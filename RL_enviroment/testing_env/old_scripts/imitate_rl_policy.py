import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.optimizers import STLSQ, SR3, ConstrainedSR3
from sklearn.model_selection import train_test_split
import dill as pickle
from itertools import combinations
from sklearn.linear_model import Lasso
from pysindy.feature_library import FourierLibrary, PolynomialLibrary, GeneralizedLibrary,  CustomLibrary
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import custom_optimizers as cust_opt
from quantization import FixedPointVisualizer
from custom_libs import hardware_efficient_library, hardware_efficient_library_pendulum



def save_sindy_model(model, save_prefix="sindy_model"):
    """
    Save SINDy model coefficients and feature names separately.
    """
    np.save(f"{save_prefix}_coefficients.npy", model.coefficients())
    with open(f"{save_prefix}_feature_names.json", "w") as f:
        json.dump(model.get_feature_names(), f)
    print(f"✅ SINDy model saved to '{save_prefix}_coefficients.npy' and '.json'")



def analyze_with_sindy(data_file="ppo_CartPole-v1_data.csv", dt=0.01):
    """
    Load collected data and apply SINDy to discover the RL policy: state -> action.
    Automatically adapts to any Gym environment based on CSV structure.
    """
    # Load data
    data = pd.read_csv(data_file)

    # Detect state and action columns dynamically
    state_columns = [col for col in data.columns if col.startswith("state_")]
    action_columns = [col for col in data.columns if col.startswith("action_")]

    X = data[state_columns].values  # (N, state_dim)
    y = data[action_columns].values  # (N, action_dim)

    print(f"State dim: {X.shape[1]}, Action dim: {y.shape[1]}, Samples: {X.shape[0]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    t = np.arange(X_train.shape[0]) * dt




    # Create a combined library
    poly_library = PolynomialLibrary(degree=3)
    fourier_library = FourierLibrary(n_frequencies=1)  
    combined_library = GeneralizedLibrary(libraries=[poly_library, fourier_library])

    # hw_lib = hardware_efficient_library()
    hw_lib = hardware_efficient_library_pendulum()

    # library = PolynomialLibrary(degree=5)
    library = hw_lib

    # This is needed for the constrained optimizer
    # combined_library.fit(X)
    library.fit(X)

    n_features = library.n_output_features_
    n_features = library.n_output_features_

    n_targets = y_train.shape[1]




    # Fit SINDy model

    ## GRIDSEARCH
    fpv = FixedPointVisualizer(int_bits=4, frac_bits=6)
    threshold_range = np.logspace(-5, 1, 15)  # for stlsq
    alpha_range = np.logspace(-5, 5, 15)      
    # threshold_range = np.logspace(1, 9, 10) # for sr3
    # alpha_range = np.logspace(-5, -1, 10)  # for sr3

    grid_results = []
    best_mse = np.inf
    best_params = None
    best_model = None

    for max_threshold in threshold_range:
        for alpha in alpha_range:
            constraint_rhs = np.array([max_threshold for _ in range(n_features * n_targets)])
            constraint_lhs = np.eye(n_features * n_targets,n_features * n_targets)

            # optimizer = Lasso(alpha=0.0003, fit_intercept=True, max_iter=500)
            # optimizer = ConstrainedSR3(constraint_rhs=constraint_rhs, 
            #                      constraint_lhs=constraint_lhs, 
            #                      inequality_constraints= True, 
            #                      thresholder="l1",
            #                      threshold=alpha,
            #                      max_iter=10000) 
            # optimizer = cust_opt.HWConstrainedSR3(constraint_rhs=constraint_rhs, 
            #                      constraint_lhs=constraint_lhs, 
            #                      inequality_constraints= True, 
            #                      thresholder="l1",
            #                      threshold=alpha,
            #                      max_iter=10000)
            optimizer = ps.STLSQ(threshold=max_threshold, alpha=alpha)
            # optimizer = cust_opt.HWConstrainedSTLSQ(threshold=threshold, alpha=alpha, quantizer=fpv)
            model = ps.SINDy(feature_library=library, optimizer=optimizer)

            try:
                model.fit(X_train, t=dt, x_dot=y_train)
                model.coefficients()[:] = fpv.quantize(np.array(   model.coefficients()).astype(np.float64))
                y_pred = model.predict(X_test)
                mse = ((y_pred - y_test) ** 2).mean()
            except Exception as e:
                mse = np.inf  # Treat failed fits as bad models

            grid_results.append(((max_threshold, alpha), mse))

            if mse < best_mse:
                best_mse = mse
                best_params = (max_threshold, alpha)
                best_model = model

    print(f"Best params from grid search: threshold={best_params[0]:.3e}, alpha={best_params[1]:.3e}")
    print(f"Best MSE: {best_mse:.6e}")

    print("\nLearned SINDy Policy:")
    best_model.print()

    # quantize
    best_model.coefficients()[:] = fpv.quantize(np.array(best_model.coefficients()).astype(np.float64))

    # Print summary
    print("\nQuantized SINDy Policy:")
    best_model.print()
   
    
    # Save model
    with open("sindy_policy.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Evaluation
    y_pred = best_model.predict(X_test)
    mse = ((y_pred - y_test) ** 2).mean()
    nonzero_count = (best_model.coefficients() != 0).sum()
    print(f"\nNonzero coefficients: {nonzero_count}")
    print(f"MSE on test set: {mse:.4f}")


if __name__ == "__main__":
    analyze_with_sindy(data_file=r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\RL_enviroment\Sindy_best_found_policies\Pendulum\sac_Pendulum-v1_data.csv", dt=0.001)  # dt may vary based on env
