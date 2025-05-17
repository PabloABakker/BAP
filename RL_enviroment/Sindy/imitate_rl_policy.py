import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.optimizers import STLSQ, SR3
from sklearn.model_selection import train_test_split
import dill as pickle
from sklearn.linear_model import Lasso
from pysindy.feature_library import FourierLibrary, PolynomialLibrary, GeneralizedLibrary,  CustomLibrary
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import custom_optimizers as cust_opt
from quantization import FixedPointVisualizer
from custom_lib import hardware_efficient_library

# setup fixed point visualizer
N_intiger_bits = 5
N_fractional_bits = 4

fpv = FixedPointVisualizer(draw=False, int_bits=N_intiger_bits, frac_bits=N_fractional_bits)




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

    hw_lib = hardware_efficient_library()
    library = PolynomialLibrary(degree=5)

    # This is needed for the constrained optimizer
    combined_library.fit(X)
    library.fit(X)

    n_features = library.n_output_features_
    n_features = library.n_output_features_

    n_targets = y_train.shape[1]

    constraint_rhs = np.array([18 for _ in range(n_features * n_targets)])
    constraint_lhs = np.eye(n_features * n_targets,n_features * n_targets)

    # Fit SINDy model
    # optimizer = STLSQ(threshold=0.001, alpha=100, fit_intercept=True)
    optimizer = cust_opt.HWConstrainedSR3(constraint_rhs=constraint_rhs, 
                                constraint_lhs=constraint_lhs, 
                                inequality_constraints= True, 
                                thresholder="l1",
                                threshold=0.001,
                                max_iter=10000)
    # optimizer = Lasso(alpha=0.0001, fit_intercept=True, max_iter=500)
    # optimizer = cust_opt.HWConstrainedSTLSQ(threshold=0.5, alpha=0.1, quantizer=fpv)

    # Create SINDy model
    model = ps.SINDy( optimizer = optimizer, feature_library=library)
    model.fit(X_train, t=t, x_dot=y_train)
    model.coefficients()[:]  = fpv.quantize(model.coefficients())

    # Save model
    with open("sindy_policy.pkl", "wb") as f:
        pickle.dump(model, f)

    # Print summary
    print("\nLearned SINDy Policy:")
    model.print()

    # Evaluation
    y_pred = model.predict(X_test)
    mse = ((y_pred - y_test) ** 2).mean()
    nonzero_count = (model.coefficients() != 0).sum()
    print(f"\nNonzero coefficients: {nonzero_count}")
    print(f"MSE on test set: {mse:.4f}")


if __name__ == "__main__":
    analyze_with_sindy(data_file=r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\RL_enviroment\sac_Pendulum-v1_data.csv", dt=0.001)  # dt may vary based on env
