import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.optimizers import STLSQ, SR3, ConstrainedSR3
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
from custom_libs import hardware_efficient_library, hardware_efficient_library_pendulum

# setup fixed point visualizer
N_intiger_bits = 4
N_fractional_bits = 7

fpv = FixedPointVisualizer(draw=False, int_bits=N_intiger_bits, frac_bits=N_fractional_bits)




def save_sindy_model(model, save_prefix="sindy_model"):
    """
    Save SINDy model coefficients and feature names separately.
    """
    np.save(f"{save_prefix}_coefficients.npy", model.coefficients())
    with open(f"{save_prefix}_feature_names.json", "w") as f:
        json.dump(model.get_feature_names(), f)
    print(f"✅ SINDy model saved to '{save_prefix}_coefficients.npy' and '.json'")

from itertools import combinations

def fixed_point_quantize(input_values, int_bits, frac_bits):
    """
    Quantizes a numpy array (1D or 2D) to the closest values in the fixed-point set
    based on ≤2-bit combinations.
    
    Parameters:
    - input_values: numpy array of float values (1D or 2D).
    - int_bits: number of integer bits.
    - frac_bits: number of fractional bits.
    
    Returns:
    - quantized array of the same shape as input_values.
    """
    total_bits = int_bits + frac_bits
    positions = list(range(total_bits))

    # Generate fixed-point values with ≤2 active bits
    values = set()
    values.add(0.0)  # Include zero

    for k in range(1, 3):  # 1 or 2 bits set
        for combo in combinations(positions, k):
            value = 0
            for pos in combo:
                value += 2 ** (int_bits - pos - 1)
            values.add(value)
            values.add(-value)

    fixed_values = sorted(values)

    input_array = np.asarray(input_values)
    fixed_arr = np.array(fixed_values)[:, np.newaxis]  # shape (N, 1)
    abs_diff = np.abs(fixed_arr - input_array.ravel())  # shape (N, M)
    closest_indices = np.argmin(abs_diff, axis=0)
    
    quantized = np.array(fixed_values)[closest_indices]
    return quantized.reshape(input_array.shape)


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

    constraint_rhs = np.array([1e8 for _ in range(n_features * n_targets)])
    constraint_lhs = np.eye(n_features * n_targets,n_features * n_targets)

    # Fit SINDy model
    optimizer = STLSQ(threshold=0.5, alpha=0.8e0, fit_intercept=True)
    # optimizer = SR3(threshold=0.001, max_iter= 10_000, nu=1e2)
    # optimizer = ConstrainedSR3(constraint_rhs=constraint_rhs, 
    #                             constraint_lhs=constraint_lhs, 
    #                             inequality_constraints= True, 
    #                             thresholder="l1",
    #                             nu= 1,
    #                             threshold=0.01,
    #                             max_iter=10000)
    # optimizer = cust_opt.HWConstrainedSR3(constraint_rhs=constraint_rhs, 
    #                             constraint_lhs=constraint_lhs, 
    #                             inequality_constraints= True, 
    #                             thresholder="l1",
    #                             threshold=0.001,
    #                             max_iter=10000)
    # optimizer = Lasso(alpha=0.0001, fit_intercept=True, max_iter=500)
    # optimizer = cust_opt.HWConstrainedSTLSQ(threshold=1e-5, alpha=1e10, quantizer=fpv)

    # Create SINDy model
    model = ps.SINDy( optimizer = optimizer, feature_library=library)
    model.fit(X_train, t=t, x_dot=y_train)
    print("\nLearned SINDy Policy:")
    # model.print()
    print(model.coefficients().shape, model.coefficients().dtype, model.coefficients())
    fpv = FixedPointVisualizer(draw=True, init_int_bits=N_intiger_bits, init_frac_bits=N_fractional_bits, input_array=model.coefficients())

    print("\n\n\n Model coeff np array:", np.array(model.coefficients()))
    print("\n\n\n -----") 
    Q_coeff  = fixed_point_quantize(np.array(model.coefficients()).astype(np.float64), int_bits=N_intiger_bits, frac_bits=N_fractional_bits)    
    
    # Save model
    with open("sindy_policy.pkl", "wb") as f:
        pickle.dump(model, f)

    # Print summary
    print("\nQuantized SINDy Policy:")
    # model.print()
    print(Q_coeff.shape, Q_coeff.dtype, Q_coeff)

    # Evaluation
    y_pred = model.predict(X_test)
    mse = ((y_pred - y_test) ** 2).mean()
    nonzero_count = (model.coefficients() != 0).sum()
    print(f"\nNonzero coefficients: {nonzero_count}")
    print(f"MSE on test set: {mse:.4f}")


if __name__ == "__main__":
    analyze_with_sindy(data_file=r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\RL_enviroment\Sindy_best_found_policies\Pendulum\sac_Pendulum-v1_data.csv", dt=0.001)  # dt may vary based on env
