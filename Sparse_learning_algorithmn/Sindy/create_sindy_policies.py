import numpy as np
import pandas as pd
import os
import sys
import copy
import json
from pathlib import Path
import dill as pickle
from itertools import product
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import pysindy as ps
from pysindy.optimizers import STLSQ, ConstrainedSR3

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))
target_dir = project_root / "Results"


# Import custom optimizers and libraries
import Custom.custom_optimizers as cust_opt
from Custom.quantization import FixedPointVisualizer
from Custom.custom_libs import create_hierarchical_dsp_library




# Environment config: (env_name, data_path, dt, library_func)
ENVIRONMENTS = [
    ("Pendulum-v1", target_dir / "Pendulum-v1/RL/ppo_Pendulum-v1_data.csv", 0.01, create_hierarchical_dsp_library),
    # Add more environments here
]


# Optimizer and library config
OPTIMIZER_TYPES = ["regular", "hardware"]
LIBRARY_TYPES = ["poly", "hw"]
FPV = FixedPointVisualizer(int_bits=4, frac_bits=6)


# Threshold and alpha ranges for hyperparameter tuning
threshold_range = np.logspace(-5, 2, 10)
threshold_range_sr3 = np.logspace(1, 10, 5)
alpha_range = np.logspace(-5, 5, 10)


def get_optimizer(opt_type, optimizer_name, alpha, threshold, constraint_rhs, constraint_lhs):
    if opt_type == "regular":
        if optimizer_name == "STLSQ":
            return STLSQ(threshold=threshold, alpha=alpha)
        elif optimizer_name == "ConstrainedSR3":
                return ConstrainedSR3(
                constraint_rhs=constraint_rhs,
                constraint_lhs=constraint_lhs,
                inequality_constraints=True,
                thresholder="l1",
                threshold=alpha,
                max_iter=10000
            )
    elif opt_type == "hardware":
        if optimizer_name == "STLSQ":
            return cust_opt.HWConstrainedSTLSQ(threshold=threshold, alpha=alpha, quantizer=FPV)
        elif optimizer_name == "ConstrainedSR3":
            return cust_opt.HWConstrainedSR3(
                constraint_rhs=constraint_rhs,
                constraint_lhs=constraint_lhs,
                inequality_constraints=True,
                thresholder="l1",
                threshold=alpha,
                max_iter=10000
            )
    raise ValueError("Invalid optimizer type")


def fit_sindy_model(env_name, dt, X_train, y_train, X_test, y_test, lib_type, opt_type, optimizer_name, library):
    library.fit(X_train)
    n_features = library.n_output_features_
    n_targets = y_train.shape[1]

    results = []
    best_metrics = np.inf
    best_model = None
    best_params = None

    def try_params(thresh, alpha):
        try:
            constraint_rhs = np.array([thresh] * (n_features * n_targets))
            constraint_lhs = np.eye(n_features * n_targets)
            optimizer = get_optimizer(opt_type, optimizer_name, alpha, thresh, constraint_rhs, constraint_lhs)
            model = ps.SINDy(feature_library=library, optimizer=optimizer)
            model.fit(X_train, t=dt, x_dot=y_train)
            mse = ((model.predict(X_test) - y_test) ** 2).mean()
            model_qat = copy.deepcopy(model)
            model_qat.coefficients()[:] = FPV.quantize(np.array(model_qat.coefficients()))
            mse_qat = ((model_qat.predict(X_test) - y_test) ** 2).mean()
            nonzero_count = int((model.coefficients() != 0).sum())
            metrics = [mse, mse_qat, nonzero_count]

            return (thresh, alpha, metrics, model)
        except Exception:
            return (thresh, alpha, np.inf, None)
        
    if optimizer_name == "ConstrainedSR3":
        results = Parallel(n_jobs=3)(
            delayed(try_params)(thresh, alpha) for thresh, alpha in product(threshold_range_sr3, alpha_range))
    else:
        results = Parallel(n_jobs=3)(
            delayed(try_params)(thresh, alpha) for thresh, alpha in product(threshold_range, alpha_range))
        

    for thresh, alpha, metrics, model in results:
        if 10*metrics[0] + 0.001*metrics[1] + 0.0*metrics[2] < best_metrics and model is not None:
            best_metrics = 10*metrics[0] + 0.001*metrics[1] + 0.0*metrics[2]
            best_mse = metrics[0]
            best_mse_qat = metrics[1]
            best_model = model
            best_params = {'threshold':thresh, 'alpha':alpha}
            nonzero_count = metrics[2]


    if best_model:
        # Prepare save paths
        save_dir = target_dir / env_name / "SINDY" / f"{lib_type}_{opt_type}_{optimizer_name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_filename = f"sindy_policy_{lib_type}_{opt_type}_{optimizer_name}.pkl"
        model_path = save_dir / model_filename

        new_filename = f"{model_path.stem}_coefficients.npy"
        coeff_path = model_path.with_name(new_filename)

        new_filename = f"{model_path.stem}_info.json"
        info_path = model_path.with_name(new_filename)

        new_filename = f"{model_path.stem}_feature_names.json"
        featnames_path = model_path.with_name(new_filename)

        new_filename = f"{model_path.stem}_search_log.csv"
        search_log_path = model_path.with_name(new_filename)

        # Save model and metadata
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        np.save(coeff_path, best_model.coefficients())

        info = {
            "mse": float(best_mse),
            "mse_qat": float(best_mse_qat),
            "params": {k: float(v) for k, v in best_params.items()},
            "Nonzero count": nonzero_count
        }
        with open(info_path, "w") as f:
            json.dump(info, f)

        with open(featnames_path, "w") as f:
            json.dump(list(best_model.get_feature_names()), f)

        # Save sweep log
        log_df = pd.DataFrame([
            {
                "env_name": env_name,
                "library_type": lib_type,
                "optimizer_type": opt_type,
                "optimizer_name": optimizer_name,
                "threshold": float(thresh),
                "alpha": float(alpha),
                "mse": float(mses[0]) if isinstance(mses, list) else np.inf,
                "mse_qat": float(mses[1]) if isinstance(mses, list) else np.inf,
                "Nonzero count": nonzero_count
            }
            for thresh, alpha, mses, _ in results if mses is not None
        ])
        log_df.to_csv(search_log_path, index=False)

        print(f"✅ Saved model: {model_path} | MSE: {best_mse:.4e} | MSE (QAT): {best_mse_qat:.4e} | Params: {best_params} | Nonzero count: {nonzero_count}")


def train_all_variants_for_env(env_name, data_path, dt, hw_lib_func):
    print(f"\n=== Training SINDy models for {env_name} ===")
    data = pd.read_csv(data_path)

    # Use the NORMALIZED states as input (X) for training SINDy.
    # The actions (y) remain the same.
    state_columns = [col for col in data.columns if col.startswith("norm_state_")]
    action_columns = [col for col in data.columns if col.startswith("action_")]
    
    # Check if the normalized columns exist
    if not state_columns:
        raise ValueError(
            "Could not find 'norm_state_' columns in the data file. "
            "Please regenerate the data with the updated generate_data.py script."
        )

    X = data[state_columns].values
    y = data[action_columns].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the state variable names for the library based on the data
    # E.g., ['x0', 'x1', 'x2']
    var_names = tuple(f"x{i}" for i in range(X_train.shape[1]))

    # Library creation is now more robust 
    poly_library = ps.PolynomialLibrary(degree=5)
    
    # Pass the correct variable names to your custom library
    hw_library = hw_lib_func(var_names=var_names)

    print(f"\nUsing {len(var_names)} state variables for SINDy: {var_names}")

    for lib_type, library in [("poly", poly_library), ("hw", hw_library)]:
        for opt_type in OPTIMIZER_TYPES:
            for optimizer_name in ["STLSQ", "ConstrainedSR3"]:
                print(f"\n--- Fitting: lib={lib_type}, opt_type={opt_type}, opt_name={optimizer_name} ---")
                fit_sindy_model(env_name, dt, X_train, y_train, X_test, y_test, lib_type, opt_type, optimizer_name, library)


if __name__ == "__main__":
    for env_name, path, dt, hw_lib_func in ENVIRONMENTS:
        train_all_variants_for_env(env_name, path, dt, hw_lib_func)
