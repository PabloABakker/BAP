import numpy as np
import pandas as pd
import os
import sys
import copy
import json
import dill as pickle
from itertools import product
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import pysindy as ps
from pysindy.optimizers import STLSQ, ConstrainedSR3

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import custom_optimizers as cust_opt
from quantization import FixedPointVisualizer
from custom_libs import hardware_efficient_library_pendulum

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 


# Environment config: (env_name, data_path, dt, library_func)
ENVIRONMENTS = [
    ("CartPole-v1", script_dir + "/Sindy_best_found_policies/Cartpole/ppo_CartPole-v1_data.csv", 0.01, hardware_efficient_library_pendulum),
    # Add more environments here
]


# Optimizer and library config
OPTIMIZER_TYPES = ["regular", "hardware"]
LIBRARY_TYPES = ["poly", "hw"]
FPV = FixedPointVisualizer(int_bits=4, frac_bits=6)

threshold_range = np.logspace(-5, 1, 10)
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
    best_mses = np.inf
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
            mses = [mse, mse_qat]

            return (thresh, alpha, mses, model)
        except Exception:
            return (thresh, alpha, np.inf, None)
        
    if optimizer_name == "ConstrainedSR3":
        results = Parallel(n_jobs=3)(
            delayed(try_params)(thresh, alpha) for thresh, alpha in product(threshold_range_sr3, alpha_range))
    else:
        results = Parallel(n_jobs=3)(
            delayed(try_params)(thresh, alpha) for thresh, alpha in product(threshold_range, alpha_range))
        

    for thresh, alpha, mses, model in results:
        if 1000*mses[0] + mses[1] < best_mses and model is not None:
            best_mses = 1000*mses[0] + mses[1]
            best_mse = mses[0]
            best_mse_qat = mses[1]
            best_model = model
            best_params = {'threshold':thresh, 'alpha':alpha}
            nonzero_count = int((best_model.coefficients() != 0).sum())


    if best_model:
        # Prepare save paths
        save_dir = os.path.join("Sindy_best_found_policies", env_name)
        os.makedirs(save_dir, exist_ok=True)
        model_filename = f"sindy_policy_{lib_type}_{opt_type}_{optimizer_name}.pkl"

        model_path = os.path.join(save_dir, model_filename)
        coeff_path = model_path.replace(".pkl", "_coefficients.npy")

        info_path = model_path.replace(".pkl", "_info.json")
        featnames_path = model_path.replace(".pkl", "_feature_names.json")
        search_log_path = model_path.replace(".pkl", "_search_log.csv")

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
                "mse_qat": float(mses[1]) if isinstance(mses, list) else np.inf
            }
            for thresh, alpha, mses, _ in results if mses is not None
        ])
        log_df.to_csv(search_log_path, index=False)

        print(f"✅ Saved model: {model_path} | MSE: {best_mse:.4e} | MSE (QAT): {best_mse_qat:.4e} | Params: {best_params} | Nonzero count: {nonzero_count}")


def train_all_variants_for_env(env_name, data_path, dt, hw_lib_func):
    print(f"\n=== Training SINDy models for {env_name} ===")
    data = pd.read_csv(data_path)
    state_columns = [col for col in data.columns if col.startswith("state_")]
    action_columns = [col for col in data.columns if col.startswith("action_")]
    X = data[state_columns].values
    y = data[action_columns].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly_library = ps.PolynomialLibrary(degree=5)
    hw_library = hw_lib_func()

    for lib_type, library in [("poly", poly_library), ("hw", hw_library)]:
        for opt_type in OPTIMIZER_TYPES:
            for optimizer_name in ["STLSQ", "ConstrainedSR3"]:
                fit_sindy_model(env_name, dt, X_train, y_train, X_test, y_test, lib_type, opt_type, optimizer_name, library)


if __name__ == "__main__":
    for env_name, path, dt, hw_lib_func in ENVIRONMENTS:
        train_all_variants_for_env(env_name, path, dt, hw_lib_func)
