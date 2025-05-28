import numpy as np
import gymnasium
import pysindy as ps
from pysindy.feature_library import CustomLibrary
from pysindy.optimizers import STLSQ, ConstrainedSR3
from scipy.integrate import solve_ivp
from itertools import permutations
from sklearn.linear_model import Lasso
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
import sys
import re

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import CustomDynamicsEnv_v2  # This runs the registration
from quantization import FixedPointVisualizer
from custom_libs import hardware_efficient_library
from testing_env.hyper_param_tuner.hyper_par_tuner import PSO, Particles

def equations_from_Xi_Theta(Xi, Theta_feature_names, var_names="xyzuvw"):
    """
    Reconstructs symbolic strings from SINDy coefficient matrix and library names.
    - Wraps terms in parentheses
    - Moves minus signs outside for clean formatting
    - Omits '* 1' for constant terms
    """
    num_eqs = Xi.shape[1]
    var_mapping = {f"x{i}": var_names[i] for i in range(num_eqs)}
    
    equations = []
    for eq_idx in range(num_eqs):
        terms = []
        for i, coeff in enumerate(Xi[:, eq_idx]):
            if np.abs(coeff) > 1e-10:
                raw_term = Theta_feature_names[i]
                fixed_term = " * ".join(raw_term.split())

                # Replace x0 → x, etc.
                for old, new in var_mapping.items():
                    fixed_term = re.sub(rf'\b{old}\b', new, fixed_term)

                # Handle constant term (i.e., just "1")
                if fixed_term.strip() == "1":
                    if coeff < 0:
                        term = f"- ({abs(coeff):.3f})"
                    else:
                        term = f"({coeff:.3f})"
                else:
                    if coeff < 0:
                        term = f"- ({abs(coeff):.3f} * {fixed_term})"
                    else:
                        term = f"({coeff:.3f} * {fixed_term})"
                
                terms.append(term)

        # Combine terms properly
        equation_str = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                equation_str += f" {term}"
            else:
                equation_str += f" + {term}"

        equations.append(equation_str.replace("* + *"," + ").replace("* - *", " - "))
    return equations



# 1. Lorenz system
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

t_span = (0, 10)
dt = 0.01
t_eval = np.arange(t_span[0], t_span[1], dt)
initial_state = [1.0, 1.0, 1.0]
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
x = sol.y.T  # (n_samples, 3)

# 2. Create library
lib = ps.PolynomialLibrary(degree=2)
lib = hardware_efficient_library()
lib.fit(x)

# 3) create function to optimize
opt = STLSQ()

def tune(particle):
    optimizer = STLSQ(threshold=particle[0], alpha=particle[1])
    model = ps.SINDy(feature_library=lib, optimizer=optimizer)
    model.fit(x, t=dt)
    X_sindy = model.simulate(x[0], t_eval)
    mse = mean_squared_error(x, X_sindy)
    return mse

# define hyper-parameters
N = 10
grid_space =  np.array([[1e-3, 1], [1e-6, 1e-1]])
momenta = (0.4, 0.9)
cognitive_constant = 0.5
social_constant = 0.5
maximum_iterations = 1

##  PSO
# pso = PSO(tune, N, grid_space, momenta, cognitive_constant, social_constant, maximum_iterations)
# pso.optimize()
# pso.animate()
# best_location = pso.get_best_location()
# print(f'best location: {best_location}')

# Fit best model
# best_threshold, best_alpha = best_location
# best_optimizer = STLSQ(threshold=best_threshold, alpha=best_alpha)
# model = ps.SINDy(feature_library=lib, optimizer=best_optimizer)
# model.fit(x, t=dt)
# print("\nLearned SINDy model:")
# model.print()
# X_sindy = model.simulate(x[0], t_eval)

## GRIDSEARCH
# Same setup
threshold_range = np.logspace(-3, 3, 1)  # 10 values from 1e-3 to 1e3
alpha_range = np.logspace(-3, 3, 1)      # 10 values from 1e-3 to 1e3

grid_results = []
best_mse = np.inf
best_params = None
best_model = None

for threshold in threshold_range:
    for alpha in alpha_range:
        optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
        model = ps.SINDy(feature_library=lib, optimizer=optimizer)

        try:
            model.fit(x, t=dt)
            X_sindy = model.simulate(x[0], t_eval)
            mse = mean_squared_error(x, X_sindy)
        except Exception as e:
            mse = np.inf  # Treat failed fits as bad models

        grid_results.append(((threshold, alpha), mse))

        if mse < best_mse:
            best_mse = mse
            best_params = (threshold, alpha)
            best_model = model

print(f"Best params from grid search: threshold={best_params[0]:.3e}, alpha={best_params[1]:.3e}")
print(f"Best MSE: {best_mse:.6e}")


equations = equations_from_Xi_Theta(model.coefficients().T, model.get_feature_names())
print("SINDy equations:")
print(equations)



# 7) Simulate & plot
labels = ["x", "y", "z"]
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(t_eval, x[:, i], "k", label="True")
    plt.plot(t_eval, X_sindy[:, i], "r--", label="SINDy")
    plt.xlabel("time")
    plt.ylabel(labels[i])
    plt.legend()
    plt.title(labels[i])
plt.tight_layout()
plt.show()

# Compute MSE
print(f"MSE (SINDy vs. Lorenz): {mean_squared_error(x, X_sindy):.6e}")
