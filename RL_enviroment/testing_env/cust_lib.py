import numpy as np
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import CustomDynamicsEnv_v2  # This runs the registration

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

vars = ['x', 'y', 'z']

# 2. Custom feature library with hardcoded functions
def hardware_efficient_library(var_names=("x", "y", "z")):
    functions = [
        # Constant
        lambda x, y, z: 1.0,

        # Linear
        lambda x, y, z: x,
        lambda x, y, z: y,
        lambda x, y, z: z,

        # Quadratic
        lambda x, y, z: x**2,
        lambda x, y, z: y**2,
        lambda x, y, z: z**2,

        # Interaction terms
        lambda x, y, z: x*y,
        lambda x, y, z: x*z,
        lambda x, y, z: y*z,

        # A + B*C combinations (non-redundant)
        lambda x, y, z: x + y*z,
        lambda x, y, z: y + x*z,
        lambda x, y, z: z + x*y,

        # (A + B)*C combinations (non-redundant)
        lambda x, y, z: (x + y)*z,
        lambda x, y, z: (x + z)*y,
        lambda x, y, z: (y + z)*x,

        # (A + B)*C + D combinations (non-redundant)
        lambda x, y, z: (x + y)*z + x,
        lambda x, y, z: (x + z)*y + z,
        lambda x, y, z: (y + z)*x + z,

        # ((A + B)^2) + C combinations (non-redundant)
        lambda x, y, z: ((y + z)**2) + x,
        lambda x, y, z: ((z + x)**2) + y,
        lambda x, y, z: ((x + y)**2) + z,

        # New composite squared terms
        lambda x, y, z: (x + y*z)**2,
        lambda x, y, z: (y + x*z)**2,
        lambda x, y, z: (z + x*y)**2,

        lambda x, y, z: ((x + y)*z)**2,
        lambda x, y, z: ((x + z)*y)**2,
        lambda x, y, z: ((y + z)*x)**2,

        lambda x, y, z: ((x + y)*z + x)**2,
        lambda x, y, z: ((x + z)*y + z)**2,
        lambda x, y, z: ((y + z)*x + z)**2,

        # Composite squared + variable
        lambda x, y, z: (x + y*z)**2 + x,
        lambda x, y, z: (y + x*z)**2 + y,
        lambda x, y, z: (z + x*y)**2 + z,

        lambda x, y, z: ((x + y)*z)**2 + x,
        lambda x, y, z: ((x + z)*y)**2 + y,
        lambda x, y, z: ((y + z)*x)**2 + z,

        lambda x, y, z: ((x + y)*z + x)**2 + x,
        lambda x, y, z: ((x + z)*y + z)**2 + y,
        lambda x, y, z: ((y + z)*x + z)**2 + z
    ]

    function_names = [
        lambda x, y, z: "1",
        lambda x, y, z: "x",
        lambda x, y, z: "y",
        lambda x, y, z: "z",
        lambda x, y, z: "x**2",
        lambda x, y, z: "y**2",
        lambda x, y, z: "z**2",
        lambda x, y, z: "x*y",
        lambda x, y, z: "x*z",
        lambda x, y, z: "y*z",
        lambda x, y, z: "x + y*z",
        lambda x, y, z: "y + x*z",
        lambda x, y, z: "z + x*y",
        lambda x, y, z: "(x + y)*z",
        lambda x, y, z: "(x + z)*y",
        lambda x, y, z: "(y + z)*x",
        lambda x, y, z: "(x + y)*z + x",
        lambda x, y, z: "(x + z)*y + z",
        lambda x, y, z: "(y + z)*x + z",
        lambda x, y, z: "((y + z)**2) + x",
        lambda x, y, z: "((z + x)**2) + y",
        lambda x, y, z: "((x + y)**2) + z",
        lambda x, y, z: "(x + y*z)**2",
        lambda x, y, z: "(y + x*z)**2",
        lambda x, y, z: "(z + x*y)**2",
        lambda x, y, z: "((x + y)*z)**2",
        lambda x, y, z: "((x + z)*y)**2",
        lambda x, y, z: "((y + z)*x)**2",
        lambda x, y, z: "((x + y)*z + x)**2",
        lambda x, y, z: "((x + z)*y + z)**2",
        lambda x, y, z: "((y + z)*x + z)**2",
        lambda x, y, z: "(x + y*z)**2 + x",
        lambda x, y, z: "(y + x*z)**2 + y",
        lambda x, y, z: "(z + x*y)**2 + z",
        lambda x, y, z: "((x + y)*z)**2 + x",
        lambda x, y, z: "((x + z)*y)**2 + y",
        lambda x, y, z: "((y + z)*x)**2 + z",
        lambda x, y, z: "((x + y)*z + x)**2 + x",
        lambda x, y, z: "((x + z)*y + z)**2 + y",
        lambda x, y, z: "((y + z)*x + z)**2 + z"
    ]

    return CustomLibrary(
        library_functions=functions,
        function_names=function_names
    )


# 5) Optional: A small wrapper around ConstrainedSR3 to allow feature-weights
class HWConstrainedSR3(ConstrainedSR3):
    def __init__(self, *args, feature_weights=None, tol=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.tol = tol
        self.feature_weights = None if feature_weights is None else np.asarray(feature_weights)

    def _fit(self, x, y):
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        Xi = np.zeros((n_features, n_targets))
        Z  = Xi.copy()

        # build weight matrix W
        if self.feature_weights is not None:
            W = np.repeat(self.feature_weights[:, None], n_targets, axis=1)
        else:
            W = np.zeros((n_features, n_targets))

        def pack(mat):
            return mat.flatten()
        def unpack(vec):
            return vec.reshape(n_features, n_targets)

        for itr in range(self.max_iter):
            Xi_old = Xi.copy()

            def objective(xi_flat):
                Xi_mat = unpack(xi_flat)
                fit    = 0.5 * np.linalg.norm(y - x @ Xi_mat)**2
                sr3    = 0.5 * self.nu * np.linalg.norm(Xi_mat - Z)**2
                ridge  = 0.5 * np.sum(W * (Xi_mat**2))
                return fit + sr3 + ridge

            def jac(xi_flat):
                Xi_mat     = unpack(xi_flat)
                grad_fit   = -x.T @ (y - x @ Xi_mat)
                grad_sr3   = self.nu * (Xi_mat - Z)
                grad_ridge = W * Xi_mat
                return pack(grad_fit + grad_sr3 + grad_ridge)

            cons = []
            if self.constraint_lhs is not None:
                kind = "ineq" if self.inequality_constraints else "eq"
                cons.append({
                    "type": kind,
                    "fun": lambda v: (self.constraint_rhs - self.constraint_lhs @ v)
                             if self.inequality_constraints
                             else (self.constraint_lhs @ v - self.constraint_rhs)
                })

            res = optimize.minimize(
                fun=objective,
                x0=pack(Xi),
                jac=jac,
                constraints=cons,
                method="SLSQP",
                options={"maxiter": 200, "ftol": 1e-12}
            )
            Xi = unpack(res.x)

            Z  = self._threshold(Xi, self.threshold)

            if np.linalg.norm(Xi - Xi_old) < self.tol:
                break

        return Xi

STLSQ()
class HWConstrainedSTLSQ(STLSQ):
    def __init__(self, alpha=0.1, threshold=0.1, max_iter=20, feature_weights=None, **kwargs):
        super().__init__(alpha=alpha, threshold=threshold, max_iter=max_iter, **kwargs)
        self.feature_weights = None if feature_weights is None else np.asarray(feature_weights)

    def _fit(self, x, y):
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        Xi = np.zeros((n_features, n_targets))

        # Precompute x^T x and x^T y for ridge solution
        XtX = x.T @ x
        Xty = x.T @ y

        if self.feature_weights is not None:
            W = np.repeat(self.feature_weights[:, None], n_targets, axis=1)
        else:
            W = np.zeros((n_features, n_targets))

        for itr in range(self.max_iter):
            Xi_old = Xi.copy()

            # Ridge regression with custom feature weighting
            for k in range(n_targets):
                ridge = self.alpha + W[:, k]  # effective regularization per feature
                Xi[:, k] = np.linalg.solve(XtX + np.diag(ridge), Xty[:, k])

            # Apply thresholding
            small_inds = np.abs(Xi) < self.threshold
            Xi[small_inds] = 0

            # Optional: Custom coefficient adjustment step
            # For example: zero out specific indices, or normalize
            # Example (dummy logic):
            # Xi[0, :] = 0  # force zeroing of first feature if needed
            # Xi = np.clip(Xi, -1, 1)

            # Convergence check
            if np.linalg.norm(Xi - Xi_old) < self.tol:
                break

        return Xi

# 6) Assemble everything & fit SINDy
lib = hardware_efficient_library(var_names=("x","y","z"))
lib = ps.PolynomialLibrary()
lib.fit(x)
n_feat = lib.n_output_features_
n_targ = x.shape[1]

rhs = np.full(n_feat * n_targ, 100.0)
lhs = np.eye(n_feat * n_targ)

# opt = HWConstrainedSR3(
#     constraint_rhs=rhs,
#     constraint_lhs=lhs,
#     inequality_constraints=True,
#     thresholder="l1",
#     max_iter=100000,
#     feature_weights=None
# )

# opt = ConstrainedSR3(
#     constraint_rhs=rhs,
#     constraint_lhs=lhs,
#     inequality_constraints=True,
#     thresholder="l1",
#     max_iter=50000
# )

opt = HWConstrainedSTLSQ(threshold=0.01, alpha=0.01, fit_intercept=True)
model = ps.SINDy(feature_library=lib, optimizer=opt)
model.fit(x, t=dt)
print("\nLearned SINDy model:")
model.print()

# 7) Simulate & plot
X_sindy = model.simulate(x[0], t_eval)
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
