import numpy as np
from scipy import optimize
from pysindy.optimizers import ConstrainedSR3, STLSQ

__all__ = ["HWConstrainedSR3"]



class HWConstrainedSR3(ConstrainedSR3):
    def __init__(
        self,
        constraint_rhs,
        constraint_lhs,
        inequality_constraints=True,
        thresholder="l1",
        threshold=0.001,
        max_iter=10000,
        tol=1e-6,
        nu=1.0,
        feature_weights=None, # shape (n_features,) with every entry correspomnding to a weight given to the feature and how heavily it impacts the hardware
    ):
        super().__init__(
            constraint_rhs=constraint_rhs,
            constraint_lhs=constraint_lhs,
            inequality_constraints=inequality_constraints,
            thresholder=thresholder,
            threshold=threshold,
            max_iter=max_iter,
            nu=nu,
        )
        self.tol = tol
        self.feature_weights = None if feature_weights is None else np.asarray(feature_weights)

    def _fit(self, x, y):
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        Xi = np.zeros((n_features, n_targets))
        Z = Xi.copy()

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

            # objective and gradient
            def objective(xi_flat):
                Xi_mat = unpack(xi_flat)
                fit = 0.5 * np.linalg.norm(y - x @ Xi_mat)**2
                sr3 = 0.5 * self.nu * np.linalg.norm(Xi_mat - Z)**2
                ridge = 0.5 * np.sum(W * (Xi_mat**2))
                return fit + sr3 + ridge

            def jac(xi_flat):
                Xi_mat = unpack(xi_flat)
                grad_fit = -x.T @ (y - x @ Xi_mat)
                grad_sr3 = self.nu * (Xi_mat - Z)
                grad_ridge = W * Xi_mat
                return pack(grad_fit + grad_sr3 + grad_ridge)

            cons = []
            if self.constraint_lhs is not None and self.constraint_rhs is not None:
                kind = 'ineq' if self.inequality_constraints else 'eq'
                cons.append({
                    'type': kind,
                    'fun': lambda v: (
                        self.constraint_rhs - self.constraint_lhs @ v
                        if self.inequality_constraints
                        else self.constraint_lhs @ v - self.constraint_rhs
                    )
                })

            res = optimize.minimize(
                fun=objective,
                x0=pack(Xi),
                jac=jac,
                constraints=cons,
                method='SLSQP',
                options={'maxiter':100, 'ftol':1e-12}
            )
            Xi = unpack(res.x)

            Z = self._threshold(Xi, self.threshold)

            if np.linalg.norm(Xi - Xi_old) < self.tol:
                break

        return Xi

class HWConstrainedSTLSQ(STLSQ):
    def __init__(self, quantizer, alpha=0.1, threshold=0.1, max_iter=20, feature_weights=None, **kwargs):
        super().__init__(alpha=alpha, threshold=threshold, max_iter=max_iter, **kwargs)
        self.feature_weights = None if feature_weights is None else np.asarray(feature_weights)
        self.quantizer = quantizer

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
            Xi = self.squantizer.quantize(Xi)

            # Convergence check
            if np.linalg.norm(Xi - Xi_old) < 1e-6:
                break

        return Xi