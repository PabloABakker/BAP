from pysindy import SINDy
import pysindy as ps
from pysindy.optimizers import STLSQ
import numpy as np
from pysindy.feature_library import PolynomialLibrary
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from pysindy.optimizers import SR3

# Load saved data
data = np.load('dynamics_data.npz')

X = data['X']
U = data['U']
if np.isnan(X).any():
    print("X contains NaN values")
# normalize the data
# X = X / np.max(np.abs(X))
# U = U / np.max(np.abs(U))
# print if x contains NaN values
# if np.isnan(X).any():
#     print("X contains NaN values")

x_test = X[0, :, :]
u_test = U[0, :, :]

# Time array and time step
t_test = np.linspace(0, 1, x_test.shape[0])
dt = t_test[1] - t_test[0]

# Differentiate
differentiation_method = ps.SmoothedFiniteDifference()
x_dot_smooth = differentiation_method._differentiate(x_test, t_test)

# Plot x vs. dx/dt
state_labels = ['u', 'v', 'theta', 'theta_dot']

# Plot each state and its derivative over time
# for i in range(x_test.shape[1]):
#     plt.figure(figsize=(8, 4))
#     plt.plot(t_test, x_test[:, i], label=f'{state_labels[i]}', linewidth=2)
#     plt.plot(t_test, x_dot_smooth[:, i], '--', label=f'd{state_labels[i]}/dt (smoothed)', linewidth=2)
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.title(f'{state_labels[i]} and its derivative')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

X_mult = [traj for traj in X]

U_mult = [traj for traj in U]

from pysindy.feature_library import FourierLibrary
poly_library = PolynomialLibrary(degree=2)
fourier_library = FourierLibrary(n_frequencies=1)  # For sin/cos terms

# Combine them
from pysindy.feature_library import GeneralizedLibrary
combined_library = GeneralizedLibrary(
    libraries=[poly_library, fourier_library]
)

# Create SINDy model
from pysindy.differentiation import SmoothedFiniteDifference
model = SINDy(
    optimizer=Lasso(alpha=0.00001, fit_intercept=True, max_iter=300), # changed from 0.01 to 0.1,
    feature_library=PolynomialLibrary(degree=2), # changed from degree 3 to degree 2
    discrete_time=False,
    differentiation_method=SmoothedFiniteDifference())


# # Fit the model
model.fit(X_mult, t=0.002, u=U_mult, library_ensemble=True, multiple_trajectories=True) #changed dt from the env.dt to 0.01

# Print the discovered equations
model.print()

# You can also use the model for prediction
x0 = X[0, 0, :]  # Initial state from first trajectory
u_test = U[0, :, :]  # Controls from first trajectory

# Create time array
dt = 0.002  # time step
t_sim = np.linspace(0, (len(u_test) - 1) * dt, len(u_test))  # proper time array

# Simulate the learned model
# print(f'x0: {x0}')
# print(f'u_test: {u_test}')
x_sim = model.simulate(x0, t=t_sim, u=u_test)


import matplotlib.pyplot as plt

# Get ground truth for first trajectory
X_true = X[0]  # shape: (n_timesteps, n_states)
u_true      = X_true[:, 0]
w_true      = X_true[:, 1]
theta_true  = X_true[:, 2]
theta_dot_true = X_true[:, 3]

# Get simulated states from SINDy
u_sim      = x_sim[:, 0]
w_sim      = x_sim[:, 1]
theta_sim  = x_sim[:, 2]
theta_dot_sim = x_sim[:, 3]

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

axs[0].plot(u_true, label='u (true)', linewidth=2)
axs[0].plot(u_sim, '--', label='u (simulated)', linewidth=2)
axs[0].legend()
axs[0].set_ylabel('u')

axs[1].plot(w_true, label='w (true)', linewidth=2)
axs[1].plot(w_sim, '--', label='w (simulated)', linewidth=2)
axs[1].legend()
axs[1].set_ylabel('w')

axs[2].plot(theta_true, label='theta (true)', linewidth=2)
axs[2].plot(theta_sim, '--', label='theta (simulated)', linewidth=2)
axs[2].legend()
axs[2].set_ylabel('theta')

axs[3].plot(theta_dot_true, label='theta_dot (true)', linewidth=2)
axs[3].plot(theta_dot_sim, '--', label='theta_dot (simulated)', linewidth=2)
axs[3].legend()
axs[3].set_ylabel('theta_dot')
axs[3].set_xlabel('Time step')

plt.tight_layout()
plt.show()


