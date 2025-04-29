from pysindy import SINDy
from pysindy.optimizers import STLSQ
import numpy as np
from pysindy.feature_library import PolynomialLibrary
import matplotlib.pyplot as plt
# Load saved data
data = np.load('dynamics_data.npz')
X = data['X']
X_dot = data['X_dot']
U = data['U']

# Reshape data (combine all trajectories)
X_combined = X.reshape(-1, X.shape[-1])
X_dot_combined = X_dot.reshape(-1, X_dot.shape[-1])
U_combined = U.reshape(-1, U.shape[-1])

from pysindy.feature_library import FourierLibrary
poly_library = PolynomialLibrary(degree=2)
fourier_library = FourierLibrary(n_frequencies=3)  # For sin/cos terms

# Combine them
from pysindy.feature_library import GeneralizedLibrary
combined_library = GeneralizedLibrary(
    libraries=[poly_library, fourier_library]
)

# Create SINDy model
model = SINDy(
    optimizer=STLSQ(threshold=0.01),
    feature_library=PolynomialLibrary(degree=3),
    discrete_time=False
)

# Fit the model
model.fit(X_combined, t=0.005, u=U_combined, x_dot=X_dot_combined) #changed dt from the env.dt to 0.005

# Print the discovered equations
model.print()

# You can also use the model for prediction
x0 = X[0, 0, :]  # Initial state from first trajectory
u_test = U[0, :, :]  # Controls from first trajectory

# Create time array
dt = 0.005  # time step
t_sim = np.linspace(0, (len(u_test) - 1) * dt, len(u_test))  # proper time array

# Simulate the learned model
x_sim = model.simulate(x0, t=t_sim, u=u_test)

data = np.load('dynamics_data.npz')
X_loaded = data['X']  # States [n_traj, steps, state_dim]
U_loaded = data['U']  # Control inputs [n_traj, steps, action_dim]

# Extract only the state 'u'
u_true = X_loaded[0, :, 0]   # 0th trajectory, 0th state (u)
plt.plot(u_true)
plt.plot(x_sim[:,0], label='u (simulated)')
plt.show()


