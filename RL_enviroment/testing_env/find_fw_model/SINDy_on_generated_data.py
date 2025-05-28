import pysindy as ps
import numpy as np
from pysindy.feature_library import PolynomialLibrary
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from pysindy.optimizers import ConstrainedSR3, STLSQ
import gymnasium as gym
from scipy.integrate import solve_ivp

# Make data
env_data = gym.make("Pendulum-v1")
obs, _ = env_data.reset(seed=42)

observations = []
controls = []
times = []

dt = 0.05
for t in range(300):
    action = env_data.action_space.sample()
    observations.append(obs)
    controls.append(action)
    times.append(t * dt)
    obs, _, terminated, truncated, _ = env_data.step(action)
    if terminated or truncated:
        break

observations = np.array(observations)     # (N, D)
controls = np.array(controls)             # (N, C)
times = np.array(times)                   # (N,)


def RLSINDyEnv(data, control, time, env_id):
    """
    Create a Gym environment with dynamics modeled by SINDy.

    Args:
        data (np.ndarray): shape (N, D) — state trajectories
        control (np.ndarray): shape (N, M) — control actions
        time (np.ndarray): time vector of length N
        env_id (str): environment ID for registration

    Returns:
        gym.Env: registered Gym environment instance
    """
    class SINDyEnv(gym.Env):
        metadata = {"render_modes": ["human"], "render_fps": 30}

        def __init__(self, model, observation_space, action_space, dt, render_mode=None):
            super().__init__()
            self.model = model
            self.dt = float(dt)
            self.render_mode = render_mode
            self.observation_space = observation_space
            self.action_space = action_space
            self.state = None

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            low = self.observation_space.low
            high = self.observation_space.high
            self.state = self.np_random.uniform(low, high)
            return self.state, {}

        def step(self, action):
            # Action-aware prediction using SINDy with control input
            def rhs(t, x):
                u = action.reshape(1, -1)
                return self.model.predict(x.reshape(1, -1), u=u).flatten()

            sol = solve_ivp(rhs, [0, self.dt], self.state, t_eval=[self.dt])
            self.state = sol.y[:, -1]

            reward = -np.linalg.norm(self.state)  # Placeholder reward
            terminated = False
            truncated = False
            return self.state, reward, terminated, truncated, {}

        def render(self):
            print(f"State: {self.state}")

        def seed(self, seed=None):
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            return [seed]

        def close(self):
            pass

    # ------------------------
    # Prepare data and SINDy
    # ------------------------
    N = data.shape[0]
    dt = float(time[1] - time[0])

    observation_space_dim = int(data.shape[1])
    action_space_dim = int(control.shape[1])

    # Ensure scalar bounds
    obs_min = float(np.min(data))
    obs_max = float(np.max(data))
    ctrl_min = float(np.min(control))
    ctrl_max = float(np.max(control))

    # Define observation and action spaces
    observation_space = gym.spaces.Box(
        low=-1.5 * abs(obs_min),
        high=1.5 * abs(obs_max),
        shape=(observation_space_dim,),
        dtype=np.float32,
    )

    action_space = gym.spaces.Box(
        low=-1.5 * abs(ctrl_min),
        high=1.5 * abs(ctrl_max),
        shape=(action_space_dim,),
        dtype=np.float32,
    )

    # Fit SINDy model with control
    library = ps.PolynomialLibrary()
    optimizer = STLSQ()
    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    model.fit(data, t=dt, u=control, library_ensemble=True)

    # Dynamically register env
    if env_id in gym.envs.registry:
        del gym.envs.registry[env_id]

    gym.envs.registration.register(
        id=env_id,
        entry_point=lambda: SINDyEnv(model, observation_space, action_space, dt),
        max_episode_steps=N
    )

    # Instantiate and return the environment
    env = gym.make(env_id)
    return env, model


env_name = "PendulumSINDy-v0"
env , gt= RLSINDyEnv(
    data=observations,
    control=controls,
    time=times,
    env_id=env_name
)

# Step 1: Use same initial condition
initial_state, _ = env.reset(seed=42)
env_data.reset(seed=42)
env.state = initial_state.copy()
env_data.unwrapped.state = initial_state.copy()

# Step 2: Rollout both environments using same control inputs
sindy_states = [initial_state.copy()]
real_states = [initial_state.copy()]
obs_data = initial_state.copy()
obs_sindy = initial_state.copy()

for i in range(len(controls)):
    action = controls[i]

    # Real env rollout
    obs_data, _, terminated_data, truncated_data, _ = env_data.step(action)
    real_states.append(obs_data.copy())

    # SINDy env rollout
    obs_sindy, _, terminated_sindy, truncated_sindy, _ = env.step(action)
    sindy_states.append(obs_sindy.copy())

    if terminated_data or truncated_data:
        break

# Step 3: Convert to arrays
real_states = np.array(real_states)
sindy_states = np.array(sindy_states)

# Step 4: Compute Mean Squared Error
mse = np.mean((real_states - sindy_states)**2, axis=0)
print("Mean Squared Error per state dimension:", mse)

# Step 5: Plot the trajectories
plt.figure(figsize=(12, 4))
labels = ['cos(theta)', 'sin(theta)', 'theta_dot']
for i in range(real_states.shape[1]):
    plt.subplot(1, 3, i+1)
    plt.plot(real_states[:, i], label="Real Env", linestyle='--')
    plt.plot(sindy_states[:, i], label="SINDy Proxy")
    plt.title(labels[i])
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

plt.suptitle("Comparison of Real vs SINDy Environment Trajectories")
plt.tight_layout()
plt.show()


# Load saved data
# data = np.load(r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\RL_enviroment\testing_env\dynamics_data.npz", allow_pickle=True)

# X = data['X']
# U = data['U']

# x_test = X[0, :, :]
# u_test = U[0, :, :]

# # Time array and time step
# t_test = np.linspace(0, 1, x_test.shape[0])
# dt = t_test[1] - t_test[0]

# # Get derivatives
# differentiation_method = ps.SmoothedFiniteDifference()
# x_dot = differentiation_method._differentiate(x_test, t_test)

# Plot x vs. dx/dt
# state_labels = ['u', 'v', 'theta', 'theta_dot']

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

# X_mult = [traj for traj in X]

# U_mult = [traj for traj in U]

# Create SINDy model
# model = ps.SINDy(
#     optimizer = STLSQ(threshold=0.1, fit_intercept=True),
#     feature_library= PolynomialLibrary(degree=2), 
#     discrete_time=False,
#     differentiation_method=differentiation_method
#     )

# # # Fit the model
# model.fit(X_mult, t=0.002, u=U_mult, library_ensemble=True, multiple_trajectories=True) 

# # Print the discovered equations
# model.print()

# # You can also use the model for prediction
# x0 = X[0, 0, :]  # Initial state from first trajectory
# u_test = U[0, :, :]  # Controls from first trajectory

# # Create time array
# dt = 0.002  # time step
# t_sim = np.linspace(0, (len(u_test) - 1) * dt, len(u_test))  # proper time array

# # Simulate the learned model
# x_sim = model.simulate(x0, t=t_sim, u=u_test)



# # # Get ground truth for first trajectory
# # X_true = X[0]  # shape: (n_timesteps, n_states)
# # u_true      = X_true[:, 0]
# # w_true      = X_true[:, 1]
# # theta_true  = X_true[:, 2]
# theta_dot_true = X_true[:, 3]

# # Get simulated states from SINDy
# u_sim      = x_sim[:, 0]
# w_sim      = x_sim[:, 1]
# theta_sim  = x_sim[:, 2]
# theta_dot_sim = x_sim[:, 3]

# # Plotting
# fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# axs[0].plot(u_true, label='u (true)', linewidth=2)
# axs[0].plot(u_sim, '--', label='u (simulated)', linewidth=2)
# axs[0].legend()
# axs[0].set_ylabel('u')

# axs[1].plot(w_true, label='w (true)', linewidth=2)
# axs[1].plot(w_sim, '--', label='w (simulated)', linewidth=2)
# axs[1].legend()
# axs[1].set_ylabel('w')

# axs[2].plot(theta_true, label='theta (true)', linewidth=2)
# axs[2].plot(theta_sim, '--', label='theta (simulated)', linewidth=2)
# axs[2].legend()
# axs[2].set_ylabel('theta')

# axs[3].plot(theta_dot_true, label='theta_dot (true)', linewidth=2)
# axs[3].plot(theta_dot_sim, '--', label='theta_dot (simulated)', linewidth=2)
# axs[3].legend()
# axs[3].set_ylabel('theta_dot')
# axs[3].set_xlabel('Time step')

# plt.tight_layout()
# plt.show()


