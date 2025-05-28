import pysindy as ps
import numpy as np
from pysindy.feature_library import PolynomialLibrary
import matplotlib.pyplot as plt
from pysindy.optimizers import STLSQ
import gymnasium as gym
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Any


class SINDyEnvWrapper:
    """
    A factory class that creates SINDy-based Gym environments from existing environments.
    """
    
    def __init__(self, base_env_id: str, seed: int = 42, n_samples: int = 300):
        """
        Initialize the wrapper with a base Gym environment.
        
        Args:
            base_env_id: ID of the Gym environment to learn
            seed: Random seed for reproducibility
            n_samples: Number of samples to collect for training
        """
        self.base_env_id = base_env_id
        self.seed = seed
        self.n_samples = n_samples
        self.observation_space = None
        self.action_space = None
        
    def collect_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect trajectories from the base environment.
        
        Returns:
            Tuple of (states, actions, times)
        """
        env = gym.make(self.base_env_id)
        obs, _ = env.reset(seed=self.seed)
        
        observations = []
        controls = []
        times = []
        
        dt = 0.05  # Default timestep, can be adjusted
        
        for t in range(self.n_samples):
            action = env.action_space.sample()
            observations.append(obs)
            controls.append(action)
            times.append(t * dt)
            obs, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
                
        env.close()
        
        return np.array(observations), np.array(controls), np.array(times)
    
    def create_sindy_model(self, data: np.ndarray, control: np.ndarray, dt: float) -> ps.SINDy:
        """
        Create and train a SINDy model from data.
        
        Args:
            data: State trajectories (N, D)
            control: Control actions (N, M)
            dt: Timestep
            
        Returns:
            Trained SINDy model
        """
        library = ps.PolynomialLibrary(degree=3)
        optimizer = STLSQ(threshold=0.1)
        model = ps.SINDy(optimizer=optimizer, feature_library=library)
        model.fit(data, t=dt, u=control)
        
        return model
    
    def make_sindy_env(self, model: ps.SINDy, 
                      observation_space: gym.spaces.Space,
                      action_space: gym.spaces.Space,
                      dt: float,
                      env_id: str) -> gym.Env:
        """
        Create a Gym environment with SINDy dynamics.
        
        Args:
            model: Trained SINDy model
            observation_space: Observation space from base env
            action_space: Action space from base env
            dt: Timestep
            env_id: ID to register the new environment
            
        Returns:
            The created Gym environment
        """
        class SINDyEnv(gym.Env):
            metadata = {"render_modes": ["human"], "render_fps": 30}

            def __init__(self):
                super().__init__()
                self.model = model
                self.dt = dt
                self.observation_space = observation_space
                self.action_space = action_space
                self.state = None
                self.steps = 0

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.state = self.np_random.uniform(
                    low=self.observation_space.low,
                    high=self.observation_space.high
                )
                self.steps = 0
                return self.state, {}

            def step(self, action):
                # Predict next state using SINDy model
                def rhs(t, x):
                    u = action.reshape(1, -1)
                    return self.model.predict(x.reshape(1, -1), u=u).flatten()

                try:
                    sol = solve_ivp(rhs, [0, self.dt], self.state, t_eval=[self.dt])
                    if hasattr(sol, 'y') and not isinstance(sol.y, list):
                        self.state = sol.y[:, -1]
                except:
                    # If integration fails, add noise to current state
                    self.state += 0.01 * self.np_random.normal(size=self.state.shape)

                # Clip to observation space bounds
                self.state = np.clip(
                    self.state,
                    self.observation_space.low,
                    self.observation_space.high
                )

                self.steps += 1
                reward = -np.linalg.norm(self.state)  # Simple reward
                terminated = False
                truncated = self.steps >= 500  # Max episode length
                
                return self.state, reward, terminated, truncated, {}

            def render(self):
                if hasattr(self, 'render_mode') and self.render_mode == "human":
                    print(f"State: {self.state}")

        # Register the environment
        if env_id in gym.envs.registry:
            del gym.envs.registry[env_id]

        gym.envs.registration.register(
            id=env_id,
            entry_point=lambda: SINDyEnv(),
            max_episode_steps=500
        )

        return gym.make(env_id)
    
    def create_and_compare(self, custom_env_id: str = None) -> Tuple[gym.Env, gym.Env, ps.SINDy]:
        """
        Main method to create and compare environments.
        
        Args:
            custom_env_id: Optional custom ID for the SINDy environment
            
        Returns:
            Tuple of (base_env, sindy_env, sindy_model)
        """
        # Step 1: Collect data from base environment
        observations, controls, times = self.collect_data()
        dt = times[1] - times[0]
        
        # Step 2: Create SINDy model
        model = self.create_sindy_model(observations, controls, dt)
        model.print()
        
        # Step 3: Create SINDy environment
        base_env = gym.make(self.base_env_id)
        env_id = custom_env_id or f"{self.base_env_id}-SINDy-v0"
        
        sindy_env = self.make_sindy_env(
            model=model,
            observation_space=base_env.observation_space,
            action_space=base_env.action_space,
            dt=dt,
            env_id=env_id
        )
        
        # Step 4: Compare trajectories
        self.compare_trajectories(base_env, sindy_env, controls)
        
        return base_env, sindy_env, model
    
    def compare_trajectories(self, 
                           base_env: gym.Env, 
                           sindy_env: gym.Env,
                           controls: np.ndarray,
                           n_steps: int = 100) -> None:
        """
        Compare trajectories between base and SINDy environments.
        
        Args:
            base_env: Original Gym environment
            sindy_env: SINDy-based environment
            controls: Array of control actions
            n_steps: Number of steps to compare
        """
        # Reset environments with same seed
        obs_base, _ = base_env.reset(seed=self.seed)
        obs_sindy, _ = sindy_env.reset(seed=self.seed)
        
        # Store trajectories
        base_traj = [obs_base]
        sindy_traj = [obs_sindy]
        
        # Run both environments with same actions
        for i in range(min(n_steps, len(controls))):
            action = controls[i]
            
            obs_base, _, term_base, trunc_base, _ = base_env.step(action)
            obs_sindy, _, term_sindy, trunc_sindy, _ = sindy_env.step(action)
            
            base_traj.append(obs_base)
            sindy_traj.append(obs_sindy)
            
            if term_base or trunc_base or term_sindy or trunc_sindy:
                break
                
        # Convert to arrays
        base_traj = np.array(base_traj)
        sindy_traj = np.array(sindy_traj)
        
        # Calculate MSE
        mse = np.mean((base_traj - sindy_traj)**2, axis=0)
        print(f"Mean Squared Error per dimension: {mse}")
        
        # Plot results
        self.plot_comparison(base_traj, sindy_traj)
        
    def plot_comparison(self, base_traj: np.ndarray, sindy_traj: np.ndarray) -> None:
        """
        Plot comparison between base and SINDy trajectories.
        """
        plt.figure(figsize=(12, 6))
        n_dims = base_traj.shape[1]
        
        for i in range(n_dims):
            plt.subplot(n_dims, 1, i+1)
            plt.plot(base_traj[:, i], label="Base Environment", linestyle='--')
            plt.plot(sindy_traj[:, i], label="SINDy Environment")
            plt.ylabel(f"Dim {i+1}")
            plt.grid(True)
            plt.legend()
            
        plt.xlabel("Timestep")
        plt.suptitle("Base vs SINDy Environment Trajectories")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    wrapper = SINDyEnvWrapper(base_env_id="Pendulum-v1", seed=42, n_samples=500)
    base_env, sindy_env, model = wrapper.create_and_compare()

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


