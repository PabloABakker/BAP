import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import solve_ivp

class CustomDynamicsEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(CustomDynamicsEnv, self).__init__()

        # Physical constants
        self.g = 9.8067  # gravity (m/s^2)
        self.m = 0.0294  # mass (kg)
        self.Iyy = 0.1   # moment of inertia (kg·m^2)
        self.bx = 0.081  # damping coefficient in x-direction
        self.bz = 0.0157 # damping coefficient in z-direction
        self.c1 = 0.0114 # force coefficient 1
        self.c2 = -0.0449 # force coefficient 2
        self.lx = 0.0    # length in x-direction (m)
        self.lz = 0.0271 # length in z-direction (m)
        self.f = 16.584013596491230  # fixed frequency (Hz)

        # State: [u, w, theta, theta_dot]
        self.state_dim = 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Action: [ld_dot] (derivative of ld)
        self.action_dim = 1
        self.action_space = spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )

        # Time step for integration
        self.dt = 0.05  # seconds
        self.max_steps = 200
        self.current_step = 0

        # Initialize state and ld
        self.state = None
        self.ld = 0.0  # initial ld
        self.reset()

    def _dynamics(self, t, y, action):
        """
        Defines the system dynamics (ODEs)
        y = [u, w, theta, theta_dot]
        action = [ld_dot]
        """
        u, w, theta, theta_dot = y
        ld_dot = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])

        # Compute derivatives
        ud = (-self.m * theta_dot * w - self.m * self.g * np.sin(theta) -
              self.bx * self.f * (u - self.lz * theta_dot + ld_dot)) / self.m

        wd = (self.m * theta_dot * u + self.m * self.g * np.cos(theta) -
              (self.c1 * self.f + self.c2) - self.bz * self.f * (w - self.ld * theta_dot)) / self.m

        theta_ddot = (-self.bx * self.f * self.lz * (u - self.lz * theta_dot + ld_dot) +
                      self.bz * self.f * self.ld * (w - self.ld * theta_dot) -
                      (self.c1 * self.f + self.c2) * self.ld) / self.Iyy

        return [ud, wd, theta_dot, theta_ddot]

    def step(self, action):
        # Integrate ODEs over time step
        sol = solve_ivp(
            fun=lambda t, y: self._dynamics(t, y, action),
            t_span=[0, self.dt],
            y0=self.state,
            t_eval=[self.dt]
        )

        # Update state and ld
        self.state = sol.y[:, -1]
        self.ld += action[0] * self.dt

        # Increment step counter
        self.current_step += 1

        # Calculate reward (penalize large deviations)
        reward = -np.sum(np.square(self.state[:3]))  # penalize u, w, theta

        # Check termination conditions
        terminated = False
        if self.current_step >= self.max_steps:
            terminated = True
        if np.any(np.abs(self.state) > 100):
            terminated = True
            reward = -100

        return self.state.copy(), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset state and ld
        self.state = np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float32)  # small initial theta
        self.ld = 0.0
        self.current_step = 0
        return self.state.copy(), {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, State: u={self.state[0]:.2f}, w={self.state[1]:.2f}, "
              f"θ={self.state[2]:.2f}, θ̇={self.state[3]:.2f}, ld={self.ld:.2f}")

    def close(self):
        pass
