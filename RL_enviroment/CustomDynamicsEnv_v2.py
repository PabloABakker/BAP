import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import solve_ivp

class CustomDynamicsEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(CustomDynamicsEnv, self).__init__()

        # Physical constantss
        self.g = 9.8067  # gravity (m/s^2)
        self.m = 0.0294  # mass (kg)
        self.Iyy = 0.1   # moment of inertia (kg·m^2)
        self.bx = 0.081  # damping coefficient in x-direction
        self.bz = 0.0157 # damping coefficient in z-direction
        self.c1 = 0.0114 # force coefficient 1
        self.c2 = -0.0449 # force coefficient 2
        self.lx = 0.0    # length in x-direction (m)
        self.lz = 0.0271 # length in z-direction (m)
        self.ly= 0.081 # length in y-direction (m)
        self.f = 16.584013596491230  # fixed frequency (Hz)

        # State: [u, w, theta, theta_dot]
        self.state_dim = 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Action: [ld_dot, ld]   ## this might  change to dihedral angle
        self.action_dim = 2
        self.action_space = spaces.Box(
            low=np.array([-np.inf, -self.ly*np.sin(18*np.pi/180)]), high=np.array([np.inf, self.ly*np.sin(18*np.pi/180)]), dtype=np.float32
        )

        # Time step for integration
        self.dt = 0.01  # seconds
        self.max_steps = 200
        self.current_step = 0

        # Initialize state and ld
        self.theta_k_1 = 0.0  

        self.state = None
        self.ld = 0.0  
        self.ld_dot = 0.0

        self.reset()

    def _dynamics(self, t, y, action):
        """
        Defines the system dynamics (ODEs)
        y = [u, w, theta, theta_dot]
        action = [ld_dot, ld]
        """
        u, w, theta, theta_dot = y
        ld_dot = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        ld = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])

        # Compute derivatives
        ud = (-self.m * theta_dot * w - self.m * self.g * np.sin(theta) -
              self.bx * self.f * (u - self.lz * theta_dot + ld_dot)) / self.m

        wd = (self.m * theta_dot * u + self.m * self.g * np.cos(theta) -
              (self.c1 * self.f + self.c2) - self.bz * self.f * (w - ld * theta_dot)) / self.m

        theta_ddot = (-self.bx * self.f * self.lz * (u - self.lz * theta_dot + ld_dot) +
                      self.bz * self.f * ld * (w - ld * theta_dot) -
                      (self.c1 * self.f + self.c2) * ld) / self.Iyy

        return [ud, wd, theta_ddot]

    def step(self, action):
        """
        Using euler forward discretization to propagate the dynamics and note 
        that for the propagation of theta we use second order information.
        """
        #Take a step in the environment using the given action.	
        f, g, h = self._dynamics(0, self.state, action)

        dy = [f, g, (self.state[2]-self.theta_k_1)/self.dt, h]  # theta is updated like a second order system

        self.theta_k_1 = self.state[2]
        self.state = self.state + self.dt * np.array(dy)

        self.current_step += 1
        reward = -np.sum(np.square(self.state[:3]))

        terminated = self.current_step >= self.max_steps or np.any(np.abs(self.state) > 100)
        if np.any(np.abs(self.state) > 100):
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
