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
        self.ly = 0.081  # length in y-direction (m)
        self.f = 16.584013596491230  # fixed frequency (Hz)

        # State: [u, w, theta, theta_dot]
        self.state_dim = 4
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -5*np.pi, -30]), 
            high=np.array([10, 10, 5*np.pi, 30]), 
            shape=(self.state_dim,), dtype=np.float64
        )

        # Action: [ld]
        self.action_dim = 1
        self.action_space = spaces.Box(
            low=np.array([-self.ly*np.sin(np.pi/10)]), high=np.array([self.ly*np.sin(np.pi/10)]), dtype=np.float64
        )
        
        # Time step for integration
        self.dt = 0.002  # seconds
        self.max_steps = 4357
        self.current_step = 0

        # Initialize state and ld
        self.ld_prev = 0.0
        self.state = None
        self.reset()

    def _full_dynamics(self, t, y, action):
        """
        Combined dynamics for all state variables.
        y = [u, w, theta, theta_dot]
        action = [ld]
        Returns [u_dot, w_dot, theta_dot, theta_ddot]
        """
        u, w, theta, theta_dot = y
        ld = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        
        # Calculate ld_dot using finite difference
        ld_dot = (ld - self.ld_prev) / self.dt if t > 0 else 0
        
        # u_dot equation
        u_dot = (-np.sin(theta)*self.g - 2*self.bx*u/self.m + 
                self.lz*theta_dot*2*self.bx/self.m + 
                2*self.bx*ld_dot/self.m - theta_dot*w)
        
        # w_dot equation
        w_dot = (np.cos(theta)*self.g - 2*(self.c1*self.f+self.c2)/self.m - 
                2*self.bz*w/self.m - 2*self.bz*(self.lx + ld)*theta_dot/self.m + 
                theta_dot*u)
        
        # theta_ddot equation
        theta_ddot = (
            2*self.bx*self.lz*u
            - 2*self.bx*self.lz**2 * theta_dot
            - 2*self.bx*self.lz*ld_dot
            - 2*self.c1*self.f*(ld + self.lx)
            - 2*self.c2*(ld + self.lx)
            - 2*self.bz*w*(ld + self.lx)
            - 2*self.bz*(ld + self.lx)**2 * theta_dot
        ) / self.Iyy
        
        return [u_dot, w_dot, theta_dot, theta_ddot]

    def step(self, action):
        """
        Step function using solve_ivp with full state integration.
        """
        # Store current action for ld_dot calculation
        current_action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Integrate all states together
        sol = solve_ivp(
            fun=lambda t, y: self._full_dynamics(t, y, current_action),
            t_span=(0, self.dt),
            y0=self.state,
            method='Radau',
            t_eval=[self.dt]
        )
        
        # Update state
        self.state = sol.y[:, -1]
        self.ld_prev = current_action[0]
        
        # Store derivatives AFTER integration
        self.u_dot, self.w_dot, _, self.theta_ddot = self._full_dynamics(
            self.dt, self.state, current_action
        )
        self.theta_dot = self.state[3]  # theta_dot comes directly from state
        
        self.current_step += 1
        
        # Calculate reward
        reward = -np.sum(np.square(self.state[2]))  # penalize u, w, theta
        
        # Check termination conditions
        terminated = (self.current_step >= self.max_steps or 
                     np.any(np.abs(self.state) > 100))
        if np.any(np.abs(self.state) > 100):
            reward = -100
            
        return self.state.copy(), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset state
        self.state = np.array([0.00, 0.00, 0.1, 0.00], dtype=np.float64)  # small initial theta (0.1 rad ~ 5.7°)
        self.current_step = 0
        self.ld_prev = 0.0
        # Reset derivatives
        self.u_dot = 0.0
        self.w_dot = 0.0
        self.theta_dot = 0.0
        self.theta_ddot = 0.0
        return self.state.copy(), {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, State: u={self.state[0]:.2f}, w={self.state[1]:.2f}, "
              f"θ={self.state[2]:.2f}, θ̇={self.state[3]:.2f}, "
              f"u_dot={self.u_dot:.2f}, w_dot={self.w_dot:.2f}, θ_ddot={self.theta_ddot:.2f}")

    def close(self):
        pass