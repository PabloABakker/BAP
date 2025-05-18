import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import solve_ivp
import pygame

class CustomDynamicsEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(CustomDynamicsEnv, self).__init__()

        self.render_mode = render_mode

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

        # Reference values
        self.ref = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # [u_ref, w_ref, theta_ref, theta_dot_ref]

        # State: [u, w, theta, theta_dot]
        self.state_dim = 4
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf]), 
            shape=(self.state_dim,), dtype=np.float64
        )

        # Action: [ld]
        self.action_dim = 1
        self.action_space = spaces.Box(
            low=np.array([-self.ly*np.sin(np.pi/10)]), high=np.array([self.ly*np.sin(np.pi/10)]), dtype=np.float64
        )
        
        # Time step for integration
        self.dt = 0.001  
        self.max_steps = 4000
        self.current_step = 0

        # Initialize state and ld
        self.ld_prev = 0.0
        self.state = None

        # Stability tracking
        self.stable_counter = 0
        self.stable_required = 10  # Steps that the agent must remain stable
        self.stability_threshold_rad = 0.005  # rad
        self.stability_threshold_radps = 0.05  # rad/s


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
    
    def reference(self, ref):
        u_ref, w_ref, theta_ref, theta_dot_ref = ref[0], ref[1], ref[2], ref[3]
        return u_ref, w_ref, theta_ref, theta_dot_ref

    def step(self, action):
        # Get reference values
        u_ref, w_ref, theta_ref, theta_dot_ref = self.reference(self.ref)

        # Store current action for ld_dot calculation
        current_action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Integrate all states together
        sol = solve_ivp(
            fun=lambda t, y: self._full_dynamics(t, y, current_action),
            t_span=(0, self.dt),
            y0=self.state,
            method='RK45',
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
        reward = -np.sum((self.state[2]-theta_ref)**2 + 0.5*(self.state[3]-theta_dot_ref)**2)  # penalize  theta, theta_dot
        
        # Bonus for stability
        theta_stable = abs(self.state[2]-theta_ref) < self.stability_threshold_rad
        theta_dot_stable = abs(self.state[3]-theta_dot_ref) < self.stability_threshold_radps

        if theta_stable and theta_dot_stable:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        # bonus reward
        if self.stable_counter >= self.stable_required:
            reward += 50  

        # Penalize oscillations or runaway values
        if np.any(np.abs(self.state-self.ref) > 5):
            reward -= 5

        # Termination conditions
        if self.current_step >= self.max_steps:
            terminated = True
        elif self.stable_counter >= self.stable_required:
            terminated = True
        else:
            terminated = False
            
        return self.state.copy(), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)
        rng = np.random.default_rng(seed)  # Create a seed-specific generator

         # Sample random initial state using local RNG
        low = np.array([-1.0, -1.0, -1.0, -1.0])
        high = np.array([1.0, 1.0, 1.0, 1.0])
        self.state = rng.uniform(low, high)
        # Reset state
        # self.state = np.array([1.00, 1.00, 1, 0.1], dtype=np.float64)  # small initial theta (0.1 rad ~ 5.7°)
        self.current_step = 0
        self.ld_prev = 0.0

        # Reset derivatives
        self.u_dot = 0.0
        self.w_dot = 0.0
        self.theta_dot = 0.0
        self.theta_ddot = 0.0
        return self.state.copy(), {}
    

    def render(self, mode='human'):
        if mode != 'human':
            return
        
        # Set background
        self.screen.fill((255, 255, 255))  # White background
        
        # Update the position based on the drone's velocity
        self.x_pos += self.state[0]  # Update x position based on u (horizontal velocity)
        self.y_pos += self.state[1]  # Update y position based on w (vertical velocity)

        # Simulate the drone flapping (use theta to show rotation)
        self.angle = self.state[2]  # Angle for the drone's rotation
        
        # Draw the drone
        # For simplicity, the drone is a rectangle with rotation
        drone_surface = pygame.Surface((self.drone_width, self.drone_height))
        drone_surface.fill((0, 0, 255))  # Blue color for the drone
        rotated_drone = pygame.transform.rotate(drone_surface, np.degrees(self.angle))  # Rotate by theta
        rotated_rect = rotated_drone.get_rect(center=(self.x_pos, self.y_pos))
        
        # Draw the drone on the screen
        self.screen.blit(rotated_drone, rotated_rect.topleft)
        
        # Render the current step and state info
        font = pygame.font.SysFont('Arial', 18)
        text = font.render(f"Step: {self.current_step}, u: {self.state[0]:.2f}, w: {self.state[1]:.2f}, "
                           f"θ: {np.degrees(self.state[2]):.2f}, θ̇: {self.state[3]:.2f}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        # Update the display
        pygame.display.flip()
        
        # Delay to create a frame rate (e.g., 30 FPS)
        self.clock.tick(30)

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

from gymnasium.envs.registration import register

register(
    id="CustomDynamicsEnv-v2",
    entry_point="CustomDynamicsEnv_v2:CustomDynamicsEnv",
    max_episode_steps=4000  
)

