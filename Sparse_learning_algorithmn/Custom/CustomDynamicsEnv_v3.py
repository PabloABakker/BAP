import gymnasium as gym
import numpy as np
from scipy.integrate import solve_ivp
from gymnasium import spaces

class CustomDynamicsEnv(gym.Env):
    def __init__(self, render_mode=None, max_steps=5000):
        super().__init__()

        self.render_mode = render_mode

        self.max_steps = max_steps


        # Reference state: track theta and theta_dot
        self.ref = np.array([0.0, 0.0, 0.0, 0.0])

        self.ld = 0.0
        self.ld_dot_prev = 0.0
        self.episode = 0
        self.stable_counter = 0
        self.state = np.zeros(4)
        self.current_step = 0

        # Physical constants
        self.g = 9.8067
        self.m = 0.0294
        self.Iyy = 0.1
        self.bx = 0.081
        self.bz = 0.0157
        self.c1 = 0.0114
        self.c2 = -0.0449
        self.lx = 0.0
        self.lz = 0.0271
        self.ly = 0.081
        # self.f = 16.584
        self.dt = 0.001


        # Observation: [u, w, theta, theta_dot]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Action: lateral deflection ld
        self.action_space = spaces.Box(low=np.array([-50, 0]), high=np.array([50, 50]), dtype=np.float64)

        # Default reward shaping parameters (will be updated)
        self.stability_threshold_rad = 0.05
        self.stability_threshold_radps = 0.5
        self.stable_required = 10
        self.stability_bonus = 50
        self.instability_penalty = 0.5

        self.last_reward_components = {}

    def set_episode_number(self, episode):
        self.episode = episode
        max_episodes = 1000  # adjust based on your training

        progress = episode / max_episodes

        # Progressive shaping (easy → hard)
        self.stability_threshold_rad = 0.05 * (1 - progress) + 0.005 * progress
        self.stability_threshold_radps = 0.5 * (1 - progress) + 0.05 * progress
        self.stable_required = int(5 * (1 - progress) + 10 * progress)

        self.stability_bonus = 50 + 25 * progress
        self.instability_penalty = 1 + 1.5 * progress

    def _dynamics(self, t, y, ld_dot, f):
        u, w, theta, theta_dot = y
        # ld_dot = (ld-ld_prev)/dt
        self.ld +=  ld_dot * self.dt

        # clip ld 
        ld = np.clip(self.ld, -self.ly * np.sin(np.pi / 10), self.ly * np.sin(np.pi / 10))

        ## Simplified dynamics equations
        # T = self.c1 * f + self.c2
        # u_dot = -self.g * np.sin(theta)
        # w_dot = self.g * np.cos(theta) - T / self.m
        # theta_ddot = -T*ld/self.Iyy

        ## Dynamics equations from matlab
        u_dot = (-np.sin(theta)*self.g - 2*self.bx*u/self.m + 
                 self.lz*theta_dot*2*self.bx/self.m + 
                 2*self.bx*ld_dot/self.m - theta_dot*w)

        w_dot = (np.cos(theta)*self.g - 2*(self.c1*f+self.c2)/self.m - 
                 2*self.bz*w/self.m - 2*self.bz*(self.lx + ld)*theta_dot/self.m + 
                 theta_dot*u)

        theta_ddot = (
            2*self.bx*self.lz*u
            - 2*self.bx*self.lz**2 * theta_dot
            - 2*self.bx*self.lz*ld_dot
            - 2*self.c1*f*(ld + self.lx)
            - 2*self.c2*(ld + self.lx)
            - 2*self.bz*w*(ld + self.lx)
            - 2*self.bz*(ld + self.lx)**2 * theta_dot
        ) / self.Iyy

        ## Dynamics equations from paper
        # u_dot = -theta_dot * w - self.g * np.sin(theta) - self.bx * f * u / self.m + self.lz * theta_dot * f * self.bx / self.m - f * self.bx * ld_dot / self.m

        # w_dot = theta_dot * u + self.g * np.cos(theta) - (self.c1 * f + self.c2) / self.m - self.bz * f * w / self.m - self.bz * f * ld_dot * theta_dot / self.m

        # theta_ddot = (
        #     -self.bx * f * self.lz * (u - self.lz * theta_dot + ld_dot)
        #     + self.bz * f * ld * (w - ld * theta_dot)
        #     - (self.c1 * f + self.c2) * ld
        # ) / self.Iyy

        return [u_dot, w_dot, theta_dot, theta_ddot]

    def step(self, action):
        # Clip and extract actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ld_dot = float(action[0])
        f = float(action[1]) 

        # Integrate dynamics
        t_span = (0, self.dt)
        y0 = self.state.copy()
        sol = solve_ivp(self._dynamics, t_span, y0, args=(ld_dot, f), method='RK45', rtol=1e-6, atol=1e-6)
        self.state = sol.y[:, -1]

        # Calculate errors
        theta_error = self.state[2] - self.ref[2]
        theta_dot_error = self.state[3] - self.ref[3]

        # Base reward
        reward = -0.1 * (theta_error**2 +  0.1* theta_dot_error**2)

        # Control smoothness penalty
        control_change = abs(ld_dot - self.ld_dot_prev)
        reward -= 0.01 * control_change

        # Stability check
        theta_stable = abs(theta_error) < self.stability_threshold_rad
        theta_dot_stable = abs(theta_dot_error) < self.stability_threshold_radps


        if theta_stable and theta_dot_stable:
            self.stable_counter += 1
            
            # Stability bonus only when first achieving stability
            if self.stable_counter == self.stable_required:
                reward += self.stability_bonus
        else:
            self.stable_counter = 0

        # Termination conditions
        if abs(self.state[2]) > np.pi/2 or abs(self.state[3]) > 10.0: # Angle > 90 deg or ang_vel too high
            reward -= self.instability_penalty # Penalize strong instability
            terminated = True
        elif np.any(np.abs(self.state[[0,1]]) > 25): # Allow velocities to go higher before termination
            reward -= self.instability_penalty
            terminated = True
        else:
            terminated = False

        # Store debug info
        self.last_reward_components = {
            "theta_error": theta_error,
            "theta_dot_error": theta_dot_error,
            "control_change": control_change,
            "stability_bonus": self.stability_bonus if self.stable_counter >= self.stable_required else 0,
            "total_reward": reward,
        }

        self.ld_dot_prev = ld_dot
        self.current_step += 1

        return self.state.copy(), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)
        rng = np.random.default_rng(seed)

        # Random initial state
        low = np.array([-1.0, -1.0, -1.0, -1.0])
        high = np.array([1.0, 1.0, 1.0, 1.0])
        self.state = rng.uniform(low, high)

        self.current_step = 0
        self.ld_prev = 0.0
        self.stable_counter = 0

        # Set reward shaping based on episode number
        self.set_episode_number(self.episode)

        return self.state.copy(), {}

    def render(self):
        print(f"Step {self.current_step}: State {self.state}, Reward components: {self.last_reward_components}")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

# from gymnasium.envs.registration import register

# register(
#     id="CustomDynamicsEnv_v3",
#     entry_point="Custom.CustomDynamicsEnv_v3:CustomDynamicsEnv",
#     max_episode_steps=5000  
# )