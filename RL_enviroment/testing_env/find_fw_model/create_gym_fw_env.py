import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pysindy as ps
from pysindy.differentiation import SmoothedFiniteDifference
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
from gymnasium.envs.registration import register

# Step 1: Fit SINDy model from dataset
optimizer = ps.STLSQ(threshold=0.1, alpha=0.01)
library = ps.PolynomialLibrary(degree=3, include_bias=True)

model_proxy = ps.SINDy(
        optimizer=optimizer,
        feature_library=library,
        differentiation_method=SmoothedFiniteDifference()
    )
model_proxy.fit(data, t=dt)


# Step 2: Create a Gymnasium-compatible env from SINDy model
class SINDyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, model, observation_space, render_mode=None, action_space=None, dt=0.05):
        super().__init__()
        self.render_mode = render_mode
        self.model = model
        self.dt = dt
        self.state = None
        self.observation_space = observation_space
        self.action_space = action_space if action_space else spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low = self.observation_space.low
        high = self.observation_space.high
        self.state = self.np_random.uniform(low, high)
        return self.state, {}

    def step(self, action):
        # Ignore action, SINDy is a passive model (unless forced dynamics added)
        def rhs(t, x): return self.model.predict(x.reshape(1, -1)).flatten()

        sol = solve_ivp(rhs, [0, self.dt], self.state, t_eval=[self.dt])
        self.state = sol.y[:, -1]

        reward = -np.linalg.norm(self.state)  # Placeholder
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

# Step 3: Register SINDy Env
def register_sindy_env(env_id, model, observation_shape, observation_bounds, dt=0.05):
    observation_space = spaces.Box(
        low=observation_bounds[0],
        high=observation_bounds[1],
        dtype=np.float32
    )

    gym.envs.registration.register(
        id=env_id,
        entry_point=lambda: SINDyEnv(model, observation_space, dt=dt),
        max_episode_steps=200,
    )
