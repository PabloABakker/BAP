import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import argparse
import numpy as np
import sys

# Add path to custom env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import CustomDynamicsEnv_v2  # This runs the registration

# Setup directories
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Custom reward wrappers
class ShiftedPendulumObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, angle_shift_rad=1.0):
        super().__init__(env)
        self.angle_shift = angle_shift_rad

    def observation(self, obs):
        cos_theta, sin_theta, theta_dot = obs
        actual_theta = np.arctan2(sin_theta, cos_theta)
        shifted_theta = actual_theta + self.angle_shift
        return np.array([np.cos(shifted_theta), np.sin(shifted_theta), theta_dot], dtype=np.float32)

class MountainCarContinuousRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        pos, vel = self.unwrapped.state
        shaped_reward = reward + 5 * abs(vel) - 1
        if pos >= 0.45:
            shaped_reward += 10
        return shaped_reward

class PendulumTrackingWrapper(gym.RewardWrapper):
    def __init__(self, env, theta_ref=0.0, theta_dot_ref=0.0):
        super().__init__(env)
        self.theta_ref = theta_ref
        self.theta_dot_ref = theta_dot_ref

    def reward(self, reward):
        theta, theta_dot = self.unwrapped.state
        error = (theta - self.theta_ref) ** 2 + (theta_dot - self.theta_dot_ref) ** 2
        return -error

def create_env(env_id, algo, for_training=True, render=False):
    render_mode = "human" if (render and for_training is False) else None
    env = gym.make(env_id, render_mode=render_mode) if render_mode else gym.make(env_id)

    if env_id == "MountainCarContinuous-v0":
        env = MountainCarContinuousRewardWrapper(env)

    env = Monitor(env)

    vecnorm_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}_vecnormalize.pkl")

    if isinstance(env.observation_space, gym.spaces.Box):
        env = DummyVecEnv([lambda: env])
        if for_training:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        else:
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False

    return env

def train(env_id="Pendulum-v1", total_timesteps=1_000_000, algo="ppo"):
    algo = algo.lower()
    env = create_env(env_id, algo, for_training=True)
    vecnorm_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}_vecnormalize.pkl")

    if algo == "qrdqn" and not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("QRDQN only supports discrete action spaces.")

    tb_log_name = f"{algo.upper()}_{env_id}"
    model_save_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}_final")

    if algo == "sac":
        model = SAC("MlpPolicy", env, 
                    learning_rate=1e-3, 
                    buffer_size=100_000, 
                    batch_size=64,
                    tau=0.001, 
                    gamma=0.99, 
                    verbose=1, device="cuda", tensorboard_log=log_dir)

    elif algo == "td3":
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, 
                    learning_rate=1e-3, 
                    buffer_size=100_000, 
                    learning_starts=8_000,
                    batch_size=64, 
                    tau=0.001, 
                    gamma=0.99, 
                    train_freq=(1, "step"),
                    gradient_steps=-1, 
                    action_noise=action_noise, policy_delay=2,
                    target_policy_noise=0.1, target_noise_clip=0.1,
                    verbose=1, device="cuda", tensorboard_log=log_dir)

    elif algo == "qrdqn":
        model = QRDQN("MlpPolicy", env, learning_rate=1e-2, buffer_size=50_000, learning_starts=1000,
                      batch_size=64, gamma=0.99, train_freq=1, gradient_steps=1,
                      target_update_interval=500, verbose=1, device="cuda", tensorboard_log=log_dir)

    else:
        model = PPO("MlpPolicy", env, 
                    learning_rate=1e-3, 
                    gamma=0.98, 
                    batch_size=64, 
                    n_steps=8,
                    verbose=1, device="cuda", tensorboard_log=log_dir)

    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(model_dir, f"{algo.upper()}_{env_id}"),
        log_path=os.path.join(log_dir, f"{algo.upper()}_{env_id}"),
        eval_freq=4_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name=tb_log_name)
    model.save(model_save_path)

    if isinstance(env, VecNormalize):
        env.save(vecnorm_path)
        print(f"Saving VecNormalize stats to {vecnorm_path}")


    env.close()

def test(env_id="Pendulum-v1", render=True, algo="ppo"):
    algo = algo.lower()
    env = create_env(env_id, algo, for_training=False, render=render)
    best_model_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}", "best_model.zip")
    final_model_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}_final.zip")

    if os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        model_path = final_model_path
    else:
        raise FileNotFoundError(f"No model found for {algo.upper()} in {model_dir}")

    if algo == "sac":
        model = SAC.load(model_path, env=env)
    elif algo == "td3":
        model = TD3.load(model_path, env=env)
    elif algo == "qrdqn":
        model = QRDQN.load(model_path, env=env)
    else:
        model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Training (PPO, SAC, TD3, QRDQN)')
    parser.add_argument('--train', action='store_true', help='Train mode')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--env', default="CartPole-v1", help='Environment ID')
    parser.add_argument('--algo', default="ppo", choices=["ppo", "sac", "td3", "qrdqn"],
                        help='Algorithm to use')
    args = parser.parse_args()

    if args.train:
        train(args.env, algo=args.algo)
    elif args.test:
        test(args.env, algo=args.algo)
    else:
        print("Please specify --train or --test")
