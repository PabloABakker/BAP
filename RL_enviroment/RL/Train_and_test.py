import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
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
        # Decompose the observation
        cos_theta, sin_theta, theta_dot = obs

        # Recover actual theta
        actual_theta = np.arctan2(sin_theta, cos_theta)

        # Add the shift
        shifted_theta = actual_theta + self.angle_shift

        # Recompute cos and sin of shifted theta
        cos_shifted = np.cos(shifted_theta)
        sin_shifted = np.sin(shifted_theta)

        # Return the modified observation
        return np.array([cos_shifted, sin_shifted, theta_dot], dtype=np.float32)

class MountainCarContinuousRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        pos, vel = self.unwrapped.state
        # Reward high velocity (to encourage swinging)
        # Encourage proximity to the goal without forcing early rightward movement
        shaped_reward = reward + 5 * abs(vel)
        shaped_reward -= 1
        # Bonus for reaching near goal position
        if pos >= 0.45:
            shaped_reward += 10
        return shaped_reward
    
class PendulumTrackingWrapper(gym.RewardWrapper):
    def __init__(self, env, theta_ref=0.0, theta_dot_ref=0.0):
        super().__init__(env)
        self.theta_ref = theta_ref
        self.theta_dot_ref = theta_dot_ref

    def reward(self, reward):
        # Get actual state from unwrapped environment
        theta, theta_dot = self.unwrapped.state

        # Tracking error
        error = (theta - self.theta_ref) ** 2 + (theta_dot - self.theta_dot_ref) ** 2

        # Negative of error to reward lower error
        return -error





def train(env_id="Pendulum-v1", total_timesteps=200_000, algo="ppo"):
    env = None
    try:
        # Create and wrap the environment correctly
        if env_id == "MountainCarContinuous-v0":
            env = gym.make(env_id)
            env = MountainCarContinuousRewardWrapper(env)
            env = Monitor(env)
        else:
            env = gym.make(env_id)
            env = Monitor(env)

        algo = algo.lower()
        tb_log_name = f"{algo.upper()}_{env_id}"
        model_save_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}_final")

        if algo == "sac":
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=5e-4,
                buffer_size=100_000,
                batch_size=64,
                tau=0.005,
                gamma=0.98,
                verbose=1,
                device="cuda",
                tensorboard_log=log_dir
            )

        elif algo == "td3":
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

            model = TD3(
                "MlpPolicy",
                env,
                learning_rate=5e-4,
                buffer_size=100_000,
                learning_starts=5_000,           
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=-1,                
                action_noise=action_noise,        
                policy_delay=1,
                verbose=1,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                device="cuda",
                tensorboard_log=log_dir
            )

        elif algo == "qrdqn":
            model = QRDQN(
                "MlpPolicy",
                env,
                learning_rate=1e-2,
                buffer_size=50_000,
                learning_starts=1000,
                batch_size=64,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=500,
                verbose=1,
                device="cuda",
                tensorboard_log=log_dir
            )

        else:  # Default to PPO
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=1e-4,
                gamma=0.98,
                batch_size=64,
                n_steps=128,
                verbose=1,
                device="cuda",
                tensorboard_log=log_dir
            )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.join(model_dir, f"{algo.upper()}_{env_id}"),
            log_path=os.path.join(log_dir, f"{algo.upper()}_{env_id}"),
            eval_freq=10_000,
            deterministic=True,
            render=False,
            verbose=1
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name=tb_log_name
        )

        model.save(model_save_path)

    finally:
        if env is not None:
            env.close()


def test(env_id="Pendulum-v1", render=True, algo="ppo"):
    env = None
    try:
        env = gym.make(env_id, render_mode="human" if render else None)

        # if env_id == "Pendulum-v1":
        #     env = ShiftedPendulumObservationWrapper(env, angle_shift_rad=np.pi/6)


        best_model_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}", "best_model")
        final_model_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}_final")

        if os.path.exists(best_model_path + ".zip"):
            model_path = best_model_path
        elif os.path.exists(final_model_path + ".zip"):
            model_path = final_model_path
        else:
            raise FileNotFoundError(f"No model found for {algo.upper()} in {model_dir}")

        algo = algo.lower()
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

    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Training (PPO, SAC, TD3, QRDQN)')
    parser.add_argument('--train', action='store_true', help='Train mode')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--env', default="CartPole-v1", help='Environment ID')
    parser.add_argument('--algo', default="ppo", choices=["ppo", "sac", "td3", "qrdqn"],
                        help='Algorithm to use (ppo, sac, td3, qrdqn)')
    args = parser.parse_args()

    if args.train:
        train(args.env, algo=args.algo)
    elif args.test:
        test(args.env, algo=args.algo)
    else:
        print("Please specify --train or --test")
