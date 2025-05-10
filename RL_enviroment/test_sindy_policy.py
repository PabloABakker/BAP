import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
import time
import os
import sys
from stable_baselines3 import PPO, SAC, TD3
from sklearn.metrics import mean_squared_error

# --- Add path to custom environment module ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import CustomDynamicsEnv_v2  # This must register the custom env

# --- Load SINDy model ---
with open("sindy_policy.pkl", "rb") as f:
    sindy_model = pickle.load(f)

def sindy_policy(obs, model):
    return model.predict(obs.reshape(1, -1))[0]

def rl_policy(obs, model):
    action, _ = model.predict(obs, deterministic=True)
    return action

def run_policy(env_id, policy_fn, model, max_steps=500, dt=0.05, render=False):
    env = gym.make(env_id, render_mode="human" if render else None)
    obs, _ = env.reset()
    obs_list, action_list = [], []
    total_reward = 0

    for _ in range(max_steps):
        action = policy_fn(obs, model)

        if isinstance(env.action_space, gym.spaces.Discrete):
            action_env = int(np.round(np.clip(action, 0, env.action_space.n - 1)))
        else:
            action_env = np.array(action, dtype=np.float32).reshape(-1)
            action_env = np.clip(action_env, env.action_space.low, env.action_space.high)

        obs_list.append(obs.copy())
        action_list.append(action)

        obs, reward, terminated, truncated, _ = env.step(action_env)
        total_reward += reward

        if render:
            time.sleep(dt)

        if terminated or truncated:
            break

    env.close()
    return np.array(obs_list), np.array(action_list), total_reward

def load_rl_model(algo, env_id):
    algo = algo.lower()
    # Updated model path
    model_dir = os.path.join(".", "models")
    model_name = f"{algo.upper()}_{env_id}_final"  # Update this to match the naming convention
    model_path = os.path.join(model_dir, model_name + ".zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained {algo.upper()} model not found for {env_id}")

    env = gym.make(env_id)
    if algo == "ppo":
        return PPO.load(model_path, env=env)
    elif algo == "sac":
        return SAC.load(model_path, env=env)
    elif algo == "td3":
        return TD3.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported RL algorithm: {algo}")


def compare_trajectories(env_id="CartPole-v1", rl_algo="ppo", dt=0.001, render=False):
    max_steps = 500

    print("\nRunning SINDy policy...")
    sindy_obs, sindy_actions, sindy_reward = run_policy(env_id, sindy_policy, sindy_model, max_steps, dt, render)
    print(f"SINDy reward: {sindy_reward:.2f}")

    print("\nLoading and running RL policy...")
    rl_model = load_rl_model(rl_algo, env_id)
    rl_obs, rl_actions, rl_reward = run_policy(env_id, rl_policy, rl_model, max_steps, dt, render)
    print(f"{rl_algo.upper()} reward: {rl_reward:.2f}")

    min_len = min(len(sindy_obs), len(rl_obs))
    sindy_obs, rl_obs = sindy_obs[:min_len], rl_obs[:min_len]
    sindy_actions, rl_actions = sindy_actions[:min_len], rl_actions[:min_len]
    timestamps = np.arange(min_len) * dt

    if sindy_obs.shape[1] != rl_obs.shape[1]:
        raise ValueError(f"State dimension mismatch: SINDy has {sindy_obs.shape[1]}, RL has {rl_obs.shape[1]}")
    mse = mean_squared_error(rl_obs, sindy_obs)
    print(f"\nTrajectory MSE (SINDy vs {rl_algo.upper()}): {mse:.6f}")

    # --- Plot state trajectories ---
    plt.figure(figsize=(12, 6))
    for i in range(sindy_obs.shape[1]):
        plt.plot(timestamps, sindy_obs[:, i], '--', label=f"SINDy state_{i}")
        plt.plot(timestamps, rl_obs[:, i], '-', label=f"{rl_algo.upper()} state_{i}")
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title(f"State Trajectories: SINDy vs {rl_algo.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot action trajectories ---
    if sindy_actions.ndim == 1:
        sindy_actions = sindy_actions.reshape(-1, 1)
    if rl_actions.ndim == 1:
        rl_actions = rl_actions.reshape(-1, 1)

    if sindy_actions.shape[1] != rl_actions.shape[1]:
        print("⚠️ Action dimension mismatch — skipping action comparison plot.")
    else:
        plt.figure(figsize=(12, 4))
        for i in range(sindy_actions.shape[1]):
            plt.plot(timestamps, sindy_actions[:, i], '--', label=f"SINDy action_{i}")
            plt.plot(timestamps, rl_actions[:, i], '-', label=f"{rl_algo.upper()} action_{i}")
        plt.xlabel("Time (s)")
        plt.ylabel("Action")
        plt.title(f"Action Trajectories: SINDy vs {rl_algo.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SINDy and RL controller trajectories")
    parser.add_argument("--env", default="CartPole-v1", help="Gym environment ID (default: CartPole-v1)")
    parser.add_argument("--algo", default="ppo", help="RL algorithm: ppo, sac, or td3 (default: ppo)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size (default: 0.05)")
    parser.add_argument("--render", action="store_true", help="Render environment during rollout")
    args = parser.parse_args()

    compare_trajectories(env_id=args.env, rl_algo=args.algo, dt=args.dt, render=args.render)
