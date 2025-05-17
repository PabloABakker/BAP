import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
import time
import os
import sys
from pathlib import Path
from stable_baselines3 import PPO, SAC, TD3
from sklearn.metrics import mean_squared_error
import dill as pickle

# --- Add path to custom environment module ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import CustomDynamicsEnv_v2  # This must register the custom env

# --- Load correct SINDy model based on env ---
def load_sindy_model(env_id):
    script_dir = Path(__file__).parent.parent.absolute()
    model_paths = {
        "CustomDynamicsEnv-v2": script_dir / "Sindy_best_found_policies" / "FlappingWing-v0" / "sindy_policy.pkl",
        "Pendulum-v1": script_dir / "Sindy_best_found_policies" / "Pendulum" / "sindy_policy_pendulum_80_params_6th_order_pol.pkl",
        "Acrobot-v1": script_dir / "Sindy_best_found_policies" / "Acrobot" / "sindy_policy_acrobot_fourier_lib4_plus_poly4_lasso.pkl",
        "CartPole-v1": script_dir / "Sindy_best_found_policies" / "Cartpole" / "sindy_policy_cartpole_15params_3rth_order_poly.pkl",
        "MountainCarContinuous-v0": script_dir / "Sindy_best_found_policies" / "Mountain car" / "sindy_policy_td3_mountaincarcontinuous_5_params_lasso.pkl",
    }

    if env_id not in model_paths:
        raise ValueError(f"No SINDy model path configured for env: {env_id}")

    model_path = model_paths[env_id]
    if not model_path.exists():
        print(f"SINDy model not found at: {model_path}. Trying fallback path...")

        # Fallback path if the model is not found in the predefined path
        fallback_path = Path(r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\RL_enviroment\sindy_policy.pkl")
        if fallback_path.exists():
            print(f"Loading fallback SINDy model from: {fallback_path}")
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Fallback SINDy model not found at: {fallback_path}")

    print(f"Loading SINDy model from: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def sindy_policy(obs, model):
    return model.predict(obs.reshape(1, -1))[0]

def rl_policy(obs, model):
    action, _ = model.predict(obs, deterministic=True)
    return action

def run_policy(env_id, policy_fn, model, max_steps=500, dt=0.001, render=False):
    env = gym.make(env_id, render_mode="human" if render else None)
    obs, _ = env.reset()
    print("Current obs shape:", obs.shape)
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
    script_dir = Path(__file__).parent.parent.absolute()
    models_root = script_dir / "RL" / "models"

    # Define possible model locations
    possible_paths = [
        models_root / f"{algo.upper()}_{env_id}_final.zip",
        models_root / f"{algo.upper()}_{env_id}" / "best_model.zip",
        models_root / f"{algo.upper()}_{env_id}.zip"
    ]

    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if not model_path:
        raise FileNotFoundError(
            f"Trained {algo.upper()} model not found for {env_id}. Checked:\n" +
            "\n".join(f"- {p}" for p in possible_paths)
        )

    print(f"Loading model from: {model_path}")
    env = gym.make(env_id)

    if algo == "ppo":
        return PPO.load(str(model_path), env=env)
    elif algo == "sac":
        return SAC.load(str(model_path), env=env)
    elif algo == "td3":
        return TD3.load(str(model_path), env=env)
    else:
        raise ValueError(f"Unsupported RL algorithm: {algo}")

def compare_from_random_inits(env_id="Pendulum-v1", rl_algo="ppo", dt=0.05, render=False, n_trials=5):
    max_steps = 500

    rl_model = load_rl_model(rl_algo, env_id)
    sindy_model = load_sindy_model(env_id)

    all_rl_rewards = []
    all_sindy_rewards = []

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # Use the same seed to keep init conditions consistent
        seed = np.random.randint(0, 10000)
        env_rl = gym.make(env_id)
        env_rl.reset(seed=seed)
        rl_obs, rl_actions, rl_reward = run_policy(env_id, rl_policy, rl_model, max_steps, dt, render)

        env_sindy = gym.make(env_id)
        env_sindy.reset(seed=seed)
        sindy_obs, sindy_actions, sindy_reward = run_policy(env_id, sindy_policy, sindy_model, max_steps, dt, render)

        all_rl_rewards.append(rl_reward)
        all_sindy_rewards.append(sindy_reward)

        # Plot one example trajectory
        min_len = min(len(rl_obs), len(sindy_obs))
        timestamps = np.arange(min_len) * dt

        plt.figure(figsize=(10, 5))
        for i in range(rl_obs.shape[1]):
            plt.plot(timestamps, rl_obs[:min_len, i], '-', label=f"RL state_{i}")
            plt.plot(timestamps, sindy_obs[:min_len, i], '--', label=f"SINDy state_{i}")
        plt.title(f"Trial {trial+1}: State Trajectories")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Report average reward
    print("\n==== Summary Across Trials ====")
    print(f"Average RL reward over {n_trials} trials: {np.mean(all_rl_rewards):.2f}")
    print(f"Average SINDy reward over {n_trials} trials: {np.mean(all_sindy_rewards):.2f}")

    print(f"\nFinal symbolic model from SINDy:")
    sindy_model.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SINDy and RL controller trajectories")
    parser.add_argument("--env", default="Pendulum-v1", help="Gym environment ID (default: Pendulum-v1)")
    parser.add_argument("--algo", default="ppo", help="RL algorithm: ppo, sac, or td3 (default: ppo)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size (default: 0.05)")
    parser.add_argument("--render", action="store_true", help="Render environment during rollout")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of random trials (default: 5)")
    args = parser.parse_args()

    compare_from_random_inits(env_id=args.env, rl_algo=args.algo, dt=args.dt, render=args.render, n_trials=args.n_trials)
