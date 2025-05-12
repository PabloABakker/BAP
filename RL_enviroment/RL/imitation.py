import numpy as np
import pandas as pd
import gymnasium as gym
import os
from stable_baselines3 import PPO, SAC, TD3
import sys

# Add path to custom env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import CustomDynamicsEnv_v2  # This runs the registration


model_dir = "models"

def collect_data(env_id="Pendulum-v1", num_episodes=10, algo="sac"):
    """
    Run the trained policy and collect (state, action) pairs.
    Returns DataFrame and saves to CSV.
    """
    env = None
    data = []
    algo = algo.lower()

    try:
        env = gym.make(env_id)

        # Determine model paths
        best_model_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}", "best_model")
        final_model_path = os.path.join(model_dir, f"{algo.upper()}_{env_id}_final")

        # Load model
        if os.path.exists(best_model_path + ".zip"):
            model_path = best_model_path
        elif os.path.exists(final_model_path + ".zip"):
            model_path = final_model_path
        else:
            raise FileNotFoundError(f"No {algo.upper()} model found for {env_id}")

        if algo == "sac":
            model = SAC.load(model_path, env=env)
            print("SAC policy network architecture:")
            print(model.actor)
        elif algo == "td3":
            model = TD3.load(model_path, env=env)
            print("TD3 policy network architecture:")
            print(model.actor)
        elif algo == "ppo":
            model = PPO.load(model_path, env=env)
            print("PPO policy network architecture:")
            print(model.policy.mlp_extractor)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)

                # Ensure action is always an array (even for discrete PPO actions)
                if np.isscalar(action):
                    action = [action]

                data.append(np.concatenate([obs, np.array(action)]))
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    finally:
        if env is not None:
            env.close()

    columns = [f"state_{i}" for i in range(len(obs))] + [f"action_{i}" for i in range(len(action))]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{algo}_{env_id}_data.csv", index=False)
    return df

if __name__ == "__main__":
    data = collect_data(env_id="CustomDynamicsEnv-v2", num_episodes=100, algo="sac")
    print("Data collection complete. First 5 rows:")
    print(data.head())
