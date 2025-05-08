# collect_rl_data.py
import numpy as np
import pandas as pd
import gymnasium as gym
import os
from stable_baselines3 import PPO, SAC

model_dir = r"RL\models"

def collect_data(env_id="Pendulum-v1", num_episodes=10, render=False, algo="sac"):
    """
    Run the trained policy and collect (state, action) pairs.
    Returns DataFrame and saves to CSV.
    """
    env = None
    data = []
    
    try:
        env = gym.make(env_id, render_mode="human" if render else None)
        
        # Determine model paths
        best_model_path = os.path.join(model_dir, f"{algo}_{env_id}", "best_model")
        final_model_path = os.path.join(model_dir, f"{algo}_{env_id}_final")
        
        # Load model
        if os.path.exists(best_model_path + ".zip"):
            model_path = best_model_path
        elif os.path.exists(final_model_path + ".zip"):
            model_path = final_model_path
        else:
            raise FileNotFoundError(f"No {algo.upper()} model found for {env_id}")
        
        model = SAC.load(model_path, env=env) if algo == "sac" else PPO.load(model_path, env=env)
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                data.append(np.concatenate([obs, action])) if algo == "sac" else data.append(np.concatenate([obs, [action]]))
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
    finally:
        if env is not None:
            env.close()
    
    columns = [f"state_{i}" for i in range(len(obs))] + [f"action_{i}" for i in range(len(action) if algo == "sac" else 1)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{algo}_{env_id}_data.csv", index=False)
    return df

if __name__ == "__main__":
    data = collect_data(env_id="Pendulum-v1", num_episodes=100, algo="sac")
    print("Data collection complete. First 5 rows:")
    print(data.head())