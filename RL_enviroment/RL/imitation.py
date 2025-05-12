import numpy as np
import pandas as pd
import gymnasium as gym
import os
from pathlib import Path
from stable_baselines3 import PPO, SAC, TD3
import sys

# Add path to custom env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import CustomDynamicsEnv_v2  # This runs the registration

def collect_data(env_id="CustomDynamicsEnv-v2", num_episodes=10, algo="sac"):
    """
    Run the trained policy and collect (state, action) pairs.
    Returns DataFrame and saves to CSV.
    """
    env = None
    data = []
    algo = algo.lower()

    try:
        # Get absolute path to models directory
        script_dir = Path(__file__).parent.absolute()
        models_root = script_dir / "models"
        
        # Define possible model locations
        final_model_path = models_root / f"{algo.upper()}_{env_id}_final.zip"
        best_model_path = models_root / f"{algo.upper()}_{env_id}" / "best_model.zip"
        
        print(f"\nLooking for models at:")
        print(f"- {final_model_path}")
        print(f"- {best_model_path}\n")
        
        if final_model_path.exists():
            model_path = final_model_path
            print(f"Loading final model from: {model_path}")
        elif best_model_path.exists():
            model_path = best_model_path
            print(f"Loading best model from: {model_path}")
        else:
            raise FileNotFoundError(
                f"No model found at:\n{final_model_path}\nor\n{best_model_path}"
            )

        # Create environment after verifying model exists
        env = gym.make(env_id)
        
        # Load the model
        if algo == "sac":
            model = SAC.load(str(model_path), env=env)  # str() for Path compatibility
            print("\nSAC policy network architecture:")
            print(model.actor)
        elif algo == "td3":
            model = TD3.load(str(model_path), env=env)
            print("\nTD3 policy network architecture:")
            print(model.actor)
        elif algo == "ppo":
            model = PPO.load(str(model_path), env=env)
            print("\nPPO policy network architecture:")
            print(model.policy)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        # Data collection
        print(f"\nCollecting {num_episodes} episodes...")
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)

                # Ensure action is always an array
                if np.isscalar(action):
                    action = [action]

                data.append(np.concatenate([obs, np.array(action)]))
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            if (ep + 1) % 10 == 0:
                print(f"Completed episode {ep + 1}/{num_episodes}")

    except Exception as e:
        print(f"\nError during data collection: {str(e)}")
        raise
    finally:
        if env is not None:
            env.close()

    # Create and save DataFrame
    columns = [f"state_{i}" for i in range(len(obs))] + [f"action_{i}" for i in range(len(action))]
    df = pd.DataFrame(data, columns=columns)
    output_file = f"{algo}_{env_id}_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\nData collection complete. Saved to {output_file}")
    print("First 5 rows:")
    print(df.head())
    return df

if __name__ == "__main__":
    data = collect_data(env_id="CustomDynamicsEnv-v2", num_episodes=100, algo="sac")
           