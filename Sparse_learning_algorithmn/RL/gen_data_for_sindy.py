import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO, SAC, TD3
import sys

# Add project root to path to import utils
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))
import Custom
from RL.utils import create_vecenv  # Import your environment creation utility

def collect_data(env_id, model_dir_name, num_episodes, algo):
    """
    Run the trained policy and collect (unnormalized_state, normalized_state, action) tuples.
    Returns DataFrame and saves to CSV.
    """
    env = None
    data_rows = []
    algo = algo.lower()

    try:
        # --- Correct Path and Environment Loading ---
        model_dir = project_root / "RL" / "models" / model_dir_name
        
        # Find the actual model file (.zip) within the directory
        model_file = None
        if (model_dir / "best_model.zip").exists():
            model_file = model_dir / "best_model.zip"
        elif (model_dir / "final_model.zip").exists():
            model_file = model_dir / "final_model.zip"
        else:
            raise FileNotFoundError(f"No best_model.zip or final_model.zip found in:\n{model_dir}")

        print(f"Found model file: {model_file}")

        # Find the vecnormalize stats associated with this specific model
        vecnorm_path = model_dir / "vecnormalize.pkl"
        if not vecnorm_path.exists():
            raise FileNotFoundError(f"VecNormalize file not found at {vecnorm_path}. It is required.")

        print(f"Loading VecNormalize stats from {vecnorm_path}")
        # Use your utility to create the properly normalized environment
        env = create_vecenv(env_id, training=False, vecnormalize_path=str(vecnorm_path))

        # Load the model with the correct environment
        model_loader = {"sac": SAC, "td3": TD3, "ppo": PPO}
        model = model_loader[algo.lower()].load(str(model_file), env=env)
        print(f"\n{algo.upper()} policy network loaded.")

        # --- MODIFIED Data Collection Loop ---
        print(f"\nCollecting {num_episodes} episodes...")
        obs = env.reset()
        for ep in range(num_episodes):
            done = np.array([False])
            while not done[0]:
                # The RL agent sees and acts on the *normalized* observation
                normalized_obs = obs
                # We also get the *unnormalized* observation for SINDy training
                unnormalized_obs = env.get_original_obs()
                
                action, _ = model.predict(normalized_obs, deterministic=True)
                flat_action = np.atleast_1d(action).flatten()

                # Store both observations along with the action
                row = np.concatenate([
                    unnormalized_obs.flatten(), 
                    normalized_obs.flatten(), 
                    flat_action
                ])
                data_rows.append(row)
                
                obs, _, done, _ = env.step(action)
            
            if (ep + 1) % 10 == 0:
                print(f"Completed episode {ep + 1}/{num_episodes}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise
    finally:
        if env is not None:
            env.close()

    if not data_rows:
        print("\nNo data was collected. Exiting.")
        return None

    # --- MODIFIED DataFrame Creation ---
    num_states = unnormalized_obs.shape[1]
    num_actions = flat_action.shape[0]
    
    # Create descriptive column names
    unnorm_cols = [f"state_{i}" for i in range(num_states)]
    norm_cols = [f"norm_state_{i}" for i in range(num_states)]
    action_cols = [f"action_{i}" for i in range(num_actions)]
    
    columns = unnorm_cols + norm_cols + action_cols
    
    df = pd.DataFrame(data_rows, columns=columns)
    
    # Save results to a CSV file
    target_dir = project_root / "Results" / env_id / "RL"
    target_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    
    output_file = target_dir / f"{algo}_{env_id}_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nData collection complete. Saved to {output_file}")
    print("First 5 rows:")
    print(df.head())
    return df

if __name__ == "__main__":
    env_id = "Pendulum-v1"  
    # Provide just the unique directory name of the model
    model_directory_name = r"PPO_Pendulum-v1_learning_rate0.001_gamma0.99_batch_size256_n_steps1024_ent_coef0.001_seed1"
    collect_data(env_id=env_id, model_dir_name=model_directory_name, num_episodes=100, algo="ppo")