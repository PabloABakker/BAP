import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from itertools import product
from pathlib import Path
from joblib import Parallel, delayed
import sys
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn
from utils import make_base_env, create_vecenv, analyze_results, load_eval_results



def train_single(env_id, algo, config, seed, total_timesteps):
    # Ensure the custom environment is available
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    sys.path.append(str(project_root))
    import Custom

    # Path and Directory Setup 
    algo = algo.lower()
    algo_name = algo.upper()
    tag = "_".join(f"{k}{v}" for k, v in config.items())

    model_folder = Path("models") / f"{algo_name}_{env_id}_{tag}_seed{seed}"
    log_folder = Path("logs") / f"{algo_name}_{env_id}_{tag}_seed{seed}"
    
    # Define a shared vecnormalize path
    # This ensures both train and eval can potentially use the same stats file
    vecnorm_path = model_folder / f"vecnormalize.pkl"

    model_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)


    # Create the training environment
    train_env = create_vecenv(env_id, training=True, vecnormalize_path=vecnorm_path)
    
    # Create the evaluation environment using the same utility
    eval_env = create_vecenv(env_id, training=False, vecnormalize_path=vecnorm_path)

    # Set up the policy architecture and model arguments
    policy_kwargs = {
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "activation_fn": nn.ReLU,
    }

    model_args = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log=str(log_folder),
    )
    model_args.update(config)

    # --- Model Initialization (no changes here) ---
    if algo == "ppo":
        model = PPO(**model_args)
    elif algo == "sac":
        model = SAC(**model_args)
    elif algo == "td3":
        n_actions = train_env.action_space.shape[0]
        low, high = np.asarray(train_env.action_space.low), np.asarray(train_env.action_space.high)
        action_range = high - low
        noise_std = 0.2 * action_range if np.all(action_range > 0) else 0.2
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std)
        model = TD3(action_noise=action_noise, **model_args)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_folder),
        log_path=str(log_folder),
        eval_freq=total_timesteps // 25,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(model_folder / "final_model.zip")
    if isinstance(train_env, VecNormalize):
        train_env.save(str(vecnorm_path))

    train_env.close()
    eval_env.close()

    
    eval_results = load_eval_results(log_folder)
    best_reward_from_eval = np.max(eval_results["episode_rewards"]) if "episode_rewards" in eval_results and len(eval_results["episode_rewards"]) > 0 else -np.inf
    mean_reward_from_eval = np.mean(eval_results["episode_rewards"]) if "episode_rewards" in eval_results and len(eval_results["episode_rewards"]) > 0 else -np.inf

    return {
        "algo": algo,
        "env_id": env_id,
        "config": config,
        "seed": seed,
        "best_reward": best_reward_from_eval,
        "mean_reward": mean_reward_from_eval,
        "std_reward": np.std(eval_results["episode_rewards"]) if "episode_rewards" in eval_results and len(eval_results["episode_rewards"]) > 0 else 0.0,
        "model_path": str(model_folder / "final_model.zip"),
        "vecnorm_path": str(vecnorm_path)
    }

def run_grid_search(env_id, algo, n_jobs, total_timesteps):

    # Define the hyperparameter grid for the specified algorithm
    grid = {
        "ppo": {
            "learning_rate": [1e-3, 1e-4, 1e-5],
            "gamma": [0.99],
            "batch_size": [256],
            "n_steps": [1024],
            "ent_coef": [0.001, 0.01]
        },
        "sac": {
            "learning_rate": [1e-3, 1e-4, 1e-5],
            "buffer_size": [500_000],
            "tau": [0.005, 0.01],
            "batch_size": [256, 512],
            "gamma": [0.99],
            "gradient_steps": [1],
            "train_freq": [ (1, "step") ],
        },
        "td3": {
            "learning_rate": [1e-3, 1e-4, 1e-5],
            "buffer_size": [500_000],
            "tau": [0.005, 0.01],
            "batch_size": [256, 512],
            "gamma": [0.99],
            "gradient_steps": [1],
        }
    }

    # Create grid search jobs
    param_keys = list(grid[algo].keys())
    param_values = list(grid[algo].values())
    all_configs = [dict(zip(param_keys, combo)) for combo in product(*param_values)]
    seeds = [0, 1, 2]

    jobs = [(env_id, algo, config, seed, total_timesteps) for config in all_configs for seed in seeds]

    # Start the grid search on multiple jobs
    print(f"Starting grid search for {len(jobs)} total runs (configs: {len(all_configs)} x seeds: {len(seeds)}) on {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_single)(env_id, algo, config, seed, total_timesteps)
        for (env_id, algo, config, seed, total_timesteps) in jobs
    )


    # Save results to a JSON file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    target_dir = project_root / "Results" / env_id / "RL"
    target_dir.mkdir(parents=True, exist_ok=True)

    results_file = target_dir / f"grid_search_{algo}_{env_id}.json"
    print(f"\nSaving results to: {results_file}") 

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Analyze and summarize the results
    df = analyze_results(results)
    print("\nGrid Search Results Summary:")
    print(df.to_string())

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices=["ppo", "sac", "td3"])
    parser.add_argument('--env', type=str)
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--timesteps', type=int)
    args = parser.parse_args()

    run_grid_search(env_id=args.env, algo=args.algo, n_jobs=args.jobs, total_timesteps=args.timesteps)
