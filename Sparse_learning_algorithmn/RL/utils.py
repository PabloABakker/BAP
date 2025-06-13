import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from typing import List, Dict, Any
from pathlib import Path
from matplotlib import pyplot as plt



def make_base_env(env_id: str, seed: int = 0, render_mode: str = None):
    """
    Correct Implementation: This function creates and returns an actual
    Gymnasium environment object.
    """
    env = gym.make(env_id, render_mode=render_mode)
    return env

def create_vecenv(env_id: str, training: bool = True, n_envs: int = 1, seed: int = 0, vecnormalize_path: str = None, render_mode: str = None):

    env_fns = [lambda: make_base_env(env_id, seed=seed + i, render_mode=render_mode) for i in range(n_envs)]

    if n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    # Correctly load or create the VecNormalize wrapper
    if vecnormalize_path and Path(vecnormalize_path).exists():
        print(f"Loading VecNormalize stats from: {vecnormalize_path}")
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        # Only normalize rewards during training
        print("Creating new VecNormalize instance.")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=training, clip_obs=10.0)

    return vec_env



def analyze_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze and summarize grid search results."""
    df = pd.DataFrame(results)

    if 'config' in df.columns and len(df) > 0 and isinstance(df['config'].iloc[0], dict):
        config_df = pd.json_normalize(df['config'])
        df = pd.concat([df.drop('config', axis=1), config_df], axis=1)

    group_cols = [col for col in df.columns if col not in ["best_reward", "mean_reward", "std_reward", "model_path", "vecnorm_path", "seed"]]

    summary = df.groupby(group_cols).agg(
        mean_best_reward=('best_reward', 'mean'),
        std_best_reward=('best_reward', 'std'),
        mean_avg_reward=('mean_reward', 'mean'),
        std_avg_reward=('mean_reward', 'std')
    ).reset_index()

    return summary




def load_eval_results(log_path: str):
    eval_file = os.path.join(log_path, "evaluations.npz")
    if not os.path.exists(eval_file):
        return {}

    data = np.load(eval_file)
    return {
        "timesteps": data["timesteps"],
        "episode_rewards": data["results"],
        "episode_lengths": data["ep_lengths"],
    }




def plot_results(results: List[Dict[str, Any]], save_path: str = None) -> None:
    """Plot learning curves from evaluation results."""
    plt.figure(figsize=(12, 8))

    for result in results:
        log_folder_name = Path(result["model_path"]).parent.name
        log_folder_path = Path("logs") / log_folder_name

        eval_data = load_eval_results(log_folder_path)
        if not eval_data:
            continue

        rewards_array = np.asarray(eval_data["episode_rewards"])
        if rewards_array.ndim == 1:
            rewards_array = rewards_array[np.newaxis, :]

        label = f"{result['algo']} {result['config']} (seed {result['seed']})"
        plt.plot(eval_data["timesteps"], np.mean(rewards_array, axis=1), label=label)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.title("Learning Curves (Mean Eval Reward per Timestep)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()