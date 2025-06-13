import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from utils import create_vecenv
import imageio 


def test_single_policy(env_id: str, algo: str, model_path: str, vecnorm_path: str, n_episodes: int = 100):
    """
    Tests a single trained policy.

    :param env_id: The environment ID.
    :param algo: The algorithm the model was trained with ('ppo', 'sac', 'td3').
    :param model_path: The path to the saved model .zip file.
    :param vecnorm_path: The path to the VecNormalize statistics.
    :param n_episodes: The number of episodes to test the policy for.
    :return: A dictionary with the evaluation results.
    """
    # Ensure the custom environment is available
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    sys.path.append(str(project_root))
    import Custom

    # Create the test environment
    eval_env = create_vecenv(env_id, training=False, vecnormalize_path=Path(vecnorm_path))

    # Load the trained model
    algo_upper = algo.upper()
    if algo_upper == "PPO":
        model = PPO.load(model_path, env=eval_env)
    elif algo_upper == "SAC":
        model = SAC.load(model_path, env=eval_env)
    elif algo_upper == "TD3":
        model = TD3.load(model_path, env=eval_env)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Evaluate the policy statistically
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )

    eval_env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Finished testing model: {model_path}")
    print(f"Mean reward over {n_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    return {
        "algo": algo,
        "env_id": env_id,
        "model_path": model_path,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_episodes": n_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }

def render_one_episode(env_id: str, algo: str, model_path: str, vecnorm_path: str):
    """
    Renders a single episode of a trained policy.

    :param env_id: The environment ID.
    :param algo: The algorithm the model was trained with.
    :param model_path: The path to the saved model .zip file.
    :param vecnorm_path: The path to the VecNormalize statistics.
    """
    print("\nRendering one episode...")

    # Create a new environment for rendering
    render_env = create_vecenv(env_id, training=False, vecnormalize_path=Path(vecnorm_path), n_envs=1, render_mode="rgb_array")

    # Load the model
    algo_upper = algo.upper()
    if algo_upper == "PPO":
        model = PPO.load(model_path, env=render_env)
    elif algo_upper == "SAC":
        model = SAC.load(model_path, env=render_env)
    elif algo_upper == "TD3":
        model = TD3.load(model_path, env=render_env)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    obs = render_env.reset()
    done = False
    total_reward = 0
    frames = []
    try:
        while not done:
            frames.append(render_env.render())
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = render_env.step(action)
            total_reward += reward
            render_env.render()
    except KeyboardInterrupt:
        print("\nRendering stopped by user.")
    finally:
        render_env.close()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    target_dir = project_root / "Results" / env_id / "RL"
    gif_path = target_dir /  f"{algo}_{env_id}_policy_render.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    
    # The reward is a numpy array because it comes from a VecEnv
    print(f"Finished rendering. Total reward for the episode: {total_reward[0]:.2f}")

def run_policy_evaluation(env_id: str, algo: str, model_path: str, n_episodes: int, render: bool):
    """
    Runs the evaluation for a specified policy and optionally renders one episode.

    :param env_id: The environment ID.
    :param algo: The algorithm the model was trained with.
    :param model_path: The path to the model file.
    :param n_episodes: The number of episodes for evaluation.
    :param render: If True, renders one episode after evaluation.
    """
    model_path_obj = Path(model_path)
    
    # Assume vecnormalize stats are in the parent directory of the model, following the training script's logic
    vecnorm_path = model_path_obj.parent / "vecnormalize.pkl"

    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not vecnorm_path.exists():
        # Add a helpful error message if the specific file is not found
        raise FileNotFoundError(f"VecNormalize file not found at the expected location:\n{vecnorm_path}\nPlease ensure it exists in the same folder as your model.")


    # Run the statistical test
    results = test_single_policy(
        env_id=env_id,
        algo=algo,
        model_path=str(model_path_obj),
        vecnorm_path=str(vecnorm_path),
        n_episodes=n_episodes
    )

    # Save results to a JSON file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    target_dir = project_root / "Results" / env_id / "RL"

    results_file = target_dir / f"test_{algo}_{env_id}_{model_path_obj.stem}.json"

    # Convert numpy arrays to lists for JSON serialization
    results_to_save = results.copy()
    results_to_save["episode_rewards"] = [float(r) for r in results["episode_rewards"]]
    results_to_save["episode_lengths"] = [int(l) for l in results["episode_lengths"]]

    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=4)

    print(f"\nTest results saved to {results_file}")

    # Display summary
    summary_df = pd.DataFrame([
        {
            "Algorithm": results["algo"],
            "Environment": results["env_id"],
            "Mean Reward": results["mean_reward"],
            "Std Reward": results["std_reward"],
            "Episodes": results["n_episodes"],
        }
    ])
    print("\nTest Summary:")
    print(summary_df.to_string(index=False))

    # Render one episode if requested
    if render:
        render_one_episode(
            env_id=env_id,
            algo=algo,
            model_path=str(model_path_obj),
            vecnorm_path=str(vecnorm_path)
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test a trained RL policy.")
    parser.add_argument('--algo', type=str, required=True, choices=["ppo", "sac", "td3"], help="Algorithm the model was trained with.")
    parser.add_argument('--env', type=str, required=True, help="Environment ID.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model's .zip file.")
    parser.add_argument('--episodes', type=int, help="Number of episodes to run the statistical test for.")
    parser.add_argument('--render', action='store_true', help="If set, render one episode after the evaluation.")
    args = parser.parse_args()

    run_policy_evaluation(
        env_id=args.env,
        algo=args.algo,
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=args.render
    )
# Best models:
  # Pendulum : "C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\Sparse_learning_algorithmn\RL\models\PPO_Pendulum-v1_learning_rate0.001_gamma0.99_batch_size256_n_steps1024_ent_coef0.001_seed1\\final_model.zip"
  # CartPole : "C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\Sparse_learning_algorithmn\RL\models\PPO_CartPole-v1_learning_rate0.001_gamma0.99_batch_size256_n_steps1024_ent_coef0.0001_seed0\final_model.zip"