import sys
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO, SAC, TD3
from sklearn.metrics import mean_squared_error
import imageio.v2 as imageio
import argparse
import csv
import pandas as pd


# Add project root to import custom envs
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))
import Custom
from RL.utils import create_vecenv

# Define model variants
SINDY_VARIANTS = [
    "poly_regular_STLSQ",
    "poly_regular_ConstrainedSR3",
    "poly_hardware_STLSQ",
    "poly_hardware_ConstrainedSR3",
    "hw_regular_STLSQ",
    "hw_regular_ConstrainedSR3",
    "hw_hardware_STLSQ",
    "hw_hardware_ConstrainedSR3",]

def load_sindy_model(env_id, variant):
    base = project_root / "Results" / env_id / "SINDY"
    model_path = base / variant / f"sindy_policy_{variant}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"SINDy model not found at: {model_path}")

    print(f"Loaded SINDy model: {variant}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_rl_model(algo, env_id, model_path, save=False):
    """
    Loads an RL model and its associated, correctly normalized environment.
    
    Returns:
        tuple: (loaded_model, normalized_environment)
    """

    algo = algo.lower()
    path = Path(model_path) 

    if not path.exists():
        raise FileNotFoundError(f"Could not find RL model at: {path}")

    # Find the vecnormalize stats associated with this specific model
    vecnorm_path = path.parent / "vecnormalize.pkl"

    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found at: {vecnorm_path}. It is required for this model.")

    # Use your utility to create the properly normalized environment
    if save:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = create_vecenv(env_id, training=False, vecnormalize_path=str(vecnorm_path), render_mode=render_mode)
    
    # Load the model with the correct environment
    model_loader = {"ppo": PPO, "sac": SAC, "td3": TD3}
    if algo not in model_loader:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    model = model_loader[algo].load(str(path), env=env)
    
    return model, env

def sindy_policy(obs, model):
    # Predict returns a 2D array like [[action]], we flatten it to a 1D array like [action]
    action = model.predict(obs.reshape(1, -1))
    return np.asarray(action).flatten()

def rl_policy(obs, model):
    action, _ = model.predict(obs, deterministic=True)
    # Ensure the output is always a numpy array
    return np.asarray(action)

def run_policy(env, policy_fn, model, max_steps=500, dt=0.05, seed=None, record=False):

    env.seed(seed)
    obs = env.reset()
    
    obs_traj, actions, frames, total_reward = [], [], [], 0

    for _ in range(max_steps):
        policy_obs =  obs

        action = policy_fn(policy_obs, model)

        if isinstance(env.action_space, gym.spaces.Discrete):
            action = int(np.round(np.clip(action, 0, env.action_space.n - 1)))
            action = np.array([[action]])
        else:
            action = np.clip(action, env.action_space.low, env.action_space.high)
            if np.isscalar(action):
                action = np.array([[action]])
            elif action.ndim == 1:
                action = np.expand_dims(action, axis=0)
            elif action.ndim == 2 and action.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 in action but got shape {action.shape}")

        obs_traj.append(policy_obs)              
        actions.append(action.flatten())         

        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        if record:
            frames.append(env.render())

        if done[0]:
            break

            
    return np.squeeze(np.array(obs_traj)), np.array(actions), total_reward, frames

def compare_all_variants(env_id, algo, dt, model_path, max_steps, n_trials, save=False):
    output_dir = project_root/ "Results" / env_id / "comparison_RL_SINDy"
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_fields = ["env_id", "trial", "variant", "mean_reward", "mse", "n_steps"]
    csv_path = output_dir / "eval_summary.csv"
    global_csv_path = output_dir / "global_eval_summary.csv"

    if save:
        for path in [csv_path, global_csv_path]:
            if not path.exists():
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    writer.writeheader()

    rl_model, rl_env = load_rl_model(algo, env_id, model_path, save=save)

    stats = {}

    for trial in range(n_trials):
        print(f"\n=== Trial {trial+1}/{n_trials} ===")
        seed = np.random.randint(0, 10000)

        rl_obs, _, rl_reward, rl_frames = run_policy(rl_env, rl_policy, rl_model, max_steps, seed=seed, record=save)
        stats.setdefault("RL", {"rewards": [], "trajectories": []})
        stats["RL"]["rewards"].append(rl_reward)
        stats["RL"]["trajectories"].append(rl_obs)

        if save:
            row = {
                "env_id": env_id,
                "trial": trial + 1,
                "variant": "RL",
                "mean_reward": rl_reward,
                "mse": float('nan'),
                "n_steps": len(rl_obs)
            }
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
            with open(global_csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
            base = output_dir / f"trial_{trial+1}_RL"
            imageio.mimsave(str(base) + ".gif", rl_frames, fps=int(1/dt))

        for variant in SINDY_VARIANTS:
            try:
                sindy_model = load_sindy_model(env_id, variant)
            except FileNotFoundError as e:
                print(f"⚠️ {e}")
                continue

                
            obs, _, reward, frames = run_policy(rl_env, sindy_policy, sindy_model, max_steps, seed=seed, record=save)
            mse = mean_squared_error(rl_obs[:len(obs)], obs[:len(rl_obs)])

            stats.setdefault(variant, {"rewards": [], "MSEs": [], "trajectories": []})
            stats[variant]["rewards"].append(reward)
            stats[variant]["MSEs"].append(mse),
            stats[variant]["trajectories"].append(obs)

            if save:
                base = output_dir / f"trial_{trial+1}_{variant}"
                imageio.mimsave(str(base) + ".gif", frames, fps=int(1/dt))

                row = {
                    "env_id": env_id,
                    "trial": trial + 1,
                    "variant": variant,
                    "mean_reward": reward,
                    "mse": mse,
                    "n_steps": len(obs)
                }
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
                with open(global_csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        if save:
            plt.figure(figsize=(10, 5))
            time_axis = np.arange(len(rl_obs)) * dt
            for i in range(rl_obs.shape[1]):
                plt.plot(time_axis, rl_obs[:, i], label=f"RL state_{i}", linewidth=2)

            for variant in SINDY_VARIANTS:
                if variant in stats:
                    traj = stats[variant]["trajectories"][-1]
                    for i in range(rl_obs.shape[1]):
                        plt.plot(time_axis[:len(traj)], traj[:len(time_axis), i], '--', label=f"{variant} state_{i}")

            plt.title(f"Trial {trial+1}: Trajectories Comparison")
            plt.xlabel("Time (s)")
            plt.ylabel("State Value")
            plt.legend(fontsize="small", ncol=2)
            plt.tight_layout()
            plt.savefig(output_dir / f"trajectories_trial_{trial+1}.png")
            plt.close()

    # Print Summary
    print("\n==== Summary Across Trials ====")
    print(f"RL mean reward: {np.mean(stats['RL']['rewards']):.2f}")

    for variant in SINDY_VARIANTS:
        if variant in stats:
            print(f"\n--- {variant} ---")
            print(f"Avg reward: {np.mean(stats[variant]['rewards']):.2f}")
            print(f"Avg MSE (states): {np.mean(stats[variant]['MSEs']):.6f}")

    # Generate summary csv report
    if save:
        df = pd.read_csv(global_csv_path)
        summary = df.groupby(["env_id", "variant"]).agg({
            "mean_reward": ["mean", "std"],
            "mse": ["mean", "std"],
            "n_steps": "mean"
        }).round(4)
        summary.to_csv(output_dir / "global_summary_stats.csv")
    
    rl_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--algo", default="ppo")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model's .zip file.")
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    compare_all_variants(
        env_id=args.env,
        algo=args.algo,
        dt=args.dt,
        model_path=args.model_path,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        save=args.save
    )
