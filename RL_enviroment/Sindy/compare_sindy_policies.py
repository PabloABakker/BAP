import os
import sys
import time
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import CustomDynamicsEnv_v2

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
    base = Path(__file__).parent.parent / "Sindy" / "Sindy_best_found_policies" / env_id
    model_path = base / f"sindy_policy_{variant}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"SINDy model not found at: {model_path}")

    print(f"Loaded SINDy model: {variant}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_rl_model(algo, env_id):
    algo = algo.lower()
    base = Path(__file__).parent.parent / "RL" / "models"
    for name in [
        f"{algo.upper()}_{env_id}_final.zip",
        f"{algo.upper()}_{env_id}/best_model.zip",
        f"{algo.upper()}_{env_id}.zip"
    ]:
        path = base / name
        if path.exists():
            env = gym.make(env_id)
            if algo == "ppo":
                return PPO.load(str(path), env=env)
            elif algo == "sac":
                return SAC.load(str(path), env=env)
            elif algo == "td3":
                return TD3.load(str(path), env=env)
    raise FileNotFoundError(f"Could not find RL model for {env_id} and algo={algo}")

def sindy_policy(obs, model):
    return model.predict(obs.reshape(1, -1))[0]

def rl_policy(obs, model):
    action, _ = model.predict(obs, deterministic=True)
    return action

def run_policy(env_id, policy_fn, model, max_steps=500, dt=0.05, seed=None, record=False):
    env = gym.make(env_id, render_mode="rgb_array" if record else None)
    obs, _ = env.reset(seed=seed)
    obs_traj, actions, frames, total_reward = [], [], [], 0

    for _ in range(max_steps):
        action = policy_fn(obs, model)
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = int(np.round(np.clip(action, 0, env.action_space.n - 1)))
        else:
            action = np.clip(np.array(action, dtype=np.float32), env.action_space.low, env.action_space.high)

        obs_traj.append(obs.copy())
        actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if record:
            frames.append(env.render())

        if terminated or truncated:
            break

    env.close()
    return np.array(obs_traj), np.array(actions), total_reward, frames

def compare_all_variants(env_id="Pendulum-v1", algo="ppo", dt=0.05, n_trials=5, save=False):
    output_dir = Path("comparison_outputs") / env_id
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_fields = ["env_id", "trial", "variant", "mean_reward", "mse", "n_steps"]
    csv_path = output_dir / "eval_summary.csv"
    global_csv_path = Path("comparison_outputs") / "global_eval_summary.csv"

    if save:
        for path in [csv_path, global_csv_path]:
            if not path.exists():
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    writer.writeheader()

    rl_model = load_rl_model(algo, env_id)

    stats = {}

    for trial in range(n_trials):
        print(f"\n=== Trial {trial+1}/{n_trials} ===")
        seed = np.random.randint(0, 10000)

        rl_obs, _, rl_reward, rl_frames = run_policy(env_id, rl_policy, rl_model, dt=dt, seed=seed, record=save)
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

        for variant in SINDY_VARIANTS:
            try:
                sindy_model = load_sindy_model(env_id, variant)
            except FileNotFoundError as e:
                print(f"⚠️ {e}")
                continue

            obs, _, reward, frames = run_policy(env_id, sindy_policy, sindy_model, dt=dt, seed=seed, record=save)
            mse = mean_squared_error(rl_obs[:len(obs)], obs[:len(rl_obs)])

            stats.setdefault(variant, {"rewards": [], "MSEs": [], "trajectories": []})
            stats[variant]["rewards"].append(reward)
            stats[variant]["MSEs"].append(mse)
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

    # Generate summary Excel report
    if save:
        df = pd.read_csv(global_csv_path)
        summary = df.groupby(["env_id", "variant"]).agg({
            "mean_reward": ["mean", "std"],
            "mse": ["mean", "std"],
            "n_steps": "mean"
        }).round(4)
        summary.to_excel(Path("comparison_outputs") / "global_summary_stats.xlsx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--algo", default="ppo")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    compare_all_variants(
        env_id=args.env,
        algo=args.algo,
        dt=args.dt,
        n_trials=args.n_trials,
        save=args.save
    )
