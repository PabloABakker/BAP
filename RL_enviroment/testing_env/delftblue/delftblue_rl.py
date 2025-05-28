import os
import gymnasium as gym
import numpy as np
import torch
import sys
import argparse
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Add path to custom env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import CustomDynamicsEnv_v2  # This runs the registration

ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}

def make_env(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return DummyVecEnv([_init])

def train_rl(config, env_id, algo_name):
    env = make_env(env_id)
    algo_class = ALGO_MAP[algo_name]
    model = algo_class(
        "MlpPolicy",
        env,
        learning_rate=config["lr"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        verbose=0,
        seed=config["seed"]
    )
    model.learn(total_timesteps=20000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    tune.report(mean_reward=mean_reward)

def run_tune(env_id, algo_name):
    config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "gamma": tune.uniform(0.9, 0.999),
        "batch_size": tune.choice([32, 64, 128]),
        "seed": tune.randint(0, 10000),
    }

    search_alg = OptunaSearch()
    scheduler = ASHAScheduler(metric="mean_reward", mode="max")

    tune.run(
        tune.with_parameters(train_rl, env_id=env_id, algo_name=algo_name),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=20,
        scheduler=scheduler,
        search_alg=search_alg,
        local_dir="ray_results",
        name=f"tune_{algo_name}_{env_id}"
    )

def run_train(env_id, algo_name):
    env = make_env(env_id)
    algo_class = ALGO_MAP[algo_name]
    model = algo_class("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save(f"{algo_name}_{env_id}_model")

def run_test(env_id, algo_name):
    env = make_env(env_id)
    algo_class = ALGO_MAP[algo_name]
    model_path = f"{algo_name}_{env_id}_model"
    model = algo_class.load(model_path, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Test reward: mean={mean_reward:.2f}, std={std_reward:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CustomDynamicsEnv-v2")
    parser.add_argument("--algo", type=str, choices=ALGO_MAP.keys(), default="ppo")
    parser.add_argument("--mode", type=str, choices=["train", "test", "tune"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.env, args.algo)
    elif args.mode == "test":
        run_test(args.env, args.algo)
    elif args.mode == "tune":
        run_tune(args.env, args.algo)

if __name__ == "__main__":
    main()
