import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse

# Setup directories
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env_id="Pendulum-v1", total_timesteps=100000, algo="ppo"):
    """
    Train using either PPO or SAC based on the 'algo' argument.
    """
    env = None
    try:
        env = gym.make(env_id)
        env = Monitor(env)
        
        # Algorithm selection
        if algo.lower() == "sac":
            # SAC hyperparameters
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                verbose=1,
                tensorboard_log=log_dir
            )
            tb_log_name = f"SAC_{env_id}"
            model_save_path = os.path.join(model_dir, f"SAC_{env_id}_final")
        else:  # Default to PPO
            # PPO hyperparameters
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=5e-4,
                gamma=0.99,
                batch_size=64,
                n_steps=256,
                verbose=1,
                tensorboard_log=log_dir
            )
            tb_log_name = f"PPO_{env_id}"
            model_save_path = os.path.join(model_dir, f"PPO_{env_id}_final")

        # Callback for evaluation and saving best model
        eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.join(model_dir, f"{algo}_{env_id}"),
            log_path=os.path.join(log_dir, f"{algo}_{env_id}"),
            eval_freq=10000,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name=tb_log_name
        )
        
        # Save final model
        model.save(model_save_path)
        
    finally:
        if env is not None:
            env.close()

def test(env_id="Pendulum-v1", render=True, algo="ppo"):
    """
    Test the trained model (either PPO or SAC).
    """
    env = None
    try:
        env = gym.make(env_id, render_mode="human" if render else None)
        
        # Try loading best model, fallback to final if not found
        best_model_path = os.path.join(model_dir, f"{algo}_{env_id}", "best_model")
        final_model_path = os.path.join(model_dir, f"{algo}_{env_id}_final")
        
        if os.path.exists(best_model_path + ".zip"):
            model_path = best_model_path
        elif os.path.exists(final_model_path + ".zip"):
            model_path = final_model_path
        else:
            raise FileNotFoundError(f"No model found for {algo.upper()} in {model_dir}")
        
        # Load the appropriate model
        if algo == "sac":
            model = SAC.load(model_path, env=env)
        else:  # PPO
            model = PPO.load(model_path, env=env)
        
        # Run the policy
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
                
    finally:
        if env is not None:
            env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Training (PPO or SAC)')
    parser.add_argument('--train', action='store_true', help='Train mode')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--env', default="CartPole-v1", help='Environment ID')
    parser.add_argument('--algo', default="ppo", choices=["ppo", "sac"], 
                        help='Algorithm to use (ppo or sac)')
    args = parser.parse_args()

    if args.train:
        train(args.env, algo=args.algo)
    elif args.test:
        test(args.env, algo=args.algo)
    else:
        print("Please specify --train or --test")