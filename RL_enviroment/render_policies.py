# import pickle
# import numpy as np
# import gymnasium as gym
# import argparse
# import os
# import sys
# import time

# # --- Add path for custom environment ---
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import CustomDynamicsEnv_v2  # Ensure env registration happens here

# # --- Base path to policies ---
# POLICY_DIR = r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\RL_enviroment\Sindy_best_found_policies"

# def load_policy(file_path):
#     with open(file_path, "rb") as f:
#         return pickle.load(f)

# def evaluate_policy(env_id, policy, policy_name, dt=0.05, max_steps=500):
#     env = gym.make(env_id, render_mode="human")
#     obs, _ = env.reset()
#     total_reward = 0

#     print(f"\n--- Running {policy_name} policy ---")

#     for _ in range(max_steps):
#         obs_input = obs.reshape(1, -1)
#         action = policy.predict(obs_input)[0]

#         if isinstance(env.action_space, gym.spaces.Discrete):
#             action_env = int(np.round(np.clip(action, 0, env.action_space.n - 1)))
#         else:
#             action_env = np.array(action, dtype=np.float32).reshape(-1)
#             action_env = np.clip(action_env, env.action_space.low, env.action_space.high)

#         obs, reward, terminated, truncated, _ = env.step(action_env)
#         total_reward += reward

#         time.sleep(dt)
#         if terminated or truncated:
#             break

#     env.close()
#     print(f"{policy_name} total reward: {total_reward:.2f}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Render SINDy and MLP policies on a Gym environment")
#     parser.add_argument("--env", type=str, required=True, help="Gym environment ID (e.g., MyCustomEnv-v0)")
#     parser.add_argument("--dt", type=float, default=0.05, help="Time delay between steps (default: 0.05)")
#     parser.add_argument("--steps", type=int, default=500, help="Maximum number of steps per rollout (default: 500)")
#     args = parser.parse_args()

#     # --- Load policies ---
#     sindy_path = os.path.join(POLICY_DIR, "sindy_policy.pkl")
#     mlp_path = os.path.join(POLICY_DIR, "mlp_policy.pkl")

#     if not os.path.exists(sindy_path) or not os.path.exists(mlp_path):
#         raise FileNotFoundError("Could not find sindy_policy.pkl or mlp_policy.pkl in the specified directory.")

#     sindy_policy = load_policy(sindy_path)
#     mlp_policy = load_policy(mlp_path)

#     # --- Evaluate both policies ---
#     evaluate_policy(args.env, sindy_policy, "SINDy", dt=args.dt, max_steps=args.steps)
#     input("\nPress Enter to continue to MLP policy...\n")
#     evaluate_policy(args.env, mlp_policy, "MLP", dt=args.dt, max_steps=args.steps)


# This needs tweeking