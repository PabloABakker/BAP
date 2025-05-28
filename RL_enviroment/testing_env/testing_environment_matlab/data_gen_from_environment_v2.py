import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import CustomDynamicsEnv_v2  # This runs the registration

def generate_training_data(env, n_trajectories=50, steps_per_trajectory=200, noise_level=0.0):
    """
    Generate training data from the custom environment.
    
    Parameters:
        env: Gym environment instance
        n_trajectories: Number of trajectories to generate
        steps_per_trajectory: Length of each trajectory
        noise_level: Amount of Gaussian noise to add
        
    Returns:
        X: State measurements (n_trajectories x steps_per_trajectory x state_dim)
        U: Control inputs (n_trajectories x steps_per_trajectory x action_dim)
    """
    X = []
    U = []
    dt = 0.001
    
    for _ in range(n_trajectories):
        # Reset environment
        x, _ = env.reset()
        x_traj = []
        u_traj = []
        
        for _ in range(steps_per_trajectory):
            # Random action
            # u = env.action_space.sample()
            u = np.array([0.01*np.sin(2*np.pi*dt*_*0.2)])  # Example control input
            # u = np.array([0])

            # Store current state and action
            x_traj.append(x)
            u_traj.append(u)
            
            # Take step in environment
            x_next, _, _, _,_ = env.step(u)
            
            x = x_next
        
        # Convert to arrays
        x_traj = np.array(x_traj)
        u_traj = np.array(u_traj)

        # Add noise if specified
        if noise_level > 0:
            x_traj += np.random.normal(scale=noise_level, size=x_traj.shape)
        
        X.append(x_traj)
        U.append(u_traj)
    
    return np.stack(X), np.stack(U)

def save_data(X, U, filename='dynamics_data.npz'):
    """Save generated data to a file"""
    np.savez(filename, X=X, U=U)
    print(f"Data saved to {filename}")

def plot_sample_trajectory(X, U, traj_idx=0):
    """Plot a sample trajectory for visualization"""
    t = np.arange(X.shape[1]) * env.dt
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot states
    for i in range(X.shape[2]):
        axs[0].plot(t, X[traj_idx, :, i], label=f'State {i}')
    axs[0].set_ylabel('States')
    axs[0].legend()
    
    # Plot controls
    for i in range(U.shape[2]):
        axs[1].plot(t, U[traj_idx, :, i], label=f'Control {i}')
    axs[1].set_ylabel('Controls')
    axs[1].set_xlabel('Time')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create environment
    from CustomDynamicsEnv_v2 import CustomDynamicsEnv
    env = CustomDynamicsEnv()
    
    # Generate data
    X, U = generate_training_data(env, 
                                        n_trajectories=2,
                                        steps_per_trajectory=7500,
                                        noise_level=0)
    
    # Save data
    save_data(X, U)
    
    print(f"Generated data shapes: X: {X.shape}, U: {U.shape}")
    # Visualize a sample trajectory
    plot_sample_trajectory(X, U)