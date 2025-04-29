import numpy as np
from CustomDynamicsEnv_v2 import CustomDynamicsEnv  # Assuming your env is saved in this file
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
import matplotlib.pyplot as plt

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
        X_dot: State derivatives (n_trajectories x steps_per_trajectory x state_dim)
        U: Control inputs (n_trajectories x steps_per_trajectory x action_dim)
    """
    X = []
    X_dot = []
    U = []
    
    for _ in range(n_trajectories):
        # Reset environment
        x, _ = env.reset()
        x_traj = []
        x_dot_traj = []
        u_traj = []
        
        for _ in range(steps_per_trajectory):
            # Random action
            u = env.action_space.sample()
            
            # Store current state and action
            x_traj.append(x)
            u_traj.append(u)
            
            # Take step in environment
            x_next, _, _, _,_ = env.step(u)
            
            # Calculate numerical derivative (simple forward difference)
            dt = env.dt
            x_dot = (x_next - x) / dt
            x_dot_traj.append(x_dot)
            
            x = x_next
        
        # Convert to arrays
        x_traj = np.array(x_traj)
        x_dot_traj = np.array(x_dot_traj)
        u_traj = np.array(u_traj)
        
        # Add noise if specified
        if noise_level > 0:
            x_traj += np.random.normal(scale=noise_level, size=x_traj.shape)
            x_dot_traj += np.random.normal(scale=noise_level, size=x_dot_traj.shape)
        
        X.append(x_traj)
        X_dot.append(x_dot_traj)
        U.append(u_traj)
    
    return np.stack(X), np.stack(X_dot), np.stack(U)

def save_data(X, X_dot, U, filename='dynamics_data.npz'):
    """Save generated data to a file"""
    np.savez(filename, X=X, X_dot=X_dot, U=U)
    print(f"Data saved to {filename}")

def plot_sample_trajectory(X, X_dot, U, traj_idx=0):
    """Plot a sample trajectory for visualization"""
    t = np.arange(X.shape[1]) * env.dt
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot states
    for i in range(X.shape[2]):
        axs[0].plot(t, X[traj_idx, :, i], label=f'State {i}')
    axs[0].set_ylabel('States')
    axs[0].legend()
    
    # Plot derivatives
    for i in range(X_dot.shape[2]):
        axs[1].plot(t, X_dot[traj_idx, :, i], label=f'Derivative {i}')
    axs[1].set_ylabel('Derivatives')
    axs[1].legend()
    
    # Plot controls
    for i in range(U.shape[2]):
        axs[2].plot(t, U[traj_idx, :, i], label=f'Control {i}')
    axs[2].set_ylabel('Controls')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create environment
    env = CustomDynamicsEnv()
    
    # Generate data
    X, X_dot, U = generate_training_data(env, 
                                        n_trajectories=100,
                                        steps_per_trajectory=200,
                                        noise_level=0.001)
    
    # Save data
    save_data(X, X_dot, U)
    
    # Visualize a sample trajectory
    plot_sample_trajectory(X, X_dot, U)