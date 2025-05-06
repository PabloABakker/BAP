import numpy as np
import matplotlib.pyplot as plt
from CustomDynamicsEnv_v2 import CustomDynamicsEnv  # Adjust filename if needed

# Load data
input_refs = np.loadtxt(r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\data.txt")
t_data = np.loadtxt(r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\time.txt")

assert len(input_refs) == len(t_data), "Mismatch in input and time data"

# Initialize environment
env = CustomDynamicsEnv()
state, _ = env.reset()

# Containers for data collection
state_history = [state.copy()]
state_derivatives = []  # Will store [u_dot, w_dot, theta_dot, theta_ddot]
ld_history = []

# Constants
deg2rad = np.pi / 180

for i in range(1, len(t_data)):
    dt = t_data[i] - t_data[i-1]
    env.dt = dt  # Update the environment's time step

    # Convert input reference to ld (dihedral length)
    angle_deg = -input_refs[i] / 9600 * 18
    angle_rad = angle_deg * deg2rad
    ld = env.ly * np.sin(angle_rad)
    ld_history.append(ld)

    # Step the environment
    action = np.array([ld])
    state, _, done, _, _ = env.step(action)
    
    # Store state and derivatives
    state_history.append(state.copy())
    state_derivatives.append([
        env.u_dot, 
        env.w_dot, 
        env.theta_dot, 
        env.theta_ddot
    ])

    if done:
        print(f"Terminated early at step {i}")
        break

# Convert to numpy arrays
state_history = np.array(state_history)
state_derivatives = np.array(state_derivatives)
time_axis = t_data[1:len(state_history)]  # Align time with steps taken

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# State variables
axs[0].plot(time_axis, state_history[1:, 0], label='u (m/s)', color='tab:blue')
axs[0].set_ylabel('u (m/s)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(time_axis, state_history[1:, 1], label='w (m/s)', color='tab:orange')
axs[1].set_ylabel('w (m/s)')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(time_axis, state_history[1:, 2], label='θ (rad)', color='tab:green')
axs[2].set_ylabel('θ (rad)')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(time_axis, state_history[1:, 3], label='θ̇ (rad/s)', color='tab:red')
axs[3].set_ylabel('θ̇ (rad/s)')
axs[3].set_xlabel('Time (s)')
axs[3].legend()
axs[3].grid(True)

plt.suptitle("State Variables Over Time")
plt.tight_layout()
plt.show()

# Plot derivatives separately
fig2, axs2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axs2[0].plot(time_axis, state_derivatives[:, 0], label='u_dot (m/s²)', color='tab:blue')
axs2[0].set_ylabel('u_dot (m/s²)')
axs2[0].legend()
axs2[0].grid(True)

axs2[1].plot(time_axis, state_derivatives[:, 1], label='w_dot (m/s²)', color='tab:orange')
axs2[1].set_ylabel('w_dot (m/s²)')
axs2[1].legend()
axs2[1].grid(True)

axs2[2].plot(time_axis, state_derivatives[:, 3], label='θ_ddot (rad/s²)', color='tab:purple')
axs2[2].set_ylabel('θ_ddot (rad/s²)')
axs2[2].set_xlabel('Time (s)')
axs2[2].legend()
axs2[2].grid(True)

plt.suptitle("State Derivatives Over Time")
plt.tight_layout()
plt.show()

# Plot control input
plt.figure(figsize=(12, 4))
plt.plot(time_axis, ld_history, label='Control Input (ld)', color='tab:brown')
plt.xlabel('Time (s)')
plt.ylabel('ld (m)')
plt.title('Control Input History')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()