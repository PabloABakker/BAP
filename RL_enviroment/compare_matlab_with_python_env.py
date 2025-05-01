import numpy as np
import matplotlib.pyplot as plt
from CustomDynamicsEnv_v2 import CustomDynamicsEnv  # Adjust filename if needed

# Load data
input_refs = np.loadtxt(r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\data.txt")       # shape (N,)
t_data = np.loadtxt(r"C:\Users\pablo\OneDrive\Bureaublad\Python\Machine learning\BAP_TOTAL\Bap_self\BAP\time.txt")       # shape (N,)

assert len(input_refs) == len(t_data), "Mismatch in input and time data"

# Initialize environment
env = CustomDynamicsEnv()
state, _ = env.reset()

# Container for states
state_history = [state.copy()]

# Constant for conversion
deg2rad = np.pi / 180

for i in range(1, len(t_data)):
    dt = t_data[i] - t_data[i - 1]
    env.dt = dt

    # Step 1: Convert input reference to angle in radians
    angle_deg = -input_refs[i] / 9600 * 18
    angle_rad = angle_deg * deg2rad

    # Step 2: Convert angle to ld using the environment parameter ly
    ld = env.ly * np.sin(angle_rad)

    # Feed to environment
    action = np.array([ld])
    state, _, done, _, _ = env.step(action)
    state_history.append(state.copy())

    if done:
        print(f"Terminated early at step {i}")
        break

# Convert state history to array
state_history = np.array(state_history)
time_axis = t_data[:len(state_history)]

# Plotting
plt.figure(figsize=(10, 6))
labels = ['u', 'w', 'theta', 'theta_dot']
for i in range(state_history.shape[1]):
    plt.plot(time_axis, state_history[:, i], label=labels[i])
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.title("State Evolution with Reference-Based Control Inputs")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
