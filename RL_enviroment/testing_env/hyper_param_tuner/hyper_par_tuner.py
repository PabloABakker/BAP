import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

class Particles:
    def __init__(self, function, N, grid_space):
        """
        Initialize the grid space: grid_space should be a matrix of size d x 2,
        where the first column is the minimum and the second column the maximum for each dimension.
        """
        self.f = function
        self.N = N
        self.grid_space = grid_space
        self.dimensions = grid_space.shape[0]
        self.min_values = grid_space[:, 0]
        self.max_values = grid_space[:, 1]

        self.locations = np.zeros((self.dimensions, N))
        self.velocities = np.zeros_like(self.locations)
        self.best_personal_locations = np.zeros_like(self.locations)
        self.global_best_location = np.zeros((self.dimensions,))

        self.initialize()

    def initialize(self):
        self.locations = np.random.uniform(
            self.min_values[:, None],
            self.max_values[:, None],
            size=(self.dimensions, self.N)
        )
        self.best_personal_locations = self.locations.copy()
        self.update_gb()

    def update_location(self):
        self.locations += self.velocities
        # Ensure particles stay within bounds
        self.locations = np.clip(self.locations, self.min_values[:, None], self.max_values[:, None])

    def update_velocities(self, momentum, c_pb, c_gb):
        r1 = np.random.rand(self.dimensions, self.N)
        r2 = np.random.rand(self.dimensions, self.N)
        cognitive = c_pb * r1 * (self.best_personal_locations - self.locations)
        social = c_gb * r2 * (self.global_best_location[:, None] - self.locations)
        self.velocities = momentum * self.velocities + cognitive + social

    def update_pb(self):
        for i in range(self.N):
            if self.f(self.locations[:, i]) < self.f(self.best_personal_locations[:, i]):
                self.best_personal_locations[:, i] = self.locations[:, i]

    def update_gb(self):
        best_index = np.argmin([self.f(self.best_personal_locations[:, i]) for i in range(self.N)])
        best_candidate = self.best_personal_locations[:, best_index]
        if self.f(best_candidate) < self.f(self.global_best_location):
            self.global_best_location = best_candidate

class PSO:
    def __init__(self, function, N, grid_space, momenta, cognitive_constant, social_constant, maximum_iterations):
        self.f = function
        self.N = N
        self.dimensions = grid_space.shape[0]
        self.maximum_iterations = maximum_iterations
        self.Mmin, self.Mmax = momenta
        self.c_pb = cognitive_constant
        self.c_gb = social_constant
        self.particles = Particles(function, N, grid_space)
        self.history = []
        self.global_best_history = []
        self.best_mse_history = []

    def optimize(self):
        momentum_schedule = np.linspace(self.Mmax, self.Mmin, self.maximum_iterations)
        for t in tqdm(range(self.maximum_iterations), desc="Optimizing"):
            self.particles.update_velocities(momentum_schedule[t], self.c_pb, self.c_gb)
            self.particles.update_location()
            
            self.particles.update_pb()
            self.particles.update_gb()
            
            self.history.append(self.particles.locations.copy())
            self.global_best_history.append(self.particles.global_best_location.copy())
            gb_location = self.particles.global_best_location
            mse = self.f(gb_location)
            self.best_mse_history.append(mse)

        plt.plot(range(self.maximum_iterations), self.best_mse_history)
        plt.xlabel("Iteration")
        plt.ylabel("Best MSE")
        plt.title("PSO Optimization Progress")
        plt.grid(True)
        plt.show()

    def get_best_location(self):
        return self.particles.global_best_location

    def animate(self):
        if self.dimensions != 2:
            print("Animation only supported for 2D problems.")
            return

        fig, ax = plt.subplots()
        ax.set_xlim(self.particles.min_values[0], self.particles.max_values[0])
        ax.set_ylim(self.particles.min_values[1], self.particles.max_values[1])
        scatter = ax.scatter([], [], c='r', s=50)

        def update(frame):
            scatter.set_offsets(self.history[frame].T)
            return scatter,

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=100, blit=True)
        ani.save('particle_swarm_optimization.gif', writer='pillow')
        plt.show()

