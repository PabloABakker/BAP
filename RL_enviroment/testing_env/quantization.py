import matplotlib.pyplot as plt
from itertools import combinations, product
from matplotlib.widgets import Slider
import numpy as np

class FixedPointVisualizer:
    def __init__(self, int_bits=None, frac_bits=None, draw=False, init_int_bits=3, init_frac_bits=3):
        self.draw = draw

        if draw:
            self.init_int_bits = init_int_bits
            self.init_frac_bits = init_frac_bits
            self._setup_plot()
        else:
            if int_bits is None or frac_bits is None:
                raise ValueError("You must provide int_bits and frac_bits if draw is False.")
            self.int_bits = int_bits
            self.frac_bits = frac_bits
            self.fixed_values = self._get_fixed_point_values(int_bits, frac_bits)

    def _get_fixed_point_values(self, int_bits, frac_bits):
        total_bits = int_bits + frac_bits
        positions = list(range(total_bits))

        values = set()
        # Include zero
        values.add(0.0)
        
        # Positive values
        for k in range(1, 3):  # 1 or 2 bits set
            for combo in combinations(positions, k):
                value = 0
                for pos in combo:
                    value += 2 ** (int_bits - pos - 1)
                values.add(value)
                values.add(-value)  # Add negative counterpart
        
        return sorted(values)

    def _get_rule_based_values(self, int_bits, frac_bits):
        m_list = list(range(int_bits))
        n_list = list(range(frac_bits + 1))
        values = set()

        # Include zero
        values.add(0.0)
        
        # Positive values
        for m in m_list:
            values.add(2 ** m)
            values.add(-(2 ** m))
        for n in n_list:
            values.add(2 ** (-n))
            values.add(-(2 ** (-n)))
        for m, n in product(m_list, n_list):
            values.add(2 ** m + 2 ** (-n))
            values.add(-(2 ** m + 2 ** (-n)))
            values.add(2 ** m - 2 ** (-n))
            values.add(-(2 ** m - 2 ** (-n)))
        for m in m_list:
            for x in range(m):
                values.add(2 ** m + 2 ** x)
                values.add(-(2 ** m + 2 ** x))
                values.add(2 ** m - 2 ** x)
                values.add(-(2 ** m - 2 ** x))
        for n in n_list:
            for y in range(n):
                values.add(2 ** (-n) + 2 ** (-y))
                values.add(-(2 ** (-n) + 2 ** (-y)))
        return sorted(values)

    def _setup_plot(self):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        
        ax_int = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_frac = plt.axes([0.25, 0.1, 0.65, 0.03])
        
        self.slider_int = Slider(ax_int, 'Int bits', 1, 8, valinit=self.init_int_bits, valstep=1)
        self.slider_frac = Slider(ax_frac, 'Frac bits', 0, 8, valinit=self.init_frac_bits, valstep=1)
        
        self.slider_int.on_changed(self._update)
        self.slider_frac.on_changed(self._update)
        
        self._plot_all(self.init_int_bits, self.init_frac_bits)
        plt.show()

    def _update(self, val):
        int_bits = int(self.slider_int.val)
        frac_bits = int(self.slider_frac.val)
        self._plot_all(int_bits, frac_bits)

    def _plot_all(self, int_bits, frac_bits):
        fixed_values = self._get_fixed_point_values(int_bits, frac_bits)
        rule_values = self._get_rule_based_values(int_bits, frac_bits)

        self.ax.clear()

        for v in fixed_values:
            self.ax.plot([v, v], [0.05, 0.25], color='blue')
        for v in rule_values:
            self.ax.plot([v, v], [-0.25, -0.05], color='red')

        self.ax.set_title(f'Ticks: ≤2 bits (blue) and rule-based (red) — int={int_bits}, frac={frac_bits}')
        self.ax.set_xlabel('Value')
        self.ax.set_yticks([])
        self.ax.grid(True, axis='x')
        plt.draw()

    def quantize(self, input_values):
        """Quantizes a numpy array (1D or 2D) to the closest values in the fixed-point set."""
        if self.draw:
            raise RuntimeError("Cannot quantize while in draw mode.")
        if not hasattr(self, 'fixed_values'):
            self.fixed_values = self._get_fixed_point_values(self.int_bits, self.frac_bits)
        
        input_array = np.asarray(input_values)
        # Reshape fixed_values for broadcasting
        fixed_arr = np.array(self.fixed_values)[:, np.newaxis]
        # Find closest values using broadcasting
        abs_diff = np.abs(fixed_arr - input_array.ravel())
        closest_indices = np.argmin(abs_diff, axis=0)
        quantized = np.array(self.fixed_values)[closest_indices]
        return quantized.reshape(input_array.shape)

# Example usage 1:
# fpv = FixedPointVisualizer(int_bits=3, frac_bits=3, draw=False) 
# random_array = np.random.rand(3, 4)-0.5  # Values between -0.5 and 0.5
# quantized_values = fpv.quantize(random_array)
# print("Original values:")
# print(random_array)
# print("Quantized values:")
# print(quantized_values)
# Example usage 2:
# fpv = FixedPointVisualizer(draw=True, init_int_bits=3, init_frac_bits=3)
