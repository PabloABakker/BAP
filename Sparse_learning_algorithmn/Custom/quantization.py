import matplotlib.pyplot as plt
from itertools import product
from matplotlib.widgets import Slider
import numpy as np

class FixedPointVisualizer:
    def __init__(self, int_bits=None, frac_bits=None, draw=False, init_int_bits=3, init_frac_bits=3, input_array=None):
        self.draw = draw
        self.input_array = input_array

        if draw:
            self.init_int_bits = init_int_bits
            self.init_frac_bits = init_frac_bits
            self._setup_plot()
        else:
            if int_bits is None or frac_bits is None:
                raise ValueError("You must provide int_bits and frac_bits if draw is False.")
            self.int_bits = int_bits
            self.frac_bits = frac_bits
            self.fixed_values = self._get_rule_based_values(int_bits, frac_bits)

    def _get_rule_based_values(self, int_bits, frac_bits):
        m_list = list(range(int_bits))
        n_list = list(range(frac_bits + 1))
        values = set()
        values.add(0.0)
        
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
                values.add(2 ** (-n) - 2 ** (-y))
                values.add(-(2 ** (-n) + 2 ** (-y)))
                values.add(-(2 ** (-n) - 2 ** (-y)))
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
        rule_values = self._get_rule_based_values(int_bits, frac_bits)
        self.ax.clear()

        # Helper: Determine if value is "special"
        def is_special(value):
            remainder = abs(value) % 1.0
            return np.isclose(remainder, 0.25, atol=1e-6) or \
                np.isclose(remainder, 0.5, atol=1e-6) or \
                np.isclose(remainder, 0.75, atol=1e-6)

        # Plot red (or cyan) ticks
        for v in rule_values:
            color = 'cyan' if is_special(v) else 'red'
            self.ax.plot([v, v], [-0.15, -0.05], color=color, linewidth=1)
            if is_special(v):
                self.ax.text(v, -0.17, f'{v:.2f}', ha='center', va='top', fontsize=7, rotation=90, color=color)

        # Plot magenta lines from input to quantized
        if self.input_array is not None:
            input_flat = np.asarray(self.input_array).ravel()
            quantized = self._quantize_to_rule_values(input_flat, rule_values)

            # Plot original input points
            self.ax.plot(input_flat, np.zeros_like(input_flat), 'mo', label='Input')

            for x, q in zip(input_flat, quantized):
                self.ax.plot([x, q], [0, -0.04], color='magenta', linestyle=(0, (3, 3)), linewidth=1)

        self.ax.set_title(f'Rule-based Quantization — int={int_bits}, frac={frac_bits}')
        self.ax.set_xlabel('Value')
        self.ax.set_yticks([])
        self.ax.grid(True, axis='x')
        self.ax.legend(loc='upper right')
        plt.draw()

    def _quantize_to_rule_values(self, input_array, rule_values):
        rule_arr = np.array(rule_values)[:, np.newaxis]  # shape (N, 1)
        abs_diff = np.abs(rule_arr - input_array.ravel())  # shape (N, M)
        closest_indices = np.argmin(abs_diff, axis=0)
        quantized = np.array(rule_values)[closest_indices]
        return quantized.reshape(input_array.shape)

    def quantize(self, input_values):
        if self.draw:
            raise RuntimeError("Cannot quantize while in draw mode.")
        if not hasattr(self, 'fixed_values'):
            self.fixed_values = self._get_rule_based_values(self.int_bits, self.frac_bits)
        
        input_array = np.asarray(input_values)
        rule_arr = np.array(self.fixed_values)[:, np.newaxis]
        abs_diff = np.abs(rule_arr - input_array.ravel())
        closest_indices = np.argmin(abs_diff, axis=0)
        quantized = np.array(self.fixed_values)[closest_indices]
        return quantized.reshape(input_array.shape)

# Example input array
# input_array = np.random.rand(1, 100) - 0.5

# # Launch visualization (only red ticks, lines from each point to quantized value)
# fpv = FixedPointVisualizer(draw=True, init_int_bits=3, init_frac_bits=4, input_array=input_array)
