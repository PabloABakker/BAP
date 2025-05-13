import numpy as np
import pysindy as ps
from pysindy.feature_library import CustomLibrary
from pysindy.optimizers import STLSQ, SR3
from scipy.integrate import solve_ivp
from itertools import combinations_with_replacement, permutations
from sklearn.linear_model import Lasso


# 1. Lorenz system
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

t_span = (0, 10)
dt = 0.01
t_eval = np.arange(t_span[0], t_span[1], dt)
initial_state = [1.0, 1.0, 1.0]
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
x = sol.y.T  # (n_samples, 3)

# # 2. Custom feature library with x, y, z, x*y, y*z, and constant
# functions = [
#     lambda x, y, z: 1.0,
#     lambda x, y, z: x,
#     lambda x, y, z: y,
#     lambda x, y, z: z,
#     lambda x, y, z: x * y,
#     lambda x, y, z: x * z,
#     lambda x, y, z: y * z,
#     lambda x, y, z: x + y * z

# ]

# function_names = [
#     lambda x, y, z: "1",
#     lambda x, y, z: "x",
#     lambda x, y, z: "y",
#     lambda x, y, z: "z",
#     lambda x, y, z: "x*y",
#     lambda x, y, z: "x*z",
#     lambda x, y, z: "y*z",
#     lambda x, y, z: "x+y*z"
# ]

vars = ['x', 'y', 'z']

def make_function(expr_str, var_names=('x', 'y', 'z')):
    code = compile(expr_str, "<string>", "eval")
    return lambda x, y, z: eval(code, {}, dict(zip(var_names, (x, y, z))))

def generate_expressions():
    exprs = set()

    # Degree 1 and 2 polynomials
    for var in vars:
        exprs.add(var)
        exprs.add(f"{var}**2")

    for a, b in combinations_with_replacement(vars, 2):
        exprs.add(f"{a}*{b}")
        exprs.add(f"{a}+{b}")
        exprs.add(f"{a}-{b}")

    # (A+B)*C, (A-B)*C
    for a, b, c in permutations(vars, 3):
        exprs.add(f"({a}+{b})*{c}")
        exprs.add(f"({a}-{b})*{c}")

    # (A+B)*C+D and variations
    for a, b, c, d in permutations(vars, 4):
        exprs.add(f"({a}+{b})*{c}+{d}")
        exprs.add(f"({a}-{b})*{c}-{d}")

    # ((C+D)**2)+A and variations
    for a, b, c in permutations(vars, 3):
        exprs.add(f"(({b}+{c})**2)+{a}")
        exprs.add(f"(({b}-{c})**2)-{a}")

    # Add constant
    exprs.add("1")

    return sorted(exprs)


expr_list = generate_expressions()

functions = [make_function(expr) for expr in expr_list]
function_names = [lambda x, y, z, s=expr: s for expr in expr_list]

library = CustomLibrary(
    library_functions=functions,
    function_names=function_names
)

# 3. Fit SINDy model
optimizer = STLSQ(threshold=0.1, alpha=100)
# optimizer = SR3(threshold=0.1, nu=1.0, tol=1e-5)
# optimizer= Lasso(alpha=100, fit_intercept=True, max_iter=500)
model = ps.SINDy(feature_library=library, optimizer=optimizer)
model.fit(x, t=dt)
model.print()

# simulate
x_sindy = model.simulate(x[0], t_eval)


# plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

labels = ['x', 'y', 'z']
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(t_eval, x[:, i], 'k', label='Lorenz')
    plt.plot(t_eval, x_sindy[:, i], 'r--', label='SINDy')
    plt.xlabel('Time')
    plt.ylabel(labels[i])
    plt.legend()
    plt.title(f'{labels[i]} vs Time')

plt.tight_layout()
plt.show()

# mse 
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(x, x_sindy)
print(f'MSE (SINDy vs Lorenz): {mse:.6f}')


