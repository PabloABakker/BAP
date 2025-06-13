from pysindy import CustomLibrary
import itertools
from sympy import sympify, expand
import numpy as np

def create_hierarchical_dsp_library(depth: int = 1, var_names: tuple = ("x",)):
    """
    Generates a hierarchical, hardware-efficient library. This version correctly
    constructs the library and names for PySINDy.
    """
    if not isinstance(var_names, tuple) or not var_names:
        raise ValueError("var_names must be a non-empty tuple of strings.")
    if depth < 0:
        raise ValueError("Depth must be a non-negative integer.")

    # Library Generation 
    base_strings = list(var_names) + ['1']
    library_functions = {s: sympify(s) for s in base_strings}
    current_level_inputs = base_strings
    
    for i in range(depth):
        print(f"Generating library functions for depth {i + 1}...")
        A_sources, B_sources, D_sources = current_level_inputs, current_level_inputs, current_level_inputs
        C_sources = list(var_names)
        new_functions = {}
        for A_str, B_str, C_str, D_str in itertools.product(A_sources, B_sources, C_sources, D_sources):
            A, B, C, D = library_functions[A_str], library_functions[B_str], library_functions[C_str], library_functions[D_str]
            expr = expand((A + B) * C + D)
            expr_str = str(expr)
            if expr_str not in library_functions:
                new_functions[expr_str] = expr
        library_functions.update(new_functions)
        current_level_inputs = list(library_functions.keys())
        print(f"Library now has {len(library_functions)} unique functions.")

    final_strings = list(library_functions.keys())
    sym_vars = [sympify(v) for v in var_names]
    eval_globals = {v.name: v for v in sym_vars}
    eval_globals['np'] = np

    # Create the list of functions that compute the feature values
    final_lambdas = [eval(f"lambda {','.join(var_names)}: {s}", eval_globals) for s in final_strings]

    # This lambda now accepts any number of arguments via *args, ignores them,
    # and returns the pre-computed string 's'. This satisfies the API.
    final_name_generators = [lambda *args, s=s: s for s in final_strings]

    print(f"\nFinal generated library contains {len(final_strings)} functions.")
    
    return CustomLibrary(
        library_functions=final_lambdas,
        function_names=final_name_generators
    )

# def hardware_efficient_library(var_names=("x", "y", "z")):
#     functions = [
#         # Constant
#         lambda x, y, z: 1.0,

#         # Linear
#         lambda x, y, z: x,
#         lambda x, y, z: y,
#         lambda x, y, z: z,

#         # Quadratic
#         lambda x, y, z: x**2,
#         lambda x, y, z: y**2,
#         lambda x, y, z: z**2,

#         # Interaction terms
#         lambda x, y, z: x*y,
#         lambda x, y, z: x*z,
#         lambda x, y, z: y*z,

#         # A + B*C combinations (non-redundant)
#         lambda x, y, z: x + y*z,
#         lambda x, y, z: y + x*z,
#         lambda x, y, z: z + x*y,

#         # (A + B)*C combinations (non-redundant)
#         lambda x, y, z: (x + y)*z,
#         lambda x, y, z: (x + z)*y,
#         lambda x, y, z: (y + z)*x,

#         # (A + B)*C + D combinations (non-redundant)
#         lambda x, y, z: (x + y)*z + x,
#         lambda x, y, z: (x + z)*y + z,
#         lambda x, y, z: (y + z)*x + z,

#         # ((A + B)^2) + C combinations (non-redundant)
#         lambda x, y, z: ((y + z)**2) + x,
#         lambda x, y, z: ((z + x)**2) + y,
#         lambda x, y, z: ((x + y)**2) + z,

#         # New composite squared terms
#         lambda x, y, z: (x + y*z)**2,
#         lambda x, y, z: (y + x*z)**2,
#         lambda x, y, z: (z + x*y)**2,

#         lambda x, y, z: ((x + y)*z)**2,
#         lambda x, y, z: ((x + z)*y)**2,
#         lambda x, y, z: ((y + z)*x)**2,

#         lambda x, y, z: ((x + y)*z + x)**2,
#         lambda x, y, z: ((x + z)*y + z)**2,
#         lambda x, y, z: ((y + z)*x + z)**2,

#         # Composite squared + variable
#         lambda x, y, z: (x + y*z)**2 + x,
#         lambda x, y, z: (y + x*z)**2 + y,
#         lambda x, y, z: (z + x*y)**2 + z,

#         lambda x, y, z: ((x + y)*z)**2 + x,
#         lambda x, y, z: ((x + z)*y)**2 + y,
#         lambda x, y, z: ((y + z)*x)**2 + z,

#         lambda x, y, z: ((x + y)*z + x)**2 + x,
#         lambda x, y, z: ((x + z)*y + z)**2 + y,
#         lambda x, y, z: ((y + z)*x + z)**2 + z
#     ]

#     function_names = [
#         lambda x, y, z: "1",
#         lambda x, y, z: "x",
#         lambda x, y, z: "y",
#         lambda x, y, z: "z",
#         lambda x, y, z: "x^2",
#         lambda x, y, z: "y^2",
#         lambda x, y, z: "z^2",
#         lambda x, y, z: "x*y",
#         lambda x, y, z: "x*z",
#         lambda x, y, z: "y*z",
#         lambda x, y, z: "x + y*z",
#         lambda x, y, z: "y + x*z",
#         lambda x, y, z: "z + x*y",
#         lambda x, y, z: "(x + y)*z",
#         lambda x, y, z: "(x + z)*y",
#         lambda x, y, z: "(y + z)*x",
#         lambda x, y, z: "(x + y)*z + x",
#         lambda x, y, z: "(x + z)*y + z",
#         lambda x, y, z: "(y + z)*x + z",
#         lambda x, y, z: "((y + z)^2) + x",
#         lambda x, y, z: "((z + x)^2) + y",
#         lambda x, y, z: "((x + y)^2) + z",
#         lambda x, y, z: "(x + y*z)^2",
#         lambda x, y, z: "(y + x*z)^2",
#         lambda x, y, z: "(z + x*y)^2",
#         lambda x, y, z: "((x + y)*z)^2",
#         lambda x, y, z: "((x + z)*y)^2",
#         lambda x, y, z: "((y + z)*x)^2",
#         lambda x, y, z: "((x + y)*z + x)^2",
#         lambda x, y, z: "((x + z)*y + z)^2",
#         lambda x, y, z: "((y + z)*x + z)^2",
#         lambda x, y, z: "(x + y*z)^2 + x",
#         lambda x, y, z: "(y + x*z)^2 + y",
#         lambda x, y, z: "(z + x*y)^2 + z",
#         lambda x, y, z: "((x + y)*z)^2 + x",
#         lambda x, y, z: "((x + z)*y)^2 + y",
#         lambda x, y, z: "((y + z)*x)^2 + z",
#         lambda x, y, z: "((x + y)*z + x)^2 + x",
#         lambda x, y, z: "((x + z)*y + z)^2 + y",
#         lambda x, y, z: "((y + z)*x + z)^2 + z"
#     ]

#     return CustomLibrary(
#         library_functions=functions,
#         function_names=function_names
#     )



def hardware_efficient_library_pendulum(var_names=("theta", "theta_dot")):
    functions = [
        # Order 0
        lambda x, y: 1.0,

        # Order 1
        lambda x, y: x,
        lambda x, y: y,

        # Order 2
        lambda x, y: x**2,
        lambda x, y: y**2,
        lambda x, y: x * y,

        # Order 3
        lambda x, y: x**3,
        lambda x, y: y**3,
        lambda x, y: x**2 * y,
        lambda x, y: x * y**2,

        # Order 4
        lambda x, y: x**4,
        lambda x, y: y**4,
        lambda x, y: x**3 * y,
        lambda x, y: x * y**3,
        lambda x, y: x**2 * y**2,

        # Order 5
        lambda x, y: x**5,
        lambda x, y: y**5,
        lambda x, y: x**4 * y,
        lambda x, y: x * y**4,
        lambda x, y: x**3 * y**2,
        lambda x, y: x**2 * y**3,

        # Order 6
        lambda x, y: x**6,
        lambda x, y: y**6,
        lambda x, y: x**5 * y,
        lambda x, y: x * y**5,
        lambda x, y: x**4 * y**2,
        lambda x, y: x**2 * y**4,
        lambda x, y: x**3 * y**3,

        # Order 7
        lambda x, y: x**7,
        lambda x, y: y**7,
        lambda x, y: x**6 * y,
        lambda x, y: x * y**6,
        lambda x, y: x**5 * y**2,
        lambda x, y: x**2 * y**5,
        lambda x, y: x**4 * y**3,
        lambda x, y: x**3 * y**4,

        # Order 8
        lambda x, y: x**8,
        lambda x, y: y**8,
        lambda x, y: x**7 * y,
        lambda x, y: x * y**7,
        lambda x, y: x**6 * y**2,
        lambda x, y: x**2 * y**6,
        lambda x, y: x**5 * y**3,
        lambda x, y: x**3 * y**5,
        lambda x, y: x**4 * y**4,

        # Composite expressions (up to order 8)
        lambda x, y: (x + y),
        lambda x, y: (x + y)**2,
        lambda x, y: (x + y)**3,
        lambda x, y: (x + y)**4,
        lambda x, y: (x + y)**5,
        lambda x, y: (x + y)**6,
        lambda x, y: (x + y)**7,
        lambda x, y: (x + y)**8,

        lambda x, y: (x + y) * x,
        lambda x, y: (x + y) * y,

        lambda x, y: ((x + y) * x)**2,
        lambda x, y: ((x + y) * y)**2,
        lambda x, y: ((x + y) * x)**3,
        lambda x, y: ((x + y) * y)**3,
        lambda x, y: ((x + y) * x)**4,
        lambda x, y: ((x + y) * y)**4,
        lambda x, y: ((x + y) * x)**5,
        lambda x, y: ((x + y) * y)**5,
        lambda x, y: ((x + y) * x)**6,
        lambda x, y: ((x + y) * y)**6,
        lambda x, y: ((x + y) * x)**7,
        lambda x, y: ((x + y) * y)**7,
        lambda x, y: ((x + y) * x)**8,
        lambda x, y: ((x + y) * y)**8,

        lambda x, y: (x + y)**2 + x,
        lambda x, y: (x + y)**2 + y,
        lambda x, y: (x + y)**3 + x,
        lambda x, y: (x + y)**3 + y,
        lambda x, y: (x + y)**4 + x,
        lambda x, y: (x + y)**4 + y,
        lambda x, y: (x + y)**5 + x,
        lambda x, y: (x + y)**5 + y,
        lambda x, y: (x + y)**6 + x,
        lambda x, y: (x + y)**6 + y,
        lambda x, y: (x + y)**7 + x,
        lambda x, y: (x + y)**7 + y,
        lambda x, y: (x + y)**8 + x,
        lambda x, y: (x + y)**8 + y,
    ]

    function_names = [  # You can adjust to use var_names[0] and var_names[1] if needed
        lambda x, y: "1",
        lambda x, y: "x",
        lambda x, y: "y",
        lambda x, y: "x^2",
        lambda x, y: "y^2",
        lambda x, y: "x*y",
        lambda x, y: "x^3",
        lambda x, y: "y^3",
        lambda x, y: "x^2*y",
        lambda x, y: "x*y^2",
        lambda x, y: "x^4",
        lambda x, y: "y^4",
        lambda x, y: "x^3*y",
        lambda x, y: "x*y^3",
        lambda x, y: "x^2*y^2",
        lambda x, y: "x^5",
        lambda x, y: "y^5",
        lambda x, y: "x^4*y",
        lambda x, y: "x*y^4",
        lambda x, y: "x^3*y^2",
        lambda x, y: "x^2*y^3",
        lambda x, y: "x^6",
        lambda x, y: "y^6",
        lambda x, y: "x^5*y",
        lambda x, y: "x*y^5",
        lambda x, y: "x^4*y^2",
        lambda x, y: "x^2*y^4",
        lambda x, y: "x^3*y^3",
        lambda x, y: "x^7",
        lambda x, y: "y^7",
        lambda x, y: "x^6*y",
        lambda x, y: "x*y^6",
        lambda x, y: "x^5*y^2",
        lambda x, y: "x^2*y^5",
        lambda x, y: "x^4*y^3",
        lambda x, y: "x^3*y^4",
        lambda x, y: "x^8",
        lambda x, y: "y^8",
        lambda x, y: "x^7*y",
        lambda x, y: "x*y^7",
        lambda x, y: "x^6*y^2",
        lambda x, y: "x^2*y^6",
        lambda x, y: "x^5*y^3",
        lambda x, y: "x^3*y^5",
        lambda x, y: "x^4*y^4",
        lambda x, y: "(x + y)",
        lambda x, y: "(x + y)^2",
        lambda x, y: "(x + y)^3",
        lambda x, y: "(x + y)^4",
        lambda x, y: "(x + y)^5",
        lambda x, y: "(x + y)^6",
        lambda x, y: "(x + y)^7",
        lambda x, y: "(x + y)^8",
        lambda x, y: "(x + y)*x",
        lambda x, y: "(x + y)*y",
        lambda x, y: "((x + y)*x)^2",
        lambda x, y: "((x + y)*y)^2",
        lambda x, y: "((x + y)*x)^3",
        lambda x, y: "((x + y)*y)^3",
        lambda x, y: "((x + y)*x)^4",
        lambda x, y: "((x + y)*y)^4",
        lambda x, y: "((x + y)*x)^5",
        lambda x, y: "((x + y)*y)^5",
        lambda x, y: "((x + y)*x)^6",
        lambda x, y: "((x + y)*y)^6",
        lambda x, y: "((x + y)*x)^7",
        lambda x, y: "((x + y)*y)^7",
        lambda x, y: "((x + y)*x)^8",
        lambda x, y: "((x + y)*y)^8",
        lambda x, y: "(x + y)^2 + x",
        lambda x, y: "(x + y)^2 + y",
        lambda x, y: "(x + y)^3 + x",
        lambda x, y: "(x + y)^3 + y",
        lambda x, y: "(x + y)^4 + x",
        lambda x, y: "(x + y)^4 + y",
        lambda x, y: "(x + y)^5 + x",
        lambda x, y: "(x + y)^5 + y",
        lambda x, y: "(x + y)^6 + x",
        lambda x, y: "(x + y)^6 + y",
        lambda x, y: "(x + y)^7 + x",
        lambda x, y: "(x + y)^7 + y",
        lambda x, y: "(x + y)^8 + x",
        lambda x, y: "(x + y)^8 + y",
    ]

    return CustomLibrary(
        library_functions=functions,
        function_names=function_names
    )




def hardware_efficient_library_cartpole_up(var_names=("x", "x_dot", "theta", "theta_dot")):
    functions = [
        # Order 0
        lambda x, dx, th, dth: 1.0,

        # Order 1
        lambda x, dx, th, dth: x,
        lambda x, dx, th, dth: dx,
        lambda x, dx, th, dth: th,
        lambda x, dx, th, dth: dth,

        # Order 2
        lambda x, dx, th, dth: x**2,
        lambda x, dx, th, dth: dx**2,
        lambda x, dx, th, dth: th**2,
        lambda x, dx, th, dth: dth**2,
        lambda x, dx, th, dth: x * th,
        lambda x, dx, th, dth: dx * dth,
        lambda x, dx, th, dth: x * dx,
        lambda x, dx, th, dth: th * dth,

        # Order 3
        lambda x, dx, th, dth: x**3,
        lambda x, dx, th, dth: th**3,
        lambda x, dx, th, dth: x**2 * th,
        lambda x, dx, th, dth: x * th**2,
        lambda x, dx, th, dth: dx**2 * dth,
        lambda x, dx, th, dth: dx * dth**2,

        # Order 4
        lambda x, dx, th, dth: x**4,
        lambda x, dx, th, dth: th**4,
        lambda x, dx, th, dth: x**2 * th**2,
        lambda x, dx, th, dth: dx**2 * dth**2,
        lambda x, dx, th, dth: (x + th)**2,
        lambda x, dx, th, dth: (dx + dth)**2,

        # Order 5
        lambda x, dx, th, dth: x**5,
        lambda x, dx, th, dth: th**5,
        lambda x, dx, th, dth: (x + th)**3,
        lambda x, dx, th, dth: (dx + dth)**3,
        lambda x, dx, th, dth: (x + th) * x * th,
        lambda x, dx, th, dth: (dx + dth) * dx * dth,

        # Order 6
        lambda x, dx, th, dth: x**6,
        lambda x, dx, th, dth: th**6,
        lambda x, dx, th, dth: x**3 * th**3,
        lambda x, dx, th, dth: dx**3 * dth**3,
        lambda x, dx, th, dth: (x + th)**4,
        lambda x, dx, th, dth: (dx + dth)**4,
    ]

    function_names = [
        lambda x, dx, th, dth: "1",
        lambda x, dx, th, dth: "x",
        lambda x, dx, th, dth: "x_dot",
        lambda x, dx, th, dth: "theta",
        lambda x, dx, th, dth: "theta_dot",

        lambda x, dx, th, dth: "x^2",
        lambda x, dx, th, dth: "x_dot^2",
        lambda x, dx, th, dth: "theta^2",
        lambda x, dx, th, dth: "theta_dot^2",
        lambda x, dx, th, dth: "x*theta",
        lambda x, dx, th, dth: "x_dot*theta_dot",
        lambda x, dx, th, dth: "x*x_dot",
        lambda x, dx, th, dth: "theta*theta_dot",

        lambda x, dx, th, dth: "x^3",
        lambda x, dx, th, dth: "theta^3",
        lambda x, dx, th, dth: "x^2*theta",
        lambda x, dx, th, dth: "x*theta^2",
        lambda x, dx, th, dth: "x_dot^2*theta_dot",
        lambda x, dx, th, dth: "x_dot*theta_dot^2",

        lambda x, dx, th, dth: "x^4",
        lambda x, dx, th, dth: "theta^4",
        lambda x, dx, th, dth: "x^2*theta^2",
        lambda x, dx, th, dth: "x_dot^2*theta_dot^2",
        lambda x, dx, th, dth: "(x + theta)^2",
        lambda x, dx, th, dth: "(x_dot + theta_dot)^2",

        lambda x, dx, th, dth: "x^5",
        lambda x, dx, th, dth: "theta^5",
        lambda x, dx, th, dth: "(x + theta)^3",
        lambda x, dx, th, dth: "(x_dot + theta_dot)^3",
        lambda x, dx, th, dth: "(x + theta)*x*theta",
        lambda x, dx, th, dth: "(x_dot + theta_dot)*x_dot*theta_dot",

        lambda x, dx, th, dth: "x^6",
        lambda x, dx, th, dth: "theta^6",
        lambda x, dx, th, dth: "x^3*theta^3",
        lambda x, dx, th, dth: "x_dot^3*theta_dot^3",
        lambda x, dx, th, dth: "(x + theta)^4",
        lambda x, dx, th, dth: "(x_dot + theta_dot)^4",
    ]

    return CustomLibrary(
        library_functions=functions,
        function_names=function_names
    )
