def hardware_efficient_library(var_names=("x", "y", "z")):
    functions = [
        # Constant
        lambda x, y, z: 1.0,

        # Linear
        lambda x, y, z: x,
        lambda x, y, z: y,
        lambda x, y, z: z,

        # Quadratic
        lambda x, y, z: x**2,
        lambda x, y, z: y**2,
        lambda x, y, z: z**2,

        # Interaction terms
        lambda x, y, z: x*y,
        lambda x, y, z: x*z,
        lambda x, y, z: y*z,

        # A + B*C combinations (non-redundant)
        lambda x, y, z: x + y*z,
        lambda x, y, z: y + x*z,
        lambda x, y, z: z + x*y,

        # (A + B)*C combinations (non-redundant)
        lambda x, y, z: (x + y)*z,
        lambda x, y, z: (x + z)*y,
        lambda x, y, z: (y + z)*x,

        # (A + B)*C + D combinations (non-redundant)
        lambda x, y, z: (x + y)*z + x,
        lambda x, y, z: (x + z)*y + z,
        lambda x, y, z: (y + z)*x + z,

        # ((A + B)^2) + C combinations (non-redundant)
        lambda x, y, z: ((y + z)**2) + x,
        lambda x, y, z: ((z + x)**2) + y,
        lambda x, y, z: ((x + y)**2) + z,

        # New composite squared terms
        lambda x, y, z: (x + y*z)**2,
        lambda x, y, z: (y + x*z)**2,
        lambda x, y, z: (z + x*y)**2,

        lambda x, y, z: ((x + y)*z)**2,
        lambda x, y, z: ((x + z)*y)**2,
        lambda x, y, z: ((y + z)*x)**2,

        lambda x, y, z: ((x + y)*z + x)**2,
        lambda x, y, z: ((x + z)*y + z)**2,
        lambda x, y, z: ((y + z)*x + z)**2,

        # Composite squared + variable
        lambda x, y, z: (x + y*z)**2 + x,
        lambda x, y, z: (y + x*z)**2 + y,
        lambda x, y, z: (z + x*y)**2 + z,

        lambda x, y, z: ((x + y)*z)**2 + x,
        lambda x, y, z: ((x + z)*y)**2 + y,
        lambda x, y, z: ((y + z)*x)**2 + z,

        lambda x, y, z: ((x + y)*z + x)**2 + x,
        lambda x, y, z: ((x + z)*y + z)**2 + y,
        lambda x, y, z: ((y + z)*x + z)**2 + z
    ]

    function_names = [
        lambda x, y, z: "1",
        lambda x, y, z: "x",
        lambda x, y, z: "y",
        lambda x, y, z: "z",
        lambda x, y, z: "x**2",
        lambda x, y, z: "y**2",
        lambda x, y, z: "z**2",
        lambda x, y, z: "x*y",
        lambda x, y, z: "x*z",
        lambda x, y, z: "y*z",
        lambda x, y, z: "x + y*z",
        lambda x, y, z: "y + x*z",
        lambda x, y, z: "z + x*y",
        lambda x, y, z: "(x + y)*z",
        lambda x, y, z: "(x + z)*y",
        lambda x, y, z: "(y + z)*x",
        lambda x, y, z: "(x + y)*z + x",
        lambda x, y, z: "(x + z)*y + z",
        lambda x, y, z: "(y + z)*x + z",
        lambda x, y, z: "((y + z)**2) + x",
        lambda x, y, z: "((z + x)**2) + y",
        lambda x, y, z: "((x + y)**2) + z",
        lambda x, y, z: "(x + y*z)**2",
        lambda x, y, z: "(y + x*z)**2",
        lambda x, y, z: "(z + x*y)**2",
        lambda x, y, z: "((x + y)*z)**2",
        lambda x, y, z: "((x + z)*y)**2",
        lambda x, y, z: "((y + z)*x)**2",
        lambda x, y, z: "((x + y)*z + x)**2",
        lambda x, y, z: "((x + z)*y + z)**2",
        lambda x, y, z: "((y + z)*x + z)**2",
        lambda x, y, z: "(x + y*z)**2 + x",
        lambda x, y, z: "(y + x*z)**2 + y",
        lambda x, y, z: "(z + x*y)**2 + z",
        lambda x, y, z: "((x + y)*z)**2 + x",
        lambda x, y, z: "((x + z)*y)**2 + y",
        lambda x, y, z: "((y + z)*x)**2 + z",
        lambda x, y, z: "((x + y)*z + x)**2 + x",
        lambda x, y, z: "((x + z)*y + z)**2 + y",
        lambda x, y, z: "((y + z)*x + z)**2 + z"
    ]

    return CustomLibrary(
        library_functions=functions,
        function_names=function_names
    )