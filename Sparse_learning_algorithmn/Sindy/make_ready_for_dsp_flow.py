import sys
import re
import numpy as np
from pathlib import Path
import dill as pickle


script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.append(str(project_root))
from Custom.quantization import FixedPointVisualizer




def load_sindy_model(env_id, variant):
    base = project_root / "Results" / env_id / "SINDY"
    model_path = base / variant / f"sindy_policy_{variant}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"SINDy model not found at: {model_path}")

    print(f"Loaded SINDy model: {variant}")
    with open(model_path, "rb") as f:
        return pickle.load(f)
    
    



def equations_from_Xi_Theta(Xi, Theta_feature_names, var_names=None):
    """
    Reconstructs symbolic strings with guaranteed variable substitution.
    Adds explicit multiplication signs between all terms.
    """
    num_eqs = Xi.shape[1]

    # Handle var_names input
    if isinstance(var_names, str):
        var_names = list(var_names)
    elif var_names is None:
        var_names = [f"x{i}" for i in range(num_eqs)]

    # Build variable pattern mapping
    detected_vars = sorted(set(v for name in Theta_feature_names for v in re.findall(r"\bx\d*\b", name)))
    var_mapping = {}

    if all(v in ["x", "x1", "x2", "x3", "x4", "x5"] for v in detected_vars):
        for i, name in enumerate(var_names):
            key = "x" if i == 0 else f"x{i}"
            var_mapping[key] = name
    else:
        for i, name in enumerate(var_names):
            var_mapping[f"x{i}"] = name

    regex_patterns = [
        (re.compile(rf'\b{re.escape(old)}\b'), new)
        for old, new in sorted(var_mapping.items(), key=lambda x: (-len(x[0]), x[0]))
    ]

    equations = []
    for eq_idx in range(num_eqs):
        terms = []
        for i, coeff in enumerate(Xi[:, eq_idx]):
            if np.abs(coeff) > 1e-10:
                raw_term = Theta_feature_names[i]

                # Apply substitutions
                processed_term = raw_term
                for pattern, new in regex_patterns:
                    processed_term = pattern.sub(new, processed_term)

                # Add multiplication signs (including for constants as 0.3 * 1)
                if processed_term == "1":
                    term = f"- ({abs(coeff):.5f} * 1)" if coeff < 0 else f"({coeff:.5f} * 1)"
                else:
                    processed_term = processed_term.replace(" ", " * ")
                    term = f"- ({abs(coeff):.5f} * {processed_term})" if coeff < 0 else f"({coeff:.5f} * {processed_term})"

                terms.append(term)

        # Combine terms
        if terms:
            equation_str = terms[0]
            for term in terms[1:]:
                if term.startswith("-"):
                    equation_str += f" {term}"
                else:
                    equation_str += f" + {term}"
            equation_str = equation_str.replace("* + *", " + ").replace("* - *", " - ")
            equations.append(equation_str)
        else:
            equations.append("0")

    return equations





def make_ready_for_dsp_flow(env_id, variant, var_names="xyzuvw"):

    model = load_sindy_model(env_id, variant)
    print(len(np.where(model.coefficients()!=0)[0]))
    model.coefficients()[:] = FPV.quantize(np.array(model.coefficients()))

    equations = equations_from_Xi_Theta(model.coefficients().T, model.get_feature_names(), var_names)
    return equations



def save_all_equations(pre_quant_dict, quant_dict, env_id):
    output_path = project_root / "Results" / env_id / "SINDY" / "equations"
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{env_id}_ALL_VARIANTS_mse_qat_optimized.txt"
    full_path = output_path / filename

    with open(full_path, "w", encoding="utf-8") as f:
        for variant in quant_dict:
            f.write(f"Variant: {variant}\n")

            # If pre-quantized version exists, write it first
            if variant in pre_quant_dict:
                f.write("  ── Pre-Quantized Equations ──\n")
                for i, eq in enumerate(pre_quant_dict[variant]):
                    f.write(f"    Eq {i}: {eq}\n")

            f.write("  ── Quantized Equations ──\n")
            for i, eq in enumerate(quant_dict[variant]):
                f.write(f"    Eq {i}: {eq}\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Saved all equations to {full_path}")




if __name__ == "__main__":

    SINDY_VARIANTS = [
        "poly_regular_STLSQ",
        "poly_regular_ConstrainedSR3",
        "poly_hardware_STLSQ",
        "poly_hardware_ConstrainedSR3",
        "hw_regular_STLSQ",
        "hw_regular_ConstrainedSR3",
        "hw_hardware_STLSQ",
        "hw_hardware_ConstrainedSR3",
    ]

    env_id = "Pendulum-v1"
    var_names = ["x", "y", "z"]
    FPV = FixedPointVisualizer(int_bits=4, frac_bits=6)

    pre_quant_equations = {}
    quantized_equations = {}

    for variant in SINDY_VARIANTS:
        model = load_sindy_model(env_id, variant)

        # Save pre-quantized equations for "regular" optimizer variants
        if "_regular_" in variant:
            eqs_pre = equations_from_Xi_Theta(model.coefficients().T, model.get_feature_names(), var_names)
            pre_quant_equations[variant] = eqs_pre

        # Quantize coefficients
        model.coefficients()[:] = FPV.quantize(np.array(model.coefficients()))
        eqs_quant = equations_from_Xi_Theta(model.coefficients().T, model.get_feature_names(), var_names)
        quantized_equations[variant] = eqs_quant

    save_all_equations(pre_quant_equations, quantized_equations, env_id)


    