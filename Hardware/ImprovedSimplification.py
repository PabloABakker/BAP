from DataFlowModel import DSPBlockNumber
import pandas as pd
import numpy as np
from collections import defaultdict
from PolynomialClass import *
import pandas as pd
from IPython.display import display
import numpy as np
from itertools import combinations


def extract_terms(node, terms=None):
    if terms is None:
        terms = []
    
    if type(node)== Addition:
        extract_terms(node.left, terms)
        extract_terms(node.right, terms)
        
    elif type(node) == Subtraction:
        extract_terms(node.left, terms)
        extract_terms(UnaryMinus(node.right), terms)

    elif type(node)== UnaryMinus:
        terms.append(node)

    else:
        terms.append(node)
    
    return terms



def try_factorizing(term, co_kernel):

    original_term = term 
    original_cokernel = co_kernel 

    if equal_terms(term, co_kernel):
        #print("Equal terms found:", term, co_kernel)

        return Constant(1.0)
    
    if co_kernel == Constant(1):
        return term

    if type(term) == UnaryMinus and type(co_kernel)!= UnaryMinus:
        inner_result = try_factorizing(term.value, co_kernel)

        if inner_result is None:  
            return None
        return UnaryMinus(inner_result)
        
    if type(co_kernel)== UnaryMinus and type(term)!= UnaryMinus:
        inner_result = try_factorizing(term, co_kernel.value)

        if inner_result is None:  
            return None
        return UnaryMinus(inner_result)
    
        
    if type(term)== UnaryMinus and type(co_kernel)== UnaryMinus:
        
        return try_factorizing(term.value, co_kernel.value)

    # Power factorization by subtracting the powers
    if type(term)== Power and type(co_kernel)== Power:
       
        if equal_terms(term.left, co_kernel.left) and term.right.value > co_kernel.right.value:
            
            new_power = term.right.value - co_kernel.right.value
            #print("Power difference:", new_power)

            if new_power == 1:
                return term.left 
            else:
                return Power(term.left, Constant(new_power))

    # Power and symbol factorization
    if type(term)== Power and type(co_kernel) in (Variable, Function):
        if equal_terms(term.left, co_kernel) and term.right.value > 1:
            new_power = term.right.value - 1
            # print(new_power)

            if new_power == 1:
                return term.left 
            else:
                return Power(term.left, Constant(new_power))

    if type(term)== Multiplication:
        term_factors = flatten_multiplication_factors(term)
    
        if type(co_kernel)== Multiplication:
            return factorize_with_multiple_factors(term_factors, flatten_multiplication_factors(co_kernel))
        else:
            remaining_factors = term_factors.copy() 
            remaining_factors, in_list = remove_factor_from_list(remaining_factors, co_kernel)

            if not in_list:
                return None
            
            return combine_factors(remaining_factors)
    
    return None


    


def factorize_with_multiple_factors(term_factors, cokernel_factors):
    # Make a copy so not to interfere, seems to be important
    remaining_factors = term_factors.copy()  
    
    for factor in cokernel_factors:
        remaining_factors, in_list = remove_factor_from_list(remaining_factors, factor)

        if not in_list:
            return None

    return combine_factors(remaining_factors)


    

def remove_factor_from_list(factors, term):
    for i, factor in enumerate(factors):

        # If there is an exact match then remove term from factors
        if equal_terms(factor, term):
            factors.pop(i)
            return factors, True
        
        # Powers need to have the same base and then the exponents are subtracted  
        if (type(factor) == Power and type(term) == Power and
            equal_terms(factor.left, term.left) and factor.right.value > term.right.value):
            
            new_power = factor.right.value - term.right.value
            if new_power == 1:
                new_factor = factor.left
            else:
                new_factor = Power(factor.left, Constant(new_power))

            # print(new_factor)

            # Remove the factored term and add the new one
            factors.pop(i)
            factors.insert(i, new_factor)
            return factors, True
        
        # Processing factorization of power with variable or function
        if (type(factor)== Power and type(term) in (Variable, Function) and 
            equal_terms(factor.left, term)):
            
            new_power = factor.right.value - 1
            if new_power == 1:
                new_factor = factor.left
            else:
                new_factor = Power(factor.left, Constant(new_power))
            
            factors.pop(i)
            factors.insert(i, new_factor)

            return factors, True
    
    return factors, False



def sort_factors(factor):
        
        if type(factor)== Constant:
            return (0, factor.value, 0)
        
        elif type(factor)== Variable:
            return (1, factor.value, 0)
        
        elif type(factor)== Function:
            return (1, factor.name, 0) 
        

        elif type(factor)== Power:
            if type(factor.left)== Variable and type(factor.right)== Constant:
                return (2, factor.left.value, factor.right.value)
            

            elif type(factor.left)== Function and type(factor.right)== Constant:
                return (2, factor.left.name, factor.right.value)
            
           
        else:
            return (3, str(factor), 0)


def equal_terms(term1, term2):
    if type(term1)!= type(term2):
        return False
    
    if type(term1)== Variable:
        return term1.value == term2.value
    
    elif type(term1)== Function:
        return term1.name == term2.name
    
    elif type(term1)== Constant:
        return (abs(term1.value - term2.value) < 1e-10)
    
    elif type(term1)== Multiplication:
        term1_type = type(term1)
        term2_type = type(term2)

        # Need to unroll the nested multiplications to be able to compare terms
        factors1 = flatten_multiplication_factors(term1)
        factors2 = flatten_multiplication_factors(term2)
        
        if len(factors1)!= len(factors2):
            return False
        
        # Need to give terms a consistent order to be able to compare
        normalize_f1 = sorted(factors1, key=sort_factors)
        normalize_f2 = sorted(factors2, key=sort_factors)

        equal_functions = all(equal_terms(f1, f2) for f1, f2 in zip(normalize_f1, normalize_f2))
        
        return equal_functions
    
    elif type(term1) in (Addition, Subtraction, Power):

        left_side_equals = equal_terms(term1.left, term2.left)
        right_side_equals = equal_terms(term1.right, term2.right)

        return (left_side_equals and right_side_equals)
    
    elif type(term1)== UnaryMinus:
        return equal_terms(term1.value, term2.value)
    
    return False



def combine_factors(factors):
    num_factors = len(factors)

    if len(factors) == 0:
        return Constant(1.0)
    elif len(factors) == 1:
        return factors[0]
    else:
        result = factors[0]
        for factor in factors[1:]:
            result = Multiplication(result, factor)
        return result
    


def flatten_multiplication_factors(expr):

    if type(expr)!= Multiplication:
        return [expr]
    
    factors = []
    factors.extend(flatten_multiplication_factors(expr.left))
    factors.extend(flatten_multiplication_factors(expr.right))

    # print(factors)
    return factors





def find_symbols(expression, node=None, symbols=None):
    if symbols is None:
        symbols = set()

    if node is None:
        node = expression

    # print(expression)
    if type(node) in (Variable, Function):
        symbols.add(node)

    elif type(node) in (Addition, Subtraction, Multiplication, Power):
        find_symbols(expression, node.left, symbols)
        find_symbols(expression, node.right, symbols)
        
    elif type(node) == UnaryMinus:
        find_symbols(expression, node.value, symbols)

    return symbols


def create_polynomial_matrix(expression_tree):
    full_terms = extract_terms(expression_tree)
    
    # Finds all variables and functions
    # parser = PolynomialParser("")
    # parser.expr_tree = expression_tree

    # Use a function to order the symbols in a conssistent structure
    # ordered by class name and variable name
    all_symbols = sorted(list(find_symbols(expression_tree)), key=lambda x: (type(x).__name__, str(x)))
    
    matrix_rows = []
    for i, original_term in enumerate(full_terms):

        term_index = i
        symbols_dict = {}
        term_sign = 1
        coefficient = 1
        
        
        if type(original_term) == UnaryMinus:
            term_sign = -1
            inner_term = original_term.value

        else:
            inner_term = original_term

        # print(term_index)
        if type(inner_term) == Constant:
            coefficient = inner_term.value
        else:
            if type(inner_term) == Multiplication:
                factors = flatten_multiplication_factors(inner_term)

                # Extracts coefficient for mnested multiplication
                coeff_factors = [f for f in factors if type(f) == Constant]

                #print("Coefficient factors found:", coeff_factors)

                if coeff_factors:
                    coefficient = coeff_factors[0].value

            # Tries to factor out each symbol from the term
            for symbol in all_symbols:
                power = 0
                current_term = inner_term
                
                while True:
                    factored_result = try_factorizing(current_term, symbol)
                    
                    if factored_result is not None:
                        power += 1
                        current_term = factored_result
                    else:
                        break
                
                if power > 0:
                    symbols_dict[symbol] = power
        


        # Starts building the dictionary which will become a row in the matrix
        row = {'term_id': i}
        
        for symbol in all_symbols:
            if symbol in symbols_dict:
                row[symbol] = symbols_dict[symbol]
            else:
                row[symbol] = 0

        row["coefficient"] = abs(coefficient)

        if coefficient >= 0:
            row["sign"] = term_sign
        else:
            row["sign"] = -term_sign
        row['original_term'] = original_term

        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    return matrix_df, all_symbols


def largest_common_cube(matrix_rows, variables):

    common_cube = {}

    for var in variables:

        var_numbers = [row[var] for row in matrix_rows]
        # print(var_numbers)
        min_power = min(var_numbers)
        # print(min_power)

        if min_power > 0:
            common_cube[var] = min_power
    
    # print(common_cube)
    return common_cube


def divide_rows_by_cube(matrix_rows, cube_divisor, variables):

    row_divided = []

    for row in matrix_rows:

        # Copying the data is CRITICAL in order to preserve the information of the original expression
        # By creating a copy the correct mapping can be made between the kernel and the original, unfactorized expression
        # Did not work without this
        divided_row = row.copy()

        # Cube divisor has all variables as keys and their powers as values
        for var in list(cube_divisor.keys()):

            # print(divided_row, type(var), type(list(cube_divisor.keys())[0]))

            # Divide by subtracting powers
            if var in cube_divisor:
                divided_row[var] = row[var] - cube_divisor[var]
            else:
                divided_row[var] = row[var]
            
            # print(divided_row)

        row_divided.append(divided_row)


    return row_divided


def cube_free(matrix_rows, variables):

    # If there is only one row it is automatically cube free because there is only one factor
    if len(matrix_rows) < 2:
        return True
    
    common_cube = largest_common_cube(matrix_rows, variables)

    if len(common_cube)== 0:
        return True
    else:
        return False


def cube_dict_to_term(cube_dict, symbols):

    
    factors = []
    sorted_symbols = sorted(symbols, key=lambda x: (type(x).__name__, str(x)))

    # Accumulates all the factors
    for symbol in sorted_symbols:
        if symbol in cube_dict and cube_dict[symbol] > 0:
            if cube_dict[symbol] ==1:
                factors.append(symbol)

            else:
                factors.append(Power(symbol, Constant(cube_dict[symbol])))
    
    # Compiles the factors into nested Multiplications
    if len(factors) == 0:
        return Constant(1)
    elif len(factors) == 1:
        return factors[0]
    else:
        result = factors[0]
        for factor in factors[1:]:
            result = Multiplication(result, factor)
        return result




def convert_row_to_term(row, symbols):
    coeff = row['coefficient']
    sign = row['sign']
    coefficient = coeff * sign
    
    symbol_factors = []
    for symbol in sorted(symbols, key=lambda x: (type(x).__name__, str(x))):
        if row[symbol] > 0:
            if row[symbol] == 1:
                symbol_factors.append(symbol) 
            else:
                symbol_factors.append(Power(symbol, Constant(row[symbol])))


    # Combine the coefficient if there is one
    # Eneter loop only if there is a symbol, otherwise it means there is only a coefficient
    if coefficient == 1 and symbol_factors:
        if len(symbol_factors) ==1:
            return symbol_factors[0]
        else:
            result = symbol_factors[0]
            for factor in symbol_factors[1:]:
                result = Multiplication(result, factor)
            return result
    
        
    elif coefficient == -1 and symbol_factors:
        if len(symbol_factors) == 1:
            return UnaryMinus(symbol_factors[0])
        
        else:
            result = symbol_factors[0]
            for factor in symbol_factors[1:]:
                result = Multiplication(result, factor)
                
            return UnaryMinus(result)
        
    elif coefficient!= 1 and symbol_factors:
        coeff_factor = Constant(abs(coefficient))
        result = coeff_factor

        for factor in symbol_factors:
            result = Multiplication(result, factor)

        if coefficient < 0:
            return UnaryMinus(result)
        
        else:
            return result
    else:
        return Constant(coefficient)



def rows_to_expression(matrix_rows, symbols):
    if len(matrix_rows) == 0:
        return Constant(0)
    
    # Converts each row to a single term
    terms = [convert_row_to_term(row, symbols) for row in matrix_rows]
    
    # Combines the terms into a combination of additions and subtractions
    if len(terms) == 1:
        return terms[0]
    else:
        result = terms[0]
        for term in terms[1:]:
            if type(term) == UnaryMinus:
                result = Subtraction(result, term.value)

            elif type(term) == Constant and term.value < 0: 
                result = Subtraction(result, Constant(abs(term.value)))

            else:
                result = Addition(result, term)

        return result
    




def display_results(kernels_cokernels):
    print("Kernel-cokernel pairs found: ", len(kernels_cokernels))
    
    for i, item in enumerate(kernels_cokernels):
        print("Pair", i+1)
        print("Cokernel:", str(item['cokernel']))
        print("Kernel: ", str(item['kernel']))
        





def extract_kernels(matrix_rows, symbols):

    kernels_cokernels = []
    
    for var_index, selected_symbol in enumerate(symbols):
        print("analyzing variable: ", selected_symbol)

        #print("Variable index:", var_index)
        
        # Find all rows in the matrix (terms) that contain the selected variable
        terms_with_var = [row for row in matrix_rows if row[selected_symbol] > 0]
        
        # If there is only one row, it is not possible to extract a kernel
        if len(terms_with_var) < 2:
            continue
        
        print("Terms that contain the selected symbol: ", selected_symbol, terms_with_var)
        
        # From all these terms finds the largest possible cube
        common_cube = largest_common_cube(terms_with_var, symbols)
        

        if len(common_cube) == 0:

            #print("No common cube found for", selected_symbol)
            continue
            
        print("Common cube found: ", common_cube)
        
        # if selected_symbol not in common_cube:
        #     print("Common cube doesn't contain symbol")
        #     continue
            

        # From the terms, divide out the common cube found
        divided_rows = divide_rows_by_cube(terms_with_var, common_cube, symbols)
        
        if not cube_free(divided_rows, symbols):
            print("Result not cube free")
            continue
            
        if len(divided_rows) < 2:
            print("Result has not enough terms")
            continue
            

        # The common cube is the cokernel and the divided rows are the kernel's terms
        # Create the expressiion version of the identified kernel-cokernel pair
        cokernel = cube_dict_to_term(common_cube, symbols)
        kernel = rows_to_expression(divided_rows, symbols)
        
        kernel_cokernel_pair = {
            'cokernel_dict': common_cube,
            'kernel_rows': divided_rows,
            'cokernel': cokernel,
            'kernel': kernel,
            'term_ids': [row['term_id'] for row in divided_rows]
        }
        
        kernels_cokernels.append(kernel_cokernel_pair)
        
        print("Found kernel-cokernel pair")
        print("Cokernel: ",cokernel)
        print("Kernel: ", kernel)


        # When a pair is found, the process should repeat for the extracted kernel to see if there are any subkernels        
       
        if len(divided_rows) > 2: 
            print("Recursively analyzing kernel")

            sub_kernels = extract_kernels(divided_rows, symbols)
            
            
            for sub_kernel in sub_kernels:
                # The cokernels need to be combined: new_cokernel = current cokernel * sub_cokernel
                # The current cokernel is stored as common_cube in the form of a dictionary
                combined_cokernel_dict = common_cube.copy()

                for var, power in sub_kernel['cokernel_dict'].items():
                    if var in combined_cokernel_dict:
                        combined_cokernel_dict[var] += power
                    else:
                        combined_cokernel_dict[var] = power
                
                combined_cokernel = cube_dict_to_term(combined_cokernel_dict, symbols)
                
                sub_kernel_pair = {
                    'cokernel_dict': combined_cokernel_dict,
                    'kernel_rows': sub_kernel['kernel_rows'],
                    'cokernel': combined_cokernel,
                    'kernel': sub_kernel['kernel'],
                    'term_ids': sub_kernel['term_ids']
                }
                
                kernels_cokernels.append(sub_kernel_pair)
                print("Recursive kernel:" + str(combined_cokernel) + "*" + str(sub_kernel['kernel']))
    
    return kernels_cokernels



def find_all_kernels_cokernels_single_expression(expression_str):

    print("Expression: ", expression_str)
    

    parser = PolynomialParser(expression_str)
    expression_tree = parser.expr_tree
    

    matrix_df, symbols = create_polynomial_matrix(expression_tree)
    
    print("Polynomial Matrix:")
    print(matrix_df)
    
    matrix_rows = matrix_df.to_dict('records')
    
    # Possible kernels and cokernels are automatically populated with the expression and a co kernel of 1
    kernels_cokernels = [{
        'cokernel_dict': {},
        'kernel_rows': matrix_rows,
        'cokernel': Constant(1),
        'kernel': expression_tree
    }]
    
   
    extracted_kernels = extract_kernels(matrix_rows, symbols)
    kernels_cokernels.extend(extracted_kernels)
    

    # As a final check make sure there are no duplicate kernel-co kernel pairs
    unique_kernels = []
    seen_pairs = set()
    
    for item in kernels_cokernels:
        kernel_co_pair = (str(item['cokernel']), str(item['kernel']))
        if kernel_co_pair not in seen_pairs:
            seen_pairs.add(kernel_co_pair)
            unique_kernels.append(item)
    
    return unique_kernels, symbols


def get_symbols_and_boundaries(expression_strs):

    all_symbols = set()
    expression_boundaries = []
    current_term_id = 0
    
    for _, expr in enumerate(expression_strs):
        expression_tree = expr.expr_tree
        
        # Find the symbols in the expression
        expr_symbols = find_symbols(expression_tree)
        all_symbols.update(expr_symbols)
        
        # Count the number of terms in the expression
        expr_terms = extract_terms(expression_tree)
        number_terms = len(expr_terms)
        
        # Define the boundaries of the expression
        start_term_id = current_term_id
        end_term_id = current_term_id + (number_terms - 1)
        expression_boundaries.append((start_term_id, end_term_id))
        
        current_term_id += number_terms
    
    # Convert the symbols to a sorted list for consistency
    all_symbols = sorted(list(all_symbols), key=lambda x: (type(x).__name__, str(x)))
    
    return all_symbols, expression_boundaries




def find_kernels_cokernels(expression_strs):
  
    print("Processing these expressions independently")

    for i, expr_str in enumerate(expression_strs):
        print("Expression: ", expr_str)

    print("=" * 20)
    
    # Parse all the expressions
    expression_trees = []
    for expr_str in expression_strs:
        parser = PolynomialParser(expr_str)
        expression_trees.append(parser)

    
    # Fetch all the symbols and the number of each term
    # Term numbering starts at the first expression at 0 and ends in the last expression
    # Boundaries are important in order to keep track of which factorized term belongs to which expression

    all_symbols, expression_boundaries = get_symbols_and_boundaries(expression_trees)
    print("Expression boundaries: ", expression_boundaries)
    print("All symbols found:", [str(s) for s in all_symbols])
    
    # Process each expression independently for kernel extraction
    all_kernels = []
    
    for expr_index, expr_str in enumerate(expression_strs):
        print("Processing Expression: ", expr_index, expr_str)
        
        # Extract the kernels for this single expression
        single_kernels, single_symbols = find_all_kernels_cokernels_single_expression(expr_str)
        
        # Fetches the start end term id of the expression being handled
        start_term_id, end_term_id = expression_boundaries[expr_index]
        

        for kernel_item in single_kernels:
            
            
            adjusted_kernel_rows = []
            for row in kernel_item['kernel_rows']:

                # Update the kernel id so that it corresponds with the correct value in the correct expression
                # When the ids were initially made the function just considered a single function
                adjusted_row = {'term_id': row['term_id'] + start_term_id, 'expression_id': expr_index}
                
                # Add all symbols to the row
                # Columns with all zeroes are removed in a different stage
                for symbol in all_symbols:
                    if symbol in row:
                        adjusted_row[symbol] = row[symbol]
                    else:
                        adjusted_row[symbol] = 0
                
                # Add the rest of the row related information
                adjusted_row["coefficient"] = row['coefficient']
                adjusted_row["sign"] = row["sign"]
                adjusted_row["original_term"] = row["original_term"]
                
                adjusted_kernel_rows.append(adjusted_row)
            
            # Create the adjusted kernel dictionary with the correct id values
            adjusted_kernel = {
                'cokernel_dict': kernel_item['cokernel_dict'],
                'kernel_rows': adjusted_kernel_rows,
                'cokernel': kernel_item['cokernel'],
                'kernel': kernel_item['kernel'],
                'term_ids': [row['term_id'] for row in adjusted_kernel_rows],
                'source_expression': expr_index
            }
            
            all_kernels.append(adjusted_kernel)
        
        
    
    print("Total kernel co kernel pairs: ", len(all_kernels))

    return all_kernels, all_symbols, expression_boundaries



def analyze_polynomial(expression_strs):

    
    kernels_cokernels, symbols, expression_boundaries = find_kernels_cokernels(expression_strs)
    
    
    
    print("Combined kernel co-kernel pairs")
    
    by_expression = {}
    for item in kernels_cokernels:
        expr_id = item['source_expression']
        
        if expr_id not in by_expression:
            by_expression[expr_id] = []
        by_expression[expr_id].append(item)

    for expr_id in sorted(by_expression.keys()):
        
        print("From Expression", expr_id)
        
        for i, item in enumerate(by_expression[expr_id]):
            print("Cokernel:", str(item['cokernel']))
            print("Kernel:", str(item['kernel']))
            print("Term IDs:", item['term_ids'])
        
    
    kcm_matrix = create_kernel_cube_matrix_multi(kernels_cokernels, symbols, expression_boundaries)
    
    print("Kernel Cube Matrix (KCM)")
    print(kcm_matrix)
    
    return kcm_matrix, kernels_cokernels, symbols, expression_boundaries




def create_kernel_cube_matrix_multi(kernels_cokernels, symbols, expression_boundaries):

    # Line to remove rows with co kernel of 1. It is excluded because a function is always factorizable by 1
    kernels_cokernels = [k for k in kernels_cokernels if str(k['cokernel']).strip() not in ["1", "1.0", "1.00"]]

    if len(kernels_cokernels) == 0:
        print("No kernels found")
        return pd.DataFrame()

    
    # Makes sure to not have the cube as a column
    cube_order = []
    seen_cubes = set()
    

    for item in kernels_cokernels:
        for row in item['kernel_rows']:
            cube_term = convert_row_to_term(row, symbols)
            if str(cube_term) not in seen_cubes:
                cube_order.append(cube_term)
                seen_cubes.add(str(cube_term))


    kcm_matrix = []
    
    for item in kernels_cokernels:
        row_data = {'cokernel': item['cokernel']}
        
        kernel_cubes = {}
        
        # This builds the kernel columns to be added to the KCM
        for row in item['kernel_rows']:
            # Convert the row to its related term
            cube_term = convert_row_to_term(row, symbols)

            if cube_term not in kernel_cubes:
                kernel_cubes[cube_term] = set()

            # Notes the term numbers which resulted in this kernel
            kernel_cubes[cube_term].add(row['term_id'])
        
        # Fill into the right location in the row whether this cube is present and if it is the index it belongs to
        for cube_term in cube_order:

            if cube_term in kernel_cubes:
                term_ids = sorted(list(kernel_cubes[cube_term]))

                if term_ids:
                    term_id_strings = []

                    for term_id in term_ids:

                        term_id_strings.append(str(term_id))
                        # print(term_id_strings)

                    merged_ids = ','.join(term_id_strings)

                    row_data[cube_term] = "1(" + merged_ids + ")"


                else:
                    row_data[cube_term] = 1
            else:
                row_data[cube_term] = 0
        
        kcm_matrix.append(row_data)
    
    return pd.DataFrame(kcm_matrix)




def distill_algorithm(kcm_df, expression_strings=None, expression_boundaries=None, seed_num = 5):

    
    cube_columns = [col for col in kcm_df.columns if col!= 'cokernel']
    if len(cube_columns) == 0:
        print("No columns")
        return None
    

    
    # Convert KCM to a binary matrix
    binary_matrix, term_id_matrix = kcm_to_list_form(kcm_df, cube_columns)
    num_rows, num_cols = binary_matrix.shape
    
    
    # Show binary matrix or term id matrix
    print("Binary matrix")
    for i in range(num_rows):
        # cokernel = kcm_df.iloc[i]['cokernel']
        printed_row = [(binary_matrix[i][j]) for j in range(num_cols)]
        print(printed_row)
    

    selected_rectangles = []
    working_binary_matrix = binary_matrix.copy()
    working_term_id_matrix = term_id_matrix.copy()
    
    rectangle_iteration = 0
    while True:
        rectangle_iteration += 1
        
        # Generate seeds from current working matrix
        seeds = generate_distill_seeds(working_binary_matrix, seed_num)
        print("Generated seeds: ", seeds)
        
        if not seeds:
            print("No more seeds available")
            break
            
        # Following the seeds, take each selected seed and greedily expand the rectangle

        possible_rectangles = []
        seen_rectangles = set()  

        for i, seed in enumerate(seeds):
            expanded_rect = expand_rectangle_greedily(seed, working_binary_matrix, kcm_df, cube_columns, working_term_id_matrix, expression_strings, expression_boundaries)
            
            if expanded_rect:

                specific_rectangle = (
                    tuple(sorted(expanded_rect['rows'])),
                    tuple(str(col) for col in expanded_rect['columns']),
                    tuple(sorted(expanded_rect['original_term_ids']))
                )
                
                # Only add if not seen before
                if specific_rectangle not in seen_rectangles:
                    seen_rectangles.add(specific_rectangle)
                    possible_rectangles.append(expanded_rect)

                # Select the best rectangle from this iteration
                best_rectangle = select_best_rectangle(possible_rectangles, kcm_df)
                
                if best_rectangle is None:
                    print("No best rectangle selected")
                    break
            
        print("Selected rectangle covering terms: ", best_rectangle['original_term_ids'])
        selected_rectangles.append(best_rectangle)
        
        # Remove covered terms from working matrices
        working_binary_matrix, working_term_id_matrix = remove_covered_terms(working_binary_matrix, working_term_id_matrix, best_rectangle)

        
        print("Updated matrix:")
        for i in range(num_rows):
            # cokernel = kcm_df.iloc[i]['cokernel']
            printed_row = [str(working_binary_matrix[i][j]) for j in range(num_cols)]
            print(printed_row)
    
    print("Rectangles selected:", selected_rectangles)
    
    if not selected_rectangles:
        return None
    
    return selected_rectangles



def kcm_to_list_form(kcm_df, cube_columns):

    binary_matrix = np.zeros((len(kcm_df), len(cube_columns)), dtype=int)
    term_id_matrix = {}
    
    
    for i in range(len(kcm_df)):
        cokernel = kcm_df.iloc[i]['cokernel']
        # kernel = kcm_df.iloc[i]['kernel']
        
        
        for j, col in enumerate(cube_columns):
            cell_value = str(kcm_df.iloc[i][col])
            
            # Parse different cell value formats
            if cell_value == '0':
                binary_matrix[i][j] = 0
                term_id_matrix[(i, j)] = []
                
                
            elif '(' in cell_value and ')' in cell_value:
                binary_matrix[i][j] = 1

                # Extracts term ids from the parentheses
                start = cell_value.find('(') + 1 
                end = cell_value.find(')')

                term_ids_str = cell_value[start:end]

              
                term_ids = [int(num) for num in term_ids_str.split(',')]
                term_id_matrix[(i, j)] = term_ids

    return binary_matrix, term_id_matrix


def generate_distill_seeds(binary_matrix, seed_num):

    num_rows, num_cols = binary_matrix.shape
    seeds = []
    
    # Finds how populated each column and row is
    row_activity = [sum(binary_matrix[i][j] for j in range(num_cols)) for i in range(num_rows)]
    col_activity = [sum(binary_matrix[i][j] for i in range(num_rows)) for j in range(num_cols)]
    

    
    # The rows are filterd based on the ones that have at least a value of 2. This means there is a potential kernel to be found
    # A new seed notes the row number as well as the corresponding columns
    for i in range(num_rows):
        if row_activity[i] > 1:
            linked_cols = [j for j in range(num_cols) if binary_matrix[i][j] ==1]
            seeds.append({
                'rows': [i],
                'cols': linked_cols,
                'activity': row_activity[i]
            })
            print("Row seed: ", i, linked_cols, row_activity[i])
    
    # Similar process with the columns
    for j in range(num_cols):
        if col_activity[j] > 1:
            linked_rows = [i for i in range(num_rows) if binary_matrix[i][j] ==1]
            seeds.append({
                'rows': linked_rows,
                'cols': [j],
                'activity': col_activity[j]
            })
            print("Column seed: ", j, linked_rows, col_activity[j])
    
    # Sort seeds by activity and select the top 10
    seeds.sort(key=lambda x: x['activity'], reverse=True)
    selected_seeds = seeds[:seed_num]
    
    print("Selected: ", len(selected_seeds), "out of", len(seeds))

    return selected_seeds



def expand_rectangle_greedily(seed, binary_matrix, kcm_df, cube_columns, term_id_matrix, expression_strings, expression_boundaries):

    num_rows, num_cols = binary_matrix.shape
    current_rows = seed['rows'][:]
    current_cols = seed['cols'][:]
    
    print("Trying to expand with initial rows and columns:", current_rows, current_cols)
    
    improved = True
    iteration = 0
    # Define maximum number of iterations to expand
    max_iterations = 4
    
    while improved and iteration < max_iterations:
        # At the start of iteration to note if the rectangle has been improved in this cycle
        improved = False
        iteration += 1
        
        # First, collect all term IDs currently in the rectangle
        current_term_ids = set()
        for row in current_rows:
            for col in current_cols:
                term_ids = term_id_matrix.get((row, col), [])
                if type(term_ids) == list:
                    current_term_ids.update(term_ids)
                else:
                    if term_ids:
                        current_term_ids.add(term_ids)
        
        # Tries to expand the rows by finding rows that have 1 in all the current columns
        for test_row in range(num_rows):

            if test_row not in current_rows:
                # Check if all current columns have 1s in this row
                if all(binary_matrix[test_row][col] ==1 for col in current_cols):


                    new_term_ids = set()
                    for col in current_cols:
                        term_ids = term_id_matrix.get((test_row, col), [])

                        if type(term_ids) == list:
                            new_term_ids.update(term_ids)
                        else:
                            if term_ids:
                                new_term_ids.add(term_ids)
                    
                    # Only add row if it brings different terms
                    if not new_term_ids.intersection(current_term_ids):
                        current_rows.append(test_row)
                        improved = True
                        break
        
        # Tries to do the same thing with the columns
        for test_col in range(num_cols):
            if test_col not in current_cols:

                # Check if all current rows have 1s in this column
                if all(binary_matrix[row][test_col] ==1 for row in current_rows):


                    new_term_ids = set()

                    for row in current_rows:
                        term_ids = term_id_matrix.get((row, test_col), [])

                        if type(term_ids) == list:
                            new_term_ids.update(term_ids)
                        else:
                            if term_ids:
                                new_term_ids.add(term_ids)
                    
                    # Only add column if it brings different terms
                    if not new_term_ids.intersection(current_term_ids):
                        current_cols.append(test_col)
                        improved = True
                        break
        
    # Check that the rectangle spans at least 1 row and 2 columns
    if len(current_rows) > 0 and len(current_cols) > 1:

        covered_term_ids = set()

        for row in current_rows:
            for col in current_cols:

                term_ids = term_id_matrix[(row, col)]

                if type(term_ids) == list:
                    covered_term_ids.update(term_ids)
                else:
                    covered_term_ids.add(term_ids)


        # Determine which expression this rectangle belongs to
        source_expression_string = None
        source_expression_tree = None
        source_expression_id = None
        source_expression_terms = []
        source_expression_term_ids = []
        
        # if expression_strings and expression_boundaries and covered_term_ids:

        # Finds the expression that contains the term ids
        for expr_id, (start_term, end_term) in enumerate(expression_boundaries):

            valid_term_ids = [term_id for term_id in covered_term_ids] #if term_id is not None] (not needed I think)

            # If one of the ids match with the range of the related expression
            if any(start_term <= term_id <= end_term for term_id in valid_term_ids):

                # Stores the initial expression
                source_expression_id = expr_id
                source_expression_string = expression_strings[expr_id]
                
                # Parses the original expression string to get the parsed expression
                parser = PolynomialParser(source_expression_string)
                source_expression_tree = parser.expr_tree
                
                # Extracts the terms from the source expression
                source_expression_terms = extract_terms(source_expression_tree)
                
                # Creates the list of term ids for this expression based on the start and end term

                source_expression_term_ids = list(range(start_term, end_term + 1))
                
                
                break
        
        # Stores all information into aa dictionary for use later on
        result = {
            'rows': current_rows,
            'columns': [cube_columns[col] for col in current_cols],
            'cokernels': [kcm_df.iloc[row_idx]['cokernel'] for row_idx in current_rows],
            'num_rows': len(current_rows),
            'num_cols': len(current_cols),
            'original_term_ids': list(covered_term_ids),
            'source_expression_string': source_expression_string,       
            'source_expression_tree': source_expression_tree,           
            'source_expression_id': source_expression_id,               
            'source_expression_terms': source_expression_terms,         
            'source_expression_term_ids': source_expression_term_ids    
        }
        
        
        return result
    

    print("Invalid rectangle, something wrong")
    return None



def select_best_rectangle(candidate_rectangles, kcm_df):
    best_rectangle = None
    lowest_cost = np.inf
    
    
    for i, rect in enumerate(candidate_rectangles):
        cost = calculate_rectangle_cost(rect, kcm_df)
        print("Rectangle ", i, "with cost", cost)
        
        if cost < lowest_cost:
            lowest_cost = cost
            best_rectangle = rect
            best_rectangle['cost'] = cost
    
    return best_rectangle



def find_existing_function(function_expression, extracted_functions):

    for existing_func in extracted_functions:
        if equal_terms(existing_func.value, function_expression):
            return existing_func
    return None



def calculate_rectangle_cost(rect, kcm_df):
    
    print("Source expression: ", rect['source_expression_string'])
    
    # Fetch the information from the rectangle
    source_terms = rect['source_expression_terms']
    source_term_ids = rect['source_expression_term_ids']
    covered_term_ids = set(rect['original_term_ids'])
    
    print("All term ids in source: ", source_term_ids)
    print("Covered (factored) term IDs: ", list(covered_term_ids))
    print("All terms: ", [str(term) for term in source_terms])
    
    # Group into factored and unfactored terms
    factored_terms = []
    unfactored_terms = []
    
    for i, term_id in enumerate(source_term_ids):

        if i < len(source_terms):
            term = source_terms[i]

            if term_id in covered_term_ids:
                factored_terms.append((term_id, term))
                print("Factored term: ", term)
            else:
                unfactored_terms.append((term_id, term))
                print("Unfactored term: ", term)
    

    # Construct the factored part of the expression co kernel * kernel
    # Build the factored part of the expression
    if len(rect['rows'])==1:
        # Single cokernel case - keep existing logic
        row_index = rect['rows'][0]
        cokernel = kcm_df.iloc[row_index]['cokernel']
        
        # The cubes that make up the kernel
        kernel_cubes = list(rect['columns'])
        
        # Combine the cubes
        if len(kernel_cubes)==1:
            kernel = kernel_cubes[0]
        else:
            kernel = kernel_cubes[0]
            for cube in kernel_cubes[1:]:

                if type(cube)== UnaryMinus:
                    kernel = Subtraction(kernel, cube.value)

                elif type(cube)== Constant and cube.value < 0:
                    kernel = Subtraction(kernel, Constant(abs(cube.value)))
                
                else:
                    kernel = Addition(kernel, cube)
        
        # Create the multiplication
        if type(cokernel)== Constant and cokernel.value == 1:
            factored_part = kernel
        else:
            factored_part = Multiplication(cokernel, kernel)
        
        factored_parts = [factored_part]
    else:
        # Multiple co kernels
        cokernels = [kcm_df.iloc[row_idx]['cokernel'] for row_idx in rect['rows']]
        
        # Combine co-kernels
        combined_cokernel = cokernels[0]
        for ck in cokernels[1:]:
            if type(ck)== UnaryMinus:
                combined_cokernel = Subtraction(combined_cokernel, ck.value)
            elif type(ck)== Constant and ck.value < 0: 
                combined_cokernel = Subtraction(combined_cokernel, Constant(abs(ck.value)))
            else:
                combined_cokernel = Addition(combined_cokernel, ck)
        
        # Build kernel from columns
        kernel_cubes = list(rect['columns'])
        if len(kernel_cubes)==1:
            kernel = kernel_cubes[0]
        else:
            kernel = kernel_cubes[0]
            for cube in kernel_cubes[1:]:
                if type(cube)== UnaryMinus:
                    kernel = Subtraction(kernel, cube.value)
                elif type(cube)== Constant and cube.value < 0:
                    kernel = Subtraction(kernel, Constant(abs(cube.value)))
                else:
                    kernel = Addition(kernel, cube)
        
        # Create (combined_cokernel) * kernel
        factored_part = Multiplication(combined_cokernel, kernel)
        factored_parts = [factored_part]
    

    # Build the complete expression by combining the factored and unfactored parts
    all_parts = []
    
    # Add the unfactored terms. Here the term id is crucial, to be able to have right term in the rightr list (factored or unfactored)
    for term_id, term in unfactored_terms:
        all_parts.append(term)
        print("Adding the unfactored term: ", term)
    
    # Add factored terms
    for factored_terms in factored_parts:
        all_parts.append(factored_terms)
        print("Adding factored part: ", factored_terms)

    
    # Combine all parts into the complete expression
    
    if len(all_parts)== 1:
        complete_expression = all_parts[0]
    else:
        complete_expression = all_parts[0]
        for part in all_parts[1:]:
            if type(part)== UnaryMinus:
                complete_expression = Subtraction(complete_expression, part.value)
            elif type(part)== Constant and part.value < 0:  # FIX: Handle negative constants
                complete_expression = Subtraction(complete_expression, Constant(abs(part.value)))
            else:
                complete_expression = Addition(complete_expression, part)
    
    
    # Store the reconstructed expression since it has been built
    rect['reconstructed_expression'] = complete_expression

    print("FULL EXPRESSION: ", complete_expression)

    cost = DSPBlockNumber(str(complete_expression))
    print("DSP Block Number Cost: ", cost)
    
    return cost




def create_function_from_rectangle(rect, kcm_df, function_name):    
   

    # The function value is the sum of its cubes that make the kernel
    function_expression = None  # Change from [] to None

    for cube_name in rect['columns']:
        if function_expression is None:  # Change condition
            function_expression = cube_name
        else:
            print(cube_name, type(cube_name))
            if type(cube_name)== UnaryMinus:
                function_expression = Subtraction(function_expression, cube_name.value)
            elif type(cube_name)== Constant and cube_name.value < 0:
                function_expression = Subtraction(function_expression, Constant(abs(cube_name.value)))

            else:
                function_expression = Addition(function_expression, cube_name)

    # Assign the function to the correct class
    extracted_function = Function(function_name)
    extracted_function.value = function_expression
    

    extracted_function.rectangle_info = {
        'rows': rect['rows'],
        'columns': rect['columns'], 
        'cokernels': rect['cokernels'],
        'cubes': rect['columns'],
        'optimization_value': rect['cost'],
        'original_term_ids': rect['original_term_ids'],
        'reconstructed_expression': rect['reconstructed_expression'],
        'polynomial_expression': function_expression 
   }
    
    print("Created Function with value: ", function_expression)
    
    return extracted_function





def group_terms(expression, position = 0):
    # Goes through and groups terms seperated by subtraction and addition. These are the groups
    terms = []
    
    if type(expression) == Addition:
        left_terms = group_terms(expression.left, position)
        terms.extend(left_terms)

        right_position = position + len(left_terms)
        right_terms = group_terms(expression.right, right_position)

        terms.extend(right_terms)
    elif type(expression) == Subtraction:
        left_terms = group_terms(expression.left, position)
        terms.extend(left_terms)

        right_position = position + len(left_terms)
        right_terms = group_terms(expression.right, right_position)

        for term, pos in right_terms:
            terms.append((UnaryMinus(term), pos))
    else:
        terms.append((expression, position))
    
    return terms



def find_existing_function(function_expression, extracted_functions):
    for existing_func in extracted_functions:
        if equal_terms(existing_func.value, function_expression):
            return existing_func
    return None



def update_expressions_with_extracted_function(all_expressions, analyzable_expressions, extracted_function, best_rect, expression_boundaries):

    # Shows which expression the rectangle came from, i.e. the expression that can be expressed in terms of the function
    source_expr_id = best_rect['source_expression_id']

    # print("Rectangle ids: ", best_rect['original_term_ids'])
   

    expr_to_modify = analyzable_expressions[source_expr_id]
    print("Expression to modify", expr_to_modify)
    
    
    # Find this expression in the list of expressions. It can either be the main function or one of the defined functions
    original_expr_index = None
    is_function_definition = False
    
    # Checks the main expression
    for i, expr in enumerate(all_expressions):
        if expr == expr_to_modify:
            original_expr_index = i
            is_function_definition = False
            break
    
    # Checks the function expressions
    if original_expr_index is None:
        for i, expr in enumerate(all_expressions):

            if '=' in expr:

                func_name, func_body = expr.split('=', 1)
                func_body = func_body.strip()

                if func_body == expr_to_modify:
                    original_expr_index = i
                    is_function_definition = True
                    break
    

    
    updated_expressions = all_expressions.copy()
    # factored_term_ids = set(best_rect['original_term_ids'])

    # Parse the expression to get terms
    parser = PolynomialParser(expr_to_modify)
    all_terms = extract_terms(parser.expr_tree)
    #print("All terms:", all_terms)

    
    # Need to identify which terms in the current expression were covered by the rectangle
    # This is done by checking which terms can be factored by the co kernel

    cokernel = best_rect['cokernels'][0]


    actually_factored_terms = []
    actually_factored_indices = []

    for i, term in enumerate(all_terms):

        factored_result = try_factorizing(term, cokernel)
        if factored_result is not None:

            # Term can be factored 
            actually_factored_terms.append(term)
            actually_factored_indices.append(i)
            

    # Build unfactored terms
    unfactored_terms = []
    for i, term in enumerate(all_terms):
        if i not in actually_factored_indices:
            unfactored_terms.append(term)

    print("Factored terms: ", [str(t) for t in actually_factored_terms])
    print("Unfactored terms: ", [str(t) for t in unfactored_terms])
    
    # Build the new expression
    new_parts = []

    # Add unfactored terms
    for term in unfactored_terms:
        new_parts.append(term) 

    # Add factored part
    cokernels = best_rect['cokernels']
    if len(cokernels) ==1:
        # Single cokernel case
        cokernel = cokernels[0]
        function_term = Function(extracted_function.name)
        
        if type(cokernel) == UnaryMinus:
            multiplication = Multiplication(cokernel.value, function_term)

            factored_part = UnaryMinus(multiplication)
            
        else:
            factored_part = Multiplication(cokernel, function_term)

        # print(factored_part)
        
        new_parts.append(factored_part)

    else:
        # Multiple cokernels - combine them
        combined_cokernel = cokernels[0]
        for ck in cokernels[1:]:
            if type(ck)== UnaryMinus:
                combined_cokernel = Subtraction(combined_cokernel, ck.value)
            else:
                combined_cokernel = Addition(combined_cokernel, ck)
        
        function_term = Function(extracted_function.name)
        factored_part = Multiplication(combined_cokernel, function_term)
        new_parts.append(factored_part)


    # Combine parts 
    
    if len(new_parts)==1:
        new_expr = new_parts[0]
    else:
        new_expr = new_parts[0]
        for part in new_parts[1:]:
            if type(part)== UnaryMinus:
                new_expr = Subtraction(new_expr, part.value)

            elif type(part) == Constant and part.value < 0:
                new_expr = Subtraction(new_expr, Constant(abs(part.value)))
                
            else:
                new_expr = Addition(new_expr, part)

    new_expr_string = str(new_expr)
        
    
    # Update the correct expression, main one or function one
    if is_function_definition:

        original_expr = updated_expressions[original_expr_index]
        func_name = original_expr.split('=')[0].strip()
        updated_expressions[original_expr_index] = str(func_name) + " = " + str(new_expr_string)
        print("Updated function definition: ", str(func_name) + " = " + str(new_expr_string))
    
    else:
        updated_expressions[original_expr_index] = new_expr_string
        print("Replaced main expression: ", new_expr_string)
    
    # Add new function definition
    updated_expressions.append( str(extracted_function.name) + " = " + str(extracted_function.value))
    
    return updated_expressions



def remove_covered_terms(binary_matrix, term_id_matrix, selected_rectangle):

    covered_term_ids = set(selected_rectangle['original_term_ids'])
    
    num_rows, num_cols = binary_matrix.shape
    
    for i in range(num_rows):
        for j in range(num_cols):
            if binary_matrix[i][j] ==1:
                cell_term_ids = term_id_matrix.get((i, j), [])
                
                if type(cell_term_ids) == list:
                    # Remove covered term IDs
                    remaining_term_ids = [tid for tid in cell_term_ids if tid not in covered_term_ids]
                    
                    if not remaining_term_ids:
                        binary_matrix[i][j] = 0  # No terms left, remove cell
                        term_id_matrix[(i, j)] = []
                    else:
                        term_id_matrix[(i, j)] = remaining_term_ids

    return binary_matrix, term_id_matrix



def greedy_kernel_intersection(expression_strings, seed_num):
    
    current_expressions = expression_strings.copy()
    iteration = 0
    extracted_functions = [] 
    
    # Continues until no better rectangle can be found
    while True:
        iteration += 1
        
        print("Iteration: ", iteration)
        print("Current expressions: ", current_expressions)
        
        # Expressions can either have an equal sign because they are functions or they do not because it is the intial input expression
        analyzable_expressions = []
        for expr in current_expressions:

            if '=' in expr:

                # The expression is a function
                func_name, function_expr = expr.split('=', 1)
                function_expr = function_expr.strip()
                
                parser = PolynomialParser(function_expr)
                terms = extract_terms(parser.expr_tree)
                if len(terms) > 1: 
                    analyzable_expressions.append(function_expr)
                    # print("Will analyze function: ", function_expr)
                

            else:
                # This is the main input expression
                parser = PolynomialParser(expr)
                terms = extract_terms(parser.expr_tree)
                analyzable_expressions.append(expr)
                
    
        
        kcm_matrix, kernels_cokernels, variables, expression_boundaries = analyze_polynomial(analyzable_expressions)
      
        
        print("Variables: ", [str(v) for v in variables])
        
        # if KCM is too small there is nothing more than can be cimplified
        if len(kcm_matrix)== 0 or len([col for col in kcm_matrix.columns if col!= 'cokernel']) < 2:
            print("KCM is too small")
            break
        

        # Find all best rectangles from this KCM
        selected_rectangles = distill_algorithm(kcm_matrix, analyzable_expressions, expression_boundaries, seed_num)

        if not selected_rectangles:
            print("No rectangles found")
            break


        # Create all functions first
        created_functions = []
        for rectangle_index, best_rect in enumerate(selected_rectangles):

            function_name = "G" + str(iteration) + chr(ord('a') + rectangle_index)
            extracted_function = create_function_from_rectangle(best_rect, kcm_matrix, function_name)
            
            extracted_functions.append(extracted_function)
            created_functions.append((extracted_function, best_rect))

            print("Created function:", function_name, extracted_function.value)

        # Update expressions with ALL rectangles at once
        current_expressions = update_expressions_with_all_functions(
            current_expressions, analyzable_expressions, created_functions, expression_boundaries
        )

        print("End of loop functions:")
        for i, func in enumerate(current_expressions):
            print(func)
            

    
    return extracted_functions, current_expressions




def update_expressions_with_all_functions(all_expressions, analyzable_expressions, created_functions, expression_boundaries):
    
    if not created_functions:
        return all_expressions
    
    # Group functions by source expression
    functions_by_expression = {}
    for extracted_function, best_rect in created_functions:
        expr_id = best_rect['source_expression_id']

        if expr_id not in functions_by_expression:
            functions_by_expression[expr_id] = []
        functions_by_expression[expr_id].append((extracted_function, best_rect))
    
    updated_expressions = all_expressions.copy()
    
    # Process each source expression that has rectangles
    for expr_id, function_rect_pairs in functions_by_expression.items():
        expr_to_modify = analyzable_expressions[expr_id]
        
        # Find the expression index
        original_expr_index = None
        is_function_definition = False
        
        for i, expr in enumerate(updated_expressions):
            if expr == expr_to_modify:
                original_expr_index = i
                is_function_definition = False
                break
        
        if original_expr_index is None:
            for i, expr in enumerate(updated_expressions):
                if '=' in expr:
                    func_name, func_body = expr.split('=', 1)
                    func_body = func_body.strip()
                    if func_body == expr_to_modify:
                        original_expr_index = i
                        is_function_definition = True
                        break
        
        if original_expr_index is None:
            continue
            
        # Parse the expression
        parser = PolynomialParser(expr_to_modify)
        all_terms = extract_terms(parser.expr_tree)
        
        # Apply ALL factorizations simultaneously
        new_parts = []
        used_term_indices = set()
        
        # Process each function/rectangle pair
        # Process each function/rectangle pair
        for extracted_function, best_rect in function_rect_pairs:
            cokernels = best_rect['cokernels']
            
            # For multiple cokernels need to check if terms can be factored by ANY of them
            factored_indices = []
            for i, term in enumerate(all_terms):
                if i not in used_term_indices:
                    # Check if this term can be factored by any cokernel
                    for cokernel in cokernels:
                        factored_result = try_factorizing(term, cokernel)

                        if factored_result is not None:
                            factored_indices.append(i)
                            break
            
            if factored_indices:
                # Mark these terms as used
                used_term_indices.update(factored_indices)
                
                # Create the factored part
                function_term = Function(extracted_function.name)
                
                if len(cokernels)==1:
                    # Single cokernel
                    cokernel = cokernels[0]
                    if type(cokernel)== UnaryMinus:
                        multiplication = Multiplication(cokernel.value, function_term)
                        factored_part = UnaryMinus(multiplication)
                    else:
                        factored_part = Multiplication(cokernel, function_term)
                else:
                    # Multiple cokernels - combine them
                    combined_cokernel = cokernels[0]
                    for ck in cokernels[1:]:
                        if type(ck)== UnaryMinus:
                            combined_cokernel = Subtraction(combined_cokernel, ck.value)
                        elif type(ck)== Constant and ck.value < 0: 
                            combined_cokernel = Subtraction(combined_cokernel, Constant(abs(ck.value)))
                        
                        else:
                            combined_cokernel = Addition(combined_cokernel, ck)
                    
                    factored_part = Multiplication(combined_cokernel, function_term)
                
                new_parts.append(factored_part)
        
        # Add remaining unfactored terms
        for i, term in enumerate(all_terms):
            if i not in used_term_indices:
                new_parts.append(term)
                print("Unfactored term", term)
        
        # Build the new expression
        if len(new_parts)==1:
            new_expr = new_parts[0]
        else:
            new_expr = new_parts[0]
            for part in new_parts[1:]:
                if type(part)== UnaryMinus:
                    new_expr = Subtraction(new_expr, part.value)
                elif type(part)== Constant and part.value < 0: 
                    new_expr = Subtraction(new_expr, Constant(abs(part.value)))
                else:
                    new_expr = Addition(new_expr, part)
        
        new_expr_string = str(new_expr)
        
        # Update the expression
        if is_function_definition:
            original_expr = updated_expressions[original_expr_index]
            func_name = original_expr.split('=')[0].strip()
            updated_expressions[original_expr_index] = str(func_name) + "=" + str(new_expr_string)
        else:
            updated_expressions[original_expr_index] = new_expr_string
        
        print("Updated expression:", new_expr_string)
    
    # Add all new function definitions
    for extracted_function, _ in created_functions:
        updated_expressions.append(str(extracted_function.name) + "=" + str(extracted_function.value))
    
    return updated_expressions





def optimize_polynomial(expression_str, seed_num):
    print("Expression: ",  expression_str)
    
    extracted_functions, final_expressions = greedy_kernel_intersection(
        expression_strings=[expression_str],
        seed_num = seed_num
    )
    
   
    print("Extracted functions:")
    for i, func in enumerate(extracted_functions):
        print(func.name, func.value)
    
    print("Final expressions:")
    for i, expr in enumerate(final_expressions):
        print(i, expr)
    
    return extracted_functions, final_expressions








def substitute_and_reconstruct(expression_list):

    parsed_expressions = {}
    main_expression = None
    
    for expr_str in expression_list:
        if '=' in expr_str:
            
            
            func_name, func_body = expr_str.split('=', 1)
            parser = PolynomialParser(func_body.strip())
            parsed_expressions[func_name.strip()] = parser.expr_tree
        else:
            # Main expression
            parser = PolynomialParser(expr_str)
            main_expression = parser.expr_tree
    

    
    final_expression = substitute_functions(main_expression, parsed_expressions)
    
    return final_expression


def substitute_functions(expression, function_definitions):

    if type(expression)== Function:
       

        func_name = expression.name
        func_body = function_definitions[func_name]
        return substitute_functions(func_body, function_definitions)
        
    
    elif type(expression)== Variable:

        var_name = expression.value
        if var_name in function_definitions:
            func_body = function_definitions[var_name]

            return substitute_functions(func_body, function_definitions)
        
        else:

            return expression
    
    elif type(expression) in (Addition, Subtraction, Multiplication, Power):

        left_substituted = substitute_functions(expression.left, function_definitions)
        right_substituted = substitute_functions(expression.right, function_definitions)
        
        if type(expression)== Addition:
            return Addition(left_substituted, right_substituted)
        
        elif type(expression)== Subtraction:
            return Subtraction(left_substituted, right_substituted)
        
        elif type(expression)== Multiplication:
            return Multiplication(left_substituted, right_substituted)
        
        elif type(expression)== Power:
            return Power(left_substituted, right_substituted)
        
    
    elif type(expression)== UnaryMinus:
        
        substituted_inner = substitute_functions(expression.value, function_definitions)
        return UnaryMinus(substituted_inner)
    
    else:
        return expression


