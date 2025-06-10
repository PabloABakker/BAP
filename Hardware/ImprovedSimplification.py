#!/usr/bin/env python
# coding: utf-8


from DataFlowModel import DSPBlockNumber
import pandas as pd
import numpy as np
from collections import defaultdict

class PolynomialTerm:
    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        return str(self.value)
    
    def is_number(self):
        return False
        
    def is_symbol(self):
        return False
    
    def could_extract_minus_sign(self):
        return False
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash(str(self))  
    
    def __eq__(self, other):
        if not isinstance(other, PolynomialTerm):
            return False
        return str(self) == str(other)  

class Variable(PolynomialTerm):
    def __init__(self, value):
        self.value = value
        
    def is_symbol(self):
        return True
    
    def __pow__(self, exponent):
        if isinstance(exponent, Constant):
            return Power(self, exponent)
        elif isinstance(exponent, (int, float)):
            return Power(self, Constant(exponent))
        
    def __hash__(self):
        return hash(self.value)
    
    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self.value == other.value
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __le__(self, other):
        return self.value <= other.value
    
    def __gt__(self, other):
        return self.value > other.value
    
    def __ge__(self, other):
        return self.value >= other.value
       


class Constant(PolynomialTerm):
    def __init__(self, value):
        self.value = float(value)
        
    def is_number(self):
        return True
        
    @property
    def is_integer(self):
        return self.value == int(self.value)
        
    def __gt__(self, second):
        if isinstance(second, Constant):
            return self.value > second.value
        return self.value > second
        
    def __eq__(self, second):
        if isinstance(second, Constant):
            return self.value == second.value
        return self.value == second
    
    def __hash__(self):
        return hash(self.value)
    
    def __mod__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value % other.value)
        return Constant(self.value % other)

    def __rmod__(self, other):
        return Constant(other % self.value)
    
    def __sub__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value - other.value)
        return Constant(self.value - other)

    def __rsub__(self, other):
        return Constant(other - self.value)
    
    def __pow__(self, exponent):
        if isinstance(exponent, Constant):
            return Constant(self.value ** exponent.value)
        return Constant(self.value ** exponent)
    
    def __floordiv__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value // other.value)
        return Constant(self.value // other)

    def __rfloordiv__(self, other):
        return Constant(other // self.value)


class BinaryOperation(PolynomialTerm):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    @property
    def args(self):
        return [self.left, self.right]


class Addition(BinaryOperation):
    def __str__(self):
        return f"{self.left} + {self.right}"
    

class Subtraction(BinaryOperation):
    def __str__(self):
        return f"{self.left} - {self.right}"
    

class Multiplication(BinaryOperation):
    def __str__(self):
        # Adds parentheses around additions and subtractions
        left_str = str(self.left)
        right_str = str(self.right)

        if isinstance(self.left, (Addition, Subtraction)):
            left_str = f"({left_str})"
            
        if isinstance(self.right, (Addition, Subtraction)):
            right_str = f"({right_str})"
            
        return f"{left_str} * {right_str}"
    

class Power(BinaryOperation):
    def __str__(self):
        # Adds parentheses around the base for clarity
        left_str = str(self.left)
        if isinstance(self.left, (Addition, Subtraction, Multiplication)):
            left_str = f"({left_str})"
            
        return f"{left_str}^{self.right}"
    

class UnaryMinus(PolynomialTerm):
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        term_str = str(self.value)
        if isinstance(self.value, (Addition, Subtraction, Multiplication)):
            term_str = f"({term_str})"
        return f"-{term_str}"
    
    def is_number(self):
        return self.value.is_number() if hasattr(self.value, 'is_number') else False
    
    def could_extract_minus_sign(self):
        return True
    
    @property
    def args(self):
        return [self.value]


class Function(PolynomialTerm):
    def __init__(self, name):
        self.name = name
        self.value = name
        
    def __str__(self):
        return self.name
    
    def is_symbol(self):
        return True
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if not isinstance(other, Function):
            return False
        return self.name == other.name


class PolynomialParser:
    def __init__(self, expr_string):
        self.expr_tree = None
        self.parse(expr_string)

    
    def parse(self, expr):
        self.expr = expr.replace(' ', '') 
        self.pos = 0
        self.expr_tree = self.parse_expression()
        return self
    
    def current_char(self):
        if self.pos >= len(self.expr):
            return None
        return self.expr[self.pos]
    
    def next_char(self):
        if self.pos >= len(self.expr):
            return None
        return self.expr[self.pos + 1] if self.pos + 1 < len(self.expr) else None

    
    def parse_expression(self):
        if self.current_char() == '-' and self.next_char() and not self.next_char().isdigit():
            self.pos += 1
            # left = Multiplication(Constant(-1.0), self.parse_term())

            left = UnaryMinus(self.parse_term())
            
        else:
            left = self.parse_term()
        
        while self.current_char() in ('+', '-'):
            op = self.current_char()
            self.pos += 1
            right = self.parse_term()
            
            if op == '+':
                left = Addition(left, right)
            else:
                left = Subtraction(left, right)
        
        return left
    
    def parse_term(self):
        left = self.parse_factor()
        
        while self.current_char() == '*':
            self.pos += 1
            right = self.parse_factor()
            left = Multiplication(left, right)
                
        return left
    
    def parse_factor(self):
        char = self.current_char()
        
        # End of expression check
        if char is None:
            return None
        
        #  unary minus
        if char == '-' and self.next_char() and not self.next_char().isdigit():
            self.pos += 1
            return UnaryMinus(self.parse_factor())
        
        # Brackets
        if char == '(':
            self.pos += 1
            expr = self.parse_expression()
            self.pos += 1  # Skip closing bracket
            
            if self.current_char() == '^':
                self.pos += 1
                exponent = self.parse_factor()
                return Power(expr, exponent)
                
            return expr
        
        #  numbers
        if char.isdigit() or (char == '-' and self.next_char() and self.next_char().isdigit()):
            return self.parse_number()
        
        #  variables
        if char.isalpha():
            var = self.parse_variable()
            
            # Check for power after variable
            if self.current_char() == '^':
                self.pos += 1
                exponent = self.parse_factor()
                return Power(var, exponent)

            return var
            
    
    def parse_number(self):
        start_pos = self.pos
        char = self.current_char()
        
        # Handle negative sign only at the beginning
        if char == '-':
            self.pos += 1
            char = self.current_char()
        
        # Parse digits and decimal point
        while char is not None and (char.isdigit() or char == '.'):
            self.pos += 1
            char = self.current_char()
        
        # Create constant
        value = float(self.expr[start_pos:self.pos])
        return Constant(value)
    
    def parse_variable(self):
        start_pos = self.pos
        char = self.current_char()
        
        while char is not None and char.isalnum():
            self.pos += 1
            char = self.current_char()
            
        variable = self.expr[start_pos:self.pos]
        return Variable(variable)
    

    def count(self, node=None):

        if node is None:
            node = self.expr_tree
        
        return self.count_terms(node)
    
    def count_terms(self, node):

        if type(node) in (Addition, Subtraction):
            left_count = self.count_terms(node.left)
            right_count = self.count_terms(node.right)
            return left_count + right_count
        else:
            return 1
        
    def find_variables(self, node = None, variables = None):

        if variables is None:
            variables = set()

        if node is None:
            node = self.expr_tree

        if isinstance(node, Variable):
            variables.add(node)

        elif isinstance(node, (Addition, Subtraction, Multiplication, Power)):
            self.find_variables(node.left, variables)
            self.find_variables(node.right, variables)
            
        elif isinstance(node, UnaryMinus):
            self.find_variables(node.value, variables)
    
        return variables
    

        
       



import pandas as pd
from IPython.display import display
import numpy as np
from itertools import combinations


# Extract terms from a node
def extract_terms(node, terms=None):
    if terms is None:
        terms = []
    
    if type(node) == Addition:
        extract_terms(node.left, terms)
        extract_terms(node.right, terms)
        
    elif type(node) == Subtraction:
        extract_terms(node.left, terms)
        extract_terms(UnaryMinus(node.right), terms)

    elif type(node) == UnaryMinus:
        terms.append(node)

    else:
        terms.append(node)
    
    return terms



# Tries to factorise a term with a cokernel
def try_factorising(term, co_kernel):

    if equal_terms(term, co_kernel):
        return Constant(1.0)
    
    if co_kernel == Constant(1):
        return term

    if type(term) == UnaryMinus and type(co_kernel) != UnaryMinus:
        inner_result = try_factorising(term.value, co_kernel)
        if inner_result is None:  
            return None
        return UnaryMinus(inner_result)
        
    if type(co_kernel) == UnaryMinus and type(term) != UnaryMinus:
        inner_result = try_factorising(term, co_kernel.value)
        if inner_result is None:  
            return None
        return UnaryMinus(inner_result)
    
        
    if type(term) == UnaryMinus and type(co_kernel) == UnaryMinus:
        
        return try_factorising(term.value, co_kernel.value)

    # Power factorisation by subtracting the powers
    if type(term) == Power and type(co_kernel) == Power:
        if equal_terms(term.left, co_kernel.left) and term.right.value > co_kernel.right.value:
            new_power = term.right.value - co_kernel.right.value
            if new_power == 1:
                return term.left 
            else:
                return Power(term.left, Constant(new_power))

    # Power and symbol factorisation
    if type(term) == Power and type(co_kernel) in (Variable, Function):
        if equal_terms(term.left, co_kernel) and term.right.value > 1:
            new_power = term.right.value - 1
            if new_power == 1:
                return term.left 
            else:
                return Power(term.left, Constant(new_power))

    if type(term) == Multiplication:
        term_factors = flatten_multiplication_factors(term)
    
        if type(co_kernel) == Multiplication:
            return factorise_with_multiple_factors(term_factors, flatten_multiplication_factors(co_kernel))
        else:
            remaining_factors = term_factors.copy() 
            remaining_factors, in_list = remove_factor_from_list(remaining_factors, co_kernel)

            if not in_list:
                return None
            
            return combine_factors(remaining_factors)
    
    return None


    


def factorise_with_multiple_factors(term_factors, cokernel_factors):
    # Make a copy of the variable so not to interfere with it
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

            # Remove the factored term and add the new one
            factors.pop(i)
            factors.insert(i, new_factor)
            return factors, True
        
        # Processing factorisation of power with variable or function
        if (type(factor) == Power and type(term) in (Variable, Function) and 
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




def factor_sort(factor):
        if type(factor) == Constant:
            return (0, factor.value, 0)
        
        elif type(factor) == Variable:
            return (1, factor.value, 0)
        
        elif type(factor) == Function:
            return (1, factor.name, 0) 
        
        elif type(factor) == Power:
            if type(factor.left) == Variable and type(factor.right) == Constant:
                return (2, factor.left.value, factor.right.value)
            
            elif type(factor.left) == Function and type(factor.right) == Constant:
                return (2, factor.left.name, factor.right.value)
            
           
        else:
            return (3, str(factor), 0)


def equal_terms(term1, term2):
    if type(term1) != type(term2):
        return False
    
    if type(term1) == Variable:
        return term1.value == term2.value
    
    elif type(term1) == Function:
        return term1.name == term2.name
    
    elif type(term1) == Constant:
        return (abs(term1.value - term2.value) < 1e-10)
    
    elif type(term1) == Multiplication:

        # Need to unroll the nested multiplications to be able to compare terms
        factors1 = flatten_multiplication_factors(term1)
        factors2 = flatten_multiplication_factors(term2)
        
        if len(factors1) != len(factors2):
            return False
        
        # Need to give terms a consistent order to be able to compare
        norm_f1 = sorted(factors1, key=factor_sort)
        norm_f2 = sorted(factors2, key=factor_sort)

        equal_functions = all(equal_terms(f1, f2) for f1, f2 in zip(norm_f1, norm_f2))
        
        return equal_functions
    
    elif type(term1) in (Addition, Subtraction, Power):

        left_side_equals = equal_terms(term1.left, term2.left)
        right_side_equals = equal_terms(term1.right, term2.right)

        return (left_side_equals and right_side_equals)
    
    elif type(term1) == UnaryMinus:
        return equal_terms(term1.value, term2.value)
    
    return False



# Function to combine the multiplication factors
def combine_factors(factors):
    if len(factors) == 0:
        return Constant(1.0)
    elif len(factors) == 1:
        return factors[0]
    else:
        result = factors[0]
        for factor in factors[1:]:
            result = Multiplication(result, factor)
        return result
    


# Makes use of recursion to add each term of the multiplication into a list
def flatten_multiplication_factors(expr):
    if type(expr) != Multiplication:
        return [expr]
    
    factors = []
    factors.extend(flatten_multiplication_factors(expr.left))
    factors.extend(flatten_multiplication_factors(expr.right))

    return factors





# Function to find the symbols in an expression with a parser
def find_symbols(expression, node=None, symbols=None):
    if symbols is None:
        symbols = set()

    if node is None:
        node = expression

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

    # Use a lambda function to order the symbols in a conssistent structure, based on class name and variable name
    all_symbols = sorted(list(find_symbols(expression_tree)), key=lambda x: (type(x).__name__, str(x)))
    
    matrix_rows = []
    for i, original_term in enumerate(full_terms):
        symbols_dict = {}
        term_sign = 1
        coefficient = 1
        
        
        if type(original_term) == UnaryMinus:
            term_sign = -1
            inner_term = original_term.value
        else:
            inner_term = original_term

        
        if type(inner_term) == Constant:
            coefficient = inner_term.value
        else:
            if type(inner_term) == Multiplication:
                factors = flatten_multiplication_factors(inner_term)

                # Extracts coefficient for mnested multiplication
                coeff_factors = [f for f in factors if type(f) == Constant]
                if coeff_factors:
                    coefficient = coeff_factors[0].value

            # Tries to factor out each symbol from the term
            for symbol in all_symbols:
                power = 0
                current_term = inner_term
                
                while True:
                    factored_result = try_factorising(current_term, symbol)
                    if factored_result is not None:
                        power += 1
                        current_term = factored_result
                    else:
                        break
                
                if power > 0:
                    symbols_dict[symbol] = power
        


        # Starts building the dictionary representing a row in the matrix
        row = {'term_id': i}
        
        for symbol in all_symbols:
            if symbol in symbols_dict:
                row[symbol] = symbols_dict[symbol]
            else:
                row[symbol] = 0

        row['coefficient'] = abs(coefficient)

        if coefficient >= 0:
            row['sign'] = term_sign
        else:
            row['sign'] = -term_sign
        row['original_term'] = original_term

        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    return matrix_df, all_symbols


# Looks for the largest common cube in a matrix row (reprsenting a term)
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


# Divides each row by cube. Does this by finding a variable with the same base and subtracting the exponent by 1
def divide_rows_by_cube(matrix_rows, cube_divisor, variables):

    row_divided = []

    for row in matrix_rows:

        # Copying the data is CRITICAL in order to preserve the information of the original expression
        # By creating a copy the correct mapping can be made between the kernel and the original, unfactorised expression
        divided_row = row.copy()

        # Cube divisor contains all variables as keys and their powers
        for var in list(cube_divisor.keys()):

            # print(divided_row, type(var), type(list(cube_divisor.keys())[0]))

            # Divides by subtracting powers
            if var in cube_divisor:
                divided_row[var] = row[var] - cube_divisor[var]
            else:
                divided_row[var] = row[var]
            
            # print(divided_row)

        row_divided.append(divided_row)


    return row_divided


# Check it term is cube free
def cube_free(matrix_rows, variables):

    # If there is only one row it is automatically cube free because there is only one factor
    if len(matrix_rows) < 2:
        return True
    
    common_cube = largest_common_cube(matrix_rows, variables)

    if len(common_cube) == 0:
        return True
    else:
        return False


# Function to convert the cube which is adictionary into variable with the PolynomialTerm class
def cube_dict_to_term(cube_dict, symbols):
    # if not cube_dict:
    #     return Constant(1)
    
    factors = []
    sorted_symbols = sorted(symbols, key=lambda x: (type(x).__name__, str(x)))

    # Accumulates all the factors
    for symbol in sorted_symbols:
        if symbol in cube_dict and cube_dict[symbol] > 0:
            if cube_dict[symbol] == 1:
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




# Convert a single row into a combined term
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
        if len(symbol_factors) == 1:
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
        
    elif coefficient != 1 and symbol_factors:
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



# Converts multiple rows into an expression
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
            else:
                result = Addition(result, term)
        return result




def display_results(kernels_cokernels):
    print("Kernel-cokernel pairs found: ", len(kernels_cokernels))
    print("=" * 15)
    
    for i, item in enumerate(kernels_cokernels):
        print("Pair", i+1)
        print("Cokernel:", str(item['cokernel']))
        print("Kernel: ", str(item['kernel']))
        





# Extract kernels based on the approach described in the paper
# Recursively iterates through the possible symbols and checking if it can be divided
def extract_kernels_method(matrix_rows, symbols):

    kernels_cokernels = []
    
    for var_index, selected_symbol in enumerate(symbols):
        print("Analysing variable: ", selected_symbol)
        
        # Find all rows in the matrix (terms) that contain the selected variable
        terms_with_var = [row for row in matrix_rows if row[selected_symbol] > 0]
        
        # If there is only one row, it is not possible to extract a kernel
        if len(terms_with_var) < 2:
            continue
        
        print("Terms that contain the selected symbol: ", selected_symbol, terms_with_var)
        
        # From all these terms finds the largest possible cube
        common_cube = largest_common_cube(terms_with_var, symbols)
        

        if len(common_cube) == 0:
            continue
            
        print("Common cube found: ", common_cube)
        
        # if selected_symbol not in common_cube:
        #     print("Common cube doesn't contain symbol")
        #     continue
            

        # From the terms, divide out the common cube found
        divided_rows = divide_rows_by_cube(terms_with_var, common_cube, symbols)
        
        # if not cube_free(divided_rows, symbols):
        #     print("Result not cube free")
        #     continue
            
        # if len(divided_rows) < 2:
        #     print("Result has not enough terms")
        #     continue
            

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
            print(f"Recursively analysing kernel")

            sub_kernels = extract_kernels_method(divided_rows, symbols)
            
            
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
                print(f"    Recursive kernel: {combined_cokernel} * {sub_kernel['kernel']}")
    
    return kernels_cokernels



def find_all_kernels_cokernels_single_expression(expression_str):

    print("Expression: ", expression_str)
    print("=" * 15)
    
    # Parse the polynomial
    parser = PolynomialParser(expression_str)
    expression_tree = parser.expr_tree
    
    # Create the polynomial matrix
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
    
   
    # Extract kernel, co kernel pairs
    extracted_kernels = extract_kernels_method(matrix_rows, symbols)
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


# Finds the various symbols and the term boundaries of each expression
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




# Function to find kernel and cokernel pairs for more than just one expression
def find_all_kernels_cokernels_multi(expression_strs):
  
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
    # Boundaries are important in order to keep track of which factorised term belongs to which expression

    all_symbols, expression_boundaries = get_symbols_and_boundaries(expression_trees)
    print(f"Expression boundaries: {expression_boundaries}")
    print(f"All symbols found: {[str(s) for s in all_symbols]}")
    
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
                adjusted_row['coefficient'] = row['coefficient']
                adjusted_row['sign'] = row['sign']
                adjusted_row['original_term'] = row['original_term']
                
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



# Function to analyse multiple polynomials
def analyse_polynomial_multi(expression_strs):

    
    kernels_cokernels, symbols, expression_boundaries = find_all_kernels_cokernels_multi(expression_strs)
    
    
    
    print("Combined kernel-cokernel pairs")
    print("=" * 15)
    
    by_expression = {}
    for item in kernels_cokernels:
        expr_id = item['source_expression']
        
        if expr_id not in by_expression:
            by_expression[expr_id] = []
        by_expression[expr_id].append(item)

    for expr_id in sorted(by_expression.keys()):
        
        print("From Expression", expr_id, ":")
        
        for i, item in enumerate(by_expression[expr_id]):
            print("Pair", i+1, ":")
            print("Cokernel:", str(item['cokernel']))
            print("Kernel:", str(item['kernel']))
            print("Term IDs:", item['term_ids'])
        
    
    print("Kernel Cube Matrix (KCM)")

    kcm_matrix = create_kernel_cube_matrix_multi(kernels_cokernels, symbols, expression_boundaries)
    print(kcm_matrix)
    
    return kcm_matrix, kernels_cokernels, symbols, expression_boundaries




# Create the KCM based on multiple expressions
def create_kernel_cube_matrix_multi(kernels_cokernels, symbols, expression_boundaries):

    # Line to remove rows with co kernel of 1. It is excluded because a function is always factorisable by 1
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
                    row_data[cube_term] = f"1({','.join(map(str, term_ids))})"
                else:
                    row_data[cube_term] = 1
            else:
                row_data[cube_term] = 0
        
        kcm_matrix.append(row_data)
    
    return pd.DataFrame(kcm_matrix)




# Implementation of the Distill algorithm to find the best rectangle in the KCM
# The search for the best rectangle is not done exhaustively. Heuristics are used

def distill_algorithm(kcm_df, expression_strings=None, expression_boundaries=None, seed_num = 10):

    print("Started the distil algorithm")
    
    cube_columns = [col for col in kcm_df.columns if col != 'cokernel']
    if len(cube_columns) == 0:
        print("DISTILL: No cube columns found")
        return None
    

    
    # Convert KCM to a binary matrix
    binary_matrix, term_id_matrix = parse_kcm_to_binary(kcm_df, cube_columns)
    num_rows, num_cols = binary_matrix.shape
    
    
    # Show binary matrix or term id matrix
    print("Binary matrix:")
    for i in range(num_rows):
        cokernel = kcm_df.iloc[i]['cokernel']
        row_str = ' '.join([str(binary_matrix[i][j]) for j in range(num_cols)])
        print(f"  Row {i} ({cokernel}): [{row_str}]")
    

    # NEW: Multiple rectangle selection approach
    selected_rectangles = []
    working_binary_matrix = binary_matrix.copy()
    working_term_id_matrix = term_id_matrix.copy()
    
    rectangle_iteration = 0
    while True:
        rectangle_iteration += 1
        print(f"\n--- Rectangle selection iteration {rectangle_iteration} ---")
        
        # Generate seeds from current working matrix
        seeds = generate_distill_seeds(working_binary_matrix, seed_num)
        print("Generated seeds: ", seeds)
        
        if not seeds:
            print("No more seeds available")
            break
            
        # Following the seeds, take each selected seed and greedily expand the rectangle
        # Following the seeds, take each selected seed and greedily expand the rectangle
        possible_rectangles = []
        seen_rectangles = set()  

        for i, seed in enumerate(seeds):
            expanded_rect = expand_rectangle_greedily(seed, working_binary_matrix, kcm_df, cube_columns, working_term_id_matrix, expression_strings, expression_boundaries)
            
            if expanded_rect:
                # Create a unique signature for the rectangle
                rect_signature = (
                    tuple(sorted(expanded_rect['rows'])),
                    tuple(str(col) for col in expanded_rect['columns']),
                    tuple(sorted(expanded_rect['original_term_ids']))
                )
                
                # Only add if not seen before
                if rect_signature not in seen_rectangles:
                    seen_rectangles.add(rect_signature)
                    possible_rectangles.append(expanded_rect)
                # Select the best rectangle from this iteration
                best_rectangle = select_best_rectangle(possible_rectangles, kcm_df)
                
                if best_rectangle is None:
                    print("No best rectangle selected")
                    break
            
        print(f"Selected rectangle covering terms: {best_rectangle['original_term_ids']}")
        selected_rectangles.append(best_rectangle)
        
        # Remove covered terms from working matrices
        remove_covered_terms(working_binary_matrix, working_term_id_matrix, best_rectangle)
        
        # Show updated working matrix
        print("Updated working binary matrix:")
        for i in range(num_rows):
            cokernel = kcm_df.iloc[i]['cokernel']
            row_str = ' '.join([str(working_binary_matrix[i][j]) for j in range(num_cols)])
            print(f"  Row {i} ({cokernel}): [{row_str}]")
    
    print(f"\nTotal rectangles selected: {len(selected_rectangles)}")
    
    if not selected_rectangles:
        return None
    
    return selected_rectangles



# Convert the data frame to a binary list format for processing
# Keeps track in a seperate matrix the term ids
def parse_kcm_to_binary(kcm_df, cube_columns):

    binary_matrix = np.zeros((len(kcm_df), len(cube_columns)), dtype=int)
    term_id_matrix = {}
    
    
    for i in range(len(kcm_df)):
        cokernel = kcm_df.iloc[i]['cokernel']
        
        
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

              
                term_ids = [int(tid) for tid in term_ids_str.split(',')]
                term_id_matrix[(i, j)] = term_ids

    return binary_matrix, term_id_matrix


# I dentidy rows and columns that should be further processed
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
            linked_cols = [j for j in range(num_cols) if binary_matrix[i][j] == 1]
            seeds.append({
                'rows': [i],
                'cols': linked_cols,
                'activity': row_activity[i]
            })
            print("Row seed: ", i, linked_cols, row_activity[i])
    
    # Similar process with the columns
    for j in range(num_cols):
        if col_activity[j] > 1:
            linked_rows = [i for i in range(num_rows) if binary_matrix[i][j] == 1]
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



# Try to expand the rectangle greedily by making it cover more adjacent rows and columns
def expand_rectangle_greedily(seed, binary_matrix, kcm_df, cube_columns, term_id_matrix, expression_strings, expression_boundaries):

    num_rows, num_cols = binary_matrix.shape
    current_rows = seed['rows'][:]
    current_cols = seed['cols'][:]
    
    print(f"DEBUG: Expanding with initial rows={current_rows}, cols={current_cols}")
    
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
                if isinstance(term_ids, list):
                    current_term_ids.update(term_ids)
                else:
                    if term_ids:
                        current_term_ids.add(term_ids)
        
        # Tries to expand the rows by finding rows that have 1 in all the current columns
        for test_row in range(num_rows):
            if test_row not in current_rows:
                # Check if all current columns have 1s in this row
                if all(binary_matrix[test_row][col] == 1 for col in current_cols):
                    # NEW: Check if this row would add any already-covered terms
                    new_term_ids = set()
                    for col in current_cols:
                        term_ids = term_id_matrix.get((test_row, col), [])
                        if isinstance(term_ids, list):
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
                if all(binary_matrix[row][test_col] == 1 for row in current_rows):
                    # NEW: Check if this column would add any already-covered terms
                    new_term_ids = set()
                    for row in current_rows:
                        term_ids = term_id_matrix.get((row, test_col), [])
                        if isinstance(term_ids, list):
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
    

    print("Invalid rectangle")
    return None



# Selects the best valued rectangle. The paper uses a formula, here the cost is implemented
# by counting the number of DSP blocks needed to build the expression
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



# Builds the expression co kernel * (kernel) + remaining terms and calculate the number of DSP blocks
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
    

    # Construct the factored part of the expression: co kernel * kernel
    # Build the factored part of the expression
    if len(rect['rows']) == 1:
        # Single cokernel case - keep existing logic
        row_index = rect['rows'][0]
        cokernel = kcm_df.iloc[row_index]['cokernel']
        
        # The cubes that make up the kernel
        kernel_cubes = list(rect['columns'])
        
        # Combine the cubes
        if len(kernel_cubes) == 1:
            kernel = kernel_cubes[0]
        else:
            kernel = kernel_cubes[0]
            for cube in kernel_cubes[1:]:
                if type(cube) == UnaryMinus:
                    kernel = Subtraction(kernel, cube.value)
                else:
                    kernel = Addition(kernel, cube)
        
        # Create the multiplication
        if type(cokernel) == Constant and cokernel.value == 1:
            factored_part = kernel
        else:
            factored_part = Multiplication(cokernel, kernel)
        
        factored_parts = [factored_part]
    else:
        # Multiple cokernels - factor as (cokernel1 + cokernel2 + ...) * kernel
        cokernels = [kcm_df.iloc[row_idx]['cokernel'] for row_idx in rect['rows']]
        
        # Combine cokernels
        combined_cokernel = cokernels[0]
        for ck in cokernels[1:]:
            if type(ck) == UnaryMinus:
                combined_cokernel = Subtraction(combined_cokernel, ck.value)
            else:
                combined_cokernel = Addition(combined_cokernel, ck)
        
        # Build kernel from columns
        kernel_cubes = list(rect['columns'])
        if len(kernel_cubes) == 1:
            kernel = kernel_cubes[0]
        else:
            kernel = kernel_cubes[0]
            for cube in kernel_cubes[1:]:
                if type(cube) == UnaryMinus:
                    kernel = Subtraction(kernel, cube.value)
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
        print(f"Adding unfactored term: ", term)
    
    # Add factored terms
    for factored_terms in factored_parts:
        all_parts.append(factored_terms)
        print("Adding factored part: ", factored_terms)

    
    # Combine all parts into the complete expression
    
    if len(all_parts) == 1:
        complete_expression = all_parts[0]
    else:
        complete_expression = all_parts[0]
        for part in all_parts[1:]:
            if type(part) == UnaryMinus:
                complete_expression = Subtraction(complete_expression, part.value)
            else:
                complete_expression = Addition(complete_expression, part)
    
    
    # Store the reconstructed expression since it has been built
    rect['reconstructed_expression'] = complete_expression

    print("FULL EXPRESSION: ", complete_expression)

    cost = DSPBlockNumber(str(complete_expression))
    print("DSP Block Number Cost: ", cost)
    
    return cost




# Convert a rectangle with its expression into a function
def create_function_from_rectangle(rect, kcm_df, function_name):    
   

    # The function value is the sum of its cubes that make the kernel
    function_expression = None  # Change from [] to None

    for cube_name in rect['columns']:
        if function_expression is None:  # Change condition
            function_expression = cube_name
        else:
            print(cube_name, type(cube_name))
            if type(cube_name) == UnaryMinus:
                function_expression = Subtraction(function_expression, cube_name.value)
            elif type(cube_name) == Constant and cube_name.value < 0:
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
        'optimisation_value': rect['cost'],
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



# Update the expressions to express them in terms of other functions
def update_expressions_with_extracted_function(all_expressions, analysable_expressions, extracted_function, best_rect, expression_boundaries):
    

    # Shows which expression the rectangle came from, i.e. the expression that can be expressed in terms of the function
    source_expr_id = best_rect['source_expression_id']
   

    expr_to_modify = analysable_expressions[source_expr_id]
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
    


    # Need to identify which terms in the current expression were covered by the rectangle
    # This is done by checking which terms can be factored by the co kernel

    cokernel = best_rect['cokernels'][0]


    actually_factored_terms = []
    actually_factored_indices = []

    for i, term in enumerate(all_terms):

        factored_result = try_factorising(term, cokernel)
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
        new_parts.append(term)  # Keep as PolynomialTerm objects

    # Add factored part
    cokernels = best_rect['cokernels']
    if len(cokernels) == 1:
        # Single cokernel case
        cokernel = cokernels[0]
        function_term = Function(extracted_function.name)
        
        if type(cokernel) == UnaryMinus:
            multiplication = Multiplication(cokernel.value, function_term)
            factored_part = UnaryMinus(multiplication)
        else:
            factored_part = Multiplication(cokernel, function_term)
        
        new_parts.append(factored_part)
    else:
        # Multiple cokernels - combine them
        combined_cokernel = cokernels[0]
        for ck in cokernels[1:]:
            if type(ck) == UnaryMinus:
                combined_cokernel = Subtraction(combined_cokernel, ck.value)
            else:
                combined_cokernel = Addition(combined_cokernel, ck)
        
        function_term = Function(extracted_function.name)
        factored_part = Multiplication(combined_cokernel, function_term)
        new_parts.append(factored_part)

    # Build the new expression
    # new_parts = []

    # # Add unfactored terms
    # for term in unfactored_terms:
    #     new_parts.append(term)  # Keep as PolynomialTerm objects

    # # Add factored part - handle multiple co-kernels
    # cokernels = best_rect['cokernels']
    # if len(cokernels) == 1:
    #     # Single co-kernel case
    #     cokernel = cokernels[0]
    #     function_term = Function(extracted_function.name)
        
    #     if type(cokernel) == UnaryMinus:
    #         multiplication = Multiplication(cokernel.value, function_term)
    #         factored_part = UnaryMinus(multiplication)
    #     else:
    #         factored_part = Multiplication(cokernel, function_term)
        
    #     new_parts.append(factored_part)
        
    # elif len(cokernels) > 1:
    #     # Multiple co-kernels - factor out the function: (cokernel1 + cokernel2 + ...) * function
    #     function_term = Function(extracted_function.name)
        
    #     # Combine all co-kernels
    #     combined_cokernel = cokernels[0]
    #     for ck in cokernels[1:]:
    #         if type(ck) == UnaryMinus:
    #             combined_cokernel = Subtraction(combined_cokernel, ck.value)
    #         else:
    #             combined_cokernel = Addition(combined_cokernel, ck)
        
    #     # Create: (combined_cokernel) * function
    #     factored_part = Multiplication(combined_cokernel, function_term)
    #     new_parts.append(factored_part)



    # Combine parts 
    
    if len(new_parts) == 1:
        new_expr = new_parts[0]
    else:
        new_expr = new_parts[0]
        for part in new_parts[1:]:
            if type(part) == UnaryMinus:
                new_expr = Subtraction(new_expr, part.value)
            else:
                new_expr = Addition(new_expr, part)

    new_expr_string = str(new_expr)
        
    
    # Update the correct expression (main one or function)
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
            if binary_matrix[i][j] == 1:
                cell_term_ids = term_id_matrix.get((i, j), [])
                
                if isinstance(cell_term_ids, list):
                    # Remove covered term IDs
                    remaining_term_ids = [tid for tid in cell_term_ids if tid not in covered_term_ids]
                    
                    if not remaining_term_ids:
                        binary_matrix[i][j] = 0  # No terms left, remove cell
                        term_id_matrix[(i, j)] = []
                    else:
                        term_id_matrix[(i, j)] = remaining_term_ids



def greedy_kernel_intersection_algorithm(expression_strings, seed_num):

    print("Greedy Kernel Intersection Algorithm with Distill")
    print("=" * 60)
    
    current_expressions = expression_strings.copy()
    iteration = 0
    extracted_functions = [] 
    
    # Continues until no better rectangle can be found
    while True:
        iteration += 1
        
        print("Iteration: ", iteration)
        print("Current expressions: ", current_expressions)
        
        # Expressions can either have an equal sign because they are functions or they do not because it is the intial input expression
        analysable_expressions = []
        for expr in current_expressions:

            if '=' in expr:

                # The expression is a function
                func_name, function_expr = expr.split('=', 1)
                function_expr = function_expr.strip()
                
                parser = PolynomialParser(function_expr)
                terms = extract_terms(parser.expr_tree)
                if len(terms) > 1: 
                    analysable_expressions.append(function_expr)
                    print("Will analyse function: ", function_expr)
                

            else:
                # This is the main input expression
                parser = PolynomialParser(expr)
                terms = extract_terms(parser.expr_tree)
                analysable_expressions.append(expr)
                
    
        
        kcm_matrix, kernels_cokernels, variables, expression_boundaries = analyse_polynomial_multi(analysable_expressions)
      
        
        print("Variables: ", [str(v) for v in variables])
        
        # if KCM is too small there is nothing more than can be cimplified
        if len(kcm_matrix) == 0 or len([col for col in kcm_matrix.columns if col != 'cokernel']) < 2:
            print("KCM is too small to continue")
            break
        

        # Find all best rectangles from this KCM
        selected_rectangles = distill_algorithm(kcm_matrix, analysable_expressions, expression_boundaries, seed_num)

        if not selected_rectangles:
            print("No rectangles found")
            break

        print(f"Found {len(selected_rectangles)} rectangles in this iteration")

        # Create all functions first
        created_functions = []
        for rect_index, best_rect in enumerate(selected_rectangles):
            function_name = "G" + str(iteration) + chr(ord('a') + rect_index)  # G1a, G1b, etc.
            extracted_function = create_function_from_rectangle(best_rect, kcm_matrix, function_name)
            extracted_functions.append(extracted_function)
            created_functions.append((extracted_function, best_rect))
            print(f"Created function {function_name}: {extracted_function.value}")

        # Update expressions with ALL rectangles at once
        current_expressions = update_expressions_with_all_functions(
            current_expressions, analysable_expressions, created_functions, expression_boundaries
        )

        print(" END OF LOOP functions:")
        for i, func in enumerate(current_expressions):
            print(func)
            

    
    return extracted_functions, current_expressions




def update_expressions_with_all_functions(all_expressions, analysable_expressions, created_functions, expression_boundaries):
    
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
        expr_to_modify = analysable_expressions[expr_id]
        print(f"Updating expression {expr_id} with {len(function_rect_pairs)} functions: {expr_to_modify}")
        
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
            
            # For multiple cokernels, we need to check if terms can be factored by ANY of them
            factored_indices = []
            for i, term in enumerate(all_terms):
                if i not in used_term_indices:
                    # Check if this term can be factored by any cokernel
                    for cokernel in cokernels:
                        factored_result = try_factorising(term, cokernel)
                        if factored_result is not None:
                            factored_indices.append(i)
                            break
            
            if factored_indices:
                # Mark these terms as used
                used_term_indices.update(factored_indices)
                
                # Create the factored part
                function_term = Function(extracted_function.name)
                
                if len(cokernels) == 1:
                    # Single cokernel
                    cokernel = cokernels[0]
                    if isinstance(cokernel, UnaryMinus):
                        multiplication = Multiplication(cokernel.value, function_term)
                        factored_part = UnaryMinus(multiplication)
                    else:
                        factored_part = Multiplication(cokernel, function_term)
                else:
                    # Multiple cokernels - combine them
                    combined_cokernel = cokernels[0]
                    for ck in cokernels[1:]:
                        if isinstance(ck, UnaryMinus):
                            combined_cokernel = Subtraction(combined_cokernel, ck.value)
                        else:
                            combined_cokernel = Addition(combined_cokernel, ck)
                    
                    factored_part = Multiplication(combined_cokernel, function_term)
                
                new_parts.append(factored_part)
                print(f"Factored terms {factored_indices} with ({' + '.join(str(c) for c in cokernels)}) * {extracted_function.name}")
        
        # Add remaining unfactored terms
        for i, term in enumerate(all_terms):
            if i not in used_term_indices:
                new_parts.append(term)
                print(f"Left unfactored: term {i} = {term}")
        
        # Build the new expression
        if len(new_parts) == 1:
            new_expr = new_parts[0]
        else:
            new_expr = new_parts[0]
            for part in new_parts[1:]:
                if isinstance(part, UnaryMinus):
                    new_expr = Subtraction(new_expr, part.value)
                else:
                    new_expr = Addition(new_expr, part)
        
        new_expr_string = str(new_expr)
        
        # Update the expression
        if is_function_definition:
            original_expr = updated_expressions[original_expr_index]
            func_name = original_expr.split('=')[0].strip()
            updated_expressions[original_expr_index] = f"{func_name} = {new_expr_string}"
        else:
            updated_expressions[original_expr_index] = new_expr_string
        
        print(f"Updated expression: {new_expr_string}")
    
    # Add all new function definitions
    for extracted_function, _ in created_functions:
        updated_expressions.append(f"{extracted_function.name} = {extracted_function.value}")
    
    return updated_expressions





def optimise_polynomial(expression_str, seed_num):
    print("COMPLETE POLYNOMIAL OPTIMISATION")
    print("=" * 15)
    print("Expression: ",  expression_str)
    
    extracted_functions, final_expressions = greedy_kernel_intersection_algorithm(
        expression_strings=[expression_str],
        seed_num = seed_num
    )
    
   
    print("Extracted functions:")
    for i, func in enumerate(extracted_functions):
        print(f"  {func.name}: {func.value}")
    
    print("Final expressions:")
    for i, expr in enumerate(final_expressions):
        print(f"  {i}: {expr}")
    
    return extracted_functions, final_expressions








def substitute_and_reconstruct(expression_list):

    # Parse all expressions
    parsed_expressions = {}
    main_expression = None
    
    for expr_str in expression_list:
        if '=' in expr_str:
            # Parse function definition
            func_name, func_body = expr_str.split('=', 1)
            parser = PolynomialParser(func_body.strip())
            parsed_expressions[func_name.strip()] = parser.expr_tree
        else:
            # Parse main expression
            parser = PolynomialParser(expr_str)
            main_expression = parser.expr_tree
    

    
    final_expression = substitute_functions(main_expression, parsed_expressions)
    
    return final_expression


def substitute_functions(expression, function_definitions):

    if type(expression) == Function:
       

        func_name = expression.name
        func_body = function_definitions[func_name]
        return substitute_functions(func_body, function_definitions)
        
    
    elif type(expression) == Variable:

        var_name = expression.value
        if var_name in function_definitions:
            func_body = function_definitions[var_name]
            return substitute_functions(func_body, function_definitions)
        
        else:

            return expression
    
    elif type(expression) in (Addition, Subtraction, Multiplication, Power):
        left_substituted = substitute_functions(expression.left, function_definitions)
        right_substituted = substitute_functions(expression.right, function_definitions)
        
        if type(expression) == Addition:
            return Addition(left_substituted, right_substituted)
        
        elif type(expression) == Subtraction:
            return Subtraction(left_substituted, right_substituted)
        
        elif type(expression) == Multiplication:
            return Multiplication(left_substituted, right_substituted)
        
        elif type(expression) == Power:
            return Power(left_substituted, right_substituted)
        
    
    elif type(expression) == UnaryMinus:
        substituted_inner = substitute_functions(expression.value, function_definitions)
        return UnaryMinus(substituted_inner)
    
    else:
        return expression


