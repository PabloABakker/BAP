
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
