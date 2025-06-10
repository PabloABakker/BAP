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
        if type(other) != PolynomialTerm:
            return False
        return str(self) == str(other)  

class Variable(PolynomialTerm):
    def __init__(self, value):
        self.value = value
        
    def is_symbol(self):
        return True
    
    def __pow__(self, exponent):
        if type(exponent) == Constant:
            return Power(self, exponent)
        elif type(exponent) in (int, float):
            return Power(self, Constant(exponent))
    
    def __repr__(self):
        return f"Variable('{self.value}')"
        
    def __hash__(self):
        return hash(self.value)
    
    def __eq__(self, other):
        if type(other) != Variable:
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
    
    def __repr__(self):
        return f"Constant({self.value})"
    
    def __eq__(self, second):
        if type(second) == Constant:
            return abs(self.value - second.value) < 1e-10 
        return abs(self.value - second) < 1e-10
        
    @property
    def is_integer(self):
        return self.value == int(self.value)
        
    def __gt__(self, second):
        if type(second) == Constant:
            return self.value > second.value
        return self.value > second
        
    def __hash__(self):
        return hash(self.value)
    
    def __mod__(self, other):
        if type(other) == Constant:
            return Constant(self.value % other.value)
        return Constant(self.value % other)

    def __rmod__(self, other):
        return Constant(other % self.value)
    
    def __sub__(self, other):
        if type(other) == Constant:
            return Constant(self.value - other.value)
        return Constant(self.value - other)

    def __rsub__(self, other):
        return Constant(other - self.value)
    
    def __pow__(self, exponent):
        if type(exponent) == Constant:
            return Constant(self.value ** exponent.value)
        return Constant(self.value ** exponent)
    
    def __floordiv__(self, other):
        if type(other) == Constant:
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
    
    def __repr__(self):
        return f"Addition({repr(self.left)}, {repr(self.right)})"
    

class Subtraction(BinaryOperation):
    def __str__(self):
        return f"{self.left} - {self.right}"
    
    def __repr__(self):
        return f"Subtraction({repr(self.left)}, {repr(self.right)})"
    

class Multiplication(BinaryOperation):
    def __str__(self):
        # Adds parentheses around additions and subtractions
        left_str = str(self.left)
        right_str = str(self.right)

        if type(self.left) in (Addition, Subtraction):
            left_str = f"({left_str})"
            
        if type(self.right) in (Addition, Subtraction):
            right_str = f"({right_str})"
            
        return f"{left_str} * {right_str}"
    
    def __repr__(self):
        return f"Multiplication({repr(self.left)}, {repr(self.right)})"
    

class Power(BinaryOperation):
    def __str__(self):
        # Adds parentheses around the base for clarity
        left_str = str(self.left)
        if type(self.left) in (Addition, Subtraction, Multiplication):
            left_str = f"({left_str})"
            
        return f"{left_str}^{self.right}"
    
    def __repr__(self):
        return f"Power({repr(self.left)}, {repr(self.right)})"
    

class UnaryMinus(PolynomialTerm):
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        term_str = str(self.value)
        if type(self.value) in (Addition, Subtraction, Multiplication):
            term_str = f"({term_str})"
        return f"-{term_str}"
    
    def __repr__(self):
        return f"UnaryMinus({repr(self.value)})"
    
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
        if self.current_char() == '-':
        # if self.current_char() == '-' and self.next_char() and not self.next_char().isdigit():
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

        if type(node) == Variable:
            variables.add(node)

        elif type(node) in (Addition, Subtraction, Multiplication, Power):
            self.find_variables(node.left, variables)
            self.find_variables(node.right, variables)
            
        elif type(node) == UnaryMinus:
            self.find_variables(node.value, variables)
    
        return variables
    
    def print_tree(self, node=None, indent=0, prefix=""):
        """Print the polynomial tree structure"""
        if node is None:
            node = self.expr_tree
        
        if node is None:
            return
        
        print("  " * indent + prefix + str(type(node).__name__) + f": {node}")
        
        if type(node) in (Addition, Subtraction, Multiplication, Power):
            self.print_tree(node.left, indent + 1, "├─ left: ")
            self.print_tree(node.right, indent + 1, "└─ right: ")
        elif type(node) == UnaryMinus:
            self.print_tree(node.value, indent + 1, "└─ value: ")
    
    def get_class_structure(self, node=None):
        """Return a string representation of the class structure"""
        if node is None:
            node = self.expr_tree
        
        if type(node) == Variable:
            return f"Variable('{node.value}')"
        elif type(node) == Constant:
            return f"Constant({node.value})"
        elif type(node) == Addition:
            return f"Addition({self.get_class_structure(node.left)}, {self.get_class_structure(node.right)})"
        elif type(node) == Subtraction:
            return f"Subtraction({self.get_class_structure(node.left)}, {self.get_class_structure(node.right)})"
        elif type(node) == Multiplication:
            return f"Multiplication({self.get_class_structure(node.left)}, {self.get_class_structure(node.right)})"
        elif type(node) == Power:
            return f"Power({self.get_class_structure(node.left)}, {self.get_class_structure(node.right)})"
        elif type(node) == UnaryMinus:
            return f"UnaryMinus({self.get_class_structure(node.value)})"
        else:
            return str(node)