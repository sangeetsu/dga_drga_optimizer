import numpy as np
from typing import List, Any

class ZDT1:
    """ZDT1 test problem."""
    
    def __init__(self, n=30):
        """
        Initialize ZDT1 problem.
        
        Args:
            n: Number of variables
        """
        self.num_variables = n
        self.num_objectives = 2
        self.name = "ZDT1"
    
    def evaluate(self, x: List[float]) -> List[float]:
        """
        Evaluate the ZDT1 function.
        
        Args:
            x: Solution vector
            
        Returns:
            Objective values
        """
        if not x or len(x) == 0:
            return [float('inf'), float('inf')]
            
        f1 = x[0]
        g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        
        return [f1, f2]

class ZDT2:
    """ZDT2 test problem."""
    
    def __init__(self, n=30):
        """
        Initialize ZDT2 problem.
        
        Args:
            n: Number of variables
        """
        self.num_variables = n
        self.num_objectives = 2
        self.name = "ZDT2"
    
    def evaluate(self, x: List[float]) -> List[float]:
        """
        Evaluate the ZDT2 function.
        
        Args:
            x: Solution vector
            
        Returns:
            Objective values
        """
        if not x or len(x) == 0:
            return [float('inf'), float('inf')]
            
        f1 = x[0]
        g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1) if len(x) > 1 else 1.0
        f2 = g * (1 - (f1 / g) ** 2)
        
        return [f1, f2]

class DTLZ2:
    """DTLZ2 multi-objective problem."""
    def __init__(self, num_objectives=3, num_variables=12):
        self.num_objectives = num_objectives
        self.num_variables = num_variables  # Common dimensionality for DTLZ2
        self.k = self.num_variables - self.num_objectives + 1
    
    def evaluate(self, x: List[float]) -> List[float]:
        """
        Evaluate DTLZ2 objective functions.
        
        Args:
            x: List of decision variables in [0,1]
            
        Returns:
            List of objective values
        """
        # Ensure x has enough elements
        if len(x) < self.num_variables:
            x = x + [0.5] * (self.num_variables - len(x))
        
        # Common term g
        g = sum((xi - 0.5)**2 for xi in x[self.num_objectives-1:])
        
        # Initialize objectives array
        f = []
        
        # Calculate objectives
        for i in range(self.num_objectives):
            product = 1.0
            
            # Product of cos terms
            for j in range(self.num_objectives - i - 1):
                product *= np.cos(x[j] * np.pi / 2)
            
            # Multiply by sin term if not the first objective
            if i > 0:
                product *= np.sin(x[self.num_objectives - i - 1] * np.pi / 2)
            
            # Add to objectives list
            f.append((1 + g) * product)
        
        return f

class SchafferF6:
    """Schaffer F6 bi-objective problem."""
    def __init__(self):
        self.num_objectives = 2
        self.num_variables = 1  # This is a 1D problem
    
    def evaluate(self, x: List[float]) -> List[float]:
        """
        Evaluate Schaffer F6 objective functions.
        Assumes x[0] is scaled from [0,1] to an appropriate range.
        
        Args:
            x: List with at least one decision variable in [0,1]
            
        Returns:
            List of objective values [f1, f2]
        """
        # Scale from [0,1] to [-2,2]
        value = x[0] * 4.0 - 2.0
        
        f1 = value**2
        f2 = (value - 2)**2
        return [f1, f2]