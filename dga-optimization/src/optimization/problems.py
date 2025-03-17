import numpy as np
from typing import List, Tuple, Optional

class OptimizationProblem:
    """Base class for optimization problems."""
    def __init__(self, population_size=100, generations=100):
        self.population_size = population_size
        self.generations = generations
        self.name = "Generic Problem"
        self.num_objectives = None
        self.num_variables = None
    
    def evaluate(self, x: List[float]) -> List[float]:
        """Evaluate the objective function for a solution."""
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def solve(self) -> List[List[float]]:
        """Solve the problem and return the Pareto front."""
        raise NotImplementedError("Subclasses must implement solve()")

class ZDT1Problem(OptimizationProblem):
    """ZDT1 problem implementation."""
    def __init__(self, population_size=100, generations=100, num_variables=30):
        super().__init__(population_size, generations)
        self.name = "ZDT1"
        self.num_objectives = 2
        self.num_variables = num_variables
    
    def evaluate(self, x: List[float]) -> List[float]:
        """Evaluate ZDT1 objective functions."""
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return [f1, f2]
    
    def solve(self) -> List[List[float]]:
        """Solve the ZDT1 problem."""
        from ..dga import DGA
        
        # Define objective function wrapper
        class ZDT1Wrapper:
            def __init__(self):
                self.num_objectives = 2
            
            def evaluate(self, x):
                f1 = x[0]
                g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
                f2 = g * (1 - np.sqrt(f1 / g))
                return [f1, f2]
        
        problem = ZDT1Wrapper()
        
        # Create and run the DGA
        dga = DGA(
            problem=problem,
            population_size=self.population_size,
            generations=self.generations,
            chromosome_length=self.num_variables,
            algorithm_type="island",  # Use island model as it performs better
            num_islands=5,
            migration_interval=5,
            migration_rate=0.1
        )
        
        pareto_front = dga.run()
        
        # Extract objectives from individuals
        return [ind.objectives for ind in pareto_front]

class ZDT2Problem(ZDT1Problem):
    """ZDT2 problem implementation."""
    def __init__(self, population_size=100, generations=100, num_variables=30):
        super().__init__(population_size, generations, num_variables)
        self.name = "ZDT2"
    
    def evaluate(self, x: List[float]) -> List[float]:
        """Evaluate ZDT2 objective functions."""
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
        f2 = g * (1 - (f1 / g)**2)
        return [f1, f2]
    
    def solve(self) -> List[List[float]]:
        """Solve the ZDT2 problem."""
        from ..dga import DGA
        
        # Define objective function wrapper
        class ZDT2Wrapper:
            def __init__(self):
                self.num_objectives = 2
            
            def evaluate(self, x):
                f1 = x[0]
                g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
                f2 = g * (1 - (f1 / g)**2)
                return [f1, f2]
        
        problem = ZDT2Wrapper()
        
        # Create and run the DGA
        dga = DGA(
            problem=problem,
            population_size=self.population_size,
            generations=self.generations,
            chromosome_length=self.num_variables,
            algorithm_type="island",  # Use island model as it performs better
            num_islands=5,
            migration_interval=5,
            migration_rate=0.1
        )
        
        pareto_front = dga.run()
        
        # Extract objectives from individuals
        return [ind.objectives for ind in pareto_front]

class DTLZ2Problem(OptimizationProblem):
    """DTLZ2 problem implementation."""
    def __init__(self, population_size=100, generations=100, num_variables=12, num_objectives=3):
        super().__init__(population_size, generations)
        self.name = "DTLZ2"
        self.num_variables = num_variables
        self.num_objectives = num_objectives
        self.k = num_variables - num_objectives + 1
    
    def evaluate(self, x: List[float]) -> List[float]:
        """Evaluate DTLZ2 objective functions."""
        # Common terms
        g = sum((xi - 0.5)**2 for xi in x[self.num_objectives-1:])
        
        # Initialize objectives
        f = [0.0] * self.num_objectives
        
        # Calculate objectives
        for i in range(self.num_objectives):
            product = 1.0
            for j in range(self.num_objectives - i - 1):
                product *= np.cos(x[j] * np.pi / 2)
            
            if i > 0:
                product *= np.sin(x[self.num_objectives - i - 1] * np.pi / 2)
            
            f[i] = (1 + g) * product
        
        return f
    
    def solve(self) -> List[List[float]]:
        """Solve the DTLZ2 problem."""
        from ..dga import DGA
        
        # Define objective function wrapper for DTLZ2
        class DTLZ2Wrapper:
            def __init__(self, num_objectives, num_variables):
                self.num_objectives = num_objectives
                self.num_variables = num_variables
                self.k = num_variables - num_objectives + 1
            
            def evaluate(self, x):
                # Common terms
                g = sum((xi - 0.5)**2 for xi in x[self.num_objectives-1:])
                
                # Initialize objectives
                f = [0.0] * self.num_objectives
                
                # Calculate objectives
                for i in range(self.num_objectives):
                    product = 1.0
                    for j in range(self.num_objectives - i - 1):
                        product *= np.cos(x[j] * np.pi / 2)
                    
                    if i > 0:
                        product *= np.sin(x[self.num_objectives - i - 1] * np.pi / 2)
                    
                    f[i] = (1 + g) * product
                
                return f
        
        problem = DTLZ2Wrapper(self.num_objectives, self.num_variables)
        
        # Create and run the DGA with more generations for this complex problem
        dga = DGA(
            problem=problem,
            population_size=self.population_size,
            generations=self.generations * 2,  # Double the generations for DTLZ2
            chromosome_length=self.num_variables,
            algorithm_type="island",
            num_islands=5,
            migration_interval=5,
            migration_rate=0.1
        )
        
        pareto_front = dga.run()
        
        # Extract objectives from individuals
        return [ind.objectives for ind in pareto_front]

class SchafferF6Problem(OptimizationProblem):
    """Schaffer F6 problem implementation."""
    def __init__(self, population_size=100, generations=100):
        super().__init__(population_size, generations)
        self.name = "Schaffer F6"
        self.num_objectives = 2
        self.num_variables = 1
    
    def evaluate(self, x: List[float]) -> List[float]:
        """Evaluate Schaffer F6 objective functions."""
        f1 = x[0]**2
        f2 = (x[0] - 2)**2
        return [f1, f2]
    
    def solve(self) -> List[List[float]]:
        """Solve the Schaffer F6 problem."""
        from ..dga import DGA
        
        # Define objective function wrapper
        class SchafferF6Wrapper:
            def __init__(self):
                self.num_objectives = 2
            
            def evaluate(self, x):
                # We only use the first element of x
                value = x[0] * 4 - 2  # Scale from [0,1] to [-2,2]
                f1 = value**2
                f2 = (value - 2)**2
                return [f1, f2]
        
        problem = SchafferF6Wrapper()
        
        # Create and run the DGA
        dga = DGA(
            problem=problem,
            population_size=self.population_size,
            generations=self.generations,
            chromosome_length=1,  # Only one variable
            algorithm_type="island",
            num_islands=5,
            migration_interval=5,
            migration_rate=0.1
        )
        
        pareto_front = dga.run()
        
        # Extract objectives from individuals
        return [ind.objectives for ind in pareto_front]