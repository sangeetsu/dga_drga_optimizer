import numpy as np
import random
from typing import List, Any, Optional

class Individual:
    """Represents an individual in the population."""
    
    def __init__(self, genes: List[float]):
        """
        Initialize an individual with specified genes.
        
        Args:
            genes: List of gene values representing the chromosome
        """
        self.genes = genes
        self.objectives = []  # Objective function values
        self.fitness_values = []  # Alias for objectives for backward compatibility
        self.evaluated = False  # Flag to track if the individual has been evaluated
        
        # For non-dominated sorting
        self.domination_count = 0
        self.dominated_solutions = []
        self.rank = 0
        self.crowding_distance = 0.0
    
    def __str__(self) -> str:
        """String representation of the individual."""
        return f"Individual(genes={self.genes[:3]}..., objectives={self.objectives})"
    
    def copy(self) -> 'Individual':
        """Create a deep copy of this individual."""
        new_ind = Individual(self.genes.copy())
        if self.evaluated:
            new_ind.objectives = self.objectives.copy()
            new_ind.fitness_values = self.objectives.copy()
            new_ind.evaluated = True
        return new_ind