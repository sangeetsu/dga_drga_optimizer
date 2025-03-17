from .individual import Individual
from .base import GeneticAlgorithmBase
from .island_model import IslandModel
from .drga import DividedRangeGA
from .diversity import fitness_sharing, clearing

class DGA:
    """Main interface for the DGA algorithm."""
    
    def __init__(self, problem, population_size=100, generations=100,
                 algorithm_type='standard', num_islands=5, migration_interval=5,
                 migration_rate=0.1, migration_topology='ring', chromosome_length=10):
        """
        Initialize the DGA algorithm.
        
        Args:
            problem: Problem instance
            population_size: Size of the population
            generations: Number of generations
            algorithm_type: Type of algorithm ('standard', 'island', 'drga')
            num_islands: Number of islands for island model or DRGA
            migration_interval: Interval for migration
            migration_rate: Rate of migration
            migration_topology: Topology for migration ('ring', 'fully_connected')
            chromosome_length: Length of chromosome (number of variables)
        """
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.algorithm_type = algorithm_type
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.migration_topology = migration_topology
        self.chromosome_length = chromosome_length
        
        # Initialize algorithm based on type
        if algorithm_type == 'standard':
            self.algorithm = GeneticAlgorithmBase(
                problem=problem,
                population_size=population_size,
                chromosome_length=chromosome_length,
                objectives=problem
            )
        elif algorithm_type == 'drga':
            self.algorithm = DividedRangeGA(
                total_population_size=population_size,
                chromosome_length=chromosome_length,
                mutation_rate=0.1,
                crossover_rate=0.9,
                objectives=problem,
                num_subpopulations=num_islands,
                migration_interval=migration_interval,
                migration_rate=migration_rate
            )
        elif algorithm_type == 'island':
            # Use DRGA for now as a simple island model
            # Future: implement a dedicated island model class
            self.algorithm = DividedRangeGA(
                total_population_size=population_size,
                chromosome_length=chromosome_length,
                mutation_rate=0.1,
                crossover_rate=0.9,
                objectives=problem,
                num_subpopulations=num_islands,
                migration_interval=migration_interval,
                migration_rate=migration_rate
            )
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    def run(self):
        """
        Run the algorithm for the specified number of generations.
        
        Returns:
            Final population
        """
        if self.algorithm_type == 'standard':
            return self.algorithm.evolve(self.generations)
        else:
            return self.algorithm.run(self.generations)