import random
from typing import List, Any, Optional
from .base import GeneticAlgorithmBase
from .individual import Individual

class IslandModel:
    """Implementation of the Island Model for distributed genetic algorithms."""
    
    def __init__(self, num_islands: int = 4, migration_interval: int = 10,
                 migration_rate: float = 0.1, topology: str = "ring",
                 selection_policy: str = "best", replacement_policy: str = "worst",
                 ga_class=GeneticAlgorithmBase, ga_params: dict = None):
        """
        Initialize the Island Model.
        
        Args:
            num_islands: Number of islands (subpopulations)
            migration_interval: Number of generations between migrations
            migration_rate: Percentage of population to migrate
            topology: Connection topology between islands ("ring", "fully_connected", "random")
            selection_policy: How to select migrants ("best", "random")
            replacement_policy: How to replace individuals with migrants ("worst", "random")
            ga_class: Genetic algorithm class to use for each island
            ga_params: Parameters for the genetic algorithm instances
        """
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.topology = topology
        self.selection_policy = selection_policy
        self.replacement_policy = replacement_policy
        
        # Default params for GA
        if ga_params is None:
            ga_params = {}
        
        # Create islands
        self.islands = []
        for i in range(num_islands):
            # Clone the parameters dictionary to avoid shared references
            island_params = ga_params.copy()
            self.islands.append(ga_class(**island_params))
        
        # Migration connections based on topology
        self.connections = self._create_connections()
        
        # Generation counter
        self.generation = 0
    
    def _create_connections(self) -> List[List[int]]:
        """Create connections between islands based on the specified topology."""
        connections = [[] for _ in range(self.num_islands)]
        
        if self.topology == "ring":
            for i in range(self.num_islands):
                connections[i].append((i + 1) % self.num_islands)
                connections[i].append((i - 1) % self.num_islands)
        
        elif self.topology == "fully_connected":
            for i in range(self.num_islands):
                for j in range(self.num_islands):
                    if i != j:
                        connections[i].append(j)
        
        elif self.topology == "random":
            # Each island connects to 2 random other islands
            for i in range(self.num_islands):
                possible_connections = [j for j in range(self.num_islands) if j != i]
                if possible_connections:
                    num_connections = min(2, len(possible_connections))
                    connections[i] = random.sample(possible_connections, num_connections)
        
        return connections
    
    def _select_migrants(self, island: GeneticAlgorithmBase) -> List[Individual]:
        """Select individuals to migrate from an island."""
        num_migrants = max(1, int(island.population_size * self.migration_rate))
        
        if self.selection_policy == "best":
            # Make sure the population is evaluated
            island.evaluate_population()
            
            # Sort population based on non-domination
            sorted_pop = island.fast_non_dominated_sort(island.population)
            return [ind.copy() for ind in sorted_pop[:num_migrants]]
        
        else:  # Random selection
            return [random.choice(island.population).copy() for _ in range(num_migrants)]
    
    def _replace_with_migrants(self, island: GeneticAlgorithmBase, migrants: List[Individual]) -> None:
        """Replace individuals in the island with migrants."""
        from .individual import Individual
        
        if not migrants:
            return
            
        if self.replacement_policy == "worst":
            # Make sure the population is evaluated
            island.evaluate_population()
            
            # Sort population by fitness (simple sum for multi-objective)
            sorted_indices = list(range(len(island.population)))
            sorted_indices.sort(key=lambda i: sum(island.population[i].objectives) if hasattr(island.population[i], 'objectives') and island.population[i].objectives else float('inf'))
            
            # Replace worst individuals with migrants
            for i, migrant in enumerate(migrants):
                if i < len(sorted_indices):
                    # Make a deep copy of the migrant
                    if isinstance(migrant, Individual):
                        new_migrant = Individual(migrant.genes.copy())
                        new_migrant.objectives = migrant.objectives.copy() if migrant.objectives else None
                        new_migrant.evaluated = migrant.evaluated
                    else:
                        # Handle case where migrant is a list
                        new_migrant = Individual(migrant)
                    
                    # Replace one of the worst individuals
                    island.population[sorted_indices[-i-1]] = new_migrant
        else:  # Random replacement
            for migrant in migrants:
                idx = random.randrange(len(island.population))
                
                # Make a deep copy of the migrant
                if isinstance(migrant, Individual):
                    new_migrant = Individual(migrant.genes.copy())
                    new_migrant.objectives = migrant.objectives.copy() if migrant.objectives else None
                    new_migrant.evaluated = migrant.evaluated
                else:
                    # Handle case where migrant is a list
                    new_migrant = Individual(migrant)
                    
                island.population[idx] = new_migrant
    
    def _perform_migration(self) -> None:
        """Perform migration between islands."""
        # Collect migrants from each island
        all_migrants = {}
        for i, island in enumerate(self.islands):
            migrants = self._select_migrants(island)
            all_migrants[i] = migrants
        
        # Distribute migrants
        for i, island in enumerate(self.islands):
            incoming_migrants = []
            for source in self.connections[i]:
                if source in all_migrants:
                    # Get a portion of the migrants from the source island
                    source_migrants = all_migrants[source]
                    # Take a random subset if there are many migrants
                    num_to_take = max(1, len(source_migrants) // len(self.connections[source]))
                    incoming_migrants.extend(random.sample(source_migrants, 
                                                         min(num_to_take, len(source_migrants))))
            
            # Replace individuals with migrants
            self._replace_with_migrants(island, incoming_migrants)
    
    def run(self, generations):
        """Run the island model GA for the specified number of generations."""
        total_pop = sum(len(island.population) for island in self.islands)
        
        for generation in range(generations):
            if generation % 25 == 0:  # Only print every 25 generations
                print(f"  Gen {generation}/{generations}: Islands = {len(self.islands)}, Total population = {total_pop}")
            
            # Evolve each island independently
            for island in self.islands:
                island.evolve()
            
            # Perform migration at specified intervals
            if generation > 0 and generation % self.migration_interval == 0:
                self._perform_migration()
        
        # Print final generation
        print(f"  Gen {generations}/{generations}: Islands = {len(self.islands)}, Total population = {total_pop} | Complete")
        
        # Combine populations from all islands
        combined_population = []
        for island in self.islands:
            combined_population.extend(island.population)
        
        return combined_population
    
    def _log_progress(self, generation: int) -> None:
        """Log current progress."""
        total_pop = sum(len(island.population) for island in self.islands)
        print(f"Generation {generation}: Islands = {self.num_islands}, Total population = {total_pop}")
    
    def _get_combined_pareto_front(self) -> List[Individual]:
        """Combine individuals from all islands and return the Pareto front."""
        # Combine populations
        combined_pop = []
        for island in self.islands:
            combined_pop.extend(island.population)
        
        # Get the non-dominated front from the first island (they all use the same algorithm)
        if self.islands:
            return self.islands[0].fast_non_dominated_sort(combined_pop)
        return []