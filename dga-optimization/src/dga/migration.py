from typing import List, Any
import random
import numpy as np

class MigrationStrategy:
    def migrate(self, population: List[Any]) -> List[Any]:
        raise NotImplementedError("Migration strategy must implement the migrate method.")

class IslandModelMigration(MigrationStrategy):
    def __init__(self, migration_rate: float, topology: str, num_islands: int = 4):
        self.migration_rate = migration_rate
        self.topology = topology
        self.num_islands = num_islands

    def migrate(self, population: List[Any]) -> List[Any]:
        if self.topology == "ring":
            return self._ring_migration(population)
        elif self.topology == "grid":
            return self._grid_migration(population)
        else:
            raise ValueError("Unknown topology type.")

    def _ring_migration(self, population: List[Any]) -> List[Any]:
        # Divide population into islands
        island_size = len(population) // self.num_islands
        islands = [population[i:i+island_size] for i in range(0, len(population), island_size)]
        
        # Ensure we have the expected number of islands
        while len(islands) > self.num_islands:
            islands[-2].extend(islands[-1])
            islands.pop()
        
        num_migrants = max(1, int(island_size * self.migration_rate))
        
        # Perform ring migration (each island sends migrants to the next island)
        new_islands = []
        for i in range(self.num_islands):
            current_island = islands[i].copy()
            next_island_idx = (i + 1) % self.num_islands
            
            # Select migrants from current island
            migrants_indices = random.sample(range(len(current_island)), num_migrants)
            migrants = [current_island[idx] for idx in migrants_indices]
            
            # Remove migrants from current island
            current_island = [ind for i, ind in enumerate(current_island) if i not in migrants_indices]
            
            # Receive migrants from previous island
            prev_island_idx = (i - 1) % self.num_islands
            prev_migrants = islands[prev_island_idx][-num_migrants:]
            
            # Add received migrants to current island
            current_island.extend(prev_migrants)
            new_islands.append(current_island)
        
        # Flatten islands back into a single population
        new_population = []
        for island in new_islands:
            new_population.extend(island)
        
        return new_population

    def _grid_migration(self, population: List[Any]) -> List[Any]:
        # Assuming a square grid topology
        grid_size = int(np.sqrt(self.num_islands))
        if grid_size * grid_size != self.num_islands:
            raise ValueError("For grid topology, num_islands must be a perfect square")
        
        # Divide population into islands
        island_size = len(population) // self.num_islands
        islands = [population[i:i+island_size] for i in range(0, len(population), island_size)]
        
        # Ensure we have the expected number of islands
        while len(islands) > self.num_islands:
            islands[-2].extend(islands[-1])
            islands.pop()
        
        num_migrants = max(1, int(island_size * self.migration_rate))
        
        # Create a grid representation
        grid = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                row.append(islands[i * grid_size + j])
            grid.append(row)
        
        # Perform grid migration (each island exchanges with 4 neighbors: up, down, left, right)
        new_grid = [[island.copy() for island in row] for row in grid]
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Find neighbors
                neighbors = []
                if i > 0: neighbors.append((i-1, j))  # up
                if i < grid_size-1: neighbors.append((i+1, j))  # down
                if j > 0: neighbors.append((i, j-1))  # left
                if j < grid_size-1: neighbors.append((i, j+1))  # right
                
                # For each neighbor, exchange migrants
                migrants_per_neighbor = num_migrants // len(neighbors)
                for ni, nj in neighbors:
                    # Select migrants from current island
                    migrants_indices = random.sample(range(len(grid[i][j])), migrants_per_neighbor)
                    migrants = [grid[i][j][idx] for idx in migrants_indices]
                    
                    # Remove migrants from current island (for the next iteration)
                    current_island = [ind for k, ind in enumerate(grid[i][j]) if k not in migrants_indices]
                    grid[i][j] = current_island
                    
                    # Add migrants to neighbor
                    new_grid[ni][nj].extend(migrants)
        
        # Flatten grid back into a single population
        new_population = []
        for row in new_grid:
            for island in row:
                new_population.extend(island)
        
        return new_population

class MigrationPolicy:
    def apply_policy(self, population: List[Any]) -> List[Any]:
        raise NotImplementedError("Migration policy must implement the apply_policy method.")

class RandomMigrationPolicy(MigrationPolicy):
    def __init__(self, migration_rate: float = 0.1):
        self.migration_rate = migration_rate
    
    def apply_policy(self, population: List[Any]) -> List[Any]:
        num_individuals = len(population)
        num_migrants = int(num_individuals * self.migration_rate)
        
        # Select random individuals to migrate
        migrant_indices = random.sample(range(num_individuals), num_migrants)
        
        # Create a new population with migrated individuals shuffled
        new_population = population.copy()
        migrants = [new_population[i] for i in migrant_indices]
        random.shuffle(migrants)
        
        for idx, migrant in zip(migrant_indices, migrants):
            new_population[idx] = migrant
            
        return new_population

# Utility function for other files to use
def migrate_population(population: List[Any], migration_rate: float = 0.1, strategy: str = "random") -> List[Any]:
    if strategy == "random":
        policy = RandomMigrationPolicy(migration_rate)
        return policy.apply_policy(population)
    elif strategy == "ring":
        migration = IslandModelMigration(migration_rate, "ring")
        return migration.migrate(population)
    elif strategy == "grid":
        migration = IslandModelMigration(migration_rate, "grid")
        return migration.migrate(population)
    else:
        raise ValueError(f"Unknown migration strategy: {strategy}")