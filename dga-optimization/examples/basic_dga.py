# This file provides an example implementation of a basic DGA.

import random
import numpy as np
from src.dga.base import GeneticAlgorithmBase
from src.optimization.objectives import ZDT1
from src.dga.migration import migrate_population
from src.dga.individual import Individual

class BasicDGA(GeneticAlgorithmBase):
    def __init__(self, population_size=100, chromosome_length=30, 
                 mutation_rate=0.1, crossover_rate=0.8, generations=50):
        # Create an objectives object that implements evaluate method
        class ObjectivesWrapper:
            def __init__(self):
                self.zdt1 = ZDT1()
                
            def evaluate(self, genes):
                return self.zdt1.evaluate(genes)
                
        objectives = ObjectivesWrapper()
        super().__init__(population_size, chromosome_length, mutation_rate, 
                         crossover_rate, objectives)
        self.generations = generations
        self.elitism_size = 10

    def run(self):
        """Run the DGA algorithm for the specified number of generations."""
        for generation in range(self.generations):
            # Evaluate the current population
            self.evaluate_population()
            
            # Create new population through selection, crossover and mutation
            new_population = []
            
            # Elitism - keep best individuals
            elites = self.select_non_dominated(self.population, self.elitism_size)
            new_population.extend(elites)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents()
                
                # Apply crossover and mutation
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Apply migration strategy
            new_population = migrate_population(new_population, migration_rate=0.1, strategy="random")
            
            # Update population
            self.population = new_population
            self.generation += 1
            
            # Log progress
            self.log_generation(generation)
            
        return self.population
            
    def log_generation(self, generation):
        """Log information about the current generation."""
        best_objectives = [ind.objectives for ind in self.select_non_dominated(self.population, 5)]
        print(f"Generation {generation}: Found {len(best_objectives)} Pareto optimal solutions")
        if best_objectives:
            print(f"Sample solution: {best_objectives[0]}")

if __name__ == "__main__":
    dga = BasicDGA(population_size=100, generations=50)
    final_population = dga.run()
    print("Optimization complete")
    pareto_front = [ind.objectives for ind in dga.select_non_dominated(final_population, len(final_population))]
    print(f"Found {len(pareto_front)} solutions in the Pareto front")