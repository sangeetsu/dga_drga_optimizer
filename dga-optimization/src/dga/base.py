import numpy as np
import random
from typing import List, Any, Optional, Union, Callable
from .individual import Individual

class GeneticAlgorithmBase:
    """Base class for genetic algorithm implementations."""
    
    def __init__(self, problem, population_size, chromosome_length, 
                 mutation_rate=0.1, crossover_rate=0.9, objectives=None):
        """
        Initialize the GA.
        
        Args:
            problem: Problem instance or objective function
            population_size: Size of the population
            chromosome_length: Length of the chromosome (number of variables)
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            objectives: Objective functions (if different from problem)
        """
        self.problem = problem
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.objectives = objectives if objectives is not None else problem
        
        # Initialize population with valid objectives
        self.population = self.initialize_population()
    
    def initialize_population(self) -> List[Individual]:
        """
        Initialize a random population.
        
        Returns:
            List of individuals
        """
        population = []
        
        for _ in range(self.population_size):
            # Create random genes
            genes = [random.random() for _ in range(self.chromosome_length)]
            
            # Create new individual
            ind = Individual(genes)
            
            # Evaluate the individual
            try:
                if hasattr(self.objectives, 'evaluate'):
                    ind.objectives = self.objectives.evaluate(ind.genes)
                elif callable(self.objectives):
                    ind.objectives = self.objectives(ind.genes)
                else:
                    # Default: use a simple sum as a placeholder objective
                    ind.objectives = [sum(ind.genes)]
                
                # Make sure objectives is a list, not a scalar
                if not isinstance(ind.objectives, (list, tuple, np.ndarray)):
                    ind.objectives = [ind.objectives]
                
                # Set fitness values (alias for objectives)
                ind.fitness_values = ind.objectives.copy()
                ind.evaluated = True
            except Exception as e:
                print(f"Error during initialization: {e}")
                # Set default objectives
                ind.objectives = [0.0, 0.0]
                ind.fitness_values = ind.objectives.copy()
                ind.evaluated = True
            
            population.append(ind)
        
        return population
    
    def selection(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """
        Select parents using tournament selection.
        
        Args:
            population: Population to select from
            num_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        parents = []
        
        for _ in range(num_parents):
            # Tournament selection
            tournament_size = min(3, len(population))
            tournament = random.sample(population, tournament_size)
            
            # Sort by sum of objectives (assuming minimization)
            tournament.sort(key=lambda ind: sum(ind.objectives))
            
            # Select the best
            parents.append(tournament[0])
        
        return parents
    
    def crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        Perform SBX crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring individuals
        """
        # Perform crossover with probability crossover_rate
        if random.random() > self.crossover_rate:
            return [parent1.copy(), parent2.copy()]
        
        # SBX crossover
        eta = 15  # Distribution index
        
        offspring1_genes = []
        offspring2_genes = []
        
        for i in range(self.chromosome_length):
            # Skip if parents are identical
            if abs(parent1.genes[i] - parent2.genes[i]) < 1e-10:
                offspring1_genes.append(parent1.genes[i])
                offspring2_genes.append(parent2.genes[i])
                continue
            
            # Ensure parent1 has the smaller value
            if parent1.genes[i] > parent2.genes[i]:
                parent1.genes[i], parent2.genes[i] = parent2.genes[i], parent1.genes[i]
            
            # Calculate beta
            if random.random() <= 0.5:
                beta = (2 * random.random()) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - random.random()))) ** (1 / (eta + 1))
            
            # Create offspring
            offspring1_genes.append(0.5 * ((1 + beta) * parent1.genes[i] + (1 - beta) * parent2.genes[i]))
            offspring2_genes.append(0.5 * ((1 - beta) * parent1.genes[i] + (1 + beta) * parent2.genes[i]))
        
        # Create offspring individuals
        offspring1 = Individual(offspring1_genes)
        offspring2 = Individual(offspring2_genes)
        
        return [offspring1, offspring2]
    
    def mutation(self, individual: Individual) -> Individual:
        """
        Perform polynomial mutation.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        eta = 20  # Distribution index
        
        for i in range(self.chromosome_length):
            # Mutate with probability mutation_rate
            if random.random() <= self.mutation_rate:
                # Calculate delta
                if random.random() <= 0.5:
                    delta = (2 * random.random()) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - random.random())) ** (1 / (eta + 1))
                
                # Mutate gene
                individual.genes[i] += delta * 0.1  # Range of mutation
                
                # Ensure gene is within [0, 1] range
                individual.genes[i] = max(0, min(1, individual.genes[i]))
        
        return individual
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Perform fast non-dominated sorting on the population.
        
        Args:
            population: Population to sort
            
        Returns:
            List of fronts, where each front is a list of individuals
        """
        fronts = []
        
        # Initialize attributes for each individual
        for ind in population:
            ind.domination_count = 0
            ind.dominated_solutions = []
        
        # Calculate domination counts
        for i in range(len(population)):
            for j in range(len(population)):
                if i == j:
                    continue
                
                # Check if i dominates j
                dominates = True
                has_better = False
                
                for k in range(len(population[i].objectives)):
                    if population[i].objectives[k] > population[j].objectives[k]:
                        dominates = False
                        break
                    elif population[i].objectives[k] < population[j].objectives[k]:
                        has_better = True
                
                if dominates and has_better:
                    population[i].dominated_solutions.append(population[j])
                elif not dominates and has_better:
                    population[i].domination_count += 1
        
        # Find the first front
        front = [ind for ind in population if ind.domination_count == 0]
        fronts.append(front)
        
        # Find subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            
            for ind in fronts[i]:
                for dominated in ind.dominated_solutions:
                    dominated.domination_count -= 1
                    
                    if dominated.domination_count == 0:
                        dominated.rank = i + 1
                        next_front.append(dominated)
            
            if not next_front:
                break
                
            fronts.append(next_front)
            i += 1
        
        return fronts
    
    def calculate_crowding_distance(self, front: List[Individual]) -> None:
        """
        Calculate crowding distance for individuals in a front.
        
        Args:
            front: List of individuals in a front
        """
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for ind in front:
            ind.crowding_distance = 0
        
        # Calculate crowding distance for each objective
        for obj_idx in range(len(front[0].objectives)):
            # Sort by the current objective
            front.sort(key=lambda ind: ind.objectives[obj_idx])
            
            # Set infinite distance for boundary points
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distance for intermediate points
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range == 0:
                continue
                
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    (front[i+1].objectives[obj_idx] - front[i-1].objectives[obj_idx]) / obj_range
                )
    
    def select_non_dominated(self, population: List[Individual], n: int) -> List[Individual]:
        """
        Select n non-dominated individuals from the population.
        
        Args:
            population: Population to select from
            n: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        if not population:
            return []
            
        # Sort by non-domination and crowding distance
        fronts = self.fast_non_dominated_sort(population)
        
        # Calculate crowding distance for each front
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Flatten fronts and sort by rank and crowding distance
        sorted_population = []
        for front in fronts:
            sorted_population.extend(front)
        
        sorted_population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))
        
        # Select top n individuals
        return sorted_population[:n]
    
    def evaluate_population(self, population: List[Individual]) -> None:
        """
        Evaluate all individuals in the population.
        
        Args:
            population: Population to evaluate
        """
        for i, ind in enumerate(population):
            if not ind.evaluated:
                try:
                    if hasattr(self.objectives, 'evaluate'):
                        ind.objectives = self.objectives.evaluate(ind.genes)
                    elif callable(self.objectives):
                        ind.objectives = self.objectives(ind.genes)
                    else:
                        # Default: use a simple sum as a placeholder objective
                        ind.objectives = [sum(ind.genes)]
                    
                    # Make sure objectives is a list, not a scalar
                    if not isinstance(ind.objectives, (list, tuple, np.ndarray)):
                        ind.objectives = [ind.objectives]
                    
                    ind.fitness_values = ind.objectives.copy()
                    ind.evaluated = True
                except Exception as e:
                    print(f"Evaluation error: {e}, creating random objectives")
                    # Generate random objectives as a fallback
                    if hasattr(self.problem, 'num_objectives'):
                        num_obj = self.problem.num_objectives
                    else:
                        num_obj = 2  # Default to 2 objectives
                    
                    ind.objectives = [random.random() for _ in range(num_obj)]
                    ind.fitness_values = ind.objectives.copy()
                    ind.evaluated = True
    
    def evolve(self, generations: int) -> List[Individual]:
        """
        Evolve the population for the specified number of generations.
        
        Args:
            generations: Number of generations to evolve
            
        Returns:
            Final population
        """
        for _ in range(generations):
            # Create offspring
            offspring = []
            
            while len(offspring) < self.population_size:
                # Select parents
                parents = self.selection(self.population, 2)
                
                # Perform crossover
                children = self.crossover(parents[0], parents[1])
                
                # Perform mutation
                for child in children:
                    self.mutation(child)
                    offspring.append(child)
            
            # Limit offspring size to population size
            if len(offspring) > self.population_size:
                offspring = offspring[:self.population_size]
            
            # Evaluate offspring
            self.evaluate_population(offspring)
            
            # Combine parent and offspring populations
            combined = self.population + offspring
            
            # Select the next generation using non-dominated sorting
            self.population = self.select_non_dominated(combined, self.population_size)
        
        return self.population