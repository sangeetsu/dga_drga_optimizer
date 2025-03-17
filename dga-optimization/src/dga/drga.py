import numpy as np
import random
from .base import GeneticAlgorithmBase
from .individual import Individual

class DividedRangeGA:
    def __init__(self, total_population_size, chromosome_length, mutation_rate, crossover_rate, 
                 objectives, num_subpopulations=2, migration_interval=5, migration_rate=0.1):
        self.total_population_size = total_population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.objectives = objectives
        
        # Determine number of objectives from the objective function
        # Default to 2 if we can't determine
        if hasattr(objectives, 'num_objectives'):
            self.num_objectives = objectives.num_objectives
        else:
            # Get a sample evaluation to determine objective count
            sample_ind = [random.random() for _ in range(chromosome_length)]
            sample_obj = objectives.evaluate(sample_ind)
            self.num_objectives = len(sample_obj)
        
        # Ensure we have at least one objective
        if self.num_objectives == 0:
            self.num_objectives = 2  # Default to 2 objectives if detection fails
        
        # Limit subpopulations to the number of objectives
        self.num_subpopulations = min(num_subpopulations, max(1, self.num_objectives))
        
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        
        # Size of each subpopulation
        self.subpop_size = max(total_population_size // self.num_subpopulations, 10)
        
        # Initialize subpopulations
        self.subpopulations = []
        for i in range(self.num_subpopulations):
            # Create a GA instance for each subpopulation
            subpop = GeneticAlgorithmBase(
                problem=objectives,
                population_size=self.subpop_size,
                chromosome_length=int(chromosome_length),
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                objectives=objectives
            )
            self.subpopulations.append(subpop)
        
        # Initialize full population
        self.population = []
        for subpop in self.subpopulations:
            self.population.extend(subpop.population)
            
        # Evaluate all individuals
        self._evaluate_all_individuals()
        
        # Divide population by objective function ranges
        self.divide_population_by_ranges()
    
    def _ensure_individual(self, ind):
        """Ensure that ind is an Individual object."""
        from .individual import Individual
        
        if not isinstance(ind, Individual):
            # Convert list to Individual object
            if isinstance(ind, list):
                return Individual(ind)
            # Handle nested lists/individuals
            elif hasattr(ind, 'genes') and isinstance(ind.genes[0], Individual):
                return Individual([g.genes[0] if isinstance(g, Individual) else g for g in ind.genes])
        return ind

    def _evaluate_all_individuals(self):
        """Evaluate all individuals in the population."""
        for i, ind in enumerate(self.population):
            # Convert to Individual if needed
            ind = self._ensure_individual(ind)
            self.population[i] = ind
            
            # Now evaluate if needed
            if not hasattr(ind, 'objectives') or ind.objectives is None:
                try:
                    ind.objectives = self.objectives.evaluate(ind.genes)
                    ind.evaluated = True
                except Exception as e:
                    print(f"Error evaluating individual: {e}")
                    # Create some default objectives as fallback
                    ind.objectives = [1.0] * self.num_objectives
                    ind.evaluated = True
    
    def divide_population_by_ranges(self):
        """Divide the population based on objective function ranges."""
        # Make sure all individuals are evaluated first
        valid_population = []
        
        for i, ind in enumerate(self.population):
            # Ensure ind is an Individual
            ind = self._ensure_individual(ind)
            self.population[i] = ind
            
            if hasattr(ind, 'objectives') and ind.objectives and len(ind.objectives) >= 1:
                valid_population.append(ind.copy())
        
        # Rest of the method remains the same...
        
        # Guard against empty population
        if not self.population:
            return
        
        # Get number of objectives from first individual or use stored value
        objective_lengths = [len(ind.objectives) for ind in self.population 
                            if hasattr(ind, 'objectives') and ind.objectives]
        
        if not objective_lengths:
            num_objectives = self.num_objectives
        else:
            # Use the minimum objective length to ensure safety
            num_objectives = min(objective_lengths)
        
        # Additional safety check
        if num_objectives == 0:
            num_objectives = 2  # Default to 2 if we still don't have a valid value
            
        # Filter out individuals with invalid objectives
        valid_population = []
        for ind in self.population:
            if hasattr(ind, 'objectives') and ind.objectives and len(ind.objectives) >= 1:
                valid_population.append(ind.copy())
        
        # If we don't have enough valid individuals, create new ones
        if len(valid_population) < self.subpop_size * self.num_subpopulations:
            while len(valid_population) < self.subpop_size * self.num_subpopulations:
                new_ind = Individual([random.random() for _ in range(self.chromosome_length)])
                new_ind.objectives = self.objectives.evaluate(new_ind.genes)
                new_ind.evaluated = True
                valid_population.append(new_ind)
        
        # Create a new combined population
        combined_population = valid_population
        
        # Clear existing subpopulations
        for subpop in self.subpopulations:
            subpop.population = []
        
        # For each subpopulation, assign a portion of the population based on objective ranges
        for i in range(self.num_subpopulations):
            # Determine which objective this subpopulation focuses on
            obj_idx = i % max(1, num_objectives)  # Ensure we never modulo by zero
            
            # Sort the combined population by the current objective with a safety check
            combined_population.sort(
                key=lambda ind: ind.objectives[obj_idx] if obj_idx < len(ind.objectives) else float('inf')
            )
            
            # Divide the population evenly among subpopulations
            subpop_size = len(combined_population) // self.num_subpopulations
            start_idx = i * subpop_size
            end_idx = start_idx + subpop_size if i < self.num_subpopulations - 1 else len(combined_population)
            
            # Assign this segment to the current subpopulation
            self.subpopulations[i].population = [ind.copy() for ind in combined_population[start_idx:end_idx]]
            
            # Ensure the subpopulation has the required minimum size
            while len(self.subpopulations[i].population) < self.subpop_size:
                # Create new individuals if needed
                new_ind = Individual([random.random() for _ in range(self.chromosome_length)])
                new_ind.objectives = self.objectives.evaluate(new_ind.genes)
                new_ind.evaluated = True
                self.subpopulations[i].population.append(new_ind)
    
    def migrate(self):
        """Migrate individuals between subpopulations."""
        all_migrants = []
        
        for subpop in self.subpopulations:
            # Make sure the population is evaluated first
            for i, ind in enumerate(subpop.population):
                # Convert to Individual if needed
                ind = self._ensure_individual(ind)
                subpop.population[i] = ind
                
                # Evaluate if needed
                if not hasattr(ind, 'objectives') or ind.objectives is None:
                    try:
                        ind.objectives = self.objectives.evaluate(ind.genes)
                        ind.evaluated = True
                    except Exception as e:
                        print(f"Error evaluating individual during migration: {e}")
                        ind.objectives = [1.0] * self.num_objectives
                        ind.evaluated = True
            
            # Sort by fitness
            candidates = sorted(subpop.population, key=lambda x: sum(x.objectives))
            
            # Select top individuals as potential migrants
            num_migrants = max(1, int(len(subpop.population) * self.migration_rate))
            
            # Get best solutions based on non-domination sorting
            if hasattr(subpop, 'fast_non_dominated_sort'):
                fronts = subpop.fast_non_dominated_sort(subpop.population)
                non_dominated = []
                for front in fronts:
                    non_dominated.extend(front)
                    if len(non_dominated) >= num_migrants:
                        break
                
                migrants = non_dominated[:num_migrants]
            else:
                # If not enough non-dominated solutions, add some random ones for diversity
                migrants = candidates[:num_migrants]
            
            all_migrants.append(migrants)
        
        # Now perform migration in a ring topology
        for i, subpop in enumerate(self.subpopulations):
            next_pop_idx = (i + 1) % self.num_subpopulations
            migrants = all_migrants[next_pop_idx]
            
            if not migrants:
                continue
                
            # Replace worst individuals in the current population
            subpop.population.sort(key=lambda ind: sum(ind.objectives))  # Simple aggregation for sorting
            
            # Replace the worst individuals with migrants
            for j, migrant in enumerate(migrants):
                migrant = self._ensure_individual(migrant)
                
                if j < len(subpop.population):
                    # Deep copy of the migrant
                    from .individual import Individual
                    new_migrant = Individual(migrant.genes.copy())
                    new_migrant.objectives = migrant.objectives.copy() if migrant.objectives else None
                    new_migrant.evaluated = migrant.evaluated
                    
                    # Replace one of the worst individuals
                    replace_idx = j
                    subpop.population[replace_idx] = new_migrant
    
    def run(self, generations):
        """Run the DRGA algorithm for the specified number of generations."""
        for gen in range(generations):
            # Evolve each subpopulation independently
            for subpop in self.subpopulations:
                subpop.evolve(1)  # Evolve for one generation
            
            # Re-evaluate all individuals to ensure objectives are up-to-date
            self.population = []
            for subpop in self.subpopulations:
                for ind in subpop.population:
                    # Ensure each individual is properly handled
                    self.population.append(self._ensure_individual(ind))
            
            self._evaluate_all_individuals()
            
            # Perform migration at specified intervals
            if gen > 0 and gen % self.migration_interval == 0:
                self.migrate()
                
            # Re-divide the population by ranges every few generations
            if gen % (self.migration_interval * 2) == 0:
                self.divide_population_by_ranges()
        
        # Gather all individuals from all subpopulations
        final_population = []
        for subpop in self.subpopulations:
            for ind in subpop.population:
                final_population.append(self._ensure_individual(ind))
        
        # Ensure all individuals are evaluated
        for i, ind in enumerate(final_population):
            ind = self._ensure_individual(ind)
            final_population[i] = ind
            if not hasattr(ind, 'objectives') or ind.objectives is None:
                ind.objectives = self.objectives.evaluate(ind.genes)
        
        # Return non-dominated solutions from the combined population
        non_dominated = self.subpopulations[0].select_non_dominated(final_population, len(final_population))
        return non_dominated