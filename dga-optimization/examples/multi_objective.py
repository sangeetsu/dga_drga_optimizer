import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.dga import DGA
from src.optimization.problems import ZDT1, ZDT2
from src.analysis import metrics
from src.analysis import visualization

def main():
    # Define the multi-objective optimization problem
    problem = ZDT1()  # or ZDT2() for a different problem

    # Initialize the Distributed Genetic Algorithm
    dga = DGA(
        problem=problem, 
        population_size=100, 
        generations=100, 
        algorithm_type="drga",  # Try "standard", "island", or "drga"
        num_islands=5,
        migration_interval=5,
        migration_rate=0.1
    )

    # Run the DGA
    final_population = dga.run()
    
    # Calculate reference point for hypervolume (slightly worse than the worst objective values)
    objectives = np.array([ind.objectives for ind in final_population])
    reference_point = [1.1, 1.1]  # Assuming normalized objectives in [0,1]
    
    # Analyze results
    metrics_results = metrics.calculate_performance_metrics(
        final_population, 
        reference_points=reference_point
    )
    
    # Plot the Pareto front
    visualization.plot_pareto_front(
        np.array([ind.objectives for ind in final_population]),
        title="Pareto Front - ZDT1 Problem"
    )

    # Print metrics
    print("Performance Metrics:", metrics_results)

if __name__ == "__main__":
    main()