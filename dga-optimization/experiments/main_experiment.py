import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import random

from src.dga import DGA
from src.optimization.objectives import ZDT1, ZDT2, DTLZ2, SchafferF6
from src.analysis import metrics, visualization, statistics
from src.utils import logging

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Experiment configuration
NUM_RUNS = 30
GENERATIONS = 100
POPULATION_SIZE = 100

def run_experiment(config):
    """Run a single experiment with the given configuration."""
    problem_class, algorithm_type, num_islands, migration_rate, migration_interval, migration_topology = config
    
    # Instantiate the problem
    problem = problem_class()
    
    # Special case for DTLZ2 - we need more generations
    additional_gens = 0
    if problem.__class__.__name__ == "DTLZ2":
        additional_gens = GENERATIONS  # Double the generations for DTLZ2
    
    start_time = time.time()
    
    # Initialize the DGA with problem-specific parameters
    dga = DGA(
        problem=problem,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS + additional_gens,
        algorithm_type=algorithm_type,
        num_islands=num_islands,
        migration_interval=migration_interval,
        migration_rate=migration_rate,
        migration_topology=migration_topology,
        chromosome_length=problem.num_variables  # Use problem-specific chromosome length
    )
    
    # Run the algorithm
    final_population = dga.run()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Create proper reference points based on the problem
    if problem.__class__.__name__ == "DTLZ2":
        # For DTLZ2, use [1.1, 1.1, 1.1] as reference point
        reference_point = np.array([1.1] * problem.num_objectives)
    elif problem.__class__.__name__ in ["ZDT1", "ZDT2"]:
        # For ZDT problems, use [1.1, 1.1]
        reference_point = np.array([1.1, 1.1])
    elif problem.__class__.__name__ == "SchafferF6":
        # For Schaffer F6, use problem-specific reference
        reference_point = np.array([4.1, 16.1])  # Based on problem bounds
    else:
        # Default adaptive reference point
        obj_values = [ind.objectives for ind in final_population if hasattr(ind, 'objectives')]
        if obj_values:
            obj_array = np.array(obj_values)
            max_vals = np.max(obj_array, axis=0)
            reference_point = max_vals * 1.1  # 10% larger than max values
        else:
            # Fallback reference point
            reference_point = np.array([1.1] * 2)
    
    # Calculate metrics
    metrics_results = metrics.calculate_performance_metrics(
        [ind for ind in final_population if hasattr(ind, 'objectives')], 
        reference_points=reference_point
    )
    
    # Add execution time and configuration details
    metrics_results['execution_time'] = execution_time
    metrics_results['problem'] = problem.__class__.__name__
    metrics_results['algorithm'] = algorithm_type
    metrics_results['num_islands'] = num_islands
    metrics_results['migration_rate'] = migration_rate
    metrics_results['migration_interval'] = migration_interval
    metrics_results['migration_topology'] = migration_topology
    
    # Save visualization of the Pareto front
    if hasattr(problem, 'num_objectives') and problem.num_objectives <= 3:
        obj_values = [ind.objectives for ind in final_population if hasattr(ind, 'objectives')]
        if obj_values:
            plot_path = f"results/{problem.__class__.__name__}_{algorithm_type}_{num_islands}.png"
            visualization.plot_pareto_front(obj_values, problem_name=problem.__class__.__name__, 
                                          algorithm_name=f"{algorithm_type}-{num_islands}", 
                                          save_path=plot_path)
    
    return metrics_results

def main():
    # Setup logging
    logging.setup_logging()
    logging.log_experiment_start("Main Experiment")
    
    # Define problems to test
    problems = [ZDT1, ZDT2, DTLZ2, SchafferF6]
    
    # Define algorithm configurations
    algorithm_configs = [
        # Standard GA
        ('standard', 1, 0, 0, 'none'),
        # Island Model with different configurations
        ('island', 5, 0.1, 5, 'ring'),
        ('island', 5, 0.1, 5, 'fully_connected'),
        ('island', 10, 0.1, 5, 'ring'),
        # DRGA with different configurations
        ('drga', 5, 0.1, 5, 'ring'),
        ('drga', 5, 0.2, 5, 'ring'),
        ('drga', 5, 0.1, 10, 'ring')
    ]
    
    # Create experiment configurations
    experiment_configs = []
    for problem_class in problems:
        for algorithm_type in ['standard', 'island', 'drga']:
            if algorithm_type == "standard":
                # Standard GA only runs with a single population
                experiment_configs.append((problem_class, algorithm_type, 1, 0.0, 0, "none"))
            elif algorithm_type == "island":
                # Try different migration parameters for island model
                for num_islands in [5, 10]:
                    for migration_topology in ["ring", "fully_connected"]:
                        if num_islands == 10 and migration_topology == "fully_connected":
                            continue  # Skip this combination to reduce experiment count
                        experiment_configs.append(
                            (problem_class, algorithm_type, num_islands, 0.1, 5, migration_topology))
            elif algorithm_type == "drga":
                # Try different parameters for DRGA
                experiment_configs.append((problem_class, algorithm_type, 5, 0.1, 5, "ring"))
                experiment_configs.append((problem_class, algorithm_type, 5, 0.2, 5, "ring"))
                experiment_configs.append((problem_class, algorithm_type, 5, 0.1, 10, "ring"))
    
    # Run the experiments in parallel
    print(f"Running {len(experiment_configs)} experiments...")
    with Pool() as pool:
        run_results = pool.map(run_experiment, experiment_configs)
    
    print("Experiments complete! Analyzing results...")
    # Convert results to DataFrame
    results_df = pd.DataFrame(run_results)
    
    # Save all results to CSV
    results_df.to_csv("experiment_results.csv", index=False)
    
    # Group by problem type, algorithm type, and parameters for summary statistics
    # Remove 'convergence_gen' since it's not calculated in the metrics function
    metrics = ['hypervolume', 'spread', 'execution_time']  # Removed 'convergence_gen'
    summary = results_df.groupby(['problem', 'algorithm', 'num_islands', 'migration_rate', 'migration_interval'])[metrics]
    
    # Calculate statistics across repeated runs
    summary_stats = summary.agg(['mean', 'std', 'min', 'max'])
    summary_stats.to_csv("experiment_summary.csv")
    
    # Create plots for each problem
    for problem in results_df['problem'].unique():
        problem_results = results_df[results_df['problem'] == problem]
        
        # Hypervolume comparison
        plt.figure(figsize=(10, 6))
        for algo in problem_results['algorithm'].unique():
            algo_data = problem_results[problem_results['algorithm'] == algo]
            if algo == "standard":
                label = "Standard GA"
            elif algo == "island":
                label = f"Island Model (5 islands)"
            else:
                label = "DRGA"
            
            plt.boxplot(algo_data['hypervolume'], positions=[['standard', 'island', 'drga'].index(algo)], 
                        widths=0.6, labels=[label])
        
        plt.title(f"Hypervolume Comparison for {problem}")
        plt.ylabel("Hypervolume")
        plt.savefig(f"{problem}_hypervolume.png")
    
    print("Analysis complete! Results saved to CSV files and plots generated.")
    logging.log_experiment_end("Main Experiment")
    print("Experiment completed. Results saved to experiment_results.csv and experiment_summary.csv")

if __name__ == "__main__":
    main()