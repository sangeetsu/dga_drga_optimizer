import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Tuple, Optional
import os

def plot_pareto_front(points: List[Tuple[float, ...]], reference_front: List[Tuple[float, ...]] = None,
                     title: str = "Pareto Front", save_path: Optional[str] = None) -> None:
    """
    Plot the Pareto front for 2D or 3D problems.
    
    Args:
        points: List of objective vectors (tuples of objective values)
        reference_front: Optional reference Pareto front for comparison
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    if not points:
        print("No points to plot")
        return
    
    # Convert to numpy array for easier manipulation
    points_array = np.array(points)
    
    # Determine dimensionality
    n_objectives = points_array.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    if n_objectives == 2:
        # 2D plot
        plt.scatter(points_array[:, 0], points_array[:, 1], c='blue', label='Approximated Front')
        
        if reference_front:
            ref_array = np.array(reference_front)
            plt.scatter(ref_array[:, 0], ref_array[:, 1], c='red', label='True Front')
            
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        
    elif n_objectives == 3:
        # 3D plot
        ax = plt.axes(projection='3d')
        ax.scatter3D(points_array[:, 0], points_array[:, 1], points_array[:, 2], c='blue', label='Approximated Front')
        
        if reference_front:
            ref_array = np.array(reference_front)
            ax.scatter3D(ref_array[:, 0], ref_array[:, 1], ref_array[:, 2], c='red', label='True Front')
            
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        
    else:
        # For more than 3 objectives, use a parallel coordinates plot
        from pandas.plotting import parallel_coordinates
        import pandas as pd
        
        # Create DataFrame for parallel coordinates
        df = pd.DataFrame(points_array, columns=[f'Obj{i+1}' for i in range(n_objectives)])
        df['Front'] = 'Approximated'
        
        if reference_front:
            ref_df = pd.DataFrame(np.array(reference_front), 
                                columns=[f'Obj{i+1}' for i in range(n_objectives)])
            ref_df['Front'] = 'True'
            df = pd.concat([df, ref_df])
            
        parallel_coordinates(df, 'Front')
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_population_history(history: List[List[Tuple[float, ...]]], 
                           interval: int = 10, save_path: Optional[str] = None) -> None:
    """
    Plot the evolution of the population over generations.
    
    Args:
        history: List of populations across generations, each containing objective vectors
        interval: Plot every nth generation
        save_path: If provided, save the plot to this path
    """
    if not history:
        print("No history to plot")
        return
    
    # Get dimensionality from the first point of the first generation
    n_objectives = len(history[0][0])
    
    if n_objectives != 2:
        print(f"Plotting population history for {n_objectives}D problems not implemented")
        return
        
    plt.figure(figsize=(12, 10))
    
    # Plot a subset of generations with different colors
    generations_to_plot = list(range(0, len(history), interval))
    if generations_to_plot and generations_to_plot[-1] != len(history) - 1:
        generations_to_plot.append(len(history) - 1)  # Always include the last generation
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(generations_to_plot)))
    
    for i, gen_idx in enumerate(generations_to_plot):
        population = history[gen_idx]
        population_array = np.array(population)
        
        plt.scatter(population_array[:, 0], population_array[:, 1], 
                    color=colors[i], label=f'Generation {gen_idx}', alpha=0.7)
    
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Population Evolution')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_pareto_front(population, problem_name, algorithm_name, save_path=None):
    """
    Plot the Pareto front for a given population.
    
    Args:
        population: List of individuals with objectives or list of objective vectors
        problem_name: Name of the problem being solved
        algorithm_name: Name of the algorithm used
        save_path: Path to save the plot (if None, plot is displayed)
        
    Returns:
        None
    """
    # Extract objective values
    if hasattr(population[0], 'objectives'):
        objectives = np.array([ind.objectives for ind in population])
    else:
        objectives = np.array(population)
    
    # Determine the dimensionality
    n_objectives = objectives.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    if n_objectives == 2:
        # 2D Pareto front
        plt.scatter(objectives[:, 0], objectives[:, 1], c='blue', alpha=0.7)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Pareto Front: {problem_name} - {algorithm_name}')
        plt.grid(True, alpha=0.3)
        
    elif n_objectives == 3:
        # 3D Pareto front
        ax = plt.axes(projection='3d')
        ax.scatter3D(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', alpha=0.7)
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title(f'Pareto Front: {problem_name} - {algorithm_name}')
        
    else:
        # For more than 3 objectives, plot pairwise scatter plots
        max_dims = min(5, n_objectives)  # Limit to first 5 objectives
        fig, axes = plt.subplots(max_dims, max_dims, figsize=(12, 10))
        plt.suptitle(f'Pairwise Objective Plots: {problem_name} - {algorithm_name}')
        
        for i in range(max_dims):
            for j in range(max_dims):
                if i == j:
                    # Histogram on the diagonal
                    axes[i, j].hist(objectives[:, i], bins=15, alpha=0.7)
                    axes[i, j].set_title(f'Obj {i+1}')
                else:
                    # Scatter plot for pairs of objectives
                    axes[i, j].scatter(objectives[:, j], objectives[:, i], alpha=0.5, s=20)
                    if i == max_dims - 1:
                        axes[i, j].set_xlabel(f'Obj {j+1}')
                    if j == 0:
                        axes[i, j].set_ylabel(f'Obj {i+1}')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_pareto_fronts(fronts_dict, problem_name, save_path=None):
    """
    Compare multiple Pareto fronts on the same plot.
    
    Args:
        fronts_dict: Dictionary with algorithm names as keys and Pareto fronts as values
        problem_name: Name of the problem being solved
        save_path: Path to save the plot (if None, plot is displayed)
        
    Returns:
        None
    """
    # Check if we have any data to plot
    if not fronts_dict:
        print("No data to plot")
        return
    
    # Determine the dimensionality from the first front
    first_algo = next(iter(fronts_dict))
    first_front = fronts_dict[first_algo]
    
    if hasattr(first_front[0], 'objectives'):
        objectives = np.array([ind.objectives for ind in first_front])
    else:
        objectives = np.array(first_front)
    
    n_objectives = objectives.shape[1]
    
    plt.figure(figsize=(12, 9))
    
    if n_objectives == 2:
        # 2D Pareto fronts
        colors = plt.cm.jet(np.linspace(0, 1, len(fronts_dict)))
        
        for (algo_name, front), color in zip(fronts_dict.items(), colors):
            if hasattr(front[0], 'objectives'):
                obj_values = np.array([ind.objectives for ind in front])
            else:
                obj_values = np.array(front)
            
            plt.scatter(obj_values[:, 0], obj_values[:, 1], color=color, alpha=0.7, label=algo_name)
        
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Pareto Front Comparison: {problem_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    elif n_objectives == 3:
        # 3D Pareto fronts
        ax = plt.axes(projection='3d')
        colors = plt.cm.jet(np.linspace(0, 1, len(fronts_dict)))
        
        for (algo_name, front), color in zip(fronts_dict.items(), colors):
            if hasattr(front[0], 'objectives'):
                obj_values = np.array([ind.objectives for ind in front])
            else:
                obj_values = np.array(front)
            
            ax.scatter3D(
                obj_values[:, 0], obj_values[:, 1], obj_values[:, 2], 
                color=color, alpha=0.7, label=algo_name
            )
        
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title(f'Pareto Front Comparison: {problem_name}')
        ax.legend()
        
    else:
        # For more than 3 objectives, create pairwise scatter plots for first two objectives only
        for (algo_name, front), color in zip(fronts_dict.items(), plt.cm.jet(np.linspace(0, 1, len(fronts_dict)))):
            if hasattr(front[0], 'objectives'):
                obj_values = np.array([ind.objectives for ind in front])
            else:
                obj_values = np.array(front)
            
            plt.scatter(obj_values[:, 0], obj_values[:, 1], color=color, alpha=0.7, label=algo_name)
        
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Pareto Front Comparison (First 2 Objectives): {problem_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metrics_comparison(results_df, metric_name, problem_name=None, save_path=None):
    """
    Plot a comparison of a metric across different algorithms.
    
    Args:
        results_df: DataFrame with experiment results
        metric_name: Name of the metric to compare
        problem_name: Optional problem name filter
        save_path: Path to save the plot (if None, plot is displayed)
        
    Returns:
        None
    """
    if problem_name:
        df = results_df[results_df['problem'] == problem_name].copy()
    else:
        df = results_df.copy()
    
    if len(df) == 0:
        print(f"No data for problem: {problem_name}")
        return
    
    # Group by problem and algorithm
    grouped = df.groupby(['problem', 'algorithm'])[metric_name]
    
    # Calculate statistics
    stats = grouped.agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Prepare for plotting
    problems = stats['problem'].unique()
    n_problems = len(problems)
    
    fig, axes = plt.subplots(1, n_problems, figsize=(6*n_problems, 8), sharey=True)
    
    # Handle the case where there's only one problem
    if n_problems == 1:
        axes = [axes]
    
    for i, problem in enumerate(problems):
        problem_stats = stats[stats['problem'] == problem]
        
        # Plot mean with error bars
        algorithms = problem_stats['algorithm'].values
        means = problem_stats['mean'].values
        stds = problem_stats['std'].values
        
        # Create x positions
        x_pos = np.arange(len(algorithms))
        
        # Plot bars with error bars
        axes[i].bar(x_pos, means, yerr=stds, align='center', alpha=0.7, 
                   ecolor='black', capsize=10)
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(algorithms, rotation=45)
        axes[i].set_title(f'{problem}')
        axes[i].yaxis.grid(True)
        
        # Add value labels
        for j, v in enumerate(means):
            axes[i].text(j, v + stds[j] + 0.01*max(means), 
                        f'{v:.3f}', ha='center', fontweight='bold')
    
    # Add common labels
    fig.suptitle(f'Comparison of {metric_name}', fontsize=16)
    fig.text(0.04, 0.5, metric_name.capitalize(), va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()