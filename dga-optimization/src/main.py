import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimization.problems import ZDT1Problem, ZDT2Problem, DTLZ2Problem, SchafferF6Problem
from dga.island_model import IslandModel
from dga.base import GeneticAlgorithmBase
from optimization.objectives import ZDT1, ZDT2, DTLZ2

def plot_2d_front(front, title="Pareto Front"):
    """Plot a 2D Pareto front."""
    front_array = np.array(front)
    plt.figure(figsize=(10, 6))
    plt.scatter(front_array[:, 0], front_array[:, 1], c='blue', s=30)
    plt.title(title)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.grid(True)
    plt.show()

def plot_3d_front(front, title="Pareto Front"):
    """Plot a 3D Pareto front."""
    front_array = np.array(front)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(front_array[:, 0], front_array[:, 1], front_array[:, 2], c='blue', s=30)
    ax.set_title(title)
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    plt.show()

def run_zdt1_example():
    """Run and visualize ZDT1 problem."""
    print("Solving ZDT1 problem...")
    problem = ZDT1Problem(population_size=100, generations=50)
    front = problem.solve()
    plot_2d_front(front, "ZDT1 Pareto Front")
    return front

def run_island_model_example():
    """Run and visualize ZDT1 problem using Island Model."""
    print("Solving ZDT1 problem with Island Model...")
    
    # Create a wrapper for the objective function
    class ObjectivesWrapper:
        def __init__(self):
            self.obj_func = ZDT1()
        
        def evaluate(self, genes):
            return self.obj_func.evaluate(genes)
    
    # Parameters for the GA instances
    ga_params = {
        'population_size': 50,  # Smaller populations per island
        'chromosome_length': 30,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'objectives': ObjectivesWrapper(),
        'elitism_size': 5
    }
    
    # Create and run the Island Model
    island_model = IslandModel(
        num_islands=4,
        migration_interval=5,
        migration_rate=0.1,
        topology="ring",
        ga_class=GeneticAlgorithmBase,
        ga_params=ga_params
    )
    
    front = island_model.run(generations=10)  # This will run for 10*5=50 generations total
    
    # Extract objectives from individuals
    pareto_front = [ind.objectives for ind in front]
    
    plot_2d_front(pareto_front, "ZDT1 Pareto Front (Island Model)")
    return pareto_front

def run_dtlz2_example():
    """Run and visualize DTLZ2 problem."""
    print("Solving DTLZ2 problem...")
    problem = DTLZ2Problem(population_size=100, generations=50, 
                          num_variables=12, num_objectives=3)
    front = problem.solve()
    plot_3d_front(front, "DTLZ2 Pareto Front")
    return front

if __name__ == "__main__":
    if len(sys.argv) > 1:
        problem_name = sys.argv[1].lower()
        if problem_name == "zdt1":
            run_zdt1_example()
        elif problem_name == "zdt2":
            problem = ZDT2Problem(population_size=100, generations=50)
            front = problem.solve()
            plot_2d_front(front, "ZDT2 Pareto Front")
        elif problem_name == "dtlz2":
            run_dtlz2_example()
        elif problem_name == "schaffer":
            problem = SchafferF6Problem(population_size=100, generations=50)
            front = problem.solve()
            plot_2d_front(front, "Schaffer F6 Pareto Front")
        elif problem_name == "island":
            run_island_model_example()
        else:
            print(f"Unknown problem: {problem_name}")
            print("Available problems: zdt1, zdt2, dtlz2, schaffer, island")
    else:
        # Default: run ZDT1
        run_zdt1_example()
