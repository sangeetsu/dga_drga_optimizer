"""Configuration settings for DGA experiments"""

# General settings
RANDOM_SEED = 42
NUM_RUNS = 30
GENERATIONS = 100
POPULATION_SIZE = 100

# Genetic algorithm parameters
CHROMOSOME_LENGTH = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01

# Island model parameters
NUM_ISLANDS = 10
ISLAND_SIZE = 10  # POPULATION_SIZE / NUM_ISLANDS
MIGRATION_INTERVAL = 5
MIGRATION_RATE = 0.1
MIGRATION_TOPOLOGIES = ["ring", "fully_connected", "random"]

# Reference points for different problems (for hypervolume calculation)
REFERENCE_POINTS = {
    "ZDT1": [1.1, 1.1],
    "ZDT2": [1.1, 1.1],
    "DTLZ2": [1.1, 1.1, 1.1],
    "SchafferF6": [1.1, 1.1]
}