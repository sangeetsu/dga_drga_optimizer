import random
import numpy as np

class Crossover:
    @staticmethod
    def single_point_crossover(parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    @staticmethod
    def uniform_crossover(parent1, parent2, probability=0.5):
        child1 = []
        child2 = []
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < probability:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)
        return child1, child2

class Mutation:
    @staticmethod
    def bit_flip_mutation(individual, mutation_rate):
        return [1 - gene if random.random() < mutation_rate else gene for gene in individual]

    @staticmethod
    def gaussian_mutation(individual, mutation_rate, sigma=0.1):
        mutated = []
        for gene in individual:
            if random.random() < mutation_rate:
                mutated_gene = gene + random.gauss(0, sigma)
                # Keep within bounds [0, 1]
                mutated_gene = max(0, min(1, mutated_gene))
                mutated.append(mutated_gene)
            else:
                mutated.append(gene)
        return mutated

class Selection:
    @staticmethod
    def binary_tournament_selection(population, fitness_values):
        selected = []
        for _ in range(len(population) // 2):
            competitors = random.sample(range(len(population)), 2)
            # Assuming lower fitness is better (minimization)
            winner = competitors[0] if fitness_values[competitors[0]] < fitness_values[competitors[1]] else competitors[1]
            selected.append(population[winner])
        return selected

    @staticmethod
    def pareto_tournament_selection(population):
        selected = []
        from .selection import pareto_dominance
        
        for _ in range(len(population) // 2):
            competitors = random.sample(population, 2)
            if pareto_dominance(competitors[0].objectives, competitors[1].objectives):
                selected.append(competitors[0])
            elif pareto_dominance(competitors[1].objectives, competitors[0].objectives):
                selected.append(competitors[1])
            else:
                selected.append(random.choice(competitors))
        
        return selected

class Replacement:
    @staticmethod
    def generational_replacement_with_elitism(population, offspring, elite_size):
        # Sort population by fitness (assuming first objective for simplicity)
        sorted_pop = sorted(population, key=lambda ind: ind.objectives[0])
        # Take the best individuals as elites
        elites = sorted_pop[:elite_size]
        # Replace the rest with offspring
        return elites + offspring[:len(population) - elite_size]