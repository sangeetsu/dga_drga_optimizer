import unittest
from dga import base, operators, selection, migration

class TestDGA(unittest.TestCase):

    def setUp(self):
        # Initialize any necessary components for the tests
        self.population_size = 100
        self.genetic_algorithm = base.GeneticAlgorithm(population_size=self.population_size)

    def test_crossover(self):
        parent1 = [0, 1, 1, 0, 1]
        parent2 = [1, 0, 0, 1, 0]
        offspring = operators.single_point_crossover(parent1, parent2)
        self.assertNotEqual(offspring, parent1)
        self.assertNotEqual(offspring, parent2)

    def test_mutation(self):
        individual = [0, 1, 1, 0, 1]
        mutated_individual = operators.bit_flip_mutation(individual)
        self.assertNotEqual(mutated_individual, individual)

    def test_selection(self):
        population = [([0, 1, 1, 0, 1], 0.8), ([1, 0, 0, 1, 0], 0.6)]
        selected = selection.tournament_selection(population)
        self.assertIn(selected, population)

    def test_migration(self):
        island_population = [[0, 1, 1], [1, 0, 0]]
        migrated_population = migration.migrate(island_population)
        self.assertNotEqual(migrated_population, island_population)

if __name__ == '__main__':
    unittest.main()