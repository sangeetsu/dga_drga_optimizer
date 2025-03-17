import unittest
from src.optimization.objectives import ZDT1, ZDT2, DTLZ2
from src.optimization.problems import MultiObjectiveProblem

class TestOptimization(unittest.TestCase):

    def test_zdt1(self):
        problem = ZDT1()
        solution = problem.evaluate([0.5, 0.5])
        self.assertEqual(len(solution), 2)
        self.assertTrue(all(isinstance(s, float) for s in solution))

    def test_zdt2(self):
        problem = ZDT2()
        solution = problem.evaluate([0.5, 0.5])
        self.assertEqual(len(solution), 2)
        self.assertTrue(all(isinstance(s, float) for s in solution))

    def test_dtlz2(self):
        problem = DTLZ2()
        solution = problem.evaluate([0.5] * problem.num_variables)
        self.assertEqual(len(solution), problem.num_objectives)
        self.assertTrue(all(isinstance(s, float) for s in solution))

    def test_multi_objective_problem(self):
        problem = MultiObjectiveProblem()
        solutions = problem.solve()
        self.assertTrue(isinstance(solutions, list))
        self.assertTrue(all(isinstance(sol, list) for sol in solutions))

if __name__ == '__main__':
    unittest.main()