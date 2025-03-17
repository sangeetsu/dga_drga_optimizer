import unittest
from src.analysis.metrics import calculate_performance_metrics
from src.analysis.visualization import plot_pareto_front
from src.analysis.statistics import perform_statistical_analysis

class TestAnalysisFunctions(unittest.TestCase):

    def test_calculate_performance_metrics(self):
        # Example input for performance metrics
        results = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
        metrics = calculate_performance_metrics(results)
        self.assertIsNotNone(metrics)
        self.assertIn('hypervolume', metrics)
        self.assertIn('igd', metrics)

    def test_plot_pareto_front(self):
        # Example input for plotting
        results = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
        try:
            plot_pareto_front(results)
        except Exception as e:
            self.fail(f"plot_pareto_front raised an exception: {e}")

    def test_perform_statistical_analysis(self):
        # Example input for statistical analysis
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        analysis_results = perform_statistical_analysis(data)
        self.assertIsNotNone(analysis_results)
        self.assertIn('mean', analysis_results)
        self.assertIn('std_dev', analysis_results)

if __name__ == '__main__':
    unittest.main()