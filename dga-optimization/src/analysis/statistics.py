from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

class Statistics:
    @staticmethod
    def mean(data):
        return np.mean(data)

    @staticmethod
    def median(data):
        return np.median(data)

    @staticmethod
    def variance(data):
        return np.var(data)

    @staticmethod
    def standard_deviation(data):
        return np.std(data)

    @staticmethod
    def confidence_interval(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean - h, mean + h

    @staticmethod
    def describe(data):
        return {
            'mean': Statistics.mean(data),
            'median': Statistics.median(data),
            'variance': Statistics.variance(data),
            'standard_deviation': Statistics.standard_deviation(data),
            'confidence_interval': Statistics.confidence_interval(data)
        }

def perform_statistical_analysis(data: List[float]) -> Dict[str, Any]:
    """
    Perform basic statistical analysis on a dataset.
    
    Args:
        data: List of values to analyze
        
    Returns:
        Dict: Dictionary containing statistical measures
    """
    if not data:
        return {
            'mean': None,
            'median': None,
            'std_dev': None,
            'min': None,
            'max': None,
            'quartiles': None
        }
    
    # Convert to numpy array
    data_array = np.array(data)
    
    # Calculate basic statistics
    mean = np.mean(data_array)
    median = np.median(data_array)
    std_dev = np.std(data_array)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    # Calculate quartiles
    q1, q3 = np.percentile(data_array, [25, 75])
    
    # Return as dictionary
    return {
        'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'min': min_val,
        'max': max_val,
        'quartiles': (q1, median, q3)
    }

def compare_algorithms(results1: List[float], results2: List[float], 
                      alpha: float = 0.05) -> Dict[str, Any]:
    """
    Statistically compare the results of two algorithms.
    
    Args:
        results1: Results from first algorithm
        results2: Results from second algorithm
        alpha: Significance level for statistical tests
        
    Returns:
        Dict: Dictionary containing statistical comparison results
    """
    if not results1 or not results2:
        return {'error': 'Empty input data'}
    
    # Convert to numpy arrays
    arr1 = np.array(results1)
    arr2 = np.array(results2)
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
    
    # Perform Wilcoxon signed-rank test for paired data if arrays have the same length
    if len(arr1) == len(arr2):
        w_stat, w_p_value = stats.wilcoxon(arr1, arr2)
    else:
        # Perform Mann-Whitney U test for unpaired data
        w_stat, w_p_value = stats.mannwhitneyu(arr1, arr2)
    
    # Calculate effect size (Cohen's d)
    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
    pooled_std = np.sqrt(((len(arr1) - 1) * std1**2 + (len(arr2) - 1) * std2**2) / 
                        (len(arr1) + len(arr2) - 2))
    effect_size = abs(mean1 - mean2) / pooled_std
    
    # Interpret effect size
    if effect_size < 0.2:
        effect_interpretation = 'negligible'
    elif effect_size < 0.5:
        effect_interpretation = 'small'
    elif effect_size < 0.8:
        effect_interpretation = 'medium'
    else:
        effect_interpretation = 'large'
    
    return {
        'means': (mean1, mean2),
        'std_devs': (std1, std2),
        't_test': {
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha
        },
        'wilcoxon_test': {
            'statistic': w_stat,
            'p_value': w_p_value,
            'significant': w_p_value < alpha
        },
        'effect_size': {
            'value': effect_size,
            'interpretation': effect_interpretation
        }
    }