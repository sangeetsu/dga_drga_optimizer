import numpy as np
from typing import List, Dict, Any, Union

def calculate_performance_metrics(population: List[Any], reference_points: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    """
    Calculate various performance metrics for the given population.
    
    Args:
        population: List of individuals
        reference_points: Reference points for hypervolume calculation
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Guard against empty population
    if not population:
        return {
            'hypervolume': 0.0,
            'spread': 0.0,
            'non_dominated_count': 0
        }
    
    # Get non-dominated solutions
    non_dominated = []
    if hasattr(population[0], 'objectives'):
        # Filter out individuals with empty objectives
        valid_individuals = [ind for ind in population if hasattr(ind, 'objectives') and 
                            ind.objectives is not None and len(ind.objectives) > 0]
        
        if not valid_individuals:
            return {
                'hypervolume': 0.0,
                'spread': 0.0,
                'non_dominated_count': 0
            }
            
        # Use existing rank if available
        if hasattr(valid_individuals[0], 'rank'):
            non_dominated = [ind for ind in valid_individuals if ind.rank == 0]
        else:
            # Simple non-dominated sorting
            non_dominated = simple_non_dominated_sort(valid_individuals)
    else:
        # Assume population is already a list of objective values
        non_dominated = simple_non_dominated_sort(population)
    
    # Count non-dominated solutions
    metrics['non_dominated_count'] = len(non_dominated)
    
    # Calculate hypervolume if possible
    if non_dominated:
        try:
            # Extract objective values
            if hasattr(non_dominated[0], 'objectives'):
                obj_values = [ind.objectives for ind in non_dominated]
            else:
                obj_values = non_dominated
                
            # Convert to numpy array for calculations
            obj_array = np.array(obj_values)
            
            # Ensure the array is not empty and has appropriate dimensions
            if obj_array.size > 0 and obj_array.shape[1] > 0:
                metrics['hypervolume'] = adaptive_hypervolume(obj_array, reference_points)
            else:
                metrics['hypervolume'] = 0.0
        except Exception as e:
            print(f"Hypervolume calculation error: {e}, setting to 0.0")
            metrics['hypervolume'] = 0.0
    else:
        metrics['hypervolume'] = 0.0
    
    # Calculate diversity metrics (spread)
    if len(non_dominated) >= 2:
        try:
            if hasattr(non_dominated[0], 'objectives'):
                obj_values = [ind.objectives for ind in non_dominated]
            else:
                obj_values = non_dominated
                
            obj_array = np.array(obj_values)
            
            if obj_array.size > 0 and obj_array.shape[1] > 0:
                metrics['spread'] = calculate_spread(obj_array)
            else:
                metrics['spread'] = 0.0
        except Exception as e:
            print(f"Spread calculation error: {e}, setting to 0.0")
            metrics['spread'] = 0.0
    else:
        metrics['spread'] = 0.0
    
    return metrics

def adaptive_hypervolume(points_array, reference_point):
    """
    Calculate hypervolume with adaptive reference point handling.
    
    Args:
        points_array: Array of points (objective values)
        reference_point: Reference point for hypervolume calculation
    
    Returns:
        Hypervolume value
    """
    # Handle empty arrays or arrays with zero dimensions
    if points_array.size == 0 or points_array.shape[0] == 0 or len(points_array.shape) < 2 or points_array.shape[1] == 0:
        print("Warning: Empty points array in hypervolume calculation")
        return 0.0
    
    # Make sure reference_point is a numpy array
    if not isinstance(reference_point, np.ndarray):
        reference_point = np.array(reference_point)
    
    # Ensure reference_point has the right shape
    if len(reference_point.shape) == 0:  # Scalar
        reference_point = np.array([reference_point] * points_array.shape[1])
    elif len(reference_point.shape) == 1 and reference_point.shape[0] != points_array.shape[1]:
        # Broadcast to match dimensions
        if points_array.shape[1] > 0:
            reference_point = np.ones(points_array.shape[1]) * np.max(reference_point)
        else:
            # Fall back to a default reference point with 2 dimensions
            reference_point = np.array([1.1, 1.1])
    
    try:
        return simplified_hypervolume_approximation(points_array, reference_point)
    except Exception as e:
        print(f"Hypervolume calculation failed: {e}")
        return 0.0

def simple_non_dominated_sort(population):
    """
    Simple implementation of non-dominated sorting.
    
    Args:
        population: List of individuals
        
    Returns:
        List of non-dominated individuals (first front)
    """
    non_dominated = []
    
    # Check if population is empty
    if not population:
        return non_dominated
    
    for i in range(len(population)):
        is_dominated = False
        ind_i = population[i]
        
        # Skip individuals with no objectives
        if not hasattr(ind_i, 'objectives') or not ind_i.objectives:
            continue
            
        for j in range(len(population)):
            if i == j:
                continue
                
            ind_j = population[j]
            
            # Skip individuals with no objectives
            if not hasattr(ind_j, 'objectives') or not ind_j.objectives:
                continue
                
            # Make sure both individuals have the same number of objectives
            if len(ind_i.objectives) != len(ind_j.objectives):
                continue
            
            # Check if j dominates i
            dominates = True
            has_better = False
            
            for k in range(len(ind_i.objectives)):
                if ind_j.objectives[k] > ind_i.objectives[k]:
                    dominates = False
                    break
                elif ind_j.objectives[k] < ind_i.objectives[k]:
                    has_better = True
            
            if dominates and has_better:
                is_dominated = True
                break
        
        if not is_dominated:
            non_dominated.append(ind_i)
    
    return non_dominated

def calculate_spread(points_array):
    """
    Calculate spread metric for a set of points.
    
    Args:
        points_array: Array of points (objective values)
    
    Returns:
        Spread value
    """
    # Handle empty arrays or invalid dimensions
    if points_array.size == 0 or points_array.shape[0] < 2 or points_array.shape[1] == 0:
        return 0.0
    
    try:
        # For 2D points, calculate the standard spread metric
        if points_array.shape[1] == 2:
            # Sort points by first objective
            sorted_points = points_array[points_array[:, 0].argsort()]
            
            # Calculate distances between consecutive points
            distances = np.sqrt(np.sum(np.diff(sorted_points, axis=0)**2, axis=1))
            
            # Calculate mean distance
            if len(distances) > 0:
                mean_dist = np.mean(distances)
                
                # Calculate spread as the sum of absolute deviations from the mean
                spread = np.sum(np.abs(distances - mean_dist)) / (len(distances) * mean_dist)
                return spread
            else:
                return 0.0
        else:
            # For higher dimensions, use a generalized spread metric
            # Based on the average distance to the nearest neighbor
            from scipy.spatial.distance import pdist, squareform
            
            # Calculate pairwise distances
            if len(points_array) >= 2:
                dists = squareform(pdist(points_array))
                
                # Set diagonal elements to infinity to ignore self-distances
                np.fill_diagonal(dists, np.inf)
                
                # Find minimum distance for each point
                min_dists = np.min(dists, axis=1)
                
                # Calculate spread as coefficient of variation of minimum distances
                if np.mean(min_dists) > 0:
                    spread = np.std(min_dists) / np.mean(min_dists)
                    return spread
                else:
                    return 0.0
            else:
                return 0.0
    except Exception as e:
        print(f"Spread calculation error: {e}")
        return 0.0

def simplified_hypervolume_approximation(points_array, reference_point):
    """
    Simplified hypervolume approximation.
    
    Args:
        points_array: Array of points (objective values)
        reference_point: Reference point for hypervolume calculation
    
    Returns:
        Hypervolume value
    """
    # Handle empty arrays or arrays with zero dimensions
    if points_array.size == 0 or points_array.shape[0] == 0 or len(points_array.shape) < 2 or points_array.shape[1] == 0:
        print("Warning: Empty points array in simplified hypervolume calculation")
        return 0.0
    
    # Make sure reference_point has the right shape
    if reference_point.shape[0] != points_array.shape[1]:
        print(f"Warning: Reference point shape {reference_point.shape} doesn't match points shape {points_array.shape}")
        if points_array.shape[1] > 0:
            reference_point = np.ones(points_array.shape[1]) * 1.1  # Default fallback
        else:
            return 0.0  # Can't proceed with empty dimension
            
    try:
        # Filter points that dominate the reference point
        valid_points = points_array[np.all(points_array <= reference_point, axis=1)]
        
        # If no points are valid, return 0
        if valid_points.size == 0 or valid_points.shape[0] == 0:
            return 0.0
            
        # Calculate the hypervolume
        if valid_points.shape[1] == 2:
            # For 2D, sort by first objective
            sorted_points = valid_points[valid_points[:, 0].argsort()]
            
            # Calculate the area under the curve
            hv = 0.0
            prev_x = reference_point[0]
            
            for point in sorted_points[::-1]:  # Start from the rightmost point
                hv += (prev_x - point[0]) * (reference_point[1] - point[1])
                prev_x = point[0]
            
            return hv
        else:
            # For higher dimensions, use Monte Carlo approximation
            num_samples = 10000
            
            # Generate random points in the hypercube defined by origin and reference point
            samples = np.random.uniform(0, 1, size=(num_samples, valid_points.shape[1])) * reference_point
            
            # Count samples that are dominated by at least one point
            dominated_count = 0
            for sample in samples:
                dominated = False
                for point in valid_points:
                    if np.all(point <= sample):
                        dominated = True
                        break
                if dominated:
                    dominated_count += 1
            
            # Calculate hypervolume as fraction of dominated samples times volume of hypercube
            hypercube_volume = np.prod(reference_point)
            hv = (dominated_count / num_samples) * hypercube_volume
            
            return hv
    except Exception as e:
        print(f"Simplified hypervolume calculation failed: {e}")
        return 0.0