import numpy as np
from typing import List, Dict

def calculate_distance(ind1, ind2, genotypic=True):
    """Calculate distance between two individuals"""
    if genotypic:
        # Euclidean distance in genotype space
        return np.sqrt(np.sum((ind1.genes - ind2.genes)**2))
    else:
        # Euclidean distance in phenotype (objective) space
        return np.sqrt(np.sum((np.array(ind1.objectives) - np.array(ind2.objectives))**2))

def fitness_sharing(population, alpha=1, sigma_share=0.2, genotypic=True):
    """
    Implements fitness sharing to reduce the fitness of similar individuals.
    
    Args:
        population: List of individuals
        alpha: Power parameter for the sharing function (typically 1)
        sigma_share: Sharing distance threshold
        genotypic: If True, use genotype for distance, else use phenotype
    """
    n = len(population)
    
    for i in range(n):
        niche_count = 0
        for j in range(n):
            distance = calculate_distance(population[i], population[j], genotypic)
            
            # Apply sharing function
            if distance < sigma_share:
                sh = 1.0 - (distance / sigma_share) ** alpha
            else:
                sh = 0
            
            niche_count += sh
        
        # Adjust fitness (assuming lower objectives are better)
        if niche_count > 0:
            for obj_idx in range(len(population[i].objectives)):
                population[i].objectives[obj_idx] *= niche_count
    
    return population

def clearing(population, kappa=1, sigma_clearing=0.2, genotypic=True):
    """
    Implements the clearing method to maintain diversity.
    Only the best kappa individuals in each niche maintain their fitness.
    
    Args:
        population: List of individuals
        kappa: Number of winners in each niche
        sigma_clearing: Clearing distance threshold
        genotypic: If True, use genotype for distance, else use phenotype
    """
    n = len(population)
    winners = [False] * n
    
    # Sort by fitness (assuming first objective for simplicity)
    sorted_indices = sorted(range(n), key=lambda i: population[i].objectives[0])
    
    for i in sorted_indices:
        if not winners[i]:
            # This is a winner
            winners[i] = True
            niche_winners = 1
            
            for j in sorted_indices:
                if i != j and niche_winners < kappa:
                    distance = calculate_distance(population[i], population[j], genotypic)
                    if distance < sigma_clearing:
                        winners[j] = True
                        niche_winners += 1
    
    # Reset non-winners' fitness to a poor value
    for i in range(n):
        if not winners[i]:
            # Set to a very high value (assuming minimization)
            population[i].objectives = [float('inf')] * len(population[i].objectives)
    
    return population