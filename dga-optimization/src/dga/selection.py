def pareto_dominance(solution_a, solution_b):
    """Evaluate if solution_a dominates solution_b."""
    return all(x <= y for x, y in zip(solution_a, solution_b)) and any(x < y for x, y in zip(solution_a, solution_b))

def non_dominated_sorting(population):
    """Perform non-dominated sorting on the population."""
    fronts = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in population:
            if pareto_dominance(p.objectives, q.objectives):
                p.dominated_solutions.append(q)
            elif pareto_dominance(q.objectives, p.objectives):
                p.domination_count += 1
        if p.domination_count == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]

def select_non_dominated(population, n):
    """Select the top n non-dominated solutions from the population."""
    fronts = non_dominated_sorting(population)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n:
            selected.extend(front)
        else:
            selected.extend(front[:n - len(selected)])
            break
    return selected