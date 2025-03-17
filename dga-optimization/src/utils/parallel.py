from multiprocessing import Pool, cpu_count

def parallel_execute(func, data, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(func, data)
    
    return results

def parallel_map(func, iterable, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    
    with Pool(processes=num_workers) as pool:
        return pool.map(func, iterable)

def parallel_starmap(func, iterable, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    
    with Pool(processes=num_workers) as pool:
        return pool.starmap(func, iterable)