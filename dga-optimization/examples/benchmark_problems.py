from src.optimization.problems import ZDT1, ZDT2, DTLZ2

def benchmark_zdt1():
    problem = ZDT1()
    # Setup and run the DGA for ZDT1
    # Example: results = run_dga(problem)
    print("Running benchmark for ZDT1...")

def benchmark_zdt2():
    problem = ZDT2()
    # Setup and run the DGA for ZDT2
    # Example: results = run_dga(problem)
    print("Running benchmark for ZDT2...")

def benchmark_dtlz2():
    problem = DTLZ2()
    # Setup and run the DGA for DTLZ2
    # Example: results = run_dga(problem)
    print("Running benchmark for DTLZ2...")

if __name__ == "__main__":
    benchmark_zdt1()
    benchmark_zdt2()
    benchmark_dtlz2()