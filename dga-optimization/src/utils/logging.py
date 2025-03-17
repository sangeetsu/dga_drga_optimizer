import logging

def setup_logging(log_file='dga_optimization.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_experiment_start(experiment_name):
    logging.info(f'Starting experiment: {experiment_name}')

def log_experiment_end(experiment_name):
    logging.info(f'Ending experiment: {experiment_name}')

def log_progress(iteration, best_solution):
    logging.debug(f'Iteration {iteration}: Best solution: {best_solution}')

def log_error(message):
    logging.error(message)