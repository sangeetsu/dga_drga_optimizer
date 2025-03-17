# Distributed Genetic Algorithms for Multi-Objective Optimization

This project implements and analyzes Distributed Genetic Algorithms (DGAs) for solving multi-objective optimization problems. The framework is designed to facilitate experimentation with various genetic operators, selection strategies, and migration policies.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

Distributed Genetic Algorithms extend traditional genetic algorithms by distributing the population across multiple subpopulations (islands). This approach allows for parallel exploration of the solution space and can enhance the diversity of solutions found.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To run the DGA, you can use the provided examples. For instance, to execute a basic DGA, run:

```
python examples/basic_dga.py
```

For multi-objective optimization problems, use:

```
python examples/multi_objective.py
```

## Modules

The project is organized into several modules:

- **dga**: Contains the core implementation of the genetic algorithm framework, including population management, genetic operators, and migration strategies.
- **optimization**: Defines the objective functions and constraints for multi-objective optimization problems.
- **analysis**: Provides tools for analyzing the performance of the DGA, including metrics and visualizations.
- **utils**: Contains utility functions for parallel execution and logging.

## Examples

The `examples` directory includes scripts demonstrating how to use the DGA framework:

- `basic_dga.py`: A simple implementation of a DGA.
- `multi_objective.py`: An example of applying DGA to multi-objective optimization problems.
- `benchmark_problems.py`: Examples of setting up and running benchmark problems.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.