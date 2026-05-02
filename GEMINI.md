# Project: intelligen

`intelligen` is a Python library providing a suite of mathematical and artificial intelligence tools. It includes functionality for linear models, statistics, numeric methods, signal processing, and more. For performance-critical parts, it incorporates C++ extensions for special mathematical functions.

## Project Structure

- `intelligen/`: Main Python package source code.
    - `linear_model/`: Implementation of linear models (regression, gradient descent).
    - `metrics/`: Performance metrics for models.
    - `special/`: Special mathematical functions, including C++ extensions.
    - `stats/`: Probability distributions (continuous and discrete).
    - `utils/`: Common utility functions.
    - `AI.py`: Basic neural network implementations.
- `src/`: C++ source code for native extensions (Faddeeva, erfinv).
- `tests/`: Unit tests for the package modules.
- `docs/`: Project documentation source (MkDocs).
- `CMakeLists.txt`: Build configuration for C++ extensions.
- `pyproject.toml`: Project metadata, dependencies, and tool configurations.

## Technology Stack

- **Language:** Python (>= 3.8), C++17
- **Numerical Computing:** NumPy
- **Plotting:** Matplotlib
- **Build System:** scikit-build-core (with CMake)
- **C++ Bindings:** nanobind
- **Linting & Formatting:** Ruff
- **Testing:** Pytest / Unittest
- **Documentation:** MkDocs (with Material theme)

## Building and Running

### Installation
To install the package and its dependencies:
```bash
pip install .
```
*Note: A C++ compiler and CMake are required to build the native extensions.*

### Development Setup
To install in editable mode with development tools:
```bash
pip install -e .[dev]
```

### Testing
Execute tests using `pytest`:
```bash
pytest
```
Alternatively, using `unittest`:
```bash
python -m unittest discover tests
```

### Linting and Formatting
Check for linting issues:
```bash
ruff check .
```
Automatically format code:
```bash
ruff format .
```

### Documentation
To serve the documentation locally:
```bash
mkdocs serve
```

## Development Conventions

- **Code Style:** Adhere to the Ruff configuration in `pyproject.toml`.
- **Docstrings:** Use the **NumPy** docstring convention.
- **Native Extensions:** C++ code should be placed in `src/` and exposed via `intelligen/special/`. We use **nanobind** for bindings. New extensions must be registered in `CMakeLists.txt`.
- **Testing:** Add unit tests for all new functionality in the `tests/` directory. Ensure existing tests pass before contributing.
- **Architecture:** Keep modules focused and maintain the clear separation between mathematical domains (stats, numeric, signals, etc.).
