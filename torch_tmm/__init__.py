__version__ = "1.0.0"

# Import key classes from submodules for a flat public API.
from .layer import Layer  # layer definitions
from .material import Material
from .model import Model  # calculation and analysis classes
from .optical_calculator import OpticalCalculator

# Define the public API for the package.
__all__ = [
    "Layer",
    "Material",
    "Model",
    "OpticalCalculator",
]
