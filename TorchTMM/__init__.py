__version__ = "1.0.0"

# Import key classes from submodules for a flat public API.
from .layer import BaseLayer  # layer definitions
from .material import BaseMaterial
from .model import Model  # calculation and analysis classes

# Define the public API for the package.
__all__ = [
    "BaseLayer",
    "BaseMaterial",
    "Model",
]
