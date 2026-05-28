"""
Ion Source Simulations
======================

A physics simulation package for modeling ion source behavior with hot surface
ion sources used in radioactive ion beam generation.

The simulation models:
- Thermal electron emission from a heated tantalum tube
- Ion generation and transport
- Electric potential well dynamics
- Large-scale particle tracking
"""

__version__ = "0.1.0"
__author__ = "Stu-Pid1"

from . import physics
from . import solvers
from . import particles
from . import visualization
from .simulation import IonSourceSimulation

__all__ = [
    "physics",
    "solvers",
    "particles",
    "visualization",
    "IonSourceSimulation",
]
