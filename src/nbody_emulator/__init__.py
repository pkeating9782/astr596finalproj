# __init__.py

"""Neural network emulator for N-body simulations."""

__version__ = "0.1.0"

from . import data, utils, emulator, inference

__all__ = ["data", "utils", "emulator", "inference"]