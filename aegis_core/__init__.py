"""
AEGIS Core Package

This package contains the core functionality for the AEGIS framework,
including error injection, MAS wrappers, and utilities.
"""

__version__ = "0.1.0"

from .malicious_factory import MaliciousSystem
from .agent_systems import get_mas_wrapper, BaseMASWrapper
from .utils import load_config, setup_logging

__all__ = [
    "MaliciousSystem",
    "get_mas_wrapper", 
    "BaseMASWrapper",
    "load_config",
    "setup_logging"
]