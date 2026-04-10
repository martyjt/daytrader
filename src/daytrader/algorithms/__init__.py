"""Algorithm layer: plugin base, registry, built-in algorithms."""

from .base import Algorithm, AlgorithmManifest
from .registry import AlgorithmRegistry

__all__ = ["Algorithm", "AlgorithmManifest", "AlgorithmRegistry"]
