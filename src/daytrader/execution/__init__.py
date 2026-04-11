"""Execution layer: paper and live trading adapters."""

from .base import ExecutionAdapter
from .loop import TradingLoop
from .paper import PaperExecutor
from .registry import ExecutionRegistry

__all__ = ["ExecutionAdapter", "ExecutionRegistry", "PaperExecutor", "TradingLoop"]
