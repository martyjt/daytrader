"""Sentiment adapters — scored news and social events.

Parallel to ``data.adapters`` (OHLCV) and ``data.macro`` (macro series).
A sentiment adapter returns timestamped events with optional numeric
scores — used to build sentiment features consumed by ML models and the
Exploration Agent.
"""

from __future__ import annotations
