"""Reinforcement learning algorithms.

RL agents implement the standard ``Algorithm`` ABC so they plug into the
existing DAG composer, walk-forward engine, and paper/live execution
pipelines with no special-casing:

* ``on_bar(ctx)`` — run the policy on the latest observation, emit a Signal
* ``train(data)`` — wrap the OHLCV DF as a gymnasium env and call
  ``model.learn(total_timesteps)``

Depends on ``stable-baselines3`` and ``gymnasium``. Install via
``pip install daytrader[rl]`` to enable. Without those packages, RL
agents are not registered — no import failure at startup.
"""

from __future__ import annotations
