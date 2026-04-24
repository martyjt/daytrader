"""Research utilities: Exploration Agent, feature lift tests, discoveries.

This package hosts the "meta-learner" side of Daytrader — background
jobs that generate hypotheses (candidate features, promising parameter
regions, algo combinations worth trying) and score them with proper
out-of-sample validation before surfacing them to the user.
"""

from __future__ import annotations
