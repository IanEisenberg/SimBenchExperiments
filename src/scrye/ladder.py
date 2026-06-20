"""The Ladder gate — the only door to the val set.

Blum & Hardt (2015, arXiv:1607.00091): release a new best only when it beats
the running best by more than a threshold eta. Under this rule the held-out
estimate's generalization error is bounded ~ (log(k·n)/n)^(1/3) regardless of
how many candidates k are tried — so an adaptive search cannot climb a noise
ladder. We choose this over Thresholdout because Thresholdout's differential-
privacy constants are impractical at our sample sizes.

eta is data-derived, not guessed: `bootstrap_eta` estimates the sampling noise
scale of the val mean-S, so the gate only fires on improvements larger than the
holdout's own finite-sample wobble.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class LadderGate:
    eta: float
    best: float = float("-inf")
    k: int = 0  # total queries spent (the global K when this gate is the run's gate)

    def consider(self, score: float) -> bool:
        """Spend one query; accept (and raise the bar) iff score > best + eta."""
        self.k += 1
        if score > self.best + self.eta:
            self.best = score
            return True
        return False


def bootstrap_eta(
    scores: Sequence[float],
    *,
    mult: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """`mult` × the std of bootstrapped means of a reference val score vector.

    This is the noise scale of the val mean-S: the smallest difference between
    two pipelines that is distinguishable from finite-sample wobble. Use a
    reference pipeline's per-record val scores as `scores`.
    """
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return float(mult * means.std())
