"""Metric primitives: rates with bootstrap intervals, paired tests, classifier metrics.

The suite is small (45 scenarios), so every headline rate comes with a 95% bootstrap
interval, and configuration comparisons use McNemar's exact test on the *paired*
per-scenario outcomes (the same scenarios run under both configurations), which is
more powerful than comparing two independent proportions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rate_ci(
    outcomes: Sequence[bool | float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap (1 - alpha) interval over scenarios."""
    x = np.asarray(outcomes, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float | int]:
    """Exact McNemar test on paired binary outcomes (two-sided binomial on discordants).

    ``b_only`` = scenarios only B solved, ``a_only`` = only A solved. The null is that
    a discordant pair is equally likely to go either way.
    """
    from scipy.stats import binomtest

    if len(a) != len(b):
        raise ValueError("Paired outcomes must have equal length.")
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_value": p}


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> float:
    """ECE: bin by confidence, weighted mean |accuracy - confidence| per bin."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else float("nan")
