"""curve_metrics.py

Metrics and feature extraction for 1D curves.

We extract simple interpretable parameters from each curve:
- mean (mu_hat) and std (sigma_hat) using weighted moments
- peak location x_peak and peak height y_max

This is useful to compare original vs generated distributions.
"""

from __future__ import annotations

import numpy as np


def extract_curve_features(x: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> dict:
    """Extract per-curve features from curves.

    Parameters
    ----------
    x:
        1D grid of shape (nbx,)
    Y:
        Curves of shape (N, nbx)
    eps:
        Small value to avoid division by zero.

    Returns
    -------
    dict of numpy arrays (length N)
    """

    x = np.asarray(x)
    Y = np.asarray(Y)

    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (N, nbx)")
    if Y.shape[1] != x.shape[0]:
        raise ValueError("Y second dimension must match x")

    # Ensure non-negative weights for moments (noise can make slightly negative values)
    W = np.clip(Y, 0.0, None)
    mass = W.sum(axis=1) + eps

    mu_hat = (W * x[None, :]).sum(axis=1) / mass
    var_hat = (W * (x[None, :] - mu_hat[:, None]) ** 2).sum(axis=1) / mass
    sigma_hat = np.sqrt(np.clip(var_hat, 0.0, None))

    idx_peak = np.argmax(Y, axis=1)
    x_peak = x[idx_peak]
    y_max = Y[np.arange(Y.shape[0]), idx_peak]

    # Negative proportion is a good sanity metric when noise is present
    neg_prop = (Y < 0).mean(axis=1)

    return {
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "x_peak": x_peak,
        "y_max": y_max,
        "neg_prop": neg_prop,
        "mass": mass,
    }


def summary_stats(arr: np.ndarray) -> dict:
    """Return mean/std/min/max/median for an array."""
    arr = np.asarray(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }
