"""gmm_clustering.py

Gaussian Mixture Model (GMM) clustering utilities for the "Data Generation"
work session.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMClusteringResult:
    """Container for GMM clustering outputs."""

    gmm: GaussianMixture
    cluster_labels: np.ndarray  # shape: (n_samples,)
    cluster_probs: np.ndarray  # shape: (n_samples, n_clusters)


def cluster_with_gmm(
    Z: np.ndarray,
    n_clusters: int,
    *,
    covariance_type: str = "full",
    random_state: int = 42,
    reg_covar: float = 1e-6,
    max_iter: int = 500,
    n_init: int = 5,
    verbose: int = 0,
) -> GMMClusteringResult:
    """Cluster PCA embeddings using a Gaussian Mixture Model.

    Parameters
    ----------
    Z:
        PCA embeddings with shape (n_samples, n_components_used).
    n_clusters:
        Number of mixture components.
    covariance_type:
        One of {"full", "tied", "diag", "spherical"}.
    random_state:
        For reproducibility.
    reg_covar:
        Non-negative regularization added to the diagonal of covariance.
        Helps avoid numerical issues.
    max_iter, n_init:
        Standard sklearn GMM settings.
    verbose:
        Verbosity level for sklearn.

    Returns
    -------
    GMMClusteringResult
        Contains:
        - gmm: fitted GaussianMixture
        - cluster_labels: argmax responsibilities
        - cluster_probs: responsibilities (soft assignment)
    """

    if not isinstance(Z, np.ndarray):
        Z = np.asarray(Z)

    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (n_samples, n_features). Got shape {Z.shape}.")

    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=random_state,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=n_init,
        verbose=verbose,
    )

    gmm.fit(Z)
    cluster_labels = gmm.predict(Z)
    cluster_probs = gmm.predict_proba(Z)

    return GMMClusteringResult(
        gmm=gmm,
        cluster_labels=cluster_labels,
        cluster_probs=cluster_probs,
    )
