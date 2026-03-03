"""curve_sampling.py

Sampling utilities to generate new curves from a fitted GMM in PCA space.



The function returns both PCA samples and the reconstructed curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


@dataclass
class SamplingResult:
    """Container for sampling outputs."""

    Z_samples: np.ndarray  # (n_samples, n_components)
    curves_samples: np.ndarray  # (n_samples, n_features)
    sample_clusters: np.ndarray  # (n_samples,)


def sample_new_curves(
    gmm: GaussianMixture,
    pca: PCA,
    scaler: StandardScaler,
    *,
    n_samples: int = 100,
) -> SamplingResult:
    """Sample new curves from a GMM in PCA space and reconstruct.

    Parameters
    ----------
    gmm:
        Fitted GaussianMixture.
    pca:
        Fitted PCA (same PCA used to produce embeddings for the GMM).
    scaler:
        Fitted StandardScaler used before PCA.
    n_samples:
        Number of new curves to generate.

    Returns
    -------
    SamplingResult
        Z_samples: samples in PCA space
        curves_samples: reconstructed curves in original feature space
        sample_clusters: cluster/component index for each sample
    """

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    Z_samples, sample_clusters = gmm.sample(n_samples)
# Inverse PCA: back to scaled feature space
# gmm may have been trained on only the first k PCA components (e.g., 2D),
# while pca can have more components (e.g., 20). sklearn's inverse_transform
# requires matching dimensionality, so we manually reconstruct using k components.
    k = Z_samples.shape[1]

    if pca.components_.shape[0] == k:
    # Same dimensionality -> standard inverse_transform works
        curves_scaled = pca.inverse_transform(Z_samples)
    else:
    # Manual inverse transform with the first k components
        curves_scaled = Z_samples @ pca.components_[:k, :] + pca.mean_

    
    # Inverse scaling: back to original space
    curves_samples = scaler.inverse_transform(curves_scaled)

    return SamplingResult(
        Z_samples=Z_samples,
        curves_samples=curves_samples,
        sample_clusters=sample_clusters,
    )
