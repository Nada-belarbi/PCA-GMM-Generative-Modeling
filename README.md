# PCA-GMM Generative Modeling

## 📌 Project Overview

This project implements a complete generative modeling pipeline based on:

- Principal Component Analysis (PCA) for dimensionality reduction
- Gaussian Mixture Models (GMM) for probabilistic density estimation
- Synthetic data generation and reconstruction

The objective is to model high-dimensional Gaussian curves and generate new statistically consistent samples.

---

## 🧠 Methodology

The pipeline follows these steps:

1. Generate high-dimensional Gaussian curves.
2. Apply PCA to reduce intrinsic dimensionality.
3. Fit a Gaussian Mixture Model in PCA space.
4. Sample new points from the GMM.
5. Reconstruct synthetic curves in the original space.
6. Compare statistical properties between original and generated data.

---

## 📊 Experiments Conducted

- Varying number of PCA components
- Varying number of GMM clusters
- Statistical comparison using:
  - Mean and standard deviation distributions
  - Reconstruction error
  - Kolmogorov–Smirnov tests
- Robustness analysis under different noise levels

---

## 📁 Project Structure

- Data_Generation.ipynb
- Data_Generation.pdf
- create_gaussian_curves.py
- analyze_pca.py
- gmm_clustering.py
- curve_sampling.py
- curve_metrics.py



---

## 🛠 Requirements

- Python 3.9+
- NumPy
- Matplotlib
- scikit-learn
- SciPy

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn scipy
