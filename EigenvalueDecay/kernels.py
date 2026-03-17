"""
Gaussian kernel on R with Gaussian-distributed samples.

Kernel:  K(x, z) = exp(-(x - z)^2 / (2 ell^2))

When samples are drawn from N(0, 1), the eigenvalues of the integral
operator (Mercer expansion under the Gaussian measure) decay as

    lambda_k = sqrt(2a / A) * B^k

For ell = 1 this simplifies to  lambda_k ~ r^(k + 1/2)  where
r = (3 - sqrt(5)) / 2  (the reciprocal golden ratio squared).
"""

import numpy as np


def sample_points(n, seed=42):
    """Draw n points from N(0, 1)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def build_kernel_matrix(x, ell=1.0):
    """Gaussian kernel matrix K_{ij} = exp(-(x_i - x_j)^2 / (2 ell^2))."""
    D = x[:, None] - x[None, :]
    return np.exp(-D ** 2 / (2.0 * ell ** 2))


def theoretical_eigenvalues(k, ell=1.0):
    """First k theoretical eigenvalues (descending).

    r = (3 - sqrt(5)) / 2 for ell = 1.
    General formula uses Fasshauer-McCourt parameters.
    """
    r = (3.0 - np.sqrt(5.0)) / 2.0
    indices = np.arange(k)
    return r ** (indices + 0.5)
