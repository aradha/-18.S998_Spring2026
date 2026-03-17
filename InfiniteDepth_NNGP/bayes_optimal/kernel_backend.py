"""
Backend for infinite-depth NNGP kernel regression demo.

Data generation and kernel computations following:
  uhlerlab/inf_depth_ntks (https://github.com/uhlerlab/inf_depth_ntks)
"""

import numpy as np
from numpy.linalg import solve, norm
from scipy.stats import dirichlet
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Data generation  (following dataset.py)
# ---------------------------------------------------------------------------

def get_dirichlet_data(n_train, n_test, alpha1, alpha2, prior1=0.5, seed=42):
    """Sample Dirichlet data on the simplex and project to the unit sphere."""
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    size1 = int(total * prior1)

    X1 = rng.dirichlet(alpha1, size=size1)
    X2 = rng.dirichlet(alpha2, size=total - size1)
    SX = np.concatenate([X1, X2], axis=0)
    X = SX / norm(SX, axis=-1, keepdims=True)

    y = np.concatenate([np.ones(size1), -np.ones(total - size1)])

    perm = rng.permutation(total)
    X, y, SX = X[perm], y[perm], SX[perm]

    return (X[:n_train], y[:n_train],
            X[n_train:], y[n_train:], SX[n_train:])


# ---------------------------------------------------------------------------
# Dual activations  (following models.py)
# ---------------------------------------------------------------------------

def dual_act(angles, act_name):
    if act_name == '2d_opt':
        return angles**7 / 2 + angles / 2
    elif act_name == '3d_opt':
        return 0.5 * angles**3 + 0.5 * angles
    elif act_name == '5d_opt':
        return angles**3 / 4 + 0.5 * angles**2 + 0.25 * angles
    elif act_name == '9d_opt':
        return angles**3 / 16 + 7 / 8 * angles**2 + 1 / 16 * angles
    elif act_name == 'erf':
        return np.arcsin(2 * angles / 3) / np.arcsin(2 / 3)
    elif 'sine' in act_name:
        a = float(act_name.split("_")[-1])
        return np.sinh(a * angles) / np.sinh(a)
    elif act_name == 'relu':
        return (1 / np.pi) * (angles * (np.pi - np.arccos(angles))
                               + np.sqrt(1 - angles**2))
    elif 'hermite' in act_name:
        degree = int(act_name.split("_")[-1])
        return angles**degree
    else:
        raise ValueError(f"Unknown activation: {act_name}")


# ---------------------------------------------------------------------------
# NNGP kernel  (following models.py)
# ---------------------------------------------------------------------------

def nngp_kernel(X1, X2, depth, act_name='2d_opt'):
    """Compute the NNGP kernel by iterating the dual activation."""
    angles = X1 @ X2.T
    angles = np.clip(angles, -1, 1)
    for _ in range(depth):
        angles = dual_act(angles, act_name)
        angles = np.clip(angles, -1, 1)
    return angles


# ---------------------------------------------------------------------------
# Kernel regression  (following models.py  —  solve_kr)
# ---------------------------------------------------------------------------

def solve_kr(K_train, y_train, K_test):
    """Kernel regression: K alpha = y, then predict with K_test.

    Sets the NNGP diagonal to 1 before solving.
    """
    K = K_train.copy()
    np.fill_diagonal(K, 1.0)
    sol = solve(K, y_train)         # (n_train,)
    return sol @ K_test             # (n_train,) @ (n_train, n_test) → (n_test,)


def run_predictions(X_train, y_train, X_test, theta_grid,
                    depth, act_name='2d_opt'):
    """Kernel regression on both a dense angle grid and the test set.

    Returns (grid_preds, test_preds).
    """
    X_grid = np.column_stack([np.cos(theta_grid), np.sin(theta_grid)])

    K_train = nngp_kernel(X_train, X_train, depth=depth, act_name=act_name)
    K_grid  = nngp_kernel(X_train, X_grid,  depth=depth, act_name=act_name)
    K_test  = nngp_kernel(X_train, X_test,  depth=depth, act_name=act_name)

    K = K_train.copy()
    np.fill_diagonal(K, 1.0)
    sol = solve(K, y_train)

    return sol @ K_grid, sol @ K_test


# ---------------------------------------------------------------------------
# Bayes-optimal classifier  (following models.py)
# ---------------------------------------------------------------------------

def dirichlet_bayes_predict(alpha1, alpha2, prior1, SX):
    """Bayes-optimal classifier for Dirichlet-distributed data."""
    pdf1 = dirichlet.pdf(SX.T, alpha1) * prior1
    pdf2 = dirichlet.pdf(SX.T, alpha2) * (1 - prior1)
    return np.where(pdf1 > pdf2, 1.0, -1.0)


def dirichlet_bayes_grid(alpha1, alpha2, prior1, theta_grid):
    """Bayes-optimal predictions on a dense grid of angles."""
    cos_t = np.cos(theta_grid)
    sin_t = np.sin(theta_grid)
    s = cos_t + sin_t
    SX = np.column_stack([cos_t / s, sin_t / s])
    return dirichlet_bayes_predict(alpha1, alpha2, prior1, SX)


# ---------------------------------------------------------------------------
# Angular density  (analytic for 2-D Dirichlet on the simplex)
# ---------------------------------------------------------------------------

def angular_density(theta, alpha):
    """Density in angle space induced by a 2-D Dirichlet on the simplex.

    f_θ(θ) = [1/B(a1,a2)] cos^{a1-1}(θ) sin^{a2-1}(θ) / (cos θ + sin θ)^{a1+a2}
    """
    a1, a2 = alpha
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    beta = gamma(a1) * gamma(a2) / gamma(a1 + a2)
    return cos_t ** (a1 - 1) * sin_t ** (a2 - 1) / (cos_t + sin_t) ** (a1 + a2) / beta
