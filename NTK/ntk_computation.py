"""
NTK (Neural Tangent Kernel) computation for a 1-hidden-layer ReLU network.

Compares:
  - Finite-width MLP with BOTH layers trained via gradient descent
  - Infinite-width NTK kernel prediction (using same initialization f_0)

Parameterization (NTK scaling):
  f(x; a, B) = (1/sqrt(k)) * sum_j a_j * relu(b_j^T x)
  a_j ~ N(0,1),  b_j ~ N(0, I_d)

Infinite-width NTK kernel:
  Theta(x, x') = (1/2pi) * [sqrt(||x||^2||x'||^2 - (x^Tx')^2) + 2(x^Tx')(pi - arccos(rho))]

where rho = (x^T x') / (||x|| ||x'||)
"""

import numpy as np
from scipy.linalg import solve

SEED = 42


# =============================================================================
# Data
# =============================================================================

def sample_data(n, seed=SEED):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = np.sin(X[:, 0] * X[:, 1] * X[:, 4]) + np.cos(2 * X[:, 2]) + 5 * np.sin(X[:, 3])
    return X, y


# =============================================================================
# Finite-width: train both layers via gradient descent
# =============================================================================

def eval_both_layers_mlp(X_train, y_train, X_test, y_test, k, seed,
                         n_steps=5000, tol=1e-12):
    """
    Train both layers of a width-k ReLU MLP via gradient descent.

    Returns the test predictions f_trained(X_test).
    """
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]

    B = rng.randn(k, d)
    a = rng.randn(k)
    inv_sqrt_k = 1.0 / np.sqrt(k)

    # Compute initial empirical NTK to set learning rate
    Z0 = X_train @ B.T
    H0 = np.maximum(0, Z0)
    S0 = (Z0 > 0).astype(np.float64)
    dots = X_train @ X_train.T
    K = (inv_sqrt_k ** 2) * (H0 @ H0.T + dots * ((S0 * a ** 2) @ S0.T))
    lam_max = np.max(np.linalg.eigvalsh(K))
    lr = 1.0 / max(lam_max, 1e-8)

    a_ckpt = a.copy(); B_ckpt = B.copy()
    prev_loss = np.inf
    for step in range(n_steps):
        Z = X_train @ B.T
        H = np.maximum(0, Z)
        pred = inv_sqrt_k * (H @ a)
        residual = pred - y_train

        loss = np.mean(residual ** 2)
        if not np.isfinite(loss):
            # Diverged — restore checkpoint and halve lr
            a = a_ckpt.copy(); B = B_ckpt.copy()
            lr *= 0.5
            prev_loss = np.inf
            continue
        if loss < tol:
            break
        if loss > prev_loss * 1.5:
            # Loss spiked — restore checkpoint and halve lr
            a = a_ckpt.copy(); B = B_ckpt.copy()
            lr *= 0.5
            prev_loss = np.inf
            continue

        prev_loss = loss
        a_ckpt = a.copy(); B_ckpt = B.copy()

        grad_a = inv_sqrt_k * (H.T @ residual)
        S = (Z > 0).astype(np.float64)
        grad_B = inv_sqrt_k * (a[:, None] * ((S * residual[:, None]).T @ X_train))

        a -= lr * grad_a
        B -= lr * grad_B

    # Test prediction
    pred_test = inv_sqrt_k * (np.maximum(0, X_test @ B.T) @ a)
    return pred_test


def eval_trial(X_train, y_train, X_test, y_test, k, seed,
               K_train, K_test, n_steps=5000):
    """
    Run one trial: train finite-width MLP and compute NTK kernel prediction
    with the same initialization f_0.

    Returns (mlp_test_mse, ntk_test_mse).
    """
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]

    B = rng.randn(k, d)
    a = rng.randn(k)
    inv_sqrt_k = 1.0 / np.sqrt(k)

    # Initial predictions f_0
    f0_train = inv_sqrt_k * (np.maximum(0, X_train @ B.T) @ a)
    f0_test  = inv_sqrt_k * (np.maximum(0, X_test  @ B.T) @ a)

    # NTK kernel prediction: f_0 + Theta_test @ Theta_train^{-1} @ (y - f_0)
    alpha = solve(K_train + 1e-10 * np.eye(len(K_train)), y_train - f0_train)
    ntk_pred = f0_test + K_test.T @ alpha
    ntk_mse = float(np.mean((ntk_pred - y_test) ** 2))

    # Train the MLP via GD (re-initialize with same seed)
    mlp_pred = eval_both_layers_mlp(X_train, y_train, X_test, y_test, k, seed,
                                    n_steps=n_steps)
    mlp_mse = float(np.mean((mlp_pred - y_test) ** 2))

    return mlp_mse, ntk_mse


# =============================================================================
# Infinite-width NTK kernel
# =============================================================================

def _rho_and_radical(X1, X2):
    norms1 = np.linalg.norm(X1, axis=1)
    norms2 = np.linalg.norm(X2, axis=1)
    dots = X1 @ X2.T
    outer_norms = np.outer(norms1, norms2)
    rho = np.clip(dots / (outer_norms + 1e-30), -1.0, 1.0)
    radical = np.sqrt(np.clip(outer_norms ** 2 - dots ** 2, 0, None))
    return dots, rho, radical


def ntk_kernel_matrix(X1, X2):
    """
    Full NTK kernel for 1-hidden-layer ReLU net with scale 1/sqrt(k):
      f(x) = (1/sqrt(k)) * a^T relu(Bx),  a~N(0,1), B~N(0,I)

      Theta(x,x') = (1/2pi) * [radical + 2 * dots * (pi - arccos(rho))]
    """
    dots, rho, radical = _rho_and_radical(X1, X2)
    return (1.0 / (2.0 * np.pi)) * (radical + 2.0 * dots * (np.pi - np.arccos(rho)))
