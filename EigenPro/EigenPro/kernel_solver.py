"""
Kernel regression solvers using Richardson iteration and preconditioned Richardson iteration (EigenPro).
"""

import numpy as np
from scipy.sparse.linalg import lobpcg
from sklearn.metrics import r2_score


def gaussian_kernel(X1, X2, bandwidth):
    """Compute Gaussian (RBF) kernel matrix K(X1, X2)."""
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    dist_sq = sq1 - 2.0 * X1 @ X2.T + sq2.T
    return np.exp(-dist_sq / (2.0 * bandwidth ** 2))


def compute_top_eigenpairs(K, k):
    """Compute top-k eigenvalues/eigenvectors of K using LOBPCG."""
    n = K.shape[0]
    # LOBPCG needs an initial guess
    rng = np.random.RandomState(42)
    X0 = rng.randn(n, k)
    # LOBPCG finds largest eigenvalues of K
    eigenvalues, eigenvectors = lobpcg(K, X0, largest=True, maxiter=200, tol=1e-8)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def richardson_iteration(K_train, y_train, K_test, y_test, num_epochs, eta=None):
    """
    Standard Richardson iteration for kernel regression.

    Update: alpha_{t+1} = alpha_t - eta * (K * alpha_t - y)
    Optimal eta: slightly less than 1 / lambda_1(K)

    Returns per-epoch R^2 on train and test.
    """
    n = K_train.shape[0]

    # Compute lambda_1 via LOBPCG (just top-1)
    rng = np.random.RandomState(42)
    X0 = rng.randn(n, 1)
    lambda_1, _ = lobpcg(K_train, X0, largest=True, maxiter=200, tol=1e-8)
    lambda_1 = lambda_1[0]

    if eta is None:
        eta = 0.95 / lambda_1

    alpha = np.zeros(n)
    train_r2 = []
    test_r2 = []

    for epoch in range(num_epochs):
        residual = K_train @ alpha - y_train
        alpha = alpha - eta * residual

        # Compute R^2
        y_pred_train = K_train @ alpha
        y_pred_test = K_test @ alpha
        train_r2.append(r2_score(y_train, y_pred_train))
        test_r2.append(r2_score(y_test, y_pred_test))

    return train_r2, test_r2


def preconditioned_richardson_iteration(K_train, y_train, K_test, y_test,
                                         num_epochs, k, eta=None):
    """
    Preconditioned Richardson iteration (EigenPro).

    Preconditioner: B = I - sum_{i=1}^{k} (1 - lambda_{k+1}/lambda_i) u_i u_i^T
    Update: alpha_{t+1} = alpha_t - eta * B * (K * alpha_t - y)
    Optimal eta: slightly less than 1 / lambda_{k+1}(K)

    Returns per-epoch R^2 on train and test.
    """
    n = K_train.shape[0]

    # Compute top-(k+1) eigenpairs
    num_eig = min(k + 1, n - 1)
    eigenvalues, eigenvectors = compute_top_eigenpairs(K_train, num_eig)

    # lambda_{k+1} (0-indexed: eigenvalues[k])
    lambda_kp1 = eigenvalues[k] if len(eigenvalues) > k else eigenvalues[-1]

    # Top-k eigenvectors and eigenvalues
    U_k = eigenvectors[:, :k]          # (n, k)
    lambdas_k = eigenvalues[:k]        # (k,)

    # Precompute coefficients: (1 - lambda_{k+1} / lambda_i)
    coeffs = 1.0 - lambda_kp1 / lambdas_k  # (k,)

    if eta is None:
        eta = 0.95 / lambda_kp1

    alpha = np.zeros(n)
    train_r2 = []
    test_r2 = []

    for epoch in range(num_epochs):
        residual = K_train @ alpha - y_train
        # B * residual = residual - U_k @ diag(coeffs) @ U_k^T @ residual
        Ut_r = U_k.T @ residual                     # (k,)
        correction = U_k @ (coeffs * Ut_r)          # (n,)
        B_residual = residual - correction
        alpha = alpha - eta * B_residual

        # Compute R^2
        y_pred_train = K_train @ alpha
        y_pred_test = K_test @ alpha
        train_r2.append(r2_score(y_train, y_pred_train))
        test_r2.append(r2_score(y_test, y_pred_test))

    return train_r2, test_r2


def generate_data(n_samples, seed=42):
    """
    Generate data: x ~ N(0, I_5), y = sin(x1*x2*x5) + cos(2*x3) + 5*sin(x4).
    Returns X_train, y_train, X_test, y_test.
    """
    rng = np.random.RandomState(seed)
    n_total = n_samples + 500  # 500 fixed test points
    X = rng.randn(n_total, 5)
    y = np.sin(X[:, 0] * X[:, 1] * X[:, 4]) + np.cos(2 * X[:, 2]) + 5 * np.sin(X[:, 3])

    X_train, X_test = X[:n_samples], X[n_samples:]
    y_train, y_test = y[:n_samples], y[n_samples:]
    return X_train, y_train, X_test, y_test


def run_comparison(n_samples, k, bandwidth, num_epochs):
    """
    Run both methods and return per-epoch R^2 curves.
    """
    X_train, y_train, X_test, y_test = generate_data(n_samples)

    K_train = gaussian_kernel(X_train, X_train, bandwidth)
    K_test = gaussian_kernel(X_test, X_train, bandwidth)

    train_r2_std, test_r2_std = richardson_iteration(
        K_train, y_train, K_test, y_test, num_epochs
    )

    train_r2_pre, test_r2_pre = preconditioned_richardson_iteration(
        K_train, y_train, K_test, y_test, num_epochs, k
    )

    return {
        "epochs": list(range(1, num_epochs + 1)),
        "standard": {
            "train_r2": train_r2_std,
            "test_r2": test_r2_std,
        },
        "eigenpro": {
            "train_r2": train_r2_pre,
            "test_r2": test_r2_pre,
        },
    }
