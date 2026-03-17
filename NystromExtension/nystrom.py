"""
Nystrom Extension and Kernel Regression Backend.

Nystrom approximation:
    Given m centers X_m from training data X (n x d), and kernel K:
    1. Compute K_mm = K(X_m, X_m)                          [m x m]
    2. Eigendecompose K_mm = U diag(s) U^T
    3. Feature map: phi(x) = diag(s^{-1/2}) U^T K(X_m, x)  [m x 1]
    4. Solve ridge regression in feature space:
       alpha = (Phi Phi^T + lambda I)^{-1} Phi y
       where Phi = phi(X) is [m x n]

Standard kernel regression:
    alpha = (K(X, X) + lambda I)^{-1} y
"""

import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
import time


def laplace_kernel(X, Z, lengthscale=1.0):
    """K(x,z) = exp(-||x - z||_2 / L)"""
    diff = X[:, None, :] - Z[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return np.exp(-dists / lengthscale)


def laplace_kernel_chunked(X, Z, lengthscale=1.0, chunk_size=2000):
    """Memory-efficient Laplace kernel using chunked computation."""
    n = X.shape[0]
    m = Z.shape[0]
    K = np.empty((n, m), dtype=np.float64)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        diff = X[i:end, None, :] - Z[None, :, :]
        K[i:end] = np.exp(-np.linalg.norm(diff, axis=2) / lengthscale)
    return K


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def generate_data(n, d=5, seed=0):
    """
    x ~ N(0, I_d)
    y = sin(3 x1 x2 x5) + cos(2 x3) + 5 sin(x4)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = (np.sin(3 * X[:, 0] * X[:, 1] * X[:, 4])
         + np.cos(2.0 * X[:, 2])
         + 5.0 * np.sin(X[:, 3]))
    return X, y


# ---------------------------------------------------------------------------
# Streaming (generator) versions — yield one step dict at a time
# ---------------------------------------------------------------------------

def nystrom_regression_stream(X_train, y_train, X_test, y_test,
                              kernel_fn, m, lam=1e-3, seed=42):
    """Yield {"name", "time"} dicts per step, then a final summary dict."""
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]

    t0 = time.time()
    indices = rng.choice(n, size=m, replace=False)
    X_m = X_train[indices]
    yield {"type": "step", "name": "Sample m centers", "time": time.time() - t0}

    t0 = time.time()
    K_mm = kernel_fn(X_m, X_m)
    yield {"type": "step", "name": "Compute K_mm (m\u00d7m)", "time": time.time() - t0}

    t0 = time.time()
    s, U = eigh(K_mm)
    threshold = 1e-10 * max(s.max(), 1e-10)
    valid = s > threshold
    s = s[valid]
    U = U[:, valid]
    s_inv_sqrt = 1.0 / np.sqrt(s)
    yield {"type": "step", "name": "Eigendecompose K_mm", "time": time.time() - t0}

    t0 = time.time()
    K_mn_train = kernel_fn(X_m, X_train)
    Phi_train = (s_inv_sqrt[:, None]) * (U.T @ K_mn_train)
    yield {"type": "step", "name": "Feature map \u03a6(X_train)", "time": time.time() - t0}

    t0 = time.time()
    A = Phi_train @ Phi_train.T + lam * np.eye(Phi_train.shape[0])
    b = Phi_train @ y_train
    L, low = cho_factor(A, lower=True)
    w = cho_solve((L, low), b)
    yield {"type": "step", "name": "Solve (\u03a6\u03a6\u1d40+\u03bbI)w = \u03a6y", "time": time.time() - t0}

    t0 = time.time()
    train_pred = Phi_train.T @ w
    yield {"type": "step", "name": "Train predictions", "time": time.time() - t0}

    t0 = time.time()
    K_mn_test = kernel_fn(X_m, X_test)
    Phi_test = (s_inv_sqrt[:, None]) * (U.T @ K_mn_test)
    test_pred = Phi_test.T @ w
    yield {"type": "step", "name": "Test predictions", "time": time.time() - t0}

    yield {
        "type": "done",
        "train_r2": float(r_squared(y_train, train_pred)),
        "test_r2": float(r_squared(y_test, test_pred)),
    }


def kernel_regression_stream(X_train, y_train, X_test, y_test,
                             kernel_fn, lam=1e-3):
    """Yield {"name", "time"} dicts per step, then a final summary dict."""
    n = X_train.shape[0]

    t0 = time.time()
    K = kernel_fn(X_train, X_train)
    K[np.diag_indices(n)] += lam
    yield {"type": "step", "name": "Compute K(X,X) + \u03bbI", "time": time.time() - t0}

    t0 = time.time()
    L, low = cho_factor(K, lower=True)
    alpha = cho_solve((L, low), y_train)
    yield {"type": "step", "name": "Cholesky solve (n\u00d7n)", "time": time.time() - t0}

    t0 = time.time()
    train_pred = K @ alpha - lam * alpha
    yield {"type": "step", "name": "Train predictions", "time": time.time() - t0}

    t0 = time.time()
    K_test = kernel_fn(X_test, X_train)
    test_pred = K_test @ alpha
    yield {"type": "step", "name": "Test predictions", "time": time.time() - t0}

    yield {
        "type": "done",
        "train_r2": float(r_squared(y_train, train_pred)),
        "test_r2": float(r_squared(y_test, test_pred)),
    }


def stream_nystrom(n_train, m_centers, lam, lengthscale=np.sqrt(5.0),
                   n_test=2000, seed=0):
    """Stream Nystrom regression steps as event dicts."""
    t0 = time.time()
    X_train, y_train = generate_data(n_train, seed=seed)
    X_test, y_test = generate_data(n_test, seed=seed + 9999)
    kern = lambda X, Z: laplace_kernel_chunked(X, Z, lengthscale=lengthscale)
    yield {"event": "data_ready", "time": float(time.time() - t0)}

    total = 0.0
    for msg in nystrom_regression_stream(
            X_train, y_train, X_test, y_test, kern,
            m=m_centers, lam=lam, seed=seed + 1):
        if msg["type"] == "step":
            total += msg["time"]
            yield {"event": "step", "name": msg["name"],
                   "time": float(msg["time"])}
        elif msg["type"] == "done":
            yield {"event": "done", "train_r2": msg["train_r2"],
                   "test_r2": msg["test_r2"], "time": float(total)}

    yield {"event": "finished"}


def stream_kernel(n_train, lam, lengthscale=np.sqrt(5.0),
                  n_test=2000, seed=0):
    """Stream kernel regression steps as event dicts."""
    t0 = time.time()
    X_train, y_train = generate_data(n_train, seed=seed)
    X_test, y_test = generate_data(n_test, seed=seed + 9999)
    kern = lambda X, Z: laplace_kernel_chunked(X, Z, lengthscale=lengthscale)
    yield {"event": "data_ready", "time": float(time.time() - t0)}

    if n_train > 15000:
        yield {"event": "skipped",
               "reason": f"n={n_train} too large for exact O(n^3) solve"}
        yield {"event": "finished"}
        return

    total = 0.0
    for msg in kernel_regression_stream(
            X_train, y_train, X_test, y_test, kern, lam=lam):
        if msg["type"] == "step":
            total += msg["time"]
            yield {"event": "step", "name": msg["name"],
                   "time": float(msg["time"])}
        elif msg["type"] == "done":
            yield {"event": "done", "train_r2": msg["train_r2"],
                   "test_r2": msg["test_r2"], "time": float(total)}

    yield {"event": "finished"}
