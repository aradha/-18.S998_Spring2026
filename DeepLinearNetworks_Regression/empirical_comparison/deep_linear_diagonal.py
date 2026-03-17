"""
Deep Linear Diagonal Networks for Sparse Linear Regression.

Parameterization: w = w_1 ⊙ w_2 ⊙ ... ⊙ w_L
Each w_l ∈ R^d is initialized to α·1.
Trained with full-batch gradient descent on MSE loss.

Data: x ~ N(0, I_d), y = <β*, x> where β* is supported on first k coordinates.
"""

import numpy as np


def generate_data(n, d=1000, k=50, seed=None):
    """Generate sparse linear regression data (noiseless).

    x ~ N(0, I_d), y = <β*, x> where β*_i = 1 for i < k, 0 otherwise.
    """
    rng = np.random.RandomState(seed)
    beta_star = np.zeros(d)
    beta_star[:k] = 1.0
    X = rng.randn(n, d)
    y = X @ beta_star
    return X, y, beta_star


def init_layers(L, d=1000, alpha=0.01):
    """Initialize L layers, each as α·1 ∈ R^d."""
    return [np.full(d, alpha) for _ in range(L)]


def compute_product(layers):
    """Compute w = w_1 ⊙ w_2 ⊙ ... ⊙ w_L."""
    w = np.ones_like(layers[0])
    for wl in layers:
        w = w * wl
    return w


def mse_loss(X, y, w):
    """Mean squared error: (1/2n) ||Xw - y||^2."""
    residual = X @ w - y
    return 0.5 * np.mean(residual ** 2)


def compute_gradients(X, y, layers):
    """Compute gradient of MSE w.r.t. each layer.

    ∂L/∂w_l = (1/n) X^T (Xw - y) ⊙ ∏_{j≠l} w_j
    """
    n = X.shape[0]
    w = compute_product(layers)
    residual = X @ w - y  # (n,)
    base_grad = X.T @ residual / n  # (d,)

    grads = []
    for l in range(len(layers)):
        # Product of all layers except l
        others = w / (layers[l] + 1e-30)  # avoid division by zero
        grads.append(base_grad * others)
    return grads


def train(X_train, y_train, X_test, y_test, L, alpha, lr, num_steps):
    """Train deep linear diagonal network with full-batch GD.

    Returns:
        layers: trained layer parameters
        history: dict with train_loss, test_loss lists
    """
    d = X_train.shape[1]
    layers = init_layers(L, d, alpha)

    history = {"train_loss": [], "test_loss": [], "w_snapshots": []}

    for step in range(num_steps + 1):
        w = compute_product(layers)
        train_loss = mse_loss(X_train, y_train, w)
        test_loss = mse_loss(X_test, y_test, w)
        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))

        if step % max(1, num_steps // 20) == 0:
            history["w_snapshots"].append(w.copy().tolist())

        if step < num_steps:
            grads = compute_gradients(X_train, y_train, layers)
            for l in range(L):
                layers[l] -= lr * grads[l]

    return layers, history


def train_streaming(X_train, y_train, X_test, y_test, L, alpha, lr, num_steps,
                    update_every=None):
    """Generator that yields training state at regular intervals for live viz."""
    if update_every is None:
        update_every = max(1, num_steps // 200)

    d = X_train.shape[1]
    layers = init_layers(L, d, alpha)

    for step in range(num_steps + 1):
        if step % update_every == 0 or step == num_steps:
            w = compute_product(layers)
            yield {
                'step': step,
                'train_loss': float(mse_loss(X_train, y_train, w)),
                'test_loss': float(mse_loss(X_test, y_test, w)),
                'w_current': w.tolist(),
            }

        if step < num_steps:
            grads = compute_gradients(X_train, y_train, layers)
            for l in range(L):
                layers[l] -= lr * grads[l]
