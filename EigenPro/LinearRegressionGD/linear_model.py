"""
Linear regression with gradient descent on a 2D Gaussian model.

Model:
    y = <w*, x> + epsilon
    x ~ N(0, S),  S = diag(1, k)   [smallest eigenvalue 1, largest k]
    epsilon ~ N(0, 1)
    w* = [1, 1]

Default learning rate: eta = 0.9 / λ_max(X^T X)

Usage (command-line demo):
    python linear_model.py --n 200 --k 10 --steps 100
"""

import argparse
import numpy as np


def generate_data(n: int, k: float, w_star=None, seed: int = 42):
    """
    Generate n samples from  y = <w*, x> + epsilon.

    Parameters
    ----------
    n      : number of samples
    k      : condition number  (largest eigenvalue of S = diag(1, k))
    w_star : true parameter vector, default [1.0, 1.0]
    seed   : random seed for reproducibility

    Returns
    -------
    X : ndarray (n, 2)  – features sampled from N(0, S)
    y : ndarray (n,)    – responses
    """
    rng = np.random.default_rng(seed)

    if w_star is None:
        w_star = np.array([1.0, 1.0])
    else:
        w_star = np.asarray(w_star, dtype=float)

    # S = diag(1, k): smallest eigenvalue = 1, largest = k
    S = np.diag([1.0, float(k)])

    X = rng.multivariate_normal(np.zeros(2), S, size=n) * 1/k  # x ~ N(0, S)
    epsilon = rng.standard_normal(n)                       # epsilon ~ N(0, 1)
    y = X @ w_star + epsilon

    return X, y


def mse(w, X, y):
    """Mean squared error  (1/n) * ||y - Xw||^2  at parameter vector w."""
    residuals = y - X @ w
    return float(np.mean(residuals ** 2))


def grad_mse(w, X, y):
    """Gradient of MSE with respect to w:  -(2/n) X^T (y - Xw)."""
    n = len(y)
    residuals = y - X @ w
    return (-2.0 / n) * (X.T @ residuals)


def gradient_descent(X, y, eta: float, n_steps: int = 100, w_init=None):
    """
    Minimize MSE by gradient descent.

    Parameters
    ----------
    X       : ndarray (n, 2)
    y       : ndarray (n,)
    eta     : learning rate
    n_steps : number of GD iterations
    w_init  : initial weight vector, default [0.0, 0.0]

    Returns
    -------
    trajectory : ndarray (n_steps+1, 2)  – w at every step (including w_0)
    losses     : ndarray (n_steps+1,)    – MSE at every step
    """
    w = np.zeros(2) if w_init is None else np.asarray(w_init, dtype=float).copy()

    trajectory = [w.copy()]
    losses = [mse(w, X, y)]

    for _ in range(n_steps):
        w = w - eta * grad_mse(w, X, y)
        trajectory.append(w.copy())
        losses.append(mse(w, X, y))

    return np.array(trajectory), np.array(losses)


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression GD demo")
    parser.add_argument("--n",     type=int,   default=200,  help="Number of samples")
    parser.add_argument("--k",     type=float, default=10.0, help="Condition number (k > 1)")
    parser.add_argument("--eta",   type=float, default=None, help="Learning rate (default 0.9/k)")
    parser.add_argument("--steps", type=int,   default=100,  help="GD iterations")
    args = parser.parse_args()

    w_star = np.array([1.0, 1.0])
    print(f"True w* = {w_star}")

    X, y = generate_data(args.n, args.k, w_star=w_star)

    # Compute top eigenvalue (largest) of X.T @ X as efficient learning rate suggestion
    XT_X = X.T @ X
    # Since XT_X is symmetric and positive definite, its spectral norm is its largest eigenvalue
    eigval_max = np.linalg.eigvalsh(XT_X).max()
    optimal_eta = .9 / eigval_max
    print(f"Suggested learning rate (2 / top eigenvalue): {optimal_eta:.6f}")
    eta    = args.eta if args.eta is not None else optimal_eta
    print(f"n={args.n}, k={args.k}, eta={eta:.6f}, steps={args.steps}")

    trajectory, losses = gradient_descent(X, y, eta=eta, n_steps=args.steps)

    print(f"\nInitial MSE : {losses[0]:.4f}")
    print(f"Final MSE   : {losses[-1]:.6f}")
    print(f"Final w     : [{trajectory[-1, 0]:.4f}, {trajectory[-1, 1]:.4f}]")
    print(f"True w*     : [1.0000, 1.0000]")
