"""
Linear Recursive Feature Machine (RFM) for Matrix Completion
=============================================================
Implements the linear RFM algorithm from https://github.com/aradha/lin-RFM
with a stateful class that supports step-by-step iteration.
"""

import numpy as np
from numpy.linalg import solve, svd


class LinearRFMSolver:
    """Stateful linear RFM solver that supports single-step iteration."""

    def __init__(self, Y, unmasked, reg=0.1, power=1.0, replace=True):
        self.n, self.m = Y.shape
        self.Y = Y.copy()
        self.unmasked = unmasked.copy()
        self.reg = reg
        self.power = power
        self.replace = replace

        self.M = np.eye(self.m)              # m×m feature matrix
        self.y = Y[unmasked].reshape(-1, 1)
        self.sol = np.zeros((self.n, self.m)) # n×m coefficients
        self.out = Y.copy()
        self.out[~unmasked] = 0.0

        self.iteration = 0
        self.best_error = np.inf
        self.prev_error = np.inf
        self.best_out = None
        self.patience = 0
        self.converged = False

    def step(self):
        """Run one iteration of linear RFM. Returns (out, error, iteration)."""
        if self.converged:
            return self.out, self.best_error, self.iteration

        n_obs = 0
        P = self.M.T @ self.M  # m×m

        for i in range(self.n):
            idxs = np.nonzero(self.unmasked[i])[0]
            A = self.M[:, idxs]               # m × |idxs|
            K = P[np.ix_(idxs, idxs)]         # |idxs| × |idxs|
            a = solve(K + np.eye(len(K)) * self.reg,
                      self.y[n_obs:n_obs + len(idxs)])
            self.sol[i, :] = (A @ a).T        # 1 × m
            n_obs += len(idxs)

        self.out = self.sol @ self.M          # n×m

        if self.replace:
            self.out[self.unmasked] = self.Y[self.unmasked]

        self.M = self.out.T @ self.out * (1.0 / self.n)  # m×m
        if self.power != 1:
            U, S, Vt = svd(self.M)
            S = np.where(S < 0.0, 0.0, S)
            S = np.power(S, self.power)
            self.M = U @ np.diag(S) @ Vt

        error = np.mean(np.square(
            self.Y[~self.unmasked] - self.out[~self.unmasked]
        ))

        if abs(error - self.prev_error) < 1e-10 or error > self.prev_error:
            self.patience += 1
        if error < self.best_error:
            self.best_out = self.out.copy()
            self.best_error = error
        self.prev_error = error

        if error < 1e-1:
            self.reg = min(self.reg, 1e-3)

        self.iteration += 1
        return self.out, error, self.iteration
