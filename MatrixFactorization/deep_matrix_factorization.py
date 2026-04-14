"""
Deep Linear Network Matrix Factorization
=========================================
Given a matrix with partially observed entries, train a deep linear network
    W_L @ W_{L-1} @ ... @ W_1
to reconstruct the observed entries, then predict the missing ones.

Architecture (for an n x m target matrix):
  - W_i for i in [1, L-1]: shape (m, m)
  - W_L: shape (n, m)

Initialization:
  - W_i for i in [1, L-1]: alpha * I_m
  - W_L: balanced with W_{L-1} so that W_L^T W_L = W_{L-1} W_{L-1}^T at init
    Since W_{L-1} = alpha * I_m, we need W_L^T W_L = alpha^2 * I_m,
    so W_L is initialized via SVD-based balancing from an (n x m) matrix.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepLinearMF(nn.Module):
    """Deep linear network for matrix factorization."""

    def __init__(self, n: int, m: int, depth: int, alpha: float = 0.01):
        super().__init__()
        assert depth >= 1, "Depth must be at least 1"
        self.n = n
        self.m = m
        self.depth = depth
        self.alpha = alpha

        self.weights = nn.ParameterList()

        if depth == 1:
            # Single layer: W_1 is (n, m), init as alpha * truncated
            W = alpha * torch.randn(n, m) / np.sqrt(max(n, m))
            self.weights.append(nn.Parameter(W))
        else:
            # Layers 1 to L-1: (m, m), initialized as alpha * I_m
            for i in range(depth - 1):
                W = alpha * torch.eye(m)
                self.weights.append(nn.Parameter(W))

            # Layer L: (n, m), balanced with W_{L-1}
            # We need W_L^T W_L = W_{L-1} W_{L-1}^T = alpha^2 * I_m
            # So W_L should have singular values all equal to alpha
            # Use a random orthogonal basis for the row space
            W_L = torch.zeros(n, m)
            if n >= m:
                Q, _ = torch.linalg.qr(torch.randn(n, m))
                W_L = alpha * Q  # (n, m) with W_L^T W_L = alpha^2 * I_m
            else:
                Q, _ = torch.linalg.qr(torch.randn(m, n))
                W_L = alpha * Q.T  # (n, m)
            self.weights.append(nn.Parameter(W_L))

    def forward(self):
        """Compute the product W_L @ W_{L-1} @ ... @ W_1."""
        result = self.weights[0]
        for i in range(1, self.depth):
            result = self.weights[i] @ result
        return result


def train_matrix_factorization(
    M: np.ndarray,
    mask: np.ndarray,
    depth: int = 3,
    alpha: float = 0.01,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    num_steps: int = 5000,
    device: str = "cpu",
    callback=None,
):
    """
    Train a deep linear network to factorize a partially observed matrix.

    Parameters
    ----------
    M : np.ndarray of shape (n, m)
        The target matrix (only observed entries matter).
    mask : np.ndarray of shape (n, m), dtype bool
        True where an entry is observed, False where missing.
    depth : int
        Number of layers in the deep linear network.
    alpha : float
        Initialization scale (W_i = alpha * I for intermediate layers).
    lr : float
        Learning rate.
    optimizer_name : str
        "adam" or "sgd".
    num_steps : int
        Number of gradient descent steps.
    device : str
        "cpu" or "cuda".
    callback : callable, optional
        Called every step with (step, model, train_loss) for live updates.

    Returns
    -------
    model : DeepLinearMF
        The trained model.
    train_losses : list of float
        Training loss at each step.
    M_pred : np.ndarray
        The full predicted matrix.
    """
    n, m = M.shape
    M_tensor = torch.tensor(M, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)

    model = DeepLinearMF(n, m, depth, alpha).to(device)

    if optimizer_name.lower() == "adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    train_losses = []

    for step in range(num_steps):
        opt.zero_grad()
        M_hat = model()
        # Loss only on observed entries
        diff = (M_hat - M_tensor) * mask_tensor
        loss = (diff ** 2).sum() / mask_tensor.sum()
        loss.backward()
        opt.step()

        train_losses.append(loss.item())

        if callback is not None:
            callback(step, model, loss.item())

    with torch.no_grad():
        M_pred = model().cpu().numpy()

    return model, train_losses, M_pred


if __name__ == "__main__":
    # Demo: random low-rank matrix completion
    np.random.seed(42)
    torch.manual_seed(42)

    n, m, rank = 10, 8, 3
    U = np.random.randn(n, rank)
    V = np.random.randn(rank, m)
    M_true = U @ V

    # Observe 50% of entries
    mask = np.random.rand(n, m) > 0.5

    model, losses, M_pred = train_matrix_factorization(
        M_true, mask, depth=3, alpha=0.01, lr=1e-3,
        optimizer_name="adam", num_steps=5000,
    )

    train_mse = np.mean((M_pred[mask] - M_true[mask]) ** 2)
    test_mse = np.mean((M_pred[~mask] - M_true[~mask]) ** 2)
    print(f"Train MSE (observed):  {train_mse:.6f}")
    print(f"Test  MSE (missing):   {test_mse:.6f}")
