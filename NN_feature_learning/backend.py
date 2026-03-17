"""
Algorithm code for NN Feature Learning webapp.
Data generation, kernel regression, and neural network training.
Based on: https://github.com/aradha/6.S088_2023/blob/main/Module3_Notebooks/NN_Low_Rank_Poly.ipynb
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from numpy.linalg import solve
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def sample_data(num_samples, d, active_indices, func_type="product", seed=None):
    """Generate multi-index model data: y = f(x_{i1}, x_{i2}, ...)"""
    if seed is not None:
        np.random.seed(seed)
    X = np.random.normal(size=(num_samples, d))
    coords = [X[:, i] for i in active_indices]

    if func_type == "product":
        y = np.ones(num_samples)
        for c in coords:
            y = y * c
    elif func_type == "sum_of_squares":
        y = np.zeros(num_samples)
        for c in coords:
            y = y + c ** 2
    elif func_type == "sum":
        y = np.zeros(num_samples)
        for c in coords:
            y = y + c
    elif func_type == "squared_sum":
        s = np.zeros(num_samples)
        for c in coords:
            s = s + c
        y = s ** 2
    elif func_type == "cubic":
        y = np.zeros(num_samples)
        for c in coords:
            y = y + c ** 3
    else:
        y = np.ones(num_samples)
        for c in coords:
            y = y * c

    return X, y.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Kernel regression
# ---------------------------------------------------------------------------

def _relu_ntk_matrix(X1, X2):
    """
    Compute the infinite-width NTK for a 1-hidden-layer ReLU network
    f(x) = W2 * relu(W1 * x + b)  with bias in the first layer.

    The bias b ~ N(0,1) is equivalent to appending a 1 to each input,
    i.e. computing the NTK on [x; 1]. So we augment the inputs internally.

    K_NTK = Sigma0 + Sigma1_w + Sigma1_b

    where:
      Sigma0   = NNGP kernel (W2 contribution)
      Sigma1_w = W1 weight contribution: x_aug^T x_aug' * (pi - theta) / (2pi)
      Sigma1_b is already folded into Sigma1_w via the augmented input.
    """
    # Augment inputs with a column of 1s to account for bias
    ones1 = np.ones((X1.shape[0], 1))
    ones2 = np.ones((X2.shape[0], 1))
    X1a = np.concatenate([X1, ones1], axis=1)
    X2a = np.concatenate([X2, ones2], axis=1)

    # Norms of augmented inputs
    norms1 = np.sqrt(np.sum(X1a ** 2, axis=1, keepdims=True))
    norms2 = np.sqrt(np.sum(X2a ** 2, axis=1, keepdims=True))
    norm_prod = norms1 @ norms2.T

    dot = X1a @ X2a.T

    # Cosine of angle, clipped for numerical stability
    cos_angle = np.clip(dot / (norm_prod + 1e-12), -1.0, 1.0)
    theta = np.arccos(cos_angle)

    # NNGP kernel (arc-cosine k=1) — W2 contribution
    sigma0 = (1.0 / (2.0 * np.pi)) * norm_prod * (np.sin(theta) + (np.pi - theta) * cos_angle)

    # W1 + bias contribution (arc-cosine k=0 on augmented input)
    sigma1 = dot * (np.pi - theta) / (2.0 * np.pi)

    return sigma0 + sigma1


def kernel_regression_ntk(X_train, y_train, X_test, y_test, reg=1e-3):
    """Kernel regression using the infinite-width NTK of our architecture."""
    K_train = _relu_ntk_matrix(X_train, X_train)
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

    train_preds = (sol @ K_train).T
    train_r2 = float(r2_score(y_train, train_preds))

    K_test = _relu_ntk_matrix(X_train, X_test)
    test_preds = (sol @ K_test).T
    test_r2 = float(r2_score(y_test, test_preds))

    return train_r2, test_r2


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """Single hidden layer ReLU network: x -> Linear(d, width, bias) -> ReLU -> Linear(width, 1)."""
    def __init__(self, input_dim, output_dim, width=128):
        super().__init__()
        self.first = nn.Linear(input_dim, width, bias=True)
        self.last = nn.Linear(width, output_dim, bias=False)

    def forward(self, x):
        return self.last(torch.relu(self.first(x)))


def train_network(X_train, y_train, X_test, y_test,
                  width=128, init_scale=None, lr=0.1, num_epochs=50,
                  train_layers="both", callback=None):
    """
    Train a 1-hidden-layer ReLU network.
    init_scale=None uses default PyTorch init (Kaiming uniform).
    train_layers: "both", "first", or "last".
    Returns (net, train_losses, test_losses, train_r2, test_r2, w1tw1).
    """
    X_tr = torch.from_numpy(X_train).double()
    y_tr = torch.from_numpy(y_train).double()
    X_te = torch.from_numpy(X_test).double()
    y_te = torch.from_numpy(y_test).double()

    trainset = list(zip(X_tr, y_tr))
    testset = list(zip(X_te, y_te))
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    net = Net(input_dim, output_dim, width=width).double()

    # Custom init for first layer only if init_scale is specified
    if init_scale is not None:
        with torch.no_grad():
            net.first.weight.copy_(torch.randn_like(net.first.weight) * init_scale)

    # Freeze layers based on train_layers setting
    if train_layers == "first":
        net.last.weight.requires_grad_(False)
    elif train_layers == "last":
        for p in net.first.parameters():
            p.requires_grad_(False)

    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    train_losses = []
    test_losses = []
    diverged = False

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
        train_loss /= len(trainset)

        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                output = net(inputs)
                loss = criterion(output, targets)
                test_loss += loss.item() * len(inputs)
        test_loss /= len(testset)

        # Check for divergence
        if math.isnan(train_loss) or math.isinf(train_loss):
            print(f"Training diverged at epoch {epoch + 1} (train_loss={train_loss}). Stopping.")
            diverged = True
            if callback:
                callback(epoch + 1, float("nan"), float("nan"), error="Training diverged (NaN loss). Try a smaller learning rate.")
            break

        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

        if callback:
            callback(epoch + 1, float(train_loss), float(test_loss))

    # Final metrics
    if diverged or len(train_losses) == 0:
        return net, train_losses, test_losses, None, None, None

    net.eval()
    with torch.no_grad():
        train_preds = net(X_tr).numpy()
        test_preds = net(X_te).numpy()

    train_r2 = float(r2_score(y_train, train_preds))
    test_r2 = float(r2_score(y_test, test_preds))

    # W1^T W1
    W1 = net.first.weight.data.numpy()
    w1tw1 = W1.T @ W1

    return net, train_losses, test_losses, train_r2, test_r2, w1tw1
