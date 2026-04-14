"""
Train a 1-hidden-layer MLP on a modular-arithmetic Cayley table.

Architecture (no biases):

    f(x) = W2 @ phi(W1 @ x)

where x is the concatenation of two one-hot vectors of length n
(representing a, b in Z_n) and the target is the one-hot encoding of
(a op b) mod n for op in {add, sub, mul, div}. Multiplication and
division require n to be prime.

Optimization: AdamW + MSE loss + mini-batches, standard PyTorch init.
"""

import math
import numpy as np
import torch
import torch.nn as nn


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _modinv(b, n):
    return pow(int(b), -1, int(n))


OPERATIONS = {
    'add': {
        'name': 'Addition  (a + b) mod n',
        'symbol': '+',
        'requires_prime': False,
        'fn': lambda a, b, n: (a + b) % n,
        'a_range': lambda n: (0, n),
        'b_range': lambda n: (0, n),
    },
    'sub': {
        'name': 'Subtraction  (a \u2212 b) mod n',
        'symbol': '\u2212',
        'requires_prime': False,
        'fn': lambda a, b, n: (a - b) % n,
        'a_range': lambda n: (0, n),
        'b_range': lambda n: (0, n),
    },
    'mul': {
        'name': 'Multiplication  (a \u00b7 b) mod n',
        'symbol': '\u00b7',
        'requires_prime': True,
        'fn': lambda a, b, n: (a * b) % n,
        'a_range': lambda n: (0, n),
        'b_range': lambda n: (0, n),
    },
    'div': {
        'name': 'Division  (a / b) mod n',
        'symbol': '/',
        'requires_prime': True,
        'fn': lambda a, b, n: (a * _modinv(b, n)) % n,
        'a_range': lambda n: (0, n),
        'b_range': lambda n: (1, n),
    },
}


def build_cayley_table(operation, n):
    """Build the full Cayley table for the given op.

    Returns (pairs, results, table) where:
      pairs   : (N, 2) int array of valid (a, b) input pairs
      results : (N,)  int array of (a op b) mod n
      table   : (n, n) int array; entries outside the valid input domain
                are set to -1 so they can be drawn as "blank" cells.
    """
    spec = OPERATIONS[operation]
    if spec['requires_prime'] and not is_prime(n):
        raise ValueError(
            f"Operation '{operation}' requires n to be prime (got n={n})."
        )

    a_lo, a_hi = spec['a_range'](n)
    b_lo, b_hi = spec['b_range'](n)

    table = np.full((n, n), -1, dtype=np.int64)
    pairs, results = [], []
    for a in range(a_lo, a_hi):
        for b in range(b_lo, b_hi):
            r = int(spec['fn'](a, b, n))
            table[a, b] = r
            pairs.append((a, b))
            results.append(r)

    return np.array(pairs, dtype=np.int64), np.array(results, dtype=np.int64), table


def encode_inputs(pairs, n):
    """Concatenate two one-hot vectors of size n. Returns (N, 2n)."""
    N = pairs.shape[0]
    X = np.zeros((N, 2 * n), dtype=np.float64)
    X[np.arange(N), pairs[:, 0]] = 1.0
    X[np.arange(N), n + pairs[:, 1]] = 1.0
    return X


def encode_targets(results, n):
    """One-hot encode targets. Returns (N, n)."""
    N = results.shape[0]
    Y = np.zeros((N, n), dtype=np.float64)
    Y[np.arange(N), results] = 1.0
    return Y


class MLP(nn.Module):
    """1 hidden layer MLP with no biases.

        f(x) = W2 phi(W1 x)

    W1: (k, 2n), W2: (n, k). phi is ReLU or quadratic (z -> z**2).
    Uses standard PyTorch nn.Linear initialization (kaiming-uniform with
    a=sqrt(5)). `init_scale` is an optional multiplier applied after
    init; init_scale=1.0 leaves the standard initialization untouched.
    """

    def __init__(self, n, k, activation='relu', init_scale=1.0):
        super().__init__()
        if activation not in ('relu', 'quadratic'):
            raise ValueError(f"activation must be 'relu' or 'quadratic'; got {activation}")
        self.n = n
        self.k = k
        self.activation = activation
        self.fc1 = nn.Linear(2 * n, k, bias=False)
        self.fc2 = nn.Linear(k, n, bias=False)
        if init_scale != 1.0:
            with torch.no_grad():
                self.fc1.weight.mul_(init_scale)
                self.fc2.weight.mul_(init_scale)

    def forward(self, x):
        z = self.fc1(x)
        h = torch.relu(z) if self.activation == 'relu' else z * z
        return self.fc2(h)


def train(operation='add', n=7, k=1024, activation='relu',
          init_scale=1.0, lr=1e-3, num_epochs=2000, train_frac=0.5,
          weight_decay=1.0, batch_size=32, track_every=10, seed=0,
          device=None, progress_callback=None):
    """AdamW + MSE training on a modular-arithmetic Cayley table.

    Returns a dict with the training history, the network, and the
    Cayley-table data needed to visualize what was trained on.
    """
    pairs, results, table = build_cayley_table(operation, n)

    rng = np.random.RandomState(seed)
    N = pairs.shape[0]
    perm = rng.permutation(N)
    n_train = int(round(train_frac * N))
    n_train = max(1, min(N - 1, n_train)) if N > 1 else N

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_all = encode_inputs(pairs, n).astype(np.float32)
    Y_all = encode_targets(results, n).astype(np.float32)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    X_train = torch.from_numpy(X_all[train_idx]).to(device)
    Y_train = torch.from_numpy(Y_all[train_idx]).to(device)
    X_test = torch.from_numpy(X_all[test_idx]).to(device)
    Y_test = torch.from_numpy(Y_all[test_idx]).to(device)

    train_pairs = pairs[train_idx]
    test_pairs = pairs[test_idx]
    train_mask = np.zeros((n, n), dtype=np.int8)
    test_mask = np.zeros((n, n), dtype=np.int8)
    for a, b in train_pairs:
        train_mask[a, b] = 1
    for a, b in test_pairs:
        test_mask[a, b] = 1

    torch.manual_seed(seed)
    net = MLP(n=n, k=k, activation=activation, init_scale=init_scale).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    y_train_idx = Y_train.argmax(dim=1)
    y_test_idx = Y_test.argmax(dim=1) if X_test.shape[0] > 0 else None

    @torch.no_grad()
    def evaluate(epoch):
        net.eval()
        Y_pred_tr = net(X_train)
        train_loss = float(loss_fn(Y_pred_tr, Y_train).item())
        train_acc = float((Y_pred_tr.argmax(dim=1) == y_train_idx).float().mean().item())
        if X_test.shape[0] > 0:
            Y_pred_te = net(X_test)
            test_loss = float(loss_fn(Y_pred_te, Y_test).item())
            test_acc = float((Y_pred_te.argmax(dim=1) == y_test_idx).float().mean().item())
        else:
            test_loss = 0.0
            test_acc = 1.0
        W1 = net.fc1.weight.detach().cpu().numpy()
        W1tW1 = W1.T @ W1

        net.train()
        return {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_error': 1.0 - train_acc,
            'test_error': 1.0 - test_acc,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'W1tW1': W1tW1,
        }

    ckpt = evaluate(0)
    history.append(ckpt)
    if progress_callback is not None:
        progress_callback(0, num_epochs, ckpt)

    g = torch.Generator(device=device)
    g.manual_seed(seed + 1)

    for epoch in range(1, num_epochs + 1):
        net.train()
        idx = torch.randperm(n_train, generator=g, device=device)
        for start in range(0, n_train, batch_size):
            batch = idx[start:start + batch_size]
            xb = X_train[batch]
            yb = Y_train[batch]
            optimizer.zero_grad()
            yp = net(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            optimizer.step()

        if epoch % track_every == 0 or epoch == num_epochs:
            ckpt = evaluate(epoch)
            history.append(ckpt)
            if progress_callback is not None:
                progress_callback(epoch, num_epochs, ckpt)
            if not math.isfinite(ckpt['train_loss']) or ckpt['train_loss'] > 1e15:
                break

    return {
        'history': history,
        'net': net,
        'table': table,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'n': n,
        'operation': operation,
        'n_train': int(n_train),
        'n_test': int(N - n_train),
    }
