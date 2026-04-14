"""
Train RFM (Recursive Feature Machine) on modular-arithmetic Cayley tables.

Uses kernel ridge regression with iterative AGOP (Average Gradient Outer
Product) feature-metric updates to classify modular arithmetic operations.
Inputs are concatenated one-hot vectors of length 2n; targets are one-hot
encoded results of length n.

Supported kernels: gaussian, laplace, quadratic.
The quadratic kernel K(x,c) = (x^T M c)^2 uses a custom AGOP computation.
"""

import numpy as np
from numpy.linalg import solve, eigh


# ── Modular arithmetic ──────────────────────────────────────────────

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
      table   : (n, n) int array; invalid cells are -1.
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


# ── Kernel functions ────────────────────────────────────────────────

def get_norm(x, M=None):
    x2 = x * x if M is None else (x @ M) * x
    return np.sum(x2, axis=1, keepdims=True)


def euclidean_distances(samples, centers, M=None, squared=True, threshold=None):
    samples_norm = get_norm(samples, M)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = get_norm(centers, M)
    if M is not None:
        distances = samples @ (M @ centers.T)
    else:
        distances = samples @ centers.T
    distances = -2 * distances + samples_norm + centers_norm.T
    if threshold is not None:
        distances[distances < threshold] = 0
    if not squared:
        distances = np.sqrt(distances)
    return distances


def gaussian_kernel(samples, centers, bandwidth, M=None):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, M=M, squared=True, threshold=0)
    kernel_mat = np.exp(-kernel_mat / (2 * bandwidth ** 2))
    return kernel_mat


def laplace_kernel(samples, centers, bandwidth, M=None):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, M=M, squared=False, threshold=0)
    kernel_mat = np.exp(-kernel_mat / bandwidth)
    return kernel_mat


def quadratic_kernel(samples, centers, bandwidth, M=None):
    """Degree-2 polynomial kernel: K(x,c) = (x^T M c)^2.

    Bandwidth is accepted for interface consistency but not used.
    """
    if M is not None:
        inner = samples @ M @ centers.T
    else:
        inner = samples @ centers.T
    return inner ** 2


def get_kernel_fn(name):
    if name == "laplace":
        return laplace_kernel
    elif name == "gaussian":
        return gaussian_kernel
    elif name == "quadratic":
        return quadratic_kernel
    raise ValueError(f"Unknown kernel: {name}")


# ── Gradient and AGOP computation ───────────────────────────────────

def get_grads(X, sol, bandwidth, M, kernel_name="laplace",
              max_num_samples=20000, centering=True):
    """Compute the gradient tensor G of shape (m, c, d).

    Dispatches to a custom implementation for the quadratic kernel.
    """
    if kernel_name == "quadratic":
        return _get_grads_quadratic(X, sol, M, max_num_samples, centering)

    if len(X) > max_num_samples:
        indices = np.random.randint(len(X), size=max_num_samples)
        x = X[indices, :]
    else:
        x = X

    n, d = X.shape
    m = len(x)
    c = sol.shape[0]

    if kernel_name == "laplace":
        K = laplace_kernel(X, x, bandwidth, M=M)
        dist = euclidean_distances(X, x, M=M, squared=False, threshold=1e-10)
        with np.errstate(divide='ignore'):
            K = K / dist
        K[K == float("Inf")] = 0.0
        factor = -1.0 / bandwidth
    elif kernel_name == "gaussian":
        K = gaussian_kernel(X, x, bandwidth, M=M)
        factor = -1.0 / (bandwidth ** 2)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    a1 = sol.T.reshape(n, c, 1)
    X1 = (X @ M).reshape(n, 1, d)
    step1 = (a1 @ X1).reshape(-1, c * d)
    del a1, X1

    step2 = (K.T @ step1).reshape(-1, c, d)
    del step1

    step3 = ((sol @ K).T).reshape(m, c, 1)
    x1 = (x @ M).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * factor
    if centering:
        G -= np.mean(G, axis=0, keepdims=True)
    return G


def _get_grads_quadratic(X, sol, M, max_num_samples=20000, centering=True):
    """Gradient tensor for the quadratic kernel K(x,c) = (x^T M c)^2.

    Derivation
    ----------
    dK(X_i, x_j)/dx_j = 2 (X_i^T M x_j) M X_i

    So for output k:
        G[j, k, :] = 2 sum_i sol[k,i] (X_i^T M x_j) (X_i @ M)

    Vectorised as:
        step1 = outer(sol.T, X@M)   -> (n, c*d)
        step2 = inner.T @ step1     -> (m, c*d)
        G     = 2 * step2
    where inner[i,j] = X_i^T M x_j (the linear inner products, not K).
    """
    if len(X) > max_num_samples:
        indices = np.random.randint(len(X), size=max_num_samples)
        x = X[indices, :]
    else:
        x = X

    n, d = X.shape
    m = len(x)
    c = sol.shape[0]

    XM = X @ M
    inner = XM @ x.T  # (n, m)

    a1 = sol.T.reshape(n, c, 1)
    X1 = XM.reshape(n, 1, d)
    step1 = (a1 @ X1).reshape(-1, c * d)
    del a1, X1

    step2 = (inner.T @ step1).reshape(-1, c, d)
    del step1

    G = 2.0 * step2
    if centering:
        G -= np.mean(G, axis=0, keepdims=True)
    return G


def agop(G):
    """Compute the Average Gradient Outer Product (AGOP)."""
    M = 0.0
    chunks = len(G) // 20 + 1
    for batch in np.array_split(G, chunks):
        batchT = np.swapaxes(batch, 1, 2)
        M += np.sum(batchT @ batch, axis=0)
    M /= len(G)
    return M


def matrix_power(M, alpha):
    """Compute M^alpha via eigendecomposition."""
    if alpha == 1.0:
        return M
    eigvals, eigvecs = eigh(M)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals ** alpha) @ eigvecs.T


# ── Main training function ──────────────────────────────────────────

def train(operation='add', n=7, kernel='gaussian', reg=1e-3, bandwidth=1.0,
          num_iters=5, alpha=1.0, train_frac=0.5, seed=0,
          progress_callback=None):
    """RFM training on a modular-arithmetic Cayley table.

    Returns a dict with the training history (including AGOP matrices M),
    the Cayley-table data, and train/test masks.
    """
    pairs, results, table = build_cayley_table(operation, n)

    rng = np.random.RandomState(seed)
    N = pairs.shape[0]
    perm = rng.permutation(N)
    n_train = int(round(train_frac * N))
    n_train = max(1, min(N - 1, n_train)) if N > 1 else N

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_all = encode_inputs(pairs, n)
    Y_all = encode_targets(results, n)

    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]
    X_test = X_all[test_idx]
    Y_test = Y_all[test_idx]

    train_pairs = pairs[train_idx]
    test_pairs = pairs[test_idx]
    train_mask = np.zeros((n, n), dtype=np.int8)
    test_mask = np.zeros((n, n), dtype=np.int8)
    for a, b in train_pairs:
        train_mask[a, b] = 1
    for a, b in test_pairs:
        test_mask[a, b] = 1

    d = 2 * n
    M = np.eye(d)
    kernel_fn = get_kernel_fn(kernel)
    history = []

    for it in range(num_iters + 1):
        K_train = kernel_fn(X_train, X_train, bandwidth, M=M)
        sol = solve(K_train + reg * np.eye(n_train), Y_train).T

        train_pred = (sol @ K_train).T
        train_acc = float(np.mean(
            np.argmax(train_pred, axis=1) == np.argmax(Y_train, axis=1)
        ))
        train_mse = float(np.mean((train_pred - Y_train) ** 2))

        if X_test.shape[0] > 0:
            K_test = kernel_fn(X_train, X_test, bandwidth, M=M)
            test_pred = (sol @ K_test).T
            test_acc = float(np.mean(
                np.argmax(test_pred, axis=1) == np.argmax(Y_test, axis=1)
            ))
            test_mse = float(np.mean((test_pred - Y_test) ** 2))
        else:
            test_acc = 1.0
            test_mse = 0.0

        ckpt = {
            'iteration': it,
            'train_loss': train_mse,
            'test_loss': test_mse,
            'train_error': 1.0 - train_acc,
            'test_error': 1.0 - test_acc,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'M': M.copy(),
        }
        history.append(ckpt)

        if progress_callback is not None:
            progress_callback(it, num_iters, ckpt)

        if it == num_iters:
            break

        G = get_grads(X_train, sol, bandwidth, M, kernel_name=kernel)
        M = agop(G)
        if alpha != 1.0:
            M = matrix_power(M, alpha)
        max_abs = np.max(np.abs(M))
        if max_abs > 0:
            M = M / max_abs

    return {
        'history': history,
        'table': table,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'n': n,
        'operation': operation,
        'n_train': int(n_train),
        'n_test': int(N - n_train),
    }
