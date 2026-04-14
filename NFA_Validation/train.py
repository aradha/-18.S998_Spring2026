"""
Train a 1-hidden-layer ReLU MLP on multi-index model data.

    f(x) = a^T ReLU(Bx)

- Balanced initialization: a_i^2 = ||B_{i:}||_2^2
- Full-batch gradient descent
- Tracks B^TB and AGOP(f) during training
"""

import numpy as np


MODELS = {
    'quadratic_2d': {
        'name': 'Quadratic: (x\u2081 + x\u2082)\u00b2',
        'fn': lambda X: (X[:, 0] + X[:, 1]) ** 2,
        'relevant_dims': [0, 1],
    },
    'product_2d': {
        'name': 'Product: x\u2081 \u00b7 x\u2082',
        'fn': lambda X: X[:, 0] * X[:, 1],
        'relevant_dims': [0, 1],
    },
    'sinusoidal_2d': {
        'name': 'Sinusoidal: sin(x\u2081 + x\u2082)',
        'fn': lambda X: np.sin(X[:, 0] + X[:, 1]),
        'relevant_dims': [0, 1],
    },
    'abs_3d': {
        'name': 'Absolute: |x\u2081 + x\u2082 + x\u2083|',
        'fn': lambda X: np.abs(X[:, 0] + X[:, 1] + X[:, 2]),
        'relevant_dims': [0, 1, 2],
    },
    'gaussian_bump': {
        'name': 'Gaussian Bump: exp(\u2212(x\u2081\u00b2 + x\u2082\u00b2))',
        'fn': lambda X: np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2)),
        'relevant_dims': [0, 1],
    },
    'quadratic_linear_2d': {
        'name': 'Quad-Linear: x\u2081\u00b2 + 2x\u2082',
        'fn': lambda X: X[:, 0] ** 2 + 2 * X[:, 1],
        'relevant_dims': [0, 1],
    },
}


def matrix_sqrt(M):
    """Compute M^{1/2} via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def entrywise_correlation(A, B):
    """Pearson correlation between flattened entries of two matrices."""
    a, b = A.ravel(), B.ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class ReLUMLP:
    """1-hidden-layer ReLU MLP: f(x) = a^T ReLU(Bx)

    Balanced initialization: a_i^2 = ||B_{i:}||_2^2
    """

    def __init__(self, d, width, init_scale=0.5, seed=42):
        rng = np.random.RandomState(seed)
        # Initialize B with orthogonal rows, scaled by init_scale
        G = rng.randn(width, d)
        if width <= d:
            Q, _ = np.linalg.qr(G.T)  # (d, width) with orthonormal columns
            self.B = Q.T * init_scale   # (width, d) with orthogonal rows
        else:
            # More neurons than dimensions: fill in blocks of d orthogonal rows
            rows = []
            for start in range(0, width, d):
                block_size = min(d, width - start)
                G_block = rng.randn(block_size, d)
                Q, _ = np.linalg.qr(G_block.T)
                rows.append(Q[:, :block_size].T * init_scale)
            self.B = np.vstack(rows)
        self.b = np.zeros(width)
        norms = np.linalg.norm(self.B, axis=1)
        signs = rng.choice([-1.0, 1.0], size=width)
        self.a = signs * norms
        self.d = d
        self.width = width
        self._Z = None

    def forward(self, X):
        """Forward pass. X: (n, d) -> (n,)"""
        self._Z = X @ self.B.T + self.b[None, :]
        H = np.maximum(self._Z, 0)
        return H @ self.a

    def compute_gradients(self, X, y, y_hat):
        """Gradients for MSE loss = (1/n)||y - y_hat||^2."""
        n = X.shape[0]
        delta = 2.0 * (y_hat - y) / n
        H = np.maximum(self._Z, 0)
        grad_a = H.T @ delta
        mask = (self._Z > 0).astype(np.float64)
        grad_z = (delta[:, None] * self.a[None, :]) * mask
        grad_B = grad_z.T @ X
        grad_b = grad_z.sum(axis=0)
        return grad_a, grad_B, grad_b

    def compute_agop(self, X):
        """AGOP of f at points X: (1/n) sum_i grad_x f(x_i) grad_x f(x_i)^T"""
        Z = X @ self.B.T + self.b[None, :]
        masks = (Z > 0).astype(np.float64)
        weighted = masks * self.a[None, :]
        grads = weighted @ self.B  # (n, d)
        return grads.T @ grads / X.shape[0]

    def get_BtB(self):
        return self.B.T @ self.B


def train(model_key, width=200, init_scale=0.5, lr=0.01,
          num_epochs=5000, d=10, n_train=500, n_test=200,
          track_every=50, seed=42, progress_callback=None):
    """Train a 1-hidden-layer ReLU MLP on multi-index model data.

    Returns (history, net) where history is a list of checkpoint dicts.
    """
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, d)
    X_test = rng.randn(n_test, d)
    y_train = MODELS[model_key]['fn'](X_train)
    y_test = MODELS[model_key]['fn'](X_test)

    net = ReLUMLP(d, width, init_scale, seed)
    history = []

    for epoch in range(num_epochs + 1):
        y_hat = net.forward(X_train)
        loss = float(np.mean((y_hat - y_train) ** 2))

        if np.isnan(loss) or loss > 1e15:
            break

        should_track = (epoch % track_every == 0) or (epoch == num_epochs)
        if should_track:
            y_hat_test = np.maximum(X_test @ net.B.T + net.b[None, :], 0) @ net.a
            test_loss = float(np.mean((y_hat_test - y_test) ** 2))

            ss_res_train = float(np.sum((y_hat - y_train) ** 2))
            ss_tot_train = float(np.sum((y_train - y_train.mean()) ** 2))
            train_r2 = 1.0 - ss_res_train / (ss_tot_train + 1e-12)

            ss_res_test = float(np.sum((y_hat_test - y_test) ** 2))
            ss_tot_test = float(np.sum((y_test - y_test.mean()) ** 2))
            test_r2 = 1.0 - ss_res_test / (ss_tot_test + 1e-12)

            BtB = net.get_BtB()
            agop_mat = net.compute_agop(X_train)
            agop_sqrt = matrix_sqrt(agop_mat)

            checkpoint = {
                'epoch': epoch,
                'train_loss': loss,
                'test_loss': test_loss,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'BtB': BtB,
                'agop': agop_mat,
                'agop_sqrt': agop_sqrt,
                'corr_agop_BtB': entrywise_correlation(agop_mat, BtB),
                'corr_agopsqrt_BtB': entrywise_correlation(agop_sqrt, BtB),
            }
            history.append(checkpoint)

            if progress_callback:
                progress_callback(epoch, num_epochs, checkpoint)

        if epoch < num_epochs:
            grad_a, grad_B, grad_b = net.compute_gradients(X_train, y_train, y_hat)
            net.a -= lr * grad_a
            net.B -= lr * grad_B
            net.b -= lr * grad_b

    return history, net
