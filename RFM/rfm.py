import numpy as np
from numpy.linalg import solve, eigh


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


def laplace_kernel(samples, centers, bandwidth, M=None):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, M=M, squared=False, threshold=0)
    kernel_mat = np.exp(-kernel_mat / bandwidth)
    return kernel_mat


def gaussian_kernel(samples, centers, bandwidth, M=None):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, M=M, squared=True, threshold=0)
    kernel_mat = np.exp(-kernel_mat / (2 * bandwidth ** 2))
    return kernel_mat


def get_kernel_fn(name):
    if name == "laplace":
        return laplace_kernel
    elif name == "gaussian":
        return gaussian_kernel
    raise ValueError(f"Unknown kernel: {name}")


def get_grads(X, sol, bandwidth, M, kernel_name="laplace",
              max_num_samples=20000, centering=True):
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


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


class RFM:
    """Recursive Feature Machine.

    Parameters
    ----------
    kernel : str
        'laplace' or 'gaussian'.
    """

    def __init__(self, kernel="laplace"):
        self.kernel = kernel
        self.X_train = None
        self.alphas = None
        self.M = None
        self.bandwidth = None
        self.reg = None
        self.history = []

    def fit(self, X_train, y_train, *, reg=1e-3, bandwidth=10.0,
            num_iters=5, alpha=1.0, M=None, centering=True,
            X_test=None, y_test=None, progress_callback=None):
        """Fit RFM iteratively.

        Parameters
        ----------
        reg : float
            Regularization for kernel ridge regression.
        bandwidth : float
            Kernel bandwidth parameter.
        num_iters : int
            Number of AGOP iterations.
        alpha : float
            Matrix power applied to the AGOP at each iteration.
        M : ndarray or None
            Initial feature metric (defaults to identity).
        X_test, y_test : ndarray or None
            If provided, test R^2 is tracked per iteration.
        progress_callback : callable or None
            Called as progress_callback(iteration, total_iterations, stage)
            where stage is 'solving' or 'agop'.
        """
        self.X_train = X_train
        self.history = []
        n, d = X_train.shape

        if M is None:
            M = np.eye(d)
        self.M = M
        self.bandwidth = bandwidth
        self.reg = reg

        kernel_fn = get_kernel_fn(self.kernel)

        for it in range(num_iters + 1):
            if progress_callback:
                progress_callback(it, num_iters, 'solving')

            K_train = kernel_fn(X_train, X_train, bandwidth, M=M)
            sol = solve(K_train + reg * np.eye(n), y_train).T
            self.alphas = sol

            train_pred = (sol @ K_train).T
            train_r2 = float(r2_score(y_train, train_pred))

            test_r2 = None
            if X_test is not None and y_test is not None:
                K_test = kernel_fn(X_train, X_test, bandwidth, M=M)
                test_pred = (sol @ K_test).T
                test_r2 = float(r2_score(y_test, test_pred))

            self.history.append({
                'iteration': it,
                'M': M.copy(),
                'train_r2': train_r2,
                'test_r2': test_r2,
            })

            if it == num_iters:
                break

            if progress_callback:
                progress_callback(it, num_iters, 'agop')

            G = get_grads(X_train, sol, bandwidth, M,
                          kernel_name=self.kernel, centering=centering)
            M = agop(G)
            if alpha != 1.0:
                M = matrix_power(M, alpha)
            M = M / np.max(np.abs(M))
            self.M = M

        return self

    def predict(self, X_test):
        kernel_fn = get_kernel_fn(self.kernel)
        K = kernel_fn(self.X_train, X_test, self.bandwidth, M=self.M)
        return (self.alphas @ K).T

    def get_M(self):
        return self.M

    def get_history(self):
        return self.history
