import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

SEED = 42
np.random.seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────

def sample_data(n):
    X = np.random.randn(n, 5)
    y = np.sin(X[:, 0] * X[:, 1] * X[:, 4]) + np.cos(2 * X[:, 2]) + 5 * np.sin(X[:, 3])
    return X, y

n_train, n_test = 32, 1000
X_train, y_train = sample_data(n_train)
X_test, y_test = sample_data(n_test)

# ── Last-layer MLP training ──────────────────────────────────────────────────
# f(x) = A @ (2/sqrt(k)) * relu(B @ x)
# B frozen (iid N(0,1)), A trained (init at 0) -- NTK parameterization

def eval_last_layer_mlp(X_train, y_train, X_test, y_test, k, seed, reg=1e-8):
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]
    B = rng.randn(k, d)                               # (k, d)
    scale = 2.0 / np.sqrt(k)

    H_train = np.maximum(0, X_train @ B.T) * scale    # (n_train, k)
    H_test  = np.maximum(0, X_test  @ B.T) * scale    # (n_test, k)

    a = solve(H_train.T @ H_train + reg * np.eye(k), H_train.T @ y_train)
    test_pred = H_test @ a
    return np.mean((test_pred - y_test) ** 2)

# ── NNGP kernel ──────────────────────────────────────────────────────────────
# Sigma(x, x') = (1/pi) * [ (x^T x')(pi - arccos(rho)) + sqrt(||x||^2 ||x'||^2 - (x^T x')^2) ]
# where rho = (x^T x') / (||x|| ||x'||)

def nngp_kernel_matrix(X1, X2):
    norms1 = np.linalg.norm(X1, axis=1)   # (n1,)
    norms2 = np.linalg.norm(X2, axis=1)   # (n2,)
    dots = X1 @ X2.T                       # (n1, n2)
    outer_norms = np.outer(norms1, norms2) # (n1, n2)

    # clamp rho to [-1, 1] for numerical safety
    rho = np.clip(dots / (outer_norms + 1e-30), -1.0, 1.0)

    # sqrt(||x||^2 ||x'||^2 - (x^T x')^2)  -- clamp inside for safety
    radical = np.sqrt(np.clip(outer_norms**2 - dots**2, 0, None))

    K = (1.0 / np.pi) * (dots * (np.pi - np.arccos(rho)) + radical)
    return K

def nngp_regression(X_train, y_train, X_test, y_test, reg=1e-10):
    K_train = nngp_kernel_matrix(X_train, X_train)
    K_test  = nngp_kernel_matrix(X_train, X_test)
    alpha = solve(K_train + reg * np.eye(len(K_train)), y_train)
    test_pred = K_test.T @ alpha
    return np.mean((test_pred - y_test) ** 2)

# ── Run experiments ──────────────────────────────────────────────────────────

widths = np.unique(np.geomspace(2, 1024, 100).astype(int))
n_trials = 50

test_mses = np.zeros((len(widths), n_trials))

for i, k in enumerate(widths):
    for t in range(n_trials):
        test_mses[i, t] = eval_last_layer_mlp(X_train, y_train, X_test, y_test, int(k), seed=t)
    print(f"width {k:5d}: test MSE = {test_mses[i].mean():.6f}")

# NNGP baseline
nngp_test_mse = nngp_regression(X_train, y_train, X_test, y_test)
print(f"NNGP:        test MSE = {nngp_test_mse:.6f}")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

median_test = np.median(test_mses, axis=1)
q25_test, q75_test = np.percentile(test_mses, [25, 75], axis=1)

ax.plot(widths, median_test, 'r-', label='Test MSE (MLP)', linewidth=2)
ax.fill_between(widths, q25_test, q75_test, color='r', alpha=0.15)

ax.axhline(nngp_test_mse, color='blue', linestyle='--', linewidth=2, label='Test MSE (NNGP)')

ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Width k', fontsize=13)
ax.set_ylabel('Test MSE', fontsize=13)
ax.set_title('Last-Layer ReLU MLP vs NNGP Kernel Regression', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nngp_width_experiment.png', dpi=150)
plt.show()
