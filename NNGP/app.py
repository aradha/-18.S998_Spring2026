"""
NNGP Width Experiment Dashboard
Dynamically visualizes finite-width MLP test MSE converging to the NNGP kernel limit.
"""

from flask import Flask, render_template, request, Response
import numpy as np
from scipy.linalg import solve
import json

app = Flask(__name__)

SEED = 42

# =============================================================================
# Data
# =============================================================================

def sample_data(n, seed=SEED):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = np.sin(X[:, 0] * X[:, 1] * X[:, 4]) + np.cos(2 * X[:, 2]) + 5 * np.sin(X[:, 3])
    return X, y

# =============================================================================
# Finite-width MLP (last layer solve)
# =============================================================================

def eval_last_layer_mlp(X_train, y_train, X_test, y_test, k, seed, reg=1e-8):
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]
    B = rng.randn(k, d)
    scale = 2.0 / np.sqrt(k)

    H_train = np.maximum(0, X_train @ B.T) * scale
    H_test  = np.maximum(0, X_test  @ B.T) * scale

    a = solve(H_train.T @ H_train + reg * np.eye(k), H_train.T @ y_train)
    test_pred = H_test @ a
    return float(np.mean((test_pred - y_test) ** 2))

# =============================================================================
# NNGP kernel
# =============================================================================

def nngp_kernel_matrix(X1, X2):
    norms1 = np.linalg.norm(X1, axis=1)
    norms2 = np.linalg.norm(X2, axis=1)
    dots = X1 @ X2.T
    outer_norms = np.outer(norms1, norms2)
    rho = np.clip(dots / (outer_norms + 1e-30), -1.0, 1.0)
    radical = np.sqrt(np.clip(outer_norms**2 - dots**2, 0, None))
    K = (1.0 / np.pi) * (dots * (np.pi - np.arccos(rho)) + radical)
    return K

def nngp_test_mse(X_train, y_train, X_test, y_test, reg=1e-10):
    K_train = nngp_kernel_matrix(X_train, X_train)
    K_test  = nngp_kernel_matrix(X_train, X_test)
    alpha = solve(K_train + reg * np.eye(len(K_train)), y_train)
    test_pred = K_test.T @ alpha
    return float(np.mean((test_pred - y_test) ** 2))

# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stream')
def stream():
    n_train = int(request.args.get('n_train', 32))
    n_test = int(request.args.get('n_test', 1000))
    n_trials = int(request.args.get('n_trials', 50))
    n_points = int(request.args.get('n_points', 100))
    max_width = int(request.args.get('max_width', 1024))

    n_train = max(2, min(200, n_train))
    n_test = max(100, min(5000, n_test))
    n_trials = max(5, min(200, n_trials))
    n_points = max(10, min(200, n_points))
    max_width = max(4, min(4096, max_width))

    X_train, y_train = sample_data(n_train)
    X_test, y_test = sample_data(n_test, seed=SEED + 1)

    # NNGP baseline
    nngp_mse = nngp_test_mse(X_train, y_train, X_test, y_test)

    widths = np.unique(np.geomspace(2, max_width, n_points).astype(int))

    def generate():
        # send NNGP baseline first
        yield f"data: {json.dumps({'type': 'nngp', 'nngp_mse': nngp_mse})}\n\n"

        for k in widths:
            k = int(k)
            mses = [eval_last_layer_mlp(X_train, y_train, X_test, y_test, k, seed=t)
                    for t in range(n_trials)]
            mses = np.array(mses)
            payload = {
                'type': 'point',
                'width': k,
                'median': float(np.median(mses)),
                'q25': float(np.percentile(mses, 25)),
                'q75': float(np.percentile(mses, 75)),
                'mean': float(np.mean(mses)),
            }
            yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  NNGP Width Experiment Dashboard")
    print("  Open http://localhost:5051 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5051)
