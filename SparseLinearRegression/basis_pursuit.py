"""Basis Pursuit for Sparse Linear Regression
==========================================
Solves min ||w||_1 s.t. Xw = y for recovering sparse signals
from Gaussian measurements, and computes the scaling law n_min ~ C * s * log(ed/s).

Usage:
    python basis_pursuit.py              # Launch web app on localhost:5000
    python basis_pursuit.py --compute    # Compute and print scaling law

Requirements: numpy, cvxpy, flask
"""

import numpy as np
import cvxpy as cp
from flask import Flask, jsonify, request, send_file
import argparse
import os


# ═══════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════

def generate_sparse_problem(n, d, s, seed=42):
    """Generate y = Xw* where w* is s-sparse and X is i.i.d. N(0,1)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w_star = np.zeros(d)
    w_star[:s] = rng.standard_normal(s)
    y = X @ w_star
    return X, y, w_star


def solve_basis_pursuit(X, y):
    """Solve min ||w||_1 s.t. Xw = y."""
    n, d = X.shape
    w = cp.Variable(d)
    prob = cp.Problem(cp.Minimize(cp.norm(w, 1)), [X @ w == y])
    try:
        prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, max_iters=10000)
        except Exception:
            return None
    if w.value is not None:
        return np.asarray(w.value).flatten()
    return None


def solve_min_l2_norm(X, y):
    """Solve min ||w||_2 s.t. Xw = y via pseudoinverse."""
    n, d = X.shape
    if n <= d:
        return X.T @ np.linalg.solve(X @ X.T, y)
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ═══════════════════════════════════════════════════════════════════════
# Scaling law computation
# ═══════════════════════════════════════════════════════════════════════

def check_exact_recovery(n, d, s, seed=42, tol=1e-3):
    """Check if basis pursuit exactly recovers w* (relative error < tol)."""
    X, y, w_star = generate_sparse_problem(n, d, s, seed)
    w_bp = solve_basis_pursuit(X, y)
    if w_bp is None:
        return False
    return np.linalg.norm(w_bp - w_star) / (np.linalg.norm(w_star) + 1e-12) < tol


def find_min_samples(d, s, num_seeds=10, success_rate=0.9, tol=1e-3):
    """Binary search for minimum n achieving exact recovery on >= success_rate of seeds."""
    required = int(np.ceil(num_seeds * success_rate))

    def succeeds_at(n):
        count = 0
        for seed in range(num_seeds):
            if check_exact_recovery(n, d, s, seed, tol):
                count += 1
            if count >= required:
                return True
            if count + (num_seeds - seed - 1) < required:
                return False
        return count >= required

    lo, hi = max(1, s), d
    if not succeeds_at(hi):
        return hi

    while lo < hi:
        mid = (lo + hi) // 2
        if succeeds_at(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def compute_scaling_law(d=100, s_values=None, num_seeds=10):
    """Compute n_min vs s*log(ed/s) for varying sparsity levels."""
    if s_values is None:
        s_values = list(range(1, min(d // 2, 25) + 1))
    results = []
    for s in s_values:
        n_min = find_min_samples(d, s, num_seeds)
        x_val = s * np.log(np.e * d / s)
        results.append({'s': int(s), 'n_min': int(n_min), 'x_val': float(x_val)})
        print(f"  s={s:3d}  n_min={n_min:4d}  s*log(ed/s)={x_val:.1f}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Flask web server
# ═══════════════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'interactive_plot.html'))


@app.route('/api/solve', methods=['POST'])
def api_solve():
    data = request.json
    n, d, s = int(data['n']), int(data['d']), int(data['s'])
    seed = int(data.get('seed', 42))

    X, y, w_star = generate_sparse_problem(n, d, s, seed)
    w_l1 = solve_basis_pursuit(X, y)
    w_l2 = solve_min_l2_norm(X, y)
    norm_ws = float(np.linalg.norm(w_star)) + 1e-12

    result = {
        'w_star': w_star.tolist(),
        'w_l1': w_l1.tolist() if w_l1 is not None else None,
        'w_l2': w_l2.tolist(),
        'l2_error': float(np.linalg.norm(w_l2 - w_star)),
        'l2_rel_error': float(np.linalg.norm(w_l2 - w_star) / norm_ws),
    }
    if w_l1 is not None:
        result['l1_error'] = float(np.linalg.norm(w_l1 - w_star))
        result['l1_rel_error'] = float(np.linalg.norm(w_l1 - w_star) / norm_ws)
    return jsonify(result)


@app.route('/api/scaling_law', methods=['POST'])
def api_scaling_law():
    data = request.json
    d = int(data.get('d', 100))
    num_seeds = int(data.get('num_seeds', 5))
    s_max = int(data.get('s_max', 20))
    s_values = list(range(1, min(s_max, d // 2) + 1))
    return jsonify(compute_scaling_law(d, s_values, num_seeds))


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basis Pursuit Sparse Recovery')
    parser.add_argument('--compute', action='store_true',
                        help='Compute scaling law and print results (no web server)')
    parser.add_argument('--d', type=int, default=100, help='Ambient dimension (default: 100)')
    parser.add_argument('--port', type=int, default=5050, help='Web server port (default: 5050)')
    args = parser.parse_args()

    if args.compute:
        print(f"Computing scaling law for d={args.d}...\n")
        compute_scaling_law(d=args.d)
    else:
        print(f"Starting web app at http://localhost:{args.port}")
        app.run(debug=True, port=args.port)
