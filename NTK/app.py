"""
NTK Width Experiment Dashboard
Visualizes finite-width MLP (both layers trained via GD) converging to the
infinite-width NTK kernel prediction.
"""

from flask import Flask, render_template, request, Response
import numpy as np
import json

from ntk_computation import (sample_data, eval_trial, ntk_kernel_matrix, SEED)

app = Flask(__name__)


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

    # Precompute analytic NTK kernel matrices (width-independent)
    K_train = ntk_kernel_matrix(X_train, X_train)
    K_test = ntk_kernel_matrix(X_train, X_test)

    widths = np.unique(np.geomspace(2, max_width, n_points).astype(int))

    def generate():
        yield f"data: {json.dumps({'type': 'init'})}\n\n"

        for k in widths:
            k = int(k)
            mlp_mses, ntk_mses = [], []
            for t in range(n_trials):
                mlp_m, ntk_m = eval_trial(
                    X_train, y_train, X_test, y_test, k, seed=t,
                    K_train=K_train, K_test=K_test)
                mlp_mses.append(mlp_m)
                ntk_mses.append(ntk_m)

            mlp_mses = np.array(mlp_mses)
            ntk_mses = np.array(ntk_mses)
            payload = {
                'type': 'point',
                'width': k,
                'mlp_median': float(np.median(mlp_mses)),
                'mlp_q25': float(np.percentile(mlp_mses, 25)),
                'mlp_q75': float(np.percentile(mlp_mses, 75)),
                'ntk_median': float(np.median(ntk_mses)),
                'ntk_q25': float(np.percentile(ntk_mses, 25)),
                'ntk_q75': float(np.percentile(ntk_mses, 75)),
            }
            yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  NTK Width Experiment Dashboard")
    print("  Open http://localhost:5052 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5052)
