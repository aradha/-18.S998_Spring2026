"""
Interactive webapp for visualizing infinite-depth NNGP kernel regression
on the non-negative orthant of the unit circle.
"""

import io
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request
from scipy.stats import dirichlet as dirichlet_dist

from kernel_backend import (
    get_dirichlet_data, run_predictions,
    dirichlet_bayes_predict, dirichlet_bayes_grid,
    angular_density,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALPHA1 = np.array([2.0, 1.0])   # y = +1  — peaks at θ ≈ 0  (negative slope)
ALPHA2 = np.array([1.0, 2.0])   # y = -1  — peaks at θ ≈ π/2 (positive slope)
PRIOR1 = 0.5
SEED = 42

ACTIVATIONS = {
    "2d_opt":  "z⁷/2 + z/2  (2D optimal)",
    "3d_opt":  "z³/2 + z/2  (3D optimal)",
    "erf":     "arcsin(2z/3) / arcsin(2/3)",
    "relu":    "ReLU dual",
    "sine_1":  "sinh(z) / sinh(1)",
}

THETA_GRID = np.linspace(1e-4, np.pi / 2 - 1e-4, 500)

# ---------------------------------------------------------------------------
# Data cache  (regenerate when n changes)
# ---------------------------------------------------------------------------
_cache = {}


def _get_data(n_samples):
    if n_samples not in _cache:
        n_test = max(n_samples, 500)
        X_tr, y_tr, X_te, y_te, SX_te = get_dirichlet_data(
            n_samples, n_test, ALPHA1, ALPHA2, PRIOR1, seed=SEED)
        theta_tr = np.arctan2(X_tr[:, 1], X_tr[:, 0])
        _cache[n_samples] = (X_tr, y_tr, X_te, y_te, SX_te, theta_tr)
    return _cache[n_samples]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
C = {
    "bg":       "#fafaf8",
    "text":     "#2d2d2d",
    "grid":     "#e0ddd8",
    "pos":      "#4a7c59",
    "neg":      "#b04a4a",
    "accent":   "#5c6d7e",
    "boundary": "#8a7e6b",
}


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(C["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ax.spines.values():
        s.set_color(C["grid"])
    ax.tick_params(colors=C["text"], labelsize=9)
    ax.set_xlabel(xlabel, color=C["text"], fontsize=10)
    ax.set_ylabel(ylabel, color=C["text"], fontsize=10)
    ax.set_title(title, color=C["text"], fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, color=C["grid"], linewidth=0.5, alpha=0.6)


def _xticks(ax):
    ax.set_xlim(0, np.pi / 2)
    ax.set_xticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
    ax.set_xticklabels(["0", "π/8", "π/4", "3π/8", "π/2"])


def _to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Distribution plot
# ---------------------------------------------------------------------------
def make_dist_plot(n_samples):
    X_tr, y_tr, _, _, _, theta_tr = _get_data(n_samples)
    t = THETA_GRID
    pdf_pos = angular_density(t, ALPHA1)
    pdf_neg = angular_density(t, ALPHA2)

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=C["bg"])
    ax.fill_between(t, pdf_pos, alpha=0.20, color=C["pos"], linewidth=0)
    ax.plot(t, pdf_pos, color=C["pos"], lw=2.2,
            label=r"$p(\theta \mid y\!=\!+1)$  [Dir(2,1)]")
    ax.fill_between(t, pdf_neg, alpha=0.20, color=C["neg"], linewidth=0)
    ax.plot(t, pdf_neg, color=C["neg"], lw=2.2,
            label=r"$p(\theta \mid y\!=\!-1)$  [Dir(1,2)]")

    ax.axvline(np.pi / 4, color=C["boundary"], ls="--", lw=1.2,
               label=r"Bayes boundary  $\theta = \pi/4$")

    # scatter training points along the baseline
    m = y_tr > 0
    ax.scatter(theta_tr[m],  -0.04 * np.ones(m.sum()),
               color=C["pos"], s=3, alpha=0.25, zorder=5)
    ax.scatter(theta_tr[~m], -0.12 * np.ones((~m).sum()),
               color=C["neg"], s=3, alpha=0.25, zorder=5)

    ax.set_ylim(bottom=-0.2)
    _xticks(ax)
    _style(ax, xlabel=r"$\theta$", ylabel="Density",
           title="Class-Conditional Densities in Angle Space")
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              bbox_to_anchor=(0.99, 0.99))
    return _to_b64(fig)


# ---------------------------------------------------------------------------
# Classification plot  (sign bands only + test accuracies)
# ---------------------------------------------------------------------------
def make_class_plot(n_samples, L, act_name):
    X_tr, y_tr, X_te, y_te, SX_te, _ = _get_data(n_samples)

    grid_preds, test_preds = run_predictions(
        X_tr, y_tr, X_te, THETA_GRID, depth=L, act_name=act_name)

    pred_signs = np.sign(grid_preds)
    bayes_grid = dirichlet_bayes_grid(ALPHA1, ALPHA2, PRIOR1, THETA_GRID)

    # test-set accuracies
    test_acc  = np.mean(np.sign(test_preds) == y_te) * 100
    bayes_preds = dirichlet_bayes_predict(ALPHA1, ALPHA2, PRIOR1, SX_te)
    bayes_acc = np.mean(bayes_preds == y_te) * 100

    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=C["bg"])

    for i in range(len(THETA_GRID) - 1):
        c = C["pos"] if pred_signs[i] > 0 else C["neg"]
        ax.axvspan(THETA_GRID[i], THETA_GRID[i + 1],
                   ymin=0.52, ymax=0.98, color=c, alpha=0.50, linewidth=0)
    for i in range(len(THETA_GRID) - 1):
        c = C["pos"] if bayes_grid[i] > 0 else C["neg"]
        ax.axvspan(THETA_GRID[i], THETA_GRID[i + 1],
                   ymin=0.02, ymax=0.48, color=c, alpha=0.50, linewidth=0)

    ax.axhline(0.5, color=C["grid"], lw=0.8)
    ax.axvline(np.pi / 4, color=C["boundary"], ls="--", lw=1.2, alpha=0.7)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Bayes Optimal", f"Kernel (L={L})"], fontsize=9)
    ax.set_ylim(0, 1)
    _xticks(ax)
    _style(ax, xlabel=r"$\theta$",
           title=(f"Test Acc: {test_acc:.1f}%  ·  "
                  f"Bayes Acc: {bayes_acc:.1f}%"))

    return _to_b64(fig)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Infinite-Depth NNGP · Bayes Optimal</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400&display=swap');

  :root {
    --bg: #fafaf8; --card: #ffffff; --border: #e8e5df;
    --text: #2d2d2d; --muted: #7a756d;
    --accent: #5c6d7e; --accent-h: #4a5a6b;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 2.5rem 1rem 1.5rem;
  }

  .container { max-width: 900px; margin: 0 auto; }

  /* ---- header ---- */
  header { text-align: center; margin-bottom: 2rem; }
  header h1 {
    font-size: 1.45rem; font-weight: 600;
    letter-spacing: -0.02em; margin-bottom: 0.35rem;
  }
  header .sub {
    color: var(--muted); font-size: 0.84rem;
    font-weight: 300; line-height: 1.5;
  }
  header .math {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: var(--accent);
    background: #f0eee9; padding: 0.45rem 1rem;
    border-radius: 6px; display: inline-block;
    margin-top: 0.7rem;
  }

  /* ---- cards ---- */
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.4rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .card h2 {
    font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); margin-bottom: 0.9rem;
  }
  .card img { width: 100%; border-radius: 6px; }

  /* ---- controls ---- */
  .controls {
    display: flex; align-items: center;
    gap: 0.9rem; flex-wrap: wrap;
  }
  .controls label {
    font-size: 0.82rem; font-weight: 400; color: var(--muted);
  }
  .controls input[type=number] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem; font-weight: 600;
    width: 5.5rem; padding: 0.35rem 0.5rem;
    border: 1px solid var(--border); border-radius: 5px;
    background: var(--bg); color: var(--accent);
    text-align: center;
  }
  .controls select {
    font-family: 'Inter', sans-serif; font-size: 0.82rem;
    padding: 0.35rem 0.5rem; border: 1px solid var(--border);
    border-radius: 5px; background: var(--bg); color: var(--text);
  }
  .controls button {
    background: var(--accent); color: #fff; border: none;
    border-radius: 6px; padding: 0.45rem 1.1rem;
    font-size: 0.82rem; font-weight: 600;
    cursor: pointer; transition: background 0.15s;
  }
  .controls button:hover { background: var(--accent-h); }

  .spinner {
    display: none; width: 16px; height: 16px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.55s linear infinite;
    margin-left: 0.2rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---- footer ---- */
  footer {
    text-align: center; margin-top: 1.2rem;
    font-size: 0.72rem; color: var(--muted);
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>Infinite-Depth NNGP Kernel Regression</h1>
    <p class="sub">Classification on the non-negative orthant of S<sup>1</sup>
      &mdash; Dirichlet data</p>
    <div class="math">φ̌(z) = z⁷/2 + z/2 &nbsp; iterated L times &nbsp;
      · &nbsp; α₊ = (2,1) &nbsp; α₋ = (1,2)</div>
  </header>

  <!-- distributions -->
  <div class="card">
    <h2>Class-Conditional Distributions</h2>
    <img src="data:image/png;base64,{{ dist_img }}" alt="distributions">
  </div>

  <!-- controls -->
  <div class="card">
    <h2>Parameters</h2>
    <form method="post" class="controls" id="ctrl">
      <label for="L">Depth L</label>
      <input type="number" id="L" name="L" min="1" max="500" value="{{ L }}">

      <label for="n">Samples</label>
      <input type="number" id="n" name="n" min="50" max="10000" step="50" value="{{ n }}">

      <label for="act">Activation</label>
      <select name="act" id="act">
        {% for key, desc in activations.items() %}
        <option value="{{ key }}" {{ 'selected' if key == act }}>{{ desc }}</option>
        {% endfor %}
      </select>

      <button type="submit">Run</button>
      <div class="spinner" id="sp"></div>
    </form>
  </div>

  <!-- classification -->
  <div class="card">
    <h2>Classification &mdash; L&thinsp;=&thinsp;{{ L }}, &thinsp;n&thinsp;=&thinsp;{{ n }}</h2>
    <img src="data:image/png;base64,{{ class_img }}" alt="classification">
  </div>

  <footer>18.S998 Spring 2026 &middot; Infinite-Depth Neural Networks</footer>

</div>
<script>
document.getElementById('ctrl').addEventListener('submit',
  function(){ document.getElementById('sp').style.display='inline-block'; });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    L   = int(request.form.get("L", 10))
    n   = int(request.form.get("n", 1000))
    act = request.form.get("act", "2d_opt")
    n = max(50, min(n, 10000))
    L = max(1, min(L, 500))

    dist_img  = make_dist_plot(n)
    class_img = make_class_plot(n, L, act)
    return render_template_string(
        TEMPLATE,
        dist_img=dist_img,
        class_img=class_img,
        L=L,
        n=n,
        act=act,
        activations=ACTIVATIONS,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
