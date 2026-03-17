"""
Flask web app — Linear Regression Gradient Descent Visualizer
Runs on http://localhost:5001

Controls exposed:
  n   : number of training samples
  k   : condition number  (covariance S = diag(1, k))
  eta : learning rate     (default 0.9 / λ_max(X^T X))

Visualization:
  • Full-screen MSE contour plot over the (w1, w2) parameter space
  • Smooth animated GD trajectory starting from w0 = [0, 0]
"""

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from linear_model import generate_data, gradient_descent

app = Flask(__name__)

W_STAR       = np.array([1.0, 1.0])
GRID_N       = 100   # contour grid resolution per axis
MAX_FRAMES   = 120   # cap animation keyframes to keep slider usable

# ---------------------------------------------------------------------------
# Embedded single-page frontend
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Linear Regression — GD Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    padding: 16px 22px 12px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow: hidden;
  }

  /* ── header ── */
  .header h1  { font-size: 1.25rem; color: #e6edf3; margin-bottom: 2px; }
  .header .sub { font-size: 0.76rem; color: #6e7681; }

  /* ── controls card ── */
  .card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
    flex-shrink: 0;
  }
  .controls { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; }
  .ctrl { display: flex; flex-direction: column; gap: 5px; }
  label { font-size: 0.72rem; color: #8b949e; letter-spacing: .04em; text-transform: uppercase; }

  input[type=number] {
    width: 100px;
    padding: 5px 8px;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #c9d1d9;
    font-size: 0.88rem;
  }
  input[type=number]:focus { outline: none; border-color: #58a6ff; }

  /* ── buttons ── */
  .btn-row { display: flex; gap: 7px; align-items: center; }
  .btn {
    padding: 5px 14px;
    border-radius: 6px;
    border: 1px solid #30363d;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .12s, border-color .12s;
    white-space: nowrap;
    user-select: none;
  }
  .btn-run  { background: #238636; border-color: #2ea043; color: #fff; }
  .btn-run:hover  { background: #2ea043; }
  .btn-play { background: #1f6feb; border-color: #388bfd; color: #fff; }
  .btn-play:hover { background: #388bfd; }
  .btn-anim { background: #1c2128; color: #c9d1d9; }
  .btn-anim:hover { background: #21262d; border-color: #58a6ff; }

  /* ── info bar ── */
  .info { margin-top: 9px; font-size: 0.78rem; color: #8b949e; min-height: 1.1em; }
  .info b { color: #58a6ff; }

  /* ── chart card fills remaining space ── */
  .chart-card {
    flex: 1 1 0;
    min-height: 0;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 8px 8px 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    overflow: hidden;
  }

  /* Plotly chart fills chart-card */
  #plt-contour { flex: 1 1 0; min-height: 0; }

  /* ── custom slider ── */
  .slider-row {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 4px;
  }
  .step-lbl {
    font-size: 0.78rem;
    color: #8b949e;
    white-space: nowrap;
    min-width: 90px;
    text-align: right;
  }
  input[type=range] {
    flex: 1;
    -webkit-appearance: none;
    height: 4px;
    background: #30363d;
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #1f6feb;
    border: 2px solid #388bfd;
    cursor: pointer;
    transition: background .12s;
  }
  input[type=range]::-webkit-slider-thumb:hover { background: #388bfd; }
  input[type=range]::-moz-range-thumb {
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #1f6feb;
    border: 2px solid #388bfd;
    cursor: pointer;
  }
</style>
</head>
<body>

<div class="header">
  <h1>Linear Regression: Gradient Descent Visualizer</h1>
  <p class="sub">
    <em>y = ⟨w*, x⟩ + ε</em> &nbsp;·&nbsp;
    <em>x ~ N(0, S)</em>, <em>S = diag(1, k)</em> &nbsp;·&nbsp;
    <em>ε ~ N(0,1)</em> &nbsp;·&nbsp;
    <em>w* = [1, 1]</em> &nbsp;·&nbsp;
    <em>w₀ = [0, 0]</em>
  </p>
</div>

<div class="card">
  <div class="controls">
    <div class="ctrl">
      <label>Samples (n)</label>
      <input id="inp-n" type="number" value="200" min="20" max="5000" step="10">
    </div>
    <div class="ctrl">
      <label>Condition number (k)</label>
      <input id="inp-k" type="number" value="10" min="1.1" max="200" step="0.5">
    </div>
    <div class="ctrl">
      <label>Learning rate (η)</label>
      <input id="inp-eta" type="number" placeholder="auto" min="0.0001" max="2" step="0.0001">
    </div>
    <div class="ctrl">
      <label>GD steps</label>
      <input id="inp-steps" type="number" value="80" min="5" max="2000" step="5">
    </div>
    <div class="ctrl" style="padding-bottom:1px">
      <div class="btn-row">
        <button class="btn btn-run"  onclick="run()">▶ Run</button>
        <button class="btn btn-play" onclick="animPlay()">⏵ Play</button>
        <button class="btn btn-anim" onclick="animPause()">⏸ Pause</button>
        <button class="btn btn-anim" onclick="animReset()">⏮ Reset</button>
      </div>
    </div>
  </div>
  <div class="info" id="info">Loading…</div>
</div>

<div class="chart-card">
  <div id="plt-contour"></div>
  <div class="slider-row">
    <input type="range" id="step-slider" min="0" max="1" value="0" step="1">
    <span class="step-lbl" id="step-lbl">Step 0</span>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

// ── η reset when k changes ────────────────────────────────────────────────
// η default is 0.9 / λ_max(X^T X), computed server-side from the data.
// Clear any manual η override when k changes so the server re-derives it.
$('inp-k').addEventListener('input', () => { $('inp-eta').value = ''; });

// ── animation state ───────────────────────────────────────────────────────
let animTx = [], animTy = [], animSteps = [];
let currentStep = 0, isPlaying = false, lastTs = 0;
const FRAME_MS = 55;   // ms per animation frame

// Scrub to a specific keyframe index using Plotly.restyle only —
// the contour (trace 0) is NEVER touched so it stays intact.
function jumpTo(i) {
  currentStep = i;
  $('step-slider').value = i;
  $('step-lbl').textContent = `Step ${animSteps[i]} / ${animSteps[animSteps.length - 1]}`;
  Plotly.restyle('plt-contour', {
    x: [animTx.slice(0, i + 1), [animTx[i]]],
    y: [animTy.slice(0, i + 1), [animTy[i]]],
  }, [1, 2]);   // indices of the two scatter traces
}

function animTick(ts) {
  if (!isPlaying) return;
  if (ts - lastTs >= FRAME_MS) {
    lastTs = ts;
    if (currentStep < animTx.length - 1) {
      jumpTo(currentStep + 1);
    } else {
      isPlaying = false;
      return;
    }
  }
  requestAnimationFrame(animTick);
}

function animPlay() {
  if (isPlaying) return;
  // Restart from beginning if already at the end
  if (currentStep >= animTx.length - 1) jumpTo(0);
  isPlaying = true;
  lastTs = 0;
  requestAnimationFrame(animTick);
}
function animPause() { isPlaying = false; }
function animReset()  { isPlaying = false; jumpTo(0); }

// Slider drag
$('step-slider').addEventListener('input', function() {
  isPlaying = false;
  jumpTo(+this.value);
});

// ── main simulation ───────────────────────────────────────────────────────
async function run() {
  isPlaying = false;
  const etaRaw = $('inp-eta').value;
  const payload = {
    n:       +$('inp-n').value,
    k:       +$('inp-k').value,
    n_steps: +$('inp-steps').value,
  };
  if (etaRaw) payload.eta = +etaRaw;   // omit → server uses 0.9 / λ_max
  $('info').textContent = 'Running simulation…';
  try {
    const res = await fetch('/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();

    const lastW = d.trajectory[d.trajectory.length - 1];
    const lastL = d.losses[d.losses.length - 1];
    // Populate η field with server-computed value if it was left on auto
    if (!etaRaw) $('inp-eta').value = d.eta.toFixed(6);
    $('info').innerHTML =
      `<b>η</b> = ${d.eta.toFixed(6)}` +
      ` &nbsp;|&nbsp; <b>λ<sub>max</sub></b> = ${d.eigval_max.toFixed(4)}` +
      ` &nbsp;|&nbsp; <b>n</b> = ${d.n}` +
      ` &nbsp;|&nbsp; <b>k</b> = ${d.k}` +
      ` &nbsp;|&nbsp; <b>ŵ final</b> = [${lastW.map(v => v.toFixed(4)).join(', ')}]` +
      ` &nbsp;|&nbsp; <b>MSE final</b> = ${lastL.toFixed(6)}`;

    drawContour(d);
  } catch(e) {
    $('info').textContent = 'Error: ' + e.message;
  }
}

// ── draw (called once per Run; animation uses Plotly.restyle only) ─────────
function drawContour(d) {
  // Cache animation arrays in module-scope so jumpTo() can reach them
  animTx    = d.anim_traj.map(w => w[0]);
  animTy    = d.anim_traj.map(w => w[1]);
  animSteps = d.anim_steps;
  currentStep = 0;

  // Wire up the slider
  const slider = $('step-slider');
  slider.max   = animTx.length - 1;
  slider.value = 0;
  $('step-lbl').textContent = `Step 0 / ${animSteps[animSteps.length - 1]}`;

  // ── traces ──────────────────────────────────────────────────────────────
  // Trace 0: MSE surface (static — never updated after initial render)
  const contour = {
    type: 'contour', name: 'MSE surface',
    x: d.w1_range, y: d.w2_range, z: d.Z,
    colorscale: 'Viridis',
    contours: { coloring: 'heatmap', showlabels: false, ncontours: 25 },
    colorbar: {
      title: { text: 'MSE', font: { color: '#8b949e', size: 11 } },
      tickfont: { color: '#8b949e', size: 10 }, len: 0.7, thickness: 14,
    },
    hovertemplate: 'w₁=%{x:.3f}<br>w₂=%{y:.3f}<br>MSE=%{z:.4f}<extra></extra>',
  };

  // Trace 1: GD path (animated via restyle)
  const path = {
    type: 'scatter', name: 'GD path',
    x: [animTx[0]], y: [animTy[0]],
    mode: 'lines',
    line: { color: '#ff6b6b', width: 2.5 },
  };

  // Trace 2: current iterate dot (animated via restyle)
  const cur = {
    type: 'scatter', name: 'Current w',
    x: [animTx[0]], y: [animTy[0]],
    mode: 'markers',
    marker: { color: '#ff6b6b', size: 12, symbol: 'circle',
              line: { color: '#fff', width: 2 } },
    showlegend: false,
  };

  // Trace 3: w* star (static)
  const wstar = {
    type: 'scatter', name: 'w* (true)',
    x: [d.w_star[0]], y: [d.w_star[1]],
    mode: 'markers+text',
    marker: { color: '#3fb950', size: 16, symbol: 'star',
              line: { color: '#fff', width: 1.5 } },
    text: ['  w*'], textposition: 'middle right',
    textfont: { color: '#3fb950', size: 13 },
  };

  // Trace 4: starting point w0 (static)
  const w0mark = {
    type: 'scatter', name: 'w₀ = [0, 0]',
    x: [animTx[0]], y: [animTy[0]],
    mode: 'markers+text',
    marker: { color: '#f0883e', size: 11, symbol: 'square',
              line: { color: '#fff', width: 1.5 } },
    text: ['  w₀'], textposition: 'middle right',
    textfont: { color: '#f0883e', size: 13 },
  };

  // ── layout (no Plotly sliders/updatemenus — we drive everything in JS) ──
  const plotH = $('plt-contour').parentElement.clientHeight
                - document.querySelector('.slider-row').offsetHeight - 16;

  const layout = {
    title: false,
    paper_bgcolor: '#161b22',
    plot_bgcolor:  '#0d1117',
    xaxis: {
      title: { text: 'w₁', font: { color: '#8b949e', size: 13 } },
      tickfont: { color: '#8b949e' },
      gridcolor: '#21262d', zerolinecolor: '#30363d',
    },
    yaxis: {
      title: { text: 'w₂', font: { color: '#8b949e', size: 13 } },
      tickfont: { color: '#8b949e' },
      gridcolor: '#21262d', zerolinecolor: '#30363d',
    },
    legend: {
      font: { color: '#adb5bd', size: 12 },
      bgcolor: 'rgba(22,27,34,0.88)',
      bordercolor: '#30363d', borderwidth: 1,
      x: 1.06, y: 1,
    },
    height: Math.max(plotH, 300),
    margin: { t: 16, b: 40, l: 58, r: 20 },
  };

  // newPlot draws everything; subsequent updates only touch traces 1 & 2
  Plotly.newPlot('plt-contour',
    [contour, path, cur, wstar, w0mark], layout, { responsive: true });
}

window.addEventListener('load', run);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/generate", methods=["POST"])
def generate():
    body    = request.get_json(force=True)
    n       = int(body.get("n",       200))
    k       = float(body.get("k",     10.0))
    n_steps = int(body.get("n_steps", 80))

    # Generate samples, then compute eigval_max for the default learning rate
    X, y = generate_data(n, k, w_star=W_STAR)
    eigval_max = float(np.linalg.eigvalsh(X.T @ X).max())
    eta     = float(body.get("eta",   0.9 / eigval_max))
    trajectory, losses = gradient_descent(X, y, eta=eta, n_steps=n_steps)

    # ── Subsample trajectory for animation ──────────────────────────────────
    # Always include step 0 and the final step; pick evenly spaced keyframes
    stride = max(1, n_steps // MAX_FRAMES)
    indices = list(range(0, n_steps + 1, stride))
    if indices[-1] != n_steps:
        indices.append(n_steps)
    anim_traj  = trajectory[indices].tolist()
    anim_steps = indices                          # original step numbers → slider labels

    # ── MSE contour grid (vectorised) ──────────────────────────────────────
    tx, ty = trajectory[:, 0], trajectory[:, 1]
    pad    = 2.0                                  # generous padding so bowl curvature is visible

    # Always show at least [-2, 3] on each axis so the landscape is legible
    w1_lo = min(tx.min(), 0.0, W_STAR[0], -2.0) - pad * 0.5
    w1_hi = max(tx.max(), 0.0, W_STAR[0],  3.0) + pad * 0.5
    w2_lo = min(ty.min(), 0.0, W_STAR[1], -2.0) - pad * 0.5
    w2_hi = max(ty.max(), 0.0, W_STAR[1],  3.0) + pad * 0.5

    # Clamp against absurd ranges (diverging GD)
    w1_lo = max(w1_lo, -12.0);  w1_hi = min(w1_hi, 14.0)
    w2_lo = max(w2_lo, -12.0);  w2_hi = min(w2_hi, 14.0)

    w1_range = np.linspace(w1_lo, w1_hi, GRID_N)
    w2_range = np.linspace(w2_lo, w2_hi, GRID_N)
    W1, W2   = np.meshgrid(w1_range, w2_range)

    # Vectorised MSE over the full grid
    Wflat     = np.column_stack([W1.ravel(), W2.ravel()])   # (GRID_N², 2)
    residuals = y[:, None] - X @ Wflat.T                    # (n, GRID_N²)
    Z         = np.mean(residuals ** 2, axis=0).reshape(W1.shape)

    return jsonify({
        "n":          n,
        "k":          k,
        "eta":        eta,
        "eigval_max": eigval_max,
        "w1_range":   w1_range.tolist(),
        "w2_range":   w2_range.tolist(),
        "Z":          Z.tolist(),
        "trajectory": trajectory.tolist(),
        "losses":     losses.tolist(),
        "anim_traj":  anim_traj,
        "anim_steps": anim_steps,
        "w_star":     W_STAR.tolist(),
    })


if __name__ == "__main__":
    print("Starting on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
