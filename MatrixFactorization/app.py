"""
Matrix Factorization Web App
=============================
A Flask + SocketIO app for interactive deep linear network matrix completion.
Lets you enter a partially observed matrix, configure hyperparameters,
and watch the network fill in the missing entries in real time.
"""

import threading
import numpy as np
import torch
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from deep_matrix_factorization import DeepLinearMF

app = Flask(__name__)
app.config["SECRET_KEY"] = "matrix-factorization-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global handle so we can stop training
training_thread = None
stop_training = threading.Event()

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deep Linear Matrix Factorization</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  :root {
    --bg:        #fafaf9;
    --surface:   #ffffff;
    --border:    #e7e5e4;
    --border-hl: #d6d3d1;
    --text:      #1c1917;
    --text-sec:  #78716c;
    --accent:    #57534e;
    --accent-hl: #44403c;
    --warm:      #f5f5f4;
    --green:     #4a7c59;
    --green-bg:  #e8f0ea;
    --red:       #9e4a4a;
    --red-bg:    #f5e6e6;
    --blue:      #4a6a9e;
    --blue-bg:   #e6edf5;
    --observed:  #d4c8be;
    --missing:   #e8e2dc;
    --radius:    10px;
    --shadow:    0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem;
  }

  .header {
    text-align: center;
    margin-bottom: 2rem;
  }
  .header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .header p {
    font-size: 0.85rem;
    color: var(--text-sec);
    margin-top: 0.25rem;
    font-weight: 400;
  }

  .layout {
    max-width: 1400px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 1.5rem;
    align-items: start;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
  }

  .panel h2 {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-sec);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  .form-row {
    margin-bottom: 0.85rem;
  }
  .form-row label {
    display: block;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 0.3rem;
  }
  .form-row-inline {
    display: flex;
    gap: 0.75rem;
  }
  .form-row-inline .form-row { flex: 1; margin-bottom: 0; }

  input[type="number"], select {
    width: 100%;
    padding: 0.5rem 0.65rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.85rem;
    font-family: inherit;
    background: var(--bg);
    color: var(--text);
    transition: border-color 0.15s;
  }
  input[type="number"]:focus, select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(87,83,78,0.1);
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.55rem 1rem;
    border: 1px solid transparent;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    font-family: inherit;
    cursor: pointer;
    transition: all 0.15s;
    gap: 0.4rem;
  }
  .btn-primary {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .btn-primary:hover { background: var(--accent-hl); }
  .btn-secondary {
    background: var(--surface);
    color: var(--text);
    border-color: var(--border);
  }
  .btn-secondary:hover { background: var(--warm); border-color: var(--border-hl); }
  .btn-danger {
    background: var(--red-bg);
    color: var(--red);
    border-color: #e0c8c8;
  }
  .btn-danger:hover { background: #f0d6d6; }
  .btn-full { width: 100%; }

  .btn-group {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
  }

  /* Matrix grid */
  .main-area {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .matrix-and-chart {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }

  .matrix-container { overflow: auto; }

  .matrix-grid {
    display: inline-grid;
    gap: 2px;
    background: var(--border);
    border-radius: 6px;
    padding: 2px;
    box-shadow: var(--shadow);
  }

  .matrix-cell {
    width: 64px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-weight: 400;
    border: none;
    outline: none;
    transition: background 0.3s, color 0.3s, box-shadow 0.15s;
  }

  .matrix-cell-input {
    background: var(--surface);
    color: var(--text);
    text-align: center;
    border-radius: 0;
  }
  .matrix-cell-input:focus {
    box-shadow: inset 0 0 0 2px rgba(87,83,78,0.25);
    z-index: 1;
    position: relative;
    background: #fff;
  }
  .matrix-cell-input::placeholder {
    color: var(--border-hl);
    font-weight: 300;
  }

  .matrix-cell-display {
    border-radius: 0;
    font-weight: 500;
    letter-spacing: -0.01em;
  }
  .matrix-cell-observed {
    background: var(--surface);
    color: var(--text);
  }
  .matrix-cell-missing {
    color: var(--blue);
    font-weight: 600;
  }

  .matrix-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }
  .matrix-label h3 {
    font-size: 0.85rem;
    font-weight: 600;
  }
  .matrix-label .tag {
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-weight: 500;
  }
  .tag-observed { background: var(--green-bg); color: var(--green); }
  .tag-predicted { background: var(--blue-bg); color: var(--blue); }

  /* Chart */
  .chart-wrapper {
    position: relative;
    height: 340px;
  }

  /* Status bar */
  .status-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    background: var(--warm);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.8rem;
    color: var(--text-sec);
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--border-hl);
    flex-shrink: 0;
  }
  .status-dot.active {
    background: var(--green);
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .status-metrics {
    display: flex;
    gap: 1.5rem;
    margin-left: auto;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.75rem;
  }
  .metric-label { color: var(--text-sec); }
  .metric-value { color: var(--text); font-weight: 600; }

  /* Color scale for predicted cells */
  .cell-anim {
    transition: background-color 0.4s ease, color 0.3s ease;
  }

  /* Responsive */
  @media (max-width: 1000px) {
    .layout { grid-template-columns: 1fr; }
    .matrix-and-chart { grid-template-columns: 1fr; }
  }

  .separator {
    height: 1px;
    background: var(--border);
    margin: 1rem 0;
  }
</style>
</head>
<body>

<div class="header">
  <h1>Deep Linear Matrix Factorization</h1>
  <p>Enter a partially observed matrix, configure the network, and watch it complete the missing entries</p>
</div>

<div class="layout">
  <!-- Left panel: controls -->
  <div>
    <div class="panel" style="margin-bottom:1rem;">
      <h2>Matrix Setup</h2>
      <div class="form-row-inline">
        <div class="form-row">
          <label>Rows (n)</label>
          <input type="number" id="nRows" value="5" min="1" max="20">
        </div>
        <div class="form-row">
          <label>Cols (m)</label>
          <input type="number" id="mCols" value="5" min="1" max="20">
        </div>
      </div>
      <div class="btn-group">
        <button class="btn btn-primary btn-full" onclick="generateGrid()">Generate Grid</button>
      </div>
      <div style="margin-top:0.6rem;">
        <button class="btn btn-secondary btn-full" onclick="fillRandom()" title="Fill with a random low-rank matrix and hide some entries">Random Low-Rank Example</button>
      </div>
    </div>

    <div class="panel">
      <h2>Network &amp; Training</h2>
      <div class="form-row">
        <label>Depth (L)</label>
        <input type="number" id="depth" value="3" min="1" max="20">
      </div>
      <div class="form-row">
        <label>Init Scale (alpha)</label>
        <input type="number" id="alpha" value="0.01" step="0.001" min="0.0001">
      </div>
      <div class="form-row">
        <label>Learning Rate</label>
        <input type="number" id="lr" value="0.003" step="0.0001" min="0.00001">
      </div>
      <div class="form-row-inline">
        <div class="form-row">
          <label>Optimizer</label>
          <select id="optimizer">
            <option value="adam" selected>Adam</option>
            <option value="sgd">SGD</option>
          </select>
        </div>
        <div class="form-row">
          <label>Steps</label>
          <input type="number" id="numSteps" value="3000" min="100" step="100">
        </div>
      </div>
      <div class="form-row">
        <label>Weight Decay</label>
        <input type="number" id="weightDecay" value="0" step="0.0001" min="0">
      </div>
      <div class="separator"></div>
      <div class="btn-group" style="margin-top:0;">
        <button class="btn btn-primary" id="trainBtn" onclick="startTraining()" style="flex:2;">Train</button>
        <button class="btn btn-danger" id="stopBtn" onclick="stopTraining()" disabled style="flex:1;">Stop</button>
      </div>
      <div style="margin-top:0.5rem;">
        <button class="btn btn-secondary btn-full" onclick="resetApp()">Reset</button>
      </div>
    </div>
  </div>

  <!-- Right: matrix + chart -->
  <div class="main-area">
    <div class="status-bar">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Ready — generate a grid and enter values</span>
      <div class="status-metrics">
        <span><span class="metric-label">Step </span><span class="metric-value" id="metricStep">—</span></span>
        <span><span class="metric-label">Loss </span><span class="metric-value" id="metricLoss">—</span></span>
      </div>
    </div>

    <div class="matrix-and-chart">
      <div>
        <div class="matrix-label">
          <h3>Matrix</h3>
          <span class="tag tag-observed">observed</span>
          <span class="tag tag-predicted">predicted</span>
        </div>
        <div class="matrix-container" id="matrixContainer">
          <p style="color:var(--text-sec); font-size:0.85rem; padding:2rem 0;">Generate a grid to begin.</p>
        </div>
      </div>
      <div>
        <div class="matrix-label"><h3>Training Loss</h3></div>
        <div class="chart-wrapper">
          <canvas id="lossChart"></canvas>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const socket = io();
let nRows = 5, mCols = 5;
let isTraining = false;
let lossChart = null;
let inputMode = true; // true = editable grid, false = display mode
let observedMask = [];
let observedValues = [];
let snapshotValues = []; // grid values saved right before training

// ---- Matrix Grid ----
function generateGrid() {
  nRows = parseInt(document.getElementById('nRows').value);
  mCols = parseInt(document.getElementById('mCols').value);
  inputMode = true;
  const container = document.getElementById('matrixContainer');
  const cellW = Math.max(52, Math.min(72, 400 / Math.max(nRows, mCols)));
  container.innerHTML = `<div class="matrix-grid" style="grid-template-columns: repeat(${mCols}, ${cellW}px);" id="matrixGrid"></div>`;
  const grid = document.getElementById('matrixGrid');
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < mCols; j++) {
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.className = 'matrix-cell matrix-cell-input';
      inp.style.width = cellW + 'px';
      inp.id = `cell-${i}-${j}`;
      inp.placeholder = '·';
      grid.appendChild(inp);
    }
  }
  resetChart();
  setStatus('idle', 'Grid ready — fill in observed entries, leave blanks for missing');
}

function fillRandom() {
  const n = parseInt(document.getElementById('nRows').value);
  const m = parseInt(document.getElementById('mCols').value);
  nRows = n; mCols = m;
  generateGrid();
  // Low-rank matrix
  const rank = Math.max(1, Math.min(3, Math.min(n, m)));
  const U = Array.from({length: n}, () => Array.from({length: rank}, () => (Math.random()-0.5)*2));
  const V = Array.from({length: rank}, () => Array.from({length: m}, () => (Math.random()-0.5)*2));
  const M = Array.from({length: n}, (_, i) =>
    Array.from({length: m}, (_, j) => {
      let s = 0; for (let k=0;k<rank;k++) s += U[i][k]*V[k][j]; return s;
    })
  );
  // Observe ~60% of entries
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (Math.random() < 0.6) {
        document.getElementById(`cell-${i}-${j}`).value = M[i][j].toFixed(2);
      }
    }
  }
}

// ---- Training ----
function readMatrix() {
  const M = [];
  const mask = [];
  for (let i = 0; i < nRows; i++) {
    const row = [], mrow = [];
    for (let j = 0; j < mCols; j++) {
      const val = document.getElementById(`cell-${i}-${j}`).value.trim();
      if (val === '' || isNaN(parseFloat(val))) {
        row.push(0);
        mrow.push(false);
      } else {
        row.push(parseFloat(val));
        mrow.push(true);
      }
    }
    M.push(row);
    mask.push(mrow);
  }
  return {M, mask};
}

function startTraining() {
  if (isTraining) return;
  const {M, mask} = readMatrix();
  // Check we have at least one observed and one missing
  const nObs = mask.flat().filter(x=>x).length;
  if (nObs === 0) { alert('Enter at least one observed value.'); return; }
  if (nObs === nRows*mCols) { alert('Leave at least one entry blank for prediction.'); return; }

  // Snapshot the raw grid text so we can restore on reset
  snapshotValues = [];
  for (let i = 0; i < nRows; i++) {
    const row = [];
    for (let j = 0; j < mCols; j++) {
      row.push(document.getElementById(`cell-${i}-${j}`).value);
    }
    snapshotValues.push(row);
  }

  observedMask = mask;
  observedValues = M;
  isTraining = true;
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;
  setStatus('active', 'Training...');
  resetChart();

  // Switch to display mode
  switchToDisplay(M, mask);

  socket.emit('start_training', {
    M, mask,
    depth: parseInt(document.getElementById('depth').value),
    alpha: parseFloat(document.getElementById('alpha').value),
    lr: parseFloat(document.getElementById('lr').value),
    optimizer: document.getElementById('optimizer').value,
    num_steps: parseInt(document.getElementById('numSteps').value),
    weight_decay: parseFloat(document.getElementById('weightDecay').value),
  });
}

function stopTraining() {
  socket.emit('stop_training');
}

function resetApp() {
  // Stop any in-flight training
  if (isTraining) {
    socket.emit('stop_training');
    isTraining = false;
  }
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;

  // Rebuild the editable grid
  generateGrid();

  // Restore the snapshot values (if we have them for this grid size)
  if (snapshotValues.length === nRows && snapshotValues[0] && snapshotValues[0].length === mCols) {
    for (let i = 0; i < nRows; i++) {
      for (let j = 0; j < mCols; j++) {
        document.getElementById(`cell-${i}-${j}`).value = snapshotValues[i][j];
      }
    }
  }

  // Reset metrics
  document.getElementById('metricStep').textContent = '—';
  document.getElementById('metricLoss').textContent = '—';
  setStatus('idle', 'Reset — ready to train again');
}

function switchToDisplay(M, mask) {
  inputMode = false;
  const container = document.getElementById('matrixContainer');
  const cellW = Math.max(52, Math.min(72, 400 / Math.max(nRows, mCols)));
  container.innerHTML = `<div class="matrix-grid" style="grid-template-columns: repeat(${mCols}, ${cellW}px);" id="matrixGrid"></div>`;
  const grid = document.getElementById('matrixGrid');
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < mCols; j++) {
      const div = document.createElement('div');
      div.className = 'matrix-cell matrix-cell-display cell-anim ' +
        (mask[i][j] ? 'matrix-cell-observed' : 'matrix-cell-missing');
      div.id = `cell-${i}-${j}`;
      div.textContent = mask[i][j] ? parseFloat(M[i][j]).toFixed(2) : '?';
      div.style.width = cellW + 'px';
      grid.appendChild(div);
    }
  }
}

// ---- Socket events ----
socket.on('training_update', (data) => {
  const {step, total_steps, loss, M_pred} = data;
  // Update matrix display
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < mCols; j++) {
      const cell = document.getElementById(`cell-${i}-${j}`);
      if (!cell) continue;
      if (!observedMask[i][j]) {
        cell.textContent = M_pred[i][j].toFixed(2);
        // Color intensity based on magnitude
        const v = M_pred[i][j];
        const absV = Math.min(Math.abs(v), 5);
        const intensity = absV / 5;
        const bgAlpha = 0.06 + intensity * 0.12;
        if (v >= 0) {
          cell.style.backgroundColor = `rgba(74, 124, 89, ${bgAlpha})`;
        } else {
          cell.style.backgroundColor = `rgba(158, 74, 74, ${bgAlpha})`;
        }
      }
    }
  }
  // Update chart
  addLossPoint(step, loss);
  // Update status metrics
  document.getElementById('metricStep').textContent = `${step+1}/${total_steps}`;
  document.getElementById('metricLoss').textContent = loss.toFixed(6);
});

socket.on('training_done', (data) => {
  isTraining = false;
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  setStatus('idle', `Done — final loss: ${data.final_loss.toFixed(6)}`);
});

socket.on('training_error', (data) => {
  isTraining = false;
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  setStatus('idle', `Error: ${data.message}`);
});

// ---- Chart ----
function resetChart() {
  if (lossChart) lossChart.destroy();
  const ctx = document.getElementById('lossChart').getContext('2d');
  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Training Loss (MSE)',
        data: [],
        borderColor: '#57534e',
        backgroundColor: 'rgba(87,83,78,0.08)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        x: {
          title: { display: true, text: 'Step', font: {size: 11, family: 'Inter'}, color: '#78716c' },
          ticks: { maxTicksLimit: 8, font: {size: 10}, color: '#a8a29e' },
          grid: { color: '#f5f5f4' },
        },
        y: {
          title: { display: true, text: 'Loss', font: {size: 11, family: 'Inter'}, color: '#78716c' },
          type: 'logarithmic',
          ticks: { font: {size: 10}, color: '#a8a29e' },
          grid: { color: '#f5f5f4' },
        }
      },
      plugins: {
        legend: { display: false },
      }
    }
  });
}

function addLossPoint(step, loss) {
  if (!lossChart) return;
  lossChart.data.labels.push(step);
  lossChart.data.datasets[0].data.push(loss);
  // Thin out points for performance if > 500
  if (lossChart.data.labels.length > 500) {
    const labels = lossChart.data.labels;
    const data = lossChart.data.datasets[0].data;
    const newLabels = [], newData = [];
    for (let i = 0; i < labels.length; i += 2) {
      newLabels.push(labels[i]);
      newData.push(data[i]);
    }
    lossChart.data.labels = newLabels;
    lossChart.data.datasets[0].data = newData;
  }
  lossChart.update();
}

// ---- Status ----
function setStatus(state, text) {
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot' + (state === 'active' ? ' active' : '');
  document.getElementById('statusText').textContent = text;
}

// Initialize
generateGrid();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@socketio.on("start_training")
def handle_start_training(data):
    global training_thread, stop_training
    stop_training.clear()

    M = np.array(data["M"], dtype=np.float32)
    mask = np.array(data["mask"], dtype=bool)
    depth = int(data["depth"])
    alpha = float(data["alpha"])
    lr = float(data["lr"])
    optimizer_name = data["optimizer"]
    num_steps = int(data["num_steps"])
    weight_decay = float(data.get("weight_decay", 0))

    def run():
        try:
            n, m = M.shape
            M_tensor = torch.tensor(M, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.bool)

            model = DeepLinearMF(n, m, depth, alpha)

            if optimizer_name.lower() == "adam":
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

            # How often to emit updates (target ~30fps feeling)
            emit_every = max(1, num_steps // 600)

            for step in range(num_steps):
                if stop_training.is_set():
                    break

                opt.zero_grad()
                M_hat = model()
                diff = (M_hat - M_tensor) * mask_tensor
                loss = (diff ** 2).sum() / mask_tensor.sum()
                loss.backward()
                opt.step()

                if step % emit_every == 0 or step == num_steps - 1:
                    with torch.no_grad():
                        pred = model().numpy().tolist()
                    socketio.emit("training_update", {
                        "step": step,
                        "total_steps": num_steps,
                        "loss": loss.item(),
                        "M_pred": pred,
                    })
                    socketio.sleep(0)  # yield to event loop

            with torch.no_grad():
                pred = model().numpy().tolist()
            final_loss = loss.item()
            socketio.emit("training_update", {
                "step": num_steps - 1,
                "total_steps": num_steps,
                "loss": final_loss,
                "M_pred": pred,
            })
            socketio.emit("training_done", {"final_loss": final_loss})

        except Exception as e:
            socketio.emit("training_error", {"message": str(e)})

    training_thread = threading.Thread(target=run, daemon=True)
    training_thread.start()


@socketio.on("stop_training")
def handle_stop_training():
    stop_training.set()


if __name__ == "__main__":
    print("Starting Matrix Factorization app at http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
