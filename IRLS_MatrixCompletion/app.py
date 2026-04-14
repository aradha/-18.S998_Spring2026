"""
Linear RFM Matrix Completion Web App
=====================================
A Flask + SocketIO app for interactive linear Recursive Feature Machine
matrix completion. Step through iterations one at a time or run continuously.
"""

import threading
import time
import numpy as np
from numpy.linalg import svd
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from linear_rfm import LinearRFMSolver

app = Flask(__name__)
app.config["SECRET_KEY"] = "linear-rfm-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global state
solver = None
solver_lock = threading.Lock()
running = threading.Event()
run_thread = None

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Linear RFM Matrix Completion</title>
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
    max-width: 1200px;
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
  .btn-success {
    background: var(--green-bg);
    color: var(--green);
    border-color: #b8d4c0;
  }
  .btn-success:hover { background: #d4e8da; }
  .btn-full { width: 100%; }

  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

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

  .cell-anim {
    transition: background-color 0.4s ease, color 0.3s ease;
  }

  /* Factor matrices + spectrum layout */
  .factors-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }

  .heatmap-grid {
    display: inline-grid;
    gap: 1px;
    background: var(--border);
    border-radius: 4px;
    padding: 1px;
  }
  .heatmap-cell {
    width: 28px;
    height: 28px;
    border-radius: 0;
    transition: background-color 0.3s ease;
  }

  .chart-wrapper {
    position: relative;
    height: 200px;
  }

  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    .factors-row { grid-template-columns: 1fr; }
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
  <h1>Linear RFM Matrix Completion</h1>
  <p>Enter a partially observed matrix, configure parameters, and step through linear RFM iterations</p>
</div>

<div class="layout">
  <!-- Left panel: controls -->
  <div>
    <div class="panel" style="margin-bottom:1rem;">
      <h2>Matrix Setup</h2>
      <div class="form-row-inline">
        <div class="form-row">
          <label>Rows (n)</label>
          <input type="number" id="nRows" value="5" min="2" max="20">
        </div>
        <div class="form-row">
          <label>Cols (n)</label>
          <input type="number" id="mCols" value="5" min="2" max="20">
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
      <h2>RFM Parameters</h2>
      <div class="form-row">
        <label>Ridge (reg)</label>
        <input type="number" id="reg" value="0.1" step="0.01" min="0.0001">
      </div>
      <div class="form-row">
        <label>Alpha (power)</label>
        <input type="number" id="power" value="1" step="0.1" min="0.1">
      </div>
      <div class="separator"></div>
      <h2>Iteration Control</h2>
      <div class="btn-group" style="margin-top:0;">
        <button class="btn btn-primary" id="stepBtn" onclick="stepOnce()" style="flex:1;">Step 1</button>
        <button class="btn btn-success" id="runBtn" onclick="runContinuous()" style="flex:1;">Run</button>
        <button class="btn btn-danger" id="stopBtn" onclick="stopRunning()" disabled style="flex:1;">Stop</button>
      </div>
      <div style="margin-top:0.5rem;">
        <button class="btn btn-secondary btn-full" onclick="resetApp()">Reset</button>
      </div>
    </div>
  </div>

  <!-- Right: matrix display -->
  <div class="main-area">
    <div class="status-bar">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Ready — generate a grid and enter values</span>
      <div class="status-metrics">
        <span><span class="metric-label">Iter </span><span class="metric-value" id="metricIter">—</span></span>
        <span><span class="metric-label">Error </span><span class="metric-value" id="metricError">—</span></span>
      </div>
    </div>

    <div>
      <div class="matrix-label">
        <h3>Predicted Matrix (A × B)</h3>
        <span class="tag tag-observed">observed</span>
        <span class="tag tag-predicted">predicted</span>
      </div>
      <div class="matrix-container" id="matrixContainer">
        <p style="color:var(--text-sec); font-size:0.85rem; padding:2rem 0;">Generate a grid to begin.</p>
      </div>
    </div>

    <div class="factors-row" id="factorsRow" style="display:none;">
      <div>
        <div class="matrix-label"><h3>A (coefficients)</h3></div>
        <div class="matrix-container" id="heatmapA"></div>
      </div>
      <div>
        <div class="matrix-label"><h3>B (features / M)</h3></div>
        <div class="matrix-container" id="heatmapB"></div>
      </div>
    </div>

    <div class="factors-row" id="spectraRow" style="display:none;">
      <div>
        <div class="matrix-label"><h3>Spectrum of A</h3></div>
        <div class="chart-wrapper"><canvas id="specChartA"></canvas></div>
      </div>
      <div>
        <div class="matrix-label"><h3>Spectrum of B</h3></div>
        <div class="chart-wrapper"><canvas id="specChartB"></canvas></div>
      </div>
    </div>
  </div>
</div>

<script>
const socket = io();
let nRows = 5, mCols = 5;
let isRunning = false;
let initialized = false;
let inputMode = true;
let observedMask = [];
let observedValues = [];
let snapshotValues = [];
let specChartA = null, specChartB = null;

// ---- Matrix Grid ----
function generateGrid() {
  nRows = parseInt(document.getElementById('nRows').value);
  mCols = parseInt(document.getElementById('mCols').value);
  inputMode = true;
  initialized = false;
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
  document.getElementById('metricIter').textContent = '—';
  document.getElementById('metricError').textContent = '—';
  setStatus('idle', 'Grid ready — fill in observed entries, leave blanks for missing');
  updateButtons();
}

function fillRandom() {
  const n = parseInt(document.getElementById('nRows').value);
  const m = parseInt(document.getElementById('mCols').value);
  nRows = n; mCols = m;
  generateGrid();
  const d = Math.min(n, m);
  const rank = Math.max(1, Math.min(3, d));
  const U = Array.from({length: n}, () => Array.from({length: rank}, () => (Math.random()-0.5)*2));
  const V = Array.from({length: rank}, () => Array.from({length: m}, () => (Math.random()-0.5)*2));
  const M = Array.from({length: n}, (_, i) =>
    Array.from({length: m}, (_, j) => {
      let s = 0; for (let k=0;k<rank;k++) s += U[i][k]*V[k][j]; return s;
    })
  );
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (Math.random() < 0.6) {
        document.getElementById(`cell-${i}-${j}`).value = M[i][j].toFixed(2);
      }
    }
  }
}

// ---- Read matrix from grid ----
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

function snapshotGrid() {
  snapshotValues = [];
  for (let i = 0; i < nRows; i++) {
    const row = [];
    for (let j = 0; j < mCols; j++) {
      row.push(document.getElementById(`cell-${i}-${j}`).value);
    }
    snapshotValues.push(row);
  }
}

// ---- Initialize solver on backend ----
function initSolver() {
  const {M, mask} = readMatrix();
  const nObs = mask.flat().filter(x=>x).length;
  if (nObs === 0) { alert('Enter at least one observed value.'); return false; }
  if (nObs === nRows*mCols) { alert('Leave at least one entry blank for prediction.'); return false; }
  snapshotGrid();
  observedMask = mask;
  observedValues = M;

  switchToDisplay(M, mask);

  socket.emit('init_solver', {
    M, mask,
    reg: parseFloat(document.getElementById('reg').value),
    power: parseFloat(document.getElementById('power').value),
  });
  initialized = true;
  return true;
}

// ---- Step / Run / Stop ----
function stepOnce() {
  if (isRunning) return;
  if (!initialized) {
    if (!initSolver()) return;
  }
  socket.emit('step_once');
  setStatus('idle', 'Stepping...');
}

function runContinuous() {
  if (isRunning) return;
  if (!initialized) {
    if (!initSolver()) return;
  }
  isRunning = true;
  updateButtons();
  setStatus('active', 'Running...');
  socket.emit('run_continuous');
}

function stopRunning() {
  socket.emit('stop_running');
}

function resetApp() {
  if (isRunning) {
    socket.emit('stop_running');
    isRunning = false;
  }
  initialized = false;
  socket.emit('reset_solver');
  generateGrid();
  if (snapshotValues.length === nRows && snapshotValues[0] && snapshotValues[0].length === mCols) {
    for (let i = 0; i < nRows; i++) {
      for (let j = 0; j < mCols; j++) {
        document.getElementById(`cell-${i}-${j}`).value = snapshotValues[i][j];
      }
    }
  }
  document.getElementById('metricIter').textContent = '—';
  document.getElementById('metricError').textContent = '—';
  // Hide and reset factor visualizations
  document.getElementById('factorsRow').style.display = 'none';
  document.getElementById('spectraRow').style.display = 'none';
  document.getElementById('heatmapA').innerHTML = '';
  document.getElementById('heatmapB').innerHTML = '';
  if (specChartA) { specChartA.destroy(); specChartA = null; }
  if (specChartB) { specChartB.destroy(); specChartB = null; }
  setStatus('idle', 'Reset — ready to iterate again');
}

function updateButtons() {
  document.getElementById('stepBtn').disabled = isRunning;
  document.getElementById('runBtn').disabled = isRunning;
  document.getElementById('stopBtn').disabled = !isRunning;
}

// ---- Display mode ----
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

// ---- Heatmap rendering ----
function valToColor(v, maxAbs) {
  const t = maxAbs > 0 ? Math.min(Math.abs(v) / maxAbs, 1) : 0;
  if (v >= 0) {
    // white -> blue
    const r = Math.round(255 * (1 - t * 0.7));
    const g = Math.round(255 * (1 - t * 0.55));
    const b = 255;
    return `rgb(${r},${g},${b})`;
  } else {
    // white -> red
    const r = 255;
    const g = Math.round(255 * (1 - t * 0.6));
    const b = Math.round(255 * (1 - t * 0.6));
    return `rgb(${r},${g},${b})`;
  }
}

function renderHeatmap(containerId, matrix) {
  const d = matrix.length;
  const container = document.getElementById(containerId);
  const cellW = Math.max(16, Math.min(28, 320 / d));
  // Find global max abs for color scaling
  let maxAbs = 0;
  for (let i = 0; i < d; i++)
    for (let j = 0; j < matrix[i].length; j++)
      maxAbs = Math.max(maxAbs, Math.abs(matrix[i][j]));

  container.innerHTML = `<div class="heatmap-grid" style="grid-template-columns: repeat(${matrix[0].length}, ${cellW}px);" ></div>`;
  const grid = container.firstChild;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      const div = document.createElement('div');
      div.className = 'heatmap-cell';
      div.style.width = cellW + 'px';
      div.style.height = cellW + 'px';
      div.style.backgroundColor = valToColor(matrix[i][j], maxAbs);
      div.title = matrix[i][j].toFixed(4);
      grid.appendChild(div);
    }
  }
}

// ---- Spectrum charts ----
function createSpecChart(canvasId) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: '#57534e', borderRadius: 3 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: 'Index', font: {size: 10, family: 'Inter'}, color: '#78716c' },
          ticks: { font: {size: 9}, color: '#a8a29e' },
          grid: { display: false },
        },
        y: {
          title: { display: true, text: 'Singular value', font: {size: 10, family: 'Inter'}, color: '#78716c' },
          ticks: { font: {size: 9}, color: '#a8a29e' },
          grid: { color: '#f5f5f4' },
          beginAtZero: true,
        }
      }
    }
  });
}

function updateSpecChart(chart, values) {
  chart.data.labels = values.map((_, i) => i + 1);
  chart.data.datasets[0].data = values;
  chart.update();
}

// ---- Socket events ----
socket.on('rfm_update', (data) => {
  const {iteration, error, M_pred, A, B, specA, specB} = data;
  // Update prediction matrix
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < mCols; j++) {
      const cell = document.getElementById(`cell-${i}-${j}`);
      if (!cell) continue;
      if (!observedMask[i][j]) {
        cell.textContent = M_pred[i][j].toFixed(2);
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
  // Show factor sections
  document.getElementById('factorsRow').style.display = 'grid';
  document.getElementById('spectraRow').style.display = 'grid';
  // Update heatmaps
  renderHeatmap('heatmapA', A);
  renderHeatmap('heatmapB', B);
  // Update spectrum charts
  if (!specChartA) specChartA = createSpecChart('specChartA');
  if (!specChartB) specChartB = createSpecChart('specChartB');
  updateSpecChart(specChartA, specA);
  updateSpecChart(specChartB, specB);
  // Update metrics
  document.getElementById('metricIter').textContent = iteration;
  document.getElementById('metricError').textContent = error.toExponential(4);
});

socket.on('rfm_stopped', (data) => {
  isRunning = false;
  updateButtons();
  const msg = data && data.message ? data.message : 'Stopped';
  setStatus('idle', msg);
});

socket.on('rfm_error', (data) => {
  isRunning = false;
  updateButtons();
  setStatus('idle', `Error: ${data.message}`);
});

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


@socketio.on("init_solver")
def handle_init_solver(data):
    global solver
    M = np.array(data["M"], dtype=np.float64)
    mask = np.array(data["mask"], dtype=bool)
    reg = float(data["reg"])
    power = float(data["power"])

    with solver_lock:
        solver = LinearRFMSolver(M, mask, reg=reg, power=power)


def build_update_payload(solver, out, error, iteration):
    """Build the SocketIO payload with matrices A (sol), B (M), and their spectra."""
    A = solver.sol
    B = solver.M
    specA = svd(A, compute_uv=False).tolist()
    specB = svd(B, compute_uv=False).tolist()
    return {
        "iteration": iteration,
        "error": float(error),
        "M_pred": out.tolist(),
        "A": A.tolist(),
        "B": B.tolist(),
        "specA": specA,
        "specB": specB,
    }


@socketio.on("step_once")
def handle_step_once():
    global solver
    with solver_lock:
        if solver is None:
            emit("rfm_error", {"message": "Solver not initialized."})
            return
        try:
            out, error, iteration = solver.step()
            emit("rfm_update", build_update_payload(solver, out, error, iteration))
        except Exception as e:
            emit("rfm_error", {"message": str(e)})


@socketio.on("run_continuous")
def handle_run_continuous():
    global solver, run_thread
    running.clear()
    running.set()

    def loop():
        try:
            while running.is_set():
                with solver_lock:
                    if solver is None:
                        break
                    out, error, iteration = solver.step()
                    socketio.emit("rfm_update",
                                  build_update_payload(solver, out, error, iteration))
                socketio.sleep(0.05)
            socketio.emit("rfm_stopped", {"message": f"Stopped at iteration {iteration}"})
        except Exception as e:
            socketio.emit("rfm_error", {"message": str(e)})
            socketio.emit("rfm_stopped", {"message": f"Error: {str(e)}"})

    run_thread = threading.Thread(target=loop, daemon=True)
    run_thread.start()


@socketio.on("stop_running")
def handle_stop_running():
    running.clear()


@socketio.on("reset_solver")
def handle_reset_solver():
    global solver
    running.clear()
    with solver_lock:
        solver = None


if __name__ == "__main__":
    print("Starting Linear RFM app at http://localhost:5003")
    socketio.run(app, host="0.0.0.0", port=5003, debug=False, allow_unsafe_werkzeug=True)
