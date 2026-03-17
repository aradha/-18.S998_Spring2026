"""
Web app for visualizing deep linear diagonal network training (live streaming).
Run: python app.py
Open: http://localhost:8765
"""

import json
import time
import http.server
import socketserver
import urllib.parse

from deep_linear_diagonal import generate_data, train_streaming

PORT = 8765

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deep Linear Diagonal Networks</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  :root {
    --bg: #fafaf9;
    --surface: #ffffff;
    --border: #e7e5e4;
    --border-hover: #d6d3d1;
    --text: #1c1917;
    --text-secondary: #78716c;
    --text-tertiary: #a8a29e;
    --accent: #57534e;
    --accent-light: #78716c;
    --train-color: #57534e;
    --test-color: #a8a29e;
    --shadow: 0 1px 3px rgba(28, 25, 23, 0.04), 0 1px 2px rgba(28, 25, 23, 0.06);
    --radius: 10px;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  .app {
    max-width: 1280px;
    margin: 0 auto;
    padding: 40px 32px;
  }

  header { margin-bottom: 36px; }

  header h1 {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.4px;
    color: var(--text);
  }

  header p {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 6px;
    font-weight: 400;
    line-height: 1.5;
  }

  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
    align-items: start;
  }

  .controls {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow);
    position: sticky;
    top: 24px;
  }

  .controls h2 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-tertiary);
    margin-bottom: 20px;
  }

  .param-group { margin-bottom: 18px; }
  .param-group:last-of-type { margin-bottom: 24px; }

  .param-group label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 8px;
  }

  .param-group label span.val {
    font-size: 12px;
    font-weight: 400;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }

  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    outline: none;
    cursor: pointer;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--surface);
    box-shadow: 0 0 0 1px var(--border);
    cursor: pointer;
    transition: transform 0.15s ease;
  }

  input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.15); }

  input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--surface);
    box-shadow: 0 0 0 1px var(--border);
    cursor: pointer;
  }

  button.run {
    width: 100%;
    padding: 10px 0;
    background: var(--text);
    color: var(--bg);
    border: none;
    border-radius: 8px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.2px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  button.run:hover { background: var(--accent); }
  button.run:active { transform: scale(0.98); }

  button.run.stop {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
  }

  button.run.stop:hover {
    background: var(--bg);
    border-color: var(--border-hover);
  }

  .status {
    margin-top: 14px;
    font-size: 12px;
    color: var(--text-tertiary);
    text-align: center;
    min-height: 18px;
    transition: color 0.2s;
  }

  .status.running { color: var(--accent); }

  .progress-track {
    width: 100%;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 10px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    width: 0%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.15s ease;
  }

  .viz {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .card-header h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.2px;
  }

  .legend {
    display: flex;
    gap: 16px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .legend-dot.train { background: var(--train-color); }
  .legend-dot.test { background: var(--test-color); }

  canvas { width: 100%; display: block; }

  .tooltip {
    position: fixed;
    background: var(--text);
    color: var(--bg);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 500;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    z-index: 100;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  .tooltip.visible { opacity: 1; }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    color: var(--text-tertiary);
    font-size: 13px;
  }

  .metrics {
    display: flex;
    gap: 12px;
  }

  .metric {
    font-size: 12px;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }

  .metric strong {
    font-weight: 600;
    color: var(--text);
  }

  @media (max-width: 768px) {
    .layout { grid-template-columns: 1fr; }
    .controls { position: static; }
  }
</style>
</head>
<body>

<div class="app">
  <header>
    <h1>Deep Linear Diagonal Networks</h1>
    <p>Sparse regression with parameterization w = w<sub>1</sub> ⊙ w<sub>2</sub> ⊙ &hellip; ⊙ w<sub>L</sub>, &ensp; x &sim; N(0, I<sub>1000</sub>), &ensp; y = &langle;&beta;*, x&rangle;, &ensp; supp(&beta;*) = {1, &hellip;, 50}</p>
  </header>

  <div class="layout">
    <div class="controls">
      <h2>Parameters</h2>

      <div class="param-group">
        <label>Depth (L) <span class="val" id="v-depth">2</span></label>
        <input type="range" id="depth" min="1" max="10" value="2" step="1">
      </div>

      <div class="param-group">
        <label>Init scale (&alpha;) <span class="val" id="v-alpha">0.01</span></label>
        <input type="range" id="alpha" min="-5" max="1" value="-2" step="0.25">
      </div>

      <div class="param-group">
        <label>Learning rate <span class="val" id="v-lr">0.001</span></label>
        <input type="range" id="lr" min="-5" max="-0.5" value="-3" step="0.25">
      </div>

      <div class="param-group">
        <label>Steps <span class="val" id="v-steps">2000</span></label>
        <input type="range" id="steps" min="100" max="20000" value="2000" step="100">
      </div>

      <div class="param-group">
        <label>Dataset size (n) <span class="val" id="v-n">200</span></label>
        <input type="range" id="n" min="20" max="2000" value="200" step="10">
      </div>

      <button class="run" id="run-btn" onclick="runExperiment()">Run Experiment</button>
      <div class="progress-track" id="progress-track" style="display:none;">
        <div class="progress-bar" id="progress-bar"></div>
      </div>
      <div class="status" id="status"></div>
    </div>

    <div class="viz">
      <div class="card">
        <div class="card-header">
          <h3>Loss Curves</h3>
          <div class="legend">
            <div class="legend-item"><div class="legend-dot train"></div>Train</div>
            <div class="legend-item"><div class="legend-dot test"></div>Test</div>
          </div>
        </div>
        <div id="loss-area">
          <canvas id="loss-canvas" height="320"></canvas>
          <div class="metrics" id="metrics" style="margin-top:12px;display:none;">
            <div class="metric">Final train: <strong id="final-train"></strong></div>
            <div class="metric">Final test: <strong id="final-test"></strong></div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3>Learned Coefficients &ensp;w<sub>1</sub> ⊙ &hellip; ⊙ w<sub>L</sub></h3>
        </div>
        <div id="bar-area">
          <div class="empty-state" id="bar-empty">Run an experiment to see results</div>
          <canvas id="bar-canvas" height="300" style="display:none;"></canvas>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
// ── Slider wiring ──
const sliders = {
  depth:  { el: document.getElementById('depth'),  vEl: document.getElementById('v-depth'),  fmt: v => v },
  alpha:  { el: document.getElementById('alpha'),  vEl: document.getElementById('v-alpha'),  fmt: v => Math.pow(10, v).toPrecision(2) },
  lr:     { el: document.getElementById('lr'),     vEl: document.getElementById('v-lr'),     fmt: v => Math.pow(10, v).toPrecision(2) },
  steps:  { el: document.getElementById('steps'),  vEl: document.getElementById('v-steps'),  fmt: v => v },
  n:      { el: document.getElementById('n'),      vEl: document.getElementById('v-n'),      fmt: v => v },
};

Object.values(sliders).forEach(s => {
  s.el.addEventListener('input', () => { s.vEl.textContent = s.fmt(parseFloat(s.el.value)); });
});

// ── Global state ──
let trainData = [];
let testData = [];
let totalSteps = 0;
let abortController = null;
let latestW = null;
let rafPending = false;

// ── Bar chart padding (shared with mousemove handler) ──
const BAR_PAD = { top: 20, right: 20, bottom: 40, left: 56 };

// ── Run / Stop toggle ──
async function runExperiment() {
  if (abortController) { stopExperiment(); return; }

  const btn = document.getElementById('run-btn');
  const status = document.getElementById('status');
  const progressTrack = document.getElementById('progress-track');
  const progressBar = document.getElementById('progress-bar');

  trainData = [];
  testData = [];
  totalSteps = parseInt(sliders.steps.el.value);
  latestW = null;

  btn.textContent = 'Stop';
  btn.classList.add('stop');
  status.textContent = 'Initializing\u2026';
  status.className = 'status running';
  progressTrack.style.display = 'block';
  progressBar.style.width = '0%';
  document.getElementById('metrics').style.display = 'none';

  // Show bar canvas
  document.getElementById('bar-empty').style.display = 'none';
  document.getElementById('bar-canvas').style.display = 'block';

  abortController = new AbortController();

  const params = new URLSearchParams({
    L: sliders.depth.el.value,
    alpha: Math.pow(10, parseFloat(sliders.alpha.el.value)),
    lr: Math.pow(10, parseFloat(sliders.lr.el.value)),
    steps: sliders.steps.el.value,
    n: sliders.n.el.value,
  });

  try {
    const resp = await fetch('/api/train?' + params.toString(), {
      signal: abortController.signal
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));

        if (data.done) {
          finishTraining(data.elapsed);
          return;
        }

        trainData.push({ step: data.step, loss: data.train_loss });
        testData.push({ step: data.step, loss: data.test_loss });
        latestW = data.w_current;

        const pct = Math.round((data.step / totalSteps) * 100);
        progressBar.style.width = pct + '%';
        status.textContent = 'Step ' + data.step + ' / ' + totalSteps;

        if (!rafPending) {
          rafPending = true;
          requestAnimationFrame(() => {
            rafPending = false;
            drawLoss();
            drawBarChart(latestW);
          });
        }
      }
    }
    finishTraining(null);
  } catch (e) {
    if (e.name === 'AbortError') {
      status.textContent = 'Stopped';
      status.className = 'status';
    } else {
      status.textContent = 'Error: ' + e.message;
      status.className = 'status';
    }
    resetButton();
  }
}

function stopExperiment() {
  if (abortController) { abortController.abort(); abortController = null; }
  resetButton();
}

function resetButton() {
  const btn = document.getElementById('run-btn');
  btn.textContent = 'Run Experiment';
  btn.classList.remove('stop');
  abortController = null;
}

function finishTraining(elapsed) {
  const status = document.getElementById('status');
  const progressBar = document.getElementById('progress-bar');

  progressBar.style.width = '100%';
  resetButton();
  status.textContent = elapsed != null ? ('Done in ' + elapsed.toFixed(1) + 's') : 'Done';
  status.className = 'status';

  drawLoss();
  if (latestW) drawBarChart(latestW);

  if (trainData.length > 0) {
    document.getElementById('metrics').style.display = 'flex';
    document.getElementById('final-train').textContent = trainData[trainData.length - 1].loss.toExponential(3);
    document.getElementById('final-test').textContent = testData[testData.length - 1].loss.toExponential(3);
  }
}

// ── Loss chart ──
function drawLoss() {
  if (trainData.length < 2) return;

  const canvas = document.getElementById('loss-canvas');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 320 * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = 320;

  const pad = { top: 16, right: 20, bottom: 40, left: 60 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  const allVals = trainData.map(d => d.loss).concat(testData.map(d => d.loss)).filter(v => v > 0);
  const logMin = Math.floor(Math.log10(Math.min(...allVals) + 1e-15));
  const logMax = Math.ceil(Math.log10(Math.max(...allVals) + 1e-15));

  const xScale = step => pad.left + (step / totalSteps) * pw;
  const yScale = v => {
    const lv = Math.log10(Math.max(v, 1e-15));
    const t = (lv - logMin) / ((logMax - logMin) || 1);
    return pad.top + (1 - t) * ph;
  };

  ctx.strokeStyle = '#f5f5f4';
  ctx.lineWidth = 1;
  for (let e = logMin; e <= logMax; e++) {
    const y = yScale(Math.pow(10, e));
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
  }

  ctx.fillStyle = '#a8a29e';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let e = logMin; e <= logMax; e++) {
    ctx.fillText('1e' + e, pad.left - 10, yScale(Math.pow(10, e)));
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 5; i++) {
    const step = Math.round((i / 5) * totalSteps);
    ctx.fillText(step, xScale(step), pad.top + ph + 10);
  }

  ctx.fillStyle = '#78716c';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Step', pad.left + pw / 2, H - 4);
  ctx.save();
  ctx.translate(14, pad.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Loss (log scale)', 0, 0);
  ctx.restore();

  function drawCurve(data, color) {
    if (data.length < 1) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(xScale(data[0].step), yScale(data[0].loss));
    for (let i = 1; i < data.length; i++) {
      ctx.lineTo(xScale(data[i].step), yScale(data[i].loss));
    }
    ctx.stroke();
  }

  drawCurve(trainData, '#57534e');
  drawCurve(testData, '#a8a29e');
}

// ── Bar chart ──
function drawBarChart(w) {
  if (!w) return;

  const canvas = document.getElementById('bar-canvas');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 300 * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = 300;

  const pad = BAR_PAD;
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  const d = w.length;
  const k = 50;  // support size

  // Y-axis range: always show 0 and at least up to 1.1 (true signal value)
  const wMin = Math.min(0, ...w);
  const wMax = Math.max(1.1, ...w);
  const yLo = wMin < -0.05 ? wMin * 1.15 : 0;
  const yHi = wMax * 1.1;
  const yRange = yHi - yLo || 1;

  const xScale = i => pad.left + (i / d) * pw;
  const yScale = v => pad.top + (1 - (v - yLo) / yRange) * ph;
  const y0 = yScale(0);

  // ── Support region background ──
  const supportEnd = xScale(k);
  ctx.fillStyle = 'rgba(87, 83, 78, 0.045)';
  ctx.fillRect(pad.left, pad.top, supportEnd - pad.left, ph);

  // ── Grid: y = 0 line ──
  ctx.strokeStyle = '#d6d3d1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, y0);
  ctx.lineTo(pad.left + pw, y0);
  ctx.stroke();

  // ── Reference: y = 1 dashed line (true signal) ──
  const y1 = yScale(1);
  ctx.strokeStyle = '#a8a29e';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, y1);
  ctx.lineTo(pad.left + pw, y1);
  ctx.stroke();
  ctx.setLineDash([]);

  // ── Draw bars ──
  const barW = Math.max(0.8, pw / d);
  for (let i = 0; i < d; i++) {
    const v = w[i];
    const x = xScale(i);
    const yVal = yScale(v);

    // Subtle color distinction: support coords slightly darker
    ctx.fillStyle = i < k ? 'rgba(87, 83, 78, 0.8)' : 'rgba(168, 162, 158, 0.55)';

    if (v >= 0) {
      ctx.fillRect(x, yVal, barW, y0 - yVal);
    } else {
      ctx.fillRect(x, y0, barW, yVal - y0);
    }
  }

  // ── Y-axis labels ──
  ctx.fillStyle = '#a8a29e';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  // Pick nice y ticks
  const yTicks = [0];
  if (yHi >= 0.5) yTicks.push(0.5);
  if (yHi >= 1.0) yTicks.push(1.0);
  if (yHi > 1.5) yTicks.push(Math.round(yHi * 10) / 10);
  if (yLo < -0.1) yTicks.push(Math.round(yLo * 10) / 10);

  for (const v of yTicks) {
    if (v < yLo || v > yHi) continue;
    const y = yScale(v);
    ctx.fillText(v.toFixed(1), pad.left - 8, y);
  }

  // ── X-axis labels ──
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xTicks = [0, 100, 200, 500, 750, 999];
  for (const idx of xTicks) {
    if (idx >= d) continue;
    ctx.fillText(idx, xScale(idx), pad.top + ph + 8);
  }

  // ── Axis titles ──
  ctx.fillStyle = '#78716c';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Coordinate index', pad.left + pw / 2, H - 4);

  ctx.save();
  ctx.translate(14, pad.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Coefficient value', 0, 0);
  ctx.restore();

  // ── Labels for support region and reference line ──
  ctx.font = '10px Inter, sans-serif';
  ctx.fillStyle = '#a8a29e';
  ctx.textAlign = 'left';
  ctx.fillText('support (0\u2013' + (k - 1) + ')', pad.left + 4, pad.top + 12);

  ctx.textAlign = 'right';
  ctx.fillText('\u03B2* = 1', pad.left + pw - 4, y1 - 6);
}

// ── Tooltip on bar chart ──
const barCanvas = document.getElementById('bar-canvas');
barCanvas.addEventListener('mousemove', function(e) {
  if (!latestW) return;
  const rect = barCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const pw = rect.width - BAR_PAD.left - BAR_PAD.right;
  const frac = (mx - BAR_PAD.left) / pw;
  const idx = Math.floor(frac * latestW.length);
  if (idx >= 0 && idx < latestW.length) {
    showTooltip(e, 'w[' + idx + '] = ' + latestW[idx].toFixed(5));
  } else {
    hideTooltip();
  }
});
barCanvas.addEventListener('mouseleave', hideTooltip);

// ── Tooltip ──
const tooltip = document.getElementById('tooltip');
function showTooltip(e, text) {
  tooltip.textContent = text;
  tooltip.className = 'tooltip visible';
  tooltip.style.left = (e.clientX + 10) + 'px';
  tooltip.style.top = (e.clientY - 32) + 'px';
}
function hideTooltip() { tooltip.className = 'tooltip'; }
</script>
</body>
</html>"""


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Suppress noisy tracebacks from client disconnects
        pass


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/train':
            self._handle_train(parsed)
        else:
            self._serve_html()

    def _serve_html(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML.encode('utf-8'))

    def _handle_train(self, parsed):
        t0 = time.time()

        params = urllib.parse.parse_qs(parsed.query)
        L = int(params.get('L', [2])[0])
        alpha = float(params.get('alpha', [0.01])[0])
        lr = float(params.get('lr', [0.001])[0])
        num_steps = int(params.get('steps', [2000])[0])
        n = int(params.get('n', [200])[0])

        d, k, n_test = 1000, 50, 1000
        X_train, y_train, _ = generate_data(n, d, k, seed=42)
        X_test, y_test, _ = generate_data(n_test, d, k, seed=123)

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        try:
            for update in train_streaming(X_train, y_train, X_test, y_test,
                                          L, alpha, lr, num_steps):
                msg = 'data: ' + json.dumps(update) + '\n\n'
                self.wfile.write(msg.encode())
                self.wfile.flush()

            elapsed = time.time() - t0
            done = 'data: ' + json.dumps({'done': True, 'elapsed': elapsed}) + '\n\n'
            self.wfile.write(done.encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    with ThreadedHTTPServer(('', PORT), Handler) as httpd:
        print(f'Server running at http://localhost:{PORT}')
        print('Press Ctrl+C to stop.')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nShutting down.')


if __name__ == '__main__':
    main()
