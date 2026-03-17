"""
Flask app for comparing Nystrom Extension vs Standard Kernel Regression.
Each method has its own play button and streams steps via SSE.
Run: python app.py
Visit: http://localhost:5001
"""

import json
from flask import Flask, render_template_string, request, Response
from nystrom import stream_nystrom, stream_kernel

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nystrom Extension Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f5f6f8;
    color: #1a1a2e;
    min-height: 100vh;
  }

  .header {
    background: #fff;
    padding: 32px 0 24px;
    border-bottom: 1px solid #e0e2e8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .header h1 { text-align: center; font-size: 26px; font-weight: 700; color: #1a1a2e; }
  .header p {
    text-align: center; color: #6b7280; margin-top: 8px; font-size: 14px;
    max-width: 720px; margin-left: auto; margin-right: auto; line-height: 1.5;
  }

  .container { max-width: 1100px; margin: 0 auto; padding: 28px 24px; }

  .formula {
    background: #fff; border: 1px solid #e0e2e8; border-radius: 10px;
    padding: 16px 22px; margin-bottom: 24px; font-size: 13px; color: #6b7280;
    line-height: 1.7; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  }
  .formula strong { color: #1a1a2e; }
  .formula .hl { color: #2563eb; font-weight: 600; }

  .controls {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;
  }
  .control-group label {
    display: block; font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.7px; color: #6b7280; margin-bottom: 6px;
  }
  .control-group select {
    width: 100%; padding: 10px 12px; background: #fff; border: 1px solid #d1d5db;
    border-radius: 8px; color: #1a1a2e; font-size: 14px; outline: none;
  }
  .control-group select:focus { border-color: #2563eb; }

  /* Two-column layout */
  .results-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start;
  }

  /* Cards */
  .result-card {
    background: #fff; border: 1px solid #e0e2e8; border-radius: 10px;
    overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .card-top {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; border-bottom: 1px solid #e0e2e8;
  }
  .card-title { font-size: 14px; font-weight: 700; }
  .card-nystrom .card-title { color: #2563eb; }
  .card-nystrom .card-top { background: #eff6ff; }
  .card-kernel .card-title { color: #b45309; }
  .card-kernel .card-top { background: #fffbeb; }

  /* Play button */
  .play-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 7px 16px; border-radius: 6px; border: none;
    font-size: 13px; font-weight: 700; cursor: pointer; transition: all 0.15s;
  }
  .play-btn svg { width: 14px; height: 14px; fill: currentColor; }
  .play-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  .card-nystrom .play-btn {
    background: #2563eb; color: #fff;
  }
  .card-nystrom .play-btn:hover:not(:disabled) { background: #1d4ed8; }

  .card-kernel .play-btn {
    background: #d97706; color: #fff;
  }
  .card-kernel .play-btn:hover:not(:disabled) { background: #b45309; }

  /* Step table */
  .step-table {
    width: 100%; border-collapse: collapse; font-size: 13px;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  }
  .step-table th {
    text-align: left; padding: 9px 20px; color: #9ca3af; font-weight: 600;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.6px;
    border-bottom: 1px solid #e0e2e8; background: #fafbfc;
  }
  .step-table th:last-child { text-align: right; }
  .step-table td {
    padding: 8px 20px; border-bottom: 1px solid #f3f4f6; color: #374151;
  }
  .step-table td:last-child { text-align: right; font-weight: 600; }
  .step-table tr:last-child td { border-bottom: none; }
  .card-nystrom .step-table td:last-child { color: #2563eb; }
  .card-kernel .step-table td:last-child { color: #b45309; }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .step-table tr.new-row { animation: fadeIn 0.3s ease; }

  /* Summary footer */
  .card-summary {
    border-top: 1px solid #e0e2e8; padding: 14px 20px;
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
    animation: fadeIn 0.3s ease;
  }
  .summary-item { display: flex; justify-content: space-between; align-items: center; }
  .summary-label {
    font-size: 11px; color: #9ca3af; text-transform: uppercase;
    letter-spacing: 0.4px; font-weight: 600;
  }
  .summary-value {
    font-size: 15px; font-weight: 800;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  }
  .card-nystrom .summary-value { color: #2563eb; }
  .card-kernel .summary-value { color: #b45309; }

  .total-row {
    grid-column: 1 / -1; display: flex; justify-content: space-between;
    align-items: center; border-top: 1px solid #e0e2e8;
    padding-top: 10px; margin-top: 4px;
  }
  .total-row .summary-label { font-size: 12px; }
  .total-row .summary-value { font-size: 18px; color: #7c3aed !important; }

  /* Status line per card */
  .card-status {
    padding: 6px 20px; font-size: 12px; color: #9ca3af;
    border-bottom: 1px solid #f3f4f6; min-height: 30px;
    display: none;
  }
  .card-status.active { display: flex; align-items: center; gap: 8px; }
  .card-status::before {
    content: ''; display: inline-block; width: 12px; height: 12px;
    border: 2px solid #e0e2e8; border-top-color: #2563eb;
    border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0;
  }
  .card-kernel .card-status::before { border-top-color: #d97706; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Empty state */
  .card-empty {
    padding: 32px 20px; text-align: center; color: #c4c8d0; font-size: 13px;
  }

  /* Skipped */
  .skipped-msg {
    padding: 28px 20px; text-align: center; color: #9ca3af; font-size: 13px;
  }

  /* Speedup banner */
  .speedup-bar {
    margin-top: 20px; background: #fff; border: 1px solid #e0e2e8;
    border-radius: 10px; padding: 18px 24px; display: none;
    align-items: center; justify-content: center; gap: 36px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    animation: fadeIn 0.4s ease;
  }
  .speedup-item { text-align: center; }
  .speedup-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.8px;
    color: #9ca3af; font-weight: 700; margin-bottom: 2px;
  }
  .speedup-value {
    font-size: 30px; font-weight: 800;
    font-family: 'SF Mono', 'Fira Code', monospace; color: #2563eb;
  }
  .speedup-divider { width: 1px; height: 44px; background: #e0e2e8; }
</style>
</head>
<body>

<div class="header">
  <h1>Nystrom Extension vs Kernel Regression</h1>
  <p>
    Data: x ~ N(0, I<sub>5</sub>), &ensp; y = sin(3x<sub>1</sub>x<sub>2</sub>x<sub>5</sub>) + cos(2x<sub>3</sub>) + 5sin(x<sub>4</sub>)<br>
    Kernel: Laplace &ensp; K(x,z) = exp(-||x-z||<sub>2</sub> / &radic;5)
  </p>
</div>

<div class="container">
  <div class="formula">
    <strong>Nystrom feature map:</strong> &ensp;
    <span class="hl">&phi;(x)</span> = S<sup>-1/2</sup> U<sup>T</sup> K(X<sub>m</sub>, x)
    &ensp; where K<sub>mm</sub> = U S U<sup>T</sup> &ensp;|&ensp;
    Solve: <span class="hl">w</span> = (&Phi;&Phi;<sup>T</sup> + &lambda;I)<sup>-1</sup> &Phi;y
  </div>

  <div class="controls">
    <div class="control-group">
      <label>Training samples (n)</label>
      <select id="n_train">
        <option value="5000">5,000</option>
        <option value="10000" selected>10,000</option>
        <option value="15000">15,000</option>
        <option value="50000">50,000</option>
        <option value="100000">100,000</option>
      </select>
    </div>
    <div class="control-group">
      <label>Nystrom centers (m)</label>
      <select id="m_centers">
        <option value="500">500</option>
        <option value="1000" selected>1,000</option>
        <option value="2000">2,000</option>
        <option value="5000">5,000</option>
        <option value="10000">10,000</option>
      </select>
    </div>
    <div class="control-group">
      <label>Ridge &lambda;</label>
      <select id="lam">
        <option value="0.0001">1e-4</option>
        <option value="0.001" selected>1e-3</option>
        <option value="0.01">1e-2</option>
        <option value="0.1">1e-1</option>
      </select>
    </div>
  </div>

  <div class="results-grid">
    <!-- Nystrom -->
    <div class="result-card card-nystrom" id="nyCard">
      <div class="card-top">
        <span class="card-title">Nystrom Extension</span>
        <button class="play-btn" id="nyBtn" onclick="runNystrom()">
          <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg> Run
        </button>
      </div>
      <div class="card-status" id="nyStatus"></div>
      <div class="card-empty" id="nyEmpty">Press Run to start</div>
      <div id="nyContent" style="display:none;"></div>
    </div>

    <!-- Kernel -->
    <div class="result-card card-kernel" id="krCard">
      <div class="card-top">
        <span class="card-title">Exact Kernel Regression</span>
        <button class="play-btn" id="krBtn" onclick="runKernel()">
          <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg> Run
        </button>
      </div>
      <div class="card-status" id="krStatus"></div>
      <div class="card-empty" id="krEmpty">Press Run to start</div>
      <div id="krContent" style="display:none;"></div>
    </div>
  </div>

  <div class="speedup-bar" id="speedupBar"></div>
</div>

<script>
function fmtTime(v) {
  if (v < 0.001) return '<0.001s';
  if (v < 1) return v.toFixed(3) + 's';
  return v.toFixed(2) + 's';
}
function fmtR2(v) { return v == null ? '\u2014' : v.toFixed(4); }

/* State for comparison */
let nyResult = null;
let krResult = null;

function getParams() {
  return new URLSearchParams({
    n_train: document.getElementById('n_train').value,
    m_centers: document.getElementById('m_centers').value,
    lam: document.getElementById('lam').value
  });
}

function tryShowSpeedup() {
  const bar = document.getElementById('speedupBar');
  if (nyResult && krResult && !krResult.skipped) {
    const speedup = krResult.time / nyResult.time;
    const gap = Math.abs(krResult.test_r2 - nyResult.test_r2);
    const gapSign = krResult.test_r2 > nyResult.test_r2 ? '-' : '+';
    const gapColor = gap < 0.01 ? '#2563eb' : '#b45309';
    bar.style.display = 'flex';
    bar.innerHTML =
      '<div class="speedup-item"><div class="speedup-label">Nystrom Speedup</div>' +
      '<div class="speedup-value">' + speedup.toFixed(1) + 'x</div></div>' +
      '<div class="speedup-divider"></div>' +
      '<div class="speedup-item"><div class="speedup-label">Test R\u00b2 Gap</div>' +
      '<div class="speedup-value" style="color:' + gapColor + '">' +
      gapSign + gap.toFixed(4) + '</div></div>';
  }
}

function runMethod(endpoint, prefix) {
  const btn = document.getElementById(prefix + 'Btn');
  const status = document.getElementById(prefix + 'Status');
  const empty = document.getElementById(prefix + 'Empty');
  const content = document.getElementById(prefix + 'Content');
  const card = document.getElementById(prefix + 'Card');

  btn.disabled = true;
  empty.style.display = 'none';
  status.textContent = 'Generating data\u2026';
  status.classList.add('active');

  /* Reset content area with a fresh table */
  content.innerHTML =
    '<table class="step-table"><thead><tr><th>Step</th><th>Time</th></tr></thead>' +
    '<tbody id="' + prefix + 'Body"></tbody></table>';
  content.style.display = 'block';

  /* Remove any old summary */
  const oldSummary = card.querySelector('.card-summary');
  if (oldSummary) oldSummary.remove();
  const oldSkipped = card.querySelector('.skipped-msg');
  if (oldSkipped) oldSkipped.remove();

  const params = getParams();
  const es = new EventSource(endpoint + '?' + params.toString());

  es.addEventListener('data_ready', function(e) {
    const d = JSON.parse(e.data);
    status.textContent = 'Data ready (' + fmtTime(d.time) + ') \u2014 computing\u2026';
  });

  es.addEventListener('step', function(e) {
    const d = JSON.parse(e.data);
    const tbody = document.getElementById(prefix + 'Body');
    const tr = document.createElement('tr');
    tr.className = 'new-row';
    tr.innerHTML = '<td>' + d.name + '</td><td>' + fmtTime(d.time) + '</td>';
    tbody.appendChild(tr);
    status.textContent = d.name + ' (' + fmtTime(d.time) + ')';
  });

  es.addEventListener('done', function(e) {
    const d = JSON.parse(e.data);
    status.classList.remove('active');
    status.style.display = 'none';

    const summary = document.createElement('div');
    summary.className = 'card-summary';
    summary.innerHTML =
      '<div class="summary-item"><span class="summary-label">Train R\u00b2</span>' +
      '<span class="summary-value">' + fmtR2(d.train_r2) + '</span></div>' +
      '<div class="summary-item"><span class="summary-label">Test R\u00b2</span>' +
      '<span class="summary-value">' + fmtR2(d.test_r2) + '</span></div>' +
      '<div class="total-row"><span class="summary-label">Total Time</span>' +
      '<span class="summary-value">' + fmtTime(d.time) + '</span></div>';
    card.appendChild(summary);

    if (prefix === 'ny') { nyResult = d; }
    else { krResult = d; }
    tryShowSpeedup();
  });

  es.addEventListener('skipped', function(e) {
    const d = JSON.parse(e.data);
    status.classList.remove('active');
    status.style.display = 'none';
    content.style.display = 'none';

    const msg = document.createElement('div');
    msg.className = 'skipped-msg';
    msg.textContent = 'Skipped \u2014 ' + d.reason;
    card.appendChild(msg);

    if (prefix === 'kr') { krResult = { skipped: true }; }
  });

  es.addEventListener('finished', function() {
    es.close();
    btn.disabled = false;
    btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M17.65 6.35A7.96 7.96 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg> Re-run';
  });

  es.onerror = function() {
    es.close();
    btn.disabled = false;
    status.textContent = 'Connection lost.';
  };
}

function runNystrom() {
  nyResult = null;
  document.getElementById('speedupBar').style.display = 'none';
  runMethod('/api/stream/nystrom', 'ny');
}

function runKernel() {
  krResult = null;
  document.getElementById('speedupBar').style.display = 'none';
  runMethod('/api/stream/kernel', 'kr');
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


def _sse_response(generator):
    def generate():
        for msg in generator:
            event = msg.pop("event")
            data = json.dumps(msg)
            yield f"event: {event}\ndata: {data}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/stream/nystrom")
def api_nystrom():
    n_train = int(request.args.get("n_train", 10000))
    m_centers = int(request.args.get("m_centers", 1000))
    lam = float(request.args.get("lam", 1e-3))
    n_train = min(max(n_train, 100), 200000)
    m_centers = min(max(m_centers, 10), n_train)
    return _sse_response(stream_nystrom(n_train, m_centers, lam))


@app.route("/api/stream/kernel")
def api_kernel():
    n_train = int(request.args.get("n_train", 10000))
    lam = float(request.args.get("lam", 1e-3))
    n_train = min(max(n_train, 100), 200000)
    return _sse_response(stream_kernel(n_train, lam))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
