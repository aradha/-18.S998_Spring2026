"""
Flask web app demonstrating preconditioned vs standard Richardson iteration
for kernel regression (EigenPro).
"""

from flask import Flask, render_template_string, jsonify, request
from kernel_solver import run_comparison

app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EigenPro: Preconditioned Richardson Iteration</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
  }
  header {
    text-align: center;
    padding: 2rem 1rem 1rem;
  }
  header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
  }
  header p {
    color: #94a3b8;
    font-size: 0.95rem;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.5;
  }
  .formula {
    font-family: 'Georgia', serif;
    font-style: italic;
    color: #cbd5e1;
    font-size: 0.88rem;
    margin-top: 0.6rem;
  }
  .controls {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    padding: 1.2rem 2rem;
    background: #1e293b;
    border-top: 1px solid #334155;
    border-bottom: 1px solid #334155;
  }
  .control-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
  }
  .control-group label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .control-group input {
    width: 100px;
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    border: 1px solid #475569;
    background: #0f172a;
    color: #e2e8f0;
    font-size: 0.95rem;
    text-align: center;
    outline: none;
    transition: border-color 0.2s;
  }
  .control-group input:focus { border-color: #60a5fa; }
  .btn {
    align-self: flex-end;
    padding: 0.5rem 1.6rem;
    border: none;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-run {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: #fff;
  }
  .btn-run:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.4); }
  .btn-run:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }
  .plots {
    display: flex;
    justify-content: center;
    gap: 1rem;
    padding: 0.5rem 2rem 2rem;
    flex-wrap: wrap;
  }
  .plot-container {
    flex: 1;
    min-width: 400px;
    max-width: 640px;
    height: 450px;
    background: #1e293b;
    border-radius: 10px;
    border: 1px solid #334155;
    overflow: hidden;
  }
  #status {
    text-align: center;
    padding: 0.3rem 0 0;
    font-size: 0.9rem;
    color: #94a3b8;
    min-height: 1.4rem;
  }
  .spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid #475569;
    border-top-color: #60a5fa;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .epoch-counter {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 700;
    color: #60a5fa;
    padding: 0 0 0.2rem;
    min-height: 1.4rem;
  }
  .legend-note {
    text-align: center;
    padding: 0.6rem 0 0;
    font-size: 1.05rem;
    color: #94a3b8;
  }
</style>
</head>
<body>

<header>
  <h1>EigenPro: Preconditioned Richardson Iteration</h1>
  <p>
    Comparing standard Richardson iteration against EigenPro's preconditioned variant
    for Gaussian kernel regression.
  </p>
  <div class="formula">
    y = sin(x&#8321;x&#8322;x&#8325;) + cos(2x&#8323;) + 5 sin(x&#8324;), &nbsp; x ~ N(0, I&#8325;)
  </div>
</header>

<div class="controls">
  <div class="control-group">
    <label>Train Samples</label>
    <input type="number" id="n_samples" value="300" min="50" max="2000" step="50">
  </div>
  <div class="control-group">
    <label>Eigenvalues (k)</label>
    <input type="number" id="k" value="10" min="1" max="100" step="1">
  </div>
  <div class="control-group">
    <label>Bandwidth</label>
    <input type="number" id="bandwidth" value="2.0" min="0.1" max="20" step="0.1">
  </div>
  <div class="control-group">
    <label>Epochs</label>
    <input type="number" id="epochs" value="100" min="10" max="1000" step="10">
  </div>
  <div class="control-group">
    <button class="btn btn-run" id="runBtn" onclick="runExperiment()">Run</button>
  </div>
</div>

<div class="legend-note">
  <span style="color:#3b82f6;">&#9644;</span> Standard Richardson &nbsp;&nbsp;
  <span style="color:#f59e0b;">&#9644;</span> EigenPro (Preconditioned)
</div>

<div id="status"></div>
<div class="epoch-counter" id="epochCounter"></div>

<div class="plots">
  <div class="plot-container" id="trainPlot"></div>
  <div class="plot-container" id="testPlot"></div>
</div>

<script>
const plotLayout = (title) => ({
  title: { text: title, font: { color: '#e2e8f0', size: 15 } },
  paper_bgcolor: '#1e293b',
  plot_bgcolor: '#1e293b',
  font: { color: '#94a3b8' },
  xaxis: {
    title: 'Epoch',
    gridcolor: '#334155',
    zerolinecolor: '#475569',
    color: '#94a3b8',
  },
  yaxis: {
    title: 'R²',
    gridcolor: '#334155',
    zerolinecolor: '#475569',
    color: '#94a3b8',
    range: [-0.1, 1.05],
  },
  height: 450,
  margin: { l: 60, r: 30, t: 50, b: 50 },
  showlegend: false,
});

const plotConfig = { responsive: true, displayModeBar: false };

// Initialize empty plots
Plotly.newPlot('trainPlot', [], plotLayout('Train R²'), plotConfig);
Plotly.newPlot('testPlot', [], plotLayout('Test R²'), plotConfig);

let animationId = null;

function setStatus(msg, loading = false) {
  const el = document.getElementById('status');
  el.innerHTML = loading ? `<span class="spinner"></span>${msg}` : msg;
}

async function runExperiment() {
  // Cancel any running animation
  if (animationId) { cancelAnimationFrame(animationId); animationId = null; }

  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  setStatus('Computing kernel matrices and running iterations...', true);
  document.getElementById('epochCounter').textContent = '';

  const params = {
    n_samples: parseInt(document.getElementById('n_samples').value),
    k: parseInt(document.getElementById('k').value),
    bandwidth: parseFloat(document.getElementById('bandwidth').value),
    epochs: parseInt(document.getElementById('epochs').value),
  };

  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await resp.json();
    if (data.error) { setStatus('Error: ' + data.error); btn.disabled = false; return; }
    setStatus('Animating results...');
    animateResults(data, params.epochs);
  } catch (e) {
    setStatus('Request failed: ' + e.message);
  }
  btn.disabled = false;
}

function animateResults(data, totalEpochs) {
  const epochs = data.epochs;
  const std = data.standard;
  const eig = data.eigenpro;

  // Determine y-axis range from data
  const allVals = [...std.train_r2, ...std.test_r2, ...eig.train_r2, ...eig.test_r2];
  const yMin = Math.min(Math.min(...allVals), 0);
  const yMax = 1.05;

  const makeTraces = (stdData, eigData) => [
    {
      x: [], y: [],
      mode: 'lines',
      line: { color: '#3b82f6', width: 2.5 },
      name: 'Standard',
    },
    {
      x: [], y: [],
      mode: 'lines',
      line: { color: '#f59e0b', width: 2.5 },
      name: 'EigenPro',
    },
  ];

  const trainTraces = makeTraces();
  const testTraces = makeTraces();

  const trainLayout = plotLayout('Train R²');
  const testLayout = plotLayout('Test R²');
  trainLayout.yaxis.range = [yMin - 0.05, yMax];
  testLayout.yaxis.range = [yMin - 0.05, yMax];
  trainLayout.xaxis.range = [1, totalEpochs];
  testLayout.xaxis.range = [1, totalEpochs];

  Plotly.newPlot('trainPlot', trainTraces, trainLayout, plotConfig);
  Plotly.newPlot('testPlot', testTraces, testLayout, plotConfig);

  let frame = 0;
  const step = Math.max(1, Math.floor(totalEpochs / 300));  // cap at ~300 animation frames

  function tick() {
    if (frame >= totalEpochs) {
      setStatus('Done.');
      document.getElementById('epochCounter').textContent = `Epoch ${totalEpochs} / ${totalEpochs}`;
      return;
    }
    // Advance by step
    const end = Math.min(frame + step, totalEpochs);
    const newX = epochs.slice(frame, end);
    const ext = { extendTraces: true };

    Plotly.extendTraces('trainPlot', {
      x: [newX, newX],
      y: [std.train_r2.slice(frame, end), eig.train_r2.slice(frame, end)],
    }, [0, 1]);

    Plotly.extendTraces('testPlot', {
      x: [newX, newX],
      y: [std.test_r2.slice(frame, end), eig.test_r2.slice(frame, end)],
    }, [0, 1]);

    frame = end;
    document.getElementById('epochCounter').textContent = `Epoch ${frame} / ${totalEpochs}`;
    animationId = requestAnimationFrame(tick);
  }

  animationId = requestAnimationFrame(tick);
}
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/run", methods=["POST"])
def run():
    try:
        params = request.get_json()
        n_samples = int(params.get("n_samples", 300))
        k = int(params.get("k", 10))
        bandwidth = float(params.get("bandwidth", 2.0))
        num_epochs = int(params.get("epochs", 100))

        # Sanity bounds
        n_samples = max(50, min(n_samples, 2000))
        k = max(1, min(k, n_samples - 2))
        num_epochs = max(1, min(num_epochs, 1000))

        results = run_comparison(n_samples, k, bandwidth, num_epochs)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
