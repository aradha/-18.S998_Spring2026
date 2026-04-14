"""Localhost dashboard for the modular-arithmetic MLP experiments."""

from flask import Flask, render_template_string, request, Response, stream_with_context
import json
import threading
import queue

from train import OPERATIONS, train, build_cayley_table, is_prime

app = Flask(__name__)


OPERATIONS_META = {
    key: {
        'name': spec['name'],
        'symbol': spec['symbol'],
        'requires_prime': spec['requires_prime'],
    }
    for key, spec in OPERATIONS.items()
}


@app.route('/')
def index():
    return render_template_string(TEMPLATE, operations=OPERATIONS_META)


@app.route('/api/cayley', methods=['POST'])
def cayley_endpoint():
    """Return the Cayley table for an operation/n without training."""
    data = request.json
    operation = data.get('operation', 'add')
    n = int(data.get('n', 7))
    try:
        _, _, table = build_cayley_table(operation, n)
    except ValueError as e:
        return {'error': str(e)}, 400
    return {
        'n': n,
        'operation': operation,
        'table': table.tolist(),
    }


@app.route('/api/train', methods=['POST'])
def train_endpoint():
    data = request.json
    operation = data.get('operation', 'add')
    n = int(data.get('n', 7))
    k = int(data.get('k', 1024))
    activation = data.get('activation', 'relu')
    init_scale = float(data.get('init_scale', 1.0))
    lr = float(data.get('lr', 1e-3))
    num_epochs = int(data.get('num_epochs', 2000))
    train_frac = float(data.get('train_frac', 0.5))
    weight_decay = float(data.get('weight_decay', 1.0))
    batch_size = int(data.get('batch_size', 32))
    seed = int(data.get('seed', 0))

    track_every = max(1, num_epochs // 200)
    q = queue.Queue()

    def _work():
        try:
            def on_progress(epoch, total, ckpt):
                q.put(json.dumps({
                    'type': 'checkpoint',
                    'epoch': ckpt['epoch'],
                    'total': total,
                    'train_loss': ckpt['train_loss'],
                    'test_loss': ckpt['test_loss'],
                    'train_error': ckpt['train_error'],
                    'test_error': ckpt['test_error'],
                    'train_acc': ckpt['train_acc'],
                    'test_acc': ckpt['test_acc'],
                    'W1tW1': ckpt['W1tW1'].tolist(),
                }))

            result = train(
                operation=operation,
                n=n,
                k=k,
                activation=activation,
                init_scale=init_scale,
                lr=lr,
                num_epochs=num_epochs,
                train_frac=train_frac,
                weight_decay=weight_decay,
                batch_size=batch_size,
                track_every=track_every,
                seed=seed,
                progress_callback=on_progress,
            )

            q.put(json.dumps({
                'type': 'meta',
                'n': result['n'],
                'operation': result['operation'],
                'table': result['table'].tolist(),
                'train_mask': result['train_mask'].tolist(),
                'test_mask': result['test_mask'].tolist(),
                'n_train': result['n_train'],
                'n_test': result['n_test'],
            }))
            q.put(json.dumps({'type': 'done'}))
        except ValueError as e:
            q.put(json.dumps({'type': 'error', 'message': str(e)}))
        except Exception as e:
            q.put(json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'}))

    threading.Thread(target=_work, daemon=True).start()

    def generate():
        while True:
            msg = q.get()
            yield f"data: {msg}\n\n"
            parsed = json.loads(msg)
            if parsed['type'] in ('done', 'error'):
                break

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Modular Arithmetic MLP</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:#f5f5f4;color:#1c1917;line-height:1.5}

.header{background:#292524;color:#fafaf9;padding:16px 28px;display:flex;align-items:baseline;gap:14px;border-bottom:1px solid #1c1917}
.header h1{font-size:18px;font-weight:600;letter-spacing:-.3px}
.header .sub{font-size:12px;color:#a8a29e}

.container{display:grid;grid-template-columns:300px 1fr;gap:20px;padding:20px;max-width:1560px;margin:0 auto}
.sidebar{display:flex;flex-direction:column;gap:14px}

.card{background:#fff;border:1px solid #e7e5e4;border-radius:10px;padding:18px}
.card-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:#78716c;margin-bottom:14px}

.fg{margin-bottom:12px}
.fg:last-child{margin-bottom:0}
.fg label{display:block;font-size:12px;font-weight:500;color:#44403c;margin-bottom:3px}
.fg select,.fg input[type="number"]{width:100%;padding:7px 10px;border:1px solid #d6d3d1;border-radius:6px;font-size:13px;color:#1c1917;background:#fafaf9;font-family:inherit;transition:border-color .15s,box-shadow .15s}
.fg select:focus,.fg input:focus{outline:none;border-color:#78716c;box-shadow:0 0 0 3px rgba(120,113,108,.1)}
.fg .hint{font-size:10px;color:#a8a29e;margin-top:3px}
.fg.range{display:flex;flex-direction:column}
.fg.range .row{display:flex;align-items:center;gap:10px}
.fg.range input[type="range"]{flex:1;accent-color:#57534e}
.fg.range .val{font-size:12px;color:#44403c;font-variant-numeric:tabular-nums;min-width:38px;text-align:right}

.btn{width:100%;padding:10px;background:#292524;color:#fafaf9;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;font-family:inherit;transition:background .15s}
.btn:hover{background:#1c1917}
.btn:disabled{background:#a8a29e;cursor:not-allowed}

.main{display:flex;flex-direction:column;gap:18px}

.metrics{display:flex;gap:24px;justify-content:center;padding:8px 0;flex-wrap:wrap}
.metric{text-align:center;min-width:110px}
.metric-val{font-size:20px;font-weight:700;color:#1c1917;font-variant-numeric:tabular-nums}
.metric-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:#78716c;margin-top:2px}

.row-2{display:grid;grid-template-columns:1fr 1fr;gap:18px}

.chart-card{background:#fff;border:1px solid #e7e5e4;border-radius:10px;padding:18px}
.chart-card h3{font-size:12px;font-weight:600;color:#57534e;margin-bottom:10px;text-transform:uppercase;letter-spacing:.4px}
.chart-card .sub{font-size:11px;color:#a8a29e;margin-bottom:10px;display:block}

.heatmap-card{background:#fff;border:1px solid #e7e5e4;border-radius:10px;padding:24px;display:flex;flex-direction:column;align-items:center}
.heatmap-card h3{font-size:12px;font-weight:600;color:#57534e;margin-bottom:4px;text-transform:uppercase;letter-spacing:.4px;align-self:flex-start}
.heatmap-card .sub{font-size:11px;color:#a8a29e;margin-bottom:18px;align-self:flex-start}
.heatmap-wrap{display:flex;justify-content:center;align-items:center;width:100%}
.legend{margin-top:16px;display:flex;align-items:center;gap:18px;justify-content:center;font-size:11px;color:#57534e}
.legend .swatch{display:inline-block;width:14px;height:14px;border:1px solid #d6d3d1;vertical-align:middle;margin-right:6px;border-radius:2px}
.legend .train{background:#1d4ed8;border-color:#1d4ed8}
.legend .test{background:#fff}

.stepper{display:flex;align-items:center;justify-content:center;gap:14px;padding:8px 0}
.step-btn{width:34px;height:34px;border-radius:50%;border:1px solid #d6d3d1;background:#fff;color:#44403c;font-size:18px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all .15s;user-select:none;font-family:inherit}
.step-btn:hover:not(:disabled){border-color:#78716c;color:#292524;background:#f5f5f4}
.step-btn:disabled{opacity:.35;cursor:not-allowed}
.step-label{font-size:13px;font-weight:600;min-width:180px;text-align:center;color:#44403c}
.step-slider{flex:1;max-width:260px;accent-color:#57534e}

.gram-section{background:#fff;border:1px solid #e7e5e4;border-radius:10px;padding:24px;display:flex;flex-direction:column;align-items:center}
.gram-section h3{font-size:12px;font-weight:600;color:#57534e;margin-bottom:4px;text-transform:uppercase;letter-spacing:.4px;align-self:flex-start}
.gram-section .sub{font-size:11px;color:#a8a29e;margin-bottom:14px;align-self:flex-start}
.gram-pair{display:flex;gap:24px;justify-content:center;width:100%;flex-wrap:wrap}
.gram-col{display:flex;flex-direction:column;align-items:center}
.gram-lbl{font-size:11px;font-weight:600;color:#78716c;margin-bottom:8px;text-transform:uppercase;letter-spacing:.3px}

.empty{text-align:center;padding:80px 20px;color:#a8a29e}
.empty h3{font-size:16px;color:#78716c;margin-bottom:6px}
.empty p{font-size:13px}
.results-wrap{animation:fadeUp .35s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

.warn{font-size:11px;color:#b91c1c;margin-top:6px;display:none}

@media(max-width:1100px){.row-2{grid-template-columns:1fr}}
@media(max-width:800px){.container{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="header">
  <h1>Modular Arithmetic MLP</h1>
  <span class="sub">1 hidden layer &middot; one-hot inputs &middot; AdamW + MSE on a Cayley-table subset</span>
</div>

<div class="container">
  <div class="sidebar">
    <div class="card">
      <div class="card-title">Task</div>
      <div class="fg">
        <label for="sel-op">Operation</label>
        <select id="sel-op">
          {% for key, op in operations.items() %}
          <option value="{{ key }}">{{ op.name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="fg range">
        <label for="in-n">Modulus n</label>
        <div class="row">
          <input type="range" id="in-n" min="3" max="59" step="1" value="11">
          <span class="val" id="val-n">11</span>
        </div>
        <div class="hint" id="hint-n">Multiplication / division require n prime.</div>
        <div class="warn" id="warn-n">n must be prime for this operation.</div>
      </div>
      <div class="fg range">
        <label for="in-trainfrac">Train Fraction</label>
        <div class="row">
          <input type="range" id="in-trainfrac" min="0.1" max="0.95" step="0.05" value="0.5">
          <span class="val" id="val-trainfrac">0.50</span>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Network</div>
      <div class="fg">
        <label for="in-k">Hidden Width (k)</label>
        <input type="number" id="in-k" value="1024" step="64" min="4" max="8192">
      </div>
      <div class="fg">
        <label for="sel-act">Activation</label>
        <select id="sel-act">
          <option value="relu">ReLU</option>
          <option value="quadratic">Quadratic (z\u00b2)</option>
        </select>
      </div>
      <div class="fg">
        <label for="in-scale">Initialization Scale</label>
        <input type="number" id="in-scale" value="1.0" step="0.1" min="0.001" max="10">
        <div class="hint">Multiplier on standard PyTorch init (1.0 = default).</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Optimizer (AdamW)</div>
      <div class="fg">
        <label for="in-lr">Learning Rate</label>
        <input type="number" id="in-lr" value="0.001" step="0.0001" min="0.00001" max="10">
      </div>
      <div class="fg">
        <label for="in-wd">Weight Decay</label>
        <input type="number" id="in-wd" value="1.0" step="0.1" min="0" max="100">
      </div>
      <div class="fg">
        <label for="in-batch">Batch Size</label>
        <input type="number" id="in-batch" value="32" step="1" min="1" max="4096">
      </div>
      <div class="fg">
        <label for="in-epochs">Epochs</label>
        <input type="number" id="in-epochs" value="2000" step="500" min="10" max="500000">
      </div>
      <div class="fg">
        <label for="in-seed">Random Seed</label>
        <input type="number" id="in-seed" value="0" step="1" min="0" max="9999">
      </div>
    </div>

    <button class="btn" id="run-btn" onclick="startTraining()">Train Network</button>
  </div>

  <div class="main">
    <div class="empty" id="empty-state">
      <h3>Configure &amp; Train</h3>
      <p>Choose an operation and modulus on the left, then click <strong>Train Network</strong>.</p>
    </div>

    <div id="results" style="display:none">
      <div class="card">
        <div class="metrics">
          <div class="metric"><div class="metric-val" id="v-trainacc">&mdash;</div><div class="metric-lbl">Train Accuracy</div></div>
          <div class="metric"><div class="metric-val" id="v-testacc">&mdash;</div><div class="metric-lbl">Test Accuracy</div></div>
          <div class="metric"><div class="metric-val" id="v-trainloss">&mdash;</div><div class="metric-lbl">Train MSE</div></div>
          <div class="metric"><div class="metric-val" id="v-testloss">&mdash;</div><div class="metric-lbl">Test MSE</div></div>
          <div class="metric"><div class="metric-val" id="v-ntrain">&mdash;</div><div class="metric-lbl">Train / Test Pairs</div></div>
        </div>
      </div>

      <div class="row-2">
        <div class="chart-card">
          <h3>Train &amp; Test Accuracy</h3>
          <span class="sub">Fraction of correctly predicted Cayley-table cells</span>
          <canvas id="acc-chart"></canvas>
        </div>
        <div class="chart-card">
          <h3>Train &amp; Test MSE</h3>
          <span class="sub">Mean squared error on one-hot targets (log scale)</span>
          <canvas id="loss-chart"></canvas>
        </div>
      </div>

      <div class="gram-section">
        <h3>W<sub>1</sub><sup>T</sup> W<sub>1</sub> through training</h3>
        <span class="sub">Gram matrix of the first-layer weights (2n &times; 2n)</span>
        <div class="gram-pair">
          <div class="gram-col"><div class="gram-lbl">With diagonal</div><canvas id="gram-canvas"></canvas></div>
          <div class="gram-col"><div class="gram-lbl">Diagonal removed</div><canvas id="gram-canvas-nodiag"></canvas></div>
        </div>
        <div class="card" style="width:100%;margin-top:14px;padding:12px 18px">
          <div class="stepper">
            <button class="step-btn" id="btn-prev" onclick="stepGram(-1)">&lsaquo;</button>
            <span class="step-label" id="gram-label">Epoch 0 / 0</span>
            <input type="range" class="step-slider" id="gram-slider" min="0" max="0" value="0"
                   oninput="goToGram(+this.value)">
            <button class="step-btn" id="btn-next" onclick="stepGram(1)">&rsaquo;</button>
          </div>
          <div style="text-align:center;margin-top:8px">
            <button class="btn" id="btn-reorder" style="width:auto;display:none;padding:8px 20px" onclick="toggleReorder()">Re-order</button>
            <span id="reorder-info" style="display:none;font-size:11px;color:#78716c;margin-left:10px"></span>
          </div>
        </div>
      </div>

      <div class="heatmap-card">
        <h3>Observed Cayley Table</h3>
        <span class="sub" id="ct-sub">&mdash;</span>
        <div class="heatmap-wrap"><canvas id="cayley-canvas"></canvas></div>
        <div class="legend">
          <span><span class="swatch train"></span>Observed (training)</span>
          <span><span class="swatch test"></span>Unobserved</span>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const OPERATIONS = {{ operations|tojson }};
let charts = {};
let isTraining = false;
let meta = null;
let checkpoints = [];
let gramIdx = 0;
let gramReordered = false;
let dlogOrder = null;   /* {g, order} computed once per run */

const CLR = { dark:'#292524', med:'#78716c', light:'#a8a29e', accent:'#7c2d12' };

function isPrime(n){
  if(n<2) return false;
  if(n<4) return true;
  if(n%2===0) return false;
  for(let i=3;i*i<=n;i+=2) if(n%i===0) return false;
  return true;
}

/* ── discrete-log reordering helpers ── */
function modPow(base, exp, mod){
  let r = 1; base = base % mod;
  while(exp > 0){
    if(exp & 1) r = (r * base) % mod;
    exp >>= 1;
    base = (base * base) % mod;
  }
  return r;
}

function findGenerator(n){
  /* Smallest primitive root of (Z/nZ)*.  n must be prime. */
  const phi = n - 1;
  const factors = [];
  let tmp = phi;
  for(let p=2; p*p<=tmp; p++){
    if(tmp % p === 0){ factors.push(p); while(tmp % p === 0) tmp /= p; }
  }
  if(tmp > 1) factors.push(tmp);
  for(let g=2; g<n; g++){
    let ok = true;
    for(const p of factors){
      if(modPow(g, phi/p, n) === 1){ ok = false; break; }
    }
    if(ok) return g;
  }
  return 2;
}

function computeDlogOrder(n){
  const g = findGenerator(n);
  const order = [];
  let v = 1;
  for(let i=0; i<n-1; i++){
    order.push(v);
    v = (v * g) % n;
  }
  order.push(0);
  return { g, order };
}

function reorderMatrix(M, order, n){
  const d = M.length;
  const idx = [];
  for(let i=0;i<n;i++) idx.push(order[i]);
  for(let i=0;i<n;i++) idx.push(n + order[i]);
  const R = [];
  for(let i=0;i<d;i++){
    const row = [];
    for(let j=0;j<d;j++) row.push(M[idx[i]][idx[j]]);
    R.push(row);
  }
  return R;
}

function toggleReorder(){
  gramReordered = !gramReordered;
  const btn = document.getElementById('btn-reorder');
  btn.textContent = gramReordered ? 'Original Order' : 'Re-order';
  goToGram(gramIdx);
}

/* ── fixed-size Cayley heatmap: blue if observed, white otherwise ── */
const HEATMAP = { grid: 520, padTop: 32, padLeft: 32, padRight: 16, padBottom: 16 };
const BLUE = '#1d4ed8';

function drawCayley(canvas, table, trainMask){
  const n = table.length;
  const G = HEATMAP.grid;
  const w = HEATMAP.padLeft + G + HEATMAP.padRight;
  const h = HEATMAP.padTop  + G + HEATMAP.padBottom;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w*dpr; canvas.height = h*dpr;
  canvas.style.width = w+'px'; canvas.style.height = h+'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,w,h);

  const cell = G / n;
  const x0 = HEATMAP.padLeft;
  const y0 = HEATMAP.padTop;

  /* fill observed cells */
  ctx.fillStyle = BLUE;
  for(let i=0;i<n;i++){
    for(let j=0;j<n;j++){
      if(table[i][j] < 0) continue;
      if(trainMask && trainMask[i][j]){
        ctx.fillRect(x0 + j*cell, y0 + i*cell, cell, cell);
      }
    }
  }

  /* grid lines */
  ctx.strokeStyle = '#e7e5e4';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for(let k=0;k<=n;k++){
    const p = Math.round(k*cell) + 0.5;
    ctx.moveTo(x0 + p, y0);
    ctx.lineTo(x0 + p, y0 + G);
    ctx.moveTo(x0,     y0 + p);
    ctx.lineTo(x0 + G, y0 + p);
  }
  ctx.stroke();

  /* outer border */
  ctx.strokeStyle = '#d6d3d1';
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, G - 1, G - 1);

  /* axis labels (only when there's room) */
  const labelStride = Math.max(1, Math.ceil(n / 12));
  if(cell >= 10){
    ctx.fillStyle = CLR.med;
    ctx.font = '600 10px Inter,system-ui,sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    for(let j=0;j<n;j+=labelStride){
      ctx.fillText(j, x0 + j*cell + cell/2, y0 - 14);
    }
    ctx.textAlign = 'right';
    for(let i=0;i<n;i+=labelStride){
      ctx.fillText(i, x0 - 8, y0 + i*cell + cell/2);
    }
  }
}

/* ── diverging heatmap for W1^T W1: blue ← white → red ── */
const GRAM_GRID = 380;

function divColor(t){
  /* t in [-1,1] → blue(neg) white(0) red(pos) */
  t = Math.max(-1, Math.min(1, t));
  const lo=[59,130,246], mid=[255,255,255], hi=[220,38,38];
  let from,to,s;
  if(t<=0){from=lo;to=mid;s=1+t;} else{from=mid;to=hi;s=t;}
  return `rgb(${Math.round(from[0]+(to[0]-from[0])*s)},${Math.round(from[1]+(to[1]-from[1])*s)},${Math.round(from[2]+(to[2]-from[2])*s)})`;
}

function zeroDiag(matrix){
  const d = matrix.length;
  const out = [];
  for(let i=0;i<d;i++){
    const row = matrix[i].slice();
    row[i] = 0;
    out.push(row);
  }
  return out;
}

function drawGram(canvas, matrix){
  const d = matrix.length;
  const G = GRAM_GRID;
  const cbW = 12;
  const cbGap = 8;
  const cbLabelW = 44;
  const pad = {top:8, right:cbGap+cbW+cbGap+cbLabelW, bottom:8, left:8};
  const w = pad.left + G + pad.right;
  const h = pad.top  + G + pad.bottom;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w*dpr; canvas.height = h*dpr;
  canvas.style.width = w+'px'; canvas.style.height = h+'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,w,h);

  const cell = G / d;
  const x0 = pad.left;
  const y0 = pad.top;

  /* find max |value| */
  let maxAbs = 0;
  for(let i=0;i<d;i++) for(let j=0;j<d;j++) maxAbs = Math.max(maxAbs, Math.abs(matrix[i][j]));
  if(maxAbs < 1e-12) maxAbs = 1;

  /* draw cells */
  for(let i=0;i<d;i++){
    for(let j=0;j<d;j++){
      ctx.fillStyle = divColor(matrix[i][j] / maxAbs);
      ctx.fillRect(x0 + j*cell, y0 + i*cell, Math.ceil(cell), Math.ceil(cell));
    }
  }

  /* outer border */
  ctx.strokeStyle = '#d6d3d1'; ctx.lineWidth = 1;
  ctx.strokeRect(x0+0.5, y0+0.5, G-1, G-1);

  /* colorbar */
  const cbX = x0 + G + cbGap;
  for(let y=0;y<G;y++){
    ctx.fillStyle = divColor(1 - 2*y/G);
    ctx.fillRect(cbX, y0+y, cbW, 1);
  }
  ctx.strokeStyle = '#e7e5e4'; ctx.lineWidth = 0.5;
  ctx.strokeRect(cbX, y0, cbW, G);

  const fmt = v => Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=1?v.toFixed(1):v.toFixed(2);
  ctx.fillStyle = CLR.med; ctx.font = '500 10px Inter,system-ui,sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(fmt(maxAbs),  cbX+cbW+4, y0+8);
  ctx.fillText('0',          cbX+cbW+4, y0+G/2+3);
  ctx.fillText(fmt(-maxAbs), cbX+cbW+4, y0+G-2);
}

/* ── gram stepper ── */
function updateGramStepper(){
  if(!checkpoints.length) return;
  const mx = checkpoints.length - 1;
  document.getElementById('gram-label').textContent =
    `Epoch ${checkpoints[gramIdx].epoch} / ${checkpoints[mx].epoch}`;
  document.getElementById('gram-slider').max = mx;
  document.getElementById('gram-slider').value = gramIdx;
  document.getElementById('btn-prev').disabled = gramIdx === 0;
  document.getElementById('btn-next').disabled = gramIdx === mx;
}
function stepGram(d){ goToGram(gramIdx + d); }
function goToGram(i){
  if(!checkpoints.length) return;
  gramIdx = Math.max(0, Math.min(i, checkpoints.length - 1));
  let mat = checkpoints[gramIdx].W1tW1;
  if(gramReordered && dlogOrder && meta){
    mat = reorderMatrix(mat, dlogOrder.order, meta.n);
  }
  drawGram(document.getElementById('gram-canvas'), mat);
  drawGram(document.getElementById('gram-canvas-nodiag'), zeroDiag(mat));
  updateGramStepper();
}

/* ── chart helpers ── */
function mkOpts(xLbl, yLbl, yLog, yMin, yMax){
  const o = {
    responsive:true, maintainAspectRatio:true,
    interaction:{mode:'index',intersect:false},
    plugins:{
      legend:{position:'bottom',labels:{usePointStyle:true,padding:14,font:{size:11,family:"'Inter',sans-serif"}}},
      tooltip:{backgroundColor:'#1c1917',titleFont:{family:"'Inter',sans-serif"},bodyFont:{family:"'Inter',sans-serif"}}
    },
    scales:{
      x:{title:{display:true,text:xLbl,font:{size:11,family:"'Inter',sans-serif"}},grid:{display:false},ticks:{font:{family:"'Inter',sans-serif",size:10}}},
      y:{title:{display:true,text:yLbl,font:{size:11,family:"'Inter',sans-serif"}},grid:{color:'#f0efed'},ticks:{font:{family:"'Inter',sans-serif",size:10}}},
    },
  };
  if(yLog) o.scales.y.type = 'logarithmic';
  if(yMin!=null) o.scales.y.min = yMin;
  if(yMax!=null) o.scales.y.max = yMax;
  return o;
}

function ds(label,color,dash){
  return {label,data:[],borderColor:color,backgroundColor:color,
          fill:false,tension:.25,pointRadius:0,borderWidth:2,borderDash:dash||[]};
}

function initCharts(){
  Object.values(charts).forEach(c=>c.destroy());
  charts = {};
  const mk = (id, datasets, xL, yL, yLog, yMin, yMax) => {
    const ctx = document.getElementById(id).getContext('2d');
    return new Chart(ctx, {type:'line',data:{labels:[],datasets},options:mkOpts(xL,yL,yLog,yMin,yMax)});
  };
  charts.acc  = mk('acc-chart',  [ds('Train accuracy',CLR.dark), ds('Test accuracy',CLR.accent,[5,4])], 'Epoch','Accuracy', false, 0, 1);
  charts.loss = mk('loss-chart', [ds('Train MSE',CLR.dark),      ds('Test MSE',CLR.accent,[5,4])],      'Epoch','MSE',     true);
}

function pushChartData(msg){
  const e = msg.epoch;
  charts.acc.data.labels.push(e);
  charts.acc.data.datasets[0].data.push(msg.train_acc);
  charts.acc.data.datasets[1].data.push(msg.test_acc);
  charts.acc.update('none');

  charts.loss.data.labels.push(e);
  charts.loss.data.datasets[0].data.push(msg.train_loss);
  charts.loss.data.datasets[1].data.push(msg.test_loss);
  charts.loss.update('none');
}

function fmtVal(v){
  if(v==null||isNaN(v)) return '\u2014';
  if(Math.abs(v)<0.0001 && v!==0) return v.toExponential(2);
  if(Math.abs(v)>=10000) return v.toExponential(2);
  return (+v).toFixed(4);
}
function updateMetrics(msg){
  document.getElementById('v-trainloss').textContent = fmtVal(msg.train_loss);
  document.getElementById('v-testloss').textContent  = fmtVal(msg.test_loss);
  document.getElementById('v-trainacc').textContent  = fmtVal(msg.train_acc);
  document.getElementById('v-testacc').textContent   = fmtVal(msg.test_acc);
}

/* ── Cayley preview while editing ── */
async function previewCayley(){
  const op = document.getElementById('sel-op').value;
  const n  = +document.getElementById('in-n').value;
  const needsPrime = OPERATIONS[op].requires_prime;
  document.getElementById('warn-n').style.display = (needsPrime && !isPrime(n)) ? 'block' : 'none';
  if(needsPrime && !isPrime(n)) return;

  try{
    const r = await fetch('/api/cayley', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({operation:op, n})
    });
    if(!r.ok) return;
    const data = await r.json();
    if(!meta || isTraining===false){
      ensureResultsVisible();
      const blank = Array.from({length:data.n},()=>Array(data.n).fill(0));
      drawCayley(document.getElementById('cayley-canvas'), data.table, blank);
      document.getElementById('ct-sub').textContent =
        `${OPERATIONS[op].name}  \u2014  n = ${data.n}  (preview, not yet trained)`;
    }
  }catch(e){}
}

function ensureResultsVisible(){
  document.getElementById('empty-state').style.display = 'none';
  const el = document.getElementById('results');
  if(el.style.display === 'none' || !el.style.display){
    el.style.display = 'flex';
    el.style.flexDirection = 'column';
    el.style.gap = '18px';
    el.classList.add('results-wrap');
  }
  if(!charts.acc) initCharts();
}

/* ── training ── */
async function startTraining(){
  const op = document.getElementById('sel-op').value;
  const n  = +document.getElementById('in-n').value;
  if(OPERATIONS[op].requires_prime && !isPrime(n)){
    alert(`Operation '${OPERATIONS[op].name}' requires n to be prime.`);
    return;
  }
  const config = {
    operation:    op,
    n:            n,
    k:           +document.getElementById('in-k').value,
    activation:   document.getElementById('sel-act').value,
    init_scale:  +document.getElementById('in-scale').value,
    lr:          +document.getElementById('in-lr').value,
    num_epochs:  +document.getElementById('in-epochs').value,
    train_frac:  +document.getElementById('in-trainfrac').value,
    weight_decay:+document.getElementById('in-wd').value,
    batch_size:  +document.getElementById('in-batch').value,
    seed:        +document.getElementById('in-seed').value,
  };

  isTraining = true;
  meta = null;
  checkpoints = [];
  gramIdx = 0;
  gramReordered = false;
  dlogOrder = null;
  document.getElementById('btn-reorder').style.display = 'none';
  document.getElementById('btn-reorder').textContent = 'Re-order';
  document.getElementById('reorder-info').style.display = 'none';
  const btn = document.getElementById('run-btn');
  btn.disabled = true; btn.textContent = 'Training\u2026';

  ensureResultsVisible();
  initCharts();

  try{
    const resp = await fetch('/api/train', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(config)
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buf += decoder.decode(value, {stream:true});
      const parts = buf.split('\n');
      buf = parts.pop();
      for(const line of parts){
        if(!line.startsWith('data: ')) continue;
        const msg = JSON.parse(line.slice(6));
        if(msg.type === 'checkpoint'){
          checkpoints.push(msg);
          gramIdx = checkpoints.length - 1;
          pushChartData(msg);
          updateMetrics(msg);
          drawGram(document.getElementById('gram-canvas'), msg.W1tW1);
          drawGram(document.getElementById('gram-canvas-nodiag'), zeroDiag(msg.W1tW1));
          updateGramStepper();
        } else if(msg.type === 'meta'){
          meta = msg;
          drawCayley(
            document.getElementById('cayley-canvas'),
            msg.table, msg.train_mask
          );
          document.getElementById('ct-sub').textContent =
            `${OPERATIONS[msg.operation].name}  \u2014  n = ${msg.n}`;
          document.getElementById('v-ntrain').textContent =
            `${msg.n_train} / ${msg.n_test}`;
        } else if(msg.type === 'error'){
          alert('Error: ' + msg.message);
        }
      }
    }
  } catch(e){
    alert('Network error: ' + e.message);
  }

  isTraining = false;
  btn.disabled = false; btn.textContent = 'Train Network';

  /* show Re-order button for mul/div */
  if(meta && (meta.operation === 'mul' || meta.operation === 'div')){
    dlogOrder = computeDlogOrder(meta.n);
    const reBtn = document.getElementById('btn-reorder');
    reBtn.style.display = 'inline-block';
    const info = document.getElementById('reorder-info');
    info.style.display = 'inline';
    info.textContent = `generator g = ${dlogOrder.g} (mod ${meta.n})`;
  }
}

/* ── wire up ── */
document.addEventListener('DOMContentLoaded', ()=>{
  const nIn = document.getElementById('in-n');
  const nVal = document.getElementById('val-n');
  nIn.addEventListener('input', ()=>{ nVal.textContent = nIn.value; previewCayley(); });

  const tfIn = document.getElementById('in-trainfrac');
  const tfVal = document.getElementById('val-trainfrac');
  tfIn.addEventListener('input', ()=>{ tfVal.textContent = (+tfIn.value).toFixed(2); });

  document.getElementById('sel-op').addEventListener('change', previewCayley);

  document.addEventListener('keydown', e => {
    if(e.target.tagName==='INPUT' || e.target.tagName==='SELECT') return;
    if(e.key==='ArrowLeft')  stepGram(-1);
    if(e.key==='ArrowRight') stepGram(1);
  });

  previewCayley();
});
</script>
</body>
</html>
'''


if __name__ == '__main__':
    app.run(debug=True, port=5012)
