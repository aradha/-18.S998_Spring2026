from flask import Flask, render_template_string, request, Response, stream_with_context
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import io
import base64
import json
import threading
import queue
from rfm import RFM

app = Flask(__name__)

MODELS = {
    'quadratic_2d': {
        'name': 'Quadratic: (x\u2081 + x\u2082)\u00b2',
        'fn': lambda X: (X[:, 0] + X[:, 1]) ** 2,
        'relevant_dims': [0, 1],
    },
    'product_2d': {
        'name': 'Product: x\u2081 \u00b7 x\u2082',
        'fn': lambda X: X[:, 0] * X[:, 1],
        'relevant_dims': [0, 1],
    },
    'sinusoidal_2d': {
        'name': 'Sinusoidal: sin(x\u2081 + x\u2082)',
        'fn': lambda X: np.sin(X[:, 0] + X[:, 1]),
        'relevant_dims': [0, 1],
    },
    'abs_3d': {
        'name': 'Absolute: |x\u2081 + x\u2082 + x\u2083|',
        'fn': lambda X: np.abs(X[:, 0] + X[:, 1] + X[:, 2]),
        'relevant_dims': [0, 1, 2],
    },
    'gaussian_bump': {
        'name': 'Gaussian Bump: exp(\u2212(x\u2081\u00b2 + x\u2082\u00b2))',
        'fn': lambda X: np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2)),
        'relevant_dims': [0, 1],
    },
    'staircase_3d': {
        'name': 'Staircase: x\u2081 + x\u2081x\u2082 + x\u2081x\u2082x\u2083',
        'fn': lambda X: X[:, 0] + X[:, 0] * X[:, 1] + X[:, 0] * X[:, 1] * X[:, 2],
        'relevant_dims': [0, 1, 2],
    },
}


def _pick_ticks(d, relevant_dims):
    """Choose tick positions that stay readable at any d."""
    if d <= 20:
        step = 1
    elif d <= 50:
        step = 5
    else:
        step = max(10, d // 10)
    ticks = set(range(0, d, step))
    ticks.add(d - 1)
    ticks.update(i for i in relevant_dims if i < d)
    return sorted(ticks)


def generate_agop_image(M, iteration, relevant_dims):
    d = M.shape[0]
    fig_w = max(5, min(8, 3.5 + d * 0.08))
    fig, ax = plt.subplots(figsize=(fig_w + 1.2, fig_w))

    max_abs = np.max(np.abs(M))
    if max_abs < 1e-10:
        max_abs = 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    im = ax.imshow(M, cmap='RdBu_r', norm=norm, aspect='equal')
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    ticks = _pick_ticks(d, relevant_dims)
    labels = [f'x{i+1}' for i in ticks]

    if d <= 20:
        fs, rot, ha = 9, 0, 'center'
    elif d <= 50:
        fs, rot, ha = 8, 45, 'right'
    else:
        fs, rot, ha = 7, 45, 'right'

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=fs, rotation=rot, ha=ha)
    ax.set_yticklabels(labels, fontsize=fs)

    for idx, tick_val in enumerate(ticks):
        if tick_val in relevant_dims:
            ax.get_xticklabels()[idx].set_color('#4361ee')
            ax.get_xticklabels()[idx].set_fontweight('bold')
            ax.get_yticklabels()[idx].set_color('#4361ee')
            ax.get_yticklabels()[idx].set_fontweight('bold')

    if d <= 8:
        for i in range(d):
            for j in range(d):
                val = M[i, j]
                txt_color = 'white' if abs(val) > 0.55 * max_abs else '#333'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=txt_color)

    ax.set_title(f'AGOP Matrix \u2014 Iteration {iteration}',
                 fontsize=11, fontweight='medium', pad=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route('/')
def index():
    return render_template_string(TEMPLATE, models=MODELS)


@app.route('/api/run', methods=['POST'])
def run_rfm():
    data = request.json
    model_key = data.get('model', 'quadratic_2d')
    kernel = data.get('kernel', 'laplace')
    alpha = float(data.get('alpha', 1.0))
    reg = float(data.get('reg', 1e-3))
    bandwidth = float(data.get('bandwidth', 5.0))
    num_iters = int(data.get('num_iters', 5))
    d = max(int(data.get('d', 10)), 3)
    n_train = int(data.get('n_train', 500))
    n_test = int(data.get('n_test', 200))

    relevant_dims = MODELS[model_key]['relevant_dims']
    q = queue.Queue()

    def _work():
        try:
            np.random.seed(42)
            X_train = np.random.randn(n_train, d)
            X_test = np.random.randn(n_test, d)
            model_fn = MODELS[model_key]['fn']
            y_train = model_fn(X_train).reshape(-1, 1)
            y_test = model_fn(X_test).reshape(-1, 1)

            def on_progress(it, total, stage):
                q.put(json.dumps({
                    'type': 'progress', 'iteration': it,
                    'total': total, 'stage': stage,
                }))

            rfm_model = RFM(kernel=kernel)
            rfm_model.fit(X_train, y_train, reg=reg, bandwidth=bandwidth,
                          num_iters=num_iters, alpha=alpha,
                          X_test=X_test, y_test=y_test,
                          progress_callback=on_progress)

            q.put(json.dumps({'type': 'progress', 'iteration': -1,
                              'total': -1, 'stage': 'rendering'}))

            results = []
            for h in rfm_model.get_history():
                agop_img = generate_agop_image(
                    h['M'], h['iteration'], relevant_dims)
                results.append({
                    'iteration': h['iteration'],
                    'train_r2': h['train_r2'],
                    'test_r2': h['test_r2'],
                    'agop_image': agop_img,
                })

            q.put(json.dumps({'type': 'done',
                              'iterations': results,
                              'relevant_dims': relevant_dims}))
        except Exception as exc:
            q.put(json.dumps({'type': 'error', 'message': str(exc)}))

    threading.Thread(target=_work, daemon=True).start()

    def generate():
        while True:
            msg = q.get()
            yield f"data: {msg}\n\n"
            if json.loads(msg)['type'] in ('done', 'error'):
                break

    return Response(stream_with_context(generate()),
                    content_type='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RFM Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:#f3f4f6;color:#1f2937;line-height:1.5}

/* ── header ── */
.header{background:#111827;color:#fff;padding:14px 28px;display:flex;align-items:baseline;gap:14px}
.header h1{font-size:18px;font-weight:600;letter-spacing:-.3px}
.header .sub{font-size:12px;color:#9ca3af}

/* ── layout ── */
.container{display:grid;grid-template-columns:300px 1fr;gap:20px;padding:20px;max-width:1440px;margin:0 auto}
.sidebar{display:flex;flex-direction:column;gap:14px}

/* ── cards ── */
.card{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);padding:18px}
.card-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:#6b7280;margin-bottom:14px}

/* ── forms ── */
.fg{margin-bottom:12px}
.fg:last-child{margin-bottom:0}
.fg label{display:block;font-size:12px;font-weight:500;color:#374151;margin-bottom:3px}
.fg select,.fg input{width:100%;padding:7px 10px;border:1px solid #d1d5db;border-radius:6px;font-size:13px;color:#1f2937;background:#fafafa;transition:border-color .2s,box-shadow .2s}
.fg select:focus,.fg input:focus{outline:none;border-color:#4f46e5;box-shadow:0 0 0 3px rgba(79,70,229,.1)}

.btn{width:100%;padding:10px;background:#4f46e5;color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:background .15s}
.btn:hover{background:#4338ca}
.btn:disabled{background:#9ca3af;cursor:not-allowed}

/* ── main ── */
.main{display:flex;flex-direction:column;gap:18px}

/* ── metrics strip ── */
.metrics{display:flex;gap:28px;justify-content:center;padding:6px 0}
.metric-val{font-size:26px;font-weight:700;color:#111827;text-align:center}
.metric-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#6b7280;text-align:center}

/* ── stepper ── */
.stepper{display:flex;align-items:center;justify-content:center;gap:14px;padding:8px 0}
.step-btn{width:34px;height:34px;border-radius:50%;border:1px solid #d1d5db;background:#fff;color:#374151;font-size:18px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all .15s;user-select:none}
.step-btn:hover:not(:disabled){border-color:#4f46e5;color:#4f46e5;background:#eef2ff}
.step-btn:disabled{opacity:.35;cursor:not-allowed}
.step-label{font-size:13px;font-weight:600;min-width:120px;text-align:center;color:#374151}
.step-slider{flex:1;max-width:180px;accent-color:#4f46e5}

/* ── viz ── */
.viz-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.viz-card{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);padding:18px}
.viz-card h3{font-size:13px;font-weight:600;color:#374151;margin-bottom:12px}
.agop-img{width:100%;border-radius:4px}

/* ── empty / loading ── */
.empty{text-align:center;padding:80px 20px;color:#9ca3af}
.empty h3{font-size:16px;color:#6b7280;margin-bottom:6px}
.empty p{font-size:13px}

.overlay{display:none;position:fixed;inset:0;background:rgba(255,255,255,.85);z-index:100;justify-content:center;align-items:center;flex-direction:column;gap:0}
.overlay.active{display:flex}
.prog-wrap{width:340px;text-align:center}
.prog-track{height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;margin-bottom:14px}
.prog-fill{height:100%;background:#4f46e5;border-radius:3px;transition:width .35s ease;width:0%}
.prog-text{font-size:13px;color:#6b7280;margin:0}

/* ── results fade-in ── */
.results-wrap{animation:fadeUp .35s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* ── keyboard hint ── */
.hint{text-align:center;font-size:11px;color:#9ca3af;margin-top:-6px}
</style>
</head>
<body>

<div class="header">
  <h1>RFM Explorer</h1>
  <span class="sub">Recursive Feature Machines &mdash; AGOP Visualization</span>
</div>

<div class="container">
  <!-- ─── sidebar ─── -->
  <div class="sidebar">
    <div class="card">
      <div class="card-title">Model</div>
      <div class="fg">
        <label for="sel-model">Target Function</label>
        <select id="sel-model">
          {% for key, m in models.items() %}
          <option value="{{ key }}">{{ m.name }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="fg">
        <label for="sel-kernel">Kernel</label>
        <select id="sel-kernel">
          <option value="laplace">Laplace</option>
          <option value="gaussian">Gaussian</option>
        </select>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Hyperparameters</div>
      <div class="fg">
        <label for="in-alpha">AGOP Power (&alpha;)</label>
        <input type="number" id="in-alpha" value="1.0" step="0.1" min="0.1" max="5.0">
      </div>
      <div class="fg">
        <label for="in-reg">Regularization (&lambda;)</label>
        <input type="number" id="in-reg" value="0.001" step="0.0001" min="0.0001">
      </div>
      <div class="fg">
        <label for="in-bw">Bandwidth</label>
        <input type="number" id="in-bw" value="5" step="0.5" min="0.1">
      </div>
      <div class="fg">
        <label for="in-iters">Iterations</label>
        <input type="number" id="in-iters" value="5" step="1" min="1" max="20">
      </div>
    </div>

    <div class="card">
      <div class="card-title">Data</div>
      <div class="fg">
        <label for="in-d">Input Dimension (d)</label>
        <input type="number" id="in-d" value="10" step="1" min="3" max="50">
      </div>
      <div class="fg">
        <label for="in-ntrain">Training Samples</label>
        <input type="number" id="in-ntrain" value="500" step="50" min="50">
      </div>
      <div class="fg">
        <label for="in-ntest">Test Samples</label>
        <input type="number" id="in-ntest" value="200" step="50" min="50">
      </div>
    </div>

    <button class="btn" id="run-btn" onclick="runRFM()">Run RFM</button>
  </div>

  <!-- ─── main ─── -->
  <div class="main">
    <div class="empty" id="empty-state">
      <h3>Configure &amp; Run</h3>
      <p>Set parameters in the sidebar, then click <strong>Run RFM</strong>.</p>
    </div>

    <div id="results" style="display:none" class="results-wrap">
      <!-- metrics -->
      <div class="card">
        <div class="metrics">
          <div><div class="metric-val" id="v-train">&mdash;</div><div class="metric-lbl">Train R&sup2;</div></div>
          <div><div class="metric-val" id="v-test">&mdash;</div><div class="metric-lbl">Test R&sup2;</div></div>
        </div>
      </div>

      <!-- stepper -->
      <div class="card">
        <div class="stepper">
          <button class="step-btn" id="btn-prev" onclick="step(-1)">&lsaquo;</button>
          <span class="step-label" id="step-label">Iteration 0 / 0</span>
          <input type="range" class="step-slider" id="step-slider" min="0" max="0" value="0"
                 oninput="goTo(+this.value)">
          <button class="step-btn" id="btn-next" onclick="step(1)">&rsaquo;</button>
        </div>
        <div class="hint">Use &larr; / &rarr; arrow keys to step through iterations</div>
      </div>

      <!-- viz -->
      <div class="viz-grid">
        <div class="viz-card">
          <h3>AGOP Matrix</h3>
          <img id="agop-img" class="agop-img" alt="AGOP heatmap">
        </div>
        <div class="viz-card">
          <h3>R&sup2; by Iteration</h3>
          <canvas id="r2-chart"></canvas>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- loading overlay -->
<div class="overlay" id="overlay">
  <div class="prog-wrap">
    <div class="prog-track"><div class="prog-fill" id="prog-fill"></div></div>
    <p class="prog-text" id="prog-text">Initializing&hellip;</p>
  </div>
</div>

<script>
let res = null, cur = 0, chart = null;

/* ── suggested bandwidth on kernel change ── */
document.getElementById('sel-kernel').addEventListener('change', function(){
  document.getElementById('in-bw').value = this.value === 'gaussian' ? 2 : 5;
});

/* ── progress helpers ── */
function setProgress(pct, text){
  document.getElementById('prog-fill').style.width = pct+'%';
  document.getElementById('prog-text').textContent = text;
}

/* ── run (SSE streaming) ── */
async function runRFM(){
  const btn = document.getElementById('run-btn');
  const ov  = document.getElementById('overlay');
  btn.disabled = true; ov.classList.add('active');
  setProgress(0, 'Initializing\u2026');

  const body = {
    model:    document.getElementById('sel-model').value,
    kernel:   document.getElementById('sel-kernel').value,
    alpha:    +document.getElementById('in-alpha').value,
    reg:      +document.getElementById('in-reg').value,
    bandwidth:+document.getElementById('in-bw').value,
    num_iters:+document.getElementById('in-iters').value,
    d:        +document.getElementById('in-d').value,
    n_train:  +document.getElementById('in-ntrain').value,
    n_test:   +document.getElementById('in-ntest').value,
  };
  const totalIters = body.num_iters;

  try {
    const resp = await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
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
        if(msg.type==='progress'){
          if(msg.stage==='rendering'){
            setProgress(85, 'Generating visualizations\u2026');
          } else {
            const pct = Math.round((msg.iteration / (totalIters+1))*80);
            const label = msg.stage==='solving'
              ? `Iteration ${msg.iteration}/${totalIters} \u2014 solving`
              : `Iteration ${msg.iteration}/${totalIters} \u2014 computing AGOP`;
            setProgress(pct, label);
          }
        } else if(msg.type==='done'){
          setProgress(100, 'Done');
          res = msg; cur = 0;
          document.getElementById('empty-state').style.display = 'none';
          const el = document.getElementById('results');
          el.style.display='flex'; el.style.flexDirection='column'; el.style.gap='18px';
          el.classList.remove('results-wrap'); void el.offsetWidth; el.classList.add('results-wrap');
          const mx = res.iterations.length-1;
          const sl = document.getElementById('step-slider');
          sl.max=mx; sl.value=0;
          buildChart(); render();
        } else if(msg.type==='error'){
          alert('Error: '+msg.message);
        }
      }
    }
  } catch(e){ alert('Error: '+e.message); }
  finally { btn.disabled=false; ov.classList.remove('active'); }
}

/* ── navigation ── */
function step(d){ goTo(cur+d); }
function goTo(i){
  if(!res) return;
  cur = Math.max(0, Math.min(i, res.iterations.length-1));
  render();
}

/* ── render current iteration ── */
function render(){
  if(!res) return;
  const it = res.iterations[cur], mx = res.iterations.length-1;
  document.getElementById('step-label').textContent = `Iteration ${it.iteration} / ${mx}`;
  document.getElementById('step-slider').value = cur;
  document.getElementById('v-train').textContent = it.train_r2.toFixed(4);
  document.getElementById('v-test').textContent  = it.test_r2.toFixed(4);
  document.getElementById('agop-img').src = 'data:image/png;base64,' + it.agop_image;
  document.getElementById('btn-prev').disabled = cur === 0;
  document.getElementById('btn-next').disabled = cur === mx;
  highlightPoint();
}

/* ── R² chart ── */
function buildChart(){
  const ctx = document.getElementById('r2-chart').getContext('2d');
  if(chart) chart.destroy();

  const labels   = res.iterations.map(i=>i.iteration);
  const testR2   = res.iterations.map(i=>i.test_r2);
  const trainR2  = res.iterations.map(i=>i.train_r2);
  const lo = Math.min(0, ...testR2, ...trainR2);

  chart = new Chart(ctx, {
    type:'line',
    data:{
      labels,
      datasets:[
        {label:'Test R\u00b2',  data:testR2,  borderColor:'#4f46e5',backgroundColor:'rgba(79,70,229,.08)',fill:true,tension:.3,pointRadius:5,pointHoverRadius:7,borderWidth:2},
        {label:'Train R\u00b2', data:trainR2, borderColor:'#9ca3af',backgroundColor:'rgba(156,163,175,.05)',fill:true,tension:.3,pointRadius:5,pointHoverRadius:7,borderWidth:2,borderDash:[5,4]},
      ]
    },
    options:{
      responsive:true,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{position:'bottom',labels:{usePointStyle:true,padding:14,font:{size:11}}}},
      scales:{
        x:{title:{display:true,text:'Iteration',font:{size:11}},grid:{display:false}},
        y:{title:{display:true,text:'R\u00b2',font:{size:11}},min:lo-.1,max:1.05,grid:{color:'#f3f4f6'}}
      },
      onClick(e){
        const pts = chart.getElementsAtEventForMode(e,'nearest',{intersect:true},false);
        if(pts.length) goTo(pts[0].index);
      }
    }
  });
}

function highlightPoint(){
  if(!chart) return;
  chart.data.datasets.forEach(ds=>{
    const base = ds.borderColor;
    ds.pointBackgroundColor = ds.data.map((_,i)=> i===cur?'#ef4444':base);
    ds.pointBorderColor     = ds.data.map((_,i)=> i===cur?'#ef4444':base);
    ds.pointRadius          = ds.data.map((_,i)=> i===cur?8:5);
  });
  chart.update('none');
}

/* ── keyboard ── */
document.addEventListener('keydown', e=>{
  if(e.key==='ArrowLeft')  step(-1);
  if(e.key==='ArrowRight') step(1);
});
</script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=5000)
