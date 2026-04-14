from flask import Flask, render_template_string, request, Response, stream_with_context
import numpy as np
import json
import threading
import queue
from train import MODELS, train

app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(TEMPLATE, models=MODELS)


@app.route('/api/train', methods=['POST'])
def train_endpoint():
    data = request.json
    model_key = data.get('model', 'quadratic_2d')
    width = int(data.get('width', 200))
    init_scale = float(data.get('init_scale', 0.5))
    lr = float(data.get('lr', 0.01))
    num_epochs = int(data.get('num_epochs', 5000))
    d = max(3, int(data.get('d', 10)))
    n_train = int(data.get('n_train', 500))
    n_test = int(data.get('n_test', 200))

    track_every = max(1, num_epochs // 100)
    q = queue.Queue()

    def _work():
        try:
            def on_progress(epoch, total, checkpoint):
                msg = {
                    'type': 'checkpoint',
                    'epoch': checkpoint['epoch'],
                    'total': total,
                    'train_loss': checkpoint['train_loss'],
                    'test_loss': checkpoint['test_loss'],
                    'train_r2': checkpoint['train_r2'],
                    'test_r2': checkpoint['test_r2'],
                    'corr_agop_BtB': checkpoint['corr_agop_BtB'],
                    'corr_agopsqrt_BtB': checkpoint['corr_agopsqrt_BtB'],
                    'BtB': checkpoint['BtB'].tolist(),
                    'agop': checkpoint['agop'].tolist(),
                    'agop_sqrt': checkpoint['agop_sqrt'].tolist(),
                }
                q.put(json.dumps(msg))

            train(
                model_key=model_key,
                width=width,
                init_scale=init_scale,
                lr=lr,
                num_epochs=num_epochs,
                d=d,
                n_train=n_train,
                n_test=n_test,
                track_every=track_every,
                progress_callback=on_progress,
            )
            q.put(json.dumps({'type': 'done'}))
        except Exception as e:
            q.put(json.dumps({'type': 'error', 'message': str(e)}))

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
<title>NFA Validation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:#f5f5f4;color:#1c1917;line-height:1.5}

/* header */
.header{background:#292524;color:#fafaf9;padding:16px 28px;display:flex;align-items:baseline;gap:14px;border-bottom:1px solid #1c1917}
.header h1{font-size:18px;font-weight:600;letter-spacing:-.3px}
.header .sub{font-size:12px;color:#a8a29e}

/* layout */
.container{display:grid;grid-template-columns:300px 1fr;gap:20px;padding:20px;max-width:1560px;margin:0 auto}
.sidebar{display:flex;flex-direction:column;gap:14px}

/* cards */
.card{background:#fff;border:1px solid #e7e5e4;border-radius:10px;padding:18px}
.card-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:#78716c;margin-bottom:14px}

/* forms */
.fg{margin-bottom:12px}
.fg:last-child{margin-bottom:0}
.fg label{display:block;font-size:12px;font-weight:500;color:#44403c;margin-bottom:3px}
.fg select,.fg input[type="number"]{width:100%;padding:7px 10px;border:1px solid #d6d3d1;border-radius:6px;font-size:13px;color:#1c1917;background:#fafaf9;font-family:inherit;transition:border-color .15s,box-shadow .15s}
.fg select:focus,.fg input:focus{outline:none;border-color:#78716c;box-shadow:0 0 0 3px rgba(120,113,108,.1)}

.btn{width:100%;padding:10px;background:#292524;color:#fafaf9;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;font-family:inherit;transition:background .15s}
.btn:hover{background:#1c1917}
.btn:disabled{background:#a8a29e;cursor:not-allowed}

/* main */
.main{display:flex;flex-direction:column;gap:18px}

/* metrics */
.metrics{display:flex;gap:24px;justify-content:center;padding:8px 0;flex-wrap:wrap}
.metric{text-align:center;min-width:110px}
.metric-val{font-size:20px;font-weight:700;color:#1c1917;font-variant-numeric:tabular-nums}
.metric-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:#78716c;margin-top:2px}

/* charts */
.chart-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.chart-card{background:#fff;border:1px solid #e7e5e4;border-radius:10px;padding:16px}
.chart-card h3{font-size:12px;font-weight:600;color:#57534e;margin-bottom:10px}

/* stepper */
.stepper{display:flex;align-items:center;justify-content:center;gap:14px;padding:8px 0}
.step-btn{width:34px;height:34px;border-radius:50%;border:1px solid #d6d3d1;background:#fff;color:#44403c;font-size:18px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all .15s;user-select:none;font-family:inherit}
.step-btn:hover:not(:disabled){border-color:#78716c;color:#292524;background:#f5f5f4}
.step-btn:disabled{opacity:.35;cursor:not-allowed}
.step-label{font-size:13px;font-weight:600;min-width:180px;text-align:center;color:#44403c}
.step-slider{flex:1;max-width:220px;accent-color:#57534e}

/* heatmaps */
.heatmap-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.heatmap-card{background:#fff;border:1px solid #e7e5e4;border-radius:10px;padding:16px;display:flex;flex-direction:column;align-items:center}
.heatmap-card h3{font-size:12px;font-weight:600;color:#57534e;margin-bottom:10px;align-self:flex-start}

/* states */
.empty{text-align:center;padding:80px 20px;color:#a8a29e}
.empty h3{font-size:16px;color:#78716c;margin-bottom:6px}
.empty p{font-size:13px}
.results-wrap{animation:fadeUp .35s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.hint{text-align:center;font-size:11px;color:#a8a29e;margin-top:-4px}

/* responsive */
@media(max-width:1100px){.chart-row,.heatmap-row{grid-template-columns:1fr 1fr}}
@media(max-width:800px){.container{grid-template-columns:1fr}.chart-row,.heatmap-row{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="header">
  <h1>NFA Validation</h1>
  <span class="sub">Neural Feature Ansatz &mdash; AGOP(f) vs B<sup>T</sup>B</span>
</div>

<div class="container">
  <!-- sidebar -->
  <div class="sidebar">
    <div class="card">
      <div class="card-title">Target Function</div>
      <div class="fg">
        <label for="sel-model">Multi-Index Model</label>
        <select id="sel-model">
          {% for key, m in models.items() %}
          <option value="{{ key }}">{{ m.name }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Network</div>
      <div class="fg">
        <label for="in-width">Hidden Width (m)</label>
        <input type="number" id="in-width" value="200" step="10" min="10" max="2000">
      </div>
      <div class="fg">
        <label for="in-scale">Initialization Scale</label>
        <input type="number" id="in-scale" value="0.5" step="0.05" min="0.01" max="5">
      </div>
    </div>

    <div class="card">
      <div class="card-title">Training</div>
      <div class="fg">
        <label for="in-lr">Learning Rate</label>
        <input type="number" id="in-lr" value="0.01" step="0.001" min="0.0001" max="1">
      </div>
      <div class="fg">
        <label for="in-epochs">Epochs</label>
        <input type="number" id="in-epochs" value="5000" step="500" min="100" max="50000">
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
        <input type="number" id="in-ntrain" value="500" step="50" min="50" max="10000">
      </div>
      <div class="fg">
        <label for="in-ntest">Test Samples</label>
        <input type="number" id="in-ntest" value="200" step="50" min="50" max="5000">
      </div>
    </div>

    <button class="btn" id="run-btn" onclick="startTraining()">Train Network</button>
  </div>

  <!-- main -->
  <div class="main">
    <div class="empty" id="empty-state">
      <h3>Configure &amp; Train</h3>
      <p>Set parameters in the sidebar, then click <strong>Train Network</strong>.</p>
    </div>

    <div id="results" style="display:none">
      <!-- metrics -->
      <div class="card">
        <div class="metrics">
          <div class="metric"><div class="metric-val" id="v-train">&mdash;</div><div class="metric-lbl">Train Loss</div></div>
          <div class="metric"><div class="metric-val" id="v-test">&mdash;</div><div class="metric-lbl">Test Loss</div></div>
          <div class="metric"><div class="metric-val" id="v-trainr2">&mdash;</div><div class="metric-lbl">Train R&sup2;</div></div>
          <div class="metric"><div class="metric-val" id="v-testr2">&mdash;</div><div class="metric-lbl">Test R&sup2;</div></div>
          <div class="metric"><div class="metric-val" id="v-corr1">&mdash;</div><div class="metric-lbl">corr(AGOP, B<sup>T</sup>B)</div></div>
          <div class="metric"><div class="metric-val" id="v-corr2">&mdash;</div><div class="metric-lbl">corr(&radic;AGOP, B<sup>T</sup>B)</div></div>
        </div>
      </div>

      <!-- charts -->
      <div class="chart-row">
        <div class="chart-card">
          <h3>Training Loss</h3>
          <canvas id="loss-chart"></canvas>
        </div>
        <div class="chart-card">
          <h3>corr(AGOP, B<sup>T</sup>B)</h3>
          <canvas id="corr1-chart"></canvas>
        </div>
        <div class="chart-card">
          <h3>corr(&radic;AGOP, B<sup>T</sup>B)</h3>
          <canvas id="corr2-chart"></canvas>
        </div>
      </div>

      <!-- stepper -->
      <div class="card">
        <div class="stepper">
          <button class="step-btn" id="btn-prev" onclick="step(-1)">&lsaquo;</button>
          <span class="step-label" id="step-label">Epoch 0 / 0</span>
          <input type="range" class="step-slider" id="step-slider" min="0" max="0" value="0"
                 oninput="goTo(+this.value)">
          <button class="step-btn" id="btn-next" onclick="step(1)">&rsaquo;</button>
        </div>
        <div class="hint">Use &larr; / &rarr; arrow keys to navigate epochs</div>
      </div>

      <!-- heatmaps -->
      <div class="heatmap-row">
        <div class="heatmap-card">
          <h3>B<sup>T</sup>B</h3>
          <canvas id="btb-canvas"></canvas>
        </div>
        <div class="heatmap-card">
          <h3>AGOP(f)</h3>
          <canvas id="agop-canvas"></canvas>
        </div>
        <div class="heatmap-card">
          <h3>&radic;AGOP(f)</h3>
          <canvas id="agopsqrt-canvas"></canvas>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let checkpoints = [];
let cur = 0;
let isTraining = false;
let charts = {};
let trainStart = 0;

const CLR = { dark:'#292524', med:'#78716c', light:'#a8a29e' };

/* ── diverging color: blue → white → red ── */
function heatColor(t) {
  const lo=[74,106,158], mid=[255,255,255], hi=[158,74,74];
  let from,to,s;
  if(t<=0){from=lo;to=mid;s=1+t;} else{from=mid;to=hi;s=t;}
  return `rgb(${Math.round(from[0]+(to[0]-from[0])*s)},${Math.round(from[1]+(to[1]-from[1])*s)},${Math.round(from[2]+(to[2]-from[2])*s)})`;
}

/* ── chart helpers ── */
function mkOpts(xLbl, yLbl, onClick, yLog){
  const o = {
    responsive:true, maintainAspectRatio:true,
    interaction:{mode:'index',intersect:false},
    plugins:{legend:{position:'bottom',labels:{usePointStyle:true,padding:14,font:{size:11,family:"'Inter',sans-serif"}}}},
    scales:{
      x:{title:{display:true,text:xLbl,font:{size:11,family:"'Inter',sans-serif"}},grid:{display:false}},
      y:{title:{display:true,text:yLbl,font:{size:11,family:"'Inter',sans-serif"}},grid:{color:'#f0efed'}},
    },
    onClick:onClick,
  };
  if(yLog) o.scales.y.type='logarithmic';
  return o;
}

function chartClick(chart,e){
  const pts=chart.getElementsAtEventForMode(e,'nearest',{intersect:false},false);
  if(pts.length) goTo(pts[0].index);
}

function ds(label,color,dash){
  return {label,data:[],borderColor:color,backgroundColor:color.replace(')',',0.05)').replace('rgb','rgba'),
          fill:false,tension:.25,pointRadius:0,borderWidth:2,borderDash:dash||[]};
}

function initCharts(){
  Object.values(charts).forEach(c=>c.destroy());
  charts={};
  const mk=(id,datasets,xL,yL,yLog)=>{
    const ctx=document.getElementById(id).getContext('2d');
    return new Chart(ctx,{type:'line',data:{labels:[],datasets},options:mkOpts(xL,yL,function(e){chartClick(this,e)},yLog)});
  };
  charts.loss  = mk('loss-chart',  [ds('Train',CLR.dark),ds('Test',CLR.light,[5,4])], 'Epoch','MSE Loss',true);
  charts.corr1 = mk('corr1-chart', [ds('Correlation',CLR.dark)],                     'Epoch','Pearson r',false);
  charts.corr2 = mk('corr2-chart', [ds('Correlation',CLR.dark)],                     'Epoch','Pearson r',false);
}

function pushChartData(msg){
  charts.loss.data.labels.push(msg.epoch);
  charts.loss.data.datasets[0].data.push(msg.train_loss);
  charts.loss.data.datasets[1].data.push(msg.test_loss);
  charts.loss.update('none');
  charts.corr1.data.labels.push(msg.epoch);
  charts.corr1.data.datasets[0].data.push(msg.corr_agop_BtB);
  charts.corr1.update('none');
  charts.corr2.data.labels.push(msg.epoch);
  charts.corr2.data.datasets[0].data.push(msg.corr_agopsqrt_BtB);
  charts.corr2.update('none');
}

function highlightCharts(){
  [charts.loss,charts.corr1,charts.corr2].forEach(chart=>{
    chart.data.datasets.forEach(d=>{
      d.pointRadius         = d.data.map((_,i)=>i===cur?6:0);
      d.pointBackgroundColor= d.data.map((_,i)=>i===cur?'#b91c1c':d.borderColor);
      d.pointBorderColor    = d.data.map((_,i)=>i===cur?'#b91c1c':d.borderColor);
    });
    chart.update('none');
  });
}

/* ── heatmap renderer ── */
function renderHeatmap(canvasId, matrix){
  const canvas=document.getElementById(canvasId);
  const d=matrix.length;
  const cellSize=Math.max(4,Math.min(36,Math.floor(300/d)));
  const pad={top:4,right:56,bottom:4,left:4};
  const gridW=d*cellSize, gridH=d*cellSize;
  const w=gridW+pad.left+pad.right, h=gridH+pad.top+pad.bottom;
  const dpr=window.devicePixelRatio||1;
  canvas.width=w*dpr; canvas.height=h*dpr;
  canvas.style.width=w+'px'; canvas.style.height=h+'px';
  const ctx=canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,w,h);

  let maxAbs=0;
  for(let i=0;i<d;i++) for(let j=0;j<d;j++) maxAbs=Math.max(maxAbs,Math.abs(matrix[i][j]));
  if(maxAbs<1e-10) maxAbs=1;

  for(let i=0;i<d;i++){
    for(let j=0;j<d;j++){
      ctx.fillStyle=heatColor(Math.max(-1,Math.min(1,matrix[i][j]/maxAbs)));
      ctx.fillRect(pad.left+j*cellSize, pad.top+i*cellSize, cellSize-.5, cellSize-.5);
    }
  }

  /* colorbar */
  const cbX=pad.left+gridW+8, cbW=10;
  for(let y=0;y<gridH;y++){
    ctx.fillStyle=heatColor(1-2*y/gridH);
    ctx.fillRect(cbX,pad.top+y,cbW,1);
  }
  ctx.strokeStyle='#e7e5e4'; ctx.lineWidth=.5;
  ctx.strokeRect(cbX,pad.top,cbW,gridH);

  const fmt=v=>Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=1?v.toFixed(1):v.toFixed(2);
  ctx.fillStyle=CLR.light; ctx.font='400 9px Inter,system-ui,sans-serif'; ctx.textAlign='left';
  ctx.fillText(fmt(maxAbs),  cbX+cbW+3, pad.top+8);
  ctx.fillText(fmt(-maxAbs), cbX+cbW+3, pad.top+gridH);
  ctx.fillText('0',          cbX+cbW+3, pad.top+gridH/2+3);
}

function renderAllHeatmaps(idx){
  if(!checkpoints.length) return;
  const cp=checkpoints[idx];
  renderHeatmap('btb-canvas',     cp.BtB);
  renderHeatmap('agop-canvas',    cp.agop);
  renderHeatmap('agopsqrt-canvas',cp.agop_sqrt);
}

/* ── metrics ── */
function fmtVal(v){
  if(v==null||isNaN(v)) return '\u2014';
  if(Math.abs(v)<0.0001&&v!==0) return v.toExponential(2);
  if(Math.abs(v)>=10000) return v.toExponential(2);
  return v.toFixed(4);
}
function updateMetrics(idx){
  if(!checkpoints.length) return;
  const cp=checkpoints[idx];
  document.getElementById('v-train').textContent=fmtVal(cp.train_loss);
  document.getElementById('v-test').textContent=fmtVal(cp.test_loss);
  document.getElementById('v-trainr2').textContent=fmtVal(cp.train_r2);
  document.getElementById('v-testr2').textContent=fmtVal(cp.test_r2);
  document.getElementById('v-corr1').textContent=fmtVal(cp.corr_agop_BtB);
  document.getElementById('v-corr2').textContent=fmtVal(cp.corr_agopsqrt_BtB);
}

/* ── stepper ── */
function updateStepper(){
  if(!checkpoints.length) return;
  const mx=checkpoints.length-1;
  let lbl=`Epoch ${checkpoints[cur].epoch} / ${checkpoints[mx].epoch}`;
  if(!isTraining){
    const sec=((performance.now()-trainStart)/1000).toFixed(1);
    lbl+=` \u2014 ${sec}s`;
  }
  document.getElementById('step-label').textContent=lbl;
  document.getElementById('step-slider').max=mx;
  document.getElementById('step-slider').value=cur;
  document.getElementById('btn-prev').disabled=cur===0;
  document.getElementById('btn-next').disabled=cur===mx;
}
function step(d){goTo(cur+d)}
function goTo(i){
  if(!checkpoints.length) return;
  cur=Math.max(0,Math.min(i,checkpoints.length-1));
  updateMetrics(cur);
  renderAllHeatmaps(cur);
  updateStepper();
  if(!isTraining) highlightCharts();
}

/* ── training ── */
async function startTraining(){
  const config={
    model:      document.getElementById('sel-model').value,
    width:     +document.getElementById('in-width').value,
    init_scale:+document.getElementById('in-scale').value,
    lr:        +document.getElementById('in-lr').value,
    num_epochs:+document.getElementById('in-epochs').value,
    d:         +document.getElementById('in-d').value,
    n_train:   +document.getElementById('in-ntrain').value,
    n_test:    +document.getElementById('in-ntest').value,
  };
  checkpoints=[]; cur=0; isTraining=true;
  trainStart=performance.now();

  const btn=document.getElementById('run-btn');
  btn.disabled=true; btn.textContent='Training\u2026';

  document.getElementById('empty-state').style.display='none';
  const el=document.getElementById('results');
  el.style.display='flex'; el.style.flexDirection='column'; el.style.gap='18px';
  el.classList.remove('results-wrap'); void el.offsetWidth; el.classList.add('results-wrap');

  initCharts();

  try{
    const resp=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});
    const reader=resp.body.getReader();
    const decoder=new TextDecoder();
    let buf='';

    while(true){
      const{done,value}=await reader.read();
      if(done) break;
      buf+=decoder.decode(value,{stream:true});
      const parts=buf.split('\n');
      buf=parts.pop();
      for(const line of parts){
        if(!line.startsWith('data: ')) continue;
        const msg=JSON.parse(line.slice(6));
        if(msg.type==='checkpoint'){
          checkpoints.push(msg);
          cur=checkpoints.length-1;
          pushChartData(msg);
          renderAllHeatmaps(cur);
          updateMetrics(cur);
          updateStepper();
        } else if(msg.type==='error'){
          alert('Error: '+msg.message);
        }
      }
    }
  }catch(e){alert('Network error: '+e.message)}

  isTraining=false;
  btn.disabled=false; btn.textContent='Train Network';
  if(checkpoints.length) highlightCharts();
  updateStepper();
}

/* keyboard nav */
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if(e.key==='ArrowLeft')  step(-1);
  if(e.key==='ArrowRight') step(1);
});
</script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=5010)
