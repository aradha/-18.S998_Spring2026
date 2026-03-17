"""
NN Feature Learning webapp.
Run: python app.py
Then open http://localhost:5001
"""

import threading
import webbrowser
import numpy as np
from flask import Flask, request, jsonify, Response
from backend import (
    sample_data, kernel_regression_ntk, train_network,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
data_store = {}
training_state = {
    "running": False,
    "train_losses": [],
    "test_losses": [],
    "epoch": 0,
    "total_epochs": 0,
    "w1tw1": None,
    "train_r2": None,
    "test_r2": None,
    "error": None,
}
training_lock = threading.Lock()

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/generate_data", methods=["POST"])
def api_generate_data():
    p = request.json
    d = int(p.get("d", 100))
    n_train = int(p.get("n_train", 2000))
    n_test = int(p.get("n_test", 10000))
    active_indices = p.get("active_indices", [0, 1])
    func_type = p.get("func_type", "product")
    seed = int(p.get("seed", 1717))

    X_train, y_train = sample_data(n_train, d, active_indices, func_type, seed=seed)
    X_test, y_test = sample_data(n_test, d, active_indices, func_type, seed=seed + 1)

    data_store.update(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    idx0 = active_indices[0] if active_indices else 0
    scatter_x = X_train[:300, idx0].tolist()
    scatter_y = y_train[:300, 0].tolist()
    idx1 = active_indices[1] if len(active_indices) > 1 else None
    scatter_x2 = X_train[:300, idx1].tolist() if idx1 is not None else None

    return jsonify(
        status="ok", n_train=n_train, n_test=n_test, d=d,
        active_indices=active_indices, func_type=func_type,
        y_train_mean=float(y_train.mean()), y_train_std=float(y_train.std()),
        scatter_x=scatter_x, scatter_x2=scatter_x2, scatter_y=scatter_y,
    )


@app.route("/api/kernel_regression", methods=["POST"])
def api_kernel_regression():
    if "X_train" not in data_store:
        return jsonify(error="Generate data first"), 400
    p = request.json
    reg = float(p.get("reg", 1e-3))
    X_tr, y_tr = data_store["X_train"], data_store["y_train"]
    X_te, y_te = data_store["X_test"], data_store["y_test"]

    train_r2, test_r2 = kernel_regression_ntk(X_tr, y_tr, X_te, y_te, reg=reg)
    return jsonify(ntk_train_r2=train_r2, ntk_test_r2=test_r2)


@app.route("/api/kernel_active", methods=["POST"])
def api_kernel_active():
    """NTK regression using only the active coordinate subset."""
    if "X_train" not in data_store:
        return jsonify(error="Generate data first"), 400
    p = request.json
    active = p.get("active_indices")
    reg = float(p.get("reg", 1e-3))
    if not active:
        return jsonify(error="No active indices provided"), 400

    X_tr = data_store["X_train"][:, active]
    X_te = data_store["X_test"][:, active]
    y_tr, y_te = data_store["y_train"], data_store["y_test"]

    train_r2, test_r2 = kernel_regression_ntk(X_tr, y_tr, X_te, y_te, reg=reg)
    return jsonify(train_r2=train_r2, test_r2=test_r2)


@app.route("/api/train_nn", methods=["POST"])
def api_train_nn():
    if "X_train" not in data_store:
        return jsonify(error="Generate data first"), 400
    if training_state["running"]:
        return jsonify(error="Training already in progress"), 409

    p = request.json
    width = int(p.get("width", 128))
    raw_init = float(p.get("init_scale", 0))
    init_scale = raw_init if raw_init > 0 else None  # 0 means default PyTorch init
    lr = float(p.get("lr", 0.1))
    num_epochs = int(p.get("num_epochs", 50))
    train_layers = p.get("train_layers", "both")  # "both", "first", "last"

    with training_lock:
        training_state.update(
            running=True, train_losses=[], test_losses=[],
            epoch=0, total_epochs=num_epochs,
            w1tw1=None, train_r2=None, test_r2=None, error=None,
        )

    def epoch_cb(epoch, tl, vl, error=None):
        with training_lock:
            if error:
                training_state["error"] = error
                training_state["running"] = False
                return
            training_state["train_losses"].append(tl)
            training_state["test_losses"].append(vl)
            training_state["epoch"] = epoch

    def run():
        _, _, _, tr2, vr2, w = train_network(
            data_store["X_train"], data_store["y_train"],
            data_store["X_test"], data_store["y_test"],
            width=width, init_scale=init_scale,
            lr=lr, num_epochs=num_epochs, train_layers=train_layers,
            callback=epoch_cb,
        )
        with training_lock:
            # If callback already flagged an error, don't overwrite it
            if training_state.get("error"):
                training_state["running"] = False
                return
            training_state.update(
                running=False,
                w1tw1=w.tolist() if w is not None else None,
                train_r2=tr2, test_r2=vr2,
            )

    threading.Thread(target=run).start()
    return jsonify(status="training_started", num_epochs=num_epochs)


@app.route("/api/training_status")
def api_training_status():
    with training_lock:
        return jsonify(**{k: v for k, v in training_state.items()})


@app.route("/api/reveal_w1tw1")
def api_reveal_w1tw1():
    with training_lock:
        if training_state["w1tw1"] is None:
            return jsonify(error="No trained model available"), 400
        M = np.array(training_state["w1tw1"])
        M_norm = (M - M.min()) / (M.max() - M.min() + 1e-12)
        n = min(10, M.shape[0])
        return jsonify(
            full=M_norm.tolist(), zoomed=M_norm[:n, :n].tolist(),
            raw_zoomed=M[:n, :n].tolist(), d=M.shape[0],
            train_r2=training_state["train_r2"], test_r2=training_state["test_r2"],
        )


# ---------------------------------------------------------------------------
# Inline frontend
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Network Feature Learning</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{--bg:#f5f5f5;--sf:#fff;--sf2:#fafafa;--bd:#e0e0e0;--tx:#222;--dim:#666;--ac:#333;--red:#c0392b;--grn:#27ae60;--amb:#b8860b}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;font-size:15px;line-height:1.5}
header{padding:28px 40px;border-bottom:2px solid var(--bd);background:#fff}
header h1{font-size:1.5rem;font-weight:700;color:#111}
header p{color:var(--dim);font-size:.9rem;margin-top:2px}
.panels{max-width:1100px;margin:0 auto;padding:24px 40px 60px;display:flex;flex-direction:column;gap:20px}
.panel{background:var(--sf);border:1px solid var(--bd);border-radius:8px}
.ph{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;cursor:pointer;user-select:none}
.ph:hover{background:var(--sf2)}
.ph h2{font-size:1.05rem;font-weight:600;display:flex;align-items:center;gap:10px}
.badge{background:var(--ac);color:#fff;font-size:.7rem;padding:2px 8px;border-radius:10px;font-weight:700}
.pb{padding:20px 24px;display:none;border-top:1px solid var(--bd)}
.panel.open .pb{display:block}
.chv{transition:transform .2s;color:var(--dim);font-size:.8rem}
.panel.open .chv{transform:rotate(180deg)}
.desc{color:var(--dim);font-size:.9rem;margin-bottom:16px;line-height:1.6}
.ctrls{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;margin-bottom:16px}
.cg{display:flex;flex-direction:column;gap:3px}
.cg label{font-size:.75rem;color:var(--dim);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
input,select{background:#fff;border:1px solid var(--bd);color:var(--tx);padding:7px 10px;border-radius:5px;font-family:inherit;font-size:.9rem;width:130px}
input:focus,select:focus{outline:none;border-color:#888}
button{background:var(--ac);color:#fff;border:none;padding:9px 18px;border-radius:5px;font-family:inherit;font-size:.9rem;font-weight:600;cursor:pointer;transition:background .15s}
button:hover{background:#555}
button:disabled{opacity:.35;cursor:not-allowed}
button.rvl{background:var(--ac);font-size:1rem;padding:12px 28px}
.tog-group{display:flex;gap:0}
.tog-group button{border-radius:0;border:1px solid var(--bd);background:var(--sf2);color:var(--dim);padding:7px 14px;font-size:.85rem;font-weight:500}
.tog-group button:first-child{border-radius:5px 0 0 5px}
.tog-group button:last-child{border-radius:0 5px 5px 0}
.tog-group button.active{background:var(--ac);color:#fff;border-color:var(--ac)}
.init-display{display:flex;align-items:center;gap:6px}
.init-tag{font-size:.8rem;color:var(--dim);background:#eee;padding:2px 8px;border-radius:4px;font-weight:500}
button.sm{padding:5px 10px;font-size:.75rem;background:#eee;border:1px solid var(--bd);color:var(--dim)}
.rg{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:14px}
.rc{background:var(--sf2);border:1px solid var(--bd);border-radius:6px;padding:14px;text-align:center}
.rc .lb{font-size:.75rem;color:var(--dim);font-weight:600;text-transform:uppercase}
.rc .vl{font-size:1.5rem;font-weight:700;margin-top:2px}
.rc .vl.g{color:var(--grn)}.rc .vl.b{color:var(--red)}.rc .vl.n{color:var(--amb)}
.pc{border:1px solid var(--bd);border-radius:6px;padding:6px;margin-top:10px;background:#fff}
.sb{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--sf2);border:1px solid var(--bd);border-radius:6px;margin-top:10px;font-size:.9rem}
.sp{width:14px;height:14px;border:2px solid var(--bd);border-top-color:var(--ac);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.pb-bar{flex:1;height:5px;background:var(--bd);border-radius:3px;overflow:hidden}
.pb-bar .fill{height:100%;background:var(--ac);transition:width .3s;border-radius:3px}
.hm-pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}
.hm-pair h3{font-size:.9rem;font-weight:600;margin-bottom:6px;color:var(--dim)}
.fd{border:1px solid var(--bd);border-radius:6px;padding:12px 16px;font-size:1.05rem;text-align:center;margin-bottom:14px;color:var(--tx);background:var(--sf2)}
.hid{display:none!important}
@media(max-width:800px){.panels{padding:12px}.hm-pair{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Neural Network Feature Learning</h1>
  <p>How neural networks discover relevant features in high-dimensional data</p>
</header>
<div class="panels">

<!-- PANEL 1 -->
<div class="panel open" id="p1">
<div class="ph" onclick="tog('p1')">
  <h2><span class="badge">1</span> Data</h2>
  <span class="chv">&#9660;</span>
</div>
<div class="pb">
  <p class="desc">
    The target depends on only a few coordinates of a high-dimensional Gaussian input:
    <b>y = f(x<sub>i</sub>, x<sub>j</sub>, ...)</b> where <b>x</b> &isin; R<sup>d</sup>.
  </p>
  <div class="ctrls">
    <div class="cg"><label>Dimension d</label><input type="number" id="d" value="100" min="2" max="500"></div>
    <div class="cg"><label>Active Indices</label><input type="text" id="idx" value="0,1" style="width:100px"></div>
    <div class="cg"><label>Target Function</label>
      <select id="fn" style="width:180px">
        <option value="product" selected>x_i &middot; x_j</option>
        <option value="sum_of_squares">x_i&sup2; + x_j&sup2;</option>
        <option value="sum">x_i + x_j</option>
        <option value="squared_sum">(x_i + x_j)&sup2;</option>
        <option value="cubic">x_i&sup3; + x_j&sup3;</option>
      </select>
    </div>
    <div class="cg"><label>n_train</label><input type="number" id="ntr" value="2000" min="100" max="10000"></div>
    <div class="cg"><label>n_test</label><input type="number" id="nte" value="10000" min="100" max="20000"></div>
    <button onclick="genData()">Generate</button>
  </div>
  <div id="formula" class="fd hid"></div>
  <div id="dataPlot" class="pc hid"></div>
</div>
</div>

<!-- PANEL 2 -->
<div class="panel" id="p2">
<div class="ph" onclick="tog('p2')">
  <h2><span class="badge">2</span> NTK Baseline (all d dimensions)</h2>
  <span class="chv">&#9660;</span>
</div>
<div class="pb">
  <p class="desc">
    The Neural Tangent Kernel (NTK) corresponds to the infinite-width limit of our network at initialization.
    It uses all d dimensions equally and cannot identify which coordinates are relevant.
  </p>
  <div class="ctrls">
    <div class="cg"><label>Regularization</label><input type="number" id="kreg" value="0.001" step="0.001" min="0"></div>
    <button onclick="runKern()">Run NTK Regression</button>
  </div>
  <div id="kstat" class="sb hid"><div class="sp"></div><span>Computing NTK...</span></div>
  <div id="kres" class="hid"><div class="rg" id="kcards"></div></div>
</div>
</div>

<!-- PANEL 3 -->
<div class="panel" id="p3">
<div class="ph" onclick="tog('p3')">
  <h2><span class="badge">3</span> Train Neural Network</h2>
  <span class="chv">&#9660;</span>
</div>
<div class="pb">
  <p class="desc">
    One hidden layer: &nbsp;<b>x &rarr; W<sub>1</sub> &rarr; ReLU &rarr; W<sub>2</sub> &rarr; y</b>.
    &nbsp; Trained with SGD.
  </p>
  <div class="ctrls">
    <div class="cg"><label>Width</label><input type="number" id="w" value="128" min="4" max="2048"></div>
    <div class="cg">
      <label>Init Scale (W1)</label>
      <div class="init-display">
        <input type="number" id="is" value="0" step="0.001" min="0" onchange="updateInitLabel()">
        <span id="initTag" class="init-tag">PyTorch default</span>
        <button class="sm" onclick="resetInit()">Reset</button>
      </div>
    </div>
    <div class="cg"><label>Learning Rate</label><input type="number" id="lr" value="0.1" step="0.01" min="0.001"></div>
    <div class="cg"><label>Epochs</label><input type="number" id="ep" value="50" min="5" max="500"></div>
  </div>
  <div class="ctrls">
    <div class="cg">
      <label>Train Layers</label>
      <div class="tog-group">
        <button id="tl-both" class="active" onclick="setTrainLayers('both')">Both</button>
        <button id="tl-first" onclick="setTrainLayers('first')">W1 only</button>
        <button id="tl-last" onclick="setTrainLayers('last')">W2 only</button>
      </div>
    </div>
    <button id="tbtn" onclick="trainNN()">Train</button>
  </div>
  <div id="tstat" class="sb hid">
    <div class="sp"></div><span id="ttxt">Training...</span>
    <div class="pb-bar"><div class="fill" id="tprog" style="width:0%"></div></div>
    <span id="tep">0/0</span>
  </div>
  <div id="nres" class="hid"><div class="rg" id="ncards"></div></div>
  <div id="lossPlot" class="pc hid"></div>
</div>
</div>

<!-- PANEL 4 -->
<div class="panel" id="p4">
<div class="ph" onclick="tog('p4')">
  <h2><span class="badge">4</span> Learned Features: W<sub>1</sub><sup>T</sup>W<sub>1</sub></h2>
  <span class="chv">&#9660;</span>
</div>
<div class="pb">
  <p class="desc">
    If the network found the relevant coordinates, <b>W<sub>1</sub><sup>T</sup>W<sub>1</sub></b>
    will be low-rank with large entries only at the active indices.
  </p>
  <div style="text-align:center;margin:16px 0">
    <button class="rvl" id="rbtn" onclick="reveal()" disabled>Reveal W<sub>1</sub><sup>T</sup>W<sub>1</sub></button>
  </div>
  <div id="rres" class="hid">
    <div class="hm-pair">
      <div>
        <h3>Full matrix (d &times; d)</h3>
        <div id="hmFull" class="pc"></div>
      </div>
      <div>
        <h3>Top-left corner (zoomed)</h3>
        <div id="hmZoom" class="pc"></div>
      </div>
    </div>
  </div>
</div>
</div>

<!-- PANEL 5 -->
<div class="panel" id="p5">
<div class="ph" onclick="tog('p5')">
  <h2><span class="badge">5</span> NTK on Active Coordinates Only</h2>
  <span class="chv">&#9660;</span>
</div>
<div class="pb">
  <p class="desc">
    What if the kernel <em>knew</em> which coordinates mattered?
    Here we run the same NTK but restricted to only the active dimensions.
    This is what feature learning effectively achieves.
  </p>
  <div class="ctrls">
    <div class="cg"><label>Regularization</label><input type="number" id="kareg" value="0.001" step="0.001" min="0"></div>
    <button onclick="runKernActive()">Run NTK on Active Coords</button>
  </div>
  <div id="kastat" class="sb hid"><div class="sp"></div><span>Computing...</span></div>
  <div id="kares" class="hid"><div class="rg" id="kacards"></div></div>
</div>
</div>

</div>
<script>
const PLT_BG="#fff",PLT_GC="#e0e0e0",PLT_FT={color:"#444",family:"-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif",size:12};
function tog(id){document.getElementById(id).classList.toggle("open")}
function $(id){return document.getElementById(id)}
function r2c(v){return v>.8?"g":v>.3?"n":"b"}
function card(lb,vl,cls){return`<div class="rc"><div class="lb">${lb}</div><div class="vl ${cls}">${vl}</div></div>`}

async function genData(){
  const indices=$("idx").value.split(",").map(s=>parseInt(s.trim())).filter(n=>!isNaN(n));
  const d=+$("d").value,ft=$("fn").value,ntr=+$("ntr").value,nte=+$("nte").value;
  const r=await(await fetch("/api/generate_data",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({d,active_indices:indices,func_type:ft,n_train:ntr,n_test:nte,seed:1717})})).json();
  const fns={product:indices.map(i=>`x<sub>${i}</sub>`).join(" &middot; "),
    sum_of_squares:indices.map(i=>`x<sub>${i}</sub><sup>2</sup>`).join(" + "),
    sum:indices.map(i=>`x<sub>${i}</sub>`).join(" + "),
    squared_sum:"("+indices.map(i=>`x<sub>${i}</sub>`).join(" + ")+")<sup>2</sup>",
    cubic:indices.map(i=>`x<sub>${i}</sub><sup>3</sup>`).join(" + ")};
  const f=$("formula");f.classList.remove("hid");
  f.innerHTML=`<b>y</b> = ${fns[ft]}, &nbsp; <b>x</b> &isin; R<sup>${d}</sup>`;
  $("dataPlot").classList.remove("hid");
  if(r.scatter_x2){
    Plotly.newPlot("dataPlot",[{x:r.scatter_x,y:r.scatter_x2,z:r.scatter_y,mode:"markers",type:"scatter3d",
      marker:{size:2,color:r.scatter_y,colorscale:"Blues",opacity:.7}}],
      {paper_bgcolor:PLT_BG,plot_bgcolor:PLT_BG,font:PLT_FT,
       margin:{l:0,r:0,t:10,b:0},height:380,
       scene:{xaxis:{title:`x_${indices[0]}`,gridcolor:PLT_GC},yaxis:{title:`x_${indices[1]}`,gridcolor:PLT_GC},
              zaxis:{title:"y",gridcolor:PLT_GC},bgcolor:PLT_BG}},{responsive:true});
  }else{
    Plotly.newPlot("dataPlot",[{x:r.scatter_x,y:r.scatter_y,mode:"markers",type:"scatter",
      marker:{size:4,color:"#555",opacity:.5}}],
      {paper_bgcolor:PLT_BG,plot_bgcolor:PLT_BG,font:PLT_FT,
       margin:{l:50,r:20,t:10,b:40},height:300,
       xaxis:{title:`x_${indices[0]}`,gridcolor:PLT_GC,zerolinecolor:PLT_GC},
       yaxis:{title:"y",gridcolor:PLT_GC,zerolinecolor:PLT_GC}},{responsive:true});
  }
  $("p2").classList.add("open");
}

async function runKern(){
  $("kstat").classList.remove("hid");$("kres").classList.add("hid");
  const r=await(await fetch("/api/kernel_regression",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({reg:+$("kreg").value})})).json();
  $("kstat").classList.add("hid");$("kres").classList.remove("hid");
  $("kcards").innerHTML=
    card("NTK Train R&sup2;",r.ntk_train_r2.toFixed(4),r2c(r.ntk_train_r2))+
    card("NTK Test R&sup2;",r.ntk_test_r2.toFixed(4),r2c(r.ntk_test_r2));
  $("p3").classList.add("open");
}

async function runKernActive(){
  const indices=$("idx").value.split(",").map(s=>parseInt(s.trim())).filter(n=>!isNaN(n));
  $("kastat").classList.remove("hid");$("kares").classList.add("hid");
  const r=await(await fetch("/api/kernel_active",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({active_indices:indices,reg:+$("kareg").value})})).json();
  $("kastat").classList.add("hid");$("kares").classList.remove("hid");
  $("kacards").innerHTML=
    card("NTK Train R&sup2; (active only)",r.train_r2.toFixed(4),r2c(r.train_r2))+
    card("NTK Test R&sup2; (active only)",r.test_r2.toFixed(4),r2c(r.test_r2));
}

let pt=null,curTrainLayers="both";
function setTrainLayers(v){
  curTrainLayers=v;
  ["both","first","last"].forEach(k=>{$("tl-"+k).classList.toggle("active",k===v)});
}
function updateInitLabel(){
  const v=+$("is").value;
  $("initTag").textContent=v>0?`N(0, ${v})`:"PyTorch default";
}
function resetInit(){$("is").value="0";updateInitLabel();}
async function trainNN(){
  if(pt){clearInterval(pt);pt=null;}
  $("tbtn").disabled=true;$("tstat").classList.remove("hid");
  $("nres").classList.add("hid");$("rbtn").disabled=true;
  Plotly.purge("lossPlot");$("lossPlot").classList.add("hid");
  $("tprog").style.width="0%";$("tep").textContent="0/0";$("ttxt").textContent="Training...";
  const resp=await fetch("/api/train_nn",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({width:+$("w").value,init_scale:+$("is").value,lr:+$("lr").value,num_epochs:+$("ep").value,train_layers:curTrainLayers})});
  if(!resp.ok){
    const err=await resp.json();
    alert(err.error||"Training failed to start");
    $("tbtn").disabled=false;$("tstat").classList.add("hid");return;
  }
  pt=setInterval(poll,500);
}
async function poll(){
  const r=await(await fetch("/api/training_status")).json();
  const pct=r.total_epochs>0?(r.epoch/r.total_epochs*100):0;
  $("tprog").style.width=pct+"%";
  $("tep").textContent=`${r.epoch}/${r.total_epochs}`;
  $("ttxt").textContent=r.running?"Training...":"Done";
  if(r.train_losses.length>0){
    $("lossPlot").classList.remove("hid");
    const ep=r.train_losses.map((_,i)=>i+1);
    Plotly.newPlot("lossPlot",[
      {x:ep,y:r.train_losses,name:"Train",line:{color:"#333",dash:"dash",width:2}},
      {x:ep,y:r.test_losses,name:"Test",line:{color:"#c0392b",width:2}}],
      {paper_bgcolor:PLT_BG,plot_bgcolor:PLT_BG,font:PLT_FT,
       margin:{l:60,r:20,t:10,b:40},height:280,
       xaxis:{title:"Epoch",gridcolor:PLT_GC,zerolinecolor:PLT_GC},
       yaxis:{title:"MSE Loss",gridcolor:PLT_GC,zerolinecolor:PLT_GC,type:"log"},
       legend:{x:.75,y:.95,bgcolor:"rgba(255,255,255,0)"}},{responsive:true});
  }
  if(!r.running){
    clearInterval(pt);pt=null;$("tstat").classList.add("hid");$("tbtn").disabled=false;
    if(r.error){
      $("nres").classList.remove("hid");
      $("ncards").innerHTML=`<div class="rc" style="grid-column:1/-1;border-color:var(--red)"><div class="lb">Error</div><div class="vl b">${r.error}</div></div>`;
      return;
    }
    $("rbtn").disabled=false;
    $("nres").classList.remove("hid");
    $("ncards").innerHTML=
      card("Train R&sup2;",r.train_r2!==null?r.train_r2.toFixed(4):"...",r2c(r.train_r2))+
      card("Test R&sup2;",r.test_r2!==null?r.test_r2.toFixed(4):"...",r2c(r.test_r2));
    $("p4").classList.add("open");
  }
}

async function reveal(){
  const r=await(await fetch("/api/reveal_w1tw1")).json();
  if(r.error){alert(r.error);return}
  $("rres").classList.remove("hid");
  const hl=(t)=>({paper_bgcolor:PLT_BG,plot_bgcolor:PLT_BG,font:PLT_FT,
    margin:{l:50,r:20,t:10,b:50},height:400,
    xaxis:{title:"Coordinate index",side:"bottom"},yaxis:{title:"Coordinate index",autorange:"reversed"}});
  Plotly.newPlot("hmFull",[{z:r.full,type:"heatmap",colorscale:"Greys",reversescale:true,
    colorbar:{tickfont:{color:"#666"}}}],hl(),{responsive:true});
  Plotly.newPlot("hmZoom",[{z:r.raw_zoomed,type:"heatmap",colorscale:"Greys",reversescale:true,
    showscale:true,colorbar:{tickfont:{color:"#666"}}}],hl(),{responsive:true});
}
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(HTML, content_type="text/html")


if __name__ == "__main__":
    print("Starting NN Feature Learning webapp...")
    print("Opening http://localhost:5001 in your browser.")
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5001")).start()
    app.run(port=5001, debug=False)
