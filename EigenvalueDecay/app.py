"""
Streamlit dashboard: eigenvalue decay of the Gaussian kernel
K(x,z) = exp(-(x-z)^2 / 2) with samples from N(0,1).

Run with:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from kernels import sample_points, build_kernel_matrix, theoretical_eigenvalues

# ── page setup ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Gaussian Kernel Eigenvalues", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .block-container { max-width: 1000px; padding-top: 2rem; }
    h1 { text-align: center; font-weight: 700; margin-bottom: 0.1rem; }
    .subtitle { text-align: center; color: #888; font-style: italic;
                margin-bottom: 1.5rem; font-size: 1.05rem; }
    .metric-card { background: #fff; border: 2px solid #3b5bdb;
                   border-radius: 12px; padding: 1rem; text-align: center; }
    .metric-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
                    color: #666; text-transform: uppercase; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #3b5bdb; }
    .metric-value-dark { font-size: 1.5rem; font-weight: 700; color: #222; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# Gaussian Kernel Eigenvalue Decay")
st.markdown(
    '<div class="subtitle">'
    "K(x, z) = exp(\u2212(x\u2212z)\u00b2 / 2),&ensp;"
    "samples x<sub>i</sub> \u223c \U0001d4a9(0, 1)"
    "</div>",
    unsafe_allow_html=True,
)

# ── controls ────────────────────────────────────────────────────────────────

c1, c2 = st.columns([3, 1])
with c1:
    n = int(st.number_input(
        "NUMBER OF SAMPLES", min_value=50, max_value=5000, value=1000, step=50,
    ))
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Update Plot", type="primary", use_container_width=True)

# ── compute ─────────────────────────────────────────────────────────────────

NUM_EIGS = 20

x = sample_points(n)
K = build_kernel_matrix(x) / n

# top-k via full eigendecomposition (fine for n <= 5000)
all_eigs = np.linalg.eigvalsh(K)
matrix_eigs = np.sort(all_eigs)[::-1][:NUM_EIGS]

theo = theoretical_eigenvalues(NUM_EIGS)
indices = np.arange(NUM_EIGS)

# ── plot ────────────────────────────────────────────────────────────────────

BLUE = "#3b5bdb"
RED = "#e03131"

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=indices, y=np.log(matrix_eigs), mode="markers",
    name="Empirical  log \u03bb(K/n)",
    marker=dict(size=9, color=BLUE, opacity=0.8,
                line=dict(width=0.8, color="#fff")),
))

fig.add_trace(go.Scatter(
    x=indices, y=np.log(theo), mode="lines",
    name="Theory  log r<sup> k+\u00bd</sup>",
    line=dict(width=3, color=RED),
))

fig.update_layout(
    title=dict(
        text="log eigenvalue  vs  index",
        x=0.5, font=dict(size=17, family="Inter"),
    ),
    xaxis=dict(
        title=dict(text="Eigenvalue index  k", font=dict(size=14)),
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        showline=True, linewidth=1, linecolor="#ccc",
        tickfont=dict(size=12),
        dtick=2,
    ),
    yaxis=dict(
        title=dict(text="log \u03bb<sub>k</sub>", font=dict(size=14)),
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        showline=True, linewidth=1, linecolor="#ccc",
        tickfont=dict(size=12),
    ),
    height=500,
    plot_bgcolor="#fff",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        x=0.50, y=0.97,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#ddd", borderwidth=1,
        font=dict(size=13, family="Inter"),
    ),
    margin=dict(l=70, r=30, t=55, b=65),
    hoverlabel=dict(font_size=13),
)

st.plotly_chart(fig, use_container_width=True)

# ── metric cards ────────────────────────────────────────────────────────────

r = (3.0 - np.sqrt(5.0)) / 2.0

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Decay</div>'
        '<div class="metric-value">Exponential</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Rate</div>'
        f'<div class="metric-value">r = {r:.4f}</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Samples</div>'
        f'<div class="metric-value-dark">{n}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center;color:#666;margin-top:0.8rem;font-size:0.92rem;">'
    "Eigenvalues of the Gaussian kernel under the Gaussian measure "
    "decay as \u03bb<sub>k</sub> = r<sup> k+\u00bd</sup> where "
    "r = (3 \u2212 \u221a5) / 2 \u2248 0.382 "
    "(reciprocal golden ratio squared)."
    "</div>",
    unsafe_allow_html=True,
)
