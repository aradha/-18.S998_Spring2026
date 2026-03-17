"""
Backend for dual activation iteration demo.

The dual activation is:
    check_phi(rho) = sum_{i=0}^{k} a_i^2 * rho^i

with the constraint check_phi(1) = 1, i.e. sum a_i^2 = 1.

We expose endpoints to:
  - Evaluate check_phi on a grid (for plotting)
  - Iterate check_phi on a single point L times (for the cobweb / trajectory)
"""

from flask import Flask, jsonify, request, send_from_directory
import numpy as np

app = Flask(__name__, static_folder=".")


# ── helpers ──────────────────────────────────────────────────────────────────

def dual_activation(rho: float, coeffs_sq: list[float]) -> float:
    """Evaluate check_phi(rho) = sum a_i^2 * rho^i."""
    val = 0.0
    for i, c2 in enumerate(coeffs_sq):
        val += c2 * (rho ** i)
    return val


def normalize_coeffs(raw_a: list[float]) -> list[float]:
    """Return a_i^2 / sum(a_i^2) so that check_phi(1) = 1."""
    sq = [a * a for a in raw_a]
    s = sum(sq)
    if s < 1e-15:
        # fallback: identity-like
        sq = [0.0] * len(raw_a)
        sq[1] = 1.0 if len(raw_a) > 1 else 1.0
        s = 1.0
    return [c / s for c in sq]


def eval_curve(coeffs_sq: list[float], n_pts: int = 500) -> dict:
    """Evaluate check_phi on [0, 1]."""
    rhos = np.linspace(0.0, 1.0, n_pts)
    vals = [dual_activation(r, coeffs_sq) for r in rhos]
    return {"rho": rhos.tolist(), "phi": vals}


def iterate_point(rho0: float, coeffs_sq: list[float], steps: int) -> list[dict]:
    """
    Return the cobweb trajectory:
      (rho0, 0) -> (rho0, phi(rho0)) -> (phi(rho0), phi(rho0)) -> ...
    Each entry is {x, y} for plotting the cobweb lines, plus the list of
    iterates for the dot.
    """
    trajectory = []  # list of {x, y} for cobweb segments
    iterates = [rho0]  # the sequence rho0, phi(rho0), phi^2(rho0), ...

    rho = rho0
    # start: move vertically from (rho, 0) to (rho, phi(rho))
    for _ in range(steps):
        phi_rho = dual_activation(rho, coeffs_sq)
        # vertical segment: (rho, rho) -> (rho, phi(rho))  [first one is from y=rho on the diagonal]
        trajectory.append({"x": rho, "y": phi_rho})
        # horizontal segment: (rho, phi(rho)) -> (phi(rho), phi(rho))  [reflect off y=x]
        trajectory.append({"x": phi_rho, "y": phi_rho})
        rho = phi_rho
        iterates.append(rho)

    return {"cobweb": trajectory, "iterates": iterates}


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/curve", methods=["POST"])
def api_curve():
    data = request.json
    raw_a = data.get("coeffs", [0.0, 1.0])
    coeffs_sq = normalize_coeffs(raw_a)
    curve = eval_curve(coeffs_sq)
    return jsonify({"curve": curve, "coeffs_sq": coeffs_sq})


@app.route("/api/iterate", methods=["POST"])
def api_iterate():
    data = request.json
    raw_a = data.get("coeffs", [0.0, 1.0])
    rho0 = float(data.get("rho0", 0.5))
    steps = int(data.get("steps", 1))
    coeffs_sq = normalize_coeffs(raw_a)
    result = iterate_point(rho0, coeffs_sq, steps)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
