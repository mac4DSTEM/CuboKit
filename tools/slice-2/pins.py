"""CuboKit slice 2: the pin generator (docs/registration-slice-2.md, sections 2 to 5).

    References/py4dstem-venv/bin/python -I -B tools/slice-2/pins.py synthetic
    References/py4dstem-venv/bin/python -I -B tools/slice-2/pins.py real [name ...]

`synthetic` writes Tests/CuboVirtualTests/Fixtures/slice-2-pins.json (and the planted cubes as .npy) BEFORE any Swift.
`real` reads slice 1's exports in $CUBOKIT_REAL_DATA and writes <name>-s2-pins.json and <name>-s2-<quantity>.npy beside them.

For every quantity: py4DSTEM's own result (f050d207, float32 where py4DSTEM computes in float32) and the float64
evaluation of the same definition written here from the registration's text (section 3), which the kit is held to.
"""
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import py4DSTEM
from py4DSTEM.datacube import DataCube
from py4DSTEM.process.calibration import get_probe_size
from py4DSTEM.process.phase import DPC

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIXTURES = os.path.join(KIT, "Tests", "CuboVirtualTests", "Fixtures")
REAL = os.environ.get("CUBOKIT_REAL_DATA", os.path.expanduser("~/Library/Caches/CuboKit/real-data"))
SEED = 20261015
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(name)
    assert ok, f"SELF-CHECK FAILED: {name} {detail}"


def g17(x):
    return "%.17g" % float(x)


def arr(a):
    return [g17(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)]


def versions():
    return {"py4dstem": py4DSTEM.__version__, "py4dstem_commit": "f050d207851f8accd3aa550365baaf1f38230428",
            "numpy": np.__version__, "python": platform.python_version()}


# ------------------------------------------------------------------------------------------------------------------
# The float64 definitions (registration section 3)
# ------------------------------------------------------------------------------------------------------------------
def com_f64(cube, weights=None):
    """3.1: per position S, the row and column moments, and the quotients, in float64."""
    ry, rx, qy, qx = cube.shape
    w = np.ones(qy * qx) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    ii, jj = np.indices((qy, qx))
    d = cube.reshape(ry * rx, qy * qx).astype(np.float64)
    S = d @ w
    Mr = d @ (w * ii.reshape(-1))
    Mc = d @ (w * jj.reshape(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return S.reshape(ry, rx), Mr.reshape(ry, rx), Mc.reshape(ry, rx), (Mr / S).reshape(ry, rx), (Mc / S).reshape(ry, rx)


def plane_f64(m):
    """3.2 .planeFit: least squares a + b r + c c over the finite positions."""
    R, C = m.shape
    r, c = np.indices((R, C))
    fin = np.isfinite(m)
    A = np.stack([np.ones(fin.sum()), r[fin], c[fin]], axis=1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, m[fin].astype(np.float64), rcond=None)
    return coef[0] + coef[1] * r + coef[2] * c


def normalize_f64(mx, my, reference, sampling=1.0):
    if reference == "plane":
        fx, fy = plane_f64(mx), plane_f64(my)
    elif reference == "constant":
        fx, fy = np.full(mx.shape, np.mean(mx)), np.full(my.shape, np.mean(my))
    elif reference == "none":
        fx, fy = np.zeros(mx.shape), np.zeros(my.shape)
    else:
        raise ValueError(reference)
    return fx, fy, np.nan_to_num(mx - fx) * sampling, np.nan_to_num(my - fy) * sampling


def rotated(nx, ny, theta, transpose):
    c, s = math.cos(theta), math.sin(theta)
    if transpose:
        return c * ny - s * nx, s * ny + c * nx
    return c * nx - s * ny, s * nx + c * ny


def curl(x, y):
    gxy = x[1:-1, 2:] - x[1:-1, :-2]
    gyx = y[2:, 1:-1] - y[:-2, 1:-1]
    return float(np.mean(np.abs(gyx - gxy)))


def div(x, y):
    gxx = x[2:, 1:-1] - x[:-2, 1:-1]
    gyy = y[1:-1, 2:] - y[1:-1, :-2]
    return float(np.mean(np.abs(gxx + gyy)))


def rotation_f64(nx, ny, angles_deg=None, maximize_divergence=False):
    """3.3, neither forced: the curve for both orientations and py4DSTEM's selection rule."""
    if angles_deg is None:
        angles_deg = np.arange(-89.0, 90.0, 1.0)
    f = div if maximize_divergence else curl
    plain = np.array([f(*rotated(nx, ny, math.radians(a), False)) for a in angles_deg])
    trans = np.array([f(*rotated(nx, ny, math.radians(a), True)) for a in angles_deg])
    if maximize_divergence:
        i, j = int(np.argmax(plain)), int(np.argmax(trans))
        transposed = not (plain[i] >= trans[j])
    else:
        i, j = int(np.argmin(plain)), int(np.argmin(trans))
        transposed = not (plain[i] <= trans[j])
    k = j if transposed else i
    cx, cy = rotated(nx, ny, math.radians(angles_deg[k]), transposed)
    return float(angles_deg[k]), transposed, plain, trans, cx, cy


def dpc_f64(cx, cy, sampling=(1.0, 1.0), padding=2, max_iter=64, step=0.5, stop=1e-6, backtrack=True):
    """3.4, py4DSTEM's iterative reconstruction in float64, step for step."""
    R, C = cx.shape
    Rp, Cp = (np.round(np.array([R, C]) * padding)).astype(int)
    kx = np.fft.fftfreq(Rp, d=sampling[0])
    ky = np.fft.fftfreq(Cp, d=sampling[1])
    kya, kxa = np.meshgrid(ky, kx)
    k_den = kxa ** 2 + kya ** 2
    k_den[0, 0] = np.inf
    k_den = 1 / k_den
    kx_op = -1j * 0.25 * kxa * k_den
    ky_op = -1j * 0.25 * kya * k_den
    phi = np.zeros((Rp, Cp))
    mask = np.zeros((Rp, Cp), dtype=bool)
    mask[:R, :C] = True
    error = np.inf
    errors, accepted = [], []
    denom = np.mean(cx.ravel() ** 2 + cy.ravel() ** 2)
    for _ in range(max_iter):
        if step < stop:
            break
        dx = (np.roll(phi, 1, axis=0) - np.roll(phi, -1, axis=0)) / (2 * sampling[0])
        dy = (np.roll(phi, 1, axis=1) - np.roll(phi, -1, axis=1)) / (2 * sampling[1])
        dx[mask] += cx.ravel()
        dy[mask] += cy.ravel()
        dx[~mask] = 0
        dy[~mask] = 0
        new_error = np.mean(dx[mask] ** 2 + dy[mask] ** 2) / denom
        if new_error > error and backtrack:
            step /= 2
            accepted.append(False)
            continue
        error = new_error
        update = np.real(np.fft.ifft2(np.fft.fft2(dx) * kx_op + np.fft.fft2(dy) * ky_op))
        phi += step * update
        errors.append(float(error))
        accepted.append(True)
    return phi[:R, :C], errors, accepted, step


def py4dstem_dpc(cube=None, com=None, shifts=None, dp_mask=None, fit="plane"):
    """py4DSTEM's own chain: preprocess (CoM float32, plane fit, rotation) and reconstruct with defaults."""
    if cube is not None:
        d = DPC(datacube=DataCube(data=cube.astype(np.float32)), energy=200e3, verbose=False)
        d.preprocess(dp_mask=dp_mask, fit_function=fit, force_com_shifts=shifts,
                     plot_center_of_mass=False, plot_rotation=False)
    else:
        d = DPC(datacube=None, energy=200e3, verbose=False)
        d.preprocess(force_com_measured=[com[0], com[1]], fit_function=fit, force_com_shifts=shifts,
                     plot_center_of_mass=False, plot_rotation=False)
    d.reconstruct(progress_bar=False)
    a = lambda v: np.asarray(v, dtype=np.float64)
    return {
        "com_x": a(d._com_measured_x), "com_y": a(d._com_measured_y),
        "fitted_x": a(d._com_fitted_x), "fitted_y": a(d._com_fitted_y),
        "normalized_x": a(d._com_normalized_x), "normalized_y": a(d._com_normalized_y),
        "degrees": float(np.rad2deg(d._rotation_best_rad)), "transposed": bool(d._rotation_best_transpose),
        "curve": a(getattr(d, "_rotation_curl", np.array([]))),
        "curve_transposed": a(getattr(d, "_rotation_curl_transpose", np.array([]))),
        "corrected_x": a(d._com_x), "corrected_y": a(d._com_y),
        "phase": a(d.object_phase), "errors": [float(e) for e in d.error_iterations],
    }


def chain_f64(mx, my, reference):
    fx, fy, nx, ny = normalize_f64(mx, my, reference)
    deg, tr, plain, trans, cx, cy = rotation_f64(nx, ny)
    phase, errors, accepted, step = dpc_f64(cx, cy)
    return {"fitted_x": fx, "fitted_y": fy, "normalized_x": nx, "normalized_y": ny, "degrees": deg, "transposed": tr,
            "curve": plain, "curve_transposed": trans, "corrected_x": cx, "corrected_y": cy, "phase": phase,
            "errors": errors, "accepted": accepted, "final_step": step}


def rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if b.size == 0:
        return 0.0
    floor = 1e-6 * max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def of_max(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def probe_entry(dp, lower=0.01, upper=0.99, n=100):
    """get_probe_size on a float64 pattern; py4DSTEM's result is float64 here, so it is the definition."""
    dp = np.asarray(dp, dtype=np.float64)
    with np.errstate(all="ignore"):
        r, x0, y0 = get_probe_size(dp, thresh_lower=lower, thresh_upper=upper, N=n)
    refused = not (np.isfinite(r) and np.isfinite(x0) and np.isfinite(y0))
    # the trusted threshold, which py4DSTEM computes and does not return: the same steps, NumPy's mean (the kit returns it)
    t = np.linspace(lower, upper, n)
    m = np.max(dp)
    rv = np.array([np.sqrt(np.sum(dp > m * x) / np.pi) for x in t])
    gv = np.gradient(rv)
    sel = (gv <= 0) * (gv >= 2 * np.median(gv))
    with np.errstate(all="ignore"):
        threshold = np.mean(t[sel])
    return {"refused": bool(refused), "radius": g17(r), "center_row": g17(x0), "center_column": g17(y0),
            "threshold": g17(threshold)}


def fit_distance(own):
    """py4DSTEM's curve_fit plane against the least-squares plane of the SAME input, its own float32 CoM."""
    if own["com_x"].size == 0 or np.allclose(own["fitted_x"], 0):
        return 0.0
    return float(max(np.max(np.abs(plane_f64(own["com_x"]) - own["fitted_x"])),
                     np.max(np.abs(plane_f64(own["com_y"]) - own["fitted_y"]))))


def chain_entry(own, f64):
    return {
        "degrees_py4dstem": own["degrees"], "transposed_py4dstem": own["transposed"],
        "degrees": f64["degrees"], "transposed": f64["transposed"],
        "fitted_x": arr(f64["fitted_x"]), "fitted_y": arr(f64["fitted_y"]),
        "fitted_x_py4dstem": arr(own["fitted_x"]), "fitted_y_py4dstem": arr(own["fitted_y"]),
        "normalized_x": arr(f64["normalized_x"]), "normalized_y": arr(f64["normalized_y"]),
        "curve": arr(f64["curve"]), "curve_transposed": arr(f64["curve_transposed"]),
        "curve_py4dstem": arr(own["curve"]), "curve_transposed_py4dstem": arr(own["curve_transposed"]),
        "corrected_x": arr(f64["corrected_x"]), "corrected_y": arr(f64["corrected_y"]),
        "phase": arr(f64["phase"]), "phase_py4dstem": arr(own["phase"]),
        "errors": [g17(e) for e in f64["errors"]], "errors_py4dstem": [g17(e) for e in own["errors"]],
        "accepted": f64["accepted"], "final_step": g17(f64["final_step"]),
        "phase_distance_py4dstem_own_vs_f64": of_max(own["phase"], f64["phase"]) if own["phase"].size else 0.0,
        "curve_distance_py4dstem_own_vs_f64": rel(own["curve"], f64["curve"]) if own["curve"].size else 0.0,
        "measured_x_py4dstem": arr(own["com_x"]), "measured_y_py4dstem": arr(own["com_y"]),
        "fit_distance_py4dstem_vs_lstsq_px": fit_distance(own),
    }


# ------------------------------------------------------------------------------------------------------------------
# Synthetic (5.1, 5.2)
# ------------------------------------------------------------------------------------------------------------------
def planted_shift_cube():
    rng = np.random.default_rng(SEED)
    ry, rx, qy, qx = 4, 5, 8, 9
    cube = np.zeros((ry, rx, qy, qx), dtype=np.float32)
    truth_r = np.zeros((ry, rx)); truth_c = np.zeros((ry, rx))
    for r in range(ry):
        for c in range(rx):
            i0, j0 = int(rng.integers(0, qy - 1)), int(rng.integers(0, qx - 1))
            a, b = float(rng.integers(1, 1000)), float(rng.integers(1, 1000))
            cube[r, c, i0, j0] = a
            cube[r, c, i0 + 1, j0 + 1] = b
            truth_r[r, c] = i0 + b / (a + b); truth_c[r, c] = j0 + b / (a + b)
    return cube, truth_r, truth_c


def disk(qy, qx, cr, cc, R, inside=1000.0, outside=1.0):
    ii, jj = np.indices((qy, qx))
    m = (ii - cr) ** 2 + (jj - cc) ** 2 < R * R
    return np.where(m, inside, outside).astype(np.float64), int(m.sum())


def planted_field(R=24, C=20):
    r, c = np.indices((R, C))
    phi = np.exp(-(((r - 11.3) / 4.0) ** 2 + ((c - 9.6) / 3.5) ** 2)) * 2.0
    phi -= phi.min()
    gx = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / 2      # the gradient the forward operator inverts
    gy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2
    return phi, gx, gy


def run_synthetic():
    out = {"versions": versions(), "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "seed": SEED, "com": [], "probe": [], "fields": []}
    slice1 = json.load(open(os.path.join(FIXTURES, "slice-1-pins.json")))
    cubes = {c["name"]: np.load(os.path.join(FIXTURES, c["file"])) for c in slice1["cubes"]}
    shift, tr, tc = planted_shift_cube()
    np.save(os.path.join(FIXTURES, "shift.npy"), shift)
    cubes["shift"] = shift
    masks = {"none": None}
    for name, cube in cubes.items():
        qy, qx = cube.shape[2:]
        ii, jj = np.indices((qy, qx))
        bf = ((ii - (qy - 1) / 2) ** 2 + (jj - (qx - 1) / 2) ** 2 < (0.3 * min(qy, qx)) ** 2).astype(np.float64)
        weighted = np.array([[0.0, 0.5, 1.0, 2.0, -1.0][(k * 7 + k // qx) % 5] for k in range(qy * qx)]).reshape(qy, qx)
        c32 = cube.astype(np.float32)
        for mname, w in (("none", None), ("bf", bf), ("weighted", weighted)):
            S, Mr, Mc, cr, cc = com_f64(c32, w)
            entry = {"cube": name, "mask": mname, "weights": arr(w) if w is not None else None,
                     "intensity": arr(S), "moment_row": arr(Mr), "moment_column": arr(Mc),
                     "row": arr(cr), "column": arr(cc)}
            # py4DSTEM's own (float32 loop), from DPC.preprocess where the scan is at least 3 x 3
            if cube.shape[0] >= 3 and cube.shape[1] >= 3 and np.all(np.isfinite(cr)) and np.all(S != 0):
                own = py4dstem_dpc(cube=c32, dp_mask=w)
                entry["row_py4dstem"] = arr(own["com_x"]); entry["column_py4dstem"] = arr(own["com_y"])
                entry["com_distance_py4dstem_own_vs_f64_px"] = float(np.max(np.abs(own["com_x"] - cr)))
                if mname == "none":
                    f64 = chain_f64(cr, cc, "plane")
                    entry["chain"] = chain_entry(own, f64)
            out["com"].append(entry)
        if name == "shift":
            check("shift cube: CoM equals the planted shift (float64)",
                  np.max(np.abs(com_f64(c32)[3] - tr)) < 1e-12 and np.max(np.abs(com_f64(c32)[4] - tc)) < 1e-12)
            out["shift_truth"] = {"row": arr(tr), "column": arr(tc)}
        for kind, dp in (("mean", np.mean(c32, axis=(0, 1), dtype=np.float64)), ("max", np.max(c32, axis=(0, 1)))):
            e = probe_entry(dp); e["cube"] = name; e["pattern"] = kind
            out["probe"].append(e)
    # probe size truth: uniform disks centred on a pixel and on a half-pixel
    for (cr, cc) in ((15.0, 16.0), (15.5, 16.5)):
        dp, n = disk(32, 33, cr, cc, 5.0)
        e = probe_entry(dp); e["cube"] = f"disk_{cr}_{cc}"; e["pattern"] = "given"
        e["pattern_values"] = arr(dp); e["shape"] = [32, 33]
        e["truth_radius"] = g17(math.sqrt(n / math.pi)); e["truth_row"] = g17(cr); e["truth_column"] = g17(cc)
        check(f"disk {cr},{cc}: py4DSTEM radius = sqrt(n/pi)", abs(float(e["radius"]) - math.sqrt(n / math.pi)) < 1e-12)
        check(f"disk {cr},{cc}: py4DSTEM centre exact", abs(float(e["center_row"]) - cr) < 1e-12
              and abs(float(e["center_column"]) - cc) < 1e-12)
        out["probe"].append(e)
    # mutant-driven cases (amendment 2026-10-15): a ring exactly at a trusted threshold (so `>` and `>=` differ where the
    # radius is averaged), and a ring exactly at the last threshold with thresholds whose computed last linspace value is
    # not the stop. Each is checked to change get_probe_size's answer under its mutation before it is pinned.
    def probe_variant(dp, lo, hi, n, ge=False, inexact_last=False):
        t = np.arange(n) * ((hi - lo) / (n - 1)) + lo
        if not inexact_last:
            t[-1] = hi
        m = np.max(dp)
        r = np.array([np.sqrt(np.sum((dp >= m * x) if ge else (dp > m * x)) / np.pi) for x in t])
        g = np.gradient(r)
        sel = (g <= 0) * (g >= 2 * np.median(g))
        return float(np.mean(r[sel])), float(np.mean(t[sel]))
    ii, jj = np.indices((32, 33))
    d2 = (ii - 15.0) ** 2 + (jj - 16.0) ** 2
    t50 = np.linspace(0.01, 0.99, 100)[50]
    tie = np.where(d2 < 25, 1000.0, np.where(d2 < 64, 1000.0 * t50, 1.0))
    check("tie: the ring sits exactly at 1000 * t50", np.sum(tie == 1000.0 * t50) > 0)
    check("tie: >= changes the answer", probe_variant(tie, 0.01, 0.99, 100) != probe_variant(tie, 0.01, 0.99, 100, ge=True))
    e = probe_entry(tie); e.update({"cube": "tie_at_threshold", "pattern": "given", "pattern_values": arr(tie), "shape": [32, 33]})
    out["probe"].append(e)
    lo, hi, nn = 0.02, 0.93, 41     # computed last 0.9299999999999999; a single 1000 pixel, a disk exactly at 1000 · 0.93
    check("thresholds: the computed last linspace value is not the stop", (nn - 1) * ((hi - lo) / (nn - 1)) + lo != hi)
    last = np.where(d2 < 0.25, 1000.0, np.where(d2 < 25, 1000.0 * hi, 1.0))
    check("last: the inexact last threshold changes the answer",
          probe_variant(last, lo, hi, nn) != probe_variant(last, lo, hi, nn, inexact_last=True))
    e = probe_entry(last, lo, hi, nn); e.update({"cube": "last_threshold", "pattern": "given", "pattern_values": arr(last),
                                                 "shape": [32, 33], "lower": lo, "upper": hi, "count": nn})
    out["probe"].append(e)
    # planted fields (5.1): a known gradient, rotated by -theta, optionally transposed; reference none
    phi, gx, gy = planted_field()
    for theta, transpose in ((0.0, False), (23.0, False), (-61.0, True)):
        t = math.radians(-theta)
        rx_, ry_ = math.cos(t) * gx - math.sin(t) * gy, math.sin(t) * gx + math.cos(t) * gy
        mx, my = (ry_, rx_) if transpose else (rx_, ry_)
        f64 = chain_f64(mx, my, "none")
        own = py4dstem_dpc(com=(mx.astype(np.float32), my.astype(np.float32)), shifts=(0.0, 0.0), fit=None)
        check(f"field {theta}: f64 solver recovers the angle and transpose",
              f64["degrees"] == theta and f64["transposed"] == transpose, f"{f64['degrees']} {f64['transposed']}")
        p = f64["phase"] - f64["phase"].mean(); q = phi - phi.mean()
        pearson = float(np.sum(p * q) / math.sqrt(np.sum(p * p) * np.sum(q * q)))
        check(f"field {theta}: reconstruction correlates with phi >= 0.99", pearson >= 0.99, f"{pearson}")
        acc_err = [e for e in f64["errors"]]
        check(f"field {theta}: accepted errors non-increasing", all(b <= a for a, b in zip(acc_err, acc_err[1:])))
        e = chain_entry(own, f64)
        e.update({"theta": theta, "transpose": transpose, "shape": list(mx.shape), "measured_x": arr(mx),
                  "measured_y": arr(my), "phi": arr(phi), "pearson": pearson})
        out["fields"].append(e)
    # a pure plane: the plane fit removes it to rounding
    r, c = np.indices((7, 9))
    px, py = 0.3 + 0.02 * r - 0.07 * c, -1.1 + 0.05 * r + 0.01 * c
    fx, fy, nx, ny = normalize_f64(px, py, "plane")
    check("pure plane: normalized ~ 0", np.max(np.abs(nx)) < 1e-12 and np.max(np.abs(ny)) < 1e-12)
    out["plane"] = {"shape": [7, 9], "x": arr(px), "y": arr(py)}
    with open(os.path.join(FIXTURES, "slice-2-pins.json"), "w") as f:
        json.dump(out, f, indent=0)
    print(f"pins.py slice 2 synthetic: {len(out['com'])} CoM entries, {len(out['probe'])} probe entries, "
          f"{len(out['fields'])} fields, {len(CHECKS)} self-checks passed")
    for e in out["com"]:
        if "chain" in e:
            ch = e["chain"]
            print(f"  {e['cube']:9s} CoM own-vs-f64 {e['com_distance_py4dstem_own_vs_f64_px']:.2e} px; rotation "
                  f"{ch['degrees']} {ch['transposed']} (py4DSTEM {ch['degrees_py4dstem']} {ch['transposed_py4dstem']}); "
                  f"phase own-vs-f64 {ch['phase_distance_py4dstem_own_vs_f64']:.2e} of max")
    for e in out["fields"]:
        print(f"  field theta {e['theta']} T={e['transpose']}: f64 {e['degrees']} {e['transposed']}, py4DSTEM "
              f"{e['degrees_py4dstem']} {e['transposed_py4dstem']}; pearson {e['pearson']:.5f}; phase own-vs-f64 "
              f"{e['phase_distance_py4dstem_own_vs_f64']:.2e}; iterations {len(e['errors'])}")
    for e in out["probe"]:
        print(f"  probe {e['cube']}/{e['pattern']}: refused={e['refused']} r={float(e['radius']):.6g}")


# ------------------------------------------------------------------------------------------------------------------
# Real data (5.3)
# ------------------------------------------------------------------------------------------------------------------
def run_real(names):
    for name in names:
        t0 = datetime.datetime.now()
        meta = json.load(open(os.path.join(REAL, f"{name}-pins.json")))
        cube = np.load(os.path.join(REAL, f"{name}.npy"), mmap_mode="r")
        c32 = np.ascontiguousarray(cube).astype(np.float32)
        qy, qx = c32.shape[2:]
        bf = next(g for g in meta["geometries"] if g["name"] == "bf")["geometry"]
        ii, jj = np.indices((qy, qx))
        bfmask = (((ii - bf["centerRow"]) ** 2 + (jj - bf["centerColumn"]) ** 2) < bf["radius"] ** 2).astype(np.float64)
        arrays, info = {}, {"name": name, "shape": list(c32.shape), "versions": versions(),
                            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "com": {}}
        for mname, w in (("none", None), ("bf", bfmask)):
            S, Mr, Mc, cr, cc = com_f64(c32, w)
            cond = np.abs(S) >= 1e-6 * (np.abs(c32.reshape(len(S.reshape(-1)), -1).astype(np.float64)) @
                                        (np.ones(qy * qx) if w is None else np.abs(w).reshape(-1))).reshape(S.shape)
            for k, v in (("intensity", S), ("moment_row", Mr), ("moment_column", Mc), ("row", cr), ("column", cc),
                         ("conditioned", cond.astype(np.float64))):
                arrays[f"com_{mname}_{k}"] = v
            own = py4dstem_dpc(cube=c32, dp_mask=w)
            arrays[f"com_{mname}_row_py4dstem"] = own["com_x"]; arrays[f"com_{mname}_column_py4dstem"] = own["com_y"]
            info["com"][mname] = {"py4dstem_own_vs_f64_px": float(np.nanmax(np.abs(own["com_x"] - cr))),
                                  "ill_conditioned_positions": int((~cond).sum())}
            if mname == "none":
                f64 = chain_f64(cr, cc, "plane")
                for k in ("fitted_x", "fitted_y", "normalized_x", "normalized_y", "curve", "curve_transposed",
                          "corrected_x", "corrected_y", "phase"):
                    arrays[f"chain_{k}"] = f64[k]
                    if k in own:
                        arrays[f"chain_{k}_py4dstem"] = own[k]
                arrays["chain_errors"] = np.array(f64["errors"]); arrays["chain_errors_py4dstem"] = np.array(own["errors"])
                info["chain"] = {"degrees": f64["degrees"], "transposed": f64["transposed"],
                                 "degrees_py4dstem": own["degrees"], "transposed_py4dstem": own["transposed"],
                                 "accepted": f64["accepted"], "final_step": f64["final_step"],
                                 "phase_distance_py4dstem_own_vs_f64": of_max(own["phase"], f64["phase"]),
                                 "curve_distance_py4dstem_own_vs_f64": rel(own["curve"], f64["curve"]) if own["curve"].size else None,
                                 "fit_distance_py4dstem_vs_lstsq_px": fit_distance(own)}
        info["probe"] = {k: probe_entry(dp) for k, dp in
                         (("mean", np.mean(c32, axis=(0, 1), dtype=np.float64)), ("max", np.max(c32, axis=(0, 1))))}
        for k, v in arrays.items():
            np.save(os.path.join(REAL, f"{name}-s2-{k}.npy"), np.ascontiguousarray(v, dtype=np.float64))
        json.dump(info, open(os.path.join(REAL, f"{name}-s2-pins.json"), "w"), indent=1)
        ch = info["chain"]
        print(f"pins.py slice 2 real {name}: CoM own-vs-f64 {info['com']['none']['py4dstem_own_vs_f64_px']:.2e} px "
              f"(ill-conditioned {info['com']['none']['ill_conditioned_positions']}), bf {info['com']['bf']['py4dstem_own_vs_f64_px']:.2e} px; "
              f"rotation f64 {ch['degrees']} {ch['transposed']} py4DSTEM {ch['degrees_py4dstem']} {ch['transposed_py4dstem']}; "
              f"phase {ch['phase_distance_py4dstem_own_vs_f64']:.2e} of max; "
              f"fit {ch['fit_distance_py4dstem_vs_lstsq_px']:.2e} px; probe mean r {float(info['probe']['mean']['radius']):.4f}; "
              f"{(datetime.datetime.now() - t0).total_seconds():.0f} s")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("synthetic", "real"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "synthetic":
        run_synthetic()
    else:
        run_real(sys.argv[2:] or ["almgsi", "auref", "simau", "particle", "py4dstem_small"])
