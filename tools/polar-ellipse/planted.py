#!/usr/bin/env python3
"""Planted-truth inputs for CuboKit's polar cube, radial background, ellipse fit, Fourier resample.

numpy only. Writes `.npy` files and one `truth.json` into --out (default: ./out beside this script; do not commit the output).

Conventions (rows and columns, never x and y):
  * A detector pixel is (r, c): r the row, c the column. The ellipse centre is (r0, c0) in pixels, pixel centres at integers.
  * Semi-axes a >= b > 0 (pixels, for the ring of scale 1). theta in (-pi/2, pi/2] is the angle of the MAJOR axis, measured from
    the +column direction towards +row. Unit vectors (row, col): e_a = (sin t, cos t), e_b = (cos t, -sin t).
      u = (r - r0) sin t + (c - c0) cos t        v = (r - r0) cos t - (c - c0) sin t
      rho = sqrt((u / a)^2 + (v / b)^2)          the ring k sits at rho = s_k (s_1 = 1)
  * The circularised frame stretches the minor axis to the major one: q = a * rho, phi = atan2(v * a / b, u). A correct
    elliptical correction therefore puts ring k at q = a * s_k for every phi. Mapping to the kit's or py4DSTEM's own (x0, y0,
    a, b, theta) is left to the Mac pin step; nothing here claims it.
  * Pattern = sum_k A_k exp(-(rho - s_k)^2 / (2 w^2)) + B(q), B(q) = bg_amp * exp(-q / bg_len) + bg_const, per scan position
    scaled by m_p. All analytic: nothing is interpolated in the truth.

Outputs:
  cube_<case>.npy        float64 (scan_rows, scan_cols, det_rows, det_cols) pattern, m_p = 1 + 0.25 * (scan index) / n
  polar_truth_<case>.npy float64 (scan_rows, scan_cols, n_q, n_phi): the pattern evaluated at the polar grid of the circularised
                         frame (q_k = (k + 0.5) * dq, phi_j = j * 2 pi / n_phi), no interpolation. Constant in phi.
  ring_points_<case>.npy (N, 2) exact (row, col) points of ring 1 for the ellipse fit; *_noisy.npy adds N(0, 0.05 px), seed 1.
  resample_*.npy         band-limited image at the input size and its exact samples at the output size.
Cases: tilted (a=24, b=18, theta=0.6), flat (a=27, b=21.6, theta=0), circle (a=b=23), near_circle (a=24, b=23.5, theta=-1.1); rings at 1 and 1.6 a.

truth.json carries the parameters, the background, the grids, the self-check numbers and the tolerance rationale.
Run:  python3 -I planted.py [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np

DET = 96
SCAN = (3, 4)
CASES = {
    "tilted": dict(r0=47.3, c0=48.6, a=24.0, b=18.0, theta=0.6),
    "flat": dict(r0=48.0, c0=47.0, a=27.0, b=21.6, theta=0.0),
    "circle": dict(r0=46.5, c0=49.5, a=23.0, b=23.0, theta=0.0),
    "near_circle": dict(r0=49.2, c0=47.8, a=24.0, b=23.5, theta=-1.1),
}
SCALES = [1.0, 1.6]         # s_k: two rings at rho = 1 and 1.6 -> radii a and 1.6 a (<= 38.4 for a=24, fits in 96/2)
AMPS = [100.0, 60.0]
WIDTH = 0.04                # ring width w in rho units
BG = dict(amp=80.0, length=25.0, const=5.0)   # B(q) = amp * exp(-q / length) + const, q in pixels of the circularised frame


def _uv(r, c, p):
    t = p["theta"]
    dr, dc = r - p["r0"], c - p["c0"]
    return dr * np.sin(t) + dc * np.cos(t), dr * np.cos(t) - dc * np.sin(t)


def pattern(r, c, p):
    u, v = _uv(r, c, p)
    rho = np.sqrt((u / p["a"]) ** 2 + (v / p["b"]) ** 2)
    q = p["a"] * rho
    out = sum(a * np.exp(-((rho - s) ** 2) / (2 * WIDTH ** 2)) for a, s in zip(AMPS, SCALES))
    return out + BG["amp"] * np.exp(-q / BG["length"]) + BG["const"]


def scan_scale():
    n = SCAN[0] * SCAN[1]
    return (1 + 0.25 * np.arange(n) / n).reshape(SCAN)


def make_cube(p):
    r, c = np.meshgrid(np.arange(DET, dtype=np.float64), np.arange(DET, dtype=np.float64), indexing="ij")
    return scan_scale()[:, :, None, None] * pattern(r, c, p)[None, None]


def polar_truth(p, n_q, n_phi):
    dq = (1.0 * DET / 2) / n_q
    q = (np.arange(n_q) + 0.5) * dq
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    rho = q / p["a"]
    prof = sum(a * np.exp(-((rho - s) ** 2) / (2 * WIDTH ** 2)) for a, s in zip(AMPS, SCALES))
    prof = prof + BG["amp"] * np.exp(-q / BG["length"]) + BG["const"]
    img = np.broadcast_to(prof[:, None], (n_q, n_phi))
    return scan_scale()[:, :, None, None] * img[None, None], q, phi, dq


def ring_points(p, n, rng=None, noise=0.0):
    ang = np.arange(n) * 2 * np.pi / n + 0.1
    u, v = p["a"] * np.cos(ang), p["b"] * np.sin(ang)
    t = p["theta"]
    r = p["r0"] + u * np.sin(t) + v * np.cos(t)
    c = p["c0"] + u * np.cos(t) - v * np.sin(t)
    pts = np.stack([r, c], axis=1)
    if noise:
        pts = pts + rng.normal(0, noise, pts.shape)
    return pts


def conic_fit(pts):
    """Direct least squares conic (Hartley-normalised SVD) -> (r0, c0, a, b, theta) with a >= b, theta in (-pi/2, pi/2]."""
    mean = pts.mean(0)
    scale = np.sqrt(((pts - mean) ** 2).sum(1).mean())
    x, y = (pts[:, 1] - mean[1]) / scale, (pts[:, 0] - mean[0]) / scale   # x = column, y = row (local fit variables only)
    d = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=1)
    coef = np.linalg.svd(d)[2][-1]
    a_, b_, c_, d_, e_, f_ = coef
    m = np.array([[a_, b_ / 2], [b_ / 2, c_]])
    cx, cy = np.linalg.solve(2 * m, -np.array([d_, e_]))
    f0 = f_ + 0.5 * (d_ * cx + e_ * cy)
    w, vec = np.linalg.eigh(m)
    axes = np.sqrt(-f0 / w)                     # semi-axes along eigenvectors
    order = np.argsort(-axes)
    a_len, b_len = axes[order[0]], axes[order[1]]
    ex, ey = vec[:, order[0]]                   # major axis as (col, row)
    t = np.arctan2(ey, ex)                      # from +column toward +row
    t = (t + np.pi / 2) % np.pi - np.pi / 2     # into [-pi/2, pi/2)
    if t <= -np.pi / 2 + 1e-12:
        t = np.pi / 2
    return dict(r0=cy * scale + mean[0], c0=cx * scale + mean[1], a=a_len * scale, b=b_len * scale, theta=t)


def fourier_resample(img, shape):
    """Reference resample by zero padding / cropping the centred spectrum (band-limited, even sizes)."""
    n0, n1 = img.shape
    m0, m1 = shape
    f = np.fft.fftshift(np.fft.fft2(img))
    out = np.zeros((m0, m1), dtype=complex)
    k0, k1 = min(n0, m0) // 2, min(n1, m1) // 2
    out[m0 // 2 - k0:m0 // 2 + k0, m1 // 2 - k1:m1 // 2 + k1] = f[n0 // 2 - k0:n0 // 2 + k0, n1 // 2 - k1:n1 // 2 + k1]
    return np.fft.ifft2(np.fft.ifftshift(out)).real * (m0 * m1) / (n0 * n1)


MODES = [(1, 0, 3.0, 0.3), (0, 2, 2.0, 1.1), (3, -2, 1.5, -0.7), (-4, 5, 1.0, 2.0), (6, 6, 0.7, 0.4)]   # (kr, kc, amp, phase)


def band_limited(n0, n1):
    """Samples of f(x,y) = 2 + sum amp cos(2 pi (kr r / L + kc c / L) + phase) on the periodic domain of 1 unit; sampling at any
    n0, n1 > 2 * max|k| reproduces the same continuous function, so the truth at the output size is exact."""
    r = (np.arange(n0) / n0)[:, None]
    c = (np.arange(n1) / n1)[None, :]
    out = np.full((n0, n1), 2.0)
    for kr, kc, amp, ph in MODES:
        out += amp * np.cos(2 * np.pi * (kr * r + kc * c) + ph)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(1)
    n_q, n_phi = 48, 90
    truth = {
        "generator": "CuboKit/tools/polar-ellipse/planted.py",
        "upstream_pin": "none; no py4DSTEM or orix was run. The Mac pin scripts add the upstream numbers.",
        "convention": "see the module docstring: (row, col), theta of the major axis from +column toward +row, "
                      "rho = sqrt((u/a)^2+(v/b)^2), q = a*rho, phi = atan2(v*a/b, u)",
        "detector": [DET, DET], "scan": list(SCAN), "ring_scales": SCALES, "ring_amplitudes": AMPS, "ring_width_rho": WIDTH,
        "scan_scale": "m_p = 1 + 0.25 * flat_scan_index / (scan_rows * scan_cols), row-major",
        "background": dict(BG, form="amp * exp(-q / length) + const, q = a * rho (circularised pixels)"),
        "polar": {"n_q": n_q, "n_phi": n_phi, "q_k": "(k + 0.5) * dq, dq = (det/2)/n_q", "phi_j": "j * 2 pi / n_phi",
                  "tolerance_rationale": "Truth is the analytic function at the grid points. A bilinear polar resample of the "
                                         "cube carries an interpolation error that grows with the ring gradient (ring "
                                         "width 0.04 a ~ 1 px): compare the ring positions (argmax of the phi mean, "
                                         "expect |dq| <= 0.25 px) and the radial background away from rings "
                                         "(rtol 1e-2); do not demand pixel equality on the ring flanks."},
        "ellipse_fit": {"tolerance_rationale": "Exact points: any direct or iterative conic fit recovers the parameters to "
                                               "round-off (use atol 1e-9 px, 1e-12 rad; theta not compared for the circle). "
                                               "Noisy points (sigma 0.05 px, N = 64, seed 1): the centre and axes scatter by "
                                               "about sigma*sqrt(2/N) = 0.009 px; the recorded selfcheck is one realisation, so "
                                               "use atol 0.05 px and 2e-3 rad (near_circle: theta atol 0.3, a - b ~ 0.5 px)."},
        "cases": {},
        "fourier_resample": {},
    }
    for name, p in CASES.items():
        cube = make_cube(p)
        np.save(os.path.join(args.out, f"cube_{name}.npy"), cube)
        pol, q, phi, dq = polar_truth(p, n_q, n_phi)
        np.save(os.path.join(args.out, f"polar_truth_{name}.npy"), pol)
        exact = ring_points(p, 64)
        noisy = ring_points(p, 64, rng, 0.05)
        np.save(os.path.join(args.out, f"ring_points_{name}.npy"), exact)
        np.save(os.path.join(args.out, f"ring_points_{name}_noisy.npy"), noisy)
        fit_e, fit_n = conic_fit(exact), conic_fit(noisy)
        keys = ["r0", "c0", "a", "b"] + ([] if name == "circle" else ["theta"])
        err_e = max(abs(fit_e[k] - p[k]) for k in keys)
        err_n = max(abs(fit_n[k] - p[k]) for k in ["r0", "c0", "a", "b"])
        err_nt = 0.0 if name == "circle" else abs(fit_n["theta"] - p["theta"])
        truth["cases"][name] = {
            "params": p, "dq_px": dq, "ring_radii_circularised_px": [p["a"] * s for s in SCALES],
            "ring_points_exact": f"ring_points_{name}.npy", "ring_points_noisy": f"ring_points_{name}_noisy.npy",
            "selfcheck": {"conic_fit_exact_max_abs_error": err_e, "conic_fit_noisy_max_abs_error_px": err_n,
                          "conic_fit_noisy_theta_error_rad": err_nt},
        }
        print(f"{name:12s} conic fit exact err {err_e:.2e}  noisy err {err_n:.3f} px, theta {err_nt:.4f} rad")
    for n_in, n_out in [((32, 32), (48, 48)), ((48, 48), (32, 32)), ((40, 24), (64, 20))]:
        src, dst = band_limited(*n_in), band_limited(*n_out)
        tag = f"{n_in[0]}x{n_in[1]}_to_{n_out[0]}x{n_out[1]}"
        np.save(os.path.join(args.out, f"resample_{tag}_input.npy"), src)
        np.save(os.path.join(args.out, f"resample_{tag}_truth.npy"), dst)
        err = float(np.abs(fourier_resample(src, n_out) - dst).max())
        truth["fourier_resample"][tag] = {"selfcheck_max_abs_error": err}
        print(f"resample {tag:22s} numpy reference error {err:.2e}")
    truth["fourier_resample"]["signal"] = {
        "form": "2 + sum amp cos(2 pi (kr r/n0 + kc c/n1) + phase) sampled at r = i/n0, c = j/n1 (periodic unit domain)",
        "modes_kr_kc_amp_phase": MODES,
        "tolerance_rationale": "Every mode has |k| <= 6 < min(n)/2 - 2 for all sizes (smallest axis 20 -> Nyquist 10), so Fourier "
                               "resampling by padding or cropping the spectrum is exact: atol 1e-10 (FFT round-off). Even sizes "
                               "only; odd sizes and the Nyquist row are not covered here."}
    with open(os.path.join(args.out, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
