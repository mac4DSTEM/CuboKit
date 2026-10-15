"""CuboKit slice 4: the pin generator (docs/registration-slice-4.md).

    References/py4dstem-venv/bin/python -I -B tools/slice-4/pins.py synthetic
    References/py4dstem-venv/bin/python -I -B tools/slice-4/pins.py export-bullseye
    References/py4dstem-venv/bin/python -I -B tools/slice-4/pins.py real [name ...]

synthetic -> Tests/CuboVirtualTests/Fixtures/slice-4-pins.json (+ planted cubes as .npy), before any Swift;
export-bullseye -> $CUBOKIT_REAL_DATA/bullseye.npy (the owner's 20 vacuum-probe images as a 1 x 20 x 250 x 250 cube);
real -> $CUBOKIT_REAL_DATA/<name>-s4-pins.json beside slice 1's exports.
py4DSTEM's own results beside the float64 evaluation of each definition (section 3), which the kit is held to.
"""
import datetime, importlib.util, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
from py4DSTEM.datacube import DataCube
from py4DSTEM.braggvectors import Probe
from py4DSTEM.preprocess.utils import get_shifted_ar
from py4DSTEM.process.calibration import get_probe_size, get_origin
from py4DSTEM.process.calibration.origin import get_origin_friedel
from py4DSTEM.process.utils.multicorr import upsampled_correlation

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s3", os.path.join(HERE, "..", "slice-3", "pins.py"))
s3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s3)
KIT, FIX, REAL = s3.KIT, s3.FIX, s3.REAL
arr, versions = s3.arr, s3.versions
CHECKS = []
DATA = os.environ.get("MACEM_DATA", os.path.join(KIT, "..", "data"))
BULLSEYE = os.path.join(DATA, "4dstem", "training_dataset", "calibrationData_bullseyeProbe.h5")


def check(name, ok, detail=""):
    CHECKS.append(name)
    assert ok, f"SELF-CHECK FAILED: {name} {detail}"


def normwise(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def relative(a, b):
    """Slice 1's relative distance with the floor 1e-6 · max|want|."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    floor = 1e-6 * np.max(np.abs(b))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), max(floor, 1e-300))))


def bools(m):
    return [int(v) for v in np.asarray(m, bool).reshape(-1)]


# ---------------- 3.1 virtual diffraction over a selection, float64 ----------------
def flat(c32):
    R, C, qy, qx = c32.shape
    return c32.reshape(R * C, qy * qx)


def blocks(n, size):
    return [(k, min(k + size, n)) for k in range(0, n, size)]


def selection_f64(c32, stat, sel):
    """sel: None (all), ('mask', bool R x C) or ('weights', float R x C). Detector blocks keep memory small."""
    R, C, qy, qx = c32.shape
    X = flat(c32)
    out = np.zeros(qy * qx)
    rows = np.arange(R * C) if sel is None or sel[0] == "weights" else np.nonzero(sel[1].reshape(-1))[0]
    w = sel[1].reshape(-1).astype(np.float64) if sel is not None and sel[0] == "weights" else None
    size = max(1, int(2.5e7 // max(len(rows), 1)))
    for a, b in blocks(qy * qx, size):
        v = X[rows, a:b].astype(np.float64)
        if w is not None:
            v = v * w[:, None]
        if stat == "mean":
            out[a:b] = np.sum(v, axis=0) / (np.sum(w) if w is not None else len(rows))
        elif stat == "max":
            out[a:b] = np.max(v, axis=0)
        else:
            out[a:b] = np.median(v, axis=0)          # sort, middle, (a + b) / 2 in float64: the definition
    return out.reshape(qy, qx)


def selection_own(cnative, stat, sel):
    dc = DataCube(data=cnative)
    mask = None if sel is None else (sel[1].astype(bool) if sel[0] == "mask" else sel[1].astype(np.float64))
    return np.asarray(dc.get_virtual_diffraction(stat, mask=mask, verbose=False).data, np.float64)


def shifted_selection_f64(c32, orow, ocol, mode, stat, sel):
    R, C, qy, qx = c32.shape
    sr, sc = np.mean(orow) - orow, np.mean(ocol) - ocol
    total = np.zeros((qy, qx)); mx = np.full((qy, qx), -np.inf); count = 0.0
    for a in range(R):
        for b in range(C):
            if sel is not None and sel[0] == "mask" and not sel[1][a, b]:
                continue
            dp = c32[a, b].astype(np.float64)
            dp = np.roll(dp, (int(np.round(sr[a, b])), int(np.round(sc[a, b]))), (0, 1)) if mode == "rounded" \
                else s3.fourier_shift(dp, sr[a, b], sc[a, b])
            w = sel[1][a, b] if sel is not None and sel[0] == "weights" else 1.0
            total += dp * w; mx = np.maximum(mx, dp * w); count += w
    return (total / count) if stat == "mean" else mx


def shifted_selection_own(c32, orow, ocol, mode, stat, sel):
    dc = s3.py4dstem_calibrated(c32, orow, ocol)
    mask = None if sel is None else (sel[1].astype(bool) if sel[0] == "mask" else sel[1].astype(np.float64))
    return np.asarray(dc.get_virtual_diffraction(stat, mask=mask, shift_center=True, subpixel=(mode == "fourier"),
                                                 verbose=False).data, np.float64)


# ---------------- 3.2 origins with a scan mask, the fast centre, remove_NaN ----------------
def remove_nan(m):
    q = np.array(m, np.float64)
    q[np.isnan(q)] = np.mean(q[~np.isnan(q)])
    return q


# ---------------- 3.3 the Friedel origin, float64 ----------------
def dft_upsample(C, u, c):
    Nr, Nc = C.shape
    n = int(np.ceil(1.5 * u))
    fr = np.fft.ifftshift(np.arange(Nr)) - np.floor(Nr / 2)
    fc = np.fft.ifftshift(np.arange(Nc)) - np.floor(Nc / 2)
    K = np.exp((-1j * 2 * np.pi / (Nc * u)) * np.outer(fc, np.arange(n) - c[1]))
    Rk = np.exp((-1j * 2 * np.pi / (Nr * u)) * np.outer(np.arange(n) - c[0], fr))
    return np.real(Rk @ np.conj(C) @ K)


def parabola(c0, cp, cm):
    with np.errstate(all="ignore"):
        return (cp - cm) / (4.0 * c0 - 2.0 * cp - 2.0 * cm)


def friedel_one(P, mask=None, u=1):
    """Section 3.3 in float64, written from the text. Returns (row, column, peak, gap) where gap is the distance of the
    second-largest cc value below the peak over max|cc| (the near-tie measure of section 4)."""
    Qr, Qc = P.shape
    im = np.pad(np.asarray(P, np.float64), ((0, Qr), (0, Qc)))
    if mask is None:
        cc = np.real(np.fft.ifft2(np.fft.fft2(im) ** 2))
    else:
        mp = np.pad(np.asarray(mask, np.float64), ((0, Qr), (0, Qc)), constant_values=(1.0, 1.0))
        M = np.fft.fft2(mp)
        t1 = np.real(np.fft.ifft2(np.fft.fft2(im) ** 2) * np.fft.ifft2(M ** 2))
        t2 = np.real(np.fft.ifft2(np.fft.fft2(im ** 2) * M))
        t3 = im * mp
        with np.errstate(all="ignore"):
            cc = (t1 - t3) / (t2 - t3)
    Nr, Nc = cc.shape
    k = int(np.argmax(cc)); x, y = divmod(k, Nc)
    fin = cc[np.isfinite(cc)].reshape(-1)
    top = np.partition(fin, -2)[-2:] if fin.size >= 2 else np.array([0.0, 0.0])
    gap = float((top[1] - top[0]) / max(np.max(np.abs(fin)), 1e-300)) if fin.size >= 2 else 0.0
    dx = parabola(cc[x, y], cc[(x + 1) % Nr, y], cc[(x - 1) % Nr, y])
    dy = parabola(cc[x, y], cc[x, (y + 1) % Nc], cc[x, (y - 1) % Nc])
    sx, sy = x + dx, y + dy
    if u > 1:
        assert u > 2
        sx, sy = np.round(sx * u) / u, np.round(sy * u) / u
        g = np.fix(np.ceil(u * 1.5) / 2)
        U = dft_upsample(np.fft.fft2(cc), u, (g - u * sx, g - u * sy))
        n = U.shape[0]
        a, b = divmod(int(np.argmax(U)), n)
        if 0 < a < n - 1 and 0 < b < n - 1:
            ex = parabola(U[a, b], U[a + 1, b], U[a - 1, b]); ey = parabola(U[a, b], U[a, b + 1], U[a, b - 1])
        else:
            ex, ey = 0.0, 0.0
        sx = sx + (a - g + ex) / u; sy = sy + (b - g + ey) / u
    with np.errstate(all="ignore"):
        return float(np.remainder(sx / 2, Qr)), float(np.remainder(sy / 2, Qc)), (x, y), gap


def friedel_f64(c32, mask=None, u=1):
    R, C = c32.shape[:2]
    orow = np.zeros((R, C)); ocol = np.zeros((R, C)); peak = np.zeros((R, C, 2), int); gap = np.zeros((R, C))
    for a in range(R):
        for b in range(C):
            orow[a, b], ocol[a, b], peak[a, b], gap[a, b] = friedel_one(c32[a, b], mask, u)
    return orow, ocol, peak, gap


def friedel_own(cube, mask=None, u=1):
    q = get_origin_friedel(DataCube(data=cube), mask=mask, upsample_factor=u)
    return np.asarray(q[0], np.float64), np.asarray(q[1], np.float64)


def friedel_entry(name, c32, cnative, mask, u):
    orow, ocol, peak, gap = friedel_f64(c32, mask, u)
    own = friedel_own(cnative, mask, u)
    alg = friedel_own(c32.astype(np.float64), mask, u)     # py4DSTEM's own algorithm in float64: the text is py4DSTEM's
    d_alg = float(np.nanmax(np.abs(np.concatenate([alg[0] - orow, alg[1] - ocol]).reshape(-1))))
    check(f"{name}: Friedel (mask {mask is not None}, u {u}) written from the text equals py4DSTEM's in float64",
          d_alg <= 1e-9, f"{d_alg}")
    with np.errstate(all="ignore"):
        d_own = float(np.nanmax(np.abs(np.concatenate([own[0] - orow, own[1] - ocol]).reshape(-1))))
    return {"mask": None if mask is None else bools(mask), "upsample": u, "row": arr(orow), "column": arr(ocol),
            "peak": peak.reshape(-1).tolist(), "gap": arr(gap), "row_py4dstem": arr(own[0]), "column_py4dstem": arr(own[1]),
            "distance_py4dstem_f64_algorithm_px": d_alg, "distance_py4dstem_own_px": d_own,
            "near_ties": int(np.sum(gap <= 1e-12))}


# ---------------- 3.4 the beamstop mask ----------------
def beamstop_f64(im, threshold=0.25, distance_edge=2.0, include_edges=True, sigma=0.0, scale_radial=None):
    im = np.array(im, np.float64)
    qy, qx = im.shape
    if sigma > 0:
        im = gaussian_filter(im, sigma, mode="nearest")
    if scale_radial is not None:
        x = np.arange(qy) * 2.0 / qy; y = np.arange(qx) * 2.0 / qx
        ya, xa = np.meshgrid(y - np.mean(y), x - np.mean(x))
        im = im * (1.0 + np.sqrt(xa ** 2 + ya ** 2) * scale_radial)
    s = np.sort(im.ravel())
    ind = int(np.round(np.clip(s.shape[0] * threshold, 0, s.shape[0])))
    keep = im >= s[ind]
    keep = np.logical_not(binary_fill_holes(np.logical_not(keep)))
    keep = binary_fill_holes(keep)
    if include_edges:
        keep[0, :] = False; keep[:, 0] = False; keep[-1, :] = False; keep[:, -1] = False
    return distance_transform_edt(keep) < distance_edge


def beamstop_own(cnative, **kw):
    dc = DataCube(data=cnative)
    return np.asarray(dc.get_beamstop_mask(returncalc=True, **kw).data, bool)


# ---------------- 3.5 the vacuum probe ----------------
def damp(probe, r0, c0, alpha, s=0.1, w=1.0, p=1.0):
    qy, qx = probe.shape
    qyy, qxx = np.meshgrid(np.arange(qx), np.arange(qy))
    qrr = np.hypot(qxx - r0, qyy - c0)
    f = 1 + (alpha * (1 + s) - qrr) / (alpha * w)
    f = np.minimum(np.maximum(f, 0), 1)
    return probe * np.sin(f) ** p


def zero(probe, r0, c0, alpha, alpha_max=1.2):
    qy, qx = probe.shape
    qyy, qxx = np.meshgrid(np.arange(qx), np.arange(qy))
    return probe * (np.hypot(qxx - r0, qyy - c0) < alpha * alpha_max)


def vacuum_f64(c32, roi=None, align=True, zero_vacuum=True, dmask=None):
    R, C, qy, qx = c32.shape
    roi = np.ones((R, C), bool) if roi is None else roi
    pos = list(zip(*np.nonzero(roi)))
    probe = c32[pos[0]].astype(np.float64)
    shifts, gaps = [], []
    for n in range(1, len(pos)):
        d = c32[pos[n]].astype(np.float64) * (1 if dmask is None else dmask)
        if align:
            cc = np.maximum(np.real(np.fft.ifft2(np.fft.fft2(probe) * np.conj(np.fft.fft2(d)))), 0)
            sr, sc = divmod(int(np.argmax(cc)), qx)
            top = np.partition(cc.reshape(-1), -2)[-2:]
            gaps.append(float((top[1] - top[0]) / max(np.max(cc), 1e-300)))
            shifts.append((sr, sc))
            d = np.roll(d, (sr, sc), axis=(0, 1))
        probe = probe * (n - 1) / n + d / n
    r, r0, c0 = get_probe_size(probe)
    final = damp(probe, r0, c0, r) if zero_vacuum else probe
    return probe, (float(r), float(r0), float(c0)), final, shifts, gaps


def vacuum_own(cube, roi=None, align=True, zero_vacuum=True, dmask=None):
    dc = DataCube(data=cube)
    p = dc.get_vacuum_probe(ROI=roi, align=align, zero_vacuum=zero_vacuum, mask=dmask, plot=False, returncalc=True)
    return np.asarray(p.probe, np.float64), (float(p.alpha), float(p.origin[0]), float(p.origin[1]))


def vacuum_entry(name, c32, cnative, roi=None, align=True, zero_vacuum=True, dmask=None):
    avg, est, final, shifts, gaps = vacuum_f64(c32, roi, align, zero_vacuum, dmask)
    alg, alg_est = vacuum_own(c32.astype(np.float64), roi, align, zero_vacuum, dmask)
    check(f"{name}: vacuum probe written from the text equals py4DSTEM's in float64 (normwise 1e-12)",
          normwise(alg, final) <= 1e-12, f"{normwise(alg, final)}")
    own, own_est = vacuum_own(cnative, roi, align, zero_vacuum, dmask)
    return {"roi": None if roi is None else bools(roi), "align": align, "zero_vacuum": zero_vacuum,
            "detector_mask": None if dmask is None else arr(dmask), "averaged": arr(avg), "probe": arr(final),
            "radius": "%.17g" % est[0], "center_row": "%.17g" % est[1], "center_column": "%.17g" % est[2],
            "shifts": [list(s) for s in shifts], "near_ties": int(sum(g <= 1e-12 for g in gaps)),
            "probe_py4dstem": arr(own), "estimate_py4dstem": ["%.17g" % v for v in own_est],
            "distance_py4dstem_f64_algorithm": normwise(alg, final), "distance_py4dstem_own": normwise(own, final)}


# ---------------- 3.6 the kernels and the synthetic probe ----------------
def wrapped(n):
    return np.mod(np.arange(n) + n // 2, n) - n // 2


def shift_f64(ar, xs, ys, bilinear):
    ar = np.asarray(ar, np.float64)
    if not bilinear:
        return s3.fourier_shift(ar, xs, ys)
    xF, yF = int(np.floor(xs)), int(np.floor(ys)); wx, wy = xs - xF, ys - yF
    return (np.roll(ar, (xF, yF), (0, 1)) * ((1 - wx) * (1 - wy)) + np.roll(ar, (xF + 1, yF), (0, 1)) * (wx * (1 - wy))
            + np.roll(ar, (xF, yF + 1), (0, 1)) * ((1 - wx) * wy) + np.roll(ar, (xF + 1, yF + 1), (0, 1)) * (wx * wy))


def kernel_f64(probe, kind, origin=None, bilinear=None, sigma=None, radii=None, profile="sine_squared"):
    probe = np.asarray(probe, np.float64)
    qy, qx = probe.shape
    if origin is None:
        _, r0, c0 = get_probe_size(probe)
    else:
        r0, c0 = origin
    if kind == "flat":
        return shift_f64(probe / np.sum(probe), -r0, -c0, bool(bilinear))
    k = shift_f64(probe, -r0, -c0, True if bilinear is None else bilinear)
    qc, qr = np.meshgrid(wrapped(qx), wrapped(qy))
    if kind == "gaussian":
        q2 = qr ** 2 + qc ** 2
        s2 = np.sum(q2 * k) / np.sum(k)
        g = np.exp(-q2 / (2 * s2 * sigma ** 2))
        return k / np.sum(k) - g / np.sum(g)
    ri, ro = radii
    q = np.hypot(qr, qc)
    if profile == "logistic":
        s = 1 / (1 + np.exp((q - 0.5 * (ro + ri)) / (0.25 * (ro - ri))))
    else:
        s = np.cos((np.pi / 2) * np.minimum(np.maximum((q - ri) / (ro - ri), 0.0), 1.0)) ** 2
    return k / np.sum(k) - s / np.sum(s)


def kernel_own(probe, kind, origin=None, bilinear=None, sigma=None, radii=None, profile="sine_squared"):
    probe = np.asarray(probe, np.float64)
    if kind == "flat":
        return Probe.get_probe_kernel_flat(probe, origin=origin, bilinear=bool(bilinear))
    b = True if bilinear is None else bilinear
    if kind == "gaussian":
        return Probe.get_probe_kernel_edge_gaussian(probe, sigma, origin=origin, bilinear=b)
    return Probe.get_probe_kernel_edge_sigmoid(probe, radii, origin=origin, type=profile, bilinear=b)


def kernels_entry(name, probe, alpha, origin=None):
    cases = [dict(kind="flat"), dict(kind="flat", bilinear=True), dict(kind="gaussian", sigma=4.0),
             dict(kind="gaussian", sigma=2.5, bilinear=False), dict(kind="sigmoid", radii=(alpha, 2 * alpha)),
             dict(kind="sigmoid", radii=(alpha, 2 * alpha), profile="logistic"),
             dict(kind="sigmoid", radii=(0.8 * alpha, 1.7 * alpha), bilinear=False)]
    out = []
    for c in cases:
        c = dict(c, origin=origin)
        k = kernel_f64(probe, **c)
        own = kernel_own(probe, **c)
        d = normwise(own, k)
        check(f"{name}: kernel {c} written from the text equals py4DSTEM's (normwise 1e-12)", d <= 1e-12, f"{d}")
        case = {key: (list(v) if isinstance(v, tuple) else v) for key, v in c.items()}
        out.append(dict(case, kernel=arr(k), distance_py4dstem=d, sum="%.17g" % float(np.sum(k))))
    return out


def synthetic_probe_entry(radius, width, shape):
    qy, qx = shape
    qyy, qxx = np.meshgrid(np.arange(qx), np.arange(qy))
    p = 1 / (1 + np.exp(4 * (np.sqrt((qxx - qy / 2.0) ** 2 + (qyy - qx / 2.0) ** 2) - radius) / width))
    own = Probe.generate_synthetic_probe(radius, width, shape).probe
    check(f"synthetic probe {radius} {width} {shape} equals py4DSTEM's", normwise(own, p) <= 1e-14)
    return {"radius": radius, "width": width, "shape": list(shape), "probe": arr(p)}


# ---------------- one cube: every claim ----------------
def pin_cube(name, cube, friedel_masked=True, friedel_upsample=True, vacuum_roi=None, weighted_own=True):
    t0 = datetime.datetime.now()
    c32 = np.ascontiguousarray(cube).astype(np.float32)
    R, C, qy, qx = c32.shape
    integer = cube.dtype.kind in "ui"
    e = {"cube": name, "integer": integer}
    # 3.1 selections
    left = np.zeros((R, C), bool); left[:, : max(1, C // 2)] = True
    r_bf, rc0, cc0 = get_probe_size(np.mean(c32.astype(np.float64), axis=(0, 1)))
    ii, jj = np.indices((qy, qx))
    bf_mask = ((ii - rc0) ** 2 + (jj - cc0) ** 2 < r_bf ** 2).astype(np.float64)
    bf = np.einsum("abij,ij->ab", c32.astype(np.float64), bf_mask)
    weights = bf / np.max(np.abs(bf))
    e["scan_mask"] = bools(left); e["weights"] = arr(weights)
    sel = {}
    for key, s in (("all", None), ("mask", ("mask", left)), ("weights", ("weights", weights))):
        for stat in ("mean", "max", "median"):
            if key == "all" and stat != "median":
                continue
            want = selection_f64(c32, stat, s)
            ent = {"value": arr(want)}
            if key != "weights" or weighted_own:
                own = selection_own(cube, stat, s)
                ent["value_py4dstem"] = arr(own)
                ent["identical_py4dstem"] = bool(np.array_equal(own, want))
                ent["distance_py4dstem"] = relative(own, want)
            sel[f"{key}_{stat}"] = ent
    e["selections"] = sel
    # shifted with a selection, on the float64 origins of slice 3
    r = get_probe_size(np.max(c32, axis=(0, 1)))[0]
    e["radius"] = "%.17g" % r
    orow, ocol, arg = s3.origins_f64(c32, r)
    e["origin_row"] = arr(orow); e["origin_column"] = arr(ocol)
    shifted = {}
    for mode in ("rounded", "fourier"):
        for key, s in (("mask", ("mask", left)), ("weights", ("weights", weights))):
            for stat in ("mean", "max"):
                want = shifted_selection_f64(c32, orow, ocol, mode, stat, s)
                own = shifted_selection_own(c32, orow, ocol, mode, stat, s)
                shifted[f"{mode}_{key}_{stat}"] = {"value": arr(want), "value_py4dstem": arr(own),
                                                   "distance_py4dstem": normwise(own, want)}
    e["shifted"] = shifted
    # 3.2 origins with the scan mask and the fast centre
    own = get_origin(DataCube(data=c32), r=r, rscale=1.2, mask=left)
    own_fast = get_origin(DataCube(data=c32), r=r, rscale=1.2, fast_center=True)
    masked_row = np.where(left, orow, np.nan); masked_col = np.where(left, ocol, np.nan)
    e["origins_masked"] = {"row": arr(masked_row), "column": arr(masked_col),
                           "row_py4dstem_data": arr(np.asarray(own[0].data)), "column_py4dstem_data": arr(np.asarray(own[1].data)),
                           "distance_py4dstem_px": float(np.max(np.abs(np.asarray(own[0].data)[left] - orow[left]))),
                           "filled_row": arr(remove_nan(masked_row)), "filled_column": arr(remove_nan(masked_col))}
    e["fast_center"] = {"row": arr(arg[..., 0]), "column": arr(arg[..., 1]),
                        "row_py4dstem": arr(own_fast[0]), "column_py4dstem": arr(own_fast[1]),
                        "differ_py4dstem": int(np.sum((np.asarray(own_fast[0]) != arg[..., 0]) | (np.asarray(own_fast[1]) != arg[..., 1])))}
    # 3.4 beamstop on the float64 mean
    mean = np.mean(c32.astype(np.float64), axis=(0, 1))
    bs = {}
    for key, kw in (("default", {}), ("no_edges", {"include_edges": False, "threshold": 0.3}),
                    ("sigma", {"sigma": 1.5, "distance_edge": 3.0}), ("radial", {"scale_radial": 0.5, "threshold": 0.2})):
        try:
            want = beamstop_f64(mean, **kw)
        except Exception as ex:
            bs[key] = {"kw": kw, "refused": str(ex)}; continue
        if not np.any(~(mean >= 0) | True):
            pass
        own = beamstop_own(cube, **kw)
        bs[key] = {"kw": kw, "mask": bools(want), "mask_py4dstem": bools(own), "differ_py4dstem": int(np.sum(own != want))}
    e["mean_f64"] = arr(mean)
    e["beamstop"] = bs
    # 3.3 Friedel
    fr = {"plain": friedel_entry(name, c32, cube, None, 1)}
    if friedel_masked:
        fr["masked"] = friedel_entry(name, c32, cube, ~beamstop_f64(mean), 1)
    if friedel_upsample:
        fr["upsampled"] = friedel_entry(name, c32, cube, None, 16)
        if friedel_masked:
            fr["masked_upsampled"] = friedel_entry(name, c32, cube, ~beamstop_f64(mean), 8)
    e["friedel"] = fr
    # 3.5, 3.6 the vacuum probe of a region and its kernels
    roi = np.zeros((R, C), bool)
    if vacuum_roi is None:
        roi[0, :] = True
    else:
        roi[vacuum_roi] = True
    v = vacuum_entry(name, c32, cube, roi)
    e["vacuum"] = v
    e["kernels"] = kernels_entry(name, np.array([float(x) for x in v["probe"]]).reshape(qy, qx), float(v["radius"]))
    e["seconds"] = (datetime.datetime.now() - t0).total_seconds()
    return e


# ---------------- synthetic and planted ----------------
def planted_median(R, C, qy=6, qx=7):
    rng = np.random.default_rng(20261015 + R * C)
    n = R * C
    cube = np.zeros((n, qy * qx))
    for p in range(qy * qx):
        cube[:, p] = rng.permutation(n) * 0.5 * (p + 1) - 3.0
    return cube.reshape(R, C, qy, qx).astype(np.float32)


def symmetric_pattern(qy, qx, cr, cc, wedge=False):
    """Point-symmetric about (cr, cc) (2cr, 2cc integers) with support inside the reflected array: a central disk, two
    Bragg disk pairs, a weak symmetric halo. With `wedge`, a beamstop arm (zeroed) and its mask."""
    ii, jj = np.indices((qy, qx)).astype(np.float64)
    d2 = (ii - cr) ** 2 + (jj - cc) ** 2
    lim = min(cr, cc, qy - 1 - cr, qx - 1 - cc)
    p = np.where(d2 < 16, 1000.0, 0.0) + 20.0 * np.exp(-d2 / 40.0)
    for vr, vc, a in ((5.0, 3.0, 300.0), (-2.0, 6.0, 180.0)):
        for s in (1, -1):
            p += np.where((ii - cr - s * vr) ** 2 + (jj - cc - s * vc) ** 2 < 4, a, 0.0)
    p = np.where(d2 <= lim ** 2, p, 0.0)
    m = np.ones((qy, qx), bool)
    if wedge:
        m = ~((np.abs(ii - cr) <= 1.5) & (jj > cc + 1.0))
        p = p * m
    return p, m


def planted_friedel():
    qy, qx = 32, 34
    centres = [(15.0, 16.0), (15.5, 16.5), (16.0, 15.5), (14.5, 17.0), (16.5, 16.0), (15.0, 17.5)]
    plain = np.zeros((2, 3, qy, qx)); masked = np.zeros((2, 3, qy, qx))
    for k, (cr, cc) in enumerate(centres):
        plain[k // 3, k % 3], _ = symmetric_pattern(qy, qx, cr, cc)
    p, m = symmetric_pattern(qy, qx, 15.5, 16.5, wedge=True)
    masked[:] = p
    return plain.astype(np.float32), masked.astype(np.float32), m, centres


def planted_gauss(cr, cc, qy=40, qx=40, s=2.5):
    ii, jj = np.indices((qy, qx)).astype(np.float64)
    return 1000.0 * np.exp(-((ii - cr) ** 2 + (jj - cc) ** 2) / (2 * s * s))


def planted_beamstop():
    qy, qx = 24, 24
    im = np.full((qy, qx), 100.0)
    im[10:14, 0:12] = 1.0           # the dark arm, reaching the left edge
    im[11, 6] = 100.0               # a bright speck inside the arm
    im[18:20, 18:20] = 1.0          # a dark hole inside the bright region
    dark = np.zeros((qy, qx), bool); dark[10:14, 0:12] = True
    dark[0, :] = dark[-1, :] = dark[:, 0] = dark[:, -1] = True     # include_edges
    want = np.zeros((qy, qx), bool)
    pts = np.argwhere(dark)
    for i in range(qy):
        for j in range(qx):
            want[i, j] = np.min((pts[:, 0] - i) ** 2 + (pts[:, 1] - j) ** 2) < 4    # d < 2 by hand: d² in {0, 1, 2}
    return im, want


def planted_vacuum():
    qy, qx = 32, 32
    ii, jj = np.indices((qy, qx)).astype(np.float64)
    disk = np.where((ii - 15) ** 2 + (jj - 16) ** 2 < 25, 100.0, 0.0) + np.where((ii - 15) ** 2 + (jj - 16) ** 2 < 36, 20.0, 0.0)
    shifts = [(0, 0), (2, -1), (-3, 2), (1, 3), (-1, -2), (4, 0)]
    cube = np.zeros((1, len(shifts), qy, qx))
    for k, s in enumerate(shifts):
        cube[0, k] = np.roll(disk, s, (0, 1))
    cube[0, 0, 2, 28] = 500.0       # a hot pixel only in p0
    return cube.astype(np.float32), disk, shifts


def run_synthetic():
    out = {"versions": versions(), "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    # median truth
    med = []
    for R, C in ((3, 5), (4, 4)):
        cube = planted_median(R, C)
        nm = f"median_{R}x{C}"
        np.save(os.path.join(FIX, nm + ".npy"), cube)
        n = R * C
        p = np.arange(cube.shape[2] * cube.shape[3])
        mid = ((n - 1) / 2.0) if n % 2 else ((n / 2 - 1) + n / 2) / 2.0
        truth = mid * 0.5 * (p + 1) - 3.0
        want = selection_f64(cube, "median", None).reshape(-1)
        check(f"{nm}: the median is the closed form", np.array_equal(want, truth))
        out_cube = cube.copy(); out_cube[0, 0] = 1e30
        np.save(os.path.join(FIX, nm + "_outlier.npy"), out_cube)
        check(f"{nm}: one outlier moves the median by at most one rank", np.max(np.abs(selection_f64(out_cube, "median", None).reshape(-1) - truth)) <= 0.5 * (p[-1] + 1))
        med.append({"cube": nm, "median": arr(truth), "median_outlier": arr(selection_f64(out_cube, "median", None)),
                    "median_py4dstem": arr(selection_own(cube, "median", None))})
    out["median_truth"] = med
    # Friedel truth
    plain, masked, m, centres = planted_friedel()
    np.save(os.path.join(FIX, "friedel_plain.npy"), plain); np.save(os.path.join(FIX, "friedel_masked.npy"), masked)
    fr, fc, peak, gap = friedel_f64(plain)
    tr = np.array([c[0] for c in centres]).reshape(2, 3); tc = np.array([c[1] for c in centres]).reshape(2, 3)
    check("Friedel recovers the planted centres (peak at twice the centre)", np.array_equal(peak[..., 0], (2 * tr).astype(int)) and np.array_equal(peak[..., 1], (2 * tc).astype(int)))
    check("Friedel recovers the planted centres within 1e-9 px", np.max(np.abs(fr - tr)) <= 1e-9 and np.max(np.abs(fc - tc)) <= 1e-9, f"{np.max(np.abs(fr - tr))}")
    mr, mc, mpeak, _ = friedel_f64(masked, m)
    out["friedel_truth"] = {"centres": [list(c) for c in centres], "mask": bools(m),
                            "plain": friedel_entry("friedel_plain", plain, plain, None, 1),
                            "plain_upsampled": friedel_entry("friedel_plain", plain, plain, None, 16),
                            "masked": friedel_entry("friedel_masked", masked, masked, m, 1),
                            "masked_peak_is_twice_centre": bool(np.all(mpeak[..., 0] == 31) and np.all(mpeak[..., 1] == 33)),
                            "masked_distance_px": float(max(np.max(np.abs(mr - 15.5)), np.max(np.abs(mc - 16.5)))),
                            "unmasked_on_beamstop_distance_px": float(max(np.max(np.abs(friedel_f64(masked)[0] - 15.5)), np.max(np.abs(friedel_f64(masked)[1] - 16.5))))}
    gcube = np.zeros((1, 3, 40, 40))
    gcentres = [(19.3, 20.6), (18.75, 21.1), (20.2, 19.45)]
    for k, c in enumerate(gcentres):
        gcube[0, k] = planted_gauss(*c)
    gcube = gcube.astype(np.float32)
    np.save(os.path.join(FIX, "friedel_gauss.npy"), gcube)
    gr, gc, _, _ = friedel_f64(gcube, None, 16)
    gd = float(max(np.max(np.abs(gr.reshape(-1) - [c[0] for c in gcentres])), np.max(np.abs(gc.reshape(-1) - [c[1] for c in gcentres]))))
    check("Friedel u = 16 recovers fractional Gaussian centres within 0.05 px", gd <= 0.05, f"{gd}")
    out["friedel_gauss"] = {"centres": [list(c) for c in gcentres], "distance_px": gd,
                            "upsampled": friedel_entry("friedel_gauss", gcube, gcube, None, 16)}
    # beamstop truth
    im, want = planted_beamstop()
    got = beamstop_f64(im)
    check("beamstop: the planted mask by hand", np.array_equal(got, want), f"{np.sum(got != want)}")
    out["beamstop_truth"] = {"pattern": arr(im), "shape": list(im.shape), "mask": bools(want), "count": int(np.sum(want))}
    # vacuum truth
    vc, disk, shifts = planted_vacuum()
    np.save(os.path.join(FIX, "vacuum_planted.npy"), vc)
    avg, est, final, sh, _ = vacuum_f64(vc, None, True, False)
    check("vacuum: aligned average is the disk at p0 (relative 1e-13)", relative(avg, disk) <= 1e-13, f"{relative(avg, disk)}")
    check("vacuum: the hot pixel of p0 is absent", avg[2, 28] == 0.0)
    plain_mean = np.mean(vc[0, 1:].astype(np.float64), axis=0)
    avg_na = vacuum_f64(vc, None, False, False)[0]
    check("vacuum: align off is the mean of p1..p(L-1)", relative(avg_na, plain_mean) <= 1e-14)
    dmask = np.ones((32, 32)); dmask[:, 30:] = 0.0
    out["vacuum_truth"] = {"disk": arr(disk), "shifts": [list(s) for s in shifts], "plain_mean": arr(plain_mean),
                           "aligned": vacuum_entry("vacuum_planted", vc, vc, None, True, False),
                           "aligned_damped": vacuum_entry("vacuum_planted", vc, vc, None, True, True),
                           "unaligned": vacuum_entry("vacuum_planted", vc, vc, None, False, True),
                           "detector_mask": vacuum_entry("vacuum_planted", vc, vc, None, True, True, dmask)}
    a = out["vacuum_truth"]["aligned_damped"]
    pr = np.array([float(x) for x in a["averaged"]]).reshape(32, 32)
    r0, c0, al = float(a["center_row"]), float(a["center_column"]), float(a["radius"])
    out["vacuum_truth"]["zero_vacuum"] = {"center": [r0, c0], "alpha": al, "alpha_max": 1.2, "value": arr(zero(pr, r0, c0, al)),
                                          "value_py4dstem": arr(_own_zero(pr, r0, c0, al, 1.2))}
    out["vacuum_truth"]["damp_vacuum"] = {"center": [r0, c0], "alpha": al, "s": 0.2, "w": 1.5, "p": 2.0,
                                          "value": arr(damp(pr, r0, c0, al, 0.2, 1.5, 2.0)),
                                          "value_py4dstem": arr(_own_damp(pr, r0, c0, al, 0.2, 1.5, 2.0))}
    # kernels and synthetic probes
    sp = synthetic_probe_entry(6.0, 2.0, (32, 30))
    out["synthetic_probes"] = [sp, synthetic_probe_entry(4.3, 1.0, (25, 27))]
    sprobe = np.array([float(x) for x in sp["probe"]]).reshape(32, 30)
    out["kernels"] = {"probe": "synthetic_probes[0]", "alpha": 6.0, "cases": kernels_entry("synthetic", sprobe, 6.0),
                      "cases_given_origin": kernels_entry("synthetic", sprobe, 6.0, origin=(15.6, 14.3))}
    ks = [float(c["sum"]) for c in out["kernels"]["cases"]]
    check("kernel sums: flat 1, others 0", abs(ks[0] - 1) <= 1e-12 and all(abs(s) <= 1e-12 for s in ks[2:]))
    # full claims on small synthetic cubes
    out["cubes"] = []
    for nm in ("planted_f32", "planted_u16", "gaussian", "counts16"):
        cube = np.load(os.path.join(FIX, nm + ".npy"))
        out["cubes"].append(pin_cube(nm, cube))
    json.dump(out, open(os.path.join(FIX, "slice-4-pins.json"), "w"))
    print(f"pins.py slice 4 synthetic: {len(out['cubes'])} cubes, {len(CHECKS)} self-checks passed")
    for e in out["cubes"]:
        summary(e)


def _own_zero(pr, r0, c0, al, amax):
    p = Probe(np.array(pr)); p.alpha = al; p.origin = (r0, c0); p.zero_vacuum(alpha_max=amax)
    return p.probe


def _own_damp(pr, r0, c0, al, s, w, pw):
    p = Probe(np.array(pr)); p.alpha = al; p.origin = (r0, c0)
    return p.zero_vacuum_sinusoid(s=s, w=w, p=pw, update_inplace=False)


def summary(e):
    sel = e["selections"]
    fr = e["friedel"]
    print(f"  {e['cube']}: median own {'identical' if sel['all_median'].get('identical_py4dstem') else sel['all_median'].get('distance_py4dstem')}, "
          f"mask mean own {sel['mask_mean'].get('distance_py4dstem'):.1e}, weights mean own {sel['weights_mean'].get('distance_py4dstem', float('nan')):.1e}; "
          f"origins masked own {e['origins_masked']['distance_py4dstem_px']:.1e} px, fast differ {e['fast_center']['differ_py4dstem']}; "
          f"beamstop differ " + ",".join(f"{k} {v.get('differ_py4dstem', 'refused')}" for k, v in e["beamstop"].items()) + "; "
          "Friedel " + ", ".join(f"{k} own {v['distance_py4dstem_own_px']:.1e} px ties {v['near_ties']}" for k, v in fr.items()) + "; "
          f"vacuum own {e['vacuum']['distance_py4dstem_own']:.1e} ties {e['vacuum']['near_ties']}; {e['seconds']:.0f} s")


def export_bullseye():
    import h5py, hashlib
    with h5py.File(BULLSEYE, "r") as h:
        stack = np.asarray(h["4DSTEM_experiment/data/diffractionslices/probe_template/data"])
    cube = np.ascontiguousarray(np.moveaxis(stack, 2, 0)[None])
    path = os.path.join(REAL, "bullseye.npy")
    np.save(path, cube)
    print(f"bullseye.npy {cube.shape} {cube.dtype} sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()}")


REAL_OPTIONS = {
    "simau": {}, "particle": {}, "py4dstem_small": {},
    "auref": {"friedel_masked": False, "friedel_upsample": False, "weighted_own": False},
    "almgsi": {"friedel_masked": False, "friedel_upsample": False, "weighted_own": False},
    "bullseye": {"vacuum_roi": (slice(0, 1), slice(0, 20))},
}


def run_real(names):
    for name in names:
        cube = np.load(os.path.join(REAL, f"{name}.npy"))
        e = pin_cube(name, cube, **REAL_OPTIONS[name])
        json.dump(e, open(os.path.join(REAL, f"{name}-s4-pins.json"), "w"))
        print(f"pins.py slice 4 real {name}:", end="")
        summary(e)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("synthetic", "real", "export-bullseye"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "synthetic":
        run_synthetic()
    elif sys.argv[1] == "export-bullseye":
        export_bullseye()
    else:
        run_real(sys.argv[2:] or ["bullseye", "simau", "particle", "py4dstem_small", "auref", "almgsi"])
