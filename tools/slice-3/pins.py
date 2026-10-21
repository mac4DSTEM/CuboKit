"""CuboKit slice 3: the pin generator (docs/registration-slice-3.md).

    ../References/python/py4dstem/bin/python -I -B tools/slice-3/pins.py synthetic
    ../References/python/py4dstem/bin/python -I -B tools/slice-3/pins.py real [name ...]

synthetic -> Tests/CuboVirtualTests/Fixtures/slice-3-pins.json (+ planted cubes as .npy), before any Swift;
real -> $CUBOKIT_REAL_DATA/<name>-s3-pins.json and <name>-s3-<quantity>.npy beside slice 1's exports.
py4DSTEM's own results beside the float64 evaluation of each definition (section 3), which the kit is held to.
"""
import datetime, json, math, os, platform, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import py4DSTEM
from py4DSTEM.datacube import DataCube
from py4DSTEM.process.calibration import get_probe_size, get_origin, fit_origin

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIX = os.path.join(KIT, "Tests", "CuboVirtualTests", "Fixtures")
REAL = os.environ.get("CUBOKIT_REAL_DATA", os.path.expanduser("~/Library/Caches/CuboKit/real-data"))
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(name)
    assert ok, f"SELF-CHECK FAILED: {name} {detail}"


def arr(a):
    return ["%.17g" % float(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)]


def versions():
    return {"py4dstem": py4DSTEM.__version__, "py4dstem_commit": "f050d207851f8accd3aa550365baaf1f38230428",
            "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()}


# ---------------- 3.1 the Gaussian filter and the origin, float64 ----------------
def kernel(sigma):
    R = int(4.0 * sigma + 0.5)
    t = np.arange(-R, R + 1)
    k = np.exp(-0.5 / (sigma * sigma) * t ** 2)
    return k / k.sum(), R


def gauss_f64(dp, sigma):
    """Rows then columns, mode nearest, Double, written from the text (checked against scipy on float64 input)."""
    k, R = kernel(sigma)
    out = np.asarray(dp, dtype=np.float64)
    for axis in (0, 1):
        n = out.shape[axis]
        idx = np.clip(np.arange(n)[:, None] + np.arange(-R, R + 1)[None, :], 0, n - 1)
        moved = np.moveaxis(out, axis, 0)
        res = np.einsum("t,itj->ij", k, moved[idx]) if moved.ndim == 2 else None
        out = np.moveaxis(res, 0, axis)
    return out


def com(ar):
    nx, ny = ar.shape
    ii, jj = np.indices((nx, ny))
    t = np.sum(ar)
    return np.sum(ii * ar) / t, np.sum(jj * ar) / t


def origins_f64(cube32, r, rscale=1.2):
    R, C, qy, qx = cube32.shape
    ii, jj = np.indices((qy, qx))
    orow = np.zeros((R, C)); ocol = np.zeros((R, C)); arg = np.zeros((R, C, 2), dtype=int)
    for a in range(R):
        g = gaussian_filter(cube32[a].astype(np.float64), sigma=(0, r, r), mode="nearest")   # float64 = the definition
        for b in range(C):
            i0, j0 = np.unravel_index(int(np.argmax(g[b])), (qy, qx))
            mask = np.hypot(ii - i0, jj - j0) < r * rscale
            orow[a, b], ocol[a, b] = com(cube32[a, b].astype(np.float64) * mask)
            arg[a, b] = (i0, j0)
    return orow, ocol, arg


def argmax_py4dstem(cube32, r):
    R, C, qy, qx = cube32.shape
    arg = np.zeros((R, C, 2), dtype=int)
    for a in range(R):
        for b in range(C):
            arg[a, b] = np.unravel_index(int(np.argmax(gaussian_filter(cube32[a, b], r, mode="nearest"))), (qy, qx))
    return arg


# ---------------- 3.2 fits, float64 ----------------
def design(shape, model):
    x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
    ry, rx = np.meshgrid(y, x)
    X, Y = rx.reshape(-1), ry.reshape(-1)
    if model == "plane":
        return np.stack([X, Y, np.ones_like(X)], 1)
    if model == "parabola":
        return np.stack([np.ones_like(X), X, Y, X ** 2, Y ** 2, X * Y], 1)
    if model == "bezier_two":
        bx = [(1 - X) ** 2, 2 * (1 - X) * X, X ** 2]; by = [(1 - Y) ** 2, 2 * (1 - Y) * Y, Y ** 2]
        return np.stack([bx[i] * by[j] for i in range(3) for j in range(3)], 1)
    raise ValueError(model)


def fit_f64(m, model, robust=False, steps=3, thresh=2):
    m = np.asarray(m, dtype=np.float64)
    if model == "constant":
        return np.full(m.shape, np.mean(m))
    A = design(m.shape, model); d = m.reshape(-1)
    mask = np.isfinite(d)
    coef = None
    for k in range((steps if robust else 0) + 1):
        if k > 0:
            err = (A @ coef - d) ** 2
            mask[err > np.mean(err) * thresh ** 2] = False
        coef, *_ = np.linalg.lstsq(A[mask], d[mask], rcond=None)
    return (A @ coef).reshape(m.shape)


# ---------------- 3.3, 3.4 shifted virtual images and patterns, float64 ----------------
def make_mask(shape, g):
    qy, qx = shape
    ii, jj = np.indices(shape)
    if g["kind"] == "circle":
        return ((ii - g["centerRow"]) ** 2 + (jj - g["centerColumn"]) ** 2 < g["radius"] ** 2).astype(np.float64)
    if g["kind"] == "annulus":
        d2 = (ii - g["centerRow"]) ** 2 + (jj - g["centerColumn"]) ** 2
        return ((d2 > g["inner"] ** 2) & (d2 < g["outer"] ** 2)).astype(np.float64)
    raise ValueError(g["kind"])


def centred(g, r0, c0, p=None):
    h = dict(g)
    if r0 is not None:
        h["centerRow"] = g["centerRow"] + r0; h["centerColumn"] = g["centerColumn"] + c0
    if p is not None:
        for k in ("centerRow", "centerColumn", "radius", "inner", "outer"):
            if k in h: h[k] = h[k] / p
    return h


def shifted_image_f64(cube32, mask, orow, ocol, mode):
    R, C = orow.shape
    dr, dc = orow - np.mean(orow), ocol - np.mean(ocol)
    out = np.zeros((R, C))
    for a in range(R):
        for b in range(C):
            if mode == "rounded":
                m = np.roll(mask, (int(np.round(dr[a, b])), int(np.round(dc[a, b]))), axis=(0, 1))
            else:
                fr, fc = int(np.floor(dr[a, b])), int(np.floor(dc[a, b])); wr, wc = dr[a, b] - fr, dc[a, b] - fc
                m = (np.roll(mask, (fr, fc), (0, 1)) * ((1 - wr) * (1 - wc)) + np.roll(mask, (fr + 1, fc), (0, 1)) * (wr * (1 - wc))
                     + np.roll(mask, (fr, fc + 1), (0, 1)) * ((1 - wr) * wc) + np.roll(mask, (fr + 1, fc + 1), (0, 1)) * (wr * wc))
            out[a, b] = np.sum(cube32[a, b].astype(np.float64) * m)
    return out


def fourier_shift(ar, sr, sc):
    nx, ny = ar.shape
    qy, qx = np.meshgrid(np.fft.fftfreq(ny, 1), np.fft.fftfreq(nx, 1))
    w = np.exp(-(2j * np.pi) * (sc * qy + sr * qx))
    return np.real(np.fft.ifft2(np.fft.fft2(ar) * w))


def shifted_dp_f64(cube32, orow, ocol, mode):
    R, C, qy, qx = cube32.shape
    sr, sc = np.mean(orow) - orow, np.mean(ocol) - ocol
    total = np.zeros((qy, qx)); mx = np.full((qy, qx), -np.inf)
    for a in range(R):
        for b in range(C):
            dp = cube32[a, b].astype(np.float64)
            if mode == "rounded":
                dp = np.roll(dp, (int(np.round(sr[a, b])), int(np.round(sc[a, b]))), axis=(0, 1))
            else:
                dp = fourier_shift(dp, sr[a, b], sc[a, b])
            total += dp; mx = np.maximum(mx, dp)
    return total / (R * C), mx


def py4dstem_calibrated(cube32, orow, ocol, p=None):
    dc = DataCube(data=cube32)
    dc.calibration.set_origin((orow, ocol))
    if p is not None:
        dc.calibration.set_Q_pixel_size(p); dc.calibration.set_Q_pixel_units("A^-1")
    return dc


def pin_cube(name, cube, r=None, geometries=None):
    c32 = np.ascontiguousarray(cube).astype(np.float32)
    if r is None:
        r, _, _ = get_probe_size(np.max(c32, axis=(0, 1)))
    e = {"cube": name, "radius": "%.17g" % r, "rscale": 1.2}
    orow, ocol, arg = origins_f64(c32, r)
    own = get_origin(DataCube(data=c32), r=r, rscale=1.2)
    parg = argmax_py4dstem(c32, r)
    e.update({"origin_row": arr(orow), "origin_column": arr(ocol), "argmax": arg.reshape(-1).tolist(),
              "argmax_py4dstem": parg.reshape(-1).tolist(),
              "origin_row_py4dstem": arr(own[0]), "origin_column_py4dstem": arr(own[1]),
              "origin_distance_py4dstem_own_vs_f64_px": float(np.nanmax(np.abs(np.asarray(own[0]) - orow)))})
    # a filtered pattern, for the filter claim
    g0 = gauss_f64(c32[0, 0], r)
    check(f"{name}: the written filter equals scipy's on float64", np.max(np.abs(g0 - gaussian_filter(c32[0, 0].astype(np.float64), r, mode="nearest"))) <= 1e-12 * np.max(np.abs(g0)))
    e["filtered_00"] = arr(g0)
    fits = {}
    for model in ("constant", "plane", "parabola", "bezier_two"):
        for robust in (False, True):
            if model == "constant" and robust: continue
            fr = fit_f64(orow, model, robust); fc = fit_f64(ocol, model, robust)
            o = fit_origin((orow, ocol), fitfunction=model, robust=robust)
            fits[f"{model}{'_robust' if robust else ''}"] = {"row": arr(fr), "column": arr(fc),
                "row_py4dstem": arr(o[0]), "distance_py4dstem_px": float(np.max(np.abs(o[0] - fr)))}
    e["fits"] = fits
    # virtual images: centred (mean origin), calibrated (p = 0.025), shifted rounded / bilinear
    qy, qx = c32.shape[2:]
    geos = geometries or [{"kind": "circle", "centerRow": 0.0, "centerColumn": 0.0, "radius": 0.3 * min(qy, qx)},
                          {"kind": "annulus", "centerRow": 0.0, "centerColumn": 0.0, "inner": 0.15 * min(qy, qx), "outer": 0.4 * min(qy, qx)}]
    r0, c0 = float(np.mean(orow)), float(np.mean(ocol))
    dcal = py4dstem_calibrated(c32, orow, ocol)
    imgs = []
    for g in geos:
        mode = "circle" if g["kind"] == "circle" else "annulus"
        geo = ((g["centerRow"], g["centerColumn"]), g["radius"]) if mode == "circle" else ((g["centerRow"], g["centerColumn"]), (g["inner"], g["outer"]))
        m_c = make_mask((qy, qx), centred(g, r0, c0))
        img_c = np.einsum("abij,ij->ab", c32.astype(np.float64), m_c)
        own_c = dcal.get_virtual_image(mode, geo, centered=True, verbose=False).data
        ent = {"geometry": g, "centered": arr(img_c), "centered_py4dstem": arr(own_c)}
        for shift, sub in (("rounded", False), ("bilinear", True)):
            img = shifted_image_f64(c32, m_c, orow, ocol, shift)
            own_s = dcal.get_virtual_image(mode, geo, centered=True, shift_center=True, subpixel=sub, verbose=False).data
            ent[shift] = arr(img); ent[f"{shift}_py4dstem"] = arr(own_s)
            ent[f"{shift}_distance_py4dstem"] = float(np.max(np.abs(own_s - img)) / max(np.max(np.abs(img)), 1e-300))
        # calibrated alone: geometry in A^-1 with p = 0.025
        p = 0.025
        gcal = {k: (v * p if k in ("centerRow", "centerColumn", "radius", "inner", "outer") else v) for k, v in centred(g, r0, c0).items()}
        dcp = py4dstem_calibrated(c32, orow, ocol, p)
        geo_cal = ((gcal["centerRow"], gcal["centerColumn"]), gcal["radius"]) if mode == "circle" else ((gcal["centerRow"], gcal["centerColumn"]), (gcal["inner"], gcal["outer"]))
        own_p = dcp.get_virtual_image(mode, geo_cal, calibrated=True, verbose=False).data
        img_p = np.einsum("abij,ij->ab", c32.astype(np.float64), make_mask((qy, qx), centred(gcal, None, None, p)))
        ent.update({"calibrated_geometry": gcal, "pixel_size": p, "calibrated": arr(img_p), "calibrated_py4dstem": arr(own_p)})
        imgs.append(ent)
    e["images"] = imgs
    # shifted mean and max patterns
    dps = {}
    for shift, sub in (("rounded", False), ("fourier", True)):
        mean, mx = shifted_dp_f64(c32, orow, ocol, shift)
        om = dcal.get_virtual_diffraction("mean", shift_center=True, subpixel=sub, verbose=False).data
        ox = dcal.get_virtual_diffraction("max", shift_center=True, subpixel=sub, verbose=False).data
        dps[shift] = {"mean": arr(mean), "max": arr(mx), "mean_py4dstem": arr(om), "max_py4dstem": arr(ox),
                      "mean_distance_py4dstem": float(np.max(np.abs(om - mean)) / np.max(np.abs(mean)))}
    e["patterns"] = dps
    return e, orow, ocol


def planted(dtype):
    R, C, qy, qx = 5, 6, 24, 25
    rng = np.random.default_rng(20261015)
    ii, jj = np.indices((qy, qx))
    cube = np.zeros((R, C, qy, qx)); tr = np.zeros((R, C)); tc = np.zeros((R, C))
    for a in range(R):
        for b in range(C):
            ci, cj = 11 + int(rng.integers(-3, 4)), 12 + int(rng.integers(-3, 4))
            cube[a, b] = np.where((ii - ci) ** 2 + (jj - cj) ** 2 < 16, 1000.0, 5.0)
            tr[a, b], tc[a, b] = ci, cj
    return cube.astype(dtype), tr, tc


def run_synthetic():
    out = {"versions": versions(), "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cubes": []}
    k, R = kernel(2.6)
    check("kernel sums to 1 and is symmetric", abs(k.sum() - 1) < 1e-15 and np.all(k == k[::-1]))
    for dtype, nm in ((np.float32, "planted_f32"), (np.uint16, "planted_u16")):
        cube, tr, tc = planted(dtype)
        np.save(os.path.join(FIX, nm + ".npy"), cube)
        e, orow, ocol = pin_cube(nm, cube)
        check(f"{nm}: origins recover the planted centres", np.max(np.abs(orow - tr)) < 1e-9 and np.max(np.abs(ocol - tc)) < 1e-9)
        mean, _ = shifted_dp_f64(cube.astype(np.float32), orow, ocol, "rounded")
        ref = cube[0, 0].astype(np.float64)
        ref = np.roll(ref, (int(np.round(np.mean(orow) - orow[0, 0])), int(np.round(np.mean(ocol) - ocol[0, 0]))), (0, 1))
        check(f"{nm}: the rounded shifted mean is the unshifted disk", np.array_equal(mean, ref))
        e["truth_row"] = arr(tr); e["truth_column"] = arr(tc)
        out["cubes"].append(e)
    for nm in ("gaussian", "counts16"):
        cube = np.load(os.path.join(FIX, nm + ".npy"))
        e, _, _ = pin_cube(nm, cube)
        out["cubes"].append(e)
    # fits truth: exact surfaces and one outlier
    r, c = np.indices((9, 11)).astype(np.float64)
    surf = {"plane": 0.3 + 0.02 * r - 0.07 * c, "parabola": 1 + 0.1 * r - 0.2 * c + 0.03 * r * r - 0.01 * c * c + 0.005 * r * c,
            "bezier_two": 2 + 0.1 * r * c * c - 0.02 * r * r * c + 0.001 * r * r * c * c}
    truth = {}
    for model, s in surf.items():
        check(f"fit {model} recovers its surface", np.max(np.abs(fit_f64(s, model) - s)) < 1e-10)
        o = s.copy(); o[4, 5] += 50
        check(f"robust {model} removes the outlier", np.max(np.abs(fit_f64(o, model, robust=True) - s)) < 1e-9)
        truth[model] = {"shape": [9, 11], "surface": arr(s), "with_outlier": arr(o)}
    out["fit_truth"] = truth
    json.dump(out, open(os.path.join(FIX, "slice-3-pins.json"), "w"))
    print(f"pins.py slice 3 synthetic: {len(out['cubes'])} cubes, {len(CHECKS)} self-checks passed")
    for e in out["cubes"]:
        ad = sum(a != b for a, b in zip(e["argmax"], e["argmax_py4dstem"]))
        print(f"  {e['cube']}: r {float(e['radius']):.4f}, origin own-vs-f64 {e['origin_distance_py4dstem_own_vs_f64_px']:.2e} px, argmax differ {ad}, "
              f"fits own-vs-f64 " + ", ".join(f"{k} {v['distance_py4dstem_px']:.1e}" for k, v in e["fits"].items()) +
              f"; bilinear own {e['images'][0]['bilinear_distance_py4dstem']:.1e}; fourier mean own {e['patterns']['fourier']['mean_distance_py4dstem']:.1e}")


def run_real(names):
    for name in names:
        t0 = datetime.datetime.now()
        cube = np.load(os.path.join(REAL, f"{name}.npy"))
        e, orow, ocol = pin_cube(name, cube)
        json.dump(e, open(os.path.join(REAL, f"{name}-s3-pins.json"), "w"))
        print(f"pins.py slice 3 real {name}: r {float(e['radius']):.4f}, origin own-vs-f64 {e['origin_distance_py4dstem_own_vs_f64_px']:.2e} px, "
              f"argmax differ {sum(a != b for a, b in zip(e['argmax'], e['argmax_py4dstem']))}, "
              f"bilinear own {e['images'][0]['bilinear_distance_py4dstem']:.1e}, fourier mean own {e['patterns']['fourier']['mean_distance_py4dstem']:.1e}; "
              f"{(datetime.datetime.now() - t0).total_seconds():.0f} s")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("synthetic", "real"):
        raise SystemExit(__doc__)
    run_synthetic() if sys.argv[1] == "synthetic" else run_real(sys.argv[2:] or ["simau", "particle", "py4dstem_small", "auref", "almgsi"])
