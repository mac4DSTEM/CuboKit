"""Pins for CuboCore FrameTransform. py4DSTEM (f050d207) moves no origin on crop/bin, so the primary pin for those is the
app source CalibrationReReference.swift at mac4DSTEM c809679c, transcribed here in Float32 (never from the Swift port).
py4DSTEM contributes the crop slice and the Q-pixel-size x bin rule, asserted from its source text.
Run: ../../tools/memcap.sh -- ../../References/python/py4dstem/bin/python -I -B tools/rereference/pins.py"""
import json, pathlib, numpy as np
f = np.float32
ref = pathlib.Path('../References/CuboKit/py4DSTEM-f050d207/py4DSTEM/preprocess/preprocess.py')
if not ref.exists(): ref = pathlib.Path('../../References/CuboKit/py4DSTEM-f050d207/py4DSTEM/preprocess/preprocess.py')
src = ref.read_text()
assert 'get_Q_pixel_size() * bin_factor' in src and 'datacube.data[\n        :, :, crop_Qx_min:crop_Qx_max' in src
def binned(v, b): return f(v) if b <= 1 else f((f(v) + f(0.5)) / f(b) - f(0.5))
def source(v, b): return f(v) if b <= 1 else f((f(v) + f(0.5)) * f(b) - f(0.5))
def inside(c, r, w, h):
    return bool(np.isfinite(c) and np.isfinite(r) and c >= f(-0.5) and c < f(w) - f(0.5) and r >= f(-0.5) and r < f(h) - f(0.5))
rng = np.random.default_rng(7)
out = {"app_source": "mac4DSTEM c809679c Core/Data/CalibrationReReference.swift", "numpy": np.__version__}
out["coords"] = [{"v": float(v), "bin": b, "binned": float(binned(v, b)), "source": float(source(v, b))}
                 for v in [0, 3.25, 31.5, -0.5, 100.75] for b in [1, 2, 3, 4]]
pos = []
for c, r, co, ro, b, w, h in [(130, 126, 64, 64, 1, 128, 128), (70, 70, 64, 64, 2, 64, 64), (64, 64, 64, 64, 4, 32, 32),
                              (10, 10, 64, 64, 1, 128, 128), (float('nan'), 5, 0, 0, 1, 8, 8), (0, 0, 0, 0, 2, 8, 8), (-0.5, -0.5, 0, 0, 1, 8, 8), (7.5, 7.49, 0, 0, 1, 8, 8), (7.5, 3, 0, 0, 1, 8, 8), (-0.51, 3, 0, 0, 1, 8, 8)]:
    mc, mr = binned(f(c) - f(co), b), binned(f(r) - f(ro), b)
    pos.append({"c": c if c == c else None, "r": r, "co": co, "ro": ro, "bin": b, "w": w, "h": h,
                "inside": inside(mc, mr, w, h), "mc": float(mc) if np.isfinite(mc) else None, "mr": float(mr)})
out["positions"] = pos
W, H = 6, 5
m = rng.uniform(0, 100, W * H).astype(f)
crop = {"w": W, "h": H, "co": 1, "ro": 2, "cw": 4, "ch": 3}
sel = m.reshape(H, W)[2:5, 1:5].ravel()
out["crop"] = {**crop, "values": m.tolist(), "expected": sel.tolist()}
cols = (rng.uniform(10, 20, 12)).astype(f); rows = (rng.uniform(10, 20, 12)).astype(f)
for k in ("a", "b"):
    co, ro, b, w, h = (8, 8, 2, 8, 8) if k == "a" else (8, 8, 1, 8, 8)
    mc = [float(binned(f(x) - f(co), b)) for x in cols]; mr = [float(binned(f(x) - f(ro), b)) for x in rows]
    first = next((i for i in range(12) if not inside(f(mc[i]), f(mr[i]), w, h)), None)
    out["map_" + k] = {"cols": cols.tolist(), "rows": rows.tolist(), "co": co, "ro": ro, "bin": b, "w": w, "h": h,
                       "mc": mc, "mr": mr, "first": first}
pathlib.Path('Tests/CuboCoreTests/Fixtures/frametransform.json').write_text(json.dumps(out, indent=1))
