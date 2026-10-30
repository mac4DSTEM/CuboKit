"""Pins for OriginFit.fitTrimmed: the MAD-trimmed origin fit of mac4DSTEM's calibration (OriginCalibration.fitOriginTrimmed).
Run from the clone root:
  MPLCONFIGDIR="$PWD/.build/matplotlib" python -I -B tools/origin/trimmed_pins.py
Two references, both recorded in the fixture:
 * `app`: a line-by-line NumPy transcription of the app's Swift source (mac4DSTEM origin/main, Core/Analysis/OriginCalibration.swift,
   `fitOriginTrimmed` and `fit2D`), in Float32 where the app is Float (residuals, median, MAD, cutoff, RMS), the least-squares surface in
   Double as the app calls the kit's OriginFit. This is the reference for the mask, the surfaces and the residuals.
 * `py4dstem`: py4DSTEM's own `fit_origin((row, column), mask=kept, fitfunction=model)` on the mask the transcription ends with, where
   py4DSTEM has an equivalent (a masked plane / parabola / bezier_two fit; its constant fit ignores the mask).
Planted outliers keep every residual away from the cutoff (asserted), so the Float32 comparisons cannot flip a mask.
Output: Tests/CuboCoreTests/Fixtures/origin-trimmed.json
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM.process.calibration import fit_origin
KIT = Path(__file__).resolve().parents[2]
import sys as _sys, pathlib as _pathlib; _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # tools/pinenv.py
from pinenv import PY4DSTEM_SRC, MAC4DSTEM_REPO
REF = PY4DSTEM_SRC
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'], text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
f32 = np.float32
APP = subprocess.check_output(['git','-C',str(MAC4DSTEM_REPO),'--no-optional-locks','rev-parse','--short','origin/main'], text=True).strip()
TERMS = {'plane': 3, 'parabola': 6, 'bezierTwo': 9}

def design(rows, cols, model):
    xs, ys = np.linspace(0, 1, rows), np.linspace(0, 1, cols)
    X, Y = np.meshgrid(xs, ys, indexing='ij'); X = X.reshape(-1); Y = Y.reshape(-1)
    if model == 'plane': return np.stack([X, Y, np.ones_like(X)], 1)
    if model == 'parabola': return np.stack([np.ones_like(X), X, Y, X*X, Y*Y, X*Y], 1)
    bx = [(1-X)**2, 2*(1-X)*X, X**2]; by = [(1-Y)**2, 2*(1-Y)*Y, Y**2]
    return np.stack([bx[i]*by[j] for i in range(3) for j in range(3)], 1)

def fit2d(z, rows, cols, model, included):
    """The app's fit2D: surface of the included positions (Float32 in, Double fit, Float32 out)."""
    z = np.asarray(z, f32)
    used = np.flatnonzero(included)
    source = used if len(used) else np.arange(len(z))
    mean_surface = np.full(len(z), f32(np.mean(z[source].astype(np.float64))), f32)
    if model == 'constant': return mean_surface
    if not (cols > 1 and rows > 1 and len(used) >= TERMS[model]): return mean_surface
    A = design(rows, cols, model)
    coef, *_ = np.linalg.lstsq(A[used], z[used].astype(np.float64), rcond=None)
    return (A @ coef).astype(f32)

def median32(v):
    if len(v) == 0: return f32('nan')
    s = np.sort(np.asarray(v, f32)); m = len(s)//2
    return f32(s[m]) if len(s) % 2 else f32((s[m-1] + s[m]) / f32(2))

def trimmed(ma, mb, rows, cols, model, sigma=3.0, rounds=3):
    """fitOriginTrimmed, step for step, in Float32. Returns the fit, the mask and the closest residual to a cutoff (relative)."""
    ma = np.asarray(ma, f32); mb = np.asarray(mb, f32); n = len(ma)
    kept = np.ones(n, bool); fa, fb = ma.copy(), mb.copy()
    sigma = f32(sigma); closest = np.inf
    def resid(a, b):
        dx = (ma - a).astype(f32); dy = (mb - b).astype(f32)
        return np.sqrt((dx*dx + dy*dy).astype(f32)).astype(f32)
    for _ in range(max(rounds, 1)):
        mask = kept.copy()
        fa = fit2d(ma, rows, cols, model, mask); fb = fit2d(mb, rows, cols, model, mask)
        r = resid(fa, fb)
        centre = median32(r)
        dev = f32(f32(1.4826) * median32(np.abs((r - centre).astype(f32))))
        if not (np.isfinite(dev) and dev > 0): break
        cutoff = f32(centre + sigma * dev)
        closest = min(closest, float(np.min(np.abs(r - cutoff) / cutoff)))
        nxt = r <= cutoff
        if not nxt.any(): break
        if np.array_equal(nxt, kept): kept = nxt; break
        kept = nxt
    if kept.any():
        fa = fit2d(ma, rows, cols, model, kept); fb = fit2d(mb, rows, cols, model, kept)
    r = resid(fa, fb)
    def rms(idx):
        if len(idx) == 0: return f32('nan')
        total = f32(0)
        for i in idx: total = f32(total + r[i]*r[i])
        return f32(np.sqrt(f32(total / f32(len(idx)))))
    return fa, fb, kept, rms(np.flatnonzero(kept)), rms(np.arange(n)), closest

def flat(a): return [float('%.9g' % v) for v in np.asarray(a, np.float64).ravel()]

rng = np.random.default_rng(20261023)
R, C = 12, 10
r_, c_ = np.indices((R, C)); x, y = r_/(R-1), c_/(C-1)
def surface(model):
    if model == 'plane': return 31.0 + 4.0*x - 2.5*y, 40.5 - 1.5*x + 3.0*y
    if model == 'parabola': return 31.0 + 4.0*x - 2.5*y + 3*x*x - 2*x*y, 40.5 - 1.5*x + 3.0*y + 2.5*y*y + x*y
    if model == 'bezierTwo':
        b = lambda t: [(1-t)**2, 2*(1-t)*t, t**2]
        gx, gy = b(x), b(y)
        wa = rng.normal(0, 3, 9); wb = rng.normal(0, 3, 9)
        return (31 + sum(wa[3*i+j]*gx[i]*gy[j] for i in range(3) for j in range(3)), 40 + sum(wb[3*i+j]*gx[i]*gy[j] for i in range(3) for j in range(3)))
    return 31.0 + 0*x, 40.5 + 0*y

def contaminate(a, b, fraction, size, clustered=False):
    a = a.copy(); b = b.copy(); n = a.size
    k = int(round(fraction*n))
    idx = np.arange(k) if clustered else rng.choice(n, k, replace=False)
    ang = rng.uniform(0, 2*np.pi, k)
    a.reshape(-1)[idx] += size*np.cos(ang); b.reshape(-1)[idx] += size*np.sin(ang)
    return a, b

cases = []
def add(name, model, a, b, sigma=3.0, rounds=3, compare_py4dstem=True):
    a = np.asarray(a, f32).astype(np.float64); b = np.asarray(b, f32).astype(np.float64)       # the app's maps are Float32: the kit gets the same numbers
    fa, fb, kept, kres, fres, closest = trimmed(a.reshape(-1), b.reshape(-1), R, C, model, sigma, rounds)
    assert closest > 1e-4, (name, closest)                      # no residual within 1e-4 of a cutoff: a Float32 flip is impossible
    d = dict(name=name, model=model, trimSigma=sigma, rounds=rounds, originRow=flat(a), originColumn=flat(b),
             app=dict(fittedRow=flat(fa), fittedColumn=flat(fb), kept=[bool(v) for v in kept], keptResidual=float(kres),
                      fullScanResidual=float(fres), closest=closest))
    if compare_py4dstem and model != 'constant' and kept.sum() >= TERMS[model]:
        qa, qb, _, _ = fit_origin((np.asarray(a, np.float64).copy(), np.asarray(b, np.float64).copy()), mask=kept.reshape(R, C), fitfunction={'bezierTwo': 'bezier_two'}.get(model, model))
        d['py4dstem'] = dict(fittedRow=flat(qa), fittedColumn=flat(qb))
    cases.append(d)
    print(f'{name:28s} kept {int(kept.sum()):3d}/{kept.size}  kept RMS {float(kres):.4g}  full RMS {float(fres):.4g}  closest {closest:.3g}')

for model in ('plane', 'parabola', 'bezierTwo'):
    sa, sb = surface(model)
    noisy = lambda s: s + rng.normal(0, 0.05, s.shape)
    a, b = contaminate(noisy(sa), noisy(sb), 0.15, 15.0)
    add(f'{model}Scattered15', model, a, b)
    a, b = contaminate(noisy(sa), noisy(sb), 0.30, 25.0)
    add(f'{model}Scattered30', model, a, b)
sa, sb = surface('plane')
add('planeNoOutliers', 'plane', sa + rng.normal(0, 0.3, sa.shape), sb + rng.normal(0, 0.3, sb.shape))
a, b = contaminate(sa + rng.normal(0, 0.05, sa.shape), sb + rng.normal(0, 0.05, sb.shape), 0.2, 12.0)
add('planeOneRound', 'plane', a, b, rounds=1)
add('planeSigma2', 'plane', a, b, sigma=2.0)
add('planeZeroRounds', 'plane', a, b, rounds=0)            # the app runs at least one round
a, b = contaminate(sa + rng.normal(0, 0.05, sa.shape), sb + rng.normal(0, 0.05, sb.shape), 0.25, 40.0, clustered=True)
add('planeClustered', 'plane', a, b)                       # the documented failure: clustered failures are kept
ca, cb = surface('constant')
a, b = contaminate(ca + rng.normal(0, 0.05, ca.shape), cb + rng.normal(0, 0.05, cb.shape), 0.2, 20.0)
add('constantScattered', 'constant', a, b)
# a flat truth with no noise and three failures: after the first trim the inliers sit exactly on the surface, the MAD is 0 and the loop stops
a = np.full((R, C), 31.0); b = np.full((R, C), 40.5)
a.reshape(-1)[[3, 17, 60]] += 9.0
add('planeExactInliers', 'plane', a, b)
# a scan too narrow for a plane (one row) falls back to the mean of the kept positions
n1 = 9
a1 = 31 + rng.normal(0, 0.05, n1); b1 = 40 + rng.normal(0, 0.05, n1); a1[2] += 18.0
R, C = 1, n1
add('oneRowScan', 'plane', a1.reshape(1, n1), b1.reshape(1, n1), compare_py4dstem=False)
out = dict(reference=f'app transcription (mac4DSTEM origin/main {APP}) and py4DSTEM f050d207 / 0.14.19', cases=cases,
           shapes={c['name']: ([1, n1] if c['name'] == 'oneRowScan' else [12, 10]) for c in cases})
dest = KIT/'Tests/CuboCoreTests/Fixtures/origin-trimmed.json'
dest.write_text(json.dumps(out, separators=(',', ':'))+'\n')
print('fixture bytes', dest.stat().st_size)
