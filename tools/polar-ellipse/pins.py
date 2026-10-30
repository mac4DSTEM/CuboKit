"""Pinned py4DSTEM polar data cube, radial background, ellipse fits and Fourier resampling (CuboKit 0.9.0), run on
small cases built from the planted truth of tools/polar-ellipse/planted.py (its `pattern`, `ring_points`, `band_limited`).
Calls py4DSTEM itself: PolarDatacube, DataCube.get_radial_bkgrnd / get_radial_bksb_dp / get_local_ave_dp(radial_bksb=True),
fit_ellipse_1D, fit_ellipse_amorphous_ring, constrain_degenerate_ellipse, ellipse_err, double_sided_gaussian(_fiterr),
convert_ellipse_params(_r), fourier_resample and resample_data_diffraction(method="fourier").
Run from the clone root with the environment of tools/requirements.txt (tools/pinenv.py finds the inputs):
  python -I -B tools/polar-ellipse/pins.py
Output: Tests/CuboVirtualTests/Fixtures/polar-radial-pins.json and Tests/CuboCoreTests/Fixtures/ellipse-resample-pins.json
Conventions: py4DSTEM's x is the kit's ROW and y its COLUMN, so the (x0, y0, a, b, theta) here are the kit's (row, column, a, b, theta)
with theta of the major axis from the +row axis towards the +column axis. planted.py measures theta of the major axis from +column
towards +row: theta_py4DSTEM = pi/2 - theta_planted (mod pi). NaN is written as null. Floats are repr (shortest round-trip).
"""
from pathlib import Path
import importlib.util
import json
import platform
import sys

import numpy as np
import scipy
import py4DSTEM
from py4DSTEM import DataCube
from py4DSTEM.process.polar import PolarDatacube
from py4DSTEM.process.calibration.ellipse import (fit_ellipse_1D, fit_ellipse_amorphous_ring, constrain_degenerate_ellipse,
                                                  ellipse_err, double_sided_gaussian, double_sided_gaussian_fiterr)
from py4DSTEM.process.utils import convert_ellipse_params, convert_ellipse_params_r, fourier_resample
from py4DSTEM.preprocess.preprocess import resample_data_diffraction

assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('planted', HERE / 'planted.py')
planted = importlib.util.module_from_spec(spec); spec.loader.exec_module(planted)
rng = np.random.default_rng(20261022)


def flat(a):
    return np.asarray(a, dtype=np.float64).ravel().tolist()


def nulls(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=np.float64).ravel()]


def py_theta(theta_planted):
    return float((np.pi / 2 - theta_planted + np.pi / 2) % np.pi - np.pi / 2)


# ---------------------------------------------------------------- polar cube and radial background
SCAN, DET = (3, 3), (40, 44)
TRUTH = dict(r0=19.6, c0=22.4, a=13.0, b=9.5, theta=0.6)       # planted convention (theta from +column towards +row)
ELL = (TRUTH['a'], TRUTH['b'], py_theta(TRUTH['theta']))       # py4DSTEM (a, b, theta)
rr_, cc_ = np.meshgrid(np.arange(DET[0], dtype=np.float64), np.arange(DET[1], dtype=np.float64), indexing='ij')
base = planted.pattern(rr_, cc_, TRUTH)
scale = 1 + 0.1 * np.arange(SCAN[0] * SCAN[1]).reshape(SCAN)
cube = (scale[:, :, None, None] * base[None, None] + rng.gamma(2.0, 1.5, SCAN + DET)).astype(np.float32)
o_row = TRUTH['r0'] + rng.normal(0, 0.15, SCAN)
o_col = TRUTH['c0'] + rng.normal(0, 0.15, SCAN)


def make_cube(ellipse):
    dc = DataCube(data=cube.copy())
    dc.calibration.set_Q_pixel_size(0.02)
    dc.calibration.set_origin((o_row, o_col))
    if ellipse is not None:
        dc.calibration.set_ellipse(ellipse)
    return dc


mask = np.ones(DET, dtype=bool); mask[:4, :] = False; mask[18:23, 20:26] = False; mask[30, 7] = False
polar_cases = []
for name, ell, kw, positions in [
    ('default', ELL, dict(n_annular=36), [(0, 0), (1, 2), (2, 1), (1, 1)]),
    ('windowed_qscale', ELL, dict(qmin=2.0, qmax=18.0, qstep=0.75, n_annular=48, qscale=2.0), [(0, 0), (2, 2)]),
    ('circular_flag_two_fold', ELL, dict(n_annular=30, ellipse=False, two_fold_symmetry=True), [(1, 1)]),
    ('masked', ELL, dict(n_annular=36, qstep=1.5, mask=mask, mask_thresh=0.3), [(0, 1), (2, 0)]),
    ('no_ellipse_calibration', None, dict(n_annular=24, qstep=2.0), [(1, 0)]),
]:
    dc = make_cube(ell)
    pc = PolarDatacube(dc, **kw)
    rec = dict(name=name, ellipse=None if ell is None else list(ell), params={k: (v if not isinstance(v, np.ndarray) else 'mask') for k, v in kw.items()},
               polarShape=[int(v) for v in pc.polar_shape], radialBins=flat(pc.radial_bins), annularBins=flat(pc.annular_bins),
               sigmaKDE=flat(pc._sigma_KDE), patterns=[])
    for (rx, ry) in positions:
        ans = pc.data[rx, ry]
        full, norm, norm_scaled, mbool = pc.transform(cube[rx, ry], origin=[rx, ry], returnval='all')
        rec['patterns'].append(dict(row=rx, column=ry, values=nulls(ans.filled(np.nan)), mask=[int(v) for v in np.ma.getmaskarray(ans).ravel()],
                                    norm=flat(norm)))
    polar_cases.append(rec)

# radial background, bksb pattern, local average (the default cube/polar; sigma 2, 0, 0.7)
dc = make_cube(ELL)
pc = PolarDatacube(dc, n_annular=36)
radial = dict(sigmaCases=[], localAverage=[])
for (rx, ry, sigma) in [(0, 0, 2), (1, 2, 2), (2, 1, 0), (1, 1, 0.7)]:
    bg = dc.get_radial_bkgrnd(rx, ry, sigma)
    bk = dc.get_radial_bksb_dp(rx, ry, sigma)
    radial['sigmaCases'].append(dict(row=rx, column=ry, sigma=sigma, background=flat(bg), bksb=flat(bk)))
for (rx, ry, sigma) in [(0, 0, 2), (1, 1, 2), (2, 1, 1.0), (1, 2, 2)]:
    radial['localAverage'].append(dict(row=rx, column=ry, sigma=sigma, values=flat(dc.get_local_ave_dp(rx, ry, radial_bksb=True, sigma=sigma))))
# a cube with no ellipse: get_radial_bkgrnd falls back to a circle (1, 1, 0)
dc0 = make_cube(None); PolarDatacube(dc0, n_annular=24, qstep=2.0)
radial['noEllipse'] = dict(row=1, column=0, sigma=2, params=dict(n_annular=24, qstep=2.0), background=flat(dc0.get_radial_bkgrnd(1, 0, 2)))

polar_doc = dict(
    versions=dict(py4dstem=py4DSTEM.__version__, py4dstem_commit='f050d207851f8accd3aa550365baaf1f38230428', numpy=np.__version__,
                  scipy=scipy.__version__, python=platform.python_version()),
    seed=20261022, scan=list(SCAN), detector=list(DET), cube=flat(cube), originRow=flat(o_row), originColumn=flat(o_col),
    qPixelSize=0.02, truthPlanted=TRUTH, ellipsePy4DSTEM=list(ELL), mask=[int(v) for v in mask.ravel()],
    polar=polar_cases, radial=radial)
(KIT / 'Tests/CuboVirtualTests/Fixtures/polar-radial-pins.json').write_text(json.dumps(polar_doc))

# ---------------------------------------------------------------- ellipse fits
NF = (48, 52)
FIT_TRUTH = {
    'tilted': dict(r0=23.4, c0=26.7, a=14.0, b=10.0, theta=0.6),
    'flat': dict(r0=24.2, c0=25.3, a=15.0, b=12.0, theta=0.0),
    'near_circle': dict(r0=22.9, c0=27.1, a=13.0, b=12.4, theta=-1.1),
}
fr_, fc_ = np.meshgrid(np.arange(NF[0], dtype=np.float64), np.arange(NF[1], dtype=np.float64), indexing='ij')


def ring_only(p):
    u, v = planted._uv(fr_, fc_, p)
    rho = np.sqrt((u / p['a']) ** 2 + (v / p['b']) ** 2)
    return 100.0 * np.exp(-((rho - 1.0) ** 2) / (2 * planted.WIDTH ** 2 * 4))     # a ring about 2 px wide


fits = []
for name, p in FIT_TRUTH.items():
    ring = ring_only(p)
    full = planted.pattern(fr_, fc_, p)
    c_guess = (p['r0'] + 0.4, p['c0'] - 0.3)
    radii = (p['b'] * 0.7, p['a'] * 1.3)
    rec = dict(name=name, truthPlanted=p, truthPy4DSTEM=dict(r0=p['r0'], c0=p['c0'], a=p['a'], b=p['b'], theta=py_theta(p['theta'])),
               shape=list(NF), ring=flat(ring), full=flat(full), centerGuess=list(c_guess), fitRadii=list(radii))
    m = np.zeros(NF, dtype=bool); m[:6, :] = True; m[20:26, 30:36] = True
    rec['ignoreMask'] = [int(v) for v in m.ravel()]
    rec['fit1D'] = list(map(float, fit_ellipse_1D(ring, center=c_guess, fitradii=radii)))
    rec['fit1DMasked'] = list(map(float, fit_ellipse_1D(ring, center=c_guess, fitradii=radii, mask=m)))
    # upstream's default fitradii raises TypeError (np.minimum(shape) with one argument): pass the evident intent (0, min(shape) / 2)
    rec['fit1DDefaults'] = list(map(float, fit_ellipse_1D(ring, fitradii=(0, min(NF) / 2))))
    keep = ~m                                                   # amorphous: fit where mask is True
    (x0, y0, a, b, th), pfull = fit_ellipse_amorphous_ring(full, c_guess, (p['b'] * 0.75, p['a'] * 1.25))
    rec['amorphous'] = dict(ellipse=[float(x0), float(y0), float(a), float(b), float(th)], p=list(map(float, pfull)))
    (x0, y0, a, b, th), pfull = fit_ellipse_amorphous_ring(full, c_guess, (p['b'] * 0.75, p['a'] * 1.25), mask=keep)
    rec['amorphousMasked'] = dict(ellipse=[float(x0), float(y0), float(a), float(b), float(th)], p=list(map(float, pfull)))
    rec['keepMask'] = [int(v) for v in keep.ravel()]
    p0 = [None] * 11; p0[3] = 2.5; p0[4] = 3.5
    (x0, y0, a, b, th), pfull = fit_ellipse_amorphous_ring(full, c_guess, (p['b'] * 0.75, p['a'] * 1.25), p0=tuple(p0))
    rec['amorphousP0'] = dict(p0=p0, ellipse=[float(x0), float(y0), float(a), float(b), float(th)], p=list(map(float, pfull)))
    fits.append(rec)

# direct function values
pts = rng.uniform(5, 40, (7, 2)); vals = rng.uniform(0.5, 3, 7)
pe = [23.4, 26.7, 0.0052, 0.0013, 0.0071]
gp = [90.0, 40.0, 3.0, 1.5, 2.5, 2.0, 23.4, 26.7, 0.0049, 0.0011, 0.0063]
funcs = dict(points=flat(pts), values=flat(vals), pEllipse=pe, pGaussian=gp,
             ellipseErr=flat(ellipse_err(pe, pts[:, 0], pts[:, 1], vals)),
             doubleSidedGaussian=flat(double_sided_gaussian(gp, pts[:, 0], pts[:, 1])),
             doubleSidedGaussianFitErr=flat(double_sided_gaussian_fiterr(gp, pts[:, 0], pts[:, 1], vals)),
             convert=[dict(abc=list(abc), aba=list(map(float, convert_ellipse_params(*abc)))) for abc in
                      [(0.0052, 0.0013, 0.0071), (0.01, 0.0, 0.02), (0.02, 0.0, 0.01), (0.01, 0.0, 0.01), (0.004, -0.002, 0.006)]],
             convertR=[dict(aba=list(abc), abc=list(map(float, convert_ellipse_params_r(*abc)))) for abc in
                       [(14.0, 10.0, 0.9), (12.0, 12.0, 0.0), (9.0, 7.0, -1.2), (15.0, 12.0, 1.5707963267948966)]])

# constrain_degenerate_ellipse: four Gaussian spots on a tilted ellipse
CD = (96, 104)
cr_, cc2_ = np.meshgrid(np.arange(CD[0], dtype=np.float64), np.arange(CD[1], dtype=np.float64), indexing='ij')
x0, y0, a_t, b_t, th = 47.6, 52.3, 22.0, 17.0, 0.35
phi_known = 1.2
spots = []
for ang in (0.45, np.pi - 0.45, np.pi + 0.45, -0.45):
    ex, ey = a_t * np.cos(ang), b_t * np.sin(ang)
    spots.append((x0 + ex * np.cos(th) - ey * np.sin(th), y0 + ex * np.sin(th) + ey * np.cos(th)))
bvm = np.zeros(CD)
for (sx, sy), amp in zip(spots, (100.0, 90.0, 95.0, 85.0)):
    bvm += amp * np.exp(-((cr_ - sx) ** 2 + (cc2_ - sy) ** 2) / (2 * 1.6 ** 2))
bvm += rng.uniform(0, 0.4, CD)
cons = []
for p_ell, ri, ro, phi, fitrad in [((x0, y0, 24.0, 15.0, th), 14.0, 34.0, phi_known, 6),
                                   ((x0 + 0.3, y0 - 0.2, 21.0, 18.5, th + 0.05), 12.0, 36.0, 1.0, 5)]:
    ac, bc = constrain_degenerate_ellipse(bvm, p_ell, ri, ro, phi, fitrad=fitrad)
    cons.append(dict(pEllipse=list(p_ell), rInner=ri, rOuter=ro, phiKnown=phi, fitRad=fitrad, aConstrained=float(ac), bConstrained=float(bc)))
constrain = dict(shape=list(CD), data=flat(bvm), cases=cons)

# ---------------------------------------------------------------- fourier resample
FR_SHAPE = (2, 3, 10, 12)
fcube = (rng.gamma(2.0, 40.0, FR_SHAPE) - 5).astype(np.float32)
fcases = []
for name, kw in [('scale2', dict(scale=2)), ('scaleHalf', dict(scale=0.5)), ('scale1p5', dict(scale=1.5)),
                 ('size15x9', dict(output_size=(15, 9))), ('size7x13', dict(output_size=(7, 13))), ('size10x20', dict(output_size=(10, 20))),
                 ('size12x12', dict(output_size=(12, 12))), ('size10x12', dict(output_size=(10, 12))),
                 ('conserve', dict(scale=2, conserve_array_sums=True)),
                 ('bandlimit', dict(output_size=(20, 24), bandlimit_nyquist=0.35, bandlimit_power=3)),
                 ('bandlimitDefaultPower', dict(output_size=(8, 6), bandlimit_nyquist=0.3)),
                 ('nonnegative', dict(output_size=(16, 18), force_nonnegative=True))]:
    out = fourier_resample(fcube, **kw)
    fcases.append(dict(name=name, args={k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}, shape=list(out.shape),
                       dtype=str(out.dtype), values=flat(out)))
odd = (rng.gamma(2.0, 40.0, (1, 2, 9, 11)) - 5).astype(np.float32)
oddcases = []
for name, kw in [('odd_to_even', dict(output_size=(12, 14))), ('odd_down', dict(output_size=(5, 7))), ('odd_scale2', dict(scale=2))]:
    out = fourier_resample(odd, **kw)
    oddcases.append(dict(name=name, args={k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}, shape=list(out.shape),
                         values=flat(out)))
two_d = rng.uniform(0, 5, (8, 6)).astype(np.float32)
two_d_out = fourier_resample(two_d, output_size=(12, 9))
# resample_data_diffraction(method="fourier"): pixel size
dcf = DataCube(data=fcube.copy()); dcf.calibration.set_Q_pixel_size(0.05)
dcf = resample_data_diffraction(dcf, resampling_factor=2, method='fourier')
dcg = DataCube(data=fcube.copy()); dcg.calibration.set_Q_pixel_size(0.05)
dcg = resample_data_diffraction(dcg, output_size=(15, 9), method='fourier')
rdd = dict(factor2=dict(qPixelSize=float(dcf.calibration.get_Q_pixel_size()), shape=list(dcf.data.shape)),
           size15x9=dict(qPixelSize=float(dcg.calibration.get_Q_pixel_size()), shape=list(dcg.data.shape), note='pixel size / (15 / 10)'))
resample = dict(inputShape=list(FR_SHAPE), input=flat(fcube), cases=fcases, oddInputShape=[1, 2, 9, 11], oddInput=flat(odd), odd=oddcases,
                twoD=dict(input=flat(two_d), shape=[8, 6], outputShape=[12, 9], values=flat(two_d_out)), resampleDataDiffraction=rdd)

ellipse_doc = dict(
    versions=polar_doc['versions'], seed=20261022, fitImageShape=list(NF), fits=fits, functions=funcs, constrain=constrain,
    resample=resample)
(KIT / 'Tests/CuboCoreTests/Fixtures/ellipse-resample-pins.json').write_text(json.dumps(ellipse_doc))
print('ok', {r['name']: r['fit1D'] for r in fits})
print({r['name']: r['amorphous']['ellipse'] for r in fits})
print({r['name']: r['truthPy4DSTEM'] for r in fits})
print(cons)
