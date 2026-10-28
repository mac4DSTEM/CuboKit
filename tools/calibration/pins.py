"""Pinned py4DSTEM calibration model and cube reshaping: DataCube.calibrate (dim vectors, meshgrids, qxxs),
Calibration setters, and preprocess.set_scan_shape, swap_RQ, swap_Rxy, swap_Qxy, bin_data_real, thin_data_real,
pad_data_diffraction, resample_data_diffraction (bilinear), median_filter_masked_pixels on a Float32 DataCube.
Run from the folder that holds this package and References/python/py4dstem (the pinned venv; see the README):
  References/python/py4dstem/bin/python -I -B CuboKit/tools/calibration/pins.py
Output: CuboKit/Tests/CuboCoreTests/Fixtures/calibration.json
"""
from pathlib import Path
import json
import numpy as np
import py4DSTEM
from py4DSTEM import DataCube
from py4DSTEM.preprocess.preprocess import (set_scan_shape, swap_RQ, swap_Rxy, swap_Qxy, bin_data_real, thin_data_real,
                                            pad_data_diffraction, resample_data_diffraction, median_filter_masked_pixels,
                                            bin_data_diffraction)
KIT = Path(__file__).resolve().parents[2]
assert py4DSTEM.__version__ == '0.14.19'
rng = np.random.default_rng(20261021)
shape = (4, 6, 7, 9)
floats = (rng.gamma(2.0, 40.0, shape) - 5).astype(np.float32)

def cube():
    return DataCube(data=floats.copy())
def flat(a):
    return np.asarray(a, dtype=np.float64).ravel().tolist()
def rec(name, dc, **extra):
    d = np.asarray(dc.data)
    return dict(name=name, shape=list(d.shape), values=flat(d), **extra)

ops = []
ops.append(rec('setScanShape', set_scan_shape(cube(), 6, 4), args=[6, 4]))
ops.append(rec('setScanShapeFlat', set_scan_shape(cube(), 24, 1), args=[24, 1]))
ops.append(rec('swapRQ', swap_RQ(cube())))
ops.append(rec('swapRxy', swap_Rxy(cube())))
ops.append(rec('swapQxy', swap_Qxy(cube())))
for b in (2, 3, 4):
    dc = cube(); dc.calibration.set_R_pixel_size(1.5); dc = bin_data_real(dc, b)
    ops.append(rec(f'binR{b}', dc, args=[b], rPixelSize=dc.calibration.get_R_pixel_size()))
for t in (2, 3, 4):
    dc = cube(); dc.calibration.set_R_pixel_size(1.5); dc = thin_data_real(dc, t)
    ops.append(rec(f'thinR{t}', dc, args=[t], rPixelSize=dc.calibration.get_R_pixel_size()))
ops.append(rec('padFactor2', pad_data_diffraction(cube(), pad_factor=2), args=['factor', 2, 2]))
ops.append(rec('padFactor1p5x2', pad_data_diffraction(cube(), pad_factor=(1.5, 2)), args=['factor', 1.5, 2]))
ops.append(rec('padSize', pad_data_diffraction(cube(), output_size=(12, 14)), args=['size', 12, 14]))
for name, kw in [('resampleHalf', dict(resampling_factor=0.5)), ('resampleDouble', dict(resampling_factor=2)),
                 ('resample1p5', dict(resampling_factor=1.5)), ('resampleAniso', dict(resampling_factor=(0.5, 2))),
                 ('resampleSize', dict(output_size=(10, 5))), ('resampleHalfConserve', dict(resampling_factor=0.5, conserve_array_sums=True)),
                 ('resampleSizeConserve', dict(output_size=(10, 5), conserve_array_sums=True))]:
    dc = cube(); dc.calibration.set_Q_pixel_size(0.04)
    dc = resample_data_diffraction(dc, method='bilinear', **kw)
    ops.append(rec(name, dc, args=kw, qPixelSize=dc.calibration.get_Q_pixel_size()))
mask = np.zeros((7, 9), bool); mask[0, 0] = mask[3, 4] = mask[3, 5] = mask[6, 8] = mask[0, 4] = True
for k in (3, 2, 4, 5):
    ops.append(rec(f'medianMasked{k}', median_filter_masked_pixels(cube(), mask, k), args=[k], mask=mask.ravel().astype(int).tolist()))

# calibration model
dc = cube()
c = dc.calibration
c.set_Q_pixel_size(0.05); c.set_Q_pixel_units('A^-1'); c.set_R_pixel_size(2.5); c.set_R_pixel_units('nm')
qx0 = 3.0 + rng.normal(0, 0.1, (4, 6)); qy0 = 4.0 + rng.normal(0, 0.1, (4, 6))
c.set_origin((qx0, qy0))
dc.calibrate()
cal = dict(qPixelSize=0.05, qPixelUnits='A^-1', rPixelSize=2.5, rPixelUnits='nm',
           qx0=flat(qx0), qy0=flat(qy0), qx0Mean=float(c.get_qx0_mean()), qy0Mean=float(c.get_qy0_mean()),
           qx0Shift=flat(c.get_qx0shift()), qy0Shift=flat(c.get_qy0shift()),
           # np.meshgrid(dim_x, dim_y) is (Ny, Nx): these are py4DSTEM's arrays, transposed in the kit
           qxxShape=list(dc.qxx.shape), qxx=flat(dc.qxx), qyy=flat(dc.qyy), rxx=flat(dc.rxx), ryy=flat(dc.ryy),
           qxxRaw=flat(dc.qxx_raw), qyyRaw=flat(dc.qyy_raw), rxxRaw=flat(dc.rxx_raw), ryyRaw=flat(dc.ryy_raw),
           qxxs=[dict(r=r, c=cc, values=flat(dc.qxxs(r, cc)), valuesY=flat(dc.qyys(r, cc))) for r, cc in [(0, 0), (2, 3), (3, 5)]],
           ellipse=None)
c.set_ellipse((1.1, 0.9, 0.3)); cal['ellipse'] = [1.1, 0.9, 0.3]
c.set_QR_rotflip_degrees((-12.5, True)); cal['qrRotationDegrees'] = c.get_QR_rotation_degrees(); cal['qrRotationRadians'] = c.get_QR_rotation()
cal['qrFlip'] = bool(c.get_QR_flip())
c.set_QR_rotation_degrees(30.0); cal['qrRotation30Radians'] = c.get_QR_rotation()
# no origin: calibrate falls back to (0, 0)
d0 = cube(); d0.calibration.set_Q_pixel_size(0.1); d0.calibrate()
cal['noOriginQxx'] = flat(d0.qxx)
# after binning the detector, calibration Q size multiplies; origin stays (upstream does not rescale it)
db = cube(); db.calibration.set_Q_pixel_size(0.05); db.calibration.set_origin((3.0, 4.0)); db = bin_data_diffraction(db, 2)
cal['afterBinQ'] = dict(qPixelSize=db.calibration.get_Q_pixel_size(), originMean=list(db.calibration.get_origin_mean()))
dp = pad_data_diffraction(cube(), output_size=(12, 14))
cal['afterPad'] = dict(qxxShape=list(dp.qxx.shape))
# process.calibration: qpixelsize and rotation
from py4DSTEM.process.calibration.qpixelsize import get_Q_pixel_size, get_dq_from_indexed_peaks
from py4DSTEM.process.calibration.rotation import get_Qvector_from_Rvector, get_Rvector_from_Qvector
qs = np.array([10.1, 14.2, 19.9, 22.6, 0.0, 28.7]); hkl = [(1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0), (0, 0, 0), (2, 2, 0)]
dq, qs_fit, hkl_fit = get_dq_from_indexed_peaks(qs, hkl, 3.905)
size, units = get_Q_pixel_size(25.0, 3.905, 'A')
vec = [(3.0, 4.0, 17.5), (-1.5, 2.5, -40.0), (0.2, -7.0, 123.0)]
rot = [dict(vx=a, vy=b, rotation=r, q=[float(x) for x in get_Qvector_from_Rvector(a, b, r)], r=[float(x) for x in get_Rvector_from_Qvector(a, b, r)]) for a, b, r in vec]
processes = dict(pixelSize=float(size), pixelSizeUnits=units, dq=float(dq), qsFit=flat(qs_fit), hklFit=[list(h) for h in hkl_fit],
                 qs=qs.tolist(), hkl=[list(h) for h in hkl], a=3.905, rotation=rot)
out = dict(reference='py4DSTEM f050d207 / 0.14.19, scipy ' + __import__('scipy').__version__, sourceShape=list(shape),
           floats=flat(floats), ops=ops, calibration=cal, processes=processes)
dest = KIT/'Tests/CuboCoreTests/Fixtures/calibration.json'
dest.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print('fixture bytes', dest.stat().st_size)
