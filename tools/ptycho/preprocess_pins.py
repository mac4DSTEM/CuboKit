"""Pinned py4DSTEM preprocessing oracle for CuboPtycho (amplitudes, scan positions, object shape, probe normalisation).
Run from the kit root:
  MPLCONFIGDIR="$PWD/.build/matplotlib" ../tools/memcap.sh -- ../References/python/py4dstem/bin/python -I -B tools/ptycho/preprocess_pins.py
Actual py4DSTEM methods (_normalize_diffraction_intensities, _calculate_scan_positions_in_pixels, _initialize_object,
_initialize_probe) are called on a SingleslicePtychography whose preprocessing attributes are set directly; centre of mass,
rotation search and calibration extraction are outside this pin (the origins and the rotation are inputs).
Where py4DSTEM has no equivalent (the app's clipped mean intensity, per-axis padding), the value is a NumPy
transcription of mac4DSTEM PtychographyPreparation.swift and is marked 'app' in the fixture.
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM.process.phase import SingleslicePtychography
from py4DSTEM.process.phase.utils import electron_wavelength_angstrom
KIT = Path(__file__).resolve().parents[2]
REF = KIT.parents[0] / 'References/CuboKit/py4DSTEM-f050d207'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'], text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
for name in ('singleslice_ptychography.py','ptychographic_methods.py','utils.py','phase_base_class.py'):
    installed = Path(py4DSTEM.__file__).parent/'process/phase'/name
    assert installed.read_bytes() == subprocess.check_output(['git','-C',str(REF),'show',f'HEAD:py4DSTEM/process/phase/{name}'])

def flat(a): return np.asarray(a).ravel().tolist()
rng = np.random.default_rng(20261020)
RX, RY, QX, QY = 5, 6, 12, 10          # scan rows, columns; detector rows, columns (non-square on purpose)
cube = np.round(rng.gamma(2.0, 30.0, (RX,RY,QX,QY)) - 8).astype(np.float32)   # includes negatives (clipping)
comx = (rng.uniform(-1.7,1.7,(RX,RY)) + 0.4).astype(np.float64)
comy = (rng.uniform(-1.7,1.7,(RX,RY)) - 0.3).astype(np.float64)
comx[0,0] = 0.0; comy[0,0] = 0.0; comx[1,1] = 2.0; comy[1,1] = -3.0      # integer shifts exercise the exact-tap path
out = dict(reference='py4DSTEM f050d207 / 0.14.19', numpy=np.__version__, scanRows=RX, scanColumns=RY,
           detectorRows=QX, detectorColumns=QY, cube=flat(cube), originRows=flat(comx), originColumns=flat(comy))

def make(**kw):
    p = SingleslicePtychography(energy=200e3, verbose=False, object_type='complex', **kw)
    return p
# amplitudes: order 1 (bilinear, get_shifted_ar) and order 3 (scipy spline, grid-wrap = py4DSTEM default)
amps = {}
for order in (1,3):
    p = make()
    a, mean, _, _ = p._normalize_diffraction_intensities(cube.copy(), comx, comy, None, False, False,
                                                         shifting_interpolation_order=order)
    amps[str(order)] = dict(amplitudes=flat(a), meanIntensity=float(mean))
    # app transcription of the mean: clipped intensities (the app sums max(value, 0))
    from scipy.ndimage import shift
    total = 0.0
    for r in range(RX):
        for c in range(RY):
            if order == 1:
                from py4DSTEM.preprocess.utils import get_shifted_ar
                s = get_shifted_ar(cube[r,c].astype(np.float32), -comx[r,c], -comy[r,c], bilinear=True)
            else:
                s = shift(cube[r,c].astype(np.float32), (-comx[r,c], -comy[r,c]), order=3, mode='grid-wrap')
            total += float(np.maximum(s,0).sum())
    amps[str(order)]['clippedMeanIntensity'] = total/(RX*RY)
out['amplitudes'] = amps

# positions, object shape, centring
scan_sampling = (0.9, 1.3)
dk = (0.7, 0.9)                         # angular sampling [mrad] per detector pixel
p0 = make()
p0._angular_sampling = dk
p0._amplitudes_shape = np.array((QX,QY))
sampling = p0.sampling
out['wavelength'] = float(electron_wavelength_angstrom(200e3)); out['angularSampling'] = list(dk)
out['objectSampling'] = list(sampling)
out['scanSampling'] = list(scan_sampling)
cases = []
for angle, transpose in [(0.0,False),(0.35,False),(-1.1,True),(2.3,True),(0.0,True)]:
    p = make()
    p._angular_sampling = dk; p._amplitudes_shape = np.array((QX,QY))
    p._grid_scan_shape = (RX,RY); p._scan_sampling = scan_sampling
    p._rotation_best_rad = angle; p._rotation_best_transpose = transpose
    p._region_of_interest_shape = np.array((QX,QY))
    pos, pad = p._calculate_scan_positions_in_pixels(None, None, None, None)
    p._object_padding_px = pad
    obj = p._initialize_object(None, pos, 'complex')
    shape = list(obj.shape)
    # centre in the field of view exactly as preprocess() does (float32 state)
    pf = np.asarray(pos, dtype=np.float32)
    pf = pf - (pf.mean(0) - np.array(shape, dtype=np.float32)/2)
    # app transcription (mac4DSTEM): per-axis padding, Double rotation, rounded object extent
    cases.append(dict(angle=angle, transpose=transpose, positions=pos.tolist(), padding=[[float(pad[0][0]),float(pad[0][1])],[float(pad[1][0]),float(pad[1][1])]],
                      objectShape=shape, centred=pf.tolist()))
out['positionCases'] = cases

# probe: ComplexProbe built and normalised to the mean intensity (py4DSTEM _initialize_probe)
probes = []
for name, kw in [('plain', dict(semiangle_cutoff=12.0, rolloff=2.0)),
                 ('aberrated', dict(semiangle_cutoff=12.0, rolloff=2.0, defocus=60.0, C12=20.0, phi12=0.4, C30=1.5e4))]:
    p = make(**kw)
    p._angular_sampling = dk; p._amplitudes_shape = np.array((QX,QY))
    p._region_of_interest_shape = np.array((QX,QY))
    p._crop_mask = None; p._crop_mask_shape = None; p._resample_exit_waves = False
    mean = 1234.5
    probe, cutoff = p._initialize_probe(None, None, mean, kw['semiangle_cutoff'], False)
    probes.append(dict(name=name, semiangle=kw['semiangle_cutoff'], rolloff=2.0, defocus=kw.get('defocus',0.0),
                       astigmatism=kw.get('C12',0.0), astigmatismAngle=kw.get('phi12',0.0), sphericalAberration=kw.get('C30',0.0),
                       meanIntensity=mean, real=flat(probe.real), imaginary=flat(probe.imag)))
out['probes'] = probes
dest = KIT/'Tests/CuboPtychoTests/Fixtures/preprocess.json'
dest.write_text(json.dumps(out,separators=(',',':'))+'\n')
print('fixture bytes', dest.stat().st_size)
