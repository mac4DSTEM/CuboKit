"""Pinned py4DSTEM oracle for position correction in the single-slice engine (fix_positions=False, positions_step_size,
max_position_update_distance, max_position_total_distance, fix_positions_com).
Run from the clone root:
  MPLCONFIGDIR="$PWD/.build/matplotlib" python -I -B tools/ptycho/position_pins.py
The real SingleslicePtychography.reconstruct runs on numerically injected preprocessing state (as pins.py). The data are generated at
positions that differ from the starting positions by up to 0.3 px (uniform), so the correction has work to do; they go through
py4DSTEM's _normalize_diffraction_intensities from a raw cube with a mild offset and noise. Output: Fixtures/positions.json
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM.process.phase import SingleslicePtychography
from py4DSTEM.process.phase.utils import ComplexProbe, electron_wavelength_angstrom, fft_shift
KIT = Path(__file__).resolve().parents[2]
import sys as _sys, pathlib as _pathlib; _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # tools/pinenv.py
from pinenv import PY4DSTEM_SRC
REF = PY4DSTEM_SRC
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'], text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.4' and platform.python_version() == '3.12.14'
for name in ('singleslice_ptychography.py','ptychographic_methods.py','ptychographic_constraints.py','utils.py','phase_base_class.py'):
    installed = Path(py4DSTEM.__file__).parent/'process/phase'/name
    assert installed.read_bytes() == subprocess.check_output(['git','-C',str(REF),'show',f'HEAD:py4DSTEM/process/phase/{name}'])

def flat(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return [float('%.9g' % v) for v in a]
def cplx(a): a = np.asarray(a); return dict(real=flat(a.real), imaginary=flat(a.imag))
rng = np.random.default_rng(20261022)
n, side = 16, 32
sampling = (0.8, 1.1)
probe = ComplexProbe(energy=200e3, gpts=(n,n), sampling=sampling, semiangle_cutoff=12, rolloff=4, defocus=20).build()._array
nominal = np.array([[8+2*r+0.25*(r%2), 8+2*c+0.5*(c%2)] for r in range(8) for c in range(8)], np.float32)
truth_positions = (nominal + rng.uniform(-0.3, 0.3, nominal.shape)).astype(np.float32)
r, c = np.indices((side, side))
obj = ((0.94+0.025*np.cos(r*0.3))*np.exp(1j*(0.15*np.sin(r*0.27)+0.2*np.cos(c*0.21)+rng.normal(0,0.003,(side,side))))).astype(np.complex64)
# a structured object (the position gradient needs contrast): amplitude texture and a few Gaussian bumps of phase
obj = (obj*(0.7+0.3*np.cos(0.55*r)*np.cos(0.43*c))).astype(np.complex64)
for (r0, c0, w, a) in [(10,12,3.0,0.5),(20,9,2.5,-0.4),(16,22,3.5,0.45),(8,24,2.0,0.35)]:
    obj = (obj*np.exp(1j*a*np.exp(-((r-r0)**2+(c-c0)**2)/(2*w*w)))).astype(np.complex64)
p = SingleslicePtychography(energy=200e3, object_type='complex', verbose=False)
p._region_of_interest_shape = np.array((n,n)); p._object_shape = np.array((side,side)); p._resample_exit_waves = False
rows, cols = p._extract_vectorized_patch_indices(truth_positions)
shifted = p._return_shifted_probes(probe, truth_positions-np.round(truth_positions))
model = np.abs(np.fft.fft2(shifted*obj[rows,cols])).astype(np.float64)**2
noise_rng = np.random.default_rng(779)
raw = (model - 0.15*model.mean() + noise_rng.normal(0, 0.03*model.mean(), model.shape)).astype(np.float32)
a, mean, _, _ = p._normalize_diffraction_intensities(raw.reshape(8,8,n,n).copy(), np.zeros((8,8)), np.zeros((8,8)), None, False, False,
                                                     shifting_interpolation_order=1)
amplitudes = np.asarray(a, np.float32).reshape(len(nominal), n, n); mean = float(mean)
start_obj = (0.97*obj*np.exp(1j*rng.normal(0,0.05,(side,side)))).astype(np.complex64)
start_probe = (0.95*fft_shift(probe, np.array([0.4,-0.3]), np)).astype(np.complex64)

def run(amps=None, mean_=None, o0=None, p0=None, iterations=4, object_step=0.35, **kw):
    q = SingleslicePtychography(energy=200e3, object_type='complex', verbose=False)
    q._region_of_interest_shape = np.array((n,n)); q._object_shape = np.array((side,side)); q._resample_exit_waves = False
    q._object = (start_obj if o0 is None else o0).copy(); q._probe = (start_probe if p0 is None else p0).copy()
    q._positions_px = nominal.copy(); q._positions_px_initial = nominal.copy()
    q._amplitudes = amplitudes if amps is None else amps; q._num_diffraction_patterns = len(nominal)
    q._mean_diffraction_intensity = mean if mean_ is None else mean_
    q._object_fov_mask_inverse = np.zeros((side,side),bool); q._preprocessed = True; q._sampling = sampling
    q._probe_initial_aperture = np.abs(np.fft.fft2(q._probe)); q.error_iterations = []
    wl = electron_wavelength_angstrom(200e3)
    q._amplitudes_shape = np.array((n,n)); q._angular_sampling = tuple(wl*1e3/(s*n) for s in sampling)      # q.sampling == sampling
    np.random.seed(1729)
    q.reconstruct(num_iter=iterations, step_size=object_step, normalization_min=0.7, fix_positions=False, fix_probe_com=False, gaussian_filter=False,
                  butterworth_filter=False, tv_denoise=False, fix_potential_baseline=False, progress_bar=False, reset=False, **kw)
    return q

cases = []
for name, kw, mine in [
    ('default', dict(), dict(stepSize=0.5)),
    ('smallStep', dict(positions_step_size=0.2), dict(stepSize=0.2)),
    ('limited', dict(positions_step_size=0.5, max_position_update_distance=0.4, max_position_total_distance=0.9),
     dict(stepSize=0.5, maxUpdateDistance=0.4, maxTotalDistance=0.9)),
    ('centreOfMass', dict(positions_step_size=0.5, fix_positions_com=False), dict(stepSize=0.5, constrainCenterOfMass=True)),
    ('fixedProbe', dict(positions_step_size=0.5, fix_probe=True), dict(stepSize=0.5, fixProbe=True)),
]:
    q = run(**kw)
    cases.append(dict(name=name, settings=mine, fixProbe=bool(kw.get('fix_probe', False)), object=cplx(q._object), probe=cplx(q._probe),
                      errors=q.error_iterations, positions=np.asarray(q._positions_px, np.float64).tolist()))
# exact object, probe and data at truth_positions; the object step is negligible, so the estimate alone moves the positions
exact_amps = np.sqrt(model).astype(np.float32)
q = run(amps=exact_amps, mean_=float(np.sum(model)/len(nominal)), o0=obj.astype(np.complex64), p0=probe.astype(np.complex64), iterations=1,
        object_step=1e-12, positions_step_size=0.5, fix_probe=True)
exact = dict(amplitudes=flat(exact_amps), meanIntensity=float(np.sum(model)/len(nominal)), object=cplx(obj), probe=cplx(probe),
             positions=np.asarray(q._positions_px, np.float64).tolist(), errors=q.error_iterations)
d0 = np.linalg.norm(nominal-truth_positions, axis=1).mean(); d1 = np.linalg.norm(np.asarray(q._positions_px)-truth_positions, axis=1).mean()
print('exact data: mean distance to truth', d0, '->', d1); assert d1 < d0
out = dict(exact=exact, reference='py4DSTEM f050d207 / 0.14.19', rows=n, columns=n, objectRows=side, objectColumns=side, rowSampling=sampling[0],
           columnSampling=sampling[1], nominalPositions=nominal.tolist(), truthPositions=truth_positions.tolist(),
           amplitudes=flat(amplitudes), meanIntensity=mean, startObject=cplx(start_obj), startProbe=cplx(start_probe), cases=cases)
dest = KIT/'Tests/CuboPtychoTests/Fixtures/positions.json'
dest.write_text(json.dumps(out,separators=(',',':'))+'\n')
print('fixture bytes', dest.stat().st_size)
for cs in cases:
    d = np.abs(np.array(cs['positions']) - truth_positions).max(); d0 = np.abs(nominal - truth_positions).max()
    print(cs['name'], 'max |position - truth| now', d, 'at start', d0, 'errors', cs['errors'][0], cs['errors'][-1])
