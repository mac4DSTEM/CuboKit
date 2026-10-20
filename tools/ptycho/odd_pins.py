"""Pinned py4DSTEM oracle for ODD, non-square detector / probe / object sizes (CuboPtycho preprocessing and the three engines).
Run from the kit root:
  MPLCONFIGDIR="$PWD/.build/matplotlib" ../tools/memcap.sh -- References/py4dstem-venv/bin/python -I -B tools/ptycho/odd_pins.py
Same method as preprocess_pins.py and engines_pins.py (real py4DSTEM methods on numerically injected state; the engine data go
through py4DSTEM's _normalize_diffraction_intensities), with a 15 x 13 probe on a 31 x 29 object (single, mixed, multislice) and a
9 x 7 detector in the preprocessing pin. Odd lengths change fftfreq (no Nyquist bin), the corner-centred patch offsets, the
Fourier shifts, the centre of mass and the top-hat frequencies. Output: Tests/CuboPtychoTests/Fixtures/odd.json
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM.process.phase import MixedstatePtychography, MultislicePtychography, SingleslicePtychography
from py4DSTEM.process.phase.utils import ComplexProbe, electron_wavelength_angstrom, fft_shift
from py4DSTEM.preprocess.utils import get_shifted_ar
from scipy.ndimage import shift as ndshift
KIT = Path(__file__).resolve().parents[2]
REF = KIT.parents[0] / 'References/CuboKit/py4DSTEM-f050d207'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'], text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.0' and platform.python_version() == '3.12.4'
for name in ('singleslice_ptychography.py','mixedstate_ptychography.py','multislice_ptychography.py','ptychographic_methods.py','ptychographic_constraints.py','utils.py','phase_base_class.py'):
    installed = Path(py4DSTEM.__file__).parent/'process/phase'/name
    assert installed.read_bytes() == subprocess.check_output(['git','-C',str(REF),'show',f'HEAD:py4DSTEM/process/phase/{name}'])

def flat(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return [float('%.9g' % v) for v in a]
def cplx(a): a = np.asarray(a); return dict(real=flat(a.real), imaginary=flat(a.imag))
rng = np.random.default_rng(20261021)
out = dict(reference='py4DSTEM f050d207 / 0.14.19', numpy=np.__version__)
def make(**kw): return SingleslicePtychography(energy=200e3, verbose=False, object_type='complex', **kw)

# ---------------- preprocessing: scan 4 x 5, detector 9 x 7 ----------------
RX, RY, QX, QY = 4, 5, 9, 7
cube = np.round(rng.gamma(2.0, 30.0, (RX,RY,QX,QY)) - 8).astype(np.float32)
comx = (rng.uniform(-1.7,1.7,(RX,RY)) + 0.4).astype(np.float64)
comy = (rng.uniform(-1.7,1.7,(RX,RY)) - 0.3).astype(np.float64)
comx[0,0] = 0.0; comy[0,0] = 0.0; comx[1,1] = 2.0; comy[1,1] = -3.0
pre = dict(scanRows=RX, scanColumns=RY, detectorRows=QX, detectorColumns=QY, cube=flat(cube), originRows=flat(comx), originColumns=flat(comy))
amps = {}
for order in (1,3):
    a, mean, _, _ = make()._normalize_diffraction_intensities(cube.copy(), comx, comy, None, False, False, shifting_interpolation_order=order)
    total = 0.0
    for r in range(RX):
        for c in range(RY):
            s = (get_shifted_ar(cube[r,c].astype(np.float32), -comx[r,c], -comy[r,c], bilinear=True) if order == 1
                 else ndshift(cube[r,c].astype(np.float32), (-comx[r,c], -comy[r,c]), order=3, mode='grid-wrap'))
            total += float(np.maximum(s,0).sum())
    amps[str(order)] = dict(amplitudes=flat(a), meanIntensity=float(mean), clippedMeanIntensity=total/(RX*RY))
pre['amplitudes'] = amps
scan_sampling = (0.9, 1.3); dk = (0.7, 0.9)
p0 = make(); p0._angular_sampling = dk; p0._amplitudes_shape = np.array((QX,QY))
pre.update(wavelength=float(electron_wavelength_angstrom(200e3)), angularSampling=list(dk), objectSampling=list(p0.sampling), scanSampling=list(scan_sampling))
cases = []
for angle, transpose in [(0.0,False),(0.35,False),(-1.1,True),(2.3,True)]:
    p = make(); p._angular_sampling = dk; p._amplitudes_shape = np.array((QX,QY))
    p._grid_scan_shape = (RX,RY); p._scan_sampling = scan_sampling
    p._rotation_best_rad = angle; p._rotation_best_transpose = transpose
    p._region_of_interest_shape = np.array((QX,QY))
    pos, pad = p._calculate_scan_positions_in_pixels(None, None, None, None)
    p._object_padding_px = pad
    obj = p._initialize_object(None, pos, 'complex')
    shape = list(obj.shape)
    pf = np.asarray(pos, dtype=np.float32)
    pf = pf - (pf.mean(0) - np.array(shape, dtype=np.float32)/2)
    cases.append(dict(angle=angle, transpose=transpose, positions=pos.tolist(), padding=[[float(pad[0][0]),float(pad[0][1])],[float(pad[1][0]),float(pad[1][1])]],
                      objectShape=shape, centred=pf.tolist()))
pre['positionCases'] = cases
probes = []
for name, kw in [('plain', dict(semiangle_cutoff=12.0, rolloff=2.0)),
                 ('aberrated', dict(semiangle_cutoff=12.0, rolloff=2.0, defocus=60.0, C12=20.0, phi12=0.4, C30=1.5e4))]:
    p = make(**kw); p._angular_sampling = dk; p._amplitudes_shape = np.array((QX,QY)); p._region_of_interest_shape = np.array((QX,QY))
    p._crop_mask = None; p._crop_mask_shape = None; p._resample_exit_waves = False
    probe, cutoff = p._initialize_probe(None, None, 1234.5, kw['semiangle_cutoff'], False)
    probes.append(dict(name=name, semiangle=kw['semiangle_cutoff'], rolloff=2.0, defocus=kw.get('defocus',0.0), astigmatism=kw.get('C12',0.0),
                       astigmatismAngle=kw.get('phi12',0.0), sphericalAberration=kw.get('C30',0.0), meanIntensity=1234.5,
                       real=flat(probe.real), imaginary=flat(probe.imag)))
pre['probes'] = probes
out['preprocessing'] = pre

# ---------------- engines: 15 x 13 probe, 31 x 29 object, 6 x 6 scan ----------------
nr, nc, sr_, sc_ = 15, 13, 31, 29
SX, SY = 6, 6
sampling = (0.8, 1.1)
base = ComplexProbe(energy=200e3, gpts=(nr,nc), sampling=sampling, semiangle_cutoff=18, rolloff=2, defocus=70, C12=25, phi12=0.31, C30=2e4).build()._array
positions = np.array([[7.5+2*r+0.25*(r%2), 6.5+2*c+0.5*(c%2)] for r in range(SX) for c in range(SY)], np.float32)
r, c = np.indices((sr_, sc_))
true_obj = ((0.94+0.025*np.cos(r*0.3))*np.exp(1j*(0.15*np.sin(r*0.27)+0.2*np.cos(c*0.21)+rng.normal(0,0.003,(sr_,sc_))))).astype(np.complex64)
start_obj = (0.98*np.exp(1j*rng.normal(0,0.02,(sr_,sc_)))).astype(np.complex64)
eng = dict(rows=nr, columns=nc, objectRows=sr_, objectColumns=sc_, rowSampling=sampling[0], columnSampling=sampling[1],
           wavelength=float(electron_wavelength_angstrom(200e3)), positions=positions.tolist(), initialObject=cplx(start_obj))
pre_obj = make()
noise_rng = np.random.default_rng(778)
ORIGIN = (0.3, -0.45)
def through_preprocessing(model_amps):
    intensity = model_amps.astype(np.float64)**2
    raw = (intensity - 0.8*intensity.mean() + noise_rng.normal(0, 0.5*intensity.mean(), intensity.shape)).astype(np.float32)
    a, mean, _, _ = pre_obj._normalize_diffraction_intensities(raw.reshape(SX,SY,nr,nc).copy(), np.full((SX,SY), ORIGIN[0]), np.full((SX,SY), ORIGIN[1]),
                                                               None, False, False, shifting_interpolation_order=1)
    a = np.asarray(a, np.float32).reshape(len(positions), nr, nc)
    energy = float(np.sum(a.astype(np.float64)**2)/len(positions))
    assert abs(float(mean)/energy - 1) > 0.02, (float(mean), energy)
    return dict(raw=flat(raw), origin=list(ORIGIN), meanIntensity=float(mean), energyPerPosition=energy), a, float(mean)

def noise_state(scale):
    z = rng.normal(0,1,(nr,nc)) + 1j*rng.normal(0,1,(nr,nc))
    z = np.fft.ifft2(np.fft.fft2(z) * (np.abs(np.fft.fft2(base)) > 0))
    return (scale * z / np.sqrt(np.sum(np.abs(z)**2)) * np.sqrt(np.sum(np.abs(base)**2))).astype(np.complex64)

def inject(p, amplitudes, probe, obj, mean):
    p._region_of_interest_shape = np.array((nr,nc)); p._object_shape = np.array((sr_,sc_)); p._resample_exit_waves = False
    p._object = obj.copy(); p._probe = probe.copy()
    p._positions_px = positions.copy(); p._positions_px_initial = positions.copy()
    p._amplitudes = amplitudes; p._num_diffraction_patterns = len(positions); p._mean_diffraction_intensity = mean
    p._object_fov_mask_inverse = np.zeros((sr_,sc_),bool); p._preprocessed = True; p._sampling = sampling
    p._probe_initial_aperture = np.abs(np.fft.fft2(probe)); p.error_iterations = []

common = dict(fix_positions=True, gaussian_filter=False, butterworth_filter=False, tv_denoise=False,
              fix_potential_baseline=False, progress_bar=False, reset=False)
fy = np.fft.fftfreq(nr)*nr; fx = np.fft.fftfreq(nc)*nc
support = (np.hypot(fy[:,None], fx[None,:]) < 5.5)
eng['support'] = [bool(v) for v in support.ravel()]

# forward models
dummy = MixedstatePtychography(energy=200e3, num_probes=3, object_type='complex', verbose=False)
inject(dummy, np.zeros((len(positions),nr,nc),np.float32), base.copy(), start_obj, 1.0); dummy._num_probes = 3
rows_idx, cols_idx = dummy._extract_vectorized_patch_indices(positions)
true_states = np.stack([base, 0.55*fft_shift(base, np.array([1.5,-2.0]), np).astype(np.complex64), 0.3*noise_state(1.0)])
start_states = np.stack([0.9*fft_shift(base, np.array([0.6,0.4]), np).astype(np.complex64), noise_state(0.5), noise_state(0.2)]).astype(np.complex64)
shifted = dummy._return_shifted_probes(true_states, positions-np.round(positions))
amps_mixed = np.sqrt(np.sum(np.abs(np.fft.fft2(shifted*true_obj[rows_idx,cols_idx][:,None]))**2, axis=1)).astype(np.float32)
mixed_pre, amps_mixed, mean_mixed = through_preprocessing(amps_mixed)
start_probe = (0.93*fft_shift(base, np.array([0.5,-0.4]), np)).astype(np.complex64)

thick = [55.0, 80.0]
true_slices = np.stack([true_obj] + [((0.95+0.02*np.cos(r*0.17+k))*np.exp(1j*(0.12*np.sin(c*0.31+k)+0.1*np.cos(r*0.23)))).astype(np.complex64) for k in (1,2)])
start_slices = np.stack([start_obj] + [(0.98*np.exp(1j*rng.normal(0,0.02,(sr_,sc_)))).astype(np.complex64) for _ in (1,2)])
msd = MultislicePtychography(energy=200e3, num_slices=3, slice_thicknesses=np.array(thick), object_type='complex', verbose=False)
props = msd._precompute_propagator_arrays((nr,nc), sampling, 200e3, np.array(thick))
inject(msd, np.zeros((len(positions),nr,nc),np.float32), base.copy(), start_slices, 1.0)
msd._num_slices = 3; msd._propagator_arrays = props
shifted_in = msd._return_shifted_probes(base.copy(), positions-np.round(positions))
_, _, overlap = msd._overlap_projection(true_slices, rows_idx, cols_idx, shifted_in)
ms_pre, amps_ms, mean_ms = through_preprocessing(np.abs(np.fft.fft2(overlap)).astype(np.float32))

eng['startProbe'] = cplx(start_probe); eng['startStates'] = cplx(start_states); eng['startSlices'] = cplx(start_slices)
eng['mixedPreprocessing'] = mixed_pre; eng['mixedAmplitudes'] = flat(amps_mixed)
eng['multislicePreprocessing'] = ms_pre; eng['amplitudes'] = flat(amps_ms)
eng['propagators'] = cplx(props); eng['sliceThicknesses'] = thick

# single-slice cases
single = []
for name, kw in [('plain', dict(fix_probe_com=False)),
                 ('constraints', dict(fix_probe_com=True, constrain_probe_amplitude=True, constrain_probe_amplitude_relative_radius=0.4,
                                       constrain_probe_amplitude_relative_width=0.1)),
                 ('apertureSupport', dict(fix_probe_com=True, fix_probe_aperture=True, probe_real_space_support_mask=support))]:
    q = SingleslicePtychography(energy=200e3, object_type='complex', verbose=False)
    inject(q, amps_ms, start_probe, start_obj, mean_ms)
    np.random.seed(1729)
    q.reconstruct(num_iter=3, step_size=0.35, normalization_min=0.7, **kw, **common)
    single.append(dict(name=name, fixProbeCenterOfMass=kw.get('fix_probe_com'), constrainProbeAmplitude=kw.get('constrain_probe_amplitude', False),
                       radius=kw.get('constrain_probe_amplitude_relative_radius', 0.5), width=kw.get('constrain_probe_amplitude_relative_width', 0.05),
                       aperture=kw.get('fix_probe_aperture', False), support=kw.get('probe_real_space_support_mask') is not None,
                       object=cplx(q._object), probe=cplx(q._probe), errors=q.error_iterations))
eng['single'] = single
# mixed cases
mixed = []
for name, k, kw in [('two', 2, dict(fix_probe_com=False, orthogonalize_probe=True)),
                    ('threeAll', 3, dict(fix_probe_com=True, orthogonalize_probe=True, constrain_probe_amplitude=True,
                                          constrain_probe_amplitude_relative_radius=0.4, constrain_probe_amplitude_relative_width=0.1,
                                          fix_probe_aperture=True, probe_real_space_support_mask=support)),
                    ('noOrthogonalize', 2, dict(fix_probe_com=False, orthogonalize_probe=False))]:
    q = MixedstatePtychography(energy=200e3, num_probes=k, object_type='complex', verbose=False)
    inject(q, amps_mixed, start_states[:k], start_obj, mean_mixed); q._num_probes = k
    np.random.seed(1729)
    q.reconstruct(num_iter=3, step_size=0.35, normalization_min=0.7, **kw, **common)
    mixed.append(dict(name=name, numProbes=k, fixProbeCenterOfMass=kw.get('fix_probe_com'), orthogonalize=kw['orthogonalize_probe'],
                      constrainProbeAmplitude=kw.get('constrain_probe_amplitude', False),
                      radius=kw.get('constrain_probe_amplitude_relative_radius', 0.5), width=kw.get('constrain_probe_amplitude_relative_width', 0.05),
                      aperture=kw.get('fix_probe_aperture', False), support=kw.get('probe_real_space_support_mask') is not None,
                      object=cplx(q._object), probes=cplx(q._probe), errors=q.error_iterations))
eng['mixed'] = mixed
# seeded initialisation of the mixed states at an odd shape
p = MixedstatePtychography(energy=200e3, num_probes=3, object_type='complex', verbose=False, semiangle_cutoff=18.0, rolloff=2.0)
p._region_of_interest_shape = (nr,nc); p._crop_mask = None; p._crop_mask_shape = None; p._resample_exit_waves = False
po = ComplexProbe(energy=200e3, gpts=(nr,nc), sampling=sampling, semiangle_cutoff=18, rolloff=2, defocus=70, C12=25, phi12=0.31, C30=2e4); po.build()
np.random.seed(5)
states, _ = p._initialize_probe(po, None, 777.0, 18.0, False)
eng['init'] = dict(seed=5, count=3, first=cplx(po._array), meanIntensity=777.0, states=cplx(states))
# multislice cases
ms = []
for name, ns, th, kw in [('three', 3, thick, dict(fix_probe_com=False)),
                         ('constraints', 3, thick, dict(fix_probe_com=True, constrain_probe_amplitude=True, constrain_probe_amplitude_relative_radius=0.4,
                                                        constrain_probe_amplitude_relative_width=0.1, fix_probe_aperture=True, probe_real_space_support_mask=support)),
                         ('two', 2, [90.0], dict(fix_probe_com=False))]:
    q = MultislicePtychography(energy=200e3, num_slices=ns, slice_thicknesses=np.array(th), object_type='complex', verbose=False)
    inject(q, amps_ms, start_probe, start_slices[:ns], mean_ms)
    q._propagator_arrays = q._precompute_propagator_arrays((nr,nc), sampling, 200e3, np.array(th))
    np.random.seed(1729)
    q.reconstruct(num_iter=3, step_size=0.35, normalization_min=0.7, tv_denoise_chambolle=False, kz_regularization_filter=False, **kw, **common)
    ms.append(dict(name=name, slices=ns, thicknesses=th, fixProbeCenterOfMass=kw.get('fix_probe_com'),
                   constrainProbeAmplitude=kw.get('constrain_probe_amplitude', False),
                   radius=kw.get('constrain_probe_amplitude_relative_radius', 0.5), width=kw.get('constrain_probe_amplitude_relative_width', 0.05),
                   aperture=kw.get('fix_probe_aperture', False), support=kw.get('probe_real_space_support_mask') is not None,
                   object=cplx(q._object), probe=cplx(q._probe), errors=q.error_iterations))
eng['multislice'] = ms
# an odd-sized built probe (aperture, rolloff, aberrations) for PtychographyProbe.build
eng['builtProbe'] = cplx(ComplexProbe(energy=200e3, gpts=(nr,nc), sampling=sampling, semiangle_cutoff=18, rolloff=2, defocus=70, C12=25, phi12=0.31, C30=2e4).build()._array)
out['engines'] = eng
dest = KIT/'Tests/CuboPtychoTests/Fixtures/odd.json'
dest.write_text(json.dumps(out,separators=(',',':'))+'\n')
print('fixture bytes', dest.stat().st_size)
