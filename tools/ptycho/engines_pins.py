"""Pinned py4DSTEM oracle for the mixed-state and multislice CPU engines (CuboPtycho).
Run from the kit root:
  MPLCONFIGDIR="$PWD/.build/matplotlib" ../tools/memcap.sh -- ../References/CuboKit/py4dstem-venv/bin/python -I -B tools/ptycho/engines_pins.py
The real MixedstatePtychography / MultislicePtychography classes run `reconstruct` on numerically injected preprocessing state
(the same method as tools/ptycho/pins.py); orthogonalisation, probe initialisation (seeded NumPy RNG) and the propagator
arrays are also called directly. Output: Tests/CuboPtychoTests/Fixtures/engines.json
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM.process.phase import MixedstatePtychography, MultislicePtychography
from py4DSTEM.process.phase.utils import ComplexProbe, electron_wavelength_angstrom, fft_shift
KIT = Path(__file__).resolve().parents[2]
REF = KIT.parents[0] / 'References/CuboKit/py4DSTEM-f050d207'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'], text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.0' and platform.python_version() == '3.12.4'
for name in ('mixedstate_ptychography.py','multislice_ptychography.py','ptychographic_methods.py','ptychographic_constraints.py','utils.py','phase_base_class.py'):
    installed = Path(py4DSTEM.__file__).parent/'process/phase'/name
    assert installed.read_bytes() == subprocess.check_output(['git','-C',str(REF),'show',f'HEAD:py4DSTEM/process/phase/{name}'])

def flat(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return [float('%.9g' % v) for v in a]      # float32 values print exactly in 9 significant digits
def cplx(a): a = np.asarray(a); return dict(real=flat(a.real), imaginary=flat(a.imag))
rng = np.random.default_rng(20261020)
n, side = 16, 32
sampling = (0.8, 1.1)
base = ComplexProbe(energy=200e3, gpts=(n,n), sampling=sampling, semiangle_cutoff=18, rolloff=2, defocus=70, C12=25, phi12=0.31, C30=2e4).build()._array
positions = np.array([[8+2*r+0.25*(r%2), 8+2*c+0.5*(c%2)] for r in range(8) for c in range(8)], np.float32)
r, c = np.indices((side, side))
true_obj = ((0.94+0.025*np.cos(r*0.3))*np.exp(1j*(0.15*np.sin(r*0.27)+0.2*np.cos(c*0.21)+rng.normal(0,0.003,(side,side))))).astype(np.complex64)
start_obj = (0.98*np.exp(1j*rng.normal(0,0.02,(side,side)))).astype(np.complex64)
out = dict(reference='py4DSTEM f050d207 / 0.14.19', numpy=np.__version__, rows=n, columns=n, objectRows=side, objectColumns=side,
           rowSampling=sampling[0], columnSampling=sampling[1], wavelength=float(electron_wavelength_angstrom(200e3)),
           positions=positions.tolist(), initialObject=cplx(start_obj))

def noise_state(scale):
    z = rng.normal(0,1,(n,n)) + 1j*rng.normal(0,1,(n,n))
    # a smooth, aperture-limited state so the intensity is concentrated like a probe
    zf = np.fft.fft2(z) * (np.abs(np.fft.fft2(base)) > 0)
    z = np.fft.ifft2(zf)
    return (scale * z / np.sqrt(np.sum(np.abs(z)**2)) * np.sqrt(np.sum(np.abs(base)**2))).astype(np.complex64)

def inject(p, amplitudes, probe, obj, shape3=None):
    p._region_of_interest_shape = np.array((n,n))
    p._object_shape = np.array((side,side))
    p._resample_exit_waves = False
    p._object = obj.copy()
    p._probe = probe.copy()
    p._positions_px = positions.copy()
    p._positions_px_initial = positions.copy()
    p._amplitudes = amplitudes
    p._num_diffraction_patterns = len(positions)
    p._mean_diffraction_intensity = float(np.sum(amplitudes.astype(np.float64)**2)/len(positions))
    p._object_fov_mask_inverse = np.zeros((side,side),bool)
    p._preprocessed = True
    p._sampling = sampling
    p._probe_initial_aperture = np.abs(np.fft.fft2(probe))
    p.error_iterations = []

common = dict(fix_positions=True, gaussian_filter=False, butterworth_filter=False, tv_denoise=False,
              fix_potential_baseline=False, progress_bar=False, reset=False)

# ---------------- mixed state ----------------
true_states = np.stack([base, 0.55*fft_shift(base, np.array([1.5,-2.0]), np).astype(np.complex64), 0.3*noise_state(1.0)])
start_states = np.stack([0.9*fft_shift(base, np.array([0.6,0.4]), np).astype(np.complex64), noise_state(0.5), noise_state(0.2)]).astype(np.complex64)
dummy = MixedstatePtychography(energy=200e3, num_probes=3, object_type='complex', verbose=False)
inject(dummy, np.zeros((len(positions),n,n),np.float32), start_states, start_obj)
dummy._num_probes = 3
rows_idx, cols_idx = dummy._extract_vectorized_patch_indices(positions)
shifted = dummy._return_shifted_probes(true_states, positions-np.round(positions))
amps_mixed = np.sqrt(np.sum(np.abs(np.fft.fft2(shifted*true_obj[rows_idx,cols_idx][:,None]))**2, axis=1)).astype(np.float32)
out['mixed'] = dict(amplitudes=flat(amps_mixed), startStates=cplx(start_states))

mixed_cases = []
for name, k, kw in [
    ('two', 2, dict(fix_probe_com=False, orthogonalize_probe=True)),
    ('threeAll', 3, dict(fix_probe_com=True, orthogonalize_probe=True, constrain_probe_amplitude=True,
                          constrain_probe_amplitude_relative_radius=0.4, constrain_probe_amplitude_relative_width=0.1)),
    ('noOrthogonalize', 2, dict(fix_probe_com=False, orthogonalize_probe=False)),
    ('fixedProbe', 2, dict(fix_probe_com=True, orthogonalize_probe=True, fix_probe=True)),
    ('purePhaseThree', 3, dict(fix_probe_com=False, orthogonalize_probe=True, pure_phase_object=True)),
]:
    p = MixedstatePtychography(energy=200e3, num_probes=k, object_type='complex', verbose=False)
    inject(p, amps_mixed, start_states[:k], start_obj)
    p._num_probes = k
    np.random.seed(1729)
    p.reconstruct(num_iter=3, step_size=0.35, normalization_min=0.7, **kw, **common)
    d = dict(name=name, numProbes=k, fixProbeCenterOfMass=kw.get('fix_probe_com'), orthogonalize=kw.get('orthogonalize_probe'),
             constrainProbeAmplitude=kw.get('constrain_probe_amplitude', False),
             radius=kw.get('constrain_probe_amplitude_relative_radius', 0.5), width=kw.get('constrain_probe_amplitude_relative_width', 0.05),
             fixProbe=kw.get('fix_probe', False), purePhase=kw.get('pure_phase_object', False),
             object=cplx(p._object), probes=cplx(p._probe), errors=p.error_iterations)
    mixed_cases.append(d)
out['mixed']['cases'] = mixed_cases

# orthogonalisation alone (injected stack with a separated spectrum), and seeded initialisation
stack = np.stack([noise_state(1.0), noise_state(0.45), noise_state(0.2), noise_state(0.08)]).astype(np.complex64)
stack[1] += 0.3*stack[0]; stack[2] += 0.2*stack[0] - 0.1j*stack[1]
p = MixedstatePtychography(energy=200e3, num_probes=4, object_type='complex', verbose=False)
p._num_probes = 4
orth = p._probe_orthogonalization_constraint(stack.copy())
out['mixed']['orthogonalization'] = dict(input=cplx(stack), output=cplx(orth), count=4)

inits = []
for seed, k in [(0,3),(1729,4)]:
    p = MixedstatePtychography(energy=200e3, num_probes=k, object_type='complex', verbose=False, semiangle_cutoff=18.0, rolloff=2.0)
    p._region_of_interest_shape = (n,n); p._crop_mask = None; p._crop_mask_shape = None; p._resample_exit_waves = False
    probe_obj = ComplexProbe(energy=200e3, gpts=(n,n), sampling=sampling, semiangle_cutoff=18, rolloff=2, defocus=70, C12=25, phi12=0.31, C30=2e4)
    probe_obj.build()
    np.random.seed(seed)
    states, _ = p._initialize_probe(probe_obj, None, 777.0, 18.0, False)
    np.random.seed(seed)
    draws = [float(np.random.rand()) for _ in range(2*(k-1))]
    inits.append(dict(seed=seed, count=k, first=cplx(probe_obj._array), meanIntensity=777.0, states=cplx(states), draws=draws))
out['mixed']['inits'] = inits

# ---------------- multislice ----------------
thick = np.array([55.0, 80.0], dtype=np.float64)
dummy = MultislicePtychography(energy=200e3, num_slices=3, slice_thicknesses=thick, object_type='complex', verbose=False)
props = dummy._precompute_propagator_arrays((n,n), sampling, 200e3, np.array(thick))
out['multislice'] = dict(sliceThicknesses=thick.tolist(), propagators=cplx(props))
true_slices = np.stack([true_obj] + [((0.95+0.02*np.cos(r*0.17+k))*np.exp(1j*(0.12*np.sin(c*0.31+k)+0.1*np.cos(r*0.23)))).astype(np.complex64) for k in (1,2)])
start_slices = np.stack([start_obj] + [(0.98*np.exp(1j*rng.normal(0,0.02,(side,side)))).astype(np.complex64) for _ in (1,2)])
inject(dummy, np.zeros((len(positions),n,n),np.float32), base.copy(), start_slices)
dummy._object_shape = np.array((side,side))
dummy._num_slices = 3
dummy._propagator_arrays = props
dummy._object_type = 'complex'
obj_for_forward = true_slices
shifted_probes_in = dummy._return_shifted_probes(base.copy(), positions-np.round(positions))
_, _, overlap = dummy._overlap_projection(true_slices, rows_idx, cols_idx, shifted_probes_in)
amps_ms = np.abs(np.fft.fft2(overlap)).astype(np.float32)
start_probe = (0.93*fft_shift(base, np.array([0.5,-0.4]), np)).astype(np.complex64)
out['multislice'].update(amplitudes=flat(amps_ms), startSlices=cplx(start_slices), startProbe=cplx(start_probe))
ms_cases = []
for name, ns, th, kw in [
    ('three', 3, [55.0, 80.0], dict(fix_probe_com=False)),
    ('threeConstraints', 3, [55.0, 80.0], dict(fix_probe_com=True, constrain_probe_amplitude=True,
                                               constrain_probe_amplitude_relative_radius=0.4, constrain_probe_amplitude_relative_width=0.1)),
    ('fixedProbe', 3, [55.0, 80.0], dict(fix_probe_com=False, fix_probe=True)),
    ('purePhase', 3, [55.0, 80.0], dict(fix_probe_com=False, pure_phase_object=True)),
    ('identicalSlices', 3, [55.0, 80.0], dict(fix_probe_com=False, identical_slices=True)),
    ('two', 2, [90.0], dict(fix_probe_com=False)),
]:
    p = MultislicePtychography(energy=200e3, num_slices=ns, slice_thicknesses=np.array(th), object_type='complex', verbose=False)
    inject(p, amps_ms, start_probe, start_slices[:ns])
    p._object_shape = np.array((side,side))
    p._propagator_arrays = p._precompute_propagator_arrays((n,n), sampling, 200e3, np.array(th))
    np.random.seed(1729)
    call = dict(common); call.update(tv_denoise_chambolle=False, kz_regularization_filter=False)
    p.reconstruct(num_iter=3, step_size=0.35, normalization_min=0.7, **kw, **call)
    ms_cases.append(dict(name=name, slices=ns, thicknesses=th, fixProbeCenterOfMass=kw.get('fix_probe_com'),
                         constrainProbeAmplitude=kw.get('constrain_probe_amplitude', False),
                         radius=kw.get('constrain_probe_amplitude_relative_radius', 0.5), width=kw.get('constrain_probe_amplitude_relative_width', 0.05),
                         fixProbe=kw.get('fix_probe', False), purePhase=kw.get('pure_phase_object', False), identicalSlices=kw.get('identical_slices', False),
                         object=cplx(p._object), probe=cplx(p._probe), errors=p.error_iterations))
out['multislice']['cases'] = ms_cases
dest = KIT/'Tests/CuboPtychoTests/Fixtures/engines.json'
dest.write_text(json.dumps(out,separators=(',',':'))+'\n')
print('fixture bytes', dest.stat().st_size)
