"""Pinned py4DSTEM CPU single-slice oracle, no app transcription.
Run from the worktree: ../../tools/memcap.sh -- ../../References/CuboKit/py4dstem-venv/bin/python -I -B tools/ptycho/pins.py
Only preprocessed numerical arrays cross the API; no calibration/state/UI is ported.
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM.process.phase import SingleslicePtychography
from py4DSTEM.process.phase.utils import ComplexProbe, electron_wavelength_angstrom
KIT = Path(__file__).resolve().parents[2]
REF = KIT.parents[1] / 'References/CuboKit/py4DSTEM-f050d207'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'], text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.0'
assert platform.python_version() == '3.12.4'
for name in ('singleslice_ptychography.py','ptychographic_methods.py','utils.py','phase_base_class.py','ptychographic_constraints.py'):
    installed = Path(py4DSTEM.__file__).parent/'process/phase'/name
    assert installed.read_bytes() == subprocess.check_output(['git','-C',str(REF),'show',f'HEAD:py4DSTEM/process/phase/{name}'])

def flat(a): return np.asarray(a).ravel().tolist()
def complex_array(a): return dict(real=flat(a.real), imaginary=flat(a.imag))
# 8x8 scan, 16x16 detector keeps the full uncompressed JSON well below 1 MB.
# Non-integer positions exercise Fourier probe translation and ties-to-even rounding.
rng = np.random.default_rng(20261016)
n, side = 16, 32
probe = ComplexProbe(energy=200e3, gpts=(n,n), sampling=(0.8,1.1), semiangle_cutoff=18,
                     rolloff=2, defocus=70, C12=25, phi12=0.31, C30=2e4).build()._array
positions = np.array([[8+2*r+0.25*(r%2),8+2*c+0.5*(c%2)] for r in range(8) for c in range(8)],np.float32)
r,c = np.indices((side,side))
obj = ((0.94+0.025*np.cos(r*0.3))*np.exp(1j*(0.15*np.sin(r*0.27)+0.2*np.cos(c*0.21)+rng.normal(0,0.003,(side,side))))).astype(np.complex64)
# Use the actual oracle forward and public reconstruct method, with numerical
# preprocessing state injected to avoid an unrelated CoM/rotation/calibration pass.
p = SingleslicePtychography(energy=200e3, object_type='complex', verbose=False)
p._region_of_interest_shape = np.array((n,n))
p._object_shape = np.array((side,side))
p._resample_exit_waves = False
rows,cols = p._extract_vectorized_patch_indices(positions)
shifted = p._return_shifted_probes(probe, positions-np.round(positions))
amplitudes = np.abs(np.fft.fft2(shifted*obj[rows,cols])).astype(np.float32)
# A fixed weak random phase avoids undefined Fourier phases of exact zeros
# in the uniform-object start; Python complex64 and Double can choose
# different phases for FFT roundoff at those zeros.
p._object = (0.98*np.exp(1j*rng.normal(0,0.02,(side,side)))).astype(np.complex64)
p._probe = probe.astype(np.complex64)
p._positions_px = positions.copy()
p._positions_px_initial = positions.copy()
p._amplitudes = amplitudes
p._num_diffraction_patterns = len(positions)
p._mean_diffraction_intensity = float(np.sum(amplitudes**2)/len(positions))
p._object_fov_mask_inverse = np.zeros((side,side),bool)
p._probe_initial_aperture = np.abs(np.fft.fft2(p._probe))
p._preprocessed = True
p._sampling = (0.8,1.1)
initial_probe = p._probe.copy()
initial_object = p._object.copy()
np.random.seed(1729)
p.reconstruct(num_iter=3, step_size=0.35, normalization_min=0.7, fix_probe_com=False,
              fix_probe=False, fix_positions=True, gaussian_filter=False, butterworth_filter=False,
              tv_denoise=False, fix_potential_baseline=False, progress_bar=False, reset=False)
out = dict(reference='py4DSTEM f050d207 / 0.14.19', numpy=np.__version__, seed=20261016,
           rows=n, columns=n, objectRows=side, objectColumns=side, positions=positions.tolist(),
           amplitudes=flat(amplitudes), initialProbe=complex_array(initial_probe), initialObject=complex_array(initial_object),
           builtProbe=complex_array(probe), wavelength=float(electron_wavelength_angstrom(200e3)),
           object=complex_array(p._object), probe=complex_array(p._probe), errors=p.error_iterations)
# Pin the two public constraint options and normalization endpoints with the same input.
variants = []
for fix, pure, minimum in [(True,False,0.0),(False,True,1.0)]:
    p._object = initial_object.copy(); p._probe = initial_probe.copy()
    p._positions_px = positions.copy()
    p.error_iterations = []
    np.random.seed(1729)
    p.reconstruct(num_iter=3, step_size=0.1, normalization_min=minimum, fix_probe=fix,
                  pure_phase_object=pure, fix_probe_com=False, fix_positions=True,
                  gaussian_filter=False, butterworth_filter=False, tv_denoise=False,
                  fix_potential_baseline=False, progress_bar=False, reset=False)
    variants.append(dict(fixProbe=fix,purePhase=pure,minimum=minimum,step=0.1,
                         object=complex_array(p._object),probe=complex_array(p._probe),errors=p.error_iterations))
out['variants'] = variants
other = ComplexProbe(energy=200e3,gpts=(7,10),sampling=(0.8,1.1),semiangle_cutoff=18,
                     rolloff=0,defocus=70,C12=25,phi12=0.31,C30=2e4).build()._array
out['rectangularProbe'] = complex_array(other)
# Probe constraints (fix_probe_com, constrain_probe_amplitude) from the same input, started from a probe that
# is off-centre by a fractional shift (py4DSTEM's own fft_shift), so the centre-of-mass constraint has work to do.
from py4DSTEM.process.phase.utils import fft_shift
offset = np.array([1.3,-0.8])
offcentre_probe = fft_shift(initial_probe, offset, np).astype(np.complex64)
cases = []
for name, com, amp, radius, width in [('com',True,False,0.5,0.05),('amplitude',False,True,0.3,0.1),
                                      ('both',True,True,0.3,0.1),('amplitudeDefaults',False,True,0.5,0.05)]:
    p._object = initial_object.copy(); p._probe = offcentre_probe.copy()
    p._positions_px = positions.copy()
    p.error_iterations = []
    np.random.seed(1729)
    p.reconstruct(num_iter=3, step_size=0.35, normalization_min=0.7, fix_probe_com=com,
                  constrain_probe_amplitude=amp, constrain_probe_amplitude_relative_radius=radius,
                  constrain_probe_amplitude_relative_width=width, fix_probe=False, fix_positions=True,
                  gaussian_filter=False, butterworth_filter=False, tv_denoise=False,
                  fix_potential_baseline=False, progress_bar=False, reset=False)
    cases.append(dict(name=name,fixProbeCenterOfMass=com,constrainProbeAmplitude=amp,radius=radius,width=width,
                      object=complex_array(p._object),probe=complex_array(p._probe),errors=p.error_iterations))
out['offcentreProbe'] = complex_array(offcentre_probe)
out['constraintCases'] = cases
(KIT/'Tests/CuboPtychoTests/Fixtures/ptycho.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
print('fixture bytes', (KIT/'Tests/CuboPtychoTests/Fixtures/ptycho.json').stat().st_size)
