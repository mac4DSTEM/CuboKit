"""Pinned oracle for the single-slice options the mac4DSTEM app uses (CuboPtycho 0.6.0 candidates).
Run from the worktree root:
  MPLCONFIGDIR="$PWD/.build/matplotlib" ../../tools/memcap.sh -- References/py4dstem-venv/bin/python -I -B tools/ptycho/options_pins.py
Needs Tests/CuboPtychoTests/Fixtures/ptycho.json (pins.py) as its input data; writes Fixtures/options.json.

1. constrainObjectAmplitude = false (the app's toggle). py4DSTEM has no switch: its _object_threshold_constraint is unconditional.
   The oracle is py4DSTEM's own reconstruct with that one method replaced by "unit amplitude if pure_phase_object else identity",
   which is what the app does when the toggle is off (the app skips the cap unless purePhaseObject).
2. Higher-order probe aberrations: ComplexProbe C21/phi21 ... C56/phi56 (py4DSTEM's set, orders 2 to 5), alone and combined,
   through PtychographyProbe.build and through the starting probe of _initialize_probe (PtychographyPreparation.initialProbe).
3. Terms py4DSTEM does not have (radial orders other than its set, e.g. (1,1), (6,0), sine components alone): a float64 NumPy
   transcription of the app's PtychographyProbeAberrations.chi (mac4DSTEM Core/Analysis/PtychographyPreparation.swift).
"""
from pathlib import Path
import json, subprocess, platform
import numpy as np
import py4DSTEM
from py4DSTEM.process.phase import SingleslicePtychography
from py4DSTEM.process.phase.utils import ComplexProbe, electron_wavelength_angstrom
KIT = Path(__file__).resolve().parents[2]
REF = KIT/'References/py4DSTEM-f050d207'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'], text=True).strip().startswith('f050d207')
assert py4DSTEM.__version__ == '0.14.19' and np.__version__ == '2.5.0' and platform.python_version() == '3.12.4'
for name in ('singleslice_ptychography.py','ptychographic_methods.py','utils.py','phase_base_class.py','ptychographic_constraints.py'):
    installed = Path(py4DSTEM.__file__).parent/'process/phase'/name
    assert installed.read_bytes() == subprocess.check_output(['git','-C',str(REF),'show',f'HEAD:py4DSTEM/process/phase/{name}'])
def flat(a): return [float('%.9g' % v) for v in np.asarray(a, dtype=np.float64).ravel()]
def cplx(a): a = np.asarray(a); return dict(real=flat(a.real), imaginary=flat(a.imag))
src = json.loads((KIT/'Tests/CuboPtychoTests/Fixtures/ptycho.json').read_text())
n, side = src['rows'], src['objectRows']
def arr(d, shape): return (np.array(d['real']) + 1j*np.array(d['imaginary'])).reshape(shape)
positions = np.array(src['positions'], np.float32)
amplitudes = np.array(src['amplitudes'], np.float32).reshape(len(positions), n, n)
mean_intensity = src['preprocessing']['meanIntensity']
initial_probe = arr(src['initialProbe'], (n,n)).astype(np.complex64)
initial_object = arr(src['initialObject'], (side,side)).astype(np.complex64)
out = dict(reference='py4DSTEM f050d207 / 0.14.19', numpy=np.__version__)

# ---------------- 1. object amplitude toggle ----------------
p = SingleslicePtychography(energy=200e3, object_type='complex', verbose=False)
p._region_of_interest_shape = np.array((n,n)); p._object_shape = np.array((side,side)); p._resample_exit_waves = False
p._amplitudes = amplitudes; p._num_diffraction_patterns = len(positions); p._mean_diffraction_intensity = mean_intensity
p._object_fov_mask_inverse = np.zeros((side,side),bool); p._probe_initial_aperture = np.abs(np.fft.fft2(initial_probe))
p._preprocessed = True; p._sampling = (0.8,1.1)
def no_cap(current_object, pure_phase_object):
    return np.exp(1j*np.angle(current_object)) if pure_phase_object else current_object
p._object_threshold_constraint = no_cap
# The data come from an object of amplitude about 0.95, so a start above 1 (here 1.1) is what lets the cap act.
start_object = (initial_object * np.float32(1.1/0.98)).astype(np.complex64)
cases = []
for name, fix, pure, step, iters in [('off',False,False,0.5,4),('offFixedProbe',True,False,0.5,4),('offPurePhase',False,True,0.35,3)]:
    p._object = start_object.copy(); p._probe = initial_probe.copy()
    p._positions_px = positions.copy(); p._positions_px_initial = positions.copy(); p.error_iterations = []
    np.random.seed(1729)
    p.reconstruct(num_iter=iters, step_size=step, normalization_min=0.7, fix_probe=fix, pure_phase_object=pure, fix_probe_com=False,
                  fix_positions=True, gaussian_filter=False, butterworth_filter=False, tv_denoise=False,
                  fix_potential_baseline=False, progress_bar=False, reset=False)
    amp = np.abs(p._object)
    if not pure: assert amp.max() > 1.002, amp.max()      # the cap would have changed the result
    cases.append(dict(name=name, fixProbe=fix, purePhase=pure, step=step, iterations=iters, maxAmplitude=float(amp.max()),
                      object=cplx(p._object), probe=cplx(p._probe), errors=p.error_iterations))
out['objectAmplitude'] = dict(startObject=cplx(start_object), cases=cases)

# ---------------- 2. higher-order terms against ComplexProbe ----------------
wl = float(electron_wavelength_angstrom(200e3))
GPTS, SAMPLING = (20,16), (0.8,1.1)
def polar_kwargs(terms): 
    kw = {}
    for t in terms:
        kw[f"C{t['n']}{t['m']}"] = t['magnitude']
        if t['m'] > 0: kw[f"phi{t['n']}{t['m']}"] = t['azimuth']
    return kw
def T(n, m, mag, az=0.0): return dict(n=n, m=m, magnitude=mag, azimuth=az)
groups = {
  'C21': [T(2,1,1500,0.4)], 'C23': [T(2,3,2500,0.2)],
  'C32': [T(3,2,1.5e5,-0.3)], 'C34': [T(3,4,2e5,0.15)],
  'C41': [T(4,1,3e6,1.1)], 'C43': [T(4,3,3e6,-0.5)], 'C45': [T(4,5,3e6,0.12)],
  'C50': [T(5,0,1e8)], 'C52': [T(5,2,1e8,0.7)], 'C54': [T(5,4,1e8,-0.2)], 'C56': [T(5,6,1e8,0.05)],
  'combined': [T(2,1,800,0.4), T(2,3,1200,-0.6), T(3,2,8e4,0.3), T(3,4,1e5,0.9), T(4,1,1.5e6,-1.0), T(4,5,2e6,0.4),
               T(5,0,5e7), T(5,2,5e7,0.1), T(5,6,5e7,-0.35)],
}
probe_cases = []
for name, terms in groups.items():
    kw = polar_kwargs(terms)
    base = dict(semiangle_cutoff=18, rolloff=2, defocus=70, C12=25, phi12=0.31, C30=2e4) if name == 'combined' else dict(semiangle_cutoff=18, rolloff=2)
    cp = ComplexProbe(energy=200e3, gpts=GPTS, sampling=SAMPLING, **base, **kw)
    alpha, phi = cp.get_scattering_angles()
    chi = cp.evaluate_chi(alpha, phi)
    assert 0.05 < np.abs(chi[alpha < 0.018]).max() < 8, (name, np.abs(chi[alpha < 0.018]).max())
    probe_cases.append(dict(name=name, terms=terms, defocus=base.get('defocus',0.0), astigmatism=base.get('C12',0.0),
                            astigmatismAngle=base.get('phi12',0.0), sphericalAberration=base.get('C30',0.0), probe=cplx(cp.build()._array)))
out['probes'] = dict(rows=GPTS[0], columns=GPTS[1], rowSampling=SAMPLING[0], columnSampling=SAMPLING[1], wavelength=wl, cutoff=0.018, rolloff=0.002, cases=probe_cases)

# the starting probe of the engine: py4DSTEM _initialize_probe (scaled to the mean intensity) with the combined terms
kw = dict(semiangle_cutoff=18.0, rolloff=2.0, defocus=70.0, C12=25.0, phi12=0.31, C30=2e4, **polar_kwargs(groups['combined']))
q = SingleslicePtychography(energy=200e3, verbose=False, object_type='complex', **kw)
dk = (0.7, 0.9)
q._angular_sampling = dk; q._amplitudes_shape = np.array(GPTS); q._region_of_interest_shape = np.array(GPTS)
q._crop_mask = None; q._crop_mask_shape = None; q._resample_exit_waves = False
probe, cutoff = q._initialize_probe(None, None, 1234.5, 18.0, False)
sampling_row = 1.0 / (GPTS[0] * dk[0] / 1e3 / wl)
out['initialProbe'] = dict(rows=GPTS[0], columns=GPTS[1], angularSampling=list(dk), meanIntensity=1234.5, semiangle=18.0, rolloff=2.0,
                           defocus=70.0, astigmatism=25.0, astigmatismAngle=0.31, sphericalAberration=2e4,
                           terms=groups['combined'], samplingRow=float(q.sampling[0]), samplingColumn=float(q.sampling[1]), wavelength=wl,
                           real=flat(probe.real), imaginary=flat(probe.imag))

# ---------------- 3. app transcription (float64) ----------------
def app_chi(alpha, theta, defocus, c12a, c12b, higher):
    ph = 0.5*alpha**2*(-defocus + c12a*np.cos(2*theta) + c12b*np.sin(2*theta))
    for (r, a, comp, c) in higher:
        rad = alpha**(r+1)/(r+1)
        if a == 0: ph = ph + c*rad
        elif comp == 0: ph = ph + c*rad*np.cos(a*theta)
        else: ph = ph + c*rad*np.sin(a*theta)
    return 2*np.pi/wl*ph
def app_probe(rows, cols, rs, cs, cutoff, rolloff, defocus, c12a, c12b, higher):
    qr = np.fft.fftfreq(rows)[:,None]/rs; qc = np.fft.fftfreq(cols)[None,:]/cs
    alpha = np.hypot(qr, qc)*wl; theta = np.arctan2(qc+0*qr, qr+0*qc)
    ap = np.where(alpha > cutoff-rolloff, np.where(alpha > cutoff, 0.0, 0.5*(1+np.cos(np.pi*(alpha-cutoff+rolloff)/rolloff))), 1.0)
    a = np.fft.ifft2(ap*np.exp(-1j*app_chi(alpha, theta, defocus, c12a, c12b, higher)))
    return a/np.sqrt((np.abs(a)**2).sum())
app_cases = []
for name, df, ca, cb, higher in [
    ('shiftAndSine', 0.0, 0.0, 0.0, [(1,1,0,300.0),(2,2,1,900.0),(3,3,1,4e4)]),
    ('sixthOrder', 55.0, 18.0, -9.0, [(6,0,0,3e10),(6,4,1,2e10),(4,2,0,1e6),(5,1,1,5e7)]),
    ('mixedWithPolarRange', 70.0, 20.0, 12.0, [(2,1,0,700.0),(2,1,1,-400.0),(3,2,0,6e4),(3,2,1,-3e4),(4,5,1,2e6)]),
]:
    pr = app_probe(GPTS[0], GPTS[1], SAMPLING[0], SAMPLING[1], 0.018, 0.002, df, ca, cb, higher)
    app_cases.append(dict(name=name, defocus=df, c12a=ca, c12b=cb,
                          terms=[dict(radialOrder=r, angularOrder=a, sine=bool(c), coefficient=v) for (r,a,c,v) in higher], probe=cplx(pr)))
out['appTranscription'] = app_cases
dest = KIT/'Tests/CuboPtychoTests/Fixtures/options.json'
dest.write_text(json.dumps(out,separators=(',',':'))+'\n')
print('fixture bytes', dest.stat().st_size)
