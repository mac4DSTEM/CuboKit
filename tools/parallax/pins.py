"""Regenerate with the environment of tools/requirements.txt, from the clone root:
python -I -B tools/parallax/pins.py

Only public py4DSTEM test data is cropped; no private dataset is copied.
"""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import scipy
import py4DSTEM
from py4DSTEM.preprocess.utils import get_shifted_ar
from py4DSTEM.process.utils.cross_correlate import align_images_fourier
from py4DSTEM.process.utils.multicorr import dftUpsample

ROOT = Path(__file__).resolve().parents[2]
PIN = 'f050d207851f8accd3aa550365baaf1f38230428'
import sys as _sys, pathlib as _pathlib; _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # tools/pinenv.py
from pinenv import PY4DSTEM_SRC, PY4DSTEM_TEST_DATA
assert subprocess.check_output(['git', '-C', str(PY4DSTEM_SRC), 'rev-parse', 'HEAD'], text=True).strip() == PIN
for source in ['process/utils/cross_correlate.py', 'process/utils/multicorr.py', 'preprocess/utils.py']:
    installed = Path(py4DSTEM.__file__).parent / source
    pinned = PY4DSTEM_SRC / 'py4DSTEM' / source
    assert installed.read_bytes() == pinned.read_bytes(), source
assert py4DSTEM.__version__ == '0.14.19'
assert np.__version__ == '2.5.4'
assert platform.python_version() == '3.12.14'
assert scipy.__version__ == '1.18.1'

rng = np.random.default_rng(20261016)

def flat(a):
    return np.asarray(a, dtype=np.float64).ravel().tolist()

def wrap(s, shape):
    return (s + np.array(shape) / 2) % shape - np.array(shape) / 2

def case(name, reference, planted, noise=0.0, running=False, provenance=None):
    frames = [get_shifted_ar(reference, *s) + noise * rng.standard_normal(reference.shape) for s in planted]
    gref = np.fft.fft2(reference)
    shifts, aligned, patches = [], [], []
    for i, frame in enumerate(frames):
        g = np.fft.fft2(frame)
        peak = align_images_fourier(gref, g, upsample_factor=8)
        shift = wrap(peak, reference.shape)
        shifted = get_shifted_ar(frame, *shift)
        shifts.append(flat(shift))
        aligned.append(flat(shifted))
        # Independent direct pin of the small matrix-DFT patch at a noninteger grid center.
        cc = gref * g.conj()
        patches.append(flat(dftUpsample(cc.conj(), 8, np.array([3.25, -2.5]))))
        if running:
            qrow, qcol = np.meshgrid(np.fft.fftfreq(reference.shape[0]), np.fft.fftfreq(reference.shape[1]), indexing='ij')
            op = np.exp(-2j * np.pi * (shift[0] * qrow + shift[1] * qcol))
            gref = gref * i / (i + 1) + (g * op) / (i + 1)
    variants = [dict(factor=u, shift=flat(wrap(align_images_fourier(np.fft.fft2(reference), np.fft.fft2(frames[0]), u), reference.shape))) for u in [3, 7, 16, 64]]
    return dict(variants=variants, name=name, rows=reference.shape[0], columns=reference.shape[1], reference=flat(reference),
                planted=planted, frames=[flat(a) for a in frames], shifts=shifts, aligned=aligned,
                mean=flat(np.mean(aligned, axis=0)), patches=patches, running=running, provenance=provenance)

cases = []
cases.append(case('integer-even-wrap', rng.normal(size=(8, 10)), [(2, -3), (-3, 4), (0, 0)]))
# Smooth periodic signal suppresses the real-projection Nyquist ambiguity; asymmetric features identify the peak.
r, c = np.meshgrid(np.arange(9), np.arange(11), indexing='ij')
smooth = np.cos(2*np.pi*(r/9+c/11)) + .7*np.sin(2*np.pi*(2*r/9-c/11)) + .3*np.cos(2*np.pi*c/11)
cases.append(case('fractional-odd', smooth, [(.375, -1.625), (-2.25, 3.125), (4.25, -5.25)]))
cases.append(case('noisy-running', rng.normal(size=(10, 12)), [(.7, -1.3), (-2.2, .4), (1.1, 3.6)], noise=.04, running=True))
path = PY4DSTEM_TEST_DATA / 'small_datacube.dm4'
# py4DSTEM's public test datacube; first diffraction pattern, a 12x14 crop around the transmitted beam.
cube = py4DSTEM.import_file(str(path))
crop = np.array(cube.data[0, 0, 250:262, 249:263], dtype=np.float64)
cases.append(case('public-dm4-crop', crop, [(1, -2), (-.625, 1.375), (3.25, -4.5)],
                  provenance={'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'selection': '[0,0,250:262,249:263]'}))
output = dict(versions=dict(py4dstem=py4DSTEM.__version__, commit=PIN, numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version()),
              upsampleFactor=8, patchCenter=[3.25, -2.5], cases=cases)
destination = ROOT / 'Tests/CuboParallaxTests/Fixtures/alignment.json'
destination.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(f'{len(cases)} stacks -> {destination} ({destination.stat().st_size} bytes)')
