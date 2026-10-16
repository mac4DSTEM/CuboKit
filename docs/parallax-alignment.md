# Parallax cross-correlation alignment

`CuboParallax` is a separate product depending only on `CuboCore`'s package-scoped Double FFT.
The README assigns `CuboPhase` to DPC; a separate product keeps the parallax API independently importable.
This slice ports the correlation, matrix-DFT refinement, periodic Fourier shifts and fixed/running-reference
stack alignment numerics. Detector grouping, coarse-to-fine schedules, reciprocal-vector least-squares
regularisation, masks, aberrations and full `Parallax.reconstruct` remain future work.

The extraction reads mac4DSTEM `c809679c8ee23a355da6af4f63665aaf64f90d0d`,
`Core/Compute/MatrixDFTCorrelation.swift` and `Core/Analysis/ParallaxAlignment.swift`, via Git.
Its Float operations become Double, reusing CuboCore's Accelerate/Bluestein FFT for arbitrary dimensions.
There are no app, UI or Metal types. The numerical oracle is py4DSTEM
`f050d207851f8accd3aa550365baaf1f38230428` (0.14.19): `align_images_fourier`,
`upsampled_correlation`, `dftUpsample`, `get_shifted_ar` and the running-reference formula in `Parallax.reconstruct`.

The spectrum is `FFT(reference) * conj(FFT(moving))`, without phase-only magnitude normalisation.
An integer maximum is polished parabolically, rounded to a half pixel with ties to even, then refined
on a `ceil(1.5 * factor)` matrix-DFT patch with a final parabolic polish. Frequencies match NumPy's
`fftfreq`, including the positive middle frequency on odd axes. The returned translation is applied to
the moving image and wrapped into `[-size/2, size/2)`. Both polish offsets are zero if the patch maximum
lies on an edge, matching Python's single try/except. Nonfinite offsets on flat peaks use zero as in the
source extraction; constant images are refused because they cannot identify a translation.

The default upsampling factor is 8; supported factors are 3...64 to bound the quadratic patch allocation.
Images must be finite, at least three pixels on each axis, with checked dimensions and lengths.
Nonfinite intermediate/output values throw. Inputs large enough to exhaust available RAM are not covered
by a streaming or memory-budget API in this slice.

The running-reference option deliberately uses complex shifted spectra: its first update replaces the
supplied reference, and later updates average shifted spectra. It does not Fourier-transform the returned
real image to reconstruct that reference, since doing so would lose imaginary Nyquist components.
Fractional shifting uses the real part of the inverse FFT, as Python does; repeated fractional shifts
of even-sized real images can therefore lose their Nyquist component.

## Pinned tests

`tools/parallax/pins.py` checks the reference checkout commit and byte-compares the installed oracle source
files to it. Versions are enforced and recorded in `Tests/CuboParallaxTests/Fixtures/alignment.json`:
Python 3.12.4, NumPy 2.5.0, SciPy 1.18.0, py4DSTEM 0.14.19. Regenerate from the package root:

```sh
../tools/memcap.sh -- ../References/CuboKit/py4dstem-venv/bin/python -I -B tools/parallax/pins.py
```

Four small three-frame stacks cover integer/wrapped shifts (8 × 10), fractional shifts on odd axes
(9 × 11), noisy fractional shifts with a running reference (10 × 12), and a public DM4 diffraction crop
(12 × 14). The latter is `small_datacube.dm4[0,0,250:262,249:263]`; its source SHA-256 is in the fixture.
Only that public test-data crop is committed; no owner's data is used. Seed: 20261016.
Additional shift pins cover factors 3, 7, 16 and 64 for the first frame of each stack.

Shifts, shifted pixels, aligned means and matrix-DFT patches are compared against direct Python outputs
with normwise relative tolerance `1e-10`, using `max(1, max(abs(expected)))` as the scale.
No bit-for-bit claim is made across FFT/BLAS implementations. Separate planted checks verify the sign and
wrapping: integer inverse shifts within that same tolerance; smooth fractional inverse shifts within
`5e-4` using the same scale (the factor-8 estimator has finite interpolation accuracy).
