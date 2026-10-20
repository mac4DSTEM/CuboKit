# Parallax cross-correlation alignment

`CuboParallax` is a separate product depending only on `CuboCore`'s package-scoped Double FFT.
The README assigns `CuboPhase` to DPC; a separate product keeps the parallax API independently importable.
This slice ports the correlation, matrix-DFT refinement, periodic Fourier shifts and fixed/running-reference
stack alignment numerics; the detector grouping, coarse-to-fine schedule, shift-basis regularisation and masks of
`Parallax.reconstruct` are ported in `ParallaxReconstruction` (last section).

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

## The reconstruct call (`ParallaxReconstruction`)

`ParallaxReconstruction.reconstruct(_:options:)` ports py4DSTEM `Parallax.reconstruct` (`process/phase/parallax.py`,
pin `f050d207`); `ParallaxReconstruction.process(_:shape:reciprocalSampling:scanSampling:energyEV:options:)` chains
`ParallaxPreprocessing.prepare`, `reconstruct` and, if asked, `ParallaxSubpixel.kernelDensity` or
`correctPositions` (with the same arguments the single steps take) and returns every product
(`ParallaxProcessingResult`: the preprocessing, the reconstruction, the wavelength, the upsampling limits, the upsampled
image and the position correction). Aberration fitting and correction and depth sectioning stay separate calls that take
`result.reconstruction.alignedStack`, `shifts` and `result.preprocessing.probeAngles`.

`ParallaxPreprocessingResult` now also carries `scanRows`, `scanColumns` and `paddedWindow` (py4DSTEM `_window_pad`,
the window in the padded frame). `correctPositions(window:)` expects `paddedWindow`, not the unpadded `window` (its
comment said so wrongly before).

Schedule and update per bin, as in the source:

- `binSchedule(detectorIndices:options:)`: the bright-field diameter in detector pixels is
  `max(rowRange, columnRange) + 1`; `maxAlignmentBin` is clipped to it; bins are `2^arange(ceil(log(min)/log 2),
  ceil(log(max)/log 2))` from the coarsest down, the last repeated `numIterAtMinBin - 1` times; `alignmentBinValues`
  replaces the schedule (clipped to `1...max`). Defaults are py4DSTEM's (`min 1`, `numIterAtMinBin 2`,
  `centeredAlignmentBins false`, `crossCorrelationUpsampleFactor 8`, `regularizerMatrixSize (1, 1)`,
  `regularizeShifts false`, `runningAverage true`).
- Detector pixels are grouped by `round((index - median(index)) / bin + 0.5)` (`+ 0` when centred; ties to even),
  the groups are visited by radius squared from the centre, ties in lexicographic order; the mean image of a group is
  correlated against the reference (`ParallaxAlignment` spectrum, matrix-DFT refinement and wrap, unchanged). With
  `runningAverage` the reference becomes the average of the shifted group spectra (the first replaces it), exactly as
  `ParallaxAlignment.align`.
- The updated shifts (`shifts + update`) are projected on the shift basis, by least squares: the reciprocal vectors
  (default, no offset term), or with `regularizeShifts` the Bernstein tensor basis of degrees `regularizerMatrixSize`
  on `u = k * 0.5 / max|k| + 0.5`. The fit is the orthogonal projection onto the basis' column space (what
  `numpy.linalg.lstsq(rcond=None)` returns as `basis @ coefs`); columns below `rows * Float.ulpOfOne` of the largest
  column norm are dropped as rank-deficient.
- The stack and the masks are Fourier-shifted by the fitted update, the accumulated shifts are centred by subtracting
  their rounded median (ties to even) and rolling stack and masks by it, and the reconstruction
  `(mean * (1 - clip(m)) + mean(stack * mask)) / (m + 1 - clip(m))`, `m = mean(mask)`, and the error
  `sum(|stack - image| * mask) / (sum(window) * count) / stack mean` are updated. `stack mean` is that of the
  preprocessed stack and does not change.
- Results: `alignedStack`, `stackMask`, `shifts` (row, column; py4DSTEM `_xy_shifts`, the shifts `subpixel_alignment`
  applies to the retained `unshiftedStack`), `image`, `mask`, `errors` (one per bin) and `binValues`.

Named option (cheap, from the source application): `emptySchedule = .singlePassAtMinimumBin`. When the bright-field
disk is one pixel wide, no binning is possible; py4DSTEM raises `IndexError` from `bin_vals[-1]` (here
`ParallaxError.invalidInput`, the default) when `numIterAtMinBin > 1`, and otherwise runs no pass; the application
runs one pass at the minimum bin. Its coarse-to-fine bin rules are otherwise py4DSTEM's; the application's level
by level driver (progress, cancellation, memory budget) is not part of the kit.

### Pins and measured gaps

`tools/parallax/reconstruct_pins.py` runs the real `Parallax(...).preprocess().reconstruct()` (and, for the end-to-end
cases, `subpixel_alignment`) on small synthetic cubes and writes `Tests/CuboParallaxTests/Fixtures/reconstruct.json`
(about 220 KB, versions recorded). Each cube is one smooth random object seen by every bright-field pixel at a shift
linear in the pixel offset, so correlation maxima are unique and float32 cannot flip a tie. Run it from the clone root
as the other pin scripts. The pinned py4DSTEM `reconstruct` ends each bin with `float(<one-element array>)`, which NumPy
2.5.0 (the pinned venv) refuses; the script makes `np.atleast_1d` return its 0-d input for the duration of the call,
which changes no arithmetic. Bright-field disks are small (16 detector pixels, 2.4 px or 2.2 px radius): NumPy's
`argsort` is a stable insertion sort only up to 16 elements, and the script asserts for every bin that the default
`argsort` of the group radii equals a stable sort. With more than 16 groups per bin and equal radii the order, and so
the running reference, can differ from the kit's (lexicographic); with `runningAverage: false` the order never matters.

| Case | Options | Bins | Shifts (px) | Image | Mask | Errors | Stack |
|---|---|---|---|---|---|---|---|
| default | py4DSTEM defaults | 2, 1, 1 | 1.1e-7 | 2.7e-7 | 1.3e-7 | 2.0e-8 | 5.4e-7 |
| regularized-2x1 | Bernstein (2, 1) | 2, 1, 1 | 1.9e-7 | 2.4e-7 | 2.2e-7 | 1.8e-8 | stack norms 2.7e-7 |
| no-running-centered | running average off, centred bins | 2, 1, 1 | 1.1e-7 | 3.6e-7 | 1.9e-7 | 3.4e-8 | 4.4e-7 |
| explicit-bins-nonsquare | 6 x 9 scan, 8 x 9 detector, bins 3, 2, 1, factor 16, Bernstein (1, 1) | 3, 2, 1 | 9.0e-7 | 5.3e-7 | 6.2e-7 | 4.8e-8 | stack norms 3.3e-7 |
| capped-single-pass | `maxAlignmentBin 2`, `numIterAtMinBin 1`, factor 4 | 1 | 1.7e-7 | 2.3e-7 | 1.4e-7 | 5.8e-9 | stack norms 8.0e-8 |

Gaps are the largest absolute difference over `max(1, max|expected|)`; the bin schedule, the detector pixels and
the number of errors are exact. Written tolerances: shifts 5e-6 px, image, mask, stack and stack norms and mask sum
2e-6, errors 5e-7. The reference works in float32 (eps 1.2e-7) through the shifts, so these are float32 gaps; none is
widened. End to end (`process`, two cases at 0.2 inverse angstrom per pixel, default factor heuristic with limits 1.06 and
2.0 and factor 1.59, and factor 2 with sigma 0.3 and two checkerboard position-correction iterations): shifts 1.1e-7,
upsampled image 2.0e-7 and 1.8e-7, probe offsets 4e-8; written 5e-6 for each, and the limits within 1e-6.

Planted truth (`plantedIntegerShiftsAreRecovered`): nine images that are integer circular shifts of one 11 x 13
random image (content moved by `(a, 2b)` for detector offset `(a, b)`) are aligned to the planted inverse shifts within
1e-6 px and onto the original image within 1e-6, error below 1e-10. It needs 30 passes at the finest bin, because
each pass reduces the offset by about 0.37 (a damped iteration, as in the source), and odd extents, because even extents
lose their Nyquist component in the first fractional shift (the effect noted above), which the pinned cases show too.
`endToEndEqualsChainedSteps` checks that `process` equals the chained single steps exactly (stack, shifts, image,
errors, limits, upsampled image and probe offsets, with and without position correction and without the subpixel stage).

### Not provided

Initial aberration shifts (`Parallax.preprocess` with polar parameters) and the normalisation orders above 0 and the
real-space mask of `preprocess` (only the constant-order path of `ParallaxPreprocessing`); `aligned_bf_image_guess`;
continuing or resetting a previous reconstruction (`reset`, the warning, `error_iterations` across calls); batching
(`max_batch_size`), GPU and plotting; `virtual_detector_mask`, `integer_pixel_rolling_alignment` and the Fourier
upsampling of `subpixel_alignment`; and cross-correlation upsampling factors outside 3...64. The `lstsq` cut for a
rank-deficient basis is a Gram-Schmidt cut at `count * Float.ulpOfOne` of the largest column norm, not NumPy's SVD
`rcond`; the fitted shifts agree for every full-rank basis (all pinned cases).
