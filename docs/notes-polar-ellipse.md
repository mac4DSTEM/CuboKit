# Notes for the merge: polar cube, radial background, ellipse fits, Fourier resample (tasks 17 and 18)

Written by the session that built them; CHANGELOG.md, docs/parity.md, docs/status.md, docs/open-items.md, README.md and
Package.swift were not touched (another session merges them). Delete this file when the merge is done.

Reference: py4DSTEM 0.14.19 (`f050d207`), NumPy 2.5.4, SciPy 1.18.1, Python 3.12.14. Pins: `tools/polar-ellipse/pins.py`
(run: `tools/memcap.sh -- References/python/py4dstem/bin/python -I -B CuboKit/tools/polar-ellipse/pins.py`), fixtures
`Tests/CuboVirtualTests/Fixtures/polar-radial-pins.json` (1.0 MB) and `Tests/CuboCoreTests/Fixtures/ellipse-resample-pins.json`
(0.9 MB), planted inputs from `tools/polar-ellipse/planted.py`. Fast tests: `PolarRadialTests` (CuboVirtualTests),
`EllipseFitTests` and `FourierResampleTests` (CuboCoreTests); `swift test -c release --parallel --skip CuboKitParityTests` green
(no parity-target test was needed: nothing here uses the real-data harness).

## New public API (all additive)

- **CuboVirtual** `PolarSampling`, `PolarCube` (`init(detectorRows:detectorColumns:sampling:detectorMask:detectorPixelSize:)`,
  `radialBins`, `annularBins`, `geometry(origin:ellipse:)`, `transform(pattern:origin:ellipse:)`,
  `pattern(of:scanRow:scanColumn:calibration:)`, `transform(source:calibration:tiling:isCancelled:progress:)`,
  `placement(of:scanRow:scanColumn:)`), `PolarPattern`, `PolarCubeData`; `RadialBackground` (`profile(of:sigma:)`,
  `ellipticalRadius`, `background`, `subtracted`, `localAverage`).
- **CuboCore** `EllipseFit` (`fit1D`, `fitAmorphousRing`, `ellipseError`, `doubleSidedGaussian`, `doubleSidedGaussianFitError`,
  `constrainDegenerate`, `semiAxes`, `canonical`), `EllipseFitPlane`, `EllipseParameters` (`.calibration` feeds
  `Calibration.ellipse`), `EllipseFitResult`, `EllipseFitError`; `CubeEditing.resampleDetectorFourier`,
  `CubeEditing.fourierResample`, `CubeEditing.fourierOutput`, `FourierResampleSpecification`, `FourierResampleOptions`.
- package-only helpers in CuboCore: `MINPACKLeastSquares` (a transcription of MINPACK `lmdif`, the routine under
  `scipy.optimize.leastsq`) and `SciPyKernels` (`arange`, `mod`, `gaussian_filter1d` and 2-D, `interp1d` linear, masked median).
  `tools/consumer-check` was not extended (the merge session should add the public names above).

## CHANGELOG bullets (suggested version 0.9.0: new public API, no source change that moves an existing number)

- **CuboVirtual `PolarCube`** (py4DSTEM `DataCube.polar`, a `PolarDatacube`): the polar-elliptical transform of every pattern in Double on the CPU: radial bins (`np.arange(qmin, qmax, qstep)`), `annularCount` angular bins over 2 pi (pi with two-fold symmetry), the four-pass bilinear binning, the Gaussian along the angle that follows the arc length, normalisation by the transformed detector mask, invalid cells below `maskThreshold`, `radialPower` (`qscale`), elliptical from `Calibration.ellipse` at the scan position or circular. `transform(source:calibration:)` does the whole cube tile by tile with positions in parallel and one shared geometry when origin and ellipse are constant (1024 patterns of 256 x 256 in 0.26 s; 0.38 s with an origin per position, Apple Silicon, release). CPU only: at that speed a Metal path would be bound by the host copy of the cube, so none was written. Against py4DSTEM's own cells: 5e-16 relative (`assert_allclose` default is 1e-7). On planted data the right ellipse makes the cube constant in phi (spread 0.6% of the mean against 8.8% for a circle assumption on the `tilted` case; the ring's radial centroid moves 0.02 px over phi against 1.2 px).
- **CuboVirtual `RadialBackground`** (`get_radial_bkgrnd`, `get_radial_bksb_dp`, `get_local_ave_dp(radial_bksb=True)`): the median over the angle of the valid polar cells, smoothed along the radius, laid back on the detector as rings (linear interpolation, extrapolating); the subtracted pattern is clamped at zero; the local average uses the 3 x 3 Gaussian kernel cut at the scan edge, with an optional per-position keep mask (`radial_bksb` with `braggmask`; CuboBragg's `BraggMasking.mask` supplies it, this module cannot call it). Against py4DSTEM: background 4e-15, subtracted 2e-13, local average 4e-13 relative. Throws `CalibrationError.notSet("origin")` without an origin.
- **CuboCore `EllipseFit`** (`process/calibration/ellipse.py`): `fit1D` (`fit_ellipse_1D`), `fitAmorphousRing` (`fit_ellipse_amorphous_ring`, with the 11 fit parameters), `ellipseError`, `doubleSidedGaussian`, `doubleSidedGaussianFitError`, `constrainDegenerate` (`constrain_degenerate_ellipse`), and the conversions `semiAxes`/`canonical` (`convert_ellipse_params`, `_r`). The ellipse is (row, column, a, b, theta) with theta of the major axis from the +row axis towards the +column axis: py4DSTEM's (x0, y0, a, b, theta) numbers, so `EllipseParameters.calibration` goes into `Calibration.ellipse` unchanged. The fits run a transcription of MINPACK `lmdif` from py4DSTEM's starting guesses: the evaluation counts equal SciPy's in 4 of 6 pinned fits, the others within 10%. Against py4DSTEM: conversions and direct evaluations 1e-7; `constrain_degenerate_ellipse` to the last digit; `fit_ellipse_1D` within 7e-10 relative; the amorphous-ring ellipse within 3e-8 relative (theta 2e-7 rad), the residual sum of squares within 4e-9. Against the planted ellipses: the amorphous-ring fit to 0.01 px in a and b and 3e-4 px in the centre; the algebraic 1D fit to 1.2% in the axes (it is biased upwards by weighting the annulus by the pixel values). `fit1D`'s default `fitRadii` is `(0, min(extents) / 2)`: py4DSTEM's default raises a `TypeError`.
- **CuboCore `CubeEditing.resampleDetectorFourier`** and **`fourierResample`** (`resample_Q(method="fourier")`, `process.utils.fourier_resample`): zero-padding or cropping of the spectrum, scale (`int(extent * scale)`) or output size, `bandlimitNyquist`/`bandlimitPower`, `forceNonnegative`, `conserveSums`, odd and even extents; the Q pixel size updates as `resample_data_diffraction` does. Float32 cube out; against py4DSTEM (whose transform is single precision) 2.1e-7 of the largest value; the Double `fourierResample` reproduces the band-limited planted truth to 1.5e-14.

### Migration
None (additive).

## docs/parity.md rows to change (same columns: py4DSTEM | CuboKit | State | Mark | Tolerance)

Replace the tolerance cell with `no upstream test; np.allclose defaults` for the polar rows (as now) and for the fits `no upstream test; np.allclose defaults on the Float64 reference; iterative fits compared at the optimiser's tolerance, see notes`.

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `.polar` (polar-transformed datacube) (row 31) | `PolarCube` (`PolarSampling`, `transform`) | present (0.9.0; CPU Double; py4DSTEM's `rr` is on the scale of the minor axis, kept; theta from +row towards +column as stored in `Calibration.ellipse`) | study | no upstream test; np.allclose defaults; measured 5e-16 relative |
| `get_radial_bkgrnd` (row 102) | `RadialBackground.background` | present (0.9.0; the lookup radius is on the major-axis scale while the polar bins are on the minor-axis scale: py4DSTEM's own, so a ring is stretched by a/b for a != b; a circle is exact) | study | no upstream test; np.allclose defaults; measured 4e-15 |
| `get_radial_bksb_dp` (row 103) | `RadialBackground.subtracted` | present (0.9.0) | study | no upstream test; np.allclose defaults; measured 2e-13 |
| `get_local_ave_dp` (`radial_bksb`, `sigma`, `braggmask`) (row 104) | `BraggMasking.localAverage` and `RadialBackground.localAverage` | present (0.9.0): `RadialBackground.localAverage` is `radial_bksb=True`, and with `keeping:` (a per-position mask, e.g. from `BraggMasking.mask`) the `radial_bksb` + `braggmask` branch; plain and Bragg-masked without radial background as before | study | no upstream test; np.allclose defaults; measured 4e-13 |
| `resample_Q` (bilinear, `conserve_array_sums`) (row 88) | `CubeEditing.resampleDetector`, `CubeEditing.resampleDetectorFourier` | present (0.9.0; method="fourier": scalar scale or output size, the Q pixel size per `resample_data_diffraction`; the kit returns Float32 where py4DSTEM returns float64 unless the sums are conserved) | study | unchanged for bilinear; fourier 1e-6 of the largest value (py4DSTEM is complex64); measured 2.1e-7 |
| `fit_ellipse_1D` (row 158) | `EllipseFit.fit1D` | present (0.9.0; default `fitRadii` repaired, upstream raises `TypeError`; `ignoring:` is upstream's `mask` with the same polarity) | useful | no upstream test; np.allclose defaults on the Float64 reference; fit compared at the optimiser's tolerance: measured 7e-10 relative |
| `fit_ellipse_amorphous_ring` (row 159) | `EllipseFit.fitAmorphousRing` | present (0.9.0; `keeping:` is upstream's `mask`, `true` = use, the opposite of `fit_ellipse_1D`; `initial:` is `p0`) | useful | as row 158; measured 3e-8 relative, cost 4e-9 |
| `ellipse_err`, `double_sided_gaussian`, `double_sided_gaussian_fiterr` (row 160) | `EllipseFit.ellipseError`, `.doubleSidedGaussian`, `.doubleSidedGaussianFitError` | present (0.9.0) | study | np.allclose defaults on the Float64 reference; measured below 1e-7 (assert_allclose default) |
| `constrain_degenerate_ellipse` (row 161) | `EllipseFit.constrainDegenerate` | present (0.9.0; Double input) | useful | np.allclose defaults; measured exact |
| `process.utils.fourier_resample`, `resample_data_diffraction(method="fourier")` (row 321) | `CubeEditing.fourierResample` (2-D, Double), `CubeEditing.resampleDetectorFourier` | present (0.9.0) | study | as row 88 |
| `convert_ellipse_params`, `convert_ellipse_params_r` (not in the table; "Further API" section if desired) | `EllipseFit.semiAxes`, `EllipseFit.canonical` | present (0.9.0) | useful | np.allclose defaults; measured below 1e-7 |
| `preprocess.get_1D_polar_background`, `get_2D_polar_background` (row 310) | unchanged | missing (not this slice: a different background, from a polynomial fit on the radial integral) | useful | unchanged |
| `process.polar.*` (`calculate_radial_statistics` ...) (row 335) | unchanged | not planned: DiffractoKit | unchanged | unchanged |

Count: rows 102, 103, 158, 159, 160, 161 and 321 go from missing to present (row 321 is in the "Further API" table), row 104 from partial to present, row 31 from not planned to present (not counted); row 88 stays present. The merge session should recount.

## docs/open-items.md

Delete these three bullets under "Missing, in the order the study needs them":

- `resample_Q(method="fourier")` and `fourier_resample`.
- `get_radial_bkgrnd`, `get_radial_bksb_dp` and `get_local_ave_dp(radial_bksb=True)`: they need the polar data cube, which is DiffractoKit's `PolarDatacube`.
- `fit_ellipse_1D`, `fit_ellipse_amorphous_ring`, `constrain_degenerate_ellipse` (the ellipse fit that feeds `Calibration.ellipse`).

Add under "Decisions to keep an eye on":

- The polar cube's radius is on the minor-axis scale (`rr = (b / a) * hypot`), the radial background's lookup radius on the major-axis scale. Both are py4DSTEM's and are ported as they are; for `a != b` the background profile is stretched by `a / b` against the polar bins. A pull request upstream (or a deliberate deviation here) is Paul's call; `testRadialBackgroundOfACircle` shows the circle agreeing with the planted exponential to 0.2%.
- The amorphous-ring fit is chaotic at rounding level in its poorly determined parameters (py4DSTEM's own result moves by up to 7e-4 in the axes when the start changes by one ulp on the `tilted` case); tests compare ellipses at 1e-6 and the residual sum of squares at 1e-6 relative, not parameter by parameter.
- `PolarCube` keeps the whole cube in Double when asked for (`PolarCubeData`: scan x 180 x ~180 doubles per pattern); a lazy `CubeSource` adapter or a Float32 variant is for the consumer that needs it.

## DiffractoKit follow-up (one line)

Consume the polar cube (`PolarCube`, `PolarCubeData`) and `EllipseFit` for the ring and ellipse work; no code there yet.

## Deviations and what a reader should know

1. **Convention**: the kit's ellipse is (row, column, a, b, theta) = py4DSTEM's (x0, y0, a, b, theta) with x the row: no conversion between the kit and py4DSTEM. `planted.py` measures theta from +column towards +row; the conversion lives in the tests only: `theta_py4DSTEM = pi/2 - theta_planted` (mod pi). Ellipses are equal modulo pi in theta; fits return theta in (-pi, pi] as py4DSTEM does.
2. **Radius scales**: see open-items above. Consequence for `planted.py`'s README: the polar truth there puts the ring at `q = a * rho`; py4DSTEM's transform puts it at `b * rho`, so the ring-position test checks bin `b` (11.97 against b = 12) instead of `a`.
3. `fit_ellipse_1D` default `fitradii` repaired (upstream `TypeError`). A non-finite fit throws `EllipseFitError.notFinite` where py4DSTEM returns NaN; a fit that did not converge (MINPACK info 5 to 8) is returned with `converged == false`.
4. `PolarCube` reads the ellipse per scan position (py4DSTEM's `get_ellipse()` takes the stored value whole and fails on an R-shaped field).
5. `RadialBackground.localAverage` clips and renormalises the kernel at the edge for any scan, including a scan one position wide, where py4DSTEM's edge branches fail.
6. `resampleDetectorFourier`: Float32 cube in a `MemoryCube`; py4DSTEM's result is float64 unless the sums are conserved (NumPy 2: float32 array times a `np.float64` scalar). Single precision on both sides: 2.1e-7 of the largest value; the Double `fourierResample` is exact on band-limited data (1.5e-14). The Nyquist row/column of an even extent goes to the negative frequencies (neither split nor doubled), as upstream; odd extents follow the same index rule.
7. `constrainDegenerate` takes Double; py4DSTEM's `gaussian_filter` of a Float32 plane rounds each pass to Float32, which can move the argmax of a flat peak by one pixel (not tested: the pins use Float64).
8. The amorphous-ring fit's guess of the ring radius uses py4DSTEM's `radial_integral` (polar-elliptical transform with a = b = 1, 2 degree bins, masked sums); ported operation by operation (`EllipseFit.radialIntegral`, internal).
9. Tolerances: direct functions and the polar numbers at `assert_allclose` Float64 (rtol 1e-7, atol 1e-12), measured 5e-16 to 4e-13. Iterative fits are compared with the stated reasons in `EllipseFitTests.check`. Nothing was loosened to pass: the first comparison of the fits at 1.49e-8 (the optimiser's xtol) was replaced after measuring that py4DSTEM's own result moves by more than that for a one-ulp change of its start.

## Measured agreement (this Mac, release)

| What | Against | Measured |
|---|---|---|
| polar cells (5 sampling configurations, 11 patterns) | py4DSTEM `PolarDatacube` | 5.2e-16 relative, validity identical, coverage 2.4e-16 |
| radial background, subtracted pattern, local average | py4DSTEM | 4.1e-15, 2.4e-13, 4.1e-13 |
| ring centroid over phi, right ellipse vs circle (planted `tilted`) | planted | 0.02 px vs 1.2 px |
| `fit_ellipse_1D` (3 shapes x 4 variants) | py4DSTEM | 6.6e-10 relative (canonical form held to 1e-5 of its largest element: B of an untilted ellipse is the fit's own noise) |
| `fit_ellipse_amorphous_ring` | py4DSTEM | 2.7e-8 relative (tilted), 5e-9 (flat), 6e-13 (near circle); cost within 4e-9 |
| `constrain_degenerate_ellipse` (2 cases) | py4DSTEM | identical to the last digit |
| `fourier_resample` (12 cube cases, 3 odd, 1 two-dimensional) | py4DSTEM (complex64) | 2.1e-7 of the largest value |
| `fourier_resample` of band-limited data | planted truth | 1.5e-14 absolute |
| fitted ellipse stored in `Calibration.ellipse`, polar cube constant in phi (planted `tilted`) | planted | spread 0.63% (true ellipse 0.63%, circle 8.8%) |

Unfinished: nothing in the two tasks; `tools/consumer-check` and the DocC pages were not touched (merge session), and no Metal path exists (CPU only,
measured speed above).
