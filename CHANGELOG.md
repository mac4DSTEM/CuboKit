# Changelog

## [0.10.0](https://github.com/mac4DSTEM/CuboKit/compare/0.9.0...0.10.0) (2026-11-01)


### ⚠ BREAKING CHANGES

* CuboPtycho, CuboStrain and CuboEmbedding (the enums) and their deprecated scope are removed; nothing replaces them. Module-qualified names such as CuboStrain.LatticeStrain now resolve. mac4DSTEM uses none of the three enums or scope, so it needs no call-site change.

### Features

* Bragg disk detection .single runs on every detector size ([#61](https://github.com/mac4DSTEM/CuboKit/issues/61)) ([f60ef41](https://github.com/mac4DSTEM/CuboKit/commit/f60ef412fd66104bef9e004fad70c9a058250cc9))
* brightest-disk seed for the iterated-window origin at low dose ([#65](https://github.com/mac4DSTEM/CuboKit/issues/65)) ([d4231bc](https://github.com/mac4DSTEM/CuboKit/commit/d4231bc9a06573e050bd4a58547f12398f8c2dad))
* deprecate the placeholder scope constants and drop dead code ([#41](https://github.com/mac4DSTEM/CuboKit/issues/41)) ([f33b33e](https://github.com/mac4DSTEM/CuboKit/commit/f33b33ea71f8b8bdf66aba2843b860e36df812e2))
* deprecate the placeholder scope constants; drop dead code ([f33b33e](https://github.com/mac4DSTEM/CuboKit/commit/f33b33ea71f8b8bdf66aba2843b860e36df812e2))
* remove the enums named like their modules ([2e4c0f6](https://github.com/mac4DSTEM/CuboKit/commit/2e4c0f6fdf097241336bb50ea210e31ede5b18ee))
* single-slice ptychography .single takes every detector size ([#59](https://github.com/mac4DSTEM/CuboKit/issues/59)) ([5d40072](https://github.com/mac4DSTEM/CuboKit/commit/5d40072220183433603cc82be769f4a32ae959f5))
* SyntheticProbe centre, public FloatFFT2D.supports, consensus diagnostics for a supplied basis ([#46](https://github.com/mac4DSTEM/CuboKit/issues/46)) ([98f5330](https://github.com/mac4DSTEM/CuboKit/commit/98f53303bc3b2a59bc4fb008195fa0ce3192a656))


### Bug Fixes

* Metal sums give +-Inf, not NaN, for infinite pixels ([#51](https://github.com/mac4DSTEM/CuboKit/issues/51)) ([5642ee1](https://github.com/mac4DSTEM/CuboKit/commit/5642ee10fbe2b5473335712586f5afd1780c6375)), closes [#12](https://github.com/mac4DSTEM/CuboKit/issues/12)


### Performance Improvements

* CPU mean and max pattern 2x faster, TwoSum state in registers ([#53](https://github.com/mac4DSTEM/CuboKit/issues/53)) ([a8b268e](https://github.com/mac4DSTEM/CuboKit/commit/a8b268ec51d9fb9b0e92afba863fb1ba6f454311))
* CPU mean pattern 2x faster, sum held in registers over four positions ([a8b268e](https://github.com/mac4DSTEM/CuboKit/commit/a8b268ec51d9fb9b0e92afba863fb1ba6f454311))
* Float32 parallax shifter for every size, Bluestein on the vDSP Float DFT ([#52](https://github.com/mac4DSTEM/CuboKit/issues/52)) ([60ddbea](https://github.com/mac4DSTEM/CuboKit/commit/60ddbeab4d11f470bdb0e16a521b7d294b575b6f))
* masked and weighted CPU mean about 1.85x faster, sum in registers ([#55](https://github.com/mac4DSTEM/CuboKit/issues/55)) ([1e4a716](https://github.com/mac4DSTEM/CuboKit/commit/1e4a716a2f4d5c127f1f4082ffda467ac59ee376))
* Metal reducers 2x to 5x faster on a cube that is not resident ([#57](https://github.com/mac4DSTEM/CuboKit/issues/57)) ([117a1f9](https://github.com/mac4DSTEM/CuboKit/commit/117a1f94df3113b13bfe17d6d2c60e0e31144e52))
* multicorr Bragg refinement 2x faster, loops ordered for contiguous reads ([#63](https://github.com/mac4DSTEM/CuboKit/issues/63)) ([b8e85f9](https://github.com/mac4DSTEM/CuboKit/commit/b8e85f9d90ef7392942a1d4f85f0fbc3610b7dc5))
* multicorr refinement 2x faster, loops ordered for contiguous reads ([b8e85f9](https://github.com/mac4DSTEM/CuboKit/commit/b8e85f9d90ef7392942a1d4f85f0fbc3610b7dc5))
* multicorr refinement as matrix products on Accelerate, 3.3x faster per cube ([#66](https://github.com/mac4DSTEM/CuboKit/issues/66)) ([1130f2f](https://github.com/mac4DSTEM/CuboKit/commit/1130f2f9b598645f4e01053c3db536942ab35339))

## 0.9.0 (2026-10-22)

The 4D-STEM study release: the polar cube and radial background, the ellipse fits and Fourier resampling, speed and
cancellation, Bragg-vector classification, and a resident cube with a memory cap (suite tasks 17, 18, 19 and 37). Pinned
against py4DSTEM 0.14.19 (`tools/polar-ellipse/pins.py`, `tools/classification/pins.py`) and planted truth
(`tools/polar-ellipse/planted.py`); `docs/parity.md` goes to 135 present, 38 partial, 42 missing of 215 counted rows
(from 121, 38, 50 of 209).

### Added
- **CuboVirtual `PolarCube`** (py4DSTEM `DataCube.polar`, a `PolarDatacube`): the polar-elliptical transform of every
  pattern in Double on the CPU: radial bins (`np.arange(qmin, qmax, qstep)`), `annularCount` angular bins over 2 pi (pi
  with two-fold symmetry), the four-pass bilinear binning, the Gaussian along the angle that follows the arc length,
  normalisation by the transformed detector mask, invalid cells below `maskThreshold`, `radialPower` (`qscale`),
  elliptical from `Calibration.ellipse` at the scan position or circular. `transform(source:calibration:)` does the
  whole cube tile by tile with positions in parallel and one shared geometry when origin and ellipse are constant (1024
  patterns of 256 x 256 in 0.26 s; 0.38 s with an origin per position, Apple Silicon, release). CPU only: at that speed
  a Metal path would be bound by the host copy of the cube, so none was written. Against py4DSTEM's own cells: 5e-16
  relative (`assert_allclose` default is 1e-7). On planted data the right ellipse makes the cube constant in phi (spread
  0.6% of the mean against 8.8% for a circle assumption on the `tilted` case; the ring's radial centroid moves 0.02 px
  over phi against 1.2 px).
- **CuboVirtual `RadialBackground`** (`get_radial_bkgrnd`, `get_radial_bksb_dp`, `get_local_ave_dp(radial_bksb=True)`):
  the median over the angle of the valid polar cells, smoothed along the radius, laid back on the detector as rings
  (linear interpolation, extrapolating); the subtracted pattern is clamped at zero; the local average uses the 3 x 3
  Gaussian kernel cut at the scan edge, with an optional per-position keep mask (`radial_bksb` with `braggmask`;
  CuboBragg's `BraggMasking.mask` supplies it, this module cannot call it). Against py4DSTEM: background 4e-15,
  subtracted 2e-13, local average 4e-13 relative. Throws `CalibrationError.notSet("origin")` without an origin.
- **CuboCore `EllipseFit`** (`process/calibration/ellipse.py`): `fit1D` (`fit_ellipse_1D`), `fitAmorphousRing`
  (`fit_ellipse_amorphous_ring`, with the 11 fit parameters), `ellipseError`, `doubleSidedGaussian`,
  `doubleSidedGaussianFitError`, `constrainDegenerate` (`constrain_degenerate_ellipse`), and the conversions
  `semiAxes`/`canonical` (`convert_ellipse_params`, `_r`). The ellipse is (row, column, a, b, theta) with theta of the
  major axis from the +row axis towards the +column axis: py4DSTEM's (x0, y0, a, b, theta) numbers, so
  `EllipseParameters.calibration` goes into `Calibration.ellipse` unchanged. The fits run a transcription of MINPACK
  `lmdif` from py4DSTEM's starting guesses: the evaluation counts equal SciPy's in 4 of 6 pinned fits, the others within
  10%. Against py4DSTEM: conversions and direct evaluations 1e-7; `constrain_degenerate_ellipse` to the last digit;
  `fit_ellipse_1D` within 7e-10 relative; the amorphous-ring ellipse within 3e-8 relative (theta 2e-7 rad), the residual
  sum of squares within 4e-9. Against the planted ellipses: the amorphous-ring fit to 0.01 px in a and b and 3e-4 px in
  the centre; the algebraic 1D fit to 1.2% in the axes (it is biased upwards by weighting the annulus by the pixel
  values). `fit1D`'s default `fitRadii` is `(0, min(extents) / 2)`: py4DSTEM's default raises a `TypeError`.
- **CuboCore `CubeEditing.resampleDetectorFourier`** and **`fourierResample`** (`resample_Q(method="fourier")`,
  `process.utils.fourier_resample`): zero-padding or cropping of the spectrum, scale (`int(extent * scale)`) or output
  size, `bandlimitNyquist`/`bandlimitPower`, `forceNonnegative`, `conserveSums`, odd and even extents; the Q pixel size
  updates as `resample_data_diffraction` does. Float32 cube out; against py4DSTEM (whose transform is single precision)
  2.1e-7 of the largest value; the Double `fourierResample` reproduces the band-limited planted truth to 1.5e-14.
- **CuboEmbedding `Featurization`** (py4DSTEM `process/classification/featurization.py`) on a feature matrix, one row
  per scan position: `pca(components:)` (scikit-learn 1.9.1's component signs, the largest-magnitude entry of each
  component positive), `nmf(maxComponents:numModels:mergeThreshold:maxIterations:randomSeed:saveAllModels:isCancelled:)`
  (`_nmf_single`, including the merge of correlated weight columns and its upstream quirks),
  `gmm(covarianceTypes:components:numModels:randomSeed:isCancelled:)` (`_gmm_single`, every fit kept), `minMaxScaled`,
  `shiftedPositive`, `reduceFeatures` (mean, median, maximum of chosen columns), `concatenate`, and
  `classPatterns(weights:patterns:pixelCount:threshold:)` (the weighted class patterns of `get_class_DPs`).
- **CuboEmbedding `NMF`**: scikit-learn's `NMF` (coordinate descent, Frobenius, no regularisation) with the NNDSVD,
  NNDSVDa, NNDSVDar, random and custom initialisations, scikit-learn's stopping rule and iteration count.
  **`GaussianMixture`**: scikit-learn's `GaussianMixture` with the four covariance types, the k-means initialisation
  (k-means++ seeding drawn from NumPy's `RandomState`), EM, `bic`, responsibilities. **`NumPyRandomState`**: NumPy's
  legacy `RandomState(seed)` (MT19937: `randomSample`, `randint`, `standardNormal`, `choice(probabilities:)`), bit
  identical. `FeaturizationError`, `FeatureMatrix`. A new C entry `cubo_embedding_svd` (LAPACK `dgesdd`) in
  `CEmbeddingLAPACK`.
- **CuboBragg `BraggVectorClassification`** (py4DSTEM `braggvectorclassification.py`): `peakLabels`
  (`get_braggpeak_labels_by_scan_position`), the data matrix X (boolean or intensity), `initialClasses`
  (`get_initial_classes`, two- or three-point function), `setInitialClassesByCooccurrence`,
  `setInitialClasses(images:)`, `nmf`, `split` and `split(classIndex:)`, `merge` (`merge_iterative`),
  `merge(classIndices:_:)`, `removeClass`, `accept`/`reject` with the candidate `W_next`/`H_next`, the class getters,
  and `classPattern` / `classPatternWithoutBraggScattering` (`classutils.get_class_DP` and
  `get_class_DP_without_Bragg_scattering`, integer and Fourier shifts, dark reference). The iterative steps take
  `isCancelled`.
- **CuboBragg `BraggVectors.virtualImage(detector:)`** (`get_virtual_image`): `.allPeaks`, `.circle(center:radius:)`,
  `.annulus(center:inner:outer:)` over the vectors in detector coordinates (the calibrations are the caller's), NumPy's
  pairwise sum order.
- Tests: `FeaturizationTests` (12), `BraggVectorClassificationTests` (11), `BraggVirtualImageTests` (2), `np.allclose`
  defaults throughout, worst measured gaps below.
- **`ResidentCube.availability(of:)`** answers `.available` or `.unavailable(reason)`: the cube's Float32 bytes against
  half of the physical memory (`ResidentCube.memoryCapFraction`, Paul's decision 2026-10-22), then the Metal device and
  its largest buffer. The reasons are `UnavailableReason.exceedsMemoryCap`, `.exceedsMetalBufferLimit` and
  `.noMetalDevice`, each with `message` in words for a status line. `ResidentCube.init(loading:)` throws
  `ResidentCubeError.unavailable(reason)`; `fits(_:)` is `availability == .available`.
- **CuboVirtual:** the CPU mean pattern (`VirtualDiffraction.mean`, `meanAndMax`, `pattern(.mean)` over `.all`) and the
  selected-area mean over a mask or weights add with a branch-free SIMD TwoSum (four pixels per step, no per-element
  finite check; the check moved to a post-pass on the result, so a NaN or Inf in the input propagates as before and a
  column of -0 keeps its sign). The detector pixels split into 512-pixel blocks across every core (4096 pixels used to
  make two blocks), and a `MemoryCube` is read in place. 100 x 100 x 64 x 64 Float32, loaded Mac: mean 19.2 to 4.9 ms
  (NumPy 2.7), mask 43.1 to 3.9 ms (NumPy 10.0), weights 54.9 to 5.1 ms (py4DSTEM 157).
- **CuboCore `CubeEditing.binScan`** sums four detector pixels per step and splits the output positions across cores
  (28.5 to 2.4 ms; NumPy 7.0), reading several output rows per tile. `medianFilterMaskedPixels` filters patterns across
  cores with a fixed small buffer per window instead of a sorted array per masked pixel and pattern (1.74 to 0.29 ms at
  64 x 64 scan; NumPy 0.64). Both are bit for bit the earlier loops.

- **`isCancelled: (@Sendable () -> Bool)? = nil` and `progress: (@Sendable (Double) -> Void)? = nil`** on
  `StrainMapping.compute` (after `options:`; the Bragg vector map takes the first tenth of the progress when it is
  built, the local fits the bulk) and `StrainMapping.histogram`. In the style of `RotationSolver.curve`: a true answer
  throws `CancellationError` and nothing partial is returned; progress never decreases and the last call is 1.
- **`isCancelled:` on the whole-cube reducers:** `VirtualImaging.image`, `VirtualDiffraction.mean`, `max`, `meanAndMax`,
  `pattern`, `CenterOfMass.measure`, `OriginMeasurement.measure(of:radius:radiusScale:estimator:...)`, asked before each
  tile, same semantics (`progress:` already existed on all but `pattern`, which gained it, covering the mask, weights,
  median and shifted paths). A cancelled Swift task still stops them as before.
- **`LatticeStrain.fitRobust` accepts intensities of 1e10 and above:** when the largest intensity is 1e10 or more every
  intensity is first divided by it in Double, so the Float32 fit no longer sees the overflow or the lost small weights;
  below 1e10 the weights are exactly those of 0.8.1. Non-finite and negative intensities, and coordinates or indices of
  1e10 and above, are still refused.
- `tools/benchmarks/bench.py speed` and `compare`, `Tests/CuboVirtualTests/SpeedBench.swift` (the alternating-rounds
  timing and its bit hash).

### Migration
Additive, with three notes. (1) `ResidentCube.init(loading:)` for a cube over the Metal buffer limit used to throw
`CuboVirtualError.metal(...)`; it now throws `ResidentCubeError.unavailable(reason)`, and for a cube over half of the
physical memory too (a machine without a Metal device still throws `CuboVirtualError.noMetalDevice`): ask
`ResidentCube.availability(of:)` first. (2) `LatticeStrain.fitRobust` (and so `StrainMapping.compute` with `fitMethod:
.robustIntensityOnce`) fitted nothing for a peak intensity of 1e10 or more (it threw `invalidInput`; the map counted a
refused fit); it now fits. An app that used that refusal as a flag must test the intensity itself; the rescale mac4DSTEM
does before `fitRobust` can go. (3) A Sonnet review of the new surface turned inputs that trapped into throws before the
tag, so these 0.9.0 signatures throw (all new in 0.9.0, none existed in 0.8.1): `BraggVectorClassification.classImage`,
`classPeakWeights`, `candidateClassImage` and `candidateClassPeakWeights` (`invalidOption` for a class that does not
exist), `PolarCubeData.pattern` (`CuboError.outOfRange`), `EllipseFit.ellipseError`, `doubleSidedGaussian` and
`doubleSidedGaussianFitError` (`invalidInput` for arrays of the wrong length), and `RadialBackground.profile`.
`PolarCube.init` refuses more than 1e7 radial bins, 65536 angular bins, 2^26 cells per pattern or a smoothing kernel
above 65536 samples a side; `transform(source:)` refuses a result above half of the physical memory; the Fourier
resample caps its output at 2^20 per side and 2^26 pixels per pattern; `initialClasses` caps the n^2 and n^3 tables at
1e8; `split` needs a finite sigma of at most four scan sides; a non-finite class-pattern shift throws; the
amorphous-ring fit refuses a centre more than four plane extents from the pattern. (4) Every cancel and progress
parameter is new and defaulted: a call that names its arguments compiles unchanged. In mac4DSTEM,
`StrainMapping.compute` can take the app's cancel flag and a progress callback, `PolarCube` and `EllipseFit` are what
the polar background and the calibration ellipse can now call, and DiffractoKit's open-items line says to consume the
polar cube (no code there yet).

## 0.8.1 (2026-10-21)

Tests and tools only; no source change and no number moves.

- Calibration pins loosened to the package's tolerance (D17). py4DSTEM has no test for these numbers, so they are held
  to `assert_allclose`'s Float64 default (rtol 1e-7, with atol 1e-12 where a value is zero in one arithmetic):
  `originMean` (1e-14 before), the axes and grids of `DataCube.calibrate`, `rxx`/`qxx`/`qxxs` and the raw grids (1e-12
  relative), `originShift`, the QR rotation in degrees (1e-12) and radians (1e-15), the pixel size after `bin_Q`
  (1e-15), the detector pixel size from a known distance (1e-15) and the fitted peak positions (1e-14), the
  scan/detector vector rotations (1e-14), and the pixel size after `resample_Q` (1e-15). Stored values, indices and
  flags stay exact; the cube edits keep their 0 and 1e-6 pins. The measured gaps are unchanged (1 ulp).
- The 0.8.0 Migration sentence about `ResidentCube` is rewritten: it named `ResidentCube` as replacing itself.
- `tools/hot-pixels/pins.py` and `tools/ptycho/options_pins.py` read the py4DSTEM checkout from `References/CuboKit`
  (they pointed at a stale path); `tools/idpc/pins.py` reads the app source from the sibling `mac4DSTEM` checkout.
  hot-pixels regenerates its fixture byte for byte; the idpc fixture was not regenerated (only its recorded NumPy
  version string would change, 2.5.0 to 2.5.4).

### Migration
None.

## 0.8.0 (2026-10-21)

The kit audit against py4DSTEM 0.14.19 (`f050d207`): the calibration model, the cube edits, the Metal selected-area
kernel with a resident cube, and the first parity count that walks the whole API (`docs/parity.md`: 121 present, 38
partial, 50 missing of 209 counted rows; 25 more not planned, with reasons). Timings against py4DSTEM at three sizes:
`docs/benchmarks.md`.

- **CuboCore `Calibration`** (value type): Q and R pixel sizes and units, the origin per scan position or as one number
  (`ScanField`, with mean and shift), measured origin and mask, the ellipse, the Q-R rotation and flip, the probe
  numbers; `detectorRowAxis`, `scanRowAxis` and the calibrated, raw and per-position-shifted grids (`rxx`, `qxx`, `qxxs`
  ... of `DataCube.calibrate`, rows x columns: py4DSTEM's calibrated grids are the transpose, the values are equal).
  `afterBinningDetector`, `afterBinningScan`, `afterThinningScan`, `afterResamplingDetector` update the pixel sizes as
  py4DSTEM's preprocess functions do (the origin is not rescaled, as upstream). `CalibrationMeasurement`:
  `get_Q_pixel_size`, `get_dq_from_indexed_peaks` (closed-form least squares, 1e-7 relative against `leastsq`),
  `get_Qvector_from_Rvector` and the reverse (1e-14). Tests: `CalibrationTests` against `tools/calibration/pins.py`.
- **CuboCore `CubeEditing`**: `setScanShape`, `swapScanAndDetector`, `swapScanAxes`, `swapDetectorAxes` (exact),
  `binScan` (`bin_R`, 1e-6 relative, NumPy's Float32 sum against the kit's Double sum), `thinScan` (exact),
  `padDetector` (factor or output size, exact), `resampleDetector` (bilinear = scipy `zoom(order=1, mode="nearest",
  grid_mode=True)`, `conserveSums`, 1e-6 relative), `medianFilterMaskedPixels` (exact for odd kernels). Each returns an
  `EditedCube` (a `MemoryCube` and the updated `Calibration`). Where py4DSTEM prints and returns the cube unchanged, the
  kit throws.
- **CuboVirtual**: `VirtualDiffraction.pattern(..., path: .metal)` runs the mean and the max over a mask or weights on
  the GPU (`cubo_weighted_pattern`: compensated Float32 sums combined in Double, held to the CPU path within relative
  1e-5; measured 6e-8 on the test cubes), with negative controls in `SelectedAreaMetalTests`. The median and a centre
  shift stay CPU and throw with `.metal`. New `ResidentCube`: a cube held in one Metal buffer (`ResidentCube.fits`
  first); every `.metal` reducer (virtual image, mean and max, centre of mass, the new pattern) reads it in place, bit
  for bit what the copying path gives; 4 ms against 26 ms for the mean at 100 x 100 x 64 x 64. The CPU mean and max now
  run on pixel blocks across cores and `mean(of:)`/`max(of:)` compute only the statistic asked for; the results are bit
  identical to the single-thread loop (`ParallelReducerTests`); the max pattern at the typical size went from 35 ms to
  5.9 ms.
- **CuboBragg `BraggMasking`**: `mask` (`get_braggmask`), `localAverage` (`get_local_ave_dp`, plain and Bragg-masked),
  `latticeMask` (`make_bragg_mask`); pins in `tools/braggmask/pins.py`. The radial-background variants need the polar
  data cube (DiffractoKit).
- `docs/benchmarks.md`, `docs/open-items.md`, `docs/status.md`; the harness is `tools/benchmarks` (`bench.py`,
  `BenchmarkHarness`). `tools/consumer-check`: 143 checks.

### Migration

Additive. `VirtualDiffraction.pattern` gained `path: ComputePath = .cpu` between `shift:` and `tiling:`; a call that
names its arguments compiles unchanged. In mac4DSTEM, to take the kit's versions when the app is switched: the kit's
`CuboVirtual.ResidentCube` replaces the app's own `ResidentCube` (the app type that holds a `FourDArray` in a Metal
buffer) for the Metal reducers; qualify the name while both exist, `VirtualDiffraction.pattern(..., path: .metal)`
replaces `MetalEngine.virtualDiffraction`, and `CubeEditing`/`Calibration` replace the app's bin, crop-and-pad helpers
and calibration struct where they overlap. `PendingLoad` and `VirtualDetector.tiledDiffraction` stay in the app
(load-screen state and tiling policy). The learned detector stays in the app.

## 0.7.1 (2026-10-21)

- **CuboEmbedding:** public `PCA.project(_:mean:basis:dims:components:centred:into:)`, the Float32 projection of one
  vector on the principal components (`vDSP_vsub` centring, exact; `vDSP_dotpr` accumulation). It is the code
  mac4DSTEM's `DiffractionEmbedding.project` ran, and `PCA.fit(variant: .diffractionEmbeddingApp)` now calls it for its
  scores (same instructions, results unchanged). Tolerance, written in `ProjectFloatTests`: against a Double sum,
  `dims * 6e-8 * sum|centred * basis|` per coordinate (measured 1.8e-8 relative at 1024 dimensions); centring bit-exact.
  `tools/consumer-check`: 138 checks.
- Not provided (documented ask): a Metal selected-area (weighted-pattern) kernel and a GPU-resident cube.
  `VirtualDiffraction.pattern` stays CPU (compensated Double). The app's `VirtualDiffraction.metal`, `FourDArray`,
  `ResidentCube`, `PendingLoad` and `VirtualDetector.tiledDiffraction` stay in mac4DSTEM until that is sized as its own
  change.

### Migration

Additive. In mac4DSTEM: `DiffractionEmbedding.project` is deleted; call `PCA.project` (same signature).

## 0.7.0 (2026-10-21)

Parity with the app's copies, so mac4DSTEM can delete its ptychography engine, parallax alignment and the numerics below
(the learned detector stays in the app). Timings: release build, one core, Mac under load 5-9, best of 5-20.

- **Speed, no API change (CuboCore `FFTPlan`/`FFT2D`):** the Double transform runs on `vDSP_DFT_zop` for the lengths
  vDSP takes (2^n x {1, 3, 5, 15}) and keeps the radix-2/Bluestein path for the rest; 5-7x faster than `vDSP_fft_zipD`
  from 128 x 128. It checks against a naive DFT (1e-12 of the largest coefficient) and against 0.6.0 (CuboPtycho object
  1.7e-11, spectrum 2.4e-13, shift 3.3e-15 on a 3-iteration case: no pinned fixture moved).
  `FFT2D.transform(real:imaginary:inverse:scratch:)` is the allocation-free `package` entry. `ParallaxAlignment.apply`
  builds the phase from a row and a column ramp, not one `sincos` per pixel.
- **CuboPtycho single-slice engine:** the Fourier-modulus projection is `measured * F / |F|` (no `atan2`/`cos`/`sin`),
  shift ramps are separable, no per-pattern allocation, pointer loops: 3 iterations of 100 patterns at 64 x 64 take
  about 30 ms in Double (0.6.0: 60 ms). New `precision: .single` (and
  `SingleslicePtychography.reconstructSingle(amplitudes: [Float], ...)`): Float32 on vDSP's DFT, about 14 ms (the app's
  engine: 15 ms). Float32 tolerance, written in `SinglePrecisionTests`: object 1e-4 absolute on a unit-modulus object
  (measured 2.9e-5; non-square detector with fractional positions as well), probe 1e-5 (measured 1.5e-6), error history
  1e-4 relative. The Float path needs a detector size vDSP takes and has no position correction (both throw
  `invalidInput`). `isCancelled:` and `progress:` on `reconstruct`.
- **CuboParallax:** `ParallaxFourierShifter.shiftedSingle` (image and batch, with `isCancelled:`/`progress:`) and
  `supportsSingle`: 0.19 ms per 160 x 160 shift against 0.47 ms Double (0.6.0: 1.28 ms; the app: 0.22 ms); agrees with
  Double to 1e-5 of the image range. `ParallaxPreprocessing.prepare(source:...)`: streamed, two passes over scan-row
  tiles of any `CubeSource`, flat Float stacks `[bright-field pixel][row][column]`, mask, probe angles in mrad,
  `maximumStackBytes` refusal (`ParallaxError.stackTooLarge`), progress, `isCancelled`; matches the resident `prepare`
  to 2e-5 relative (measured 3.9e-7) and is exact across tile sizes.
- **CuboVirtual:** `OriginEstimator` (`.py4DSTEM`, `.iteratedWindow(passes:minimumMargin:binnedCoarse:)` = the app's
  estimator) with `OriginMeasurement.measure(of:radius:radiusScale:estimator:path:...)` on CPU and Metal
  (`cubo_measure_origin`, Metal against CPU 1e-5) and `OriginMeasurement.windowSensitivity`. Not provided: a Metal
  weighted-pattern kernel (`VirtualDiffraction.pattern` sums in compensated Double).
- **CuboStrain:** public `LatticeStrain.fitRobust`, `scoreBasis`, `indexingTolerance`, `conditionNumber`,
  `indexByRounding`, `StrainFrame.rotate`; `StrainFitMethod.robustIntensityOnce`; `StrainMapOptions.indexing`
  (`.nearestDirection` default, `.basisRounding`) and `indexingTolerance`; `StrainMapResult.refusedFitCount`,
  `firstRefusal`, `basisSupportFraction`, `basisConditionNumber`. Double helpers agree with the app's Float formulas to
  1e-6 (tests 1e-9 against a Double transcription).
- **CuboBragg:** `BraggDisks.detectAll(in:kernel:options:tiling:failurePolicy:isCancelled:progress:)` returning
  `BraggScanResult` (`vectors`, `failedPositions`, `firstFailure`), `BraggFailurePolicy` (`.throwOnFirst`,
  `.recordEmpty`), `BraggDiskOptions.validate()`. The old `detectAll` is unchanged in behaviour. `FloatFFT2D` moved to
  CuboCore (`package`).
- **CuboCore:** `HotPixelFilter.replacing(in:base:plan:validatingFinite:)`, the in-place loop entry on a `Replacer`
  (checks only the pixels the plan reads); bit-identical to `replacing`.
- `tools/consumer-check`: 137 checks. No new negative controls (`controls.sh` not extended this release).
- Package.swift has no sibling-kit dependencies, so no ranges were widened.

### Migration

Additive, with two edges: `ParallaxError` gains `stackTooLarge` (an exhaustive `switch` needs a default), and the Double
FFT results move by about 1e-15 relative. In mac4DSTEM: `SingleslicePtychography.reconstruct(precision: .single,
isCancelled:, progress:)` replaces the app's engine (no position correction there, as in the app);
`ParallaxFourierShifter.shiftedSingle` and `correlationShift` replace `ParallaxAlignment`'s Float path, with
`MatrixDFTCorrelation` going through `ParallaxAlignment.shift` (Double, about 4 ms per peak);
`ParallaxPreprocessing.prepare(source:)` replaces `ParallaxPreprocessor.run` (kit angles are mrad; sampling per axis);
`OriginMeasurement.measure(estimator: .iteratedWindow, path: .metal)` replaces `MetalEngine.measureOrigins`;
`BraggDisks.detectAll(failurePolicy: .recordEmpty)` replaces `DiskDetection.detectAll`; `LatticeStrain.fitRobust` and
`StrainMapOptions.indexing: .basisRounding` replace `StrainMapping`'s fit and indexing. Stays in the app: the learned
detector (training, refine), display Metal, `ParallaxPhysicalCalibration`, `PhaseContrastMemoryBudget`, the `condition
<= 8` refusal of a given basis (use `LatticeStrain.conditionNumber`).

## 0.6.0 (2026-10-21)

The app's remaining kit asks, so mac4DSTEM can drop its copies. Every addition is opt-in or additive; defaults keep
0.5.0 results (CuboParallax's default position correction now throws instead of trapping on a flat score map, below).
Tolerances are written in each test and measured against py4DSTEM 0.14.19 (`f050d207`), scikit-learn 1.4.2 or a
transcription of the app source.

- **CuboBragg, single precision:** `BraggDiskOptions.precision` (`BraggPrecision.single`) runs the pattern copy,
  filters, 2-D FFT, correlation and maxima search in Float32 (vDSP) with per-worker scratch; accepted peaks are refined
  in Double. Sizes vDSP's DFT does not support (f·2^n, f in 1, 3, 5, 15) throw `invalidOption`;
  `BraggDetector.supportsSinglePrecision` asks. 256 x 256, poly: 0.47 ms serial / 0.14 ms wall on 18 cores against 5.31
  / 0.87 for `.double`; the app's own detector measured 0.24-0.27 ms wall. Pins: positions 1.3e-5 px (tolerance 5e-5),
  intensities 1.1e-6 (5e-6). `correlationPower` near 0 is not held in Float32 (documented). Also `BraggFunnel` and
  `detectWithFunnel` (per-stage counts as the app reports them, peaks identical to `detect`),
  `BraggMaxima.findWithFunnel`, public `BraggDetector.fillNonFinite` (Double and Float) and Float-buffer `detect`.
- **CuboPtycho:** `constrainObjectAmplitude` on the single-slice engine (off skips the |object| <= 1 cap; DEVIATION,
  pinned to py4DSTEM with only that constraint replaced, gap 1.8e-7 at 2e-6); `PtychographyAberrationTerm` (any
  radial/angular order, cosine or sine, `.polar` for py4DSTEM's C_nm, phi_nm) as `higherOrder:` on
  `PtychographyProbe.build`, `PtychographyPreparation.initialProbe` and
  `PtychographyPreparationOptions.higherOrderAberrations` (C21 to C56 pinned to py4DSTEM at 1.2e-7; orders py4DSTEM
  cannot express to a NumPy transcription of the app, 4.8e-10).
- **CuboParallax:** `ParallaxFourierShifter` (plan built once, batch shifts, bit-identical to
  `ParallaxAlignment.shifted`); `ParallaxAlignment.correlationShift` on `ParallaxSpectrum` (the app's spectrum-level
  estimate, 3.3e-14 px); `ParallaxCorrectionResult.transferReal/transferImaginary`; `isCancelled:` and `progress:` on
  alignment, batch shift, `reconstruct`, `process`, `section` and `correctPositions` (`CancellationError`);
  `ParallaxDepthSectioner` (forward FFTs once, `plane(atDepth:)`; `section` planes bit-identical);
  `ParallaxPositionCorrectionVariant.application` (the app's Float loop, offsets exact, image 5.5e-8). Behaviour change:
  the default position-correction loop on an exactly flat score map used to trap on NaN coordinates and now throws
  `ParallaxError.nonfiniteValues`.
- **CuboEmbedding:** `PCACovarianceAccumulator` (streamed, mergeable, Chan/Welford in Double),
  `PCA.fit(covariance:count:mean:components:)` and `PCASignResolver` (svd_flip signs in a second pass; streamed fit
  equals scikit-learn to 2e-12 on components, 1e-15 on variance, invariant to tile order);
  `DiffractionFeatures.binnedVector(implementation: .accelerate)` (the app's vDSP path, 2.7e-7 relative to the scalar
  one; non-finite patterns fall back to scalar; default `.scalar`); `isCancelled:` on `KMeans.fit` and `fitSeeded`.
- **CuboStrain:** `LatticeStrain.chooseBasisConsensusResult` (`ConsensusBasis`: basis, observation and support counts,
  residual, condition number, indexing tolerance; `isCancelled:`), `referenceMADResult` (`ReferenceMAD`: reference and
  inlier indices), `StrainMapResult.referenceInlierCount`. The plain calls return identical bases.
- **CuboCore:** `HotPixelFilter.Replacer`, the mask's clipped neighbour lists built once; `apply` replaces in place with
  no allocation, 0.005 ms against 0.054 ms per 256 x 256 pattern with 40 masked pixels, bit-identical to `replacing`,
  which keeps its validate-and-copy contract (and its time).
- `tools/consumer-check`: 105+ checks and the new controls for every new public symbol; CuboParallax is covered for the
  first time.

### Migration

Additive. In mac4DSTEM: `DiskDetector` can call `BraggDetector` with `.single` (and `detectWithFunnel`,
`fillNonFinite`); the ptychography settings map onto `constrainObjectAmplitude` and `PtychographyAberrationTerm`
(cartesian C12 to polar: `C12 = hypot(a, b)`, `phi12 = atan2(b, a) / 2`); parallax drops its Fourier shift, correlation
estimate, transfer-function recomputation and per-plane re-FFT for the shifter, spectrum estimate, transfer fields and
sectioner; embedding drops `BinGeometry`, the `embed` fast path and the covariance eigen shim; strain drops the
re-scoring in `estimateLatticeBasis` and `robustReferenceIndices`; `CalibratedDataCubeWriter` can reuse one
`HotPixelFilter.Replacer`. Not provided: object-amplitude toggle on the mixed-state and multislice engines; a Float
matrix DFT (multicorr is Double, about 4 ms per peak); `LatticeStrain.diagnostics(basis:positions:)` for the
`initialBasis` path; a scan mask on `chooseBasisConsensus`.

## 0.5.0 (2026-10-20)

App-resident computation moves into the kit, each piece pinned to py4DSTEM 0.14.19 (`f050d207`), NumPy or a
transcription of
the app source, with the tolerance written in its test (`docs/ptycho.md` has the measured gaps).

- **CuboPtycho preprocessing:** `PtychographyPreparation` (`amplitudes`, `scanPositions`, `objectShape`,
  `centerPositions`, `initialProbe`, `prepare`) builds the amplitude stack, scan positions, object canvas and starting
  probe from a `CubeSource`. Bilinear and cubic-spline (SciPy grid-wrap) pattern shifts, py4DSTEM's
  rotation/transpose/padding, Float32 rotation matrix reproduced. Measured gaps 4.2e-8 (amplitudes), 0 (positions), 6e-8
  (probe) at tolerance 2e-6. The app's choices are named options (`PtychographyPreparationOptions.mac4DSTEM`: bilinear
  shifts, per-axis padding, no re-centring, clipped mean intensity).
- **CuboPtycho engines:** `MixedstatePtychography` (`reconstruct`, `orthogonalize`, `initialProbes` with
  NumPy-compatible seeded states) and `MultislicePtychography` (`reconstruct`, `propagators`, `identicalSlices`), CPU
  Double beside `SingleslicePtychography`; new `PtychographyStack` type. Measured gaps 3.9e-7 at 2e-6 over 11 py4DSTEM
  cases; one slice equals the single-slice engine to 2e-16.
- **CuboPtycho constraints:** `fixProbeAperture` and `probeSupportMask` on all three engines (off by default), pinned to
  py4DSTEM on each.
- **CuboCore:** `ViewedCube`, `CubeViewSpecification`, `CubeViewGeometry`, `AxisCrop`: scan crop, detector crop and
  detector bin applied at read time to any `CubeSource` (py4DSTEM `crop_data_real`, `crop_data_diffraction`,
  `bin_data_diffraction`: crop first, remainder trimmed off the end, blocks summed). Shapes and integer sums exact,
  float sums within 1e-6 (measured 1.2e-7). `AxisCrop.enclosing` is the app's drag-rectangle arithmetic. New error case
  `CuboError.invalidView`.
- **CuboPhase:** `DPCViews` (magnitude, angle in [0, 2 pi) with py4DSTEM's `row + i column` direction or the image-frame
  `atan2(row, column)`, milliradians per pixel) and `RotationConvention` (the sign flip between py4DSTEM's R-Q angle and
  an image-frame caller's, display text), NumPy-pinned.
- **CuboPtycho error normalisation:** the engines take `errorNormalization` (`.meanIntensity(m)`: py4DSTEM's residual
  over `m * positions`, with `PtychographyAmplitudes.errorNormalization` giving it for preprocessed amplitudes;
  `.suppliedAmplitudeEnergy`: the previous behaviour, still the default because bare amplitudes carry no mean) and the
  mixed-state engine `zeroFarFieldPixelError` (`.reference`, the default: py4DSTEM's infinite term for an exactly zero
  far-field pixel; `.finite`: the previous `measured^2`). The pins now go through py4DSTEM's preprocessing with a mean
  that differs from the amplitude energy by more than 2 percent; the old normalisation fails them.
- **CuboPtycho position correction:** `SingleslicePtychography.reconstruct(positionCorrection:)`
  (`PtychographyPositionCorrection`: step size, maximum update and total distances in Å, centre-of-mass constraint) and
  `Result.positions`; five py4DSTEM cases, positions 6.2e-7 px, others 4e-7 at 2e-6, plus an exact-data sign check. Off
  by default.
- **CuboPtycho odd sizes:** preprocessing and the three engines pinned on odd, non-square detectors, probes and objects
  (`odd.json`); no defect found.
- **CuboCore:** `OriginFit.fitTrimmed` (`TrimmedOriginFit`), the MAD-trimmed origin fit of the app's calibration, beside
  `OriginFit.fit`: joint residual, `median + 3 * 1.4826 * MAD`, up to three rounds, final fit on the kept mask; fourteen
  cases pinned to a Float32 transcription of the app source (mask exact, surfaces 6e-8) and to py4DSTEM's masked
  `fit_origin` (1.3e-8). `AxisCrop.rowRange`, `columnRange` and `fits` saturate or refuse instead of trapping on extreme
  integers, and `AxisCrop.enclosing` returns `nil` when a stride product overflows.
- `docs/parity.md`: 79 done, 37 partial, 66 absent of 182 (crop_Q, crop_R, bin_Q done; mixed-state and multislice
  partial).

### Migration

Source-compatible except that `CuboError` gains a case (`invalidView`): an exhaustive `switch` over it needs a default,
and `SingleslicePtychography.Result` gains `positions`. `SingleslicePtychography.reconstruct` gains defaulted parameters
(`fixProbeAperture`, `probeSupportMask`, `errorNormalization`, `positionCorrection`); the other engines gain
`errorNormalization`, and the mixed-state one `zeroFarFieldPixelError`. Behaviour change: with a mixed-state engine, an
exactly zero far-field pixel now makes the iteration's `error` infinite (py4DSTEM); pass `zeroFarFieldPixelError:
.finite` for the old value. A caller that wants py4DSTEM's `errors` passes `errorNormalization:
amplitudes.errorNormalization`. A caller in the app moves: the preparer (`PtychographyPreparer`) to
`PtychographyPreparation.prepare(..., options: .mac4DSTEM(...))` after relabelling its axes to py4DSTEM's (its x and y
are the kit's column and row); the load view's crop and bin to `ViewedCube` (its bin factors 2, 4, 8 stay the app's
policy); the DPC magnitude and angle images to `DPCViews` (Double, not Float32 interleaved) and the rotation sign to
`RotationConvention`. Not moved: the colour wheel (a display), the byte-count and residency arithmetic of the load
configuration, the UI selection state.

## 0.4.0 (2026-10-20)

Three additions, each pinned to py4DSTEM 0.14.19 (`f050d207`) with the tolerance written in its test.

- **CuboPtycho:** `SingleslicePtychography.reconstruct` gains `fixProbeCenterOfMass`, `constrainProbeAmplitude`,
  `probeAmplitudeRelativeRadius` (0.5) and `probeAmplitudeRelativeWidth` (0.05), all off by default so every earlier
  result is unchanged (py4DSTEM turns the centre-of-mass constraint on by default; here it is the caller's choice). Four
  new py4DSTEM cases, measured gap 3.6e-7, tolerance 2e-6. Radius outside 0...0.5 or width outside (0, 0.5] is
  `invalidInput`. Other probe constraints (aperture, Fourier amplitude, aberration fitting, support) remain absent.
- **CuboParallax:** `ParallaxReconstruction.reconstruct` ports `Parallax.reconstruct` (alignment-bin schedule, running
  reference, Bernstein regularisation, median centring, per-bin error), and `ParallaxReconstruction.process` runs
  prepare, reconstruct and the optional KDE upsampling and position correction in one call. Seven py4DSTEM cases, gaps
  1e-7 to 9e-7 (tolerance 5e-6 on shifts, 2e-6 on images). Named option `emptySchedule = .singlePassAtMinimumBin` is the
  app's behaviour where py4DSTEM raises. Aberration fit and correction and depth sectioning stay separate calls. Not
  provided: initial aberration shifts, real-space mask, continue/reset, batching.
- **CuboStrain:** `StrainMapping` (`histogram`, `chooseBasisVectors`, `compute`, `centredPeaks`) is the scan-level
  pipeline of py4DSTEM's `StrainMap`: Bragg-vector histogram, basis choice, per-position fits, reference, strain maps
  with coordinate rotation. Pinned to six `StrainMap` runs and five histograms (strain gap 1.8e-15, tolerance 1e-9).
  `LatticeStrain.chooseBasis` gains an overload with explicit `indexOrigin/G1/G2`. `BraggDiskOptions.tieOrder`
  (`.rasterOrder` default, `.numpy` for py4DSTEM's order of equal maxima) is new in CuboBragg. `tools/consumer-check`
  scope checks, which tested a string that no longer existed, now test real scope text.
- `ParallaxPreprocessingResult` gains `scanRows`, `scanColumns`, `paddedWindow` (`correctPositions(window:)` takes
  `paddedWindow`, as its comment now says).
- `docs/parity.md`: 76 done, 35 partial, 71 absent of 182.

### Migration

Additive for the source. `CuboStrain` now depends on `CuboBragg` (and through it `CuboVirtual` and `CuboParallax`), so a
package that imports only `CuboStrain` builds those too; `StrainDirections.maxima` uses `BraggPeak`, so callers of the
maxima path also `import CuboBragg`. `ParallaxPreprocessingResult` has new public fields (its initialiser is internal).
Everything else keeps its default.

## 0.3.0 (2026-10-16)

Consumer notes for 0.0.1 to 0.2.0 are archived.

0.3.0 adds seven pieces on the CPU, each pinned, with the tolerance stated in its test, to py4DSTEM 0.14.19
(`f050d207`),
scikit-learn 1.4.2 or a transcription of the mac4DSTEM source (Double, except hot pixels and origin re-referencing in
Float32): `CuboParallax` (alignment, preprocessing, aberration fit and
correction, depth sectioning, subpixel upsampling), `CuboPtycho`, `CuboStrain`, `CuboBragg`, `CuboEmbedding`, Fourier
iDPC in `CuboPhase`, and hot-pixel filtering and origin re-referencing in `CuboCore`. Where the mac4DSTEM source differs
from the reference, the difference is a named option and the reference is the default. `docs/parity.md` now counts 73
done, 36 partial and 73 absent of 182 rows in scope.

### CuboParallax

- Add the CPU-only `CuboParallax` product: Double-precision matrix-DFT correlation refinement, periodic Fourier
  translation, and fixed/running-reference stack alignment (default upsampling factor 8). Public inputs and results are
  Sendable and independent of app types. Four small stacks pin shifts, shifted images, means and DFT patches to py4DSTEM
  0.14.19 (`f050d207`), with `1e-10` normwise tolerance.

- Add CuboParallax constant-order bright-field preprocessing: detector selection, reciprocal vectors, radian probe
  angles, sine-squared blending and centered padding. Default windowed stack retention follows py4DSTEM; `.unwindowed`
  retains the app KDE input.

- Add CuboParallax aberration fitting (`ParallaxAberrationFitter`: polar-decomposition low order C1/C12/rotation,
  higher-order global/recursive fits) and CTF phase correction (`ParallaxAberrationCorrection`). Pinned to py4DSTEM
  0.14.19 `aberration_fit`/`aberration_correct` (fit coefficients to 2e-4, relative with a unit floor (absolute below
  1), measured 6e-7, against a Float32 reference basis, low order and rotation to 1e-10, corrected image to 2e-6). The
  reference ignores `q_highpass`; the app applies it, so `.reference` is the default and `.apply` is the app option,
  with `.detectorGrid`/`.selectedCentroid` basis conventions, the app pinned to a NumPy transcription of the app source.

- Add CuboParallax depth sectioning (`ParallaxDepthSectioning`), py4DSTEM `depth_section` semantics. The reference path
  (defocus-only sign CTF, optional information-limit envelope) is pinned to py4DSTEM 0.14.19 at 2e-7 measured (tolerance
  1e-5 of the maximum over all planes, complex64 reference); py4DSTEM's `use_CTF_fit=True` raises a TypeError at that
  version, so the full-fit CTF (`.fullFit`, the app's behaviour) is pinned to a NumPy transcription of the app source.

- Add CuboParallax subpixel kernel-density upsampling (`ParallaxSubpixel`): the upsample-factor heuristic and
  bilinear/Lanczos deposition with Gaussian blur and optional sinc low-pass, pinned to py4DSTEM 0.14.19
  `subpixel_alignment` (measured gap 4e-7, tolerance 1e-4; the reference accumulates in Float32). The app's KDE source
  (mac4DSTEM c809679c) was read against the reference and uses the same formulas, including the Lanczos second-axis
  window quirk, so there is no separate app option; this was a reading, not a pinned comparison. Probe-position
  correction (`correctPositions`: gradient and checkerboard steps, Gaussian and Butterworth regularisation) is pinned to
  the reference over three 3-iteration cases (offset gap 1.7e-5, image gap 6e-7, tolerance 1e-4). The app's variant (no
  regularisation; clamps the step floor only on rejected sites; guards a zero gradient) is not ported as an option.

### CuboPtycho

- Add CuboPtycho CPU probe construction (defocus, two-fold astigmatism, C30) and full-batch single-slice complex-object
  gradient descent from preprocessed amplitudes, with fixed positions, optional probe updates and pure-phase object
  constraints. Pinned py4DSTEM 0.14.19 fixtures cover probe pixels and object/probe after three iterations.

### CuboStrain

- Add CuboStrain basis selection from histogram maxima, measured-direction indexing, intensity-squared lattice fits,
  median references and rotated infinitesimal strain. Separate intensity-once, MAD-reference and cluster-consensus
  compatibility APIs preserve the original source numerics. Small binary fixtures pin both conventions.

### CuboBragg

- Add the CPU `CuboBragg` product: py4DSTEM's `find_Bragg_disks` for one pattern, a stack or a cube in Double (hybrid
  cross-correlation with a `ProbeKernel`, `get_maxima_2D` with its filter cascade, parabolic and multicorr subpixel
  refinement), plus the app's `ProbeRingHint`. Defaults are py4DSTEM's; the mac4DSTEM differences are named options
  (`relativeReferenceMinimumRadius`, `nonFinitePixels`, `multicorrOrder`, `BraggDiskOptions.mac4DSTEM`,
  `detectorAdapted`). Thirty py4DSTEM 0.14.19 cases pin positions and intensities to 1e-9 px / 1e-10 (measured 1.5e-12 /
  1.3e-13); ten cases pin the options to the unchanged app source in Float32 (1e-3 px / 1e-5; measured 1.8e-4 / 1.2e-6).
  `GaussianFilter.apply` gains an `edge:` parameter (`.reflect`, scipy's default); the default `.nearest` is unchanged.
  `CuboParallax` exposes its matrix-DFT refinement to the package (`package`, no public change).

### CuboEmbedding

- Add CuboEmbedding: `PCA` (scikit-learn 1.4.2 `svd_flip` sign convention, covariance divisor rows - 1) and `KMeans`
  (scikit-learn Lloyd from caller-given centroids), pinned to scikit-learn. mac4DSTEM's Float32 covariance PCA, seeded
  k-means and binned diffraction feature vector are separate named options pinned to its original source. Binary
  fixtures under 50 KB; see `docs/embedding.md`.

### CuboPhase and CuboCore

- Add CPU Fourier iDPC integration in CuboPhase, with periodic or centered zero-padded boundaries, explicit
  regularization, anisotropic sampling and projected-phase calibration. Regression pins use a NumPy float64
  transcription of the app Fourier integration; this is separate from the existing py4DSTEM iterative DPC.

- Add CuboCore hot-pixel filtering, wrapped 21-pixel classification and sequential clipped-window median replacement.
  Float32 mean accumulation matches py4DSTEM, including the contiguous pairwise reduction for a 1×1 detector. Eight
  small fixtures pin means, masks and all filtered pixels bit for bit to py4DSTEM 0.14.19.

- Add CuboCore `FrameTransform`: detector-position, scan-crop and per-position origin-map re-referencing under crop and
  bin, with the pixel-centre bin rule `(v+0.5)/bin-0.5`. py4DSTEM moves no origin on crop or bin (it only multiplies the
  Q pixel size by the bin factor), so these rules are the app's, pinned Float32 bit for bit to a transcription of
  mac4DSTEM `CalibrationReReference.swift` at c809679c (`tools/rereference/pins.py`). Calibration objects, provenance
  and invalidation messages stay in the app.

### Migration

Additive: every 0.2.0 product and API is unchanged, and the new modules (`CuboParallax`, `CuboStrain`, `CuboBragg`,
`CuboPtycho`, `CuboEmbedding`) are new library products a consumer adds only if it wants them. One public signature
gains a defaulted parameter: `GaussianFilter.apply` takes `edge:` (default `.nearest`, as before). `CuboParallax`'s
matrix-DFT refinement is shared with `CuboBragg` through `package` access (no public change). Nothing a consumer
calls moves. Full parallax reconstruction (grouping, schedules, regularisation) is not yet provided.

## 0.0.1 to 0.2.0

Pre-releases, no entries.

---

From 0.10.0 on, release-please writes each entry from the merged pull-request titles, and the version and compare link
come with it. Where a consumer must change something, the linked pull request has a Migration paragraph.
