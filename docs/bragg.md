# Bragg disk detection

`CuboBragg` is the numeric core of disk detection: py4DSTEM's `find_Bragg_disks` for one pattern, a stack or a cube, on the
CPU in Double, or (opt in, `BraggDiskOptions.precision = .single`) in Float32 on Accelerate. Rows and columns, never x and y: py4DSTEM's `qx` is the row. Probe kernels come from CuboVirtual
(`ProbeKernel`, corner-centred); nothing here builds a kernel, controls a UI, validates a user interface, records
provenance or schedules tiles beyond `ScanTiling`.

## The definition

For a pattern `P` and a corner-centred kernel `K` (py4DSTEM's `template`):

1. optional `sigma_dp` Gaussian smoothing of `P` (scipy `gaussian_filter`, mode `reflect`);
2. `m = FFT2(P) · conj(FFT2(K))`; `cc = |m|^p · exp(i arg m)` (`cc = m` at `corrPower` 1; a bin with `m = 0` is 1 at `p = 0`, as
   NumPy's `0**0`, else 0);
3. `CC = max(Re IFFT2(cc), 0)` (a NaN stays NaN);
4. `BraggMaxima.find` on `CC` (py4DSTEM's `get_maxima_2D`): `sigma_cc` Gaussian smoothing, the 8-neighbour maxima (`>=` toward
   increasing indices, `>` toward decreasing), the edge boundary on the integer maxima, then the filters in py4DSTEM's order:
   absolute intensity, relative intensity (the reference is the maximum of rank `relativeToPeak`), minimum spacing (brightest
   first, a peak is dropped if within the spacing of a kept one), count;
5. subpixel: `.pixel` stops there; `.parabolic` adds the three-point parabola per axis on the smoothed correlation and replaces the
   intensity by its bilinear interpolation at the refined position; `.multicorr` then refines each position on the unsmoothed
   complex `cc` by matrix-DFT upsampling around the half-pixel-rounded position (`CuboParallax`'s `MatrixDFTCorrelation`, the same
   code that pins `upsampled_correlation`), and sorts by the refined intensity.

Equal intensities are ordered by raster position (py4DSTEM's order there is NumPy's unstable sort); the pins contain no ties. `BraggDiskOptions.tieOrder = .numpy` selects NumPy's order instead (larger row, then larger column first), which the strain map's Bragg vector map uses (docs/strain.md).

## What differs in mac4DSTEM, as named options

The primary behaviour is py4DSTEM's. The app (source `c809679c`, `Core/Analysis/DiskDetection.swift`) differs in the places below;
each is an option of `BraggDiskOptions`, off by default, and is pinned to the app source, not to the new Swift.

| Option | py4DSTEM (default) | mac4DSTEM |
|---|---|---|
| `relativeReferenceMinimumRadius` | the reference of the relative cut is the brightest maximum anywhere | when above 0, the brightest maximum at least this far from the brightest one (a saturated direct beam does not set the scale) |
| `nonFinitePixels` | `.propagate`: NaN or infinity spreads through the FFT and the pattern yields no peak | `.neighbourMedian`: each such pixel takes the median of its finite 8-neighbours (0 if none), before smoothing |
| `multicorrOrder` | `.refinedIntensity`: sorted again by the refined intensity | `.correlationIntensity`: left in the order of the integer maxima |
| `BraggDiskOptions.detectorAdapted` | (not in py4DSTEM; its literal spacing 60 and edge 20 suit ~512 px) | spacing = rounded probe radius clamped to [4, max(4, round(min(q)/8))], edge = max(2, trunc(min(q)/24)) |
| `BraggDiskOptions.mac4DSTEM` | | the three switches above plus the app's default `minimumRelativeIntensity` 0.0015 (py4DSTEM 0.005) |

Not replicated, because they are not numerics of the detector or are below its resolution: the Gaussian taps of radius
`max(1, round(4σ))` against scipy's `int(4σ + 0.5)` (they differ only below σ = 0.125), the kernel construction from a measured
probe with a cosine shoulder (kernels are CuboVirtual's), parameter validation and provenance, and the learned-candidate
refinement. The app's Float32 arithmetic is the opt-in `.single` path below (the default stays the float64 definition, the pins
record the Float32 gap); its per-pattern funnel is `BraggFunnel`. `radial_bksb` and `filter_function` are not provided: the caller preprocesses the pattern.

## Single precision (`BraggDiskOptions.precision = .single`)

`BraggPrecision.double` (the default) is everything above, unchanged. `.single` is the same definition in Float32, for speed:
the pattern copy, the non-finite fill, both Gaussian filters (scipy's taps rounded once to Float; `vDSP_conv` along reflect-padded
rows, row-wise `vDSP_vsma` across rows), the 2-D transform (`vDSP_DFT_zop` along the rows, a transpose, the same along the
columns; the spectrum is kept transposed so no transpose is spent on the way back), the correlation product (`vDSP_zvmul`, or the
hybrid power in `powf`), `max(Re, 0)` and the maxima search run in Float. The few accepted peaks are refined in Double: the parabola
from the Float neighbourhood, the matrix-DFT upsampling on the Float complex correlation widened to Double. The reported intensity
is the Float correlation, widened.

- **Sizes.** The transform is the exact N-point DFT at the pattern's own size, so only detector sizes vDSP supports on both axes are
  available: f·2ⁿ with f in 1, 3, 5, 15, ... (64, 128, 192, 240, 256, 320, 384, 480, 512, ...; not 56, 100, 127, 130).
  `BraggDetector.supportsSinglePrecision(detectorRows:detectorColumns:)` asks vDSP itself. Any other size is refused with
  `CuboBraggError.invalidOption` (naming the size), never computed another way; `.double` runs on every size.
- **Scratch.** A detector keeps a pool of scratch buffers (six `rows × columns` Float planes); a call checks one out and returns
  it. `BraggDisks.detectAll` therefore allocates one per concurrent worker, not one per pattern, and with `.single` hands each
  Float pattern straight from the tile to the detector (no widening). The Float plan and kernel spectrum are built on the first
  `.single` call of a detector.
- **Float entry.** `BraggDetector.detect(_:options:)` and `detectWithFunnel(_:options:)` also take an `UnsafeBufferPointer<Float>`
  (`rows · columns` values) for a caller that holds Float patterns; with `.double` it is widened first.
- **Not held: phase correlation.** At `correlationPower` 0 every bin has unit weight, including bins at rounding-noise level, whose
  phase in Float32 is noise; on the pins the peak positions move by up to 11 px. `.single` is for `correlationPower` of about 0.5 and
  above (the tests hold 0.5, 0.75 and 1); use `.double` for phase correlation.
- **Faint peaks.** The Float transform's absolute error is about 1e-6 of the correlation maximum, so a peak's accuracy falls with its
  height: below 1e-3 of the brightest peak the gaps grow (the pins with the spacing filter off reach peaks of 4e-5).

Measured against py4DSTEM and `.double` (`Tests/CuboBraggTests/SinglePrecisionTests.swift`, the 64 × 64 pins; all counts and
orders equal; case `correlationPower` 0 excluded):

| Quantity | Measured maximum | Tolerance (per quantity) |
|---|---|---|
| Position, peaks above 1e-3 of the brightest, against py4DSTEM | 1.3e-5 px | 5e-5 px |
| Intensity, same peaks, relative | 1.1e-6 (py4DSTEM), 1.8e-6 (`.double`) | 5e-6 |
| Position, fainter peaks (4e-5 to 1e-3 of the brightest) | 4.4e-4 px | 1e-3 px |
| Intensity, fainter peaks, relative | 2.4e-4 | 5e-4 |
| Real correlation map, fraction of its maximum (`correlationPower` 0.5 to 1, with σ) | 2.1e-6 | 1e-5 |
| Complex product `cc`, fraction of its largest magnitude | 1.2e-5 | 3e-5 |
| Non-square and wide-filter grids (64 × 48, 48 × 64, 16 × 64, 16 × 16 with σ 5, 32 × 80) map / product | 4.0e-7 / 2.5e-7 | as above |
| 256 × 256 benchmark scene, position / relative intensity, poly | 1.0e-5 px / 3.6e-7 | |

The app's own tolerance against py4DSTEM is 2e-4 px; `.single` is inside it by a factor of 15 for peaks of any usable height.

### Timing (256 × 256, Apple M5 Pro, 18 cores)

`CUBOKIT_BENCH=1 swift test -c release --filter BraggBenchmark` (a 9-disk synthetic scene with noise, `BraggDiskOptions.mac4DSTEM`,
edge 10, spacing 12, sigma 2). Wall ms per pattern.

| Path | `.parabolic` serial | `.parabolic` `detectAll`, 18 cores | `.multicorr` serial | `.multicorr` `detectAll`, 18 cores |
|---|---|---|---|---|
| `.double` | 5.31 | 0.87 | 40.2 | 9.7 |
| `.single` | **0.47** | **0.140** | 35.3 | 9.3 |
| mac4DSTEM's Float path (open-items, 2026-10-20, its own scene) | 2.05 | 0.24 to 0.27 | | about 21 |

The app was not re-run here (its working tree is read-only from this repository), so its row is the figure its lane measured
(4096 positions, 18 workers), not a same-run comparison; `.single` is ahead of it on `.parabolic` by a wide margin on this
machine class. With `.multicorr` the time is the matrix-DFT refinement in Double (`CuboParallax`, about 4 ms per peak; a Float matrix DFT there is the next speed-up), not the
correlation, which is why the two precisions are close there.

## Funnel

`BraggDetector.detectWithFunnel(_:options:)` returns the peaks of `detect` (identical) and a `BraggFunnel`: the number of maxima
after each stage in py4DSTEM's order (local maxima inside the edge boundary, absolute cut, relative cut, spacing, accepted),
whether the peak cap dropped any, the relative cut's reference intensity (nil when the cut is off or no maximum of rank
`relativeToPeak` exists; `relativeReferenceWasAvailable` tells which), the maximum of the smoothed correlation, and the number of
non-finite pixels `.neighbourMedian` replaced. `BraggMaxima.findWithFunnel` is the same on a correlation map the caller supplies
(its `nonFinitePixelCount` is 0). The fields mirror mac4DSTEM's `DiskDetectionPatternDiagnostics`. In `.single`, the maxima
count can differ from `.double`'s by maxima that are rounding noise of the zero-valued parts of the map; counts after the relative cut
agree on the pins.

## Non-finite pixels

`BraggDetector.fillNonFinite(_:rows:columns:)` (a Double or a Float image, in place; returns the count replaced) is the app's
fill: each NaN or infinite pixel becomes the median of its finite 8-neighbours, 0 when it has none, neighbours read before any
replacement. `BraggDiskOptions.nonFinitePixels = .neighbourMedian` applies it before any smoothing; it is public for callers that
fill elsewhere (the origin calibration's Friedel path). A finite image is untouched.

## Probe ring hint

`ProbeRingHint.outerEdge(of:)` is the app's suggestion for the radius of a trench kernel when the probe is a bullseye: the outer
edge of the ring structure of the mean pattern. It has no py4DSTEM counterpart and is pinned to the app source
(`ProbeRingHint.swift` at `c809679c`), with the probe size from py4DSTEM's `get_probe_size` (the kit's `ProbeSize.estimate` is
bit for bit the same) feeding the app's code. The 20 % edge, the 0.25 dip and the guard `outer <= min(q)/8` are the app's, measured
on its datasets: a hint to offer, not a default.

## Pins and tolerances

`tools/bragg/pins.py` runs two oracles and writes `Tests/CuboBraggTests/Fixtures/bragg.{json,f64}` (519 KB; the scenes are synthetic
logistic-edge disks at fractional positions with seeded noise, three scenes: 64 × 64, 40 × 56 which is Bluestein on both axes, and a
variant scene):

- py4DSTEM 0.14.19 (`f050d207`), the public `find_Bragg_disks`: 30 cases covering pixel, poly and multicorr (factors 3, 8, 16), corr
  power 1, 0.5 and 0, `sigma_dp`, `sigma_cc` 0, absolute, relative and ranked-relative cuts, spacing, edge 0, 1, 5 and 30, the peak
  cap, a non-square non-power-of-two detector, and a pattern with NaN and infinity (no peaks). The generator asserts there are no
  tied intensities.
- the app source, compiled unchanged apart from access qualifiers and shims (`ProbeKernel`'s fields, a probe-size stub fed from
  py4DSTEM), linked with ImagoKit 0.1.1's Float32 `FFT2D`: ten cases including the three options above, which the generator asserts
  change the answer.

Measured gaps (peak counts and order equal in every case): py4DSTEM, position 1.5e-12 px and relative intensity 1.3e-13, tolerance
1e-9 px and 1e-10; app, position 1.8e-4 px and relative intensity 1.2e-6 (Float32), tolerance 1e-3 px and 1e-5. The Gaussian filter
gained an `Edge` mode (`reflect` is scipy's default and what `find_Bragg_disks` uses; `nearest` stays the default of the origin
measurement), pinned to scipy in `GaussianEdgeTests`.

Regenerate from a suite worktree:

```sh
../../tools/memcap.sh -- ../../References/python/py4dstem/bin/python -I -B tools/bragg/pins.py
```
