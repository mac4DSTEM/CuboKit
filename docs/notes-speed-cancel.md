# Notes for the merge: speed and cancellation (suite task 19)

For the session that merges CHANGELOG.md, docs/open-items.md, docs/status.md, docs/parity.md and README.md. Nothing in those files was touched by this slice.
Version: written as 0.9.0 in code comments and docs/benchmarks.md (change it if the merge numbers it differently).

## CHANGELOG bullets (paste under the new version; none moves a number)

Speed (results bit for bit those of 0.8.1, D16 reference unchanged):

- **CuboVirtual:** the CPU mean pattern (`VirtualDiffraction.mean`, `meanAndMax`, `pattern(.mean)` over `.all`) and the selected-area mean over a mask or weights add with a branch-free SIMD TwoSum (four pixels per step, no per-element finite check; the check moved to a post-pass on the result, so a NaN or Inf in the input propagates as before and a column of -0 keeps its sign). The detector pixels split into 512-pixel blocks across every core (4096 pixels used to make two blocks), and a `MemoryCube` is read in place. 100 x 100 x 64 x 64 Float32, loaded Mac: mean 19.2 to 4.9 ms (NumPy 2.7), mask 43.1 to 3.9 ms (NumPy 10.0), weights 54.9 to 5.1 ms (py4DSTEM 157).
- **CuboCore `CubeEditing.binScan`** sums four detector pixels per step and splits the output positions across cores (28.5 to 2.4 ms; NumPy 7.0), reading several output rows per tile. `medianFilterMaskedPixels` filters patterns across cores with a fixed small buffer per window instead of a sorted array per masked pixel and pattern (1.74 to 0.29 ms at 64 x 64 scan; NumPy 0.64). Both are bit for bit the earlier loops.

Added (all new parameters default, so every existing call compiles):

- **`isCancelled: (@Sendable () -> Bool)? = nil` and `progress: (@Sendable (Double) -> Void)? = nil`** on `StrainMapping.compute` (after `options:`; the Bragg vector map takes the first tenth of the progress when it is built, the local fits the bulk) and `StrainMapping.histogram`. In the style of `RotationSolver.curve`: a true answer throws `CancellationError` and nothing partial is returned; progress never decreases and the last call is 1.
- **`isCancelled:` on the whole-cube reducers:** `VirtualImaging.image`, `VirtualDiffraction.mean`, `max`, `meanAndMax`, `pattern`, `CenterOfMass.measure`, `OriginMeasurement.measure(of:radius:radiusScale:estimator:...)`, asked before each tile, same semantics (`progress:` already existed on all but `pattern`, which gained it, covering the mask, weights, median and shifted paths). A cancelled Swift task still stops them as before.
- **`LatticeStrain.fitRobust` accepts intensities of 1e10 and above:** when the largest intensity is 1e10 or more every intensity is first divided by it in Double, so the Float32 fit no longer sees the overflow or the lost small weights; below 1e10 the weights are exactly those of 0.8.1. Non-finite and negative intensities, and coordinates or indices of 1e10 and above, are still refused.
- `tools/benchmarks/bench.py speed` and `compare`, `Tests/CuboVirtualTests/SpeedBench.swift` (the alternating-rounds timing and its bit hash).

### Migration (text for the entry)

Additive. A call that names its arguments compiles unchanged. One behaviour change: `LatticeStrain.fitRobust` (and so `StrainMapping.compute` with `fitMethod: .robustIntensityOnce`) fitted nothing for a peak intensity of 1e10 or more (it threw `invalidInput`; the map counted a refused fit); it now fits. An app that used that refusal as a flag must test the intensity itself. In mac4DSTEM, `StrainMapping.compute` can be given the app's cancel flag and a progress callback; the rescale before `fitRobust` that the app does for 1e10 and above can go.

## docs/open-items.md

Delete item 1 (the CPU mean pattern is no longer 8x behind NumPy; see below for what remains).
Delete item 3 entirely (`binScan` and `medianFilterMaskedPixels` are faster than py4DSTEM now). Keep its last sentence ("`bin_Q` (2.8x), `pad_Q` (3.9x), `thin_R` (2.3x), `resample_Q` (19x) and `dp_median` (3.9x) are faster than py4DSTEM") if wanted, or move it to docs/benchmarks.md.
Replace the lead sentence ("Slower than py4DSTEM at the typical size ... most slipped first") with: "Slower than py4DSTEM at the typical size (100 x 100 x 64 x 64, Float32); the numbers are in `docs/benchmarks.md`."
Add as the new item 1 (renumber the rest):

> 1. **Mean pattern on the CPU, 1.8x NumPy** (`VirtualDiffraction.mean`: 4.9 ms against 2.7 ms on a loaded Mac, best 4.3 against 2.6; the max pattern is 5.9 ms against 2.8). Since 0.9.0 the sum is a SIMD TwoSum across cores and the pass is bandwidth-bound (about 40 GB/s); NumPy's float32 reduction reads the cube once without widening. A resident cube on Metal is ahead (3.4 ms). What would change it: reading Float32 pairs without widening to Double is not an option (D16), so only a resident cube or the same sum on Float32 pairs (`df_add`) goes lower.

Item 2 (Metal host copy per tile) stays; add: "The CPU path is now faster than the host-copy Metal path for the mean (4.9 against 27 ms), the mask (3.9 against 29) and the weights (5.1 against 28): the CPU is the default and the host-copy Metal path should only be asked for by a caller that wants to test it."
Item 4 (centre of mass on Metal) stays.

Under "Missing", the bullet "The app asks that remain" loses `LatticeStrain.fitRobust` for intensities >= 1e10 and "cancel and progress on `StrainMapping.compute`".

## docs/status.md, docs/parity.md

No row changes (no number moved). status.md could say: "cancel and progress on StrainMapping.compute and the whole-cube reducers".

## Before / after / Python (docs/benchmarks.md has the full section)

Three alternating rounds (old sources, new sources, Python), 15 calls after one warm-up, median of the call medians, ms; the Mac was shared (load 9 to 13).

| case | before | after | Python | after vs Python |
|---|---|---|---|---|
| dp_mean, 100x100x64x64 | 19.2 | 4.9 | 2.7 | 0.56x |
| selected-area mean, mask | 43.1 | 3.9 | 10.0 | 2.6x |
| selected-area mean, weights | 54.9 | 5.1 | 157.2 | 31x |
| bin_R(2), typical | 28.5 | 2.4 | 7.0 | 2.9x |
| bin_R(2), 64x64x32x32 | 2.89 | 0.63 | 0.72 | 1.1x |
| median_filter_masked_pixels, 64x64x32x32 | 1.74 | 0.29 | 0.64 | 2.2x |

Numbers: the hash of the result bits is identical before and after for all six cases (existing pin `ParallelReducerTests` also passes unchanged); new tests compare against a transcription of the old scalar loops on random, NaN, +Inf, -Inf, +Inf with -Inf and -0 columns. Against NumPy on the same cube, mask and weights the kit differs by NumPy's own Float32 accumulation error: mean 3.9e-6 and mask 3.1e-6 relative to the largest value (kit against the float64 truth: 3e-18 and 0), weights 1.9e-16 (NumPy accumulates in float64 there), `bin_R` 7.3e-8 (one Float32 rounding), the masked-pixel median equal bit for bit.

## Deviations and things to know

- Task-local plumbing: `isCancelled:` on the CuboVirtual reducers is handed to the passes through a `@TaskLocal` (`Interruption.swift`) rather than as a parameter on every internal function; the public signatures are as asked. `Task.checkCancellation()` in those passes became `checkInterrupted()`, which does both checks.
- Two existing tests changed because they encoded the old refusal at 1e12: `robustFitScalesBrightWeightsAndHandlesFewPeaks` no longer expects 1e12 to throw (the new tests `RobustLargeIntensityTests` cover 1e10 to 1e300), and the 9th scan position in `scan()` carries a negative intensity (still refused) instead of 1e12; `defaultsKeepTheNearestDirectionPath` sets that one peak to 1e12 itself because the py4DSTEM fit accepts it.
- Dropped by measurement: a register-resident 16-pixel accumulator (1.5x slower than the row-contiguous loop) and other block sizes (128 to 1024 within noise).
- `dp_mean` is still 1.8x behind NumPy (see the open-items text above); the target "within 1.2x" was not reached on a loaded Mac, the 8x gap is gone.
- The `-0` handling: the old loop skipped zero corrections, so an all(-0) column kept -0; the TwoSum adds +0 corrections, so `CompensatedSum.value` returns the sum itself for a -0 sum with a zero correction. Tested.
- `MemoryCube` is read in place by `withTileValues` / `withScanRows` (no copy per tile); other sources still go through `readScanTile`. No tile is held after the call returns.
- `CubeEditing.swift` edits are local to `binScan` and `medianFilterMaskedPixels` (their bodies); the new kernels are in `Sources/CuboCore/CubeEditingParallel.swift`. The Fourier `resampleQ` slice only needs to avoid those two bodies.
- Warnings: my files compile without warnings; two older test files carry warnings that predate this slice (`Tests/CuboBraggTests/SinglePrecisionTests.swift:182`, `Tests/CuboVirtualTests/SelectedAreaMetalTests.swift:30`), left alone.
- Checks run: `swift test -c release --parallel --skip CuboKitParityTests` green, `tools/consumer-check/run.sh` 143 checks pass. The parity target was not run (no number moved; it compares the same bits).
