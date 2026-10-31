# CuboKit open items

Slower than py4DSTEM at the typical size (100 x 100 x 64 x 64, Float32); the numbers are in `docs/benchmarks.md`.

1. **Metal mean, max and selected-area patterns pay a host copy per tile** (28 ms against 3.4 ms resident). Use
   `ResidentCube` when a cube is reduced more than once; an automatic policy (hold the cube when the first call reads it
   twice) is the app's. The CPU path is faster than the host-copy Metal path for the mean (2.2 against 27 ms), the mask
   and the weights (3.9 and 5.1 against 29 and 28 ms before the 1.85x of CuboKit#55): the CPU is the default.
2. Centre of mass on Metal is slower than the CPU at both sizes measured (27 ms against 11 ms): the kernel reads each
   pattern with one thread; only a resident cube would change that. CPU stays the default.

Closed: the CPU mean pattern is no longer behind NumPy (CuboKit#7). Since CuboKit#53 each pixel keeps its TwoSum sum and
correction in registers over four scan positions per pass, in 128-pixel blocks: `VirtualDiffraction.mean` 4.47 -> 2.17 ms
(2.1x) against NumPy's 2.7, the max pattern 2.24 -> 1.67 ms, `meanAndMax` 5.73 -> 2.45 ms (resident 100 x 100 x 64 x 64
Float32, `MeanMaxBench`, median of 15). CuboKit#55 put the masked and weighted mean on the same pass (about 1.85x faster;
a scattered half mask stays near 1.8 ms, bound by memory access). Results are bit for bit unchanged.

Missing, in the order the study needs them (`docs/parity.md`, "study" rows):

- The Double FFT on lengths vDSP does not take (Bluestein, the padded parallax sizes and ptychography detectors such as
  165) is bound by vDSP's Double FFT: a vectorised, allocation-free Bluestein on vDSP's DFT measured 1.1x and was not
  kept. Float32 callers use `ParallaxFourierShifter.shiftedSingle`, which takes every size (2.6x at 165 x 165). The
  hot-pixel mean stays Float32 on purpose (`HotPixelFilter` documents why).
- A single `reconstruct` for parallax; ptychography probe constraints, mixed-state and multislice engines.

Decisions to keep an eye on:

- `Calibration` follows py4DSTEM: binning, thinning, padding and resampling change the pixel sizes but not the origin (in
  detector pixels), so an origin measured before `bin_Q(2)` is wrong afterwards. A caller that stores the origin rescales it.
- `ResidentCube` is capped at half of the physical memory and needs the cube in one Metal buffer (`ResidentCube.availability(of:)` says which limit and how far over); a larger cube needs a chunked variant (decided 2026-10-22: no chunked variant; the app falls back to the tiled path).
- The Metal sums are Float32 within a tile: a per-tile sum or a product past about 3.4e38, or a selection weight below the
  Float32 range (it rounds to 0), gives ±∞ or NaN where the CPU Double path stays finite. Not reachable with detector counts.

0.9.0 notes (polar cube, ellipse, classification):

- The polar cube's radius is on the minor-axis scale (`rr = (b / a) * hypot`), the radial background's lookup radius on the major-axis scale. Both are py4DSTEM's and are ported as they are; for `a != b` the background profile is stretched by `a / b` against the polar bins. A pull request upstream (or a deliberate deviation here) is Paul's call; `testRadialBackgroundOfACircle` shows the circle agreeing with the planted exponential to 0.2%.
- The amorphous-ring fit is chaotic at rounding level in its poorly determined parameters (py4DSTEM's own result moves by up to 7e-4 in the axes when the start changes by one ulp on the `tilted` case); tests compare ellipses at 1e-6 and the residual sum of squares at 1e-6 relative, not parameter by parameter.
- `PolarCube` keeps the whole cube in Double when asked for (`PolarCubeData`: scan x 180 x ~180 doubles per pattern); a lazy `CubeSource` adapter or a Float32 variant is for the consumer that needs it. The polar cube and radial background are CPU only (1024 patterns of 256 x 256 in 0.26 s); a Metal path would be bound by the host copy.
- The NMF, the Gaussian filter, the median and the SVD are shared since #14 (CuboBragg's `lstsq` runs on CuboEmbedding's LAPACK SVD), except `Featurization`'s feature median (CuboEmbedding does not depend on CuboCore) and the sorted-window medians of `HotPixelFilter` and `medianFilterMaskedPixels`.
- `Featurization.from_braggvectors`, ICA, `RobustScaler`, `spatial_separation` and `consensus` are not ported. The GMM k-means start is a transcription of scikit-learn 1.4.2's Lloyd loop with the data centred as 1.9.1 does (every pin agrees on labels). Upstream `NMF(init=None)` with `random_state=None` is not bit repeatable (an unseeded randomised SVD).
- `fit_ellipse_1D` with default `fitradii` raises `TypeError` upstream; the kit's default is `(0, min(shape)/2)`. The DocC landing pages list the new public names; `tools/consumer-check` does not exercise them yet.
- DiffractoKit follow-up: consume the polar cube (`PolarCube`, `PolarCubeData`) and `EllipseFit` for the ring and ellipse work; no code there yet.
