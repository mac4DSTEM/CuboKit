# CuboKit open items

Slower than py4DSTEM at the typical size (100 x 100 x 64 x 64, Float32); the numbers are in `docs/benchmarks.md`.

1. **Mean pattern on the CPU, 1.8x NumPy** (`VirtualDiffraction.mean`: 4.9 ms against 2.7 ms on a loaded Mac, best 4.3 against 2.6; the max pattern is 5.9 ms against 2.8). Since 0.9.0 the sum is a SIMD TwoSum across cores and the pass is bandwidth-bound (about 40 GB/s); NumPy's float32 reduction reads the cube once without widening. A resident cube on Metal is ahead (3.4 ms). Reading Float32 pairs without widening to Double is not an option (D16), so only a resident cube or the same sum on Float32 pairs (`df_add`) goes lower.
2. **Metal mean, max and selected-area patterns lose to the CPU on a cube that is not resident.** Since CuboKit#8 a
   `MemoryCube` reaches the GPU without a copy and other sources go through one reused staging buffer (before, a new
   buffer per tile cost about 20 of the 26 ms). At the typical size the Metal mean and max is now 6.4 ms (staged 13.4,
   resident 3.8) against 2.4 ms on the CPU, and the mask mean 9.7 against 1.6: these passes are bandwidth-bound and the
   CPU reads the cube in place, so the CPU stays the default. The virtual image (5.0 against 9.6 ms) and the centre of
   mass (8.6 against 11.6) are now faster on Metal for a `MemoryCube`; the centre of mass at 128 x 128 detectors is not
   (29.8 against 17.5: the kernel reads each pattern with one thread). An automatic choice of path is the app's.

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
