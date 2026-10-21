# CuboKit open items

Slower than py4DSTEM at the typical size (100 x 100 x 64 x 64, Float32), most slipped first; the numbers are in
`docs/benchmarks.md`.

1. **Mean pattern on the CPU, 8x slower than NumPy** (`VirtualDiffraction.mean`: 22 ms against 2.8 ms; the selected-area
   mask 48 ms against 11 ms; the max pattern is 5.9 ms, 2x NumPy's, since the block-parallel loop of 0.8.0 took it from
   35 ms). Cause, measured: the Neumaier loop adds every element in Double with a data-dependent branch, about 1.2 G
   elements/s per core, and the parallel loop gains little while the Mac is loaded (load 10 to 11). Evidence that the
   traversal is not the limit: the same cube in a `ResidentCube` runs the Metal path in 3.4 ms. What would refute it: a
   branch-free SIMD compensated sum that still takes 20 ms. Fix: a SIMD TwoSum without the finite check (the check moves
   to a post-pass on the result), bit identical or within NumPy's own float32 accumulation error.
2. **Metal mean, max and selected-area patterns pay a host copy per tile** (28 ms against 3.4 ms resident). Use
   `ResidentCube` when a cube is reduced more than once; an automatic policy (hold the cube when the first call reads it
   twice) is the app's.
3. **`CubeEditing.binScan` 4.6x slower than NumPy** (30 ms against 6.4 ms at the typical size) and
   **`medianFilterMaskedPixels` 2x slower at 64 x 64 scan** (1.7 ms against 0.9 ms; faster at 32 x 32): Double accumulation
   per element in a scalar loop, and a sorted scratch array per masked pixel and pattern; NumPy vectorises over the scan.
   The split across cores applies to the first; the second needs the median taken across patterns for one pixel at a time.
   `bin_Q` (2.8x), `pad_Q` (3.9x), `thin_R` (2.3x), `resample_Q` (19x) and `dp_median` (3.9x) are faster than py4DSTEM.
4. Centre of mass on Metal is slower than the CPU at both sizes measured (27 ms against 11 ms): the kernel reads each
   pattern with one thread; only a resident cube would change that. CPU stays the default.

Missing, in the order the study needs them (`docs/parity.md`, "study" rows):

- `resample_Q(method="fourier")` and `fourier_resample`.
- `get_radial_bkgrnd`, `get_radial_bksb_dp` and `get_local_ave_dp(radial_bksb=True)`: they need the polar data cube, which
  is DiffractoKit's `PolarDatacube`.
- `fit_ellipse_1D`, `fit_ellipse_amorphous_ring`, `constrain_degenerate_ellipse` (the ellipse fit that feeds `Calibration.ellipse`).
- The app asks that remain: `LatticeStrain.fitRobust` for intensities >= 1e10 (rescale first), cancel and progress on
  `StrainMapping.compute`, a vDSP-backed shifter for padded parallax sizes, `SyntheticProbe.make` with a centre, the
  hot-pixel mean in Double, `svd_flip` signs in embedding, `FloatFFT2D.supports` public, names that clash with module names.
- A single `reconstruct` for parallax; ptychography probe constraints, mixed-state and multislice engines.

Decisions to keep an eye on:

- `Calibration` follows py4DSTEM: binning, thinning, padding and resampling change the pixel sizes but not the origin (in
  detector pixels), so an origin measured before `bin_Q(2)` is wrong afterwards. A caller that stores the origin rescales it.
- `ResidentCube` needs the cube in one Metal buffer (`ResidentCube.fits`); a larger cube needs a chunked variant.
- `cubo_weighted_pattern` (and `cubo_sum_and_max`) use a plain TwoSum: a +Inf in a selected pattern gives NaN on Metal where the
  CPU gives +Inf. Only hot-pixel data with infinities is affected; `df_add` keeps a non-finite sum and could replace it.
