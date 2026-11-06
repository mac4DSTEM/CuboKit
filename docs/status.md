# CuboKit status (0.11.0, 2026-11-06)

Live and tested: the cube model (eager, memory-mapped, `ViewedCube` crop and bin, `ResidentCube` held in a Metal buffer),
the calibration model (`Calibration`, `ScanField`, coordinate axes and grids, `CalibrationMeasurement`), the cube edits
(`CubeEditing`: reshape, swaps, `bin_R`, `thin_R`, `pad_Q`, `resample_Q` bilinear, masked-pixel median), virtual images,
mean, max, median and selected-area patterns (CPU and Metal), centre of mass, origins (the iterated window with a
brightest-disk seed at low dose, and from the Bragg peaks with a symmetry centre guess: `measure_origin`, `fit_origin`), probe size, DPC and iDPC, strain,
Bragg disks and masks (`BraggMasking`, `.single` on every detector size), Bragg-vector classification and virtual images from Bragg vectors, the polar cube and radial background, the ellipse fits, Fourier resampling, parallax (Float32 shifter for every size), single-slice ptychography (`.single` on every detector size), PCA, NMF, GMM and k-means; cancel and progress on the long reducers and the strain map; `ResidentCube` capped at half of the physical memory, with the reason.

Since 0.9.0: 0.10.0 (2026-11-01) removed the enums named like their modules (`CuboPtycho`, `CuboStrain`, `CuboEmbedding`) and their deprecated scope, gave `SyntheticProbe` a centre and `FloatFFT2D.supports` public, and made the CPU mean and max patterns, the Metal reducers on a non-resident cube and the multicorr Bragg refinement 2 to 5 times faster; 0.11.0 (2026-11-05) added the origin from the Bragg peaks. Details and the migration notes: `CHANGELOG.md`.

Parity with py4DSTEM 0.14.19 (`f050d207`): 135 present, 38 partial, 42 missing of 215 counted rows (`docs/parity.md`).
Timings against py4DSTEM at three cube sizes: `docs/benchmarks.md`. What is open: the GitHub issues; known limits: `docs/parity.md`.

Tests (split 2026-10-21): the real-data comparisons against the pinned float64 / py4DSTEM exports (virtual, centre of mass, origin fit and probes, in `$CUBOKIT_REAL_DATA`, skipped by name when absent) sit in `CuboKitParityTests`; the pins, origin-fit and probe checks and helpers they share with the fast tests are in `CuboKitTestSupport`. `swift test -c release --parallel --skip CuboKitParityTests` is the per-session gate; CI runs the full suite on every push (the real-data tests skip there). At 0.9.0: 331 tests (310 fast + 21 parity); Mac, release, `--parallel`: fast target 13 s, full suite 3 min 38 s.
