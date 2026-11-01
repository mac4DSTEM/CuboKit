# CuboKit status (0.9.0, 2026-10-22)

Live and tested: the cube model (eager, memory-mapped, `ViewedCube` crop and bin, `ResidentCube` held in a Metal buffer),
the calibration model (`Calibration`, `ScanField`, coordinate axes and grids, `CalibrationMeasurement`), the cube edits
(`CubeEditing`: reshape, swaps, `bin_R`, `thin_R`, `pad_Q`, `resample_Q` bilinear, masked-pixel median), virtual images,
mean, max, median and selected-area patterns (CPU and Metal), centre of mass, origins, probe size, DPC and iDPC, strain,
Bragg disks and masks (`BraggMasking`), Bragg-vector classification and virtual images from Bragg vectors, the polar cube and radial background, the ellipse fits, Fourier resampling, parallax, single-slice ptychography, PCA, NMF, GMM and k-means; cancel and progress on the long reducers and the strain map; `ResidentCube` capped at half of the physical memory, with the reason.

Parity with py4DSTEM 0.14.19 (`f050d207`): 135 present, 38 partial, 42 missing of 215 counted rows (`docs/parity.md`).
Timings against py4DSTEM at three cube sizes: `docs/benchmarks.md`. What is open: the GitHub issues; known limits: `docs/parity.md`.

Tests (2026-10-21): the real-data comparisons against the pinned float64 / py4DSTEM exports (virtual, centre of mass, origin fit and probes, in `$CUBOKIT_REAL_DATA`, skipped by name when absent) sit in `CuboKitParityTests`; the pins, origin-fit and probe checks and helpers they share with the fast tests are in `CuboKitTestSupport`. 331 tests before and after (310 fast + 21 parity). Mac, release, `--parallel`: fast target (`--skip CuboKitParityTests`) 13 s, full suite 3 min 38 s; before the split the one full suite took 4 min 3 s. CI keeps running the full suite (the real-data tests skip there).
