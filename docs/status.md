# CuboKit status (0.8.1, 2026-10-21)

Live and tested: the cube model (eager, memory-mapped, `ViewedCube` crop and bin, `ResidentCube` held in a Metal buffer),
the calibration model (`Calibration`, `ScanField`, coordinate axes and grids, `CalibrationMeasurement`), the cube edits
(`CubeEditing`: reshape, swaps, `bin_R`, `thin_R`, `pad_Q`, `resample_Q` bilinear, masked-pixel median), virtual images,
mean, max, median and selected-area patterns (CPU and Metal), centre of mass, origins, probe size, DPC and iDPC, strain,
Bragg disks and masks (`BraggMasking`), parallax, single-slice ptychography, PCA and k-means.

Parity with py4DSTEM 0.14.19 (`f050d207`): 121 present, 38 partial, 50 missing of 209 counted rows (`docs/parity.md`).
Timings against py4DSTEM at three cube sizes: `docs/benchmarks.md`. What is next: `docs/open-items.md`.

Tests (2026-10-21): the real-data comparisons against the pinned float64 / py4DSTEM exports (slices 1 to 4, in `$CUBOKIT_REAL_DATA`, skipped by name when absent) sit in `CuboKitParityTests`; the pins, S3/S4 checks and helpers they share with the fast tests are in `CuboKitTestSupport`. 331 tests before and after (310 fast + 21 parity). Mac, release, `--parallel`: fast target (`--skip CuboKitParityTests`) 13 s, full suite 3 min 38 s; before the split the one full suite took 4 min 3 s. CI keeps running the full suite (the real-data tests skip there).
