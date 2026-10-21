# Benchmarks

Kit against py4DSTEM, same inputs, Float32 cubes, one run of `tools/benchmarks` on 2026-10-21. Harness: `Tests/CuboVirtualTests/BenchmarkHarness.swift`
(Swift, built with `swift test -c release`, skipped unless `CUBOKIT_BENCH_DIR` is set) and `tools/benchmarks/bench.py` (the py4DSTEM side
and the merge; the commands are in its docstring). Both read the same `.npy` cubes. Best of five after one warm-up; cells are
best / median (spread = slowest - best) in milliseconds; "CPU vs py4DSTEM" is py4DSTEM's best divided by the kit CPU's best
(above 1 the kit is faster). The cubes are held in memory (`MemoryCube`; `DataCube(data=ndarray)`). The Metal columns copy each
tile to the GPU on every call; the `_resident` rows hold the cube in a `ResidentCube` (one Metal buffer) first.

Sizes (scan rows x columns x detector rows x columns): small 32x32x32x32 (1.0 M values), mid 64x64x32x32 (4.2 M), typical
100x100x64x64 (41.9 M, 168 MB). The large size (330x330x64x64, 1.8 GB) and the Bragg, strain, DPC, parallax and ptychography
functions are the nightly review's; this first table is what a Mac can measure while five other sessions use it.
The machine was shared during the run (load average 5 to 11 from other sessions' builds), so the spreads are wider than on a quiet Mac.

Machine: Apple M5 Pro, macOS 27.0.1; Swift 6.4; py4DSTEM 0.14.19, NumPy 2.5.4, Python 3.12.14.
Float32 cubes, best of five after one warm-up; cells are best / median (spread) in ms.

| function | size | kit CPU | kit Metal | py4DSTEM | CPU vs py4DSTEM |
|---|---|---|---|---|---|
| bin_Q_2 | small | 1.3 / 1.3 (0.2) | — | 3.4 / 3.6 (0.8) | 2.64x |
| bin_Q_2 | mid | 5.4 / 5.6 (0.9) | — | 12.9 / 13.1 (0.2) | 2.40x |
| bin_Q_2 | typical | 54.0 / 55.3 (2.7) | — | 116.5 / 120.0 (4.2) | 2.16x |
| bin_R_2 | small | 0.8 / 0.9 (0.1) | — | 0.4 / 0.5 (0.0) | 0.53x |
| bin_R_2 | mid | 3.2 / 3.4 (0.3) | — | 0.7 / 0.8 (0.6) | 0.22x |
| bin_R_2 | typical | 32.3 / 33.0 (1.5) | — | 6.4 / 6.8 (2.0) | 0.20x |
| center_of_mass | small | 0.4 / 0.6 (0.5) | 1.6 / 1.6 (0.5) | — | — |
| center_of_mass | mid | 1.2 / 1.7 (0.6) | 2.1 / 2.3 (1.7) | — | — |
| center_of_mass | typical | 12.8 / 13.2 (0.8) | 31.4 / 31.9 (0.9) | — | — |
| dp_max | small | 0.1 / 0.1 (0.0) | 1.5 / 1.6 (0.5) | 0.2 / 0.2 (0.0) | 2.55x |
| dp_max | mid | 0.3 / 0.3 (0.0) | 3.5 / 3.5 (0.2) | 0.4 / 0.4 (0.0) | 1.48x |
| dp_max | typical | 5.9 / 5.9 (0.4) | 28.7 / 29.1 (0.6) | 2.8 / 3.1 (3.9) | 0.48x |
| dp_mean | small | 0.8 / 0.8 (0.4) | 1.5 / 1.8 (0.7) | 0.2 / 0.2 (0.1) | 0.23x |
| dp_mean | mid | 3.4 / 3.8 (0.7) | 3.4 / 3.5 (0.2) | 0.4 / 0.4 (0.1) | 0.12x |
| dp_mean | typical | 22.0 / 23.6 (1.8) | 27.5 / 28.7 (3.1) | 2.8 / 3.2 (1.2) | 0.13x |
| dp_mean_resident | small | — | 0.5 / 0.5 (0.5) | — | — |
| dp_mean_resident | mid | — | 1.6 / 1.7 (0.1) | — | — |
| dp_mean_resident | typical | — | 3.4 / 3.5 (0.2) | — | — |
| dp_median | small | 8.6 / 9.1 (0.8) | — | 13.8 / 14.2 (0.4) | 1.61x |
| dp_median | mid | 36.8 / 38.3 (4.9) | — | 65.3 / 68.0 (5.1) | 1.77x |
| dp_median | typical | 233.4 / 251.8 (34.5) | — | 785.1 / 821.1 (51.6) | 3.36x |
| median_masked_pixels | small | 0.5 / 0.5 (0.0) | — | 0.5 / 0.8 (0.8) | 1.07x |
| median_masked_pixels | mid | 2.0 / 2.1 (0.2) | — | 0.9 / 0.9 (0.3) | 0.42x |
| pad_Q_2 | small | 0.8 / 0.8 (0.2) | — | 2.1 / 2.3 (0.4) | 2.72x |
| pad_Q_2 | mid | 2.9 / 3.0 (0.3) | — | 8.3 / 8.8 (0.7) | 2.86x |
| pad_Q_2 | typical | 34.6 / 46.3 (23.0) | — | 98.3 / 100.5 (10.6) | 2.84x |
| resample_Q_half | small | 0.6 / 0.6 (0.2) | — | 10.0 / 10.8 (1.1) | 16.94x |
| resample_Q_half | mid | 2.4 / 2.4 (0.1) | — | 39.3 / 45.6 (16.0) | 16.51x |
| selected_area_mean_mask | small | 1.3 / 1.4 (0.1) | 1.7 / 1.9 (0.3) | 0.5 / 0.6 (0.5) | 0.38x |
| selected_area_mean_mask | mid | 4.5 / 5.1 (1.1) | 3.7 / 3.9 (0.3) | 1.3 / 1.4 (0.2) | 0.29x |
| selected_area_mean_mask | typical | 47.6 / 49.4 (2.4) | 29.0 / 30.3 (2.4) | 10.8 / 12.8 (2.8) | 0.23x |
| selected_area_mean_mask_resident | small | — | 1.2 / 1.3 (0.3) | — | — |
| selected_area_mean_mask_resident | mid | — | 1.8 / 2.0 (0.3) | — | — |
| selected_area_mean_mask_resident | typical | — | 4.7 / 6.2 (6.0) | — | — |
| selected_area_mean_weights | small | 1.5 / 1.7 (0.2) | 1.5 / 1.7 (0.4) | 2.9 / 3.1 (0.6) | 1.89x |
| selected_area_mean_weights | mid | 5.8 / 6.2 (1.3) | 3.5 / 3.6 (0.2) | 11.8 / 12.1 (0.6) | 2.03x |
| selected_area_mean_weights | typical | 61.7 / 63.1 (3.5) | 27.7 / 29.2 (2.3) | 166.5 / 168.1 (3.6) | 2.70x |
| selected_area_mean_weights_resident | small | — | 1.1 / 1.1 (0.0) | — | — |
| selected_area_mean_weights_resident | mid | — | 1.7 / 1.8 (0.3) | — | — |
| selected_area_mean_weights_resident | typical | — | 3.6 / 3.8 (0.3) | — | — |
| thin_R_2 | small | 0.1 / 0.1 (0.1) | — | 0.6 / 0.6 (0.4) | 7.23x |
| thin_R_2 | mid | 0.3 / 0.3 (0.0) | — | 0.8 / 0.8 (0.1) | 3.02x |
| thin_R_2 | typical | 2.4 / 2.7 (0.4) | — | 4.5 / 5.0 (4.4) | 1.91x |
| virtual_image_circle | small | 0.2 / 0.2 (0.0) | 0.9 / 0.9 (0.4) | 1.9 / 2.0 (0.2) | 10.09x |
| virtual_image_circle | mid | 0.7 / 0.7 (0.0) | 2.2 / 2.3 (0.6) | 7.0 / 7.1 (0.2) | 9.88x |
| virtual_image_circle | typical | 10.3 / 10.8 (1.0) | 23.5 / 24.1 (1.9) | 31.3 / 32.7 (2.8) | 3.04x |

## Backend decision per function (D16)

The break-even is read off the three sizes above; between two measured sizes it is interpolated, and says so. The Mac was
loaded (load average 10 to 11) during the final run, so a crossing near a tie moves with load.

| function | decision | why |
|---|---|---|
| `VirtualDiffraction.max` | CPU always | block-parallel since 0.8.0: 5.9 ms at the typical size against NumPy's 2.8 and 29 ms with the host copy; a `vmax` pass is memory-bound, so a copy to the GPU never pays |
| `VirtualDiffraction.mean`, `meanAndMax` | CPU; **resident Metal always** | the compensated Double sum is compute-bound. Host-copy Metal ties the CPU at 4.2 M values (3.4 against 3.4 ms) and loses at 41.9 M on this loaded Mac (27.5 against 22.0), so it does not pay; the resident cube beats the CPU at every size (0.5 against 0.8 ms at 1.0 M, 3.4 against 22.0 at 41.9 M) and comes within 1.2x of NumPy (3.4 against 2.8 ms) |
| `VirtualDiffraction.pattern` with a mask | Metal above about 4 M values; resident Metal always | CPU 47.6 ms, host-copy Metal 29.0, resident Metal 4.6 at the typical size; level at 4.2 M (4.5 against 3.7) |
| `VirtualDiffraction.pattern` with weights | Metal above about 1 M values; resident Metal always | the CPU path pays Double products (62 ms at the typical size); host-copy Metal is 1.7x faster at 4.2 M (3.5 against 5.8) and 2.2x at 41.9 M; resident Metal 4.3 ms. The CPU path is already 2.7x faster than py4DSTEM |
| `VirtualDiffraction.median` | CPU only | no Metal path; 3.4x faster than py4DSTEM at the typical size. A GPU sort per pixel needs the whole scan axis resident; revisit with the resident cube |
| `VirtualImaging.image` | CPU only for a non-resident cube | the CPU dot product is 9.5 ms at the typical size against 21 ms with the host copy; resident Metal is not measured (open item) |
| `CenterOfMass.measure` | CPU only | CPU 11.0 ms against Metal 26.7 at the typical size; the weighted-moment kernel is one thread per position |
| `CubeEditing.binScan`, `thinScan`, `padDetector`, `resampleDetector`, `medianFilterMaskedPixels`, `ViewedCube` bin | CPU only | memory-bound copies and small per-pattern arithmetic; the copy to the GPU would cost more than the work. Faster than py4DSTEM except `binScan` and the masked-pixel median at 64 x 64 (open item 3) |
| `Calibration`, `CalibrationMeasurement`, `BraggMasking` | CPU only | scalars and one pattern at a time |
