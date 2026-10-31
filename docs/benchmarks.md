# Benchmarks

Kit against py4DSTEM, same inputs, Float32 cubes, one run of `tools/benchmarks` on 2026-10-21. Harness: `Tests/CuboVirtualTests/BenchmarkHarness.swift`
(Swift, built with `swift test -c release`, skipped unless `CUBOKIT_BENCH_DIR` is set) and `tools/benchmarks/bench.py` (the py4DSTEM side
and the merge). Both read the same `.npy` cubes. Best of five after one warm-up; cells are
best / median (spread = slowest - best) in milliseconds; "CPU vs py4DSTEM" is py4DSTEM's best divided by the kit CPU's best
(above 1 the kit is faster). The cubes are held in memory (`MemoryCube`; `DataCube(data=ndarray)`). The Metal columns copy each
tile to the GPU on every call; the `_resident` rows hold the cube in a `ResidentCube` (one Metal buffer) first.

Sizes (scan rows x columns x detector rows x columns): small 32x32x32x32 (1.0 M values), mid 64x64x32x32 (4.2 M), typical
100x100x64x64 (41.9 M, 168 MB). The large size (330x330x64x64, 1.8 GB) and the Bragg, strain, DPC, parallax and ptychography
functions are not measured yet.
The machine was shared during the run (load average 5 to 11 from other builds), so the spreads are wider than on a quiet Mac.

Re-run from a fresh clone of the kit, on a quiet Mac, one benchmark at a time (the typical cube is 168 MB on disk and in memory):

```sh
python3.12 -m venv .venv-bench && .venv-bench/bin/pip install py4DSTEM==0.14.19
DIR=$(mktemp -d)
.venv-bench/bin/python -I -B tools/benchmarks/bench.py make "$DIR"      # writes the small, mid and typical .npy cubes
.venv-bench/bin/python -I -B tools/benchmarks/bench.py python "$DIR"    # py4DSTEM side: python.json
CUBOKIT_BENCH_DIR="$DIR" swift test -c release --filter BenchmarkHarness   # kit side: swift.json
.venv-bench/bin/python -I -B tools/benchmarks/bench.py merge "$DIR"     # prints the table below
```

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

## Speed round, 2026-10-22: the CPU mean, selected-area means and bin_R

The first table above is the 2026-10-21 run. Five of its rows (`dp_mean`, `selected_area_mean_mask`, `selected_area_mean_weights`, `bin_R_2`,
`median_masked_pixels`) were then the slowest against py4DSTEM; this round replaced their scalar loops and measured again. What changed:
the mean and the selected-area mean add with a branch-free SIMD TwoSum (four pixels per step, no per-element finite check; the check
moved to a post-pass on the result, `CompensatedSum`), the pixels split into 512-pixel blocks across every core (the old split made
two blocks at 4096 pixels), a `MemoryCube` is read in place instead of through a 64 MiB tile copy; `binScan` sums four pixels per step and
splits the output positions across cores; `medianFilterMaskedPixels` filters the patterns across cores with a fixed small buffer per
window instead of a sorted array allocated per masked pixel and pattern.

Method: three alternating rounds of old sources, new sources and Python, each 15 calls after one warm-up, `Tests/CuboVirtualTests/SpeedBench.swift`
(Swift, both builds, `CUBOKIT_SPEED_DIR="$DIR" swift test -c release --filter SpeedBench`) and `tools/benchmarks/bench.py speed "$DIR"` (py4DSTEM/NumPy 2.5.4); cells are the median of the call medians in
milliseconds, the three rounds' medians in brackets. The Mac was shared (load average 9 to 13 throughout), so every number is a
loaded-machine number: the old column is also on the loaded machine, and the ratios are the finding. Float32 cubes of the benchmark
above (`bench.py make`), kit CPU path, `MemoryCube`.

| function | size | before | after | py4DSTEM / NumPy | before vs Python | after vs Python |
|---|---|---|---|---|---|---|
| dp_mean | typical | 19.2 (22.0, 18.9, 19.2) | 4.9 (5.0, 4.8, 4.9) | 2.7 (2.6, 3.0, 2.7) | 0.14x | 0.56x |
| selected_area_mean_mask | typical | 43.1 (48.5, 43.1, 42.0) | 3.9 (3.8, 4.0, 3.9) | 10.0 (10.0, 10.0, 12.4) | 0.23x | 2.6x |
| selected_area_mean_weights | typical | 54.9 (60.9, 54.1, 54.9) | 5.1 (5.2, 5.1, 4.6) | 157.2 (157.2, 160.4, 156.9) | 2.9x | 31x |
| bin_R_2 | typical | 28.5 (31.5, 28.2, 28.5) | 2.4 (2.4, 2.5, 2.2) | 7.0 (11.2, 7.0, 6.1) | 0.25x | 2.9x |
| median_masked_pixels | 64 x 64 scan, 32 x 32 detector | 1.74 (1.87, 1.73, 1.74) | 0.29 (0.26, 0.29, 0.52) | 0.64 (0.58, 0.81, 0.64) | 0.37x | 2.2x |
| bin_R_2 | 64 x 64 scan, 32 x 32 detector | 2.89 (3.06, 2.89, 2.89) | 0.63 (0.65, 0.63, 0.63) | 0.72 (0.72, 0.76, 0.70) | 0.25x | 1.1x |

Ratios are Python's time over the kit's (above 1 the kit is faster). `dp_mean` stays 1.8x behind NumPy at the typical size (best of 15:
4.3 against 2.6 ms): the pass is bandwidth-bound and NumPy's reduction reads the cube once with no widening; the resident Metal path
(3.4 ms, table above) is the way past it. Block sizes of 128 to 1024 pixels and a register-resident 16-pixel accumulator were tried
and were no faster (the register variant was 1.5x slower: the strided reads between positions lose to row-contiguous ones).

Numbers did not move. The hash of every result's bits is the same before and after for all six cases (`SpeedBench` prints it), so the
results are bit identical to the earlier scalar loops, which are the CPU Float64/Double reference. Against NumPy on the same
cube, mask and weights (`bench.py compare`; the largest difference relative to the largest value):

| result | kit vs float64 truth | py4DSTEM/NumPy vs float64 truth | kit vs py4DSTEM |
|---|---|---|---|
| dp_mean | 3.0e-18 | 3.9e-06 (Float32 accumulation) | 3.9e-06 |
| selected_area_mean_mask | 0 | 3.1e-06 (Float32 accumulation) | 3.1e-06 |
| selected_area_mean_weights | 6.0e-15 | 6.0e-15 (NumPy accumulates in Float64 here) | 1.9e-16 |
| bin_R_2 | 3.6e-08 | 9.1e-08 | 7.3e-08 (both inside one Float32 rounding, 6e-08 relative to each value) |
| median_masked_pixels | bit for bit equal to py4DSTEM | | |

So the kit's means differ from NumPy by NumPy's own Float32 accumulation error, as before, and are closer to the exact value than
NumPy is.

## Speed round, 2026-10-31: the CPU mean and max in registers

The mean turned out compute-bound, not bandwidth-bound: every scan position stored each pixel's TwoSum sum and correction and
loaded them again for the next position. Since CuboKit#53, `CompensatedSum.addRows` takes four positions per pass with the sum and
correction in SIMD registers (each pixel still takes the positions in scan order with the same TwoSum), and the pixel blocks shrank
from 512 to 128 so every core takes part. CuboKit#55 did the same for the masked and weighted mean (`CompensatedSum.addPicked`).
Results are bit for bit unchanged, NaN and ±∞ included.

Method: `Tests/CuboVirtualTests/MeanMaxBench.swift` and `SelectedMeanBench.swift` (skipped unless their environment variable is set),
release build, resident 100 x 100 x 64 x 64 Float32 cube, median of 15, milliseconds.

| function | before | after | speed-up |
|---|---|---|---|
| dp_mean | 4.47 | 2.17 | 2.1x (NumPy 2.7: the kit is now ahead) |
| dp_max | 2.24 | 1.67 | 1.3x |
| meanAndMax | 5.73 | 2.45 | 2.3x |
| selected_area_mean_mask, selected_area_mean_weights | | | about 1.85x (a scattered half mask stays near 1.8 ms, bound by memory access) |

The 2026-10-22 note above that a register-resident accumulator was slower tried 16 pixels per position with strided reads; the
kept variant holds the pixels in registers over four positions and still reads each row contiguously.

## Backend decision per function

The break-even is read off the three sizes above; between two measured sizes it is interpolated, and says so. The Mac was
loaded (load average 10 to 11) during the final run, so a crossing near a tie moves with load.

| function | decision | why |
|---|---|---|
| `VirtualDiffraction.max` | CPU always | block-parallel since 0.8.0: 5.9 ms at the typical size on the loaded 2026-10-21 run against NumPy's 2.8 and 29 ms with the host copy; 1.67 ms (was 2.24) on the resident `MeanMaxBench` since CuboKit#53; a `vmax` pass is memory-bound, so a copy to the GPU never pays |
| `VirtualDiffraction.mean`, `meanAndMax` | CPU; **resident Metal always** | since 0.9.0 the compensated Double sum is a SIMD TwoSum across cores; since CuboKit#53 it is held in registers over four positions: 2.17 ms at the typical size (was 4.47, 22.0 before 0.9.0), ahead of NumPy's 2.7; host-copy Metal (27.5) does not pay; the resident cube (3.4 ms at 41.9 M values, measured before 0.9.0) has not been measured against the new CPU pass |
| `VirtualDiffraction.pattern` with a mask | CPU (0.9.0); resident Metal for repeated reductions | CPU 3.9 ms at the typical size since 0.9.0 (was 47.6), 2.6x faster than py4DSTEM; host-copy Metal (29.0 before) now loses to it; resident Metal 4.6 ms is level |
| `VirtualDiffraction.pattern` with weights | CPU (0.9.0); resident Metal for repeated reductions | CPU 5.1 ms at the typical size since 0.9.0 (was 61.7), 31x faster than py4DSTEM (157 ms); host-copy Metal (27.7) loses to it; resident Metal 3.6 ms |
| `VirtualDiffraction.median` | CPU only | no Metal path; 3.4x faster than py4DSTEM at the typical size. A GPU sort per pixel needs the whole scan axis resident; revisit with the resident cube |
| `VirtualImaging.image` | CPU only for a non-resident cube | the CPU dot product is 9.5 ms at the typical size against 21 ms with the host copy; resident Metal is not measured (open item) |
| `CenterOfMass.measure` | CPU only | CPU 11.0 ms against Metal 26.7 at the typical size; the weighted-moment kernel is one thread per position |
| `CubeEditing.binScan`, `thinScan`, `padDetector`, `resampleDetector`, `medianFilterMaskedPixels`, `ViewedCube` bin | CPU only | memory-bound copies and small per-pattern arithmetic; the copy to the GPU would cost more than the work. Faster than py4DSTEM since 0.9.0 (`binScan` 2.4 against 7.0 ms, the masked-pixel median 0.29 against 0.64 ms at 64 x 64; see the speed round) |
| `Calibration`, `CalibrationMeasurement`, `BraggMasking` | CPU only | scalars and one pattern at a time |
