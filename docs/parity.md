# Parity with py4DSTEM

A checklist against py4DSTEM's documented API in the areas CuboKit holds or plans, at the pinned commit `f050d207`
(0.14.19), so "complete" is a measured claim. Updated at every tag. The areas are the
`DataCube` (`datacube/`), the calibration of origins and probes (`process/calibration/`, `data/calibration.py`), the
`Probe` class (`braggvectors/probe.py`), DPC and the phase base class's preprocessing (`process/phase/`), and the headline
lists for strain, Bragg vectors and disk detection, parallax and ptychography. Crystallography (`Crystal`, orientation,
ACOM) is DiffractoKit's; file formats (`read`, `import_file`, `save`) are FormatoKit's; simulation is SimulatoKit's;
plotting and `show*` are not ported.

Where py4DSTEM takes a Python-only argument (`verbose`, `progress_bar`, `returncalc`, `name`, `plot`, `device="gpu"` for
CuPy, `dask`), the row says n/a once per method, not per argument.

## Totals (0.9.0)

**135 present, 38 partial, 42 missing** of 215 counted rows (study 53, useful 162); 24 more rows are not planned, with the reason in the row. The count table is near the end.

States: **present** (pinned against py4DSTEM or planted truth; numerics held to the tolerance in the last column, never stricter; readers, I/O, indices and metadata exact), **partial** (what is missing says so), **missing**, **not planned** (the reason is in the Mark column). Marks: **study** (on the path of the TEM study and the two apps' open kit asks), **useful**, **not planned**. The tolerance column names the py4DSTEM test that states it; py4DSTEM's `test/` asserts almost no values, so most rows use `np.allclose`/`assert_allclose` defaults for the dtype (suite rule D17). The legend of the earlier versions ("done", "absent", "app", "n/a") maps to present, missing and not planned.

## `DataCube`: container and shape (datacube/datacube.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `DataCube(data)`, `.data` | `MemoryCube(shape:values:)`, `MappedCube(npy:)` (lazy, memory-mapped), `CubeSource` | present (Float32 on read; ten element types) | useful | exact (shapes, indices, values read) |
| `R_Nx`, `R_Ny`, `Q_Nx`, `Q_Ny` | `CubeShape.scanRows`, `scanColumns`, `detectorRows`, `detectorColumns` | present | useful | exact (shapes, indices, values read) |
| `Rshape`, `Qshape`, `R_N` | `CubeShape`, `scanCount`, `detectorCount` | present | useful | exact (shapes, indices, values read) |
| `datacube[rx, ry]` | `readPattern(row:column:)` | present | useful | exact (shapes, indices, values read) |
| `bin_Q_mmap`, dask-backed lazy data | `MappedCube` | partial: lazy and memory-mapped; crop and bin on read through `ViewedCube` (0.5.0) | study | exact (shapes, indices, values read) |
| `DataCube(calibration=..., slicelabels=...)` | — | missing (planned: a calibration model, below) | study | exact (shapes, indices, values read) |
| `copy`, `add`, the emdfile tree and `metadata` | — | not planned (value types; emdfile is FormatoKit's) | not planned: value types; emdfile is FormatoKit's | — |
| `.polar` (polar-transformed datacube) | `PolarCube` (`PolarSampling`, `transform`) | present (0.9.0; CPU Double; py4DSTEM's `rr` is on the scale of the minor axis, kept; theta from +row towards +column as stored in `Calibration.ellipse`) | study | no upstream test; np.allclose defaults; measured 5e-16 relative |

## `DataCube`: virtual images (datacube/virtualimage.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `get_virtual_image(mode, geometry)`: `point`, `circle`, `annulus`, `rectangle` | `VirtualImaging.image(of:geometry:)` | present (CPU and Metal) | useful | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `get_virtual_image(mode="mask", geometry=array)` | `VirtualImaging.image(of:geometry:)` with a mask geometry | present (CPU and Metal) | useful | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `get_virtual_image(return_mask=True)`, `make_detector` | `DetectorMask.make` | present | useful | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `get_virtual_image(centered=True)` | `VirtualImaging.image(of:geometry:centeredOn:)` | present | useful | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `get_virtual_image(calibrated=True)` | `VirtualImaging.image(of:geometry:pixelSize:)` | present (centred and calibrated together refused) | study | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `get_virtual_image(shift_center=True, subpixel=False)` | `shift: .rounded` | present (CPU) | useful | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `get_virtual_image(shift_center=True, subpixel=True)` | `shift: .bilinear` | present (CPU) | useful | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `get_calibrated_detector_geometry` | applied inside `VirtualImaging.image` | partial: not exposed as a function | study | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `make_bragg_mask` | `BraggMasking.latticeMask(...)` | present (0.8.0; exact counts; `return_sum=False` is the loop over the lattice points) | study | no upstream test; np.allclose defaults on the Float64 reference; Metal against CPU 1e-5 |
| `get_virtual_image(dask=True)`, `test_config` | — | not planned (Python-only; the lazy cube tiles instead) | not planned: Python-only; the lazy cube tiles instead | — |
| `position_detector`, `show` | — | not planned (plotting) | not planned: plotting | — |

## `DataCube`: virtual diffraction (datacube/virtualdiffraction.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `get_dp_mean` | `VirtualDiffraction.mean` | present | useful | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_dp_max` | `VirtualDiffraction.max` | present | useful | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_dp_median` | `VirtualDiffraction.median(of:)` | present (CPU) | useful | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_virtual_diffraction(method="mean")`, `"max"` | `VirtualDiffraction.mean`, `max`, `meanAndMax` | present | useful | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_virtual_diffraction(method="median")` | `VirtualDiffraction.pattern(of:statistic: .median, scan:)` | present (CPU) | useful | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_virtual_diffraction(mask=)`, boolean or float real-space mask | `VirtualDiffraction.pattern(of:statistic:scan: .mask / .weights, path:)` | present (CPU compensated Double; Metal `cubo_weighted_pattern` for mean and max, 0.8.0, held to the CPU at 1e-5; `ResidentCube` reads in place) | study | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_virtual_diffraction(mask=)`, complex mask | — | missing (not ported) | not planned: absent (not ported) | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_virtual_diffraction(shift_center=True, subpixel=False)` (mean, max) | `VirtualDiffraction.meanAndMax(of:shift: .rounded)` | present (CPU) | useful | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_virtual_diffraction(shift_center=True, subpixel=True)` (mean, max) | `shift: .fourier` | present (CPU) | useful | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |
| `get_virtual_diffraction(shift_center=True, mask=)` (mean, max) | `pattern(of:statistic:scan:shift:)` | present (CPU) | useful | no upstream test; np.allclose defaults; sums Float64-compensated, NumPy Float32 gap < 1e-6 relative; Metal against CPU 1e-5 |

## `DataCube`: calibration-bound properties and coordinates (datacube/datacube.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `calibrate()` (rebuild the coordinate vectors from the calibration) | `Calibration` (axes and grids are derived from the value) | present (0.8.0; a value type: no target registration) | study | values equal to the Double arithmetic of `DataCube.calibrate`, `assert_allclose` Float64 default (rtol 1e-7, atol 1e-12); no upstream test |
| `rxx`, `ryy`, `qxx`, `qyy` (calibrated meshgrids) | `Calibration.scanRowGrid/scanColumnGrid/detectorRowGrid/detectorColumnGrid`, `*Axis` | present (0.8.0; rows x columns, the transpose of py4DSTEM's (Ny, Nx) array, values equal) | study | values equal to the Double arithmetic of `DataCube.calibrate`, `assert_allclose` Float64 default (rtol 1e-7, atol 1e-12); no upstream test |
| `rxx_raw`, `ryy_raw`, `qxx_raw`, `qyy_raw` (pixel meshgrids) | `Calibration.rawScanRowGrid` ... `rawDetectorColumnGrid` | present (0.8.0) | study | values equal to the Double arithmetic of `DataCube.calibrate`, `assert_allclose` Float64 default (rtol 1e-7, atol 1e-12); no upstream test |
| `qxxs(rx, ry)`, `qyys(rx, ry)` (per-position shifted Q axes) | `Calibration.shiftedDetectorRowGrid/ColumnGrid(scanRow:scanColumn:)` | present (0.8.0; throws `CalibrationError.notSet` without an origin, as py4DSTEM raises) | study | values equal to the Double arithmetic of `DataCube.calibrate`, `assert_allclose` Float64 default (rtol 1e-7, atol 1e-12); no upstream test |
| `Q_pixel_size`, `Q_pixel_units`, `R_pixel_size`, `R_pixel_units` | `Calibration.detectorPixelSize/detectorUnits/scanPixelSize/scanUnits` | present (0.8.0) | study | values equal to the Double arithmetic of `DataCube.calibrate`, `assert_allclose` Float64 default (rtol 1e-7, atol 1e-12); no upstream test |

## `DataCube`: reshaping and pre-processing (datacube/datacube.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `set_scan_shape` | `CubeEditing.setScanShape` | present (0.8.0; throws where py4DSTEM prints and returns the cube) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `swap_RQ` | `CubeEditing.swapScanAndDetector` | present (0.8.0; exact) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `swap_Rxy` | `CubeEditing.swapScanAxes` | present (0.8.0; exact) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `swap_Qxy` | `CubeEditing.swapDetectorAxes` | present (0.8.0; exact) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `crop_Q` | `ViewedCube` with `CubeViewSpecification.detectorCrop` | present (0.5.0; a read-time view of any `CubeSource`; shapes exact, `tools/cubeview/pins.py`) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `crop_R` | `ViewedCube` with `.scanCrop` | present (0.5.0; exact) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `bin_Q` | `ViewedCube` with `.detectorBin` | present (0.5.0; py4DSTEM's rule: crop first, trim the remainder off the end, sum `bin x bin` blocks; Double sums rounded once to Float32, measured 1.2e-7 relative; any positive factor, the app's 2, 4, 8 is its own policy) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `bin_R` | `CubeEditing.binScan` | present (0.8.0; 1e-6 relative; R pixel size updated) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `thin_R` | `CubeEditing.thinScan` | present (0.8.0; exact) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `pad_Q` | `CubeEditing.padDetector` | present (0.8.0; exact; factor and output size) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `resample_Q` (bilinear, `conserve_array_sums`, `method="fourier"`) | `CubeEditing.resampleDetector`, `CubeEditing.resampleDetectorFourier` | present (0.9.0; method="fourier": scalar scale or output size, the Q pixel size per `resample_data_diffraction`; the kit returns Float32 where py4DSTEM returns float64 unless the sums are conserved) | study | bilinear unchanged; fourier 1e-6 of the largest value (py4DSTEM is complex64); measured 2.1e-7 |
| `filter_hot_pixels` (`thresh`, `ind_compare`, `return_mask`) | `HotPixelFilter.filter`, `.mask`, `.replacing` | present: eager Float32 cube, means/masks/pixels pinned exact today; wrapped classification and clipped sequential replacement | useful | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |
| `median_filter_masked_pixels` | `CubeEditing.medianFilterMaskedPixels` | present (0.8.0; exact for odd kernels, in place and row-major as upstream) | study | index moves exact; sums 1e-6 relative of max (NumPy Float32); resample 1e-6 (scipy Float32); no upstream test |

## `DataCube`: probe, beam stop and background methods (datacube/datacube.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `get_probe_size` (`thresh_lower`, `thresh_upper`, `N`) | `ProbeSize.estimate` | present (exact today) | useful | no upstream test; np.allclose defaults |
| `get_vacuum_probe(ROI=boolean array)` | `VacuumProbe.measure(of:scanMask:)` | present | useful | no upstream test; np.allclose defaults |
| `get_vacuum_probe(ROI=4-tuple)` | — | missing (not ported: pass the equivalent boolean ROI) | not planned: absent (not ported: pass the equivalent boolean ROI) | no upstream test; np.allclose defaults |
| `get_vacuum_probe(align, zero_vacuum, mask)` | `VacuumProbe.measure(of:scanMask:align:zeroVacuum:detectorMask:)` | present | useful | no upstream test; np.allclose defaults |
| `get_beamstop_mask` (`threshold`, `distance_edge`, `include_edges`, `sigma`, `scale_radial`) | `BeamstopMask.make(from:threshold:distanceEdge:includeEdges:sigma:scaleRadial:)` | present | useful | no upstream test; np.allclose defaults |
| `get_beamstop_mask(use_max_dp=True)` | the caller passes the max pattern | present (via the caller's pattern) | useful | no upstream test; np.allclose defaults |
| `get_radial_bkgrnd` | `RadialBackground.background` | present (0.9.0; the lookup radius is on the major-axis scale while the polar bins are on the minor-axis scale: py4DSTEM's own, so a ring is stretched by a/b for a != b; a circle is exact) | study | no upstream test; np.allclose defaults; measured 4e-15 |
| `get_radial_bksb_dp` | `RadialBackground.subtracted` | present (0.9.0) | study | no upstream test; np.allclose defaults; measured 2e-13 |
| `get_local_ave_dp` (`radial_bksb`, `sigma`, `braggmask`) | `BraggMasking.localAverage`, `RadialBackground.localAverage` | present (0.9.0): `RadialBackground.localAverage` is `radial_bksb=True`, and with `keeping:` (a per-position mask, e.g. from `BraggMasking.mask`) the `radial_bksb` + `braggmask` branch (hand-composed, not pinned); plain and Bragg-masked without radial background as before | study | no upstream test; np.allclose defaults; measured 4e-13 |
| `get_braggmask` | `BraggMasking.mask(vectors:scanRow:scanColumn:detectorRows:detectorColumns:radius:)` | present (0.8.0; exact) | study | no upstream test; np.allclose defaults |

## Calibration entries (data/calibration.py)

CuboKit has no stateful `Calibration` object yet; its measurements return plain values (`OriginMap`, `OriginFit`,
`RotationSolution`, `ProbeEstimate`) that the caller stores.

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `Calibration` container: `attach`, `root`, `register_target`, `targets`, `calibrate`, `copy` | `Calibration` (value type) | present (0.8.0; `attach`, `root`, `register_target`, `copy` have no meaning for a value: the caller owns the value) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `Calibration.to_h5`, `from_h5` | — | not planned (FormatoKit) | not planned: FormatoKit | — |
| `Q_pixel_size` (`set_`, `get_`, `qpixsize`) | `Calibration.detectorPixelSize` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `Q_pixel_units` (`set_`, `get_`, `qpixunits`) | `Calibration.detectorUnits` | present (0.8.0; `DetectorUnits`: pixels, A^-1, mrad) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `R_pixel_size` (`set_`, `get_`, `rpixsize`) | `Calibration.scanPixelSize` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `R_pixel_units` (`set_`, `get_`, `rpixunits`) | `Calibration.scanUnits` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `qx0`, `qy0` (per-position origin, `set_`, `get_`) | `Calibration.setOrigin`, `origin(row:column:)`, `ScanField` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `qx0_mean`, `qy0_mean`, `get_origin_mean` | `Calibration.originMean` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `qx0shift`, `qy0_shift`, `get_origin_shift`, `origin_shift` | `Calibration.originShift(row:column:)`, `ScanField.shift` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `origin`, `set_origin`, `get_origin(rx, ry)` | `Calibration.setOrigin`, `origin(row:column:)` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `qx0_meas`, `qy0_meas`, `origin_meas`, `set_origin_meas`, `get_origin_meas` | `Calibration.measuredOriginRow/Column` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `origin_meas_mask`, `set_origin_meas_mask`, `get_origin_meas_mask` | `Calibration.measuredOriginMask` | present (0.8.0) | useful | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `a`, `b`, `theta`, `set_a`/`get_a`, `set_b`/`get_b`, `set_theta`/`get_theta` | `Calibration.ellipse` (`EllipseCalibration`) | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `ellipse`, `set_ellipse`, `get_ellipse` (a, b, theta) | `Calibration.ellipse` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `p_ellipse`, `set_p_ellipse`, `get_p_ellipse` | `Calibration.ellipse` with `originMean`/`origin(row:column:)` | present (0.8.0; like upstream the setter does not move the origin) | useful | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `QR_rotation_degrees`, `set_QR_rotation`, `get_QR_rotation`, `..._degrees` | `Calibration.qrRotationRadians/qrRotationDegrees`, `setQRRotation` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `QR_flip`, `set_QR_flip`, `get_QR_flip` | `Calibration.qrFlip` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `QR_rotflip`, `set_QR_rotflip`, `get_QR_rotflip`, `..._degrees` | `Calibration.setQRRotationAndFlip(degrees:flip:)` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `probe_semiangle`, `probe_param`, `set_probe_param`, `get_probe_param` | `Calibration.probeSemiangle`, `setProbeParameters` | present (0.8.0; the set writes the MEAN origin only, as upstream) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `probe_center`, `set_probe_center`, `get_probe_center` | `Calibration.probeCenter` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `probe_convergence_semiangle_pixels`, `set_`, `get_` | `Calibration.convergenceSemiangleInPixels` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |
| `probe_convergence_semiangle_mrad`, `set_`, `get_` | `Calibration.convergenceSemiangleInMilliradians` | present (0.8.0) | study | exact (stored values; `test/test_calibration.py` asserts `get_Q_pixel_size() == 10` after a bin) |

## Calibration processes (process/calibration/)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `get_origin_single_dp` | the per-pattern step of `OriginMeasurement.measure` | partial: no single-pattern entry point | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_origin(r, rscale)` | `OriginMeasurement.measure(radius:radiusScale:)` | present (CPU) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_origin(dp_max=)` | the max pattern is computed internally | partial: no argument to pass it | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_origin(mask=)` | `OriginMeasurement.measure(of:scanMask:)` | present (NaN outside the mask) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_origin(fast_center=True)` | `OriginMeasurement.measure(of:scanMask:fastCenter:)` | present | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_origin(remove_NaN=True)` | `OriginMap.replacingNaNWithMean()` | present | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_origin_friedel(mask, upsample_factor)` | `FriedelOrigin.measure(of:mask:upsampleFactor:)`, `.origin(of:)` | present (CPU only) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_origin_friedel(device="gpu")` | — | missing (planned: a Metal path, later) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `fit_origin(fitfunction="constant")` | `OriginFit.fit(model: .constant)` | present | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `fit_origin(fitfunction="plane")` | `OriginFit.fit(model: .plane)` | present | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `fit_origin(fitfunction="parabola")` | `OriginFit.fit(model: .parabola)` | present | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `fit_origin(fitfunction="bezier_two")` | `OriginFit.fit(model: .bezierTwo)` | present | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `fit_origin(robust, robust_steps, robust_thresh)` | `OriginFit.fit(robust:robustSteps:robustThreshold:)` | present | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `fit_origin(mask=)` | `OriginFit.fit` | partial: NaN entries are left out of the fit; no boolean mask argument | useful | no upstream test; np.allclose defaults on the Float64 reference |
| mac4DSTEM trimmed fit (`fitOriginTrimmed`, MAD cut, up to three rounds) | `OriginFit.fitTrimmed` | named option beside `fit(robust:)`; mask and surfaces pinned to a Float32 transcription of the app source, the final fit also to `fit_origin(mask=kept)` (not counted twice) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `fit_origin(returnfitp=True)` | — | missing (the fit coefficients are not returned) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_probe_size` (`process/calibration/probe.py`) | `ProbeSize.estimate` | present (exact today) | study | no upstream test; np.allclose defaults on the Float64 reference |
| `fit_ellipse_1D` | `EllipseFit.fit1D` | present (0.9.0; default `fitRadii` repaired, upstream raises `TypeError`; `ignoring:` is upstream's `mask` with the same polarity) | useful | no upstream test; np.allclose defaults on the Float64 reference; the fit is compared at 1e-6 (the optimiser's own result moves by 1e-7 to 7e-4 for one ulp of start): measured 7e-10 relative |
| `fit_ellipse_amorphous_ring` | `EllipseFit.fitAmorphousRing` | present (0.9.0; `keeping:` is upstream's `mask`, `true` = use, the opposite of `fit_ellipse_1D`; `initial:` is `p0`) | useful | as `fit_ellipse_1D`; measured 3e-8 relative, cost 4e-9 |
| `ellipse_err`, `double_sided_gaussian`, `double_sided_gaussian_fiterr` | `EllipseFit.ellipseError`, `.doubleSidedGaussian`, `.doubleSidedGaussianFitError` | present (0.9.0) | study | np.allclose defaults on the Float64 reference; measured below 1e-7 |
| `constrain_degenerate_ellipse` | `EllipseFit.constrainDegenerate` | present (0.9.0; Double input) | useful | np.allclose defaults; measured exact |
| `get_Q_pixel_size(q_meas, q_known, units)` | `CalibrationMeasurement.detectorPixelSize(measuredPixels:knownDistance:units:)` | present (0.8.0; exact) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_dq_from_indexed_peaks` | `CalibrationMeasurement.detectorPixelSize(peakPositions:indices:cellSize:)` | present (0.8.0; closed-form least squares, 1e-7 relative against `leastsq`) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `get_Qvector_from_Rvector`, `get_Rvector_from_Qvector` | `CalibrationMeasurement.detectorVector(fromScanVector:rotationDegrees:)`, `scanVector(fromDetectorVector:rotationDegrees:)` | present (0.8.0; 1e-14) | useful | no upstream test; np.allclose defaults on the Float64 reference |
| `compare_QR_rotation` | — | not planned (plotting) | not planned: plotting | — |

## `Probe` (braggvectors/probe.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `Probe(data)` with `probe`, `kernel`, `alpha`, `origin` | functions on a detector image | partial: no stateful `Probe` object | study | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.from_vacuum_data` (`mask`, `threshold`, `expansion`, `opening`) | — | missing (planned: the Bragg-disk module) | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.generate_synthetic_probe(radius, width, Qshape)` | `SyntheticProbe.make(radius:width:detectorRows:detectorColumns:centre:)` | present; `centre: nil` is py4DSTEM's (Q/2 in floats), a given centre (for example a pixel) is an option, tested exact against the closed form | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.measure_disk` (`thresh_lower`, `thresh_upper`, `N`, `zero_vacuum`, `alpha_max`) | `ProbeSize.estimate` (the same algorithm), then `VacuumProbe.zeroVacuum` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.measure_disk(plot=True)` | — | not planned (plotting) | not planned: plotting | — |
| `Probe.zero_vacuum` | `VacuumProbe.zeroVacuum(_:center:alpha:alphaMax:)` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.zero_vacuum_sinusoid` (`s`, `w`, `p`) | `VacuumProbe.dampVacuum(_:center:alpha:start:width:power:)` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.get_kernel(mode="flat")` | `ProbeKernel.flat` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.get_kernel(mode="gaussian", sigma)` | `ProbeKernel.gaussian` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.get_kernel(mode="sigmoid", radii)` | `ProbeKernel.sigmoid(profile: .sineSquared)` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.get_kernel(mode="sigmoid_log", radii)` | `ProbeKernel.sigmoid(profile: .sineSquared)` (py4DSTEM dispatches this mode to the sine² trench; `.logistic` is `type="logistic"`) | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.get_probe_kernel_flat(origin, bilinear)` | `ProbeKernel.flat(_:origin:bilinear:)` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.get_probe_kernel_edge_gaussian(sigma, origin, bilinear)` | `ProbeKernel.gaussian(_:sigma:origin:bilinear:)` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.get_probe_kernel_edge_sigmoid(type="sine_squared")` | `ProbeKernel.sigmoid(_:innerRadius:outerRadius:profile: .sineSquared)` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |
| `Probe.get_probe_kernel_edge_sigmoid(type="logistic")` | `ProbeKernel.sigmoid(..., profile: .logistic)` | present | useful | no upstream test (`test_probe.py` checks types and shapes); np.allclose defaults |

## DPC and the phase base class's preprocessing (process/phase/dpc.py, phase_base_class.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `DPC(datacube, energy, ...)` | `CenterOfMass`, `CenterOfMassReference`, `RotationSolver`, `DPC` | partial: functions, no stateful reconstruction object | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `_calculate_intensities_center_of_mass` | `CenterOfMass.measure` | present (CPU and Metal) | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `_extract_intensities_and_calibrations_from_datacube` (reciprocal sampling, mask) | `CenterOfMass.measure(mask:)`, `reciprocalSampling:` | partial: the calibration is passed in, not read from a cube | study | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(dp_mask)` | `CenterOfMass.measure(mask:)` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(fit_function="plane")`, `"constant"` | `CenterOfMassReference.planeFit`, `.constant` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(fit_function="parabola")`, `"bezier_two"` | `OriginFit` has them | partial: not wired into `CenterOfMassReference` | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(force_com_shifts)` | `CenterOfMassReference.given` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(force_com_measured)` | the measured `ScanVectorField` is an input | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(rotation_angles_deg)` | `RotationSolver.solve(anglesDegrees:)` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(maximize_divergence=False)`: curl | `RotationCriterion.minimizeCurl` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(maximize_divergence=True)`: divergence | `RotationCriterion.maximizeDivergence` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(force_com_rotation, force_com_transpose)` | `forceRotationDegrees:`, `forceTranspose:` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(padding_factor)` | `DPC.reconstruct(paddingFactor:)` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| the rotation search curves (`plot_rotation` data) | `RotationSolution.curve`, `curveTransposed` | present (the plot itself n/a) | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.preprocess(vectorized_com_calculation, plot_*, device, clear_fft_cache)` | — | not planned (Python memory strategy, plotting, CuPy) | not planned: Python memory strategy, plotting, CuPy | — |
| `DPC.reconstruct(max_iter, step_size, stopping_criterion, backtrack)` | `DPC.reconstruct(maxIterations:stepSize:stoppingCriterion:backtrack:)` | present (CPU, Double) | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.reconstruct(gaussian_filter_sigma, gaussian_filter)` | — | missing (planned: DPC filters, later) | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.reconstruct(q_lowpass, q_highpass, butterworth_order, butterworth_filter)` | — | missing (planned: DPC filters, later) | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.reconstruct(store_iterations=True)` | `DPCReconstruction.errors`, `accepted` | partial: the error history is kept; the per-iteration objects are not | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.reconstruct(device="gpu")` | — | missing (planned: a Metal DPC, later) | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.reconstruct(reset, progress_bar)` | — | not planned (Python state and progress bar) | not planned: Python state and progress bar | — |
| result: `object_phase`, `error`, `sampling`, rotation and transpose | `DPCReconstruction.phase`, `errors`, `RotationSolution` | present | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `DPC.visualize`, `show_complex_CoM`, `to_h5` | — | not planned (plotting; FormatoKit) | not planned: plotting; FormatoKit | — |
| complex CoM magnitude and angle (`show_complex_CoM` data) | `DPCViews.magnitude`, `.angle` | not counted (derived views): py4DSTEM's `row + i column` direction or the image-frame `atan2(row, column)`, wrapped to [0, 2 pi); NumPy-pinned (`tools/dpcviews/pins.py`), `RotationConvention` states the sign flip between conventions | useful | none upstream (`test/` holds no phase test); np.allclose defaults |
| `set_device`, `set_storage`, `clear_device_mem`, `attach_datacube`, `reinitialize_parameters`, `set_save_defaults` | — | not planned (Python object plumbing) | not planned: Python object plumbing | — |

## Strain (process/strain/)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `StrainMap(braggvectors)`, state properties, `reset_calstate` | `StrainMapping` | partial: the scan-level steps are stateless calls on centred peaks; no calibration-state object | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `choose_basis_vectors` | `StrainMapping.chooseBasisVectors`, `LatticeStrain.chooseBasis` | present: histogram, maxima (multicorr, parabolic, pixel), `index_origin/g1/g2`; pinned (`docs/strain.md`) | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `set_hkl` | — | missing | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `set_max_peak_spacing` | `index(...maximumSpacing:)` | partial: stateless argument, no visualization/state | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `fit_basis_vectors` (`mask`, `max_peak_spacing`) | `StrainMapping.compute` | present: scan mask, maximum spacing, per-position fits | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `get_strain` (`gvects`, `coordinate_rotation`) | `StrainMapping.compute` | present: reference region or median, rotation | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `get_reference_g1g2` | `LatticeStrain.reference` | present for caller-selected bases | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `latticevectors.index_bragg_directions` | `LatticeStrain.index(directions:basis:)` | present: minimum-norm SVD and ties-to-even | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `latticevectors.add_indices_to_braggvectors` | `LatticeStrain.index(peaks:directions:maximumSpacing:)` | partial: caller handles scan mask/coordinate shifts | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `latticevectors.fit_lattice_vectors` | `LatticeStrain.fit` | present | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `latticevectors.fit_lattice_vectors_all_DPs` | per-pattern `fit` | partial: caller maps scan positions | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `latticevectors.get_reference_g1g2`, `get_strain_from_reference_g1g2` | `reference`, `strain` | present | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `latticevectors.get_rotated_strain_map` | `strain(...coordinateRotation:)` | present | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `BraggVectors.to_strainmap` | `StrainMapping.centredPeaks(_:origin:)` | partial: converts a `CuboBragg` result; calibrations (ellipse, pixel size, R–Q rotation) are the caller's | useful | `test_strain.py` (runs the workflow, no value assert); np.allclose defaults |
| `show_strain`, `show_reference_directions`, `show_lattice_vectors`, `show_bragg_indexing` | — | not planned (plotting) | not planned: plotting | — |

## Bragg vectors and disk detection (braggvectors/)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `find_Bragg_disks(DataCube, template)` | `BraggDisks.detectAll(in:kernel:options:tiling:)` | present: per-position `BraggDisks.detect`, tiled, patterns on every core; pins on the per-pattern path (`docs/bragg.md`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `find_Bragg_disks` on a 2D array, a 3D stack, or `(DataCube, rx, ry)` | `BraggDisks.detect(_:kernel:options:)`, `detect(patterns:kernel:options:)` | partial: one pattern and a stack; no `(DataCube, rx, ry)` tuple (the caller reads the patterns) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `find_Bragg_disks(corrPower, sigma, sigma_dp, sigma_cc, filter_function)` | `BraggDiskOptions.correlationPower`, `correlationSigma`, `patternSigma` | partial: all but `filter_function` (the caller filters the pattern) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `find_Bragg_disks(radial_bksb)` | — | missing (planned: with `get_radial_bksb_dp`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `find_Bragg_disks(subpixel="none"/"poly"/"multicorr", upsample_factor)` | `BraggDiskOptions.subpixel` (`.pixel`, `.parabolic`, `.multicorr`), `upsampleFactor` | present: pinned, factors 3, 8, 16; a factor of 2 or less for multicorr is refused (py4DSTEM asserts) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `find_Bragg_disks(minAbsoluteIntensity, minRelativeIntensity, relativeToPeak, minPeakSpacing, edgeBoundary, maxNumPeaks)` | the same six options | present: pinned; `BraggMaxima.find` is `get_maxima_2D` | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `find_Bragg_disks(CUDA, CUDA_batched)` | — | missing (planned: a Metal path, later) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `find_Bragg_disks(distributed, ML, ml_*)`, the `diskdetection_parallel`, `_aiml`, `_cuda` variants | — | not planned (Python parallel tooling, TensorFlow) | not planned: Python parallel tooling, TensorFlow | — |
| `BraggVectors` (`raw`, `cal`, `setcal`, `calibrate`, `get_vectors`, `shape`, `qx`, `qy`, `I`) | `BraggVectors` (peaks per scan position) | partial: the raw value container only; no calibration state | study | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `BraggVectors.get_virtual_image` | `BraggVectors.virtualImage(detector:)` | present (0.9.0): `.allPeaks`, `.circle`, `.annulus` over vectors already in detector coordinates; the four calibration flags are the caller's | useful | `test_braggvectors.py` (counts and shapes); np.allclose defaults, measured 0 |
| `BraggVectors.histogram` | — | missing (planned: with `BraggVectors`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `BraggVectors.measure_origin` | — | missing (planned: with `BraggVectors`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `BraggVectors.measure_origin_beamstop` | — | missing (planned: with `BraggVectors`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `BraggVectors.fit_origin` | — | missing (planned: with `BraggVectors`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `BraggVectors.fit_p_ellipse` | — | missing (planned: with `BraggVectors`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `BraggVectors.mask_in_Q`, `get_masked_peaks`, `mask_in_R` | — | missing (planned: with `BraggVectors`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `BraggVectorMap` | — | missing (planned: with `BraggVectors`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `threshold_Braggpeaks`, `universal_threshold`, `get_pointlistarray_intensities` | — | missing (planned: with `BraggVectors`) | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |
| `BraggVectors.plot`, `copy`, `to_h5` | — | not planned (plotting; value types; FormatoKit) | not planned: plotting; value types; FormatoKit | — |
| `Crystal`, `orientation_plan`, `match_orientations` | — | DiffractoKit | useful | `test_braggvectors.py`, `test_workflow/test_disk_detection_*.py` (counts and shapes); positions np.allclose defaults |

## Parallax (process/phase/parallax.py)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `Parallax(datacube, energy, ...)` | `CuboPtycho` | missing (planned: `CuboPtycho`, placeholder) | useful | none upstream; np.allclose defaults |
| `Parallax.preprocess` (`edge_blend`, `dp_mask`, `threshold_intensity`, descan correction, `realspace_mask`) | `ParallaxPreprocessing.prepare` | partial: constant normalization, selection/mask, edge window and padding; no descan/polynomial normalization/real-space mask. Default windowed retention follows reference; `.unwindowed` follows app (independent source transcription pin). | useful | none upstream; np.allclose defaults |
| `Parallax.reconstruct` (alignment bins, upsampling, regularisation) | `CuboParallax.ParallaxReconstruction.reconstruct`, `.process` (end to end with KDE) | partial: bin schedule, running reference, Bernstein regularisation, centring, per-bin error, pinned (1e-7 to 9e-7); no initial aberration shifts, real-space mask, continue/reset, batching (`docs/parallax-alignment.md`) | useful | none upstream; np.allclose defaults |
| `Parallax.guess_common_aberrations` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `Parallax.subpixel_alignment` | `CuboParallax.ParallaxSubpixel` | partial: factor heuristic, bilinear/Lanczos KDE, Gaussian, low-pass; position correction (`correctPositions`, reference semantics incl. regularisation); no `virtual_detector_mask` or integer-pixel rolling | useful | none upstream; np.allclose defaults |
| `Parallax.aberration_fit` | `CuboParallax.ParallaxAberrationFitter` | present: low order (C1, C12a/b, rotation, transpose) and higher-order global/recursive fits; reference basis default, app basis (`.selectedCentroid`) as option; tolerances in `AberrationTests` | useful | none upstream; np.allclose defaults |
| `Parallax.aberration_correct` | `CuboParallax.ParallaxAberrationCorrection` | present: CTF sign/phase correction; reference ignores `q_highpass` (default), `.apply` is the app behaviour | useful | none upstream; np.allclose defaults |
| `Parallax.depth_section` | `CuboParallax.ParallaxDepthSectioning` | present: CPU Double; `.defocusOnly` pinned to py4DSTEM (`use_CTF_fit=False`); `.fullFit` pinned to the app transcription because the reference's `use_CTF_fit=True` crashes at 0.14.19; plotting not ported | not planned: done: CPU Double; `.defocusOnly` pinned to py4DSTEM (`use_CTF_fit=False`); `.fullFit` pinned to the app transcription because the reference's `use_CTF_fit=True` crashes at 0.14.19; plotting not ported | none upstream; np.allclose defaults |
| `Parallax.object_cropped` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `show_shifts`, `show_probe_position_shifts`, `visualize` | — | not planned (plotting) | not planned: plotting | — |

`CuboParallax` also exposes the alignment utilities used within the partial reconstruction row:

| py4DSTEM utility | CuboKit | State |
|---|---|---|
| `align_images_fourier`, `upsampled_correlation`, `dftUpsample` | `ParallaxAlignment.shift(reference:moving:rows:columns:upsampleFactor:)` | done for finite, nonconstant real images; CPU Double, factors 3...64; default factor 8; `1e-10` normwise relative tolerance with unit floor |
| `get_shifted_ar(periodic=True, bilinear=False)` | `ParallaxAlignment.shifted(_:rows:columns:by:)` | done for finite real images at least 3 × 3; CPU Double; `1e-10` normwise relative tolerance with unit floor |

## Ptychography (process/phase/)

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `SingleslicePtychography` (`preprocess`, `reconstruct`) | `CuboPtycho` | partial: CPU single-slice gradient descent from preprocessed amplitudes; scope below | useful | none upstream; np.allclose defaults |
| `MultislicePtychography` (`preprocess`, `reconstruct`) | `MultislicePtychography`, `PtychographyPreparation` | partial (0.5.0): CPU Double gradient descent through slices with Fresnel propagators, identical-slices option; tilted propagators, z-regularisation, TV and `middle_focus` absent; scope in the ptychography details below | useful | none upstream; np.allclose defaults |
| `MixedstatePtychography` (`preprocess`, `reconstruct`) | `MixedstatePtychography`, `PtychographyPreparation` | partial (0.5.0): CPU Double gradient descent with several probe states, orthogonalisation, seeded NumPy-compatible initial states, probe constraints; scope in the ptychography details below | useful | none upstream; np.allclose defaults |
| `MixedstateMultislicePtychography` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `DirectPtychography`: `SSB` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `DirectPtychography`: `OBF` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `DirectPtychography`: `WDD` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `MagneticPtychography` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `PtychographicTomography`, `MagneticPtychographicTomography` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `XRayMagneticPtychography` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| `PtychographyOptimizer`, `OptimizationParameter` | `CuboPtycho` | missing (planned: `CuboPtycho`) | useful | none upstream; np.allclose defaults |
| the constraint and method mixins (`ObjectNDConstraintsMixin`, `ProbeMethodsMixin`, ...), `ComplexProbe`, `AffineTransform` | — | not planned (internals; their features count under the classes above) | not planned: internals; their features count under the classes above | — |

## Further API audited 2026-10-21 (modules the tables above did not reach)

Source: an AST walk of the public functions and methods of `datacube`, `preprocess`, `process/{calibration,phase,strain,utils,fit,rdf,polar,diffraction,classification,wholepatternfit}`, `braggvectors`, `data`, `io`, `visualize` and `utils` at f050d207, checked against the names in the tables above.

| py4DSTEM | CuboKit | State | Mark | Tolerance (source) |
|---|---|---|---|---|
| `preprocess.electron_count`, `electron_count_GPU`, `calculate_thresholds`, `torch_bin`, `counted_datacube_to_pointlistarray`, `counted_pointlistarray_to_datacube` | — | missing | useful | none upstream; counted events exact |
| `preprocess.get_darkreference`, `get_bksbtr_DP` | — | missing | useful | none upstream; allclose defaults |
| `preprocess.get_background_streaks`, `_x`, `_y` | — | missing | useful | none upstream; allclose defaults |
| `preprocess.get_1D_polar_background`, `get_2D_polar_background` | — | missing | useful (with the radial background) | none upstream; allclose defaults |
| `preprocess.datacube_diffraction_shift` (Fourier and bilinear, per-pattern shifts) | `VirtualDiffraction.pattern(shift:)` shifts patterns inside the reducers | partial: no function that returns the shifted cube | useful | none upstream; allclose defaults |
| `preprocess.bin_data_mmap` (`DataCube.bin_Q_mmap`) | `ViewedCube` with `.detectorBin` | present (0.5.0; lazy) | useful | 1.2e-7 relative measured |
| `preprocess.median_filter_masked_pixels_2D` | — | missing | useful | exact for odd kernels |
| `preprocess.utils.bin2D` | `ViewedCube` / `CubeEditing.binScan` on cubes; no 2-D entry point | partial | useful | exact for integers |
| `preprocess.utils.make_Fourier_coords2D` | `FFT2D` frequencies are internal | missing | useful | exact |
| `process.utils.get_CoM` (2-D array centre of mass) | `CenterOfMass.measure` (per pattern in the cube) | partial: no single-array entry point | useful | none upstream; allclose defaults |
| `process.utils.radial_reduction`, `radial_integral`, `radial_elliptical_integral` | — | missing | useful (radial integration of patterns; DiffractoKit owns the polar transform) | none upstream |
| `process.utils.sector_mask`, `masks.make_circular_mask` | `DetectorMask.make` (circle, annulus, rectangle) | partial: no sector | useful | exact masks |
| `process.utils.get_qx_qy_1d` | `Calibration.detectorRowAxis` / `detectorColumnAxis` | present (0.8.0) | useful | exact to the Double arithmetic |
| `process.utils.get_maxima_1D`, `linear_interpolation_1D`, `add_to_2D_array_from_floats` | `BraggMaxima` (2-D); no 1-D | partial | useful | none upstream |
| `process.utils.fourier_resample`, `resample_data_diffraction(method="fourier")` | `CubeEditing.fourierResample` (2-D, Double), `CubeEditing.resampleDetectorFourier` | present (0.9.0) | study | as `resample_Q`: measured 2.1e-7 against py4DSTEM, 1.5e-14 against the band-limited truth |
| `process.utils.get_voronoi_vertices`, `get_ewpc_filter_function`, `electron_wavelength_angstrom`, `electron_interaction_parameter` | wavelength lives in `PtychographyProbe`; the rest absent | partial | useful | none upstream |
| `process.utils.multicorr.upsampleFFT`, `cross_correlate.align_and_shift_images` | `MatrixDFTCorrelation`, `ParallaxAlignment` (the same DFT upsampling) | partial: not exposed as py4DSTEM's function | useful | none upstream; allclose defaults |
| `process.utils.single_atom_scatter`, `elliptical_coords.*` (`cartesian_to_polarelliptical_transform`, `elliptical_resample`) | — | not planned | not planned: DiffractoKit (polar and scattering factors) and SimulatoKit | — |
| `process.utils.cluster.Cluster` | — | not planned | not planned: ACOM-adjacent clustering is DiffractoKit | — |
| `process.phase.utils.ComplexProbe` (`evaluate_aperture`, `evaluate_chi`, envelopes, `get_scattering_angles`, `get_spatial_frequencies`) | `PtychographyProbe.build` | partial: no coherence envelopes (see the ptychography details) | useful | none upstream |
| `process.phase.utils.AffineTransform`, `estimate_global_transformation`, `estimate_global_transformation_ransac` | — | missing | useful (scan-to-detector affine for ptycho) | none upstream |
| `process.phase.utils.fourier_ring_correlation`, `return_1D_profile`, `fourier_rotate_real_volume`, `fft_shift`, `fourier_translation_operator`, `spatial_frequencies`, `generate_batches`, `subdivide_into_batches`, `array_slice` | `FFT2D`, `ParallaxFourierShifter` cover `fft_shift`; the rest missing | partial | useful | none upstream |
| `process.phase.utils` periodic gradients, divergence and the 3-D Poisson/Nesterov helpers (`compute_gradient_periodic`, `compute_divergence_periodic`, `preconditioned_poisson_solver_periodic_3D`, `nesterov_*`) | `IDPC` (2-D Poisson only) | missing | not planned: 3-D tomography is outside the kit until multislice ptychography is registered | — |
| `process.phase.parameter_optimize.PtychographyOptimizer` (`grid_search`, `optimize`) | — | missing | useful | none upstream |
| `braggvectors.find_Bragg_disks_aiml*`, `_CUDA`, `_ipp`, `_dask`, `beta_parallel_disk_detection` | the learned detector stays in the app; `BraggDisks.detectAll` tiles in process | not planned | not planned: ML, CUDA and cluster backends | — |
| `BraggVectors.set_raw_vectors`, `BraggVectorMap.pixelsize/pixelunits` | `BraggScanResult`, no map object | partial | useful | exact |
| `BraggVectorClassification.__init__` (`braggpeaks`, `Qx`, `Qy`, `X_is_boolean`, `max_dist`), `get_braggpeak_labels_by_scan_position` | `BraggVectorClassification.init(vectors:referenceRows:referenceColumns:booleanData:maximumDistance:)`, `.peakLabels(...)` | present (0.9.0): labels and X exact (rows are `qx`) | useful | none upstream; labels exact, X exact |
| `get_initial_classes`, `get_initial_classes_by_cooccurrence`, `get_initial_classes_from_images` | `.initialClasses(...)`, `.setInitialClassesByCooccurrence(...)`, `.setInitialClasses(images:)` | present (0.9.0): class sets exact, W exact, H by minimum-norm least squares (LAPACK `dgesdd` SVD against `lstsq`; a non-finite matrix throws as NumPy raises) | useful | none upstream; np.allclose defaults, measured 1.5e-8 of the bound |
| `nmf`, `split`, `split_by_class_index`, `merge_iterative`, `merge_by_class_index`, `remove_class`, `accept`, `reject`, `get_class*`, `get_candidate_class*` | `.nmf`, `.split`, `.split(classIndex:)`, `.merge`, `.merge(classIndices:_:)`, `.removeClass`, `.accept`, `.reject`, `classPeakWeights`, `classImage`, `candidate*` | present (0.9.0): `merge()` itself is an empty method in 0.14.19; `merge` here is `merge_iterative` (see the notes); a lone class makes `merge` throw where NumPy raises `IndexError` | useful | none upstream; np.allclose defaults, measured 2.3e-8 (nmf), 0 (split, remove, merge by index), 3.3e-11 (merge) |
| `classutils.get_class_DP`, `get_class_DP_without_Bragg_scattering` | `BraggVectorClassification.classPattern(of:...)`, `.classPatternWithoutBraggScattering(of:...)` | present (0.9.0): integer shifts (ties to even) and Fourier shifts, dark reference | useful | none upstream; np.allclose defaults, measured 5e-11 |
| `Featurization` (`PCA`, `NMF` incl. `_nmf_single`, `GMM` incl. `_gmm_single`, `MinMaxScaler`, `shift_positive`, `mean/median/max_feature`, `concatenate_features`, `get_class_DPs`, `get_class_ims`) | `Featurization.pca/nmf/gmm/minMaxScaled/shiftedPositive/reduceFeatures/concatenate/classPatterns`, `NMF`, `GaussianMixture`, `NumPyRandomState` | partial (0.9.0): scikit-learn 1.9.1 numbers, NumPy's MT19937 reproduced bit for bit; NNDSVD uses an exact SVD (scikit-learn's is randomised); `get_class_DPs` nmf/gmm branches are buggy upstream and not copied | useful | none upstream; np.allclose defaults, worst measured 1.0e-3 of the bound (GMM), 1.4e-6 (iterative NMF), 8e-9 (PCA) |
| `Featurization.from_braggvectors`, `ICA`, `RobustScaler`, `spatial_separation`, `consensus`, `add_features`, `delete_features` | — | missing: `from_braggvectors` needs the Bragg types in CuboEmbedding; ICA and the two clustering steps are not ported | useful | none upstream |
| `process.rdf.*` (`get_radial_intensity`, `get_rdf`, `fit_scattering_factor`, `amorph.*`) | — | not planned | not planned: DiffractoKit (amorphous and polar analysis) | — |
| `process.polar.*` (`PolarDatacube`, `calculate_radial_statistics`, `calculate_FEM_local`, `calculate_annular_symmetry`, `calculate_pair_dist_function`) | — | not planned | not planned: DiffractoKit (polar elliptical transforms) | — |
| `process.wholepatternfit.*` (`WholePatternFit`, models) | — | not planned | not planned: DiffractoKit/SimulatoKit (whole-pattern forward fits) | — |
| `process.diffraction.*` (`Crystal`, orientation, ACOM, digital dark field, moire) | — | not planned | not planned: DiffractoKit | — |
| `io.filereaders.*` (`read_empad`, `load_mib`, `read_gatan_K2_bin`, `read_arina`, `read_dm`, `read_abTEM`) | `MappedCube(npy:)` only | not planned | not planned: FormatoKit | — |
| `visualize.*` (`show*`, overlays, `vis_RQ`, `vis_grid`), `utils.configuration_checker` | — | not planned | not planned: plotting and Python environment checks | — |

## Count (0.9.0; audited 2026-10-21, rows of the 0.9.0 slices updated)

One row is one method or option group. Counted: the rows marked study or useful. Not planned rows (plotting, Python-only,
other kits' territory, the learned detector) are listed with their reason and not counted. Earlier counts: 0.7.1 "79 done,
37 partial, 66 absent of 182" (the same tables without the audit of `preprocess`, `process.utils`, `process.phase.utils`,
`braggvectors` backends and the other modules; rows added by the audit are marked below).

| Area | present | partial | missing | not planned |
|---|---|---|---|---|
| DataCube: container and shape | 5 | 1 | 1 | 1 |
| DataCube: virtual images | 8 | 1 | 0 | 2 |
| DataCube: virtual diffraction | 9 | 0 | 1 | 0 |
| DataCube: calibration-bound properties and coordinates | 5 | 0 | 0 | 0 |
| DataCube: reshaping and pre-processing | 13 | 0 | 0 | 0 |
| DataCube: probe, beam stop and background methods | 9 | 0 | 1 | 0 |
| Calibration entries | 21 | 0 | 0 | 1 |
| Calibration processes | 18 | 4 | 2 | 1 |
| Probe | 12 | 1 | 1 | 1 |
| DPC and the phase base class's preprocessing | 13 | 5 | 3 | 4 |
| Strain | 8 | 5 | 1 | 1 |
| Bragg vectors and disk detection | 4 | 4 | 10 | 2 |
| Parallax | 3 | 3 | 3 | 1 |
| Ptychography | 0 | 3 | 8 | 1 |
| Further API audited 2026-10-21 (modules the tables above did not reach) | 7 | 11 | 11 | 9 |
| **Total** | **135** | **38** | **42** | **24** |

Study rows: 53, of which 47 present, 5 partial, 1 missing.
## 0.2.0 notes (2026-10-15)

- `RotationSolver` gains `curve(_:anglesDegrees:criterion:transpose:isCancelled:)` (py4DSTEM computes the same curve inside
  `DPC.preprocess`) and runs the grid in parallel; the CPU centre of mass runs positions in parallel; both bit identical.
- The Metal centre of mass agrees with the CPU path where |S| ≥ 1e-3 · Σ|wP|: Float32's absolute noise floor on
  a Metal reduction is about 3e-10 of the total intensity, so a pattern whose weighted sum is far below its total is
  outside the claim. py4DSTEM has no GPU centre of mass to compare with.
- The counts above are unchanged by 0.2.0 (no new py4DSTEM method).

## Fourier iDPC (app algorithm; outside the py4DSTEM method totals)

| Reference | CuboKit | State |
|---|---|---|
| NumPy transcription of app Fourier iDPC | `IDPC.integrate`, `integratePhysical`, `IDPCPhysicalCalibration`, `IDPCBoundary` | done: periodic/centered zero-padded boundaries, regularization and physical sampling; CPU Double |

Generator `tools/idpc/pins.py` explicitly distinguishes this oracle from py4DSTEM's
iterative `DPC`. Absolute phase tolerance `1e-12` rad; details in `docs/idpc.md`.

## Hot-pixel filtering details

| py4DSTEM f050d207 | CuboKit | State |
|---|---|---|
| `DataCube.filter_hot_pixels` | `HotPixelFilter.filter` | done for eager Float32 cubes; exact NumPy strided/pairwise mean order, wrapped 21-pixel classification, row-major sequential clipped medians |

This refines the preprocessing row above (not counted twice). Generator
`tools/hot-pixels/pins.py` calls the actual pinned method. Eight fixtures (30,403 bytes)
pin means, masks and every cleaned sample exactly today (measured gap 0; a tolerance against py4DSTEM would be allowed).
Supported input and numerical details: `docs/hot-pixels.md`.

## CPU ptychography details (refines the headline above; not counted twice)

| py4DSTEM f050d207 | CuboKit | State |
|---|---|---|
| `ComplexProbe.build` | `PtychographyProbe.build` | partial: defocus, C12, C30, hard/cosine aperture; no coherence envelopes |
| `SingleslicePtychography.reconstruct` gradient descent | `SingleslicePtychography.reconstruct` | partial: preprocessed amplitudes, complex object, fixed positions, full batches, probe updates, threshold/pure-phase object constraints, probe centre-of-mass (`fixProbeCenterOfMass`, default off), probe amplitude top-hat, fixed aperture (`fixProbeAperture`) and real-space support (`probeSupportMask`), py4DSTEM's error normalisation (`errorNormalization`), position correction (`positionCorrection`: `positions_step_size`, `max_position_update_distance`, `max_position_total_distance`, `fix_positions_com`); Fourier-amplitude, aberration-fitting, object filters, batching and the projection schemes absent |
| `MixedstatePtychography.reconstruct` (gradient descent) | `MixedstatePtychography.reconstruct`, `.orthogonalize`, `.initialProbes` | partial: as single-slice, with `orthogonalizeProbes`; `fix_probe_com` shifts every state by state 0's centre of mass; aperture constrains state 0, amplitude and support every state; the NumPy `RandomState` initial states are reproduced for a given seed; per-state eigenvector phase is a convention (tests align it) |
| `MultislicePtychography.reconstruct` (gradient descent), `_precompute_propagator_arrays` | `MultislicePtychography.reconstruct`, `.propagators` | partial: slice stack, `slice_thicknesses` per interval, `identical_slices`, probe constraints as single-slice; `kz_regularization_filter`, `theta_x/theta_y`, TV and `middle_focus` absent; the caller gives one thickness per interval (py4DSTEM broadcasts a scalar) |
| `_normalize_diffraction_intensities` | `PtychographyPreparation.amplitudes` | partial: per-position origins, order-1 (bilinear) and order-3 (spline, grid-wrap) shifts, square root of the clipped values, the mean intensity; `crop_patterns`, `positions_mask`, `region_of_interest_shape` resampling absent |
| `_calculate_scan_positions_in_pixels` (raster) | `PtychographyPreparation.scanPositions` | done: rotation (Float32 matrix as py4DSTEM), transpose, padding; `initial_scan_positions` and `positions_offset_ang` absent |
| `_initialize_object`, `center_positions_in_fov` | `objectShape`, `centerPositions` | done (`object_fov_ang` absent) |
| `_initialize_probe` (no vacuum probe) | `PtychographyPreparation.initialProbe` | done for the aperture probe with defocus, C12, C30; vacuum probes absent |
| odd and non-square probe, detector and object sizes | all of the above | pinned (`odd.json`) |
| mac4DSTEM `PtychographyPreparer` | `PtychographyPreparationOptions.mac4DSTEM(...)`: bilinear shifts, per-axis padding, no re-centring, clipped mean | named option; padding and mean pinned to the app source by a transcription, the rest to py4DSTEM |

Pins: `tools/ptycho/pins.py` (`ptycho.json`), `tools/ptycho/preprocess_pins.py` (`preprocess.json`), `tools/ptycho/engines_pins.py` (`engines.json`), all in `Tests/CuboPtychoTests/Fixtures/`.
Tolerance `2e-6` normwise maximum error with unit floor; see DocC Scope for numerical limits.

## Reciprocal lattice strain details

| Reference / source | CuboKit | State |
|---|---|---|
| `BraggVectors.histogram`, `StrainMap.choose_basis_vectors`, `fit_basis_vectors`, `get_strain` | `StrainMapping.histogram`, `chooseBasisVectors`, `compute` | done: pinned to py4DSTEM (tolerance 1e-9, measured 1.4e-14; histogram identical); `BraggDiskOptions.tieOrder = .numpy` reproduces py4DSTEM's order of equal maxima; pins `tools/strain/mapping_pins.py` |
| `LatticeStrain.chooseBasis` (supplied maxima) | `LatticeStrain.chooseBasis` | numeric selection done |
| `index_bragg_directions`, `StrainMap.fit_basis_vectors` indexing | `LatticeStrain.index` overloads | done for supplied histogram directions; nearest-direction inclusive spacing and ties-to-even indices |
| `fit_lattice_vectors` | `LatticeStrain.fit` | done: intensity squared objective, minimum-norm SVD, mean-distance residual |
| `get_reference_g1g2` | `LatticeStrain.reference` | done: plain component medians of caller-selected valid ROI |
| `get_strain_from_reference_g1g2`, `get_rotated_strain_map` | `LatticeStrain.strain` | done: py4DSTEM reference, rotation and theta signs; caller maps per scan position |
| App c809679c intensity-once fit | `fitIntensityOnce` | named option: Float32 vectors, Double normal equations, weight floor 1e-6, RMS residual; original helper pin |
| App c809679c MAD reference | `referenceMAD` | named option: Float32 MAD rejection with 2% scale floor and half-population fallback; original helper pin |
| App c809679c cluster-consensus selection | `chooseBasisConsensus`, `consensusDiagnostics` (the same scoring for a supplied basis) | named option: repeated vector clusters, bounded condition/support scoring; original helper pin |

These rows refine the strain headline; plotting and the `StrainMap` state object are not APIs.
`tools/strain/pins.py` calls the actual pinned Python functions and independently
compiles the original app helper text. Binary f64 pins cover planted strain and
a public DM4 crop. Absolute tolerance 1e-10, maximum measured fit gap 5.33e-15
and strain gap 8.88e-16; all three original-source options have gap/tolerance 0.
Conventions, bounds and reproducibility: `docs/strain.md`.
Preprocessing pins: `tools/parallax/preprocess_pins.py`, four cases, approximately 114 KB.
Normwise maximum tolerance `2e-6` with unit floor (Double port vs Float32 storage);
measured maximum gap `1.34e-7`. App unwindowed stack independently transcribed from
`ParallaxPreprocessing.swift` at the app source pin; no app types are included.



## Frame transforms (crop and bin of origins)

| py4DSTEM | CuboKit | Status |
|---|---|---|
| `crop_data_diffraction` / `crop_data_real` (slicing) | `FrameTransform.cropMap` (scan-indexed maps), `movePosition` (offsets) | done; exact selection |
| `bin_data_diffraction` Q pixel size × N | `FrameTransform.binnedPixelSize` | done |
| origins and lengths under crop/bin | `movePosition`, `moveOriginMap`, `binnedLength`, `binnedCoordinate`, `sourceCoordinate` | py4DSTEM leaves them untouched; these are the mac4DSTEM app rules (pixel-centre bin, half-open bounds), pinned to the app source at c809679c, not to py4DSTEM |

`binnedLength` and `binnedPixelSize` are trivial scalar rules, tested by hand-written asserts, not by the source pin; callers own the check that a scan map matches the source scan before `cropMap`.


## Embedding: PCA and k-means (not py4DSTEM; scikit-learn 1.4.2 is the reference)

These rows sit outside the py4DSTEM tallies above. py4DSTEM's own featurization calls scikit-learn's PCA.

| Reference / source | CuboKit | State |
|---|---|---|
| `sklearn.decomposition.PCA(n_components=k, svd_solver="full")`: `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, `transform` | `PCA.fit`, `PCA.transform` | done: `svd_flip` u-based sign; covariance eigen route, not SVD; whiten, `n_components` fractions/`mle`, other solvers, `inverse_transform`, `score` absent |
| `sklearn.cluster.KMeans(init=<array>, n_init=1, algorithm="lloyd")`: `cluster_centers_`, `labels_`, `inertia_`, `n_iter_` | `KMeans.fit` | done: strict and tolerance stopping, empty-cluster relocation (single empty cluster pinned), final relabel; k-means++ / random init, `n_init`, sample weights, elkan, `predict`, `transform` absent |
| mac4DSTEM `DiffractionEmbedding` covariance PCA (c809679c) | `PCA.fit(..., variant: .diffractionEmbeddingApp)`, `PCA.project` (Float32 vDSP projection of one vector) | named option: Float32, divisor rows, raw LAPACK sign, clamped component count; original-source pin |
| mac4DSTEM `DiffractionEmbedding.kMeans` | `KMeans.fitSeeded` | named option: splitmix64 k-means++, 50 passes, no tolerance; original-source pin |
| mac4DSTEM `DiffractionEmbedding.embed` (log1p, max-normalise, box bin) | `DiffractionFeatures.binnedVector` | no external counterpart; original-source pin to Float32 rounding |

Pins: `tools/embedding/pins.py`, `Tests/CuboEmbeddingTests/Fixtures/embedding.json`. Gaps and conventions: `docs/embedding.md`.


## Bragg disk detection details

| Reference / source | CuboKit | State |
|---|---|---|
| `find_Bragg_disks` (`_find_Bragg_disks_single`), `get_cross_correlation_FT` | `BraggDetector.detect` / `correlate`, `BraggDisks.detect` | done: hybrid cross-correlation in Double, any detector size; 30 py4DSTEM cases, gap 1.5e-12 px |
| `get_maxima_2D`, `filter_2D_maxima`, `linear_interpolation_2D` | `BraggMaxima.find` | done: the filter order, the half-open edge, parabolic and multicorr (via CuboParallax's `MatrixDFTCorrelation`) |
| `gaussian_filter` (mode `reflect`) | `GaussianFilter.apply(..., edge: .reflect)` | done: pinned to scipy 1.18.0 |
| App `relativeReferenceMinimumRadiusPx` | `relativeReferenceMinimumRadius` | named option; pinned to the app source (Float32, gap 1.8e-4 px) |
| App non-finite pixel fill (D019) | `nonFinitePixels = .neighbourMedian` | named option; py4DSTEM returns no peaks |
| App multicorr peak order | `multicorrOrder = .correlationIntensity` | named option; py4DSTEM sorts by refined intensity |
| App `detectorAdapted` | `BraggDiskOptions.detectorAdapted` | named helper; pinned to the app source |
| App `ProbeRingHint` | `ProbeRingHint.outerEdge(of:)` | app-only (no py4DSTEM counterpart); pinned to the app source |

Peak lists are `row`/`column`/`intensity`, brightest first. The app's Float32 arithmetic, kernel construction from a measured
probe, funnel diagnostics, validation and provenance are not ported (`docs/bragg.md`). Tolerances: py4DSTEM 1e-9 px and 1e-10
relative; app 1e-3 px and 1e-5 relative (measured 1.5e-12 / 1.3e-13 and 1.8e-4 / 1.2e-6). Reproduction: `tools/bragg/pins.py`.

## Known limits and decisions

Open work is in the [issues](https://github.com/mac4DSTEM/CuboKit/issues), including the ones closed as *not planned* and
parked for later. Speed against py4DSTEM is in `docs/benchmarks.md`.

- **Metal against the CPU without a resident cube.** The mean, max and selected-area patterns are bandwidth-bound, and the
  CPU reads the cube in place, so the CPU stays their default (typical size, 100 x 100 x 64 x 64 Float32: Metal mean and max
  6.4 ms staged against 2.4 ms on the CPU, mask mean 9.7 against 1.6). The virtual image and the centre of mass are faster on
  Metal for a `MemoryCube`, the centre of mass at 128 x 128 detectors is not (one thread per pattern). Choosing the path is
  the app's.
- **Double FFT on lengths vDSP does not take** (Bluestein: padded parallax sizes, ptychography detectors such as 165) is bound
  by vDSP's Double FFT; a vectorised Bluestein on vDSP's DFT measured 1.1x and was not kept. Float32 callers use
  `ParallaxFourierShifter.shiftedSingle` and single-slice ptychography `.single`, which take every size. The hot-pixel mean
  stays Float32 on purpose (`HotPixelFilter`).
- **`Calibration` follows py4DSTEM:** binning, thinning, padding and resampling change the pixel sizes but not the origin (in
  detector pixels), so an origin measured before `bin_Q(2)` is wrong afterwards. A caller that stores the origin rescales it.
- **`ResidentCube`** is capped at half of the physical memory and needs the cube in one Metal buffer
  (`ResidentCube.availability(of:)` says which limit and how far over). No chunked variant (decided 2026-10-22): the app falls
  back to the tiled path.
- **Metal sums are Float32 within a tile:** a per-tile sum or product past about 3.4e38, or a selection weight below the
  Float32 range, gives ±∞ or NaN where the CPU Double path stays finite. Not reachable with detector counts.
- **Polar cube radius scale:** the polar cube's radius is on the minor-axis scale (`rr = (b / a) * hypot`), the radial
  background's lookup radius on the major-axis scale. Both are py4DSTEM's, ported as they are; for `a != b` the background
  profile is stretched by `a / b` against the polar bins (`testRadialBackgroundOfACircle`: a circle agrees with the planted
  exponential to 0.2 %).
- **Amorphous-ring fit:** chaotic at rounding level in its poorly determined parameters (py4DSTEM's own result moves by up to
  7e-4 in the axes when the start changes by one ulp on the `tilted` case); tests compare ellipses at 1e-6 and the residual sum
  of squares at 1e-6 relative, not parameter by parameter. `fit_ellipse_1D` with default `fitradii` raises `TypeError`
  upstream; the kit's default is `(0, min(shape)/2)`.
- **`PolarCube`** keeps the whole cube in Double when asked for (`PolarCubeData`); a lazy `CubeSource` adapter or a Float32
  variant is for the consumer that needs it. The polar cube and radial background are CPU only (a Metal path would be bound
  by the host copy).
- **Shared numerics:** NMF, the Gaussian filter, the median and the SVD are shared (CuboBragg's `lstsq` runs on
  CuboEmbedding's LAPACK SVD), except `Featurization`'s feature median and the sorted-window medians of `HotPixelFilter` and
  `medianFilterMaskedPixels`.
- **Not ported:** `Featurization.from_braggvectors`, ICA, `RobustScaler`, `spatial_separation` and `consensus`. The GMM k-means
  start is scikit-learn 1.4.2's Lloyd loop with the data centred as 1.9.1 does (every pin agrees on labels). Upstream
  `NMF(init=None)` with `random_state=None` is not bit repeatable (an unseeded randomised SVD).
