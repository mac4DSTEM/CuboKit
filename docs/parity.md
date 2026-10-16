# Parity with py4DSTEM

A checklist against py4DSTEM's documented API in the areas CuboKit holds or plans, at the pinned commit `f050d207`
(0.14.19), so "complete" is a measured claim (suite ROADMAP, "Kits", phase 4). Updated at every tag. The areas are the
`DataCube` (`datacube/`), the calibration of origins and probes (`process/calibration/`, `data/calibration.py`), the
`Probe` class (`braggvectors/probe.py`), DPC and the phase base class's preprocessing (`process/phase/`), and the headline
lists for strain, Bragg vectors and disk detection, parallax and ptychography. Crystallography (`Crystal`, orientation,
ACOM) is DiffractoKit's; file formats (`read`, `import_file`, `save`) are FormatoKit's; simulation is SimulatoKit's;
plotting and `show*` are not ported.

Legend: **done** (pinned against py4DSTEM and planted truth, `docs/registration-slice-1.md` and its successors),
**partial** (what is missing says so), **absent** (not in the kit; "planned: ..." names the slice or module where a plan
exists), **app** (lives in mac4DSTEM, not in a kit), **n/a** (plotting or Python-only). "done (slice 4)" rows are
`docs/registration-slice-4.md`.

Where py4DSTEM takes a Python-only argument (`verbose`, `progress_bar`, `returncalc`, `name`, `plot`, `device="gpu"` for
CuPy, `dask`), the row says n/a once per method, not per argument.

## `DataCube`: container and shape (datacube/datacube.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `DataCube(data)`, `.data` | `MemoryCube(shape:values:)`, `MappedCube(npy:)` (lazy, memory-mapped), `CubeSource` | done (Float32 on read; ten element types) |
| `R_Nx`, `R_Ny`, `Q_Nx`, `Q_Ny` | `CubeShape.scanRows`, `scanColumns`, `detectorRows`, `detectorColumns` | done |
| `Rshape`, `Qshape`, `R_N` | `CubeShape`, `scanCount`, `detectorCount` | done |
| `datacube[rx, ry]` | `readPattern(row:column:)` | done |
| `bin_Q_mmap`, dask-backed lazy data | `MappedCube` | partial: lazy and memory-mapped; no bin on read |
| `DataCube(calibration=..., slicelabels=...)` | — | absent (planned: a calibration model, below) |
| `copy`, `add`, the emdfile tree and `metadata` | — | n/a (value types; emdfile is FormatoKit's) |
| `.polar` (polar-transformed datacube) | — | n/a (DiffractoKit: polar elliptical transforms) |

## `DataCube`: virtual images (datacube/virtualimage.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `get_virtual_image(mode, geometry)`: `point`, `circle`, `annulus`, `rectangle` | `VirtualImaging.image(of:geometry:)` | done (CPU and Metal) |
| `get_virtual_image(mode="mask", geometry=array)` | `VirtualImaging.image(of:geometry:)` with a mask geometry | done (CPU and Metal) |
| `get_virtual_image(return_mask=True)`, `make_detector` | `DetectorMask.make` | done |
| `get_virtual_image(centered=True)` | `VirtualImaging.image(of:geometry:centeredOn:)` | done (slice 3) |
| `get_virtual_image(calibrated=True)` | `VirtualImaging.image(of:geometry:pixelSize:)` | done (slice 3; centred and calibrated together refused) |
| `get_virtual_image(shift_center=True, subpixel=False)` | `shift: .rounded` | done (slice 3; CPU) |
| `get_virtual_image(shift_center=True, subpixel=True)` | `shift: .bilinear` | done (slice 3; CPU) |
| `get_calibrated_detector_geometry` | applied inside `VirtualImaging.image` | partial: not exposed as a function |
| `make_bragg_mask` | — | absent (planned: with `get_braggmask`, the Bragg-disk module) |
| `get_virtual_image(dask=True)`, `test_config` | — | n/a (Python-only; the lazy cube tiles instead) |
| `position_detector`, `show` | — | n/a (plotting) |

## `DataCube`: virtual diffraction (datacube/virtualdiffraction.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `get_dp_mean` | `VirtualDiffraction.mean` | done |
| `get_dp_max` | `VirtualDiffraction.max` | done |
| `get_dp_median` | `VirtualDiffraction.median(of:)` | done (slice 4; CPU) |
| `get_virtual_diffraction(method="mean")`, `"max"` | `VirtualDiffraction.mean`, `max`, `meanAndMax` | done |
| `get_virtual_diffraction(method="median")` | `VirtualDiffraction.pattern(of:statistic: .median, scan:)` | done (slice 4; CPU) |
| `get_virtual_diffraction(mask=)`, boolean or float real-space mask | `pattern(of:statistic:scan: .mask / .weights)` | done (slice 4; CPU) |
| `get_virtual_diffraction(mask=)`, complex mask | — | absent (not ported) |
| `get_virtual_diffraction(shift_center=True, subpixel=False)` (mean, max) | `VirtualDiffraction.meanAndMax(of:shift: .rounded)` | done (slice 3; CPU) |
| `get_virtual_diffraction(shift_center=True, subpixel=True)` (mean, max) | `shift: .fourier` | done (slice 3; CPU) |
| `get_virtual_diffraction(shift_center=True, mask=)` (mean, max) | `pattern(of:statistic:scan:shift:)` | done (slice 4; CPU) |

## `DataCube`: calibration-bound properties and coordinates (datacube/datacube.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `calibrate()` (rebuild the coordinate vectors from the calibration) | — | absent (planned: a calibration model) |
| `rxx`, `ryy`, `qxx`, `qyy` (calibrated meshgrids) | — | absent (planned: a calibration model) |
| `rxx_raw`, `ryy_raw`, `qxx_raw`, `qyy_raw` (pixel meshgrids) | — | absent (planned: a calibration model) |
| `qxxs(rx, ry)`, `qyys(rx, ry)` (per-position shifted Q axes) | — | absent (planned: a calibration model) |
| `Q_pixel_size`, `Q_pixel_units`, `R_pixel_size`, `R_pixel_units` | `pixelSize:` argument of calibrated imaging | partial: Q size as an argument, not stored; no R size, no units |

## `DataCube`: reshaping and pre-processing (datacube/datacube.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `set_scan_shape` | — | absent (planned: reshape slice) |
| `swap_RQ` | — | absent (planned: reshape slice) |
| `swap_Rxy` | — | absent (planned: reshape slice) |
| `swap_Qxy` | — | absent (planned: reshape slice) |
| `crop_Q` | — | absent (planned: reshape slice; mac4DSTEM's `LoadSpecification` crops at read time, app today) |
| `crop_R` | — | absent (planned: reshape slice; `LoadSpecification`, app today) |
| `bin_Q` | — | absent (planned: reshape slice; `LoadSpecification`, app today) |
| `bin_R` | — | absent (planned: reshape slice; `LoadSpecification`, app today) |
| `thin_R` | — | absent (planned: reshape slice) |
| `pad_Q` | — | absent (planned: reshape slice) |
| `resample_Q` (bilinear, `conserve_array_sums`) | — | absent (planned: reshape slice) |
| `filter_hot_pixels` (`thresh`, `ind_compare`, `return_mask`) | `HotPixelFilter.filter`, `.mask`, `.replacing` | done: eager Float32 cube, means/masks/pixels pinned bit for bit; wrapped classification and clipped sequential replacement |
| `median_filter_masked_pixels` | — | absent (planned: with `filter_hot_pixels`) |

## `DataCube`: probe, beam stop and background methods (datacube/datacube.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `get_probe_size` (`thresh_lower`, `thresh_upper`, `N`) | `ProbeSize.estimate` | done (slice 2; bit for bit) |
| `get_vacuum_probe(ROI=boolean array)` | `VacuumProbe.measure(of:scanMask:)` | done (slice 4) |
| `get_vacuum_probe(ROI=4-tuple)` | — | absent (not ported: pass the equivalent boolean ROI) |
| `get_vacuum_probe(align, zero_vacuum, mask)` | `VacuumProbe.measure(of:scanMask:align:zeroVacuum:detectorMask:)` | done (slice 4) |
| `get_beamstop_mask` (`threshold`, `distance_edge`, `include_edges`, `sigma`, `scale_radial`) | `BeamstopMask.make(from:threshold:distanceEdge:includeEdges:sigma:scaleRadial:)` | done (slice 4) |
| `get_beamstop_mask(use_max_dp=True)` | the caller passes the max pattern | done (slice 4; via the caller's pattern) |
| `get_radial_bkgrnd` | — | absent (planned: Bragg-disk module) |
| `get_radial_bksb_dp` | — | absent (planned: Bragg-disk module) |
| `get_local_ave_dp` (`radial_bksb`, `sigma`, `braggmask`) | — | absent (planned: Bragg-disk module) |
| `get_braggmask` | — | absent (planned: Bragg-disk module) |

## Calibration entries (data/calibration.py)

CuboKit has no stateful `Calibration` object yet; its measurements return plain values (`OriginMap`, `OriginFit`,
`RotationSolution`, `ProbeEstimate`) that the caller stores.

| py4DSTEM | CuboKit | State |
|---|---|---|
| `Calibration` container: `attach`, `root`, `register_target`, `targets`, `calibrate`, `copy` | — | absent (planned: a calibration model) |
| `Calibration.to_h5`, `from_h5` | — | n/a (FormatoKit) |
| `Q_pixel_size` (`set_`, `get_`, `qpixsize`) | `pixelSize:` argument | partial: a number passed in; not stored |
| `Q_pixel_units` (`set_`, `get_`, `qpixunits`) | — | absent (planned: a calibration model) |
| `R_pixel_size` (`set_`, `get_`, `rpixsize`) | — | absent (planned: a calibration model) |
| `R_pixel_units` (`set_`, `get_`, `rpixunits`) | — | absent (planned: a calibration model) |
| `qx0`, `qy0` (per-position origin, `set_`, `get_`) | `OriginFit.fit` result | partial: the fitted map is returned, not stored |
| `qx0_mean`, `qy0_mean`, `get_origin_mean` | `OriginMap.meanRow`, `meanColumn` | partial: values only |
| `qx0shift`, `qy0_shift`, `get_origin_shift`, `origin_shift` | — | absent (planned: with the calibration model; the shift is the origin minus its mean) |
| `origin`, `set_origin`, `get_origin(rx, ry)` | `OriginFit` maps | partial: maps are returned; no `(rx, ry)` lookup or store |
| `qx0_meas`, `qy0_meas`, `origin_meas`, `set_origin_meas`, `get_origin_meas` | `OriginMap.row`, `column` | partial: values only |
| `origin_meas_mask`, `set_origin_meas_mask`, `get_origin_meas_mask` | `scanMask` of `OriginMeasurement.measure` | partial (slice 4): the mask is an argument; not stored |
| `a`, `b`, `theta`, `set_a`/`get_a`, `set_b`/`get_b`, `set_theta`/`get_theta` | — | absent (planned: a calibration model; the fit itself may be DiffractoKit's geometry) |
| `ellipse`, `set_ellipse`, `get_ellipse` (a, b, theta) | — | absent (planned: a calibration model) |
| `p_ellipse`, `set_p_ellipse`, `get_p_ellipse` | — | absent (planned: a calibration model) |
| `QR_rotation_degrees`, `set_QR_rotation`, `get_QR_rotation`, `..._degrees` | `RotationSolution.degrees`, `radians` | partial: solved, not stored |
| `QR_flip`, `set_QR_flip`, `get_QR_flip` | `RotationSolution.transposed` | partial: solved, not stored |
| `QR_rotflip`, `set_QR_rotflip`, `get_QR_rotflip`, `..._degrees` | `RotationSolution` | partial: the pair is returned, not stored |
| `probe_semiangle`, `probe_param`, `set_probe_param`, `get_probe_param` | `ProbeEstimate.radius` | partial: radius in pixels only |
| `probe_center`, `set_probe_center`, `get_probe_center` | `ProbeEstimate.centerRow`, `centerColumn` | partial: values only |
| `probe_convergence_semiangle_pixels`, `set_`, `get_` | `ProbeEstimate.radius` | partial: values only |
| `probe_convergence_semiangle_mrad`, `set_`, `get_` | — | absent (planned: a calibration model; needs the Q pixel size and units) |

## Calibration processes (process/calibration/)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `get_origin_single_dp` | the per-pattern step of `OriginMeasurement.measure` | partial: no single-pattern entry point |
| `get_origin(r, rscale)` | `OriginMeasurement.measure(radius:radiusScale:)` | done (slice 3; CPU) |
| `get_origin(dp_max=)` | the max pattern is computed internally | partial: no argument to pass it |
| `get_origin(mask=)` | `OriginMeasurement.measure(of:scanMask:)` | done (slice 4; NaN outside the mask) |
| `get_origin(fast_center=True)` | `OriginMeasurement.measure(of:scanMask:fastCenter:)` | done (slice 4) |
| `get_origin(remove_NaN=True)` | `OriginMap.replacingNaNWithMean()` | done (slice 4) |
| `get_origin_friedel(mask, upsample_factor)` | `FriedelOrigin.measure(of:mask:upsampleFactor:)`, `.origin(of:)` | done (slice 4; CPU only) |
| `get_origin_friedel(device="gpu")` | — | absent (planned: a Metal path, later lane) |
| `fit_origin(fitfunction="constant")` | `OriginFit.fit(model: .constant)` | done (slice 3) |
| `fit_origin(fitfunction="plane")` | `OriginFit.fit(model: .plane)` | done (slice 3) |
| `fit_origin(fitfunction="parabola")` | `OriginFit.fit(model: .parabola)` | done (slice 3) |
| `fit_origin(fitfunction="bezier_two")` | `OriginFit.fit(model: .bezierTwo)` | done (slice 3) |
| `fit_origin(robust, robust_steps, robust_thresh)` | `OriginFit.fit(robust:robustSteps:robustThreshold:)` | done (slice 3) |
| `fit_origin(mask=)` | `OriginFit.fit` | partial: NaN entries are left out of the fit; no boolean mask argument |
| `fit_origin(returnfitp=True)` | — | absent (the fit coefficients are not returned) |
| `get_probe_size` (`process/calibration/probe.py`) | `ProbeSize.estimate` | done (slice 2; bit for bit) |
| `fit_ellipse_1D` | — | absent (planned: a calibration model; the fit may be DiffractoKit's geometry) |
| `fit_ellipse_amorphous_ring` | — | absent (planned: with `fit_ellipse_1D`) |
| `ellipse_err`, `double_sided_gaussian`, `double_sided_gaussian_fiterr` | — | absent (planned: with `fit_ellipse_1D`) |
| `constrain_degenerate_ellipse` | — | absent (planned: with `fit_ellipse_1D`) |
| `get_Q_pixel_size(q_meas, q_known, units)` | — | absent (planned: a calibration model) |
| `get_dq_from_indexed_peaks` | — | absent (planned: a calibration model; needs indexed peaks) |
| `get_Qvector_from_Rvector`, `get_Rvector_from_Qvector` | — | absent (planned: a calibration model; needs the QR rotation) |
| `compare_QR_rotation` | — | n/a (plotting) |

## `Probe` (braggvectors/probe.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `Probe(data)` with `probe`, `kernel`, `alpha`, `origin` | functions on a detector image | partial (slice 4): no stateful `Probe` object |
| `Probe.from_vacuum_data` (`mask`, `threshold`, `expansion`, `opening`) | — | absent (planned: the Bragg-disk module) |
| `Probe.generate_synthetic_probe(radius, width, Qshape)` | `SyntheticProbe.make(radius:width:detectorRows:detectorColumns:)` | done (slice 4) |
| `Probe.measure_disk` (`thresh_lower`, `thresh_upper`, `N`, `zero_vacuum`, `alpha_max`) | `ProbeSize.estimate` (the same algorithm), then `VacuumProbe.zeroVacuum` | done (slice 4) |
| `Probe.measure_disk(plot=True)` | — | n/a (plotting) |
| `Probe.zero_vacuum` | `VacuumProbe.zeroVacuum(_:center:alpha:alphaMax:)` | done (slice 4) |
| `Probe.zero_vacuum_sinusoid` (`s`, `w`, `p`) | `VacuumProbe.dampVacuum(_:center:alpha:start:width:power:)` | done (slice 4) |
| `Probe.get_kernel(mode="flat")` | `ProbeKernel.flat` | done (slice 4) |
| `Probe.get_kernel(mode="gaussian", sigma)` | `ProbeKernel.gaussian` | done (slice 4) |
| `Probe.get_kernel(mode="sigmoid", radii)` | `ProbeKernel.sigmoid(profile: .sineSquared)` | done (slice 4) |
| `Probe.get_kernel(mode="sigmoid_log", radii)` | `ProbeKernel.sigmoid(profile: .sineSquared)` (py4DSTEM dispatches this mode to the sine² trench; `.logistic` is `type="logistic"`) | done (slice 4) |
| `Probe.get_probe_kernel_flat(origin, bilinear)` | `ProbeKernel.flat(_:origin:bilinear:)` | done (slice 4) |
| `Probe.get_probe_kernel_edge_gaussian(sigma, origin, bilinear)` | `ProbeKernel.gaussian(_:sigma:origin:bilinear:)` | done (slice 4) |
| `Probe.get_probe_kernel_edge_sigmoid(type="sine_squared")` | `ProbeKernel.sigmoid(_:innerRadius:outerRadius:profile: .sineSquared)` | done (slice 4) |
| `Probe.get_probe_kernel_edge_sigmoid(type="logistic")` | `ProbeKernel.sigmoid(..., profile: .logistic)` | done (slice 4) |

## DPC and the phase base class's preprocessing (process/phase/dpc.py, phase_base_class.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `DPC(datacube, energy, ...)` | `CenterOfMass`, `CenterOfMassReference`, `RotationSolver`, `DPC` | partial: functions, no stateful reconstruction object |
| `_calculate_intensities_center_of_mass` | `CenterOfMass.measure` | done (slice 2; CPU and Metal) |
| `_extract_intensities_and_calibrations_from_datacube` (reciprocal sampling, mask) | `CenterOfMass.measure(mask:)`, `reciprocalSampling:` | partial: the calibration is passed in, not read from a cube |
| `DPC.preprocess(dp_mask)` | `CenterOfMass.measure(mask:)` | done (slice 2) |
| `DPC.preprocess(fit_function="plane")`, `"constant"` | `CenterOfMassReference.planeFit`, `.constant` | done (slice 2) |
| `DPC.preprocess(fit_function="parabola")`, `"bezier_two"` | `OriginFit` has them | partial: not wired into `CenterOfMassReference` |
| `DPC.preprocess(force_com_shifts)` | `CenterOfMassReference.given` | done (slice 2) |
| `DPC.preprocess(force_com_measured)` | the measured `ScanVectorField` is an input | done (slice 2) |
| `DPC.preprocess(rotation_angles_deg)` | `RotationSolver.solve(anglesDegrees:)` | done (slice 2) |
| `DPC.preprocess(maximize_divergence=False)`: curl | `RotationCriterion.minimizeCurl` | done (slice 2) |
| `DPC.preprocess(maximize_divergence=True)`: divergence | `RotationCriterion.maximizeDivergence` | done (slice 2) |
| `DPC.preprocess(force_com_rotation, force_com_transpose)` | `forceRotationDegrees:`, `forceTranspose:` | done (slice 2) |
| `DPC.preprocess(padding_factor)` | `DPC.reconstruct(paddingFactor:)` | done (slice 2) |
| the rotation search curves (`plot_rotation` data) | `RotationSolution.curve`, `curveTransposed` | done (slice 2; the plot itself n/a) |
| `DPC.preprocess(vectorized_com_calculation, plot_*, device, clear_fft_cache)` | — | n/a (Python memory strategy, plotting, CuPy) |
| `DPC.reconstruct(max_iter, step_size, stopping_criterion, backtrack)` | `DPC.reconstruct(maxIterations:stepSize:stoppingCriterion:backtrack:)` | done (slice 2; CPU, Double) |
| `DPC.reconstruct(gaussian_filter_sigma, gaussian_filter)` | — | absent (planned: DPC filters, later lane) |
| `DPC.reconstruct(q_lowpass, q_highpass, butterworth_order, butterworth_filter)` | — | absent (planned: DPC filters, later lane) |
| `DPC.reconstruct(store_iterations=True)` | `DPCReconstruction.errors`, `accepted` | partial: the error history is kept; the per-iteration objects are not |
| `DPC.reconstruct(device="gpu")` | — | absent (planned: a Metal DPC, later lane) |
| `DPC.reconstruct(reset, progress_bar)` | — | n/a (Python state and progress bar) |
| result: `object_phase`, `error`, `sampling`, rotation and transpose | `DPCReconstruction.phase`, `errors`, `RotationSolution` | done (slice 2) |
| `DPC.visualize`, `show_complex_CoM`, `to_h5` | — | n/a (plotting; FormatoKit) |
| `set_device`, `set_storage`, `clear_device_mem`, `attach_datacube`, `reinitialize_parameters`, `set_save_defaults` | — | n/a (Python object plumbing) |

## Strain (process/strain/)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `StrainMap(braggvectors)`, `braggvectors`, `rshape`, `qshape`, `origin`, `reset_calstate` | `CuboStrain` | absent (planned: `CuboStrain`, placeholder) |
| `choose_basis_vectors` (`index_g1`, `index_g2`, `index_origin`, `subpixel`, `upsample_factor`, `sigma`, intensity and spacing thresholds) | `CuboStrain` | absent (planned: `CuboStrain`) |
| `set_hkl` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `set_max_peak_spacing` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `fit_basis_vectors` (`mask`, `max_peak_spacing`) | `CuboStrain` | absent (planned: `CuboStrain`) |
| `get_strain` (`gvects`, `coordinate_rotation`) | `CuboStrain` | absent (planned: `CuboStrain`) |
| `get_reference_g1g2` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `latticevectors.index_bragg_directions` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `latticevectors.add_indices_to_braggvectors` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `latticevectors.fit_lattice_vectors` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `latticevectors.fit_lattice_vectors_all_DPs` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `latticevectors.get_reference_g1g2`, `get_strain_from_reference_g1g2` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `latticevectors.get_rotated_strain_map` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `BraggVectors.to_strainmap` | `CuboStrain` | absent (planned: `CuboStrain`) |
| `show_strain`, `show_reference_directions`, `show_lattice_vectors`, `show_bragg_indexing` | — | n/a (plotting) |

## Bragg vectors and disk detection (braggvectors/)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `find_Bragg_disks(DataCube, template)` | — | absent (planned: a module decision at its registration: here or DiffractoKit's geometry) |
| `find_Bragg_disks` on a 2D array, a 3D stack, or `(DataCube, rx, ry)` | — | absent (planned: with `find_Bragg_disks`) |
| `find_Bragg_disks(corrPower, sigma, sigma_dp, sigma_cc, filter_function)` | — | absent (planned: with `find_Bragg_disks`) |
| `find_Bragg_disks(radial_bksb)` | — | absent (planned: with `get_radial_bksb_dp`) |
| `find_Bragg_disks(subpixel="none"/"poly"/"multicorr", upsample_factor)` | — | absent (planned: with `find_Bragg_disks`) |
| `find_Bragg_disks(minAbsoluteIntensity, minRelativeIntensity, relativeToPeak, minPeakSpacing, edgeBoundary, maxNumPeaks)` | — | absent (planned: with `find_Bragg_disks`) |
| `find_Bragg_disks(CUDA, CUDA_batched)` | — | absent (planned: a Metal path, later lane) |
| `find_Bragg_disks(distributed, ML, ml_*)`, the `diskdetection_parallel`, `_aiml`, `_cuda` variants | — | n/a (Python parallel tooling, TensorFlow) |
| `BraggVectors` (`raw`, `cal`, `setcal`, `calibrate`, `get_vectors`, `shape`, `qx`, `qy`, `I`) | — | absent (planned: with `find_Bragg_disks`) |
| `BraggVectors.get_virtual_image` | — | absent (planned: with `BraggVectors`) |
| `BraggVectors.histogram` | — | absent (planned: with `BraggVectors`) |
| `BraggVectors.measure_origin` | — | absent (planned: with `BraggVectors`) |
| `BraggVectors.measure_origin_beamstop` | — | absent (planned: with `BraggVectors`) |
| `BraggVectors.fit_origin` | — | absent (planned: with `BraggVectors`) |
| `BraggVectors.fit_p_ellipse` | — | absent (planned: with `BraggVectors`) |
| `BraggVectors.mask_in_Q`, `get_masked_peaks`, `mask_in_R` | — | absent (planned: with `BraggVectors`) |
| `BraggVectorMap` | — | absent (planned: with `BraggVectors`) |
| `threshold_Braggpeaks`, `universal_threshold`, `get_pointlistarray_intensities` | — | absent (planned: with `BraggVectors`) |
| `BraggVectors.plot`, `copy`, `to_h5` | — | n/a (plotting; value types; FormatoKit) |
| `Crystal`, `orientation_plan`, `match_orientations` | — | DiffractoKit |

## Parallax (process/phase/parallax.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `Parallax(datacube, energy, ...)` | `CuboPtycho` | absent (planned: `CuboPtycho`, placeholder) |
| `Parallax.preprocess` (`edge_blend`, `dp_mask`, `threshold_intensity`, descan correction, `realspace_mask`) | `ParallaxPreprocessing.prepare` | partial: constant normalization, selection/mask, edge window and padding; no descan/polynomial normalization/real-space mask. Default windowed retention follows reference; `.unwindowed` follows app (independent source transcription pin). |
| `Parallax.reconstruct` (alignment bins, upsampling, regularisation) | `CuboParallax.ParallaxAlignment` | partial: CPU Double cross-correlation, matrix-DFT refinement (default factor 8), periodic shifts and fixed/running-reference stacks; no grouping, schedules, regularisation or full reconstruction (`docs/parallax-alignment.md`) |
| `Parallax.guess_common_aberrations` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `Parallax.subpixel_alignment` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `Parallax.aberration_fit` | `CuboParallax.ParallaxAberrationFitter` | low order (C1, C12a/b, rotation, transpose) and higher-order global/recursive fits; reference basis default, app basis (`.selectedCentroid`) as option; tolerances in `AberrationTests` |
| `Parallax.aberration_correct` | `CuboParallax.ParallaxAberrationCorrection` | CTF sign/phase correction; reference ignores `q_highpass` (default), `.apply` is the app behaviour |
| `Parallax.depth_section` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `Parallax.object_cropped` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `show_shifts`, `show_probe_position_shifts`, `visualize` | — | n/a (plotting) |

`CuboParallax` also exposes the alignment utilities used within the partial reconstruction row:

| py4DSTEM utility | CuboKit | State |
|---|---|---|
| `align_images_fourier`, `upsampled_correlation`, `dftUpsample` | `ParallaxAlignment.shift(reference:moving:rows:columns:upsampleFactor:)` | done for finite, nonconstant real images; CPU Double, factors 3...64; default factor 8; `1e-10` normwise relative tolerance with unit floor |
| `get_shifted_ar(periodic=True, bilinear=False)` | `ParallaxAlignment.shifted(_:rows:columns:by:)` | done for finite real images at least 3 × 3; CPU Double; `1e-10` normwise relative tolerance with unit floor |

## Ptychography (process/phase/)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `SingleslicePtychography` (`preprocess`, `reconstruct`) | `CuboPtycho` | partial: CPU single-slice gradient descent from preprocessed amplitudes; scope below |
| `MultislicePtychography` (`preprocess`, `reconstruct`) | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `MixedstatePtychography` (`preprocess`, `reconstruct`) | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `MixedstateMultislicePtychography` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `DirectPtychography`: `SSB` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `DirectPtychography`: `OBF` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `DirectPtychography`: `WDD` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `MagneticPtychography` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `PtychographicTomography`, `MagneticPtychographicTomography` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `XRayMagneticPtychography` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| `PtychographyOptimizer`, `OptimizationParameter` | `CuboPtycho` | absent (planned: `CuboPtycho`) |
| the constraint and method mixins (`ObjectNDConstraintsMixin`, `ProbeMethodsMixin`, ...), `ComplexProbe`, `AffineTransform` | — | n/a (internals; their features count under the classes above) |

## Count (slices 1 to 4 and initial parallax alignment)

Rows in CuboKit's scope only (the n/a, app, DiffractoKit and FormatoKit rows excluded). One row is one method or option group.

| Area | Rows | done | partial | absent |
|---|---|---|---|---|
| DataCube: container and shape | 6 | 4 | 1 | 1 |
| DataCube: virtual images | 9 | 7 | 1 | 1 |
| DataCube: virtual diffraction | 10 | 9 | 0 | 1 |
| DataCube: calibration-bound properties and coordinates | 5 | 0 | 1 | 4 |
| DataCube: reshaping and pre-processing | 13 | 1 | 0 | 12 |
| DataCube: probe, beam stop and background methods | 10 | 5 | 0 | 5 |
| Calibration entries | 21 | 0 | 12 | 9 |
| Calibration processes | 23 | 11 | 3 | 9 |
| Probe | 14 | 12 | 1 | 1 |
| DPC and the phase base class's preprocessing | 20 | 13 | 4 | 3 |
| Strain | 14 | 0 | 0 | 14 |
| Bragg vectors and disk detection | 17 | 0 | 0 | 17 |
| Parallax | 9 | 0 | 2 | 7 |
| Ptychography | 11 | 0 | 1 | 10 |
| **Total** | **182** | **62** | **26** | **94** |

Of the 182 rows in scope: done 62 (of which the slice-4 rows are marked), partial 26, absent 94.
Of the done rows, 25 are marked slice 4, 36 are slices 1 to 3, and one is hot-pixel filtering. The previous file lumped methods and counted 28 rows.

## 0.2.0 notes (2026-10-15)

- `RotationSolver` gains `curve(_:anglesDegrees:criterion:transpose:isCancelled:)` (py4DSTEM computes the same curve inside
  `DPC.preprocess`) and runs the grid in parallel; the CPU centre of mass runs positions in parallel; both bit identical.
- The Metal centre of mass agrees with the CPU path where |S| ≥ 1e-3 · Σ|wP| (slice 2): Float32's absolute noise floor on
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
pin means, masks and every cleaned sample bit for bit. Measured gap 0; tolerance 0.
Supported input and numerical details: `docs/hot-pixels.md`.

## CPU ptychography details (refines the headline above; not counted twice)

| py4DSTEM f050d207 | CuboKit | State |
|---|---|---|
| `ComplexProbe.build` | `PtychographyProbe.build` | partial: defocus, C12, C30, hard/cosine aperture; no coherence envelopes |
| `SingleslicePtychography.reconstruct` gradient descent | `SingleslicePtychography.reconstruct` | partial: preprocessed amplitudes, complex object, fixed positions, full batches, probe updates and threshold/pure-phase constraints; other constraints disabled |

Pins: `tools/ptycho/pins.py`, `Tests/CuboPtychoTests/Fixtures/ptycho.json`.
Tolerance `2e-6` normwise maximum error with unit floor; see DocC Scope for numerical limits.

Preprocessing pins: `tools/parallax/preprocess_pins.py`, four cases, approximately 114 KB.
Normwise maximum tolerance `2e-6` with unit floor (Double port vs Float32 storage);
measured maximum gap `1.34e-7`. App unwindowed stack independently transcribed from
`ParallaxPreprocessing.swift` at the app source pin; no app types are included.
