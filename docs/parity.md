# Parity with py4DSTEM

A checklist against py4DSTEM's documented `DataCube` API and the processing modules CuboKit's scope covers, at the pinned
commit `f050d207` (0.14.19), so "complete" is a measured claim (suite ROADMAP, "Kits", phase 4). Updated at every tag.
Crystallography (`Crystal`, orientation, ACOM) is DiffractoKit's; file formats (`read`, `import_file`, `save`) are
FormatoKit's; simulation is SimulatoKit's.

Legend: **done** (pinned against py4DSTEM and planted truth, `docs/registration-slice-1.md`), **partial** (what is missing
says so), **planned** (a slice is named), **app** (lives in mac4DSTEM, not in a kit), **n/a** (plotting or Python-only).

## `DataCube` (py4DSTEM/datacube/datacube.py, virtualimage.py, virtualdiffraction.py)

| py4DSTEM | CuboKit | State |
|---|---|---|
| `DataCube(data)`, `.data` | `MemoryCube(shape:values:)`, `MappedCube(npy:)` (lazy, memory-mapped), `CubeSource` | done (Float32 on read; ten element types) |
| `R_Nx`, `R_Ny`, `Q_Nx`, `Q_Ny`, `Rshape`, `Qshape`, `R_N` | `CubeShape.scanRows`, `scanColumns`, `detectorRows`, `detectorColumns`, `scanCount`, `detectorCount` | done |
| `datacube[rx, ry]` | `readPattern(row:column:)` | done |
| `bin_Q_mmap`, dask-backed lazy data | `MappedCube` | partial: lazy and memory-mapped; no bin on read |
| `get_virtual_image(mode, geometry)`: `point`, `circle`, `annulus`, `rectangle`, `mask` | `VirtualImaging.image(of:geometry:)` | done (CPU and Metal) |
| `get_virtual_image(..., return_mask=True)`, `make_detector` | `DetectorMask.make` | done |
| `get_virtual_image(centered=True)`, `calibrated=True` | — | planned (needs the calibration model, slice 2 or 3) |
| `get_virtual_image(shift_center=True, subpixel=...)` | — | planned (needs per-position origins, slice 2) |
| `get_dp_mean`, `get_dp_max` | `VirtualDiffraction.mean`, `max`, `meanAndMax` | done |
| `get_dp_median` | — | planned |
| `get_virtual_diffraction(method, mask=...)` (a real-space region) | — | planned; mac4DSTEM's `tiledDiffraction` is the seed (app today) |
| `get_virtual_diffraction(shift_center=True)` | — | planned (slice 2, with the origins) |
| `position_detector`, `show` | — | n/a (plotting) |
| `calibration`: `Q_pixel_size`, `R_pixel_size`, units, origin, ellipse, rotation | — | planned (a calibration model; ellipse and Q scale are DiffractoKit's geometry) |
| `set_scan_shape`, `swap_RQ`, `swap_Rxy`, `swap_Qxy` | — | planned |
| `crop_Q`, `crop_R`, `bin_Q`, `bin_R`, `thin_R`, `pad_Q`, `resample_Q` | — | planned; mac4DSTEM's `LoadSpecification` does crop and bin at read time (app today) |
| `filter_hot_pixels`, `median_filter_masked_pixels` | — | planned; mac4DSTEM's `HotPixelFilter` is the seed |
| `get_probe_size` | `ProbeSize.estimate` | done (slice 2; bit for bit) |
| `get_vacuum_probe` | — | planned |
| `get_beamstop_mask`, `get_radial_bkgrnd`, `get_radial_bksb_dp`, `get_local_ave_dp`, `get_braggmask` | — | planned |
| `find_Bragg_disks` | — | planned (a module decision at its registration: here or DiffractoKit's geometry) |
| `copy`, `add`, the emdfile tree | — | n/a |

## Processing beyond `DataCube`

| py4DSTEM | CuboKit | State |
|---|---|---|
| centre of mass (`process/phase` DPC preprocessing: `_calculate_intensities_center_of_mass`, `fit_origin`) | `CenterOfMass.measure`, `CenterOfMassReference.normalize` | done (slice 2; CPU and Metal) |
| per-position origins (`get_origin`, `get_origin_friedel`) | — | planned (a calibration slice) |
| `DPC` (`preprocess`: curl or divergence rotation and transpose; `reconstruct`: the iterative phase, no filters) | `RotationSolver.solve`, `DPC.reconstruct` | done (slice 2; CPU) |
| `DPC.reconstruct` constraints (Gaussian, Butterworth), a Metal DPC | — | planned |
| `StrainMap` (`choose_basis_vectors`, `fit_basis_vectors`, `get_strain`) | `CuboStrain` | planned (placeholder) |
| `Parallax` (tilt-corrected bright field, aberration fit) | `CuboPtycho` | planned (placeholder) |
| `SingleslicePtychography`, `MultislicePtychography`, `MixedstatePtychography` | `CuboPtycho` | planned (placeholder) |
| `Crystal`, `orientation_plan`, `match_orientations` | — | DiffractoKit |
| `read`, `import_file`, `save` | `NPYFile` (the `.npy` exchange only) | FormatoKit |

## Count (2026-10-15, slices 1 and 2)

Of the 27 rows above in CuboKit's scope (the n/a, DiffractoKit and FormatoKit rows excluded): done 9, partial 1,
planned 17.
