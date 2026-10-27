# Reciprocal lattice strain

`LatticeStrain` uses row = py4DSTEM x and column = y, in a calibrated,
centered detector frame supplied by the caller. No calibration state, UI,
progress, tiling or provenance object is required. All public values are Sendable.

Primary path, pinned to py4DSTEM 0.14.19 (`f050d207`):

1. `chooseBasis(maxima:origin:)` receives intensity-ordered histogram maxima,
   including the direct beam. The nearest maximum supplies the origin, the next
   nearest g1, and g2 minimizes distance times (cos²(angle difference) + 0.1).
   The overload with `indexOrigin`, `indexG1` and `indexG2` takes explicit indices as
   `choose_basis_vectors` does (a given index is used as is; the others are guessed).
   Histogram construction and peak extraction are in `StrainMapping` (below).
2. `index(directions:basis:)` gives histogram directions integer indices, rounding
   ties to even. `index(peaks:directions:maximumSpacing:)` assigns measured peaks
   to the nearest measured direction, retaining the first on ties, and rejecting
   distances greater than the inclusive limit. This follows StrainMap's measured
   direction matching rather than rounding every measured peak independently.
3. `fit` fits an intercept and two vectors with design [1,h,k], multiplying both
   design and data by intensity: the objective weights intensity **squared**.
   Modern Accelerate LAPACK SVD computes the minimum-norm solution with NumPy's
   rank cutoff max(rows,columns)*epsilon. Residual is intensity-weighted mean
   Euclidean distance, not RMS. Fewer than minimumPeaks (default 5) returns nil.
4. `reference` computes plain component medians of explicitly selected valid
   fits. The caller applies scan and ROI masks before passing the selected bases.
5. `strain` computes beta = lstsq(reference rows, local rows).transpose;
   exx=1-beta00, eyy=1-beta11, exy=-(beta01+beta10)/2,
   theta=(beta01-beta10)/2. Coordinate rotation is counterclockwise in degrees;
   theta remains in radians with py4DSTEM's sign.

The three compatibility options are independent:

- `fitIntensityOnce`: Float32 inputs/output, Double normal equations weighted by
  intensity once, weights floored at 1e-6, RMS residual. This exposes the original
  fit helper; the app's subsequent local robust refit/orchestration is not implied.
- `referenceMAD`: Float32 component medians followed by rejection when the sum
  of the two vector distances exceeds median distance + max(3*1.4826*MAD,
  0.02*mean basis length). Retain all candidates if fewer than half survive.
- `chooseBasisConsensus`: Float32 repeated-vector spatial clustering within
  max(0.5, 18% of median nearest radius), minimum cluster support 8% of occupied
  positions, at most 24 candidates, condition number at most 8, at least 50%
  observation support. Pair ties prefer lower residual, then shorter vectors.
  Minimum noncentral radius defaults to 2. Empty/no-consensus inputs return nil.

Result-returning variants (round 6, from the app's strain switch; the plain calls are unchanged and now delegate to them):

- `chooseBasisConsensusResult(_:origin:minimumRadius:isCancelled:)` returns `LatticeStrain.ConsensusBasis`: the `basis`
  (identical to `chooseBasisConsensus`), `observationCount` and `supportCount` (exact integers), and `residual` (RMS
  distance of the supporting observations to their node), `conditionNumber` and `indexingTolerance` (the Float32
  quantities the search computed, widened to Double). `isCancelled` is an optional `@Sendable () -> Bool` polled while
  clustering and before each candidate pair; true throws `CancellationError`. Measured against a test-local Double
  rescoring on a planted 60-position lattice: counts equal, residual/condition/tolerance gaps 1.4e-7, 3.7e-7, 8.1e-8
  (written tolerance 1e-5, Float32).
- `referenceMADResult(_:)` returns `LatticeStrain.ReferenceMAD`: the `reference` (identical to `referenceMAD`) and the
  ascending `inlierIndices` into the input bases (`inlierCount`); all indices when the half-population fallback applies.
- `StrainMapResult.referenceInlierCount`: positions the reference is the median of (equals `referencePositionCount`
  for the median method, the MAD-retained count for `.madRejection`, 0 for a supplied reference).

Their source is the read-only app commit `c809679c8ee23a355da6af4f63665aaf64f90d0d`,
`Core/Analysis/StrainMapping.swift`. Only numeric helpers are retained; debug
printing is removed. The generator independently compiles the **original** helper
text from that commit, removing only access/isolation qualifiers and supplying
minimal value shims. It never imports or executes CuboStrain to generate pins.

Regenerate from a suite worktree:

```sh
../../References/python/py4dstem/bin/python -I -B tools/strain/pins.py
```

The generator checks Python 3.12.4, NumPy 2.5.0, py4DSTEM 0.14.19, the reference
commit and installed source identity. Inputs include exact planted lattices,
unequal-intensity noise that distinguishes weighting, a reference population
that distinguishes MAD rejection, repeated vector clusters, and peaks measured
on public `small_datacube.dm4` crop [0,0,200:312,200:312]. Its input SHA256 and
source SHA256 are in `Fixtures/strain.json`. Numeric arrays are little-endian
binary f64, under 5 KB total. Oracle execution under an 8 GB memory cap peaked at
689 MB. The Swift source oracle is also a descendant of that capped process.

Absolute tolerance 1e-10 for primary fits/strains (no bit-exact claim across SVD
implementations); measured maximum fit gap 5.33e-15, strain gap 8.88e-16 on the
pins. Compatibility helpers match their original-source pins exactly (tolerance
0). These pins establish reference agreement and recover planted strain; they
are not microscope validation or performance measurements.

## Scan-level API: `StrainMapping`

`StrainMapping` (Sources/CuboStrain/StrainMapping.swift) composes the functions above into py4DSTEM's `StrainMap`
workflow. Everything is Double, CPU, public, Sendable; no calibration object, plotting or state. Peaks are plain
`StrainPeak(row, column, intensity)` values, **centred** (diffraction origin subtracted, in detector pixels), one list
per scan position, row-major (`position = row * scanColumns + column`; every output array uses the same order). This is
what `BraggVectors.get_vectors(center=True, pixel=False)` returns. `StrainMapping.centredPeaks(_:origin:)` converts a
`CuboBragg.BraggVectors` detection result. Calibrations beyond the origin (elliptical distortion, the R-Q rotation, a
pixel size) are applied by the caller before the peaks arrive.

1. `histogram(...)` is `BraggVectors.histogram(mode="cal")`: each peak goes to `(coordinate + origin) * sampling` and
   shares its intensity bilinearly between the four neighbouring cells, in py4DSTEM's order of operations and
   accumulation (four separate sums, added in that order); peaks whose floor or ceiling cells fall outside the grid are
   dropped; the grid is `round(detector * sampling)` with ties to even. Options: `sampling`, a boolean `scanMask`, or
   numeric `scanWeights` (positions with weight above `weightsThreshold` 0.005 are used, intensities scaled).
2. `chooseBasisVectors(histogram:options:)` is `choose_basis_vectors` without the figure: `get_maxima_2D` (through
   `CuboBragg.BraggMaxima`) at py4DSTEM's defaults (`StrainBasisOptions`: multicorr, upsample 16, sigma 0, no intensity
   cuts, `minSpacing` 0, `edgeBoundary` 1, `maxNumPeaks` 10), then `LatticeStrain.chooseBasis` with optional explicit
   indices, then `index(directions:basis:)` for the indexed directions. Equal intensities are ordered as NumPy orders
   them (`np.sort(order="intensity")[::-1]`: larger row first, then larger column), through the new
   `BraggDiskOptions.tieOrder = .numpy`; the Bragg disk detector keeps its raster order (`.rasterOrder`, the default).
   Mirrored peaks of a centrosymmetric pattern tie, so this decides which of +g and -g becomes g1.
   `StrainDirections.lattice(basis:order:)` builds the directions from a basis the caller already has.
3. `compute(...)` runs the histogram and basis choice (unless `directions:` is given), then for every position in the
   scan mask: nearest-direction indexing within `maximumPeakSpacing` (inclusive, first direction on ties), the
   weighted `LatticeStrain.fit` (`minimumPeaks` 5), then the reference and `LatticeStrain.strain` with the coordinate
   rotation (degrees, counterclockwise). `StrainReference`: `.fittedMedian` (`gvects=None`), `.region(mask)` (a boolean
   scan array, intersected with the fitted positions), `.supplied(basis)` (the tuple form). The result
   (`StrainMapResult`) has the maxima and directions used, the validity flags (scan mask and enough indexed peaks),
   per-position origin, g1, g2 and residual, the reference, and exx, eyy, exy, theta, all flat Double arrays with NaN at
   unfitted positions (py4DSTEM stores 0 in the fit maps and meaningless strain values there, flagged only by `mask`).
4. The mac4DSTEM variants stay the existing named options, now selectable per run: `fitMethod = .intensityOnce`,
   `referenceMethod = .madRejection`; the consensus basis (`chooseBasisConsensus`) is used by passing
   `StrainDirections.lattice(basis:)` as the directions. py4DSTEM is the default of every option.

Validation (`StrainMappingError.invalidInput`, `LatticeStrainError`): peak lists that do not match the scan, a mask,
weights or region of the wrong length or non-finite, non-positive sampling, a negative or non-finite
`maximumPeakSpacing`, `minimumPeaks` below 1, a non-finite rotation or peak, fewer than three maxima, an index out of
range, a degenerate basis (`singularBasis`), no fitted position for a median reference (`emptyReference`). Where py4DSTEM
would index out of range or return NaN, the kit refuses with the reason.

### Pins

`tools/strain/mapping_pins.py` (a sibling of `pins.py`, which is unchanged, as are its fixtures) runs py4DSTEM's real
`BraggVectors.histogram`, `StrainMap.choose_basis_vectors`, `fit_basis_vectors` and `get_strain` (Agg backend, the
plots discarded) and `get_maxima_2D`, and writes `Tests/CuboStrainTests/Fixtures/map_*.f64` and `strain_mapping.json`
(61 little-endian f64 arrays, 277 KB; the oracle peaked at 199 MB under an 8 GB memory cap). Datasets:

- d1: noise-free planted lattice (7 x 8 scan, 48 x 48 detector, fractional origin, non-orthogonal reference lattice,
  planted exx, eyy, exy, theta ramps, columns 0 to 2 exactly unstrained).
- d2: the same with 0.04 px position noise, intensity noise, 12 % of the peaks dropped, four positions with three
  peaks only, a boolean scan mask, a reference region, coordinate rotations 20 and -35 degrees.
- d3: peaks measured by `get_maxima_2D` on an 8 x 8 block of `simulation_4DSTEM` in the public `Au_sim.h5` (py4DSTEM
  test data; SHA256 in the JSON). That cube is a polycrystalline simulation, so these are real peaks with irregular
  lattices: it pins the algorithm on measured data, it is not a strain measurement.
- Runs r1 to r6 cover the default multicorr, parabolic and pixel maxima, sigma, minimum spacing and relative-intensity
  cuts, an explicit `index_g1`, median, region and supplied references, and rotation. Histogram cases: default,
  sampling 1.5, boolean mask, numeric weights. A tie array pins NumPy's order of equal maxima.

### Tolerances and measured gaps (tests print them)

| Quantity | Tolerance | Measured maximum gap |
|---|---|---|
| Bragg vector map (5 cases) | 1e-12 times the map maximum | 0 (identical) |
| Maxima (position, intensity), including multicorr | 1e-9 absolute | 2.3e-13 |
| Origin, g1, g2; directions' positions | 1e-9 | 1.4e-14 |
| Direction indices h, k; validity mask | exact | exact |
| Fits (origin, g1, g2, residual), reference | 1e-9 | 1.4e-14 |
| exx, eyy, exy, theta (rotated or not) | 1e-9 | 1.8e-15 |
| Planted truth recovered (d1, reference region = unstrained columns) | 1e-9 | 1.2e-15 |

The tolerance is not tuned to the gaps; the fit quantities are of order 1 to 100.

### Not provided

- Plotting and display (`show_strain`, `show_reference_directions`, `show_lattice_vectors`, `show_bragg_indexing`,
  `set_max_peak_spacing`'s figure), `set_hkl` and the hkl bookkeeping, the `StrainMap` object with its `calstate` check
  and `reset_calstate`, `histogram(mode="raw")` as a mode (pass the raw vectors with origin (0, 0) instead), and the
  `BraggVectors` calibrations (ellipse, pixel size, R-Q rotation; the caller centres and calibrates).
- With a pixel-calibrated `BraggVectors`, py4DSTEM's histogram is in detector pixels but its fit works in calibrated
  units; the kit takes everything in detector pixels, so that mixed-unit case is not reproduced.
- mac4DSTEM's other pipeline pieces beyond the named options: its inverse-matrix indexing gate with the +-8 index
  limit, the robust local refit, the cluster-consensus orchestration, the Float32 diagnostics.
- A position whose indexed intensities sum to zero is "not fitted" here; py4DSTEM stores NaN and fails later.
- The Bragg disk detector (`CuboBragg`) is a separate step; the strain input is its centred output.
