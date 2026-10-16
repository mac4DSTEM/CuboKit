# Reciprocal lattice strain

`LatticeStrain` uses row = py4DSTEM x and column = y, in a calibrated,
centered detector frame supplied by the caller. No calibration state, UI,
progress, tiling or provenance object is required. All public values are Sendable.

Primary path, pinned to py4DSTEM 0.14.19 (`f050d207`):

1. `chooseBasis(maxima:origin:)` receives intensity-ordered histogram maxima,
   including the direct beam. The nearest maximum supplies the origin, the next
   nearest g1, and g2 minimizes distance times (cos²(angle difference) + 0.1).
   Histogram construction and peak extraction are separate from this module.
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

Their source is the read-only app commit `c809679c8ee23a355da6af4f63665aaf64f90d0d`,
`Core/Analysis/StrainMapping.swift`. Only numeric helpers are retained; debug
printing is removed. The generator independently compiles the **original** helper
text from that commit, removing only access/isolation qualifiers and supplying
minimal value shims. It never imports or executes CuboStrain to generate pins.

Regenerate from a suite worktree:

```sh
../../tools/memcap.sh -- ../../References/CuboKit/py4dstem-venv/bin/python -I -B tools/strain/pins.py
```

The generator checks Python 3.12.4, NumPy 2.5.0, py4DSTEM 0.14.19, the reference
commit and installed source identity. Inputs include exact planted lattices,
unequal-intensity noise that distinguishes weighting, a reference population
that distinguishes MAD rejection, repeated vector clusters, and peaks measured
on public `small_datacube.dm4` crop [0,0,200:312,200:312]. Its input SHA256 and
source SHA256 are in `Fixtures/strain.json`. Numeric arrays are little-endian
binary f64, under 5 KB total. Oracle execution under the 8 GB memcap peaked at
689 MB. The Swift source oracle is also a descendant of that capped process.

Absolute tolerance 1e-10 for primary fits/strains (no bit-exact claim across SVD
implementations); measured maximum fit gap 5.33e-15, strain gap 8.88e-16 on the
pins. Compatibility helpers match their original-source pins exactly (tolerance
0). These pins establish reference agreement and recover planted strain; they
are not microscope validation or performance measurements.
