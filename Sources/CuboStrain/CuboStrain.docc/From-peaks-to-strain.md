# From peaks to strain

Strain comes from how far a lattice has moved from a reference, measured in reciprocal space. A lattice is two vectors,
`g1` and `g2`, as a ``LatticeBasis`` in detector pixels from the lattice origin. A reference basis and a local basis give a
``StrainTensor``: `exx` along the first (row) axis, `eyy` along the second, the shear `exy`, and the rotation `theta` in
radians.

At the smallest scale, ``LatticeStrain/strain(local:reference:coordinateRotation:)`` compares two bases.

```swift
import CuboStrain

let reference = LatticeBasis(g1: LatticeVector(row: 20, column: 0), g2: LatticeVector(row: 0, column: 20))
let stretched = LatticeBasis(g1: LatticeVector(row: 19.8, column: 0), g2: LatticeVector(row: 0, column: 20))
let tensor = try LatticeStrain.strain(local: stretched, reference: reference)
print(tensor.exx, tensor.eyy, tensor.exy, tensor.theta)
```

For a pattern, choose the basis from the maxima of the Bragg vector map with
``LatticeStrain/chooseBasis(maxima:origin:)``, index the peaks with ``LatticeStrain/index(peaks:directions:maximumSpacing:)``
and fit them with ``LatticeStrain/fit(_:origin:minimumPeaks:)``. A fit returns a ``LatticeFit`` with the basis and a residual.
Take the reference from a set of fitted bases with ``LatticeStrain/reference(_:)``.

## A whole scan

``StrainMapping/compute(peaks:scanRows:scanColumns:detectorRows:detectorColumns:origin:directions:basisOptions:options:)``
runs those steps for every scan position, the way py4DSTEM's `StrainMap` does. It needs the peaks of every position as
``StrainPeak`` values centred on the lattice origin, which `StrainMapping.centredPeaks(_:origin:)` makes from the
``BraggVectors`` of CuboBragg. The ``StrainMapOptions`` take `maximumPeakSpacing`, the farthest a peak may be from a
direction and still be indexed, and the reference: ``StrainReference/fittedMedian``, a ``StrainReference/region(_:)`` of the
scan, or a ``StrainReference/supplied(_:)`` basis.

The ``StrainMapResult`` has flat row-major maps `exx`, `eyy`, `exy` and `theta` of `scanRows * scanColumns` values, NaN where
a position could not be fitted, and `valid`, `indexedPeakCount` and `residual` for each position. A value from a position
that is not `valid` is not a measurement.

Agreement with py4DSTEM shows that the port follows its definition. It does not show that a strain map is the specimen's
strain. That needs a calibrated detector and a specimen with a known lattice.
