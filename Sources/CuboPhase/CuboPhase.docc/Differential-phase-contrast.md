# Differential phase contrast

Differential phase contrast (DPC) turns the centre of mass of every pattern into a phase image in three steps. Each step
is a separate call, so a field measured by another method can enter at any of them.

1. ``CenterOfMassReference/normalize(_:reference:reciprocalSampling:)`` subtracts a reference from the measured field.
   The reference is a least-squares plane (`.planeFit`, py4DSTEM's default), the mean (`.constant`), nothing (`.none`),
   or per-position origins measured elsewhere (`.given`). The result is the fitted reference and the normalised shift,
   with NaN set to zero.
2. ``RotationSolver/solve(_:anglesDegrees:criterion:forceRotationDegrees:forceTranspose:)`` finds the rotation between
   scan and detector, and whether the components are transposed. By default it minimises the curl of the field over
   angles from -89 to 89 degrees in steps of 1. The result carries the corrected field. The scan needs at least three
   positions in each direction.
3. ``DPC/reconstruct(_:scanSampling:paddingFactor:maxIterations:stepSize:stoppingCriterion:backtrack:)`` runs
   py4DSTEM's iterative reconstruction on the corrected field. It returns the phase on the scan grid and the error of
   each accepted iteration.

The measured field is a ``ScanVectorField``: one row component and one column component per position, in detector
pixels. The centre of mass from `CenterOfMass` in CuboVirtual gives these values.

```swift
import CuboCore
import CuboPhase
import CuboVirtual
import Foundation

let cube = try MappedCube(npy: URL(fileURLWithPath: "cube.npy"))
let com = try await CenterOfMass.measure(of: cube)
let measured = try ScanVectorField(scanRows: com.row.scanRows, scanColumns: com.row.scanColumns,
                                   row: com.row.values, column: com.column.values)
let normalized = try CenterOfMassReference.normalize(measured, reference: .planeFit).normalized
let rotation = try RotationSolver.solve(normalized)                     // degrees, transposed, corrected field
let phase = try DPC.reconstruct(rotation.corrected)                     // phase on the scan grid
```

The reference, the rotation and the reconstruction are computed in Double. Reconstruction runs on the CPU only, with
Accelerate's double-precision FFT. On five real cubes, the phase agrees within 3.3e-12 of its maximum with the float64
evaluation of the same steps. py4DSTEM computes the same steps in float32, and that difference is recorded, not asserted.

Agreement with py4DSTEM shows that the port follows its definition. It does not show that the phase is the specimen's
projected potential. That needs a calibrated rotation and a known specimen.
