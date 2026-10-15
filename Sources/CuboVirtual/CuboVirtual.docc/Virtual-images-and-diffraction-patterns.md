# Virtual images and diffraction patterns

A virtual image applies a detector mask to every pattern: for each scan position, it sums the pattern times the mask
over the detector. The masks are py4DSTEM's `make_detector` shapes, in detector pixels counted from the corner:
`.point` for one pixel, `.circle` for a bright-field disk, `.annulus` for a dark-field ring, `.rectangle`, and `.mask`
for any weights on the detector grid. ``DetectorGeometry`` holds them, and ``DetectorMask`` builds the grid.

``VirtualDiffraction`` gives the mean and maximum pattern, the median, and a pattern over chosen scan positions through
``ScanSelection``: `.all`, `.mask` for the positions where a Boolean is true, or `.weights`, where the mean is
Σ P·w / Σ w and the maximum and median are taken over P·w. ``CenterOfMass`` returns the intensity-weighted centre of mass of every pattern,
in detector pixels, with the intensity sum. A pattern with zero total intensity gives NaN. Negative intensities are summed
as they are. ``OriginMeasurement`` and ``FriedelOrigin`` find an origin per position.

```swift
import CuboCore
import CuboVirtual
import Foundation

let cube = try MappedCube(npy: URL(fileURLWithPath: "cube.npy"))
let mean = try await VirtualDiffraction.mean(of: cube)                  // the mean pattern, on the CPU
let probe = try ProbeSize.estimate(mean)                                // radius and centre of the probe
let bright = try await VirtualImaging.image(of: cube,
    geometry: .circle(centerRow: probe.centerRow, centerColumn: probe.centerColumn, radius: probe.radius))
let annular = try await VirtualImaging.image(of: cube,
    geometry: .annulus(centerRow: probe.centerRow, centerColumn: probe.centerColumn, inner: 18, outer: 30),
    path: .metal)
let com = try await CenterOfMass.measure(of: cube)                      // com.row and com.column, in detector pixels
```

Results are Double, whatever the cube's type. ``ScanImage`` holds one value per scan position, and ``DetectorImage``
one per detector pixel. Each result records the ``ComputePath`` that made it.

The CPU path is the reference. It uses Accelerate and accumulates in Double. The Metal path runs compute kernels with
Float32 accumulation and combines the tiles in Double. Nothing chooses a path for you: a number depends on the path only
when the caller asks for `.metal`, and the result says which path ran. Metal virtual images and mean patterns agree with
the CPU path to a relative 1e-5, a tolerance fixed before the code was written. The maximum pattern is bit for bit on
both paths. The median, the selected patterns and the origins run on the CPU only. A `.metal` request on a machine without a Metal
device throws; ``VirtualDiffraction/metalIsAvailable`` says in advance.
