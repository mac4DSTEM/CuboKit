// The Swift code of README.md's "The first ten minutes" block, copied verbatim, and nothing else of the kit's API
// beyond writing the input file it opens. run.sh checks that every line of every ```swift block in README.md appears
// here, so the README and this file cannot drift apart: change the README, and run.sh fails until this file follows.
// run.sh runs the binary in an empty temporary directory, so "cube.npy" below is the file written here.
import CuboCore
import CuboVirtual
import Foundation

// Input only: the cube.npy that the README's Python line would have exported. A 3 x 4 scan of 64 x 64 patterns, each a
// bright disk of radius 12 centred on pixel (31, 32) over a flat background.
let scanRows: Int = 3
let scanColumns: Int = 4
let detector: Int = 64
var inputValues: [Float] = []
inputValues.reserveCapacity(scanRows * scanColumns * detector * detector)
for scan: Int in 0..<(scanRows * scanColumns) {
    for row: Int in 0..<detector {
        for column: Int in 0..<detector {
            let dr: Float = Float(row - 31)
            let dc: Float = Float(column - 32)
            let inside: Bool = dr * dr + dc * dc <= 144
            inputValues.append(inside ? 100 + Float(scan) : 1)
        }
    }
}
try NPYFile.write(inputValues, shape: [scanRows, scanColumns, detector, detector], to: URL(fileURLWithPath: "cube.npy"))

// The README's block, verbatim.
// Lazy: nothing is read until a pass needs it.
let cube = try MappedCube(npy: URL(fileURLWithPath: "cube.npy"))
let mean = try await VirtualDiffraction.mean(of: cube)                  // get_dp_mean, in Double
let probe = try ProbeSize.estimate(mean)                                // get_probe_size: radius, centre
let bf = try await VirtualImaging.image(of: cube, geometry:
    .circle(centerRow: probe.centerRow, centerColumn: probe.centerColumn, radius: probe.radius))
let com = try await CenterOfMass.measure(of: cube)                      // com.row, com.column: detector pixels
let adf = try await VirtualImaging.image(of: cube,
    geometry: .annulus(centerRow: 31, centerColumn: 32, inner: 18, outer: 30), path: .metal)

import CuboParallax

// Three asymmetric features in a small row-major reference image.
let alignmentRows = 8, alignmentColumns = 10
var alignmentReference = [Double](repeating: 0, count: alignmentRows * alignmentColumns)
alignmentReference[13] = 1
alignmentReference[36] = 0.7
alignmentReference[62] = 0.3
let moving = try ParallaxAlignment.shifted(alignmentReference,
    rows: alignmentRows, columns: alignmentColumns, by: .init(row: 2, column: -3))
let alignment = try ParallaxAlignment.align([moving], reference: alignmentReference,
    rows: alignmentRows, columns: alignmentColumns)
print(alignment.shifts[0])  // translation applied to moving; alignment.images and alignment.mean are row-major

print("readme-consumer summary: 2 swift blocks from README.md compiled and ran")
