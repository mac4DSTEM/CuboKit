// What mac4DSTEM (and any other app) reads from CuboKit, through the public API only. Each check prints one line; the
// summary line is what run.sh reads. A cube is written to a temporary .npy, read back lazily and eagerly, and reduced on
// both paths.
import CuboCore
import CuboPtycho
import CuboStrain
import CuboVirtual
import Foundation

var checks = 0
var failures = 0
@MainActor func check(_ name: String, _ ok: Bool) {
    checks += 1
    if !ok { failures += 1 }
    print("\(ok ? "ok  " : "FAIL") \(name)")
}

let shape = try CubeShape(scanRows: 3, scanColumns: 4, detectorRows: 8, detectorColumns: 9)
check("CubeShape counts", shape.scanCount == 12 && shape.detectorCount == 72 && shape.elementCount == 864)
check("CubeShape extents", shape.extents == [3, 4, 8, 9])
let values = (0..<shape.elementCount).map { Float($0 % 97) - 10 }
let url = FileManager.default.temporaryDirectory.appendingPathComponent("cubokit-consumer-\(UUID().uuidString).npy")
defer { try? FileManager.default.removeItem(at: url) }
try NPYFile.write(values, shape: shape.extents, to: url)
let header = try NPYFile.header(of: url)
check("NPYFile.header", header.elementType == .float32 && header.shape == shape.extents && header.dataOffset % 64 == 0)
let lazy = try MappedCube(npy: url)
check("MappedCube shape and type", lazy.shape == shape && lazy.elementType == .float32)
let eager = try await MemoryCube(loading: lazy, tiling: .rows(1))
check("MemoryCube(loading:) equals the file", eager.values == values)
let pattern = try await lazy.readPattern(row: 2, column: 3)
check("readPattern", pattern.values == Array(values[(11 * 72)..<(12 * 72)]) && pattern[7, 8] == values[11 * 72 + 71])
let tile = try await eager.readScanTile(rows: 1..<3)
check("readScanTile", tile.positionCount == 8 && tile.values.count == 576)
check("ScanTiling.automatic", try ScanTiling.automatic.rowsPerTile(for: shape) == 3)

let geometries: [DetectorGeometry] = [
    .circle(centerRow: 3.5, centerColumn: 4, radius: 2.5),
    .annulus(centerRow: 3.5, centerColumn: 4, inner: 1, outer: 3.6),
    .rectangle(rowMin: 1, rowMax: 4, columnMin: 2, columnMax: 7),
    .point(row: 7, column: 8),
    .mask((0..<72).map { Float($0 % 3) - 1 }),
]
let paths: [ComputePath] = VirtualDiffraction.metalIsAvailable ? [.cpu, .metal] : [.cpu]
for geometry in geometries {
    let mask = try DetectorMask.make(geometry, shape: shape)
    var want = [Double]()
    for s in 0..<shape.scanCount {
        want.append((0..<72).reduce(0.0) { $0 + Double(values[s * 72 + $1]) * Double(mask[$1]) })
    }
    for path in paths {
        let image = try await VirtualImaging.image(of: lazy, geometry: geometry, path: path, tiling: .automatic)
        check("VirtualImaging.image \(geometry) \(path)", image.path == path && image.values == want
              && image.scanRows == 3 && image.scanColumns == 4)
    }
}
for path in paths {
    let (mean, max) = try await VirtualDiffraction.meanAndMax(of: eager, path: path)
    let m = (0..<72).map { k in (0..<12).reduce(0.0) { $0 + Double(values[$1 * 72 + k]) } / 12 }
    let x = (0..<72).map { k in (0..<12).map { Double(values[$0 * 72 + k]) }.max()! }
    check("VirtualDiffraction.meanAndMax \(path)", mean.values == m && max.values == x && mean[0, 0] == m[0])
    let onlyMean = try await VirtualDiffraction.mean(of: lazy, path: path)
    let onlyMax = try await VirtualDiffraction.max(of: lazy, path: path)
    check("VirtualDiffraction.mean / max \(path)", onlyMean.values == m && onlyMax.values == x)
}
do {
    _ = try DetectorMask.make(.annulus(centerRow: 0, centerColumn: 0, inner: 2, outer: 1), shape: shape)
    check("CuboVirtualError.innerNotBelowOuter", false)
} catch let error as CuboVirtualError {
    check("CuboVirtualError.innerNotBelowOuter", error == .innerNotBelowOuter(inner: 2, outer: 1)
          && error.errorDescription?.contains("mac4DSTEM") == false)
}
do {
    _ = try CubeShape([1, 2, 3])
    check("CuboError.invalidShape", false)
} catch let error as CuboError {
    check("CuboError.invalidShape", error == .invalidShape([1, 2, 3]))
}
check("CuboStrain scope", CuboStrain.scope.contains("not registered"))
check("CuboPtycho scope", CuboPtycho.scope.contains("not registered"))
check("ElementType list", ElementType.allCases.count == 10 && ElementType.uint16.npyDescriptor == "<u2")
print("consumer-check summary: \(checks) checks, \(failures) failures")
exit(failures == 0 ? 0 : 1)
