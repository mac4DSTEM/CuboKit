// What an app reads from CuboKit, through the public API only. Each check prints one line; the
// summary line is what run.sh reads. A cube is written to a temporary .npy, read back lazily and eagerly, and reduced on
// both paths.
import CuboCore
import CuboPhase
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
          && error.errorDescription?.contains("app") == false)
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
// Slice 2: centre of mass, probe size, the reference, the rotation, DPC.
for path in paths {
    let com = try await CenterOfMass.measure(of: lazy, mask: .circle(centerRow: 3.5, centerColumn: 4, radius: 3), path: path)
    check("CenterOfMass.measure \(path)", com.row.scanRows == 3 && com.row.values.count == 12 && com.row.path == path
          && com.intensity.values.count == 12 && com.momentRow.values.count == 12 && com.momentColumn.values.count == 12)
}
var disk = [Double](repeating: 1, count: 32 * 33)
for i in 0..<32 { for j in 0..<33 where (i - 15) * (i - 15) + (j - 16) * (j - 16) < 25 { disk[i * 33 + j] = 1000 } }
let probe = try ProbeSize.estimate(DetectorImage(detectorRows: 32, detectorColumns: 33, values: disk, path: .cpu))
check("ProbeSize.estimate", probe.centerRow == 15 && probe.centerColumn == 16 && probe.radius > 4 && probe.threshold > 0)
let rows = 9, cols = 8
var gx = [Double](), gy = [Double]()
for r in 0..<rows { for c in 0..<cols {
    let f = { (r: Double, c: Double) in exp(-((r - 4) * (r - 4) + (c - 3.5) * (c - 3.5)) / 4) }
    gx.append((f(Double(r + 1), Double(c)) - f(Double(r - 1), Double(c))) / 2)
    gy.append((f(Double(r), Double(c + 1)) - f(Double(r), Double(c - 1))) / 2)
} }
let field = try ScanVectorField(scanRows: rows, scanColumns: cols, row: gx, column: gy)
let (fitted, normalized) = try CenterOfMassReference.normalize(field, reference: .none, reciprocalSampling: 1)
check("CenterOfMassReference.normalize", fitted.row.allSatisfy { $0 == 0 } && normalized == field)
let rotation = try RotationSolver.solve(normalized, anglesDegrees: RotationSolver.defaultAnglesDegrees,
                                        criterion: .minimizeCurl, forceRotationDegrees: nil, forceTranspose: nil)
check("RotationSolver.solve", rotation.degrees == 0 && !rotation.transposed && rotation.curve.count == 179)
let dpc = try DPC.reconstruct(rotation.corrected, scanSampling: (1, 1), paddingFactor: 2, maxIterations: 16)
check("DPC.reconstruct", dpc.phase.count == rows * cols && dpc.errors.count <= 16 && dpc.finalStepSize > 0)
do {
    _ = try DPC.reconstruct(try ScanVectorField(scanRows: 3, scanColumns: 3, row: Array(repeating: 0, count: 9),
                                                column: Array(repeating: 0, count: 9)))
    check("CuboPhaseError.zeroField", false)
} catch let error as CuboPhaseError {
    check("CuboPhaseError.zeroField", error == .zeroField && error.errorDescription?.contains("app") == false)
}
// Slice 3: origins, fits, centred and shifted imaging, shifted patterns.
let (origins, originRadius) = try await OriginMeasurement.measure(of: eager, radius: 2.5, radiusScale: 1.2)
check("OriginMeasurement.measure", originRadius == 2.5 && origins.row.values.count == 12 && origins.meanRow.isFinite)
let fittedPlane = try OriginFit.fit(origins.row.values, rows: 3, columns: 4, model: .plane, robust: true)
check("OriginFit.fit", fittedPlane.count == 12 && OriginFitModel.allCases.count == 4)
let centredImage = try await VirtualImaging.image(of: lazy, geometry: .circle(centerRow: 0, centerColumn: 0, radius: 2),
                                                  centeredOn: (row: origins.meanRow, column: origins.meanColumn),
                                                  pixelSize: nil, shift: .bilinear(origins))
check("VirtualImaging.image centred and shifted", centredImage.values.count == 12)
let shiftedPatterns = try await VirtualDiffraction.meanAndMax(of: lazy, shift: .fourier(origins))
let unitFilter = try GaussianFilter(sigma: 1)
check("VirtualDiffraction.meanAndMax shifted", shiftedPatterns.mean.values.count == 72 && unitFilter.kernel.count == 9)
// Slice 4: the median pattern, selections, origins with a scan mask, the Friedel origin, the beamstop mask, the vacuum
// probe and its kernels. Every expectation is closed-form: a consumer can verify it by hand.
func isInvalidInput(_ body: () async throws -> Void) async -> Bool {
    do { try await body(); return false } catch let error as CuboVirtualError {
        if case .invalidInput = error { return true }
        return false
    } catch { return false }
}
let tinyShape = try CubeShape(scanRows: 1, scanColumns: 3, detectorRows: 4, detectorColumns: 4)
let tinyP0 = (0..<16).map { Float($0) }, tinyP1 = (0..<16).map { Float(2 * $0 + 1) }, tinyP2 = (0..<16).map { Float(100 - $0) }
let tiny = try MemoryCube(shape: tinyShape, values: tinyP0 + tinyP1 + tinyP2)
let tinyMiddle = (0..<16).map { k in [Double(tinyP0[k]), Double(tinyP1[k]), Double(tinyP2[k])].sorted()[1] }
let tinyMedian = try await VirtualDiffraction.median(of: tiny)
check("VirtualDiffraction.median, odd count: the middle value", tinyMedian.values == tinyMiddle
      && tinyMedian.detectorRows == 4 && tinyMedian.detectorColumns == 4)
let evenMedian = try await VirtualDiffraction.median(of: eager)
let evenWant = (0..<72).map { k -> Double in
    let v = (0..<12).map { Double(values[$0 * 72 + k]) }.sorted()
    return (v[5] + v[6]) / 2
}
check("VirtualDiffraction.median, even count: the mean of the two middle values", evenMedian.values == evenWant)
check("DiffractionStatistic cases", DiffractionStatistic.allCases == [.mean, .max, .median]
      && DiffractionStatistic.median.rawValue == "median")
let selectedOne = try await VirtualDiffraction.pattern(of: tiny, statistic: .mean, scan: .mask([false, true, false]))
check("VirtualDiffraction.pattern: a mask selecting one position gives that pattern", selectedOne.values == tinyP1.map(Double.init))
let selectedTwoMax = try await VirtualDiffraction.pattern(of: tiny, statistic: .max, scan: .mask([true, false, true]))
check("VirtualDiffraction.pattern .max over a two-position mask",
      selectedTwoMax.values == (0..<16).map { Swift.max(Double(tinyP0[$0]), Double(tinyP2[$0])) })
let selectedMedian = try await VirtualDiffraction.pattern(of: tiny, statistic: .median, scan: .mask([true, true, true]))
check("VirtualDiffraction.pattern .median over a full mask equals the median", selectedMedian.values == tinyMiddle)
let unitWeights = try await VirtualDiffraction.pattern(of: eager, statistic: .mean, scan: .weights(Array(repeating: 1, count: 12)))
let plainMean = try await VirtualDiffraction.pattern(of: eager, statistic: .mean)
check("VirtualDiffraction.pattern: weights all one give the mean",
      zip(unitWeights.values, plainMean.values).allSatisfy { abs($0 - $1) < 1e-12 })
let firstWeighted = try await VirtualDiffraction.pattern(of: tiny, statistic: .mean, scan: .weights([1, 0, 0]))
check("VirtualDiffraction.pattern: weights [1, 0, 0] give the first pattern", firstWeighted.values == tinyP0.map(Double.init))
let allScan = try await VirtualDiffraction.pattern(of: eager, statistic: .max, scan: .all)
let slice1Max = try await VirtualDiffraction.max(of: eager, path: .cpu)
check("VirtualDiffraction.pattern .all .max equals slice 1's max", allScan.values == slice1Max.values && plainMean.values.count == 72)
check("ScanSelection is Hashable", ScanSelection.all == ScanSelection.all && ScanSelection.mask([true]) != ScanSelection.mask([false])
      && Set([ScanSelection.all, .weights([1]), .weights([1])]).count == 2)
let shiftedSelection = try await VirtualDiffraction.pattern(of: lazy, statistic: .mean, scan: .all, shift: .rounded(origins))
check("VirtualDiffraction.pattern with a rounded shift", shiftedSelection.values.count == 72)
check("VirtualDiffraction.pattern: an all-false mask is invalidInput",
      await isInvalidInput { _ = try await VirtualDiffraction.pattern(of: tiny, statistic: .mean, scan: .mask([false, false, false])) })
check("VirtualDiffraction.pattern: the median with a shift is invalidInput",
      await isInvalidInput { _ = try await VirtualDiffraction.pattern(of: eager, statistic: .median, shift: .rounded(origins)) })
do {
    _ = try await VirtualDiffraction.pattern(of: tiny, statistic: .mean, scan: .mask([true]))
    check("VirtualDiffraction.pattern: a short mask is maskLengthMismatch", false)
} catch let error as CuboVirtualError {
    check("VirtualDiffraction.pattern: a short mask is maskLengthMismatch", error == .maskLengthMismatch(expected: 3, received: 1))
}
let selectMask = (0..<12).map { $0 % 2 == 0 }
let (maskedOrigins, maskedRadius) = try await OriginMeasurement.measure(of: eager, radius: 2.5, radiusScale: 1.2,
                                                                         scanMask: selectMask, fastCenter: false)
check("OriginMeasurement.measure with a scan mask: NaN exactly where excluded", maskedRadius == 2.5
      && (0..<12).allSatisfy { maskedOrigins.row.values[$0].isNaN == !selectMask[$0]
                               && maskedOrigins.column.values[$0].isNaN == !selectMask[$0] })
let (fastOrigins, _) = try await OriginMeasurement.measure(of: eager, radius: 2.5, radiusScale: 1.2, scanMask: nil, fastCenter: true)
check("OriginMeasurement.measure fastCenter: integer row and column",
      fastOrigins.row.values.allSatisfy { $0 == $0.rounded() } && fastOrigins.column.values.allSatisfy { $0 == $0.rounded() })
let filledOrigins = maskedOrigins.replacingNaNWithMean()
let keptRows = (0..<12).filter { selectMask[$0] }.map { maskedOrigins.row.values[$0] }
let keptMean = keptRows.reduce(0, +) / Double(keptRows.count)
check("OriginMap.replacingNaNWithMean fills NaN with the mean of the rest",
      filledOrigins.row.values.allSatisfy { !$0.isNaN } && filledOrigins.column.values.allSatisfy { !$0.isNaN }
      && (0..<12).allSatisfy { selectMask[$0] ? filledOrigins.row.values[$0] == maskedOrigins.row.values[$0]
                                              : abs(filledOrigins.row.values[$0] - keptMean) < 1e-9 })
// Friedel: a pattern point-symmetric about the half-integer centre (3.5, 3.5) of an 8 x 8 detector.
let half: [Double] = [1, 2, 5, 9, 9, 5, 2, 1]
let symmetric = (0..<64).map { half[$0 / 8] * half[$0 % 8] }
let symmetricImage = DetectorImage(detectorRows: 8, detectorColumns: 8, values: symmetric, path: .cpu)
let friedel = try FriedelOrigin.origin(of: symmetricImage)
check("FriedelOrigin.origin of a point-symmetric pattern returns its centre",
      abs(friedel.row - 3.5) < 1e-9 && abs(friedel.column - 3.5) < 1e-9)
let friedelUp = try FriedelOrigin.origin(of: symmetricImage, upsampleFactor: 4)
check("FriedelOrigin.origin with upsampleFactor 4 keeps the centre", abs(friedelUp.row - 3.5) < 1e-9 && abs(friedelUp.column - 3.5) < 1e-9)
let friedelCubeShape = try CubeShape(scanRows: 1, scanColumns: 2, detectorRows: 8, detectorColumns: 8)
let friedelCube = try MemoryCube(shape: friedelCubeShape, values: (symmetric + symmetric.map { 3 * $0 }).map(Float.init))
let friedelMap = try await FriedelOrigin.measure(of: friedelCube)
check("FriedelOrigin.measure: every position at the centre", friedelMap.row.values.count == 2
      && friedelMap.row.values.allSatisfy { abs($0 - 3.5) < 1e-9 } && friedelMap.column.values.allSatisfy { abs($0 - 3.5) < 1e-9 })
let friedelMasked = try FriedelOrigin.origin(of: symmetricImage, mask: Array(repeating: true, count: 64))
check("FriedelOrigin.origin with a mask", friedelMasked.row.isFinite && friedelMasked.column.isFinite)
check("FriedelOrigin: an upsampleFactor of 2 is invalidInput",
      await isInvalidInput { _ = try FriedelOrigin.origin(of: symmetricImage, upsampleFactor: 2) })
// Beamstop: a 12 x 12 pattern of 100 with a dark 4 x 4 block at the corner (connected to the border).
let inBlock = { (k: Int) in k / 12 < 4 && k % 12 < 4 }
let beamstopImage = DetectorImage(detectorRows: 12, detectorColumns: 12, values: (0..<144).map { inBlock($0) ? 0.0 : 100.0 }, path: .cpu)
let beamstop = try BeamstopMask.make(from: beamstopImage)
check("BeamstopMask.make: true on the dark border-connected block, false at the centre",
      beamstop.count == 144 && (0..<144).allSatisfy { !inBlock($0) || beamstop[$0] } && !beamstop[6 * 12 + 6])
let beamstopNoEdge = try BeamstopMask.make(from: beamstopImage, threshold: 0.25, distanceEdge: 1, includeEdges: false,
                                           sigma: 0, scaleRadial: nil)
check("BeamstopMask.make without the edges: the block only", (0..<144).allSatisfy { beamstopNoEdge[$0] == inBlock($0) })
check("BeamstopMask.make: a NaN pattern is invalidInput",
      await isInvalidInput { _ = try BeamstopMask.make(from: DetectorImage(detectorRows: 2, detectorColumns: 2, values: [1, .nan, 1, 1], path: .cpu)) })
// The vacuum probe, from the 32 x 33 disk above in a 2 x 2 scan whose position s holds (s + 1) times the disk.
let diskImage = DetectorImage(detectorRows: 32, detectorColumns: 33, values: disk, path: .cpu)
let vacuumShape = try CubeShape(scanRows: 2, scanColumns: 2, detectorRows: 32, detectorColumns: 33)
let sameDisk = try MemoryCube(shape: vacuumShape, values: (0..<4).flatMap { _ in disk.map(Float.init) })
let scaledDisk = try MemoryCube(shape: vacuumShape, values: (0..<4).flatMap { s in disk.map { Float($0) * Float(s + 1) } })
let vacuum = try await VacuumProbe.measure(of: sameDisk, zeroVacuum: false)
check("VacuumProbe.measure of identical patterns returns that pattern", vacuum.averaged.values.count == 32 * 33
      && zip(vacuum.averaged.values, disk).allSatisfy { abs($0 - $1) < 1e-9 } && vacuum.probe == vacuum.averaged)
check("VacuumProbe.estimate is the disk's", vacuum.estimate.centerRow == 15 && vacuum.estimate.centerColumn == 16 && vacuum.estimate.radius > 4)
let damped = try await VacuumProbe.measure(of: sameDisk)
check("VacuumProbe.probe is damped: sin(1) at the centre, 0 far away",
      abs(damped.probe[15, 16] - damped.averaged[15, 16] * sin(1.0)) < 1e-9 && damped.probe[0, 0] == 0)
let oneScaled = try await VacuumProbe.measure(of: scaledDisk, scanMask: [false, false, true, false], zeroVacuum: false)
check("VacuumProbe.measure of a scan mask selecting one position gives that pattern",
      zip(oneScaled.averaged.values, disk).allSatisfy { abs($0 - 3 * $1) < 1e-9 })
check("VacuumProbe.measure: an empty selection is invalidInput",
      await isInvalidInput { _ = try await VacuumProbe.measure(of: sameDisk, scanMask: [false, false, false, false]) })
let zeroed = VacuumProbe.zeroVacuum(diskImage, center: (15, 16), alpha: 5, alphaMax: 1.2)
check("VacuumProbe.zeroVacuum: kept inside alpha * alphaMax, zero outside",
      zeroed[15, 16] == diskImage[15, 16] && zeroed[0, 0] == 0 && zeroed[15, 16 + 5] == diskImage[15, 16 + 5] && zeroed[15, 16 + 7] == 0)
let dampedImage = VacuumProbe.dampVacuum(diskImage, center: (15, 16), alpha: 5)
check("VacuumProbe.dampVacuum: sin(1) inside alpha (1 + start), 0 far away",
      abs(dampedImage[15, 16] - diskImage[15, 16] * sin(1.0)) < 1e-12 && dampedImage[0, 0] == 0)
// Kernels from the disk probe.
let flatKernel = try ProbeKernel.flat(diskImage)
check("ProbeKernel.flat sums to 1", abs(flatKernel.values.reduce(0, +) - 1) < 1e-12)
let flatCorner = try ProbeKernel.flat(diskImage, origin: (15, 16), bilinear: true)
check("ProbeKernel.flat with an integer origin and bilinear: the centre moves to the corner",
      abs(flatCorner[0, 0] - 1000 / disk.reduce(0, +)) < 1e-12 && abs(flatCorner.values.reduce(0, +) - 1) < 1e-12)
let gaussianKernel = try ProbeKernel.gaussian(diskImage, sigma: 0.5)
check("ProbeKernel.gaussian sums to 0", abs(gaussianKernel.values.reduce(0, +)) < 1e-12 && gaussianKernel.detectorRows == 32)
let sigmoidKernel = try ProbeKernel.sigmoid(diskImage, innerRadius: 3, outerRadius: 8)
check("ProbeKernel.sigmoid sums to 0", abs(sigmoidKernel.values.reduce(0, +)) < 1e-12)
let logisticKernel = try ProbeKernel.sigmoid(diskImage, innerRadius: 3, outerRadius: 8, profile: .logistic, origin: (15, 16),
                                             bilinear: false)
check("ProbeKernel.sigmoid .logistic sums to 0",
      abs(logisticKernel.values.reduce(0, +)) < 1e-12 && SigmoidProfile.allCases == [.sineSquared, .logistic])
do {
    _ = try ProbeKernel.sigmoid(diskImage, innerRadius: 8, outerRadius: 3)
    check("ProbeKernel.sigmoid: outer not above inner is innerNotBelowOuter", false)
} catch let error as CuboVirtualError {
    check("ProbeKernel.sigmoid: outer not above inner is innerNotBelowOuter", error == .innerNotBelowOuter(inner: 8, outer: 3))
}
let synthetic = try SyntheticProbe.make(radius: 5, width: 2, detectorRows: 32, detectorColumns: 32)
check("SyntheticProbe.make is 1/2 on the radius, near 1 inside and near 0 outside",
      synthetic.detectorRows == 32 && synthetic[16, 21] == 0.5 && synthetic[16, 11] == 0.5 && synthetic[16, 16] > 0.9
      && synthetic[0, 0] < 0.001)
check("SyntheticProbe.make: a zero width is invalidInput",
      await isInvalidInput { _ = try SyntheticProbe.make(radius: 5, width: 0, detectorRows: 32, detectorColumns: 32) })
check("CuboVirtualError.invalidInput", CuboVirtualError.invalidInput("x") == .invalidInput("x")
      && CuboVirtualError.invalidInput("x") != .invalidInput("y")
      && CuboVirtualError.invalidInput("why").errorDescription == "Invalid input: why.")
print("consumer-check summary: \(checks) checks, \(failures) failures")
exit(failures == 0 ? 0 : 1)
