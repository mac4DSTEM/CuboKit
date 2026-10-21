// What an app reads from CuboKit, through the public API only. Each check prints one line; the
// summary line is what run.sh reads. A cube is written to a temporary .npy, read back lazily and eagerly, and reduced on
// both paths.
import CuboBragg
import CuboCore
import CuboEmbedding
import CuboPhase
import CuboPtycho
import CuboStrain
import CuboVirtual
import Foundation

func isEmbeddingError(_ body: () throws -> Void) async -> Bool {
    do { try body(); return false } catch is EmbeddingError { return true } catch { return false }
}

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
// Bragg disk detection: a kernel from the public probe API, two disks, py4DSTEM's defaults plus a spacing and an edge.
let braggProbe = try SyntheticProbe.make(radius: 4, width: 2, detectorRows: 48, detectorColumns: 48)
let braggKernel = try ProbeKernel.sigmoid(braggProbe, innerRadius: 4, outerRadius: 8, origin: (24, 24))
var braggValues = [Double](repeating: 0, count: 48 * 48)
for (cr, cc, a) in [(24.3, 23.6, 100.0), (10.0, 30.0, 40.0)] {
    for i in 0..<48 { for j in 0..<48 { braggValues[i * 48 + j] += a / (1 + exp(4 * (hypot(Double(i) - cr, Double(j) - cc) - 4) / 2)) } }
}
var braggOptions = BraggDiskOptions()
braggOptions.minimumPeakSpacing = 8
braggOptions.edgeBoundary = 4
let braggPeaks = try BraggDisks.detect(DetectorImage(detectorRows: 48, detectorColumns: 48, values: braggValues, path: .cpu),
                                       kernel: braggKernel, options: braggOptions)
check("BraggDisks.detect", braggPeaks.count == 2 && abs(braggPeaks[0].row - 24.3) < 0.2 && abs(braggPeaks[0].column - 23.6) < 0.2
      && abs(braggPeaks[1].row - 10) < 0.2 && braggPeaks[0].intensity > braggPeaks[1].intensity)
check("BraggDiskOptions variants", BraggDiskOptions.mac4DSTEM.nonFinitePixels == .neighbourMedian
      && BraggDiskOptions.detectorAdapted(detectorRows: 64, detectorColumns: 64).edgeBoundary == 2
      && ProbeRingHint.outerEdge(of: DetectorImage(detectorRows: 8, detectorColumns: 8, values: [Double](repeating: 0, count: 64), path: .cpu)) == nil)
do {
    var bad = BraggDiskOptions()
    bad.upsampleFactor = 2
    _ = try BraggDisks.detect(DetectorImage(detectorRows: 48, detectorColumns: 48, values: braggValues, path: .cpu), kernel: braggKernel, options: bad)
    check("CuboBraggError.invalidOption", false)
} catch let error as CuboBraggError {
    check("CuboBraggError.invalidOption", { if case .invalidOption = error { return true } else { return false } }())
}
do {
    // The Float32 path, the funnel and the non-finite fill (CuboBragg 0.6.0).
    let image = DetectorImage(detectorRows: 48, detectorColumns: 48, values: braggValues, path: .cpu)
    let detector = try BraggDetector(kernel: braggKernel)
    var singleOptions = braggOptions
    singleOptions.precision = .single
    let (singlePeaks, singleFunnel) = try detector.detectWithFunnel(image, options: singleOptions)
    let (doublePeaks, doubleFunnel) = try detector.detectWithFunnel(image, options: braggOptions)
    check("BraggPrecision / BraggDiskOptions.precision", BraggDiskOptions().precision == .double
          && BraggPrecision.allCases == [.double, .single] && singlePeaks.count == 2
          && abs(singlePeaks[0].row - doublePeaks[0].row) < 1e-3 && abs(singlePeaks[1].column - doublePeaks[1].column) < 1e-3)
    check("BraggDetector.supportsSinglePrecision", BraggDetector.supportsSinglePrecision(detectorRows: 48, detectorColumns: 48)
          && !BraggDetector.supportsSinglePrecision(detectorRows: 56, detectorColumns: 40))
    let plainPeaks = try detector.detect(image, options: braggOptions)
    check("BraggDetector.detectWithFunnel(DetectorImage)", doublePeaks == plainPeaks
          && doubleFunnel.acceptedCount == 2 && doubleFunnel.localMaximumCount >= 2 && singleFunnel.acceptedCount == 2)
    check("BraggFunnel fields", doubleFunnel.localMaximumCount >= doubleFunnel.afterAbsoluteCutCount
          && doubleFunnel.afterAbsoluteCutCount >= doubleFunnel.afterRelativeCutCount
          && doubleFunnel.afterRelativeCutCount >= doubleFunnel.afterSpacingCount && doubleFunnel.afterSpacingCount >= doubleFunnel.acceptedCount
          && !doubleFunnel.wasCountLimited && doubleFunnel.relativeReferenceIntensity != nil && doubleFunnel.relativeReferenceWasAvailable
          && doubleFunnel.correlationMaximum > 0 && doubleFunnel.nonFinitePixelCount == 0)
    let made = BraggFunnel(localMaximumCount: 5, afterAbsoluteCutCount: 4, afterRelativeCutCount: 3, afterSpacingCount: 2,
                           acceptedCount: 1, wasCountLimited: true, relativeReferenceIntensity: 10,
                           relativeReferenceWasAvailable: true, correlationMaximum: 12, nonFinitePixelCount: 3)
    check("BraggFunnel.init", made.acceptedCount == 1 && made.wasCountLimited && made.nonFinitePixelCount == 3 && made == made)
    let floats = braggValues.map { Float($0) }
    let viaFloat = try floats.withUnsafeBufferPointer { try detector.detectWithFunnel($0, options: singleOptions) }
    check("BraggDetector.detectWithFunnel(UnsafeBufferPointer<Float>)", viaFloat.peaks.count == 2 && viaFloat.funnel.acceptedCount == 2)
    let detectedFloat = try floats.withUnsafeBufferPointer { try detector.detect($0, options: singleOptions) }
    check("BraggDetector.detect(UnsafeBufferPointer<Float>)", detectedFloat == viaFloat.peaks)
    var filled: [Double] = [1, 2, 3, .nan, 5, 6, 7, 8, 9]
    var filledFloat: [Float] = [1, 2, 3, .infinity, 5, 6, 7, 8, 9]
    check("BraggDetector.fillNonFinite(Double)", BraggDetector.fillNonFinite(&filled, rows: 3, columns: 3) == 1 && filled[3] == 5)
    check("BraggDetector.fillNonFinite(Float)", BraggDetector.fillNonFinite(&filledFloat, rows: 3, columns: 3) == 1 && filledFloat[3] == 5)
    var mapOptions = BraggDiskOptions()
    mapOptions.correlationSigma = 0; mapOptions.subpixel = .pixel; mapOptions.edgeBoundary = 1; mapOptions.minimumPeakSpacing = 0
    var map = [Double](repeating: 0, count: 100); map[55] = 3; map[22] = 1
    let found = try BraggMaxima.findWithFunnel(in: map, rows: 10, columns: 10, options: mapOptions)
    let plain = try BraggMaxima.find(in: map, rows: 10, columns: 10, options: mapOptions)
    check("BraggMaxima.findWithFunnel", found.peaks.count == 2 && found.funnel.localMaximumCount == 2 && found.funnel.correlationMaximum == 3
          && found.peaks == plain)
    do {
        var tooBig = singleOptions
        tooBig.subpixel = .parabolic
        _ = try BraggDetector(kernel: DetectorImage(detectorRows: 40, detectorColumns: 56, values: [Double](repeating: 1, count: 40 * 56), path: .cpu))
            .detect(DetectorImage(detectorRows: 40, detectorColumns: 56, values: [Double](repeating: 1, count: 40 * 56), path: .cpu), options: tooBig)
        check("BraggPrecision.single refuses an unsupported size", false)
    } catch let error as CuboBraggError {
        check("BraggPrecision.single refuses an unsupported size", { if case .invalidOption = error { return true } else { return false } }())
    }
} catch {
    check("Bragg single path, funnel", false)
}
check("CuboStrain scope", CuboStrain.scope.contains("scan-level"))
do {
    // Scan-level strain: a 2 x 2 scan of a square lattice, known directions, supplied reference.
    var lists: [[StrainPeak]] = []
    for _ in 0..<4 { lists.append((-1...1).flatMap { h in (-1...1).map { k in StrainPeak(row: Double(8 * h), column: Double(8 * k), intensity: 1) } }) }
    let lattice = try StrainDirections.lattice(basis: LatticeBasis(g1: LatticeVector(row: 8, column: 0), g2: LatticeVector(row: 0, column: 8)))
    var options = StrainMapOptions(maximumPeakSpacing: 2)
    options.reference = .supplied(lattice.basis)
    let map = try StrainMapping.compute(peaks: lists, scanRows: 2, scanColumns: 2, detectorRows: 32, detectorColumns: 32,
                                        origin: LatticeVector(row: 16, column: 16), directions: lattice, options: options)
    let histogram = try StrainMapping.histogram(peaks: lists, scanRows: 2, scanColumns: 2, detectorRows: 32, detectorColumns: 32,
                                                origin: LatticeVector(row: 16, column: 16))
    var basisOptions = StrainBasisOptions()
    basisOptions.subpixel = .parabolic
    let chosen = try StrainMapping.chooseBasisVectors(histogram: histogram, options: basisOptions)
    check("StrainMapping scan-level", map.valid.allSatisfy { $0 } && map.exx.allSatisfy { abs($0) < 1e-12 } && histogram.rows == 32
          && chosen.directions.count == chosen.maxima.count && BraggDiskOptions().tieOrder == .rasterOrder
          && (StrainMapping.centredPeaks(BraggVectors(scanRows: 1, scanColumns: 1, peaks: [[]]), origin: LatticeVector(row: 0, column: 0)).count == 1))
} catch {
    check("StrainMapping scan-level", false)
}
do {
    // Round 6: consensus diagnostics with cancellation, MAD inliers.
    let positions = (0..<20).map { _ in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 0.0), (0.0, -1.0)].map { LatticePeak(row: 8 * $0.0, column: 10 * $0.1, intensity: 1) } }
    let result = try LatticeStrain.chooseBasisConsensusResult(positions, isCancelled: { false })
    let plain = try LatticeStrain.chooseBasisConsensus(positions)
    var cancelled = false
    do { _ = try LatticeStrain.chooseBasisConsensusResult(positions, isCancelled: { true }) } catch is CancellationError { cancelled = true }
    check("LatticeStrain.chooseBasisConsensusResult", result?.basis == plain && result?.observationCount == 100 && result?.supportCount == 100
          && (result?.residual ?? 1) < 1e-5 && (result?.conditionNumber ?? 0) >= 1 && (result?.indexingTolerance ?? 0) >= 0.5 && cancelled)
    let mad = try LatticeStrain.referenceMADResult([LatticeBasis(g1: .init(row: 8, column: 0), g2: .init(row: 0, column: 10)),
                                                    LatticeBasis(g1: .init(row: 8, column: 0), g2: .init(row: 0, column: 10)),
                                                    LatticeBasis(g1: .init(row: 8, column: 0), g2: .init(row: 0, column: 10))])
    let again = try LatticeStrain.referenceMAD([mad.reference, mad.reference, mad.reference])
    check("LatticeStrain.referenceMADResult", mad.inlierIndices == [0, 1, 2] && mad.inlierCount == 3 && mad.reference == again)
    var one = StrainMapOptions(maximumPeakSpacing: 2); one.referenceMethod = .madRejection
    let lists = (0..<4).map { _ in (-1...1).flatMap { h in (-1...1).map { k in StrainPeak(row: Double(8 * h), column: Double(8 * k), intensity: 1) } } }
    let lat = try StrainDirections.lattice(basis: LatticeBasis(g1: .init(row: 8, column: 0), g2: .init(row: 0, column: 8)))
    let m = try StrainMapping.compute(peaks: lists, scanRows: 2, scanColumns: 2, detectorRows: 32, detectorColumns: 32,
                                      origin: LatticeVector(row: 16, column: 16), directions: lat, options: one)
    check("StrainMapResult.referenceInlierCount", m.referenceInlierCount == 4 && m.referencePositionCount == 4)
} catch {
    check("round 6 strain", false)
}
check("CuboPtycho scope", CuboPtycho.scope.contains("probe-centre"))
do {
    // Crop-and-bin view, ptychography preprocessing and the three engines, through the public API.
    let viewed = try ViewedCube(base: eager, specification: CubeViewSpecification(
        scanCrop: AxisCrop(rowOffset: 1, columnOffset: 0, rows: 2, columns: 3),
        detectorCrop: AxisCrop(rowOffset: 0, columnOffset: 1, rows: 7, columns: 8), detectorBin: 2))
    let loaded = try await MemoryCube(loading: viewed)
    var firstBlock: Float = 0                      // scan (1, 0), detector rows 0..<2, columns 1..<3 summed
    for r in 0..<2 { for c in 0..<2 { firstBlock += values[(1 * 4) * 72 + r * 9 + 1 + c] } }
    check("AxisCrop overflow guards",
          AxisCrop.enclosing(firstRow: 1, firstColumn: 1, lastRow: 2, lastColumn: 2, strideRows: Int.max, planeRows: 10, planeColumns: 10) == nil
          && AxisCrop(rowOffset: Int.max - 1, columnOffset: 0, rows: Int.max, columns: 1).rowRange.upperBound == Int.max
          && !AxisCrop(rowOffset: 1, columnOffset: 1, rows: 1, columns: 1).fits(rows: Int.min, columns: 4))
    check("ViewedCube crop and bin", viewed.shape.extents == [2, 3, 3, 4] && viewed.geometry.discardedDetectorRows == 1
          && loaded.values.count == 72 && loaded.values[0] == firstBlock)
    let prepared = try await PtychographyPreparation.prepare(
        source: eager, origins: .uniform(row: 0.3, column: -0.2), geometry: PtychographyScanGeometry(scanSamplingRow: 1, scanSamplingColumn: 1),
        objectSamplingRow: 0.5, objectSamplingColumn: 0.5, wavelength: 0.025, semiangle: 10)
    let states = try MixedstatePtychography.initialProbes(from: prepared.probe, count: 2, seed: 1)
    let mixed = try MixedstatePtychography.reconstruct(amplitudes: prepared.amplitudes.values, positions: prepared.positions,
                                                       object: prepared.object, probes: states, iterations: 1)
    let propagators = try MultislicePtychography.propagators(rows: 8, columns: 9, rowSampling: 0.5, columnSampling: 0.5,
                                                             wavelength: 0.025, sliceThicknesses: [20])
    let multi = try MultislicePtychography.reconstruct(
        amplitudes: prepared.amplitudes.values, positions: prepared.positions,
        object: PtychographyStack([prepared.object, prepared.object]), probe: prepared.probe, sliceThicknesses: [20],
        wavelength: 0.025, rowSampling: 0.5, columnSampling: 0.5, iterations: 1)
    // Error normalisation (py4DSTEM's preprocessing mean), the zero-pixel option and position correction.
    let normalized = try MixedstatePtychography.reconstruct(
        amplitudes: prepared.amplitudes.values, positions: prepared.positions, object: prepared.object, probes: states, iterations: 1,
        errorNormalization: prepared.amplitudes.errorNormalization, zeroFarFieldPixelError: .finite)
    let energyNormalized = try MixedstatePtychography.reconstruct(
        amplitudes: prepared.amplitudes.values, positions: prepared.positions, object: prepared.object, probes: states, iterations: 1,
        errorNormalization: .suppliedAmplitudeEnergy, zeroFarFieldPixelError: .reference)
    let corrected = try SingleslicePtychography.reconstruct(
        amplitudes: prepared.amplitudes.values, positions: prepared.positions, object: prepared.object, probe: prepared.probe, iterations: 1,
        errorNormalization: .meanIntensity(prepared.amplitudes.meanIntensity),
        positionCorrection: PtychographyPositionCorrection(stepSize: 0.5, maxUpdateDistance: 1, maxTotalDistance: 2,
                                                           rowSampling: 0.5, columnSampling: 0.5, constrainCenterOfMass: true))
    check("CuboPtycho error normalisation and position correction", normalized.errors.count == 1 && energyNormalized.errors.count == 1
          && corrected.positions.count == prepared.positions.count && corrected.errors.count == 1
          && PtychographyPositionCorrection().stepSize == 0.5)
    // App options: the object amplitude toggle and the higher-order probe terms.
    let terms = PtychographyAberrationTerm.polar(radialOrder: 2, angularOrder: 3, magnitude: 1000, azimuth: 0.1)
        + [PtychographyAberrationTerm(radialOrder: 6, angularOrder: 0, coefficient: 1e9)]
    let aberrated = try PtychographyProbe.build(rows: 8, columns: 9, rowSampling: 0.5, columnSampling: 0.5, wavelength: 0.025, cutoff: 0.02,
                                                higherOrder: terms)
    let startingProbe = try PtychographyPreparation.initialProbe(rows: 8, columns: 9, samplingRow: 0.5, samplingColumn: 0.5, wavelength: 0.025,
                                                                 semiangle: 20, higherOrder: terms, meanIntensity: 10)
    let uncapped = try SingleslicePtychography.reconstruct(
        amplitudes: prepared.amplitudes.values, positions: prepared.positions, object: prepared.object, probe: prepared.probe, iterations: 1,
        constrainObjectAmplitude: false)
    var preparationOptions = PtychographyPreparationOptions()
    preparationOptions.higherOrderAberrations = terms
    check("CuboPtycho app options", terms.count == 3 && terms[2].component == .cosine && terms[1].component == .sine
          && aberrated.real.allSatisfy(\.isFinite) && startingProbe.real.count == 72 && uncapped.errors.count == 1
          && preparationOptions.higherOrderAberrations == terms)
    check("CuboPtycho preprocessing and engines", prepared.amplitudes.positionCount == 12 && mixed.probes.count == 2
          && mixed.errors.count == 1 && multi.object.count == 2 && propagators.count == 1
          && mixed.object.real.allSatisfy(\.isFinite) && multi.probe.real.allSatisfy(\.isFinite))
}
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
let dpcMagnitude = DPCViews.magnitude(of: field)
let dpcAngle = DPCViews.angle(of: field, convention: .rowOverColumn)
check("DPCViews and RotationConvention", dpcMagnitude.count == rows * cols && dpcAngle.allSatisfy { $0 >= 0 && $0 < 2 * Double.pi }
      && RotationConvention.kit(fromImageFrame: 0.5) == -0.5 && RotationConvention.degreesText(-0.01) == "0.0°"
      && DPCViews.milliradiansPerPixel(wavelength: 0.02, reciprocalSampling: 0.05) == 1.0)
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
let trimmedPlane = try OriginFit.fitTrimmed(originRow: origins.row.values, originColumn: origins.column.values, rows: 3, columns: 4,
                                            model: .plane, trimSigma: 3, madScale: 1.4826, rounds: 3)
check("OriginFit.fitTrimmed", trimmedPlane.fittedRow.count == 12 && trimmedPlane.fittedColumn.count == 12 && trimmedPlane.kept.count == 12
      && trimmedPlane.kept.contains(true) && trimmedPlane.fullScanResidual >= 0 && trimmedPlane.keptResidual >= 0
      && trimmedPlane.excludedFraction != nil)
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
// HotPixelFilter.Replacer: the reusable in-place replacement plan (equals the one-frame `replacing`).
do {
    let hpMask = (0..<20).map { $0 % 7 == 3 }
    let hpFrame = (0..<20).map { Float(($0 * 37) % 11) + (($0 == 3) ? 50 : 0) }
    var hpPlanned = hpFrame
    let replacer = try HotPixelFilter.Replacer(rows: 4, columns: 5, mask: hpMask)
    replacer.apply(to: &hpPlanned)
    let hpWant = try HotPixelFilter.replacing(hpFrame, rows: 4, columns: 5, mask: hpMask)
    var hpPadded = [Float(-1)] + hpFrame
    hpPadded.withUnsafeMutableBufferPointer { replacer.apply(to: $0, base: 1) }
    check("HotPixelFilter.Replacer equals replacing", hpPlanned == hpWant && Array(hpPadded.dropFirst()) == hpWant
          && hpPadded[0] == -1 && replacer.rows == 4 && replacer.columns == 5)
    do { _ = try HotPixelFilter.Replacer(rows: 2, columns: 2, mask: [true]); check("HotPixelFilter.Replacer mask size", false) }
    catch { check("HotPixelFilter.Replacer mask size", error as? HotPixelFilterError == .invalidInput) }
}

// CuboEmbedding 0.6.0: the streamed PCA, the vDSP binning and the cancellation hook.
let embeddingRows = 40, embeddingColumns = 5
let embeddingData = (0..<(embeddingRows * embeddingColumns)).map { i -> Double in
    let r = Double(i / embeddingColumns), c = Double(i % embeddingColumns)
    return sin(r * 0.7 + c) * (c + 1) + cos(r * 1.3 * (c + 1)) + 0.1 * r
}
var embeddingAccumulator = try PCACovarianceAccumulator(columns: embeddingColumns)
var embeddingHalf = try PCACovarianceAccumulator(columns: embeddingColumns)
try embeddingAccumulator.add(Array(embeddingData[0..<(25 * embeddingColumns)]), rows: 25)
try embeddingHalf.add(Array(embeddingData[(25 * embeddingColumns)...]).map { Float($0) }, rows: 15)
try embeddingAccumulator.merge(embeddingHalf)
check("PCACovarianceAccumulator count, columns, mean", embeddingAccumulator.count == 40 && embeddingAccumulator.columns == 5
      && embeddingAccumulator.mean.count == 5)
let streamedCovariance = try embeddingAccumulator.covariance()
let streamedFit = try PCA.fit(covariance: streamedCovariance, count: embeddingAccumulator.count,
                              mean: embeddingAccumulator.mean, components: 2)
let accumulatorFit = try PCA.fit(embeddingAccumulator, components: 2)
check("PCA.fit(covariance:count:mean:components:) and PCA.fit(_:components:) agree, scores are empty",
      streamedFit == accumulatorFit && streamedFit.scores.isEmpty && streamedFit.rows == 40 && streamedFit.componentCount == 2)
let memoryFit = try PCA.fit(embeddingData, rows: embeddingRows, columns: embeddingColumns, components: 2)
var embeddingResolver = PCASignResolver(result: streamedFit)
var embeddingResolverTail = PCASignResolver(result: streamedFit)
try embeddingResolver.add(Array(embeddingData[0..<(10 * embeddingColumns)]), rows: 10, firstRow: 0)
try embeddingResolverTail.add(Array(embeddingData[(10 * embeddingColumns)...]).map { Float($0) }, rows: 30, firstRow: 10)
try embeddingResolver.merge(embeddingResolverTail)
let resolvedFit = embeddingResolver.resolved()
check("PCASignResolver gives scikit-learn's signs (components equal the in-memory fit)",
      zip(resolvedFit.components, memoryFit.components).allSatisfy { abs($0 - $1) < 1e-4 }
      && zip(resolvedFit.explainedVariance, memoryFit.explainedVariance).allSatisfy { abs($0 - $1) < 1e-4 })
check("PCACovarianceAccumulator refuses a ragged block", await isEmbeddingError {
    var a = try PCACovarianceAccumulator(columns: 2); try a.add([1, 2, 3] as [Double], rows: 2) })
let binPattern = (0..<64).map { Float(($0 * 37) % 101) }
let binScalar = try DiffractionFeatures.binnedVector(pattern: binPattern, rows: 8, columns: 8, binnedSize: 4)
let binAccelerated = try DiffractionFeatures.binnedVector(pattern: binPattern, rows: 8, columns: 8, binnedSize: 4,
                                                           implementation: .accelerate)
let binPointer = try binPattern.withUnsafeBufferPointer {
    try DiffractionFeatures.binnedVector(pattern: $0, rows: 8, columns: 8, binnedSize: 4, implementation: .accelerate)
}
check("DiffractionFeatures.binnedVector(implementation:) .accelerate agrees with .scalar",
      DiffractionFeatures.Implementation.scalar != .accelerate && binAccelerated == binPointer
      && zip(binScalar, binAccelerated).allSatisfy { abs($0 - $1) < 1e-5 })
let kmeansCentroids = Array(embeddingData[0..<(2 * embeddingColumns)])
let kmeansCancelled: Bool
do { _ = try KMeans.fit(embeddingData, rows: embeddingRows, columns: embeddingColumns, initialCentroids: kmeansCentroids,
                        isCancelled: { true }); kmeansCancelled = false } catch is CancellationError { kmeansCancelled = true }
let seededCancelled: Bool
do { _ = try KMeans.fitSeeded(embeddingData, rows: embeddingRows, columns: embeddingColumns, groups: 2, seed: 3,
                              isCancelled: { true }); seededCancelled = false } catch is CancellationError { seededCancelled = true }
check("KMeans.fit and KMeans.fitSeeded isCancelled throw CancellationError", kmeansCancelled && seededCancelled)
let kmeansRun = try KMeans.fitSeeded(embeddingData, rows: embeddingRows, columns: embeddingColumns, groups: 2, seed: 3,
                                     isCancelled: { false })
check("KMeans.fitSeeded runs unchanged with a never-cancelling hook", kmeansRun.labels.count == embeddingRows)
print("consumer-check summary: \(checks) checks, \(failures) failures")
exit(failures == 0 ? 0 : 1)
