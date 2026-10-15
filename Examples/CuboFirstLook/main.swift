//
//  main.swift: a first look at a 4D-STEM cube, the "first ten minutes" of CuboKit.
//
//      swift run cubo-first-look <cube.npy> [output-folder]
//      swift run cubo-first-look --synthetic [output-folder]
//
//  The cube is a 4D .npy file in py4DSTEM's (R_Nx, R_Ny, Q_Nx, Q_Ny) C-order layout, written in Python with
//  `np.save("cube.npy", datacube.data)`. It is opened lazily (nothing is read until a pass needs it). The example finds
//  the probe on the mean pattern, makes a bright-field image with a circle on the probe, measures the centre of mass of
//  every pattern, and writes four PNGs and numbers.json. Rows and columns, never x and y.
//

import CuboCore
import CuboVirtual
import Foundation
import ImageIO
import CoreGraphics

// MARK: - Output helpers

struct UsageError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

/// Writes `values` (`rows × columns`) as an 8-bit grey PNG, scaled linearly from the smallest to the largest finite
/// value (non-finite values draw black). Returns the (min, max) used.
@discardableResult
func writeGreyPNG(_ values: [Double], rows: Int, columns: Int, to url: URL) throws -> (min: Double, max: Double) {
    let finite = values.filter(\.isFinite)
    guard let lo = finite.min(), let hi = finite.max() else { throw UsageError("\(url.lastPathComponent): no finite values") }
    let span = hi - lo
    let pixels: [UInt8] = values.map { v in
        guard v.isFinite, span > 0 else { return 0 }
        return UInt8(((v - lo) / span * 255).rounded())
    }
    guard let provider = CGDataProvider(data: Data(pixels) as CFData),
          let image = CGImage(width: columns, height: rows, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: columns,
                              space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                              provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
    else { throw UsageError("could not create \(url.lastPathComponent)") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw UsageError("could not write \(url.path)") }
    return (lo, hi)
}

func mean(_ values: [Double]) -> Double {
    let finite = values.filter(\.isFinite)
    return finite.isEmpty ? .nan : finite.reduce(0, +) / Double(finite.count)
}

/// Numbers for numbers.json: JSON has no NaN, so a missing value is written as null.
func json(_ x: Double) -> Any { x.isFinite ? x : NSNull() }

// MARK: - A small cube without data

/// 16 × 16 scan positions, a 32 × 32 detector: a disk of radius 5 (value 1000) whose centre moves by 0.2 pixel per
/// scan row (detector row) and per scan column (detector column), plus uniform noise from a fixed-seed generator.
func syntheticCube() throws -> MemoryCube {
    let shape = try CubeShape(scanRows: 16, scanColumns: 16, detectorRows: 32, detectorColumns: 32)
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15                       // SplitMix64, fixed seed
    func noise() -> Float {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Float((z ^ (z >> 31)) >> 40) / Float(1 << 24) * 5     // 0 ..< 5
    }
    var values = [Float]()
    values.reserveCapacity(shape.elementCount)
    for scanRow in 0..<16 {
        for scanColumn in 0..<16 {
            let centerRow = 16 + 0.2 * (Double(scanRow) - 7.5)
            let centerColumn = 16 + 0.2 * (Double(scanColumn) - 7.5)
            for row in 0..<32 {
                for column in 0..<32 {
                    let inside = (Double(row) - centerRow) * (Double(row) - centerRow)
                        + (Double(column) - centerColumn) * (Double(column) - centerColumn) <= 25
                    values.append((inside ? 1000 : 0) + noise())
                }
            }
        }
    }
    return try MemoryCube(shape: shape, values: values)
}

// MARK: - The first look

func firstLook(of cube: some CubeSource, label: String, into folder: URL) async throws {
    let shape = cube.shape
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    print("\(label): \(shape.scanRows) scan rows × \(shape.scanColumns) scan columns, "
          + "\(shape.detectorRows) detector rows × \(shape.detectorColumns) detector columns")

    // 1. The mean diffraction pattern (py4DSTEM get_dp_mean), in Double.
    let meanPattern = try await VirtualDiffraction.mean(of: cube)
    let logged = meanPattern.values.map { log1p(max($0, 0)) }
    try writeGreyPNG(logged, rows: shape.detectorRows, columns: shape.detectorColumns,
                     to: folder.appendingPathComponent("mean-pattern.png"))

    // 2. The probe: radius and centre of the bright central disk of the mean pattern (py4DSTEM get_probe_size).
    let probe = try ProbeSize.estimate(meanPattern)
    print(String(format: "probe: radius %.3f px, centre row %.3f, column %.3f", probe.radius, probe.centerRow, probe.centerColumn))

    // 3. One virtual bright-field image: the sum over a circle on the probe, at every scan position.
    let brightField = try await VirtualImaging.image(
        of: cube, geometry: .circle(centerRow: probe.centerRow, centerColumn: probe.centerColumn, radius: probe.radius))
    let bf = try writeGreyPNG(brightField.values, rows: shape.scanRows, columns: shape.scanColumns,
                              to: folder.appendingPathComponent("bright-field.png"))

    // 4. The centre of mass of every pattern, in detector pixels (CPU reference path).
    let com = try await CenterOfMass.measure(of: cube)
    try writeGreyPNG(com.row.values, rows: shape.scanRows, columns: shape.scanColumns,
                     to: folder.appendingPathComponent("com-row.png"))
    try writeGreyPNG(com.column.values, rows: shape.scanRows, columns: shape.scanColumns,
                     to: folder.appendingPathComponent("com-column.png"))

    // 5. The numbers.
    let numbers: [String: Any] = [
        "shape": ["scanRows": shape.scanRows, "scanColumns": shape.scanColumns,
                  "detectorRows": shape.detectorRows, "detectorColumns": shape.detectorColumns],
        "probe": ["radius": json(probe.radius), "centerRow": json(probe.centerRow),
                  "centerColumn": json(probe.centerColumn)],
        "brightField": ["min": bf.min, "max": bf.max, "mean": mean(brightField.values)],
        "centerOfMass": ["meanRow": json(mean(com.row.values)), "meanColumn": json(mean(com.column.values))],
    ]
    let data = try JSONSerialization.data(withJSONObject: numbers, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: folder.appendingPathComponent("numbers.json"))
    print(String(format: "bright field: min %.6g, max %.6g, mean %.6g", bf.min, bf.max, mean(brightField.values)))
    print(String(format: "centre of mass: mean row %.4f, mean column %.4f", mean(com.row.values), mean(com.column.values)))
    print("wrote mean-pattern.png, bright-field.png, com-row.png, com-column.png and numbers.json to \(folder.path)")
}

// MARK: - Command line

let arguments = Array(CommandLine.arguments.dropFirst())
do {
    guard let first = arguments.first, arguments.count <= 2, !first.hasPrefix("-") || first == "--synthetic" else {
        throw UsageError("usage: cubo-first-look <cube.npy> [output-folder]   or   cubo-first-look --synthetic [output-folder]")
    }
    let folder = URL(fileURLWithPath: arguments.count == 2 ? arguments[1] : "cubo-first-look-output")
    if first == "--synthetic" {
        try await firstLook(of: try syntheticCube(), label: "synthetic cube", into: folder)
    } else {
        let path = URL(fileURLWithPath: first)
        guard FileManager.default.fileExists(atPath: path.path) else { throw UsageError("no such file: \(path.path)") }
        try await firstLook(of: try MappedCube(npy: path), label: path.lastPathComponent, into: folder)
    }
} catch {
    FileHandle.standardError.write(Data("cubo-first-look: \(error.localizedDescription)\n".utf8))
    exit(1)
}
