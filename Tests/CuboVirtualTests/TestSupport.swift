//
//  TestSupport.swift
//  The fixtures directory (read by #filePath, not as resources), the pins written by tools/slice-1/pins.py before any
//  Swift existed (registration section 4), the real-data directory (outside the repository), and the registered
//  comparison: relative `t` with the floor 1e-6 · max|want| over the image (registration 3.4).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual

enum Fixtures {
    static let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    static func url(_ name: String) -> URL { directory.appendingPathComponent(name) }

    /// `$CUBOKIT_REAL_DATA`, default `~/Library/Caches/CuboKit/real-data` (registration 4).
    static var realData: URL {
        if let env = ProcessInfo.processInfo.environment["CUBOKIT_REAL_DATA"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/CuboKit/real-data")
    }
}

/// A float64 written by the generator as a 17-significant-digit string, parsed by Double(String) (correctly rounded).
struct D: Decodable, Sendable {
    let value: Double
    init(from decoder: any Decoder) throws {
        let text = try decoder.singleValueContainer().decode(String.self)
        guard let v = Double(text) else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: text))
        }
        value = v
    }
}

struct Pins: Decodable, Sendable {
    let versions: [String: String]
    let generatedAt: String
    let seed: Int
    let boundaryPixelsFloat32Disagrees: Int
    let cubes: [CubePin]

    // Explicit keys: Foundation's .convertFromSnakeCase capitalises a letter after a digit ("py4dstem" → "Py4Dstem").
    enum CodingKeys: String, CodingKey {
        case versions, seed, cubes
        case generatedAt = "generated_at"
        case boundaryPixelsFloat32Disagrees = "boundary_pixels_float32_disagrees"
    }

    static let shared: Pins = {
        do {
            let data = try Data(contentsOf: Fixtures.url("slice-1-pins.json"))
            return try JSONDecoder().decode(Pins.self, from: data)
        } catch { fatalError("slice-1-pins.json: \(error)") }
    }()

    func cube(_ name: String) throws -> CubePin {
        try #require(cubes.first { $0.name == name }, "the pins hold no cube \(name)")
    }
}

struct CubePin: Decodable, Sendable {
    let name: String
    let file: String
    let sha256: String
    let shape: [Int]
    let dtype: String
    let exactInFloat32: Bool
    let meanF64: [D]
    let meanPy4dstemFloat32Cube: [D]
    let meanPy4dstemNative: [D]
    let max: [D]
    let meanDistancePy4dstemOwnVsF64: Double
    let geometries: [GeometryPin]

    enum CodingKeys: String, CodingKey {
        case name, file, sha256, shape, dtype, max, geometries
        case exactInFloat32 = "exact_in_float32"
        case meanF64 = "mean_f64"
        case meanPy4dstemFloat32Cube = "mean_py4dstem_float32_cube"
        case meanPy4dstemNative = "mean_py4dstem_native"
        case meanDistancePy4dstemOwnVsF64 = "mean_distance_py4dstem_own_vs_f64"
    }

    var isInteger: Bool { dtype.hasPrefix("uint") || dtype.hasPrefix("int") }
    var url: URL { Fixtures.url(file) }
    func mapped() throws -> MappedCube { try MappedCube(npy: url) }
}

struct GeometryPin: Decodable, Sendable {
    let name: String
    let geometry: GeometrySpec
    let maskCount: Int
    let maskPixels: [Int]
    let imageF64: [D]
    let imagePy4dstemFloat32Cube: [D]
    let imagePy4dstemNative: [D]
    let imageDistancePy4dstemOwnVsF64: Double

    enum CodingKeys: String, CodingKey {
        case name, geometry
        case maskCount = "mask_count"
        case maskPixels = "mask_pixels"
        case imageF64 = "image_f64"
        case imagePy4dstemFloat32Cube = "image_py4dstem_float32_cube"
        case imagePy4dstemNative = "image_py4dstem_native"
        case imageDistancePy4dstemOwnVsF64 = "image_distance_py4dstem_own_vs_f64"
    }
}

/// The generator's geometry dictionary (registration 5.2), in the kit's names.
struct GeometrySpec: Decodable, Sendable {
    let kind: String
    let centerRow: Double?, centerColumn: Double?, radius: Double?, inner: Double?, outer: Double?
    let rowMin: Double?, rowMax: Double?, columnMin: Double?, columnMax: Double?
    let row: Int?, column: Int?
    let weights: [D]?

    var geometry: DetectorGeometry {
        switch kind {
        case "circle": return .circle(centerRow: centerRow!, centerColumn: centerColumn!, radius: radius!)
        case "annulus": return .annulus(centerRow: centerRow!, centerColumn: centerColumn!, inner: inner!, outer: outer!)
        case "rectangle": return .rectangle(rowMin: rowMin!, rowMax: rowMax!, columnMin: columnMin!, columnMax: columnMax!)
        case "point": return .point(row: row!, column: column!)
        case "mask": return .mask(weights!.map { Float($0.value) })
        default: fatalError("unknown geometry kind \(kind)")
        }
    }
}

/// The registered comparison (3.4): |got − want| ≤ t · max(|want|, 1e-6 · max_k |want_k|). Returns the largest
/// distance in units of that denominator, so a test can print what it measured.
@discardableResult
func assertRelative(_ got: [Double], _ want: [Double], _ t: Double, _ message: @autoclosure () -> String,
                    sourceLocation: SourceLocation = SourceLocation(fileID: #fileID, filePath: #filePath,
                                                                    line: #line, column: #column)) -> Double {
    #expect(got.count == want.count, "\(message()): count", sourceLocation: sourceLocation)
    guard got.count == want.count, !want.isEmpty else { return .infinity }
    let floor = 1e-6 * (want.map(abs).max() ?? 0)
    var worst = 0.0, worstIndex = 0
    for k in want.indices {
        let denominator = Swift.max(abs(want[k]), floor)
        let d = denominator > 0 ? abs(got[k] - want[k]) / denominator : abs(got[k] - want[k])
        if !(d <= worst) { worst = d; worstIndex = k }
    }
    #expect(worst <= t, "\(message()): worst relative distance \(worst) at \(worstIndex) (got \(got[worstIndex]), want \(want[worstIndex]))",
            sourceLocation: sourceLocation)
    return worst
}

/// Bit for bit.
func assertIdentical(_ got: [Double], _ want: [Double], _ message: @autoclosure () -> String,
                     sourceLocation: SourceLocation = SourceLocation(fileID: #fileID, filePath: #filePath,
                                                                     line: #line, column: #column)) {
    #expect(got.count == want.count, "\(message()): count", sourceLocation: sourceLocation)
    guard got.count == want.count else { return }
    let k = got.indices.first { got[$0].bitPattern != want[$0].bitPattern }
    #expect(k == nil, "\(message()): differs at \(k ?? -1): got \(k.map { got[$0] } ?? 0), want \(k.map { want[$0] } ?? 0)",
            sourceLocation: sourceLocation)
}

/// Whether the real-data directory holds `name`'s export (registration 4); the real-data tests are enabled by it.
func realDataIsPresent(_ name: String) -> Bool {
    FileManager.default.fileExists(atPath: Fixtures.realData.appendingPathComponent("\(name)-pins.json").path)
}

func doubles(_ pins: [D]) -> [Double] { pins.map(\.value) }
