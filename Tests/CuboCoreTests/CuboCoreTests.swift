//
//  CuboCoreTests.swift
//  The cube model (registration 1.1, 3.5, 5.5): shapes, tilings, the eager and lazy cubes, the .npy container and the
//  element conversions.
//

import Foundation
import XCTest
import CuboCore

final class CuboCoreTests: XCTestCase {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("CuboVirtualTests/Fixtures")
    var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory.appendingPathComponent("CuboCoreTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: scratch) }

    // MARK: Shapes and tilings

    func testShapeValidatesAndCounts() throws {
        let s = try CubeShape([2, 3, 5, 7])
        XCTAssertEqual(s.scanCount, 6); XCTAssertEqual(s.detectorCount, 35); XCTAssertEqual(s.elementCount, 210)
        XCTAssertEqual(s.elementsPerScanRow, 105); XCTAssertEqual(s.scanIndex(row: 1, column: 2), 5)
        XCTAssertThrowsError(try CubeShape([2, 0, 5, 7]))
        XCTAssertThrowsError(try CubeShape([2, 3, 5]))
        XCTAssertThrowsError(try CubeShape([Int.max / 2, 4, 4, 4]))
        let decoded = try JSONDecoder().decode(CubeShape.self, from: JSONEncoder().encode(s))
        XCTAssertEqual(decoded, s)
        XCTAssertThrowsError(try JSONDecoder().decode(CubeShape.self, from: Data(
            #"{"scanRows":0,"scanColumns":1,"detectorRows":1,"detectorColumns":1}"#.utf8)))
    }

    /// `.automatic` is a function of the shape alone: the largest row count whose Float32 tile is at most 64 MiB.
    func testAutomaticTilingIsAFunctionOfTheShape() throws {
        let almgsi = try CubeShape([330, 330, 64, 64])          // 5 406 720 bytes per scan row
        XCTAssertEqual(try ScanTiling.automatic.rowsPerTile(for: almgsi), 12)
        let big = try CubeShape([20, 20, 512, 512])             // 20 971 520 bytes per scan row
        XCTAssertEqual(try ScanTiling.automatic.rowsPerTile(for: big), 3)
        let huge = try CubeShape([4, 64, 512, 512])             // one row is over 64 MiB
        XCTAssertEqual(try ScanTiling.automatic.rowsPerTile(for: huge), 1)
        let small = try CubeShape([2, 3, 5, 7])
        XCTAssertEqual(try ScanTiling.automatic.rowsPerTile(for: small), 2)
        XCTAssertEqual(try ScanTiling.rows(4).ranges(for: try CubeShape([10, 1, 1, 1])), [0..<4, 4..<8, 8..<10])
        XCTAssertEqual(try ScanTiling.rows(99).rowsPerTile(for: small), 2)
        XCTAssertThrowsError(try ScanTiling.rows(0).rowsPerTile(for: small))
    }

    // MARK: Eager and lazy

    func testEagerCubeReadsAndRefuses() async throws {
        let shape = try CubeShape([2, 3, 5, 7])
        let values = (0..<shape.elementCount).map(Float.init)
        let cube = try MemoryCube(shape: shape, values: values)
        let p = try await cube.readPattern(row: 1, column: 2)
        XCTAssertEqual(p.values, Array(values[175..<210]))
        XCTAssertEqual(p[4, 6], 209)
        let t = try await cube.readScanTile(rows: 1..<2)
        XCTAssertEqual(t.values, Array(values[105..<210])); XCTAssertEqual(t.positionCount, 3)
        await assertThrowsAsync { try await cube.readPattern(row: 2, column: 0) }
        await assertThrowsAsync { try await cube.readPattern(row: 0, column: -1) }
        await assertThrowsAsync { try await cube.readScanTile(rows: 1..<3) }
        await assertThrowsAsync { try await cube.readScanTile(rows: 1..<1) }
        XCTAssertThrowsError(try MemoryCube(shape: shape, values: [1, 2, 3]))
    }

    /// The lazy cube reads the same Float32 values as NumPy's astype(np.float32) of the file, for every fixture.
    func testLazyCubeEqualsTheFileForEveryFixture() async throws {
        for name in ["digits", "gaussian", "counts16", "counts32", "thirds", "negative", "boundary"] {
            let url = Self.fixtures.appendingPathComponent("\(name).npy")
            let lazy = try MappedCube(npy: url)
            let (shape, doubles) = try NPYFile.readDoubles(url)
            XCTAssertEqual(lazy.shape.extents, shape, name)
            let want = doubles.map { Float($0) }       // Float(Double) rounds to nearest even, as astype(np.float32)
            let eager = try await MemoryCube(loading: lazy, tiling: .rows(1))
            XCTAssertEqual(eager.values, want, name)
            let last = try await lazy.readPattern(row: shape[0] - 1, column: shape[1] - 1)
            XCTAssertEqual(last.values, Array(want.suffix(shape[2] * shape[3])), name)
        }
    }

    /// thirds holds k/3 in float64: the lazy read rounds each to the nearest Float32 (not truncates).
    func testFloat64RoundsToNearest() async throws {
        let url = Self.fixtures.appendingPathComponent("thirds.npy")
        let lazy = try MappedCube(npy: url)
        XCTAssertEqual(lazy.elementType, .float64)
        let tile = try await lazy.readScanTile(rows: 0..<2)
        let doubles = try NPYFile.readDoubles(url).values
        for (f, d) in zip(tile.values, doubles) {
            XCTAssertLessThanOrEqual(abs(Double(f) - d), Double(f.ulp) / 2)
        }
        XCTAssertTrue(zip(tile.values, doubles).contains { Double($0) != $1 }, "k/3 must not all be exact in Float32")
    }

    /// Every element type converts as Swift's Float(_:) does (exact up to 2^24, nearest even above).
    func testEveryElementTypeConverts() async throws {
        func check<T: FixedWidthInteger>(_ type: T.Type, _ descr: String, _ raw: [T]) async throws {
            let url = scratch.appendingPathComponent("\(descr.dropFirst()).npy")
            try writeNPY(raw, descr: descr, shape: [1, 1, 1, raw.count], to: url)
            let cube = try MappedCube(npy: url)
            let p = try await cube.readPattern(row: 0, column: 0)
            XCTAssertEqual(p.values, raw.map { Float($0) }, descr)
        }
        try await check(UInt8.self, "|u1", [0, 1, 127, 128, 255])
        try await check(Int8.self, "|i1", [-128, -1, 0, 1, 127])
        try await check(UInt16.self, "<u2", [0, 1, 32767, 32768, 65535])
        try await check(Int16.self, "<i2", [-32768, -1, 0, 1, 32767])
        try await check(UInt32.self, "<u4", [0, 60000, 16_777_216, 16_777_217, 4_294_967_295])
        try await check(Int32.self, "<i4", [-2_147_483_648, -16_777_217, -1, 0, 2_147_483_647])
        try await check(UInt64.self, "<u8", [0, 16_777_217, 1 << 40 + 1, UInt64.max])
        try await check(Int64.self, "<i8", [Int64.min, -(1 << 40) - 1, -1, 0, Int64.max])
        let f: [Float] = [-1.5, 0, .leastNonzeroMagnitude, 3.4e38]
        let urlF = scratch.appendingPathComponent("f4.npy")
        try NPYFile.write(f, shape: [1, 1, 2, 2], to: urlF)
        let pf = try await MappedCube(npy: urlF).readPattern(row: 0, column: 0)
        XCTAssertEqual(pf.values, f)
    }

    // MARK: The .npy container

    func testHeaderOfANumPyFile() throws {
        let h = try NPYFile.header(of: Self.fixtures.appendingPathComponent("gaussian.npy"))
        XCTAssertEqual(h.majorVersion, 1); XCTAssertEqual(h.elementType, .float32)
        XCTAssertEqual(h.shape, [6, 5, 16, 16]); XCTAssertEqual(h.dataOffset % 64, 0)
        let u = try NPYFile.header(of: Self.fixtures.appendingPathComponent("counts16.npy"))
        XCTAssertEqual(u.elementType, .uint16)
    }

    func testWriteRoundTripsAndPadsAsNumPy() async throws {
        let url = scratch.appendingPathComponent("w.npy")
        let values = (0..<(2 * 3 * 4 * 5)).map { Float($0) * 0.25 - 3 }
        try NPYFile.write(values, shape: [2, 3, 4, 5], to: url)
        let h = try NPYFile.header(of: url)
        XCTAssertEqual(h.dataOffset % 64, 0); XCTAssertEqual(h.shape, [2, 3, 4, 5])
        let eager = try await MemoryCube(loading: try MappedCube(npy: url))
        XCTAssertEqual(eager.values, values)
    }

    func testVersion2HeaderIsRead() async throws {
        let url = scratch.appendingPathComponent("v2.npy")
        let dict = "{'descr': '<u2', 'fortran_order': False, 'shape': (1, 2, 1, 3), }"
        var header = dict + String(repeating: " ", count: (64 - (12 + dict.count + 1) % 64) % 64) + "\n"
        header = header.replacingOccurrences(of: "\u{0}", with: "")
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 2, 0])
        let n = UInt32(header.utf8.count)
        data.append(contentsOf: [UInt8(n & 0xFF), UInt8(n >> 8 & 0xFF), UInt8(n >> 16 & 0xFF), UInt8(n >> 24)])
        data.append(contentsOf: Array(header.utf8))
        for v: UInt16 in [1, 2, 3, 40000, 5, 6] { data.append(contentsOf: [UInt8(v & 0xFF), UInt8(v >> 8)]) }
        try data.write(to: url)
        let cube = try MappedCube(npy: url)
        XCTAssertEqual(cube.shape.extents, [1, 2, 1, 3])
        let eager = try await MemoryCube(loading: cube)
        XCTAssertEqual(eager.values, [1, 2, 3, 40000, 5, 6])
    }

    func testRefusals() throws {
        func refuses(_ descr: String, fortran: String = "False", shape: String = "(1, 1, 2, 2)", payload: Int = 16,
                     _ want: CuboError, line: UInt = #line) throws {
            let url = scratch.appendingPathComponent("r\(line).npy")
            try writeRawNPY(dict: "{'descr': '\(descr)', 'fortran_order': \(fortran), 'shape': \(shape), }",
                            payload: payload, to: url)
            XCTAssertThrowsError(try MappedCube(npy: url), line: line) { XCTAssertEqual($0 as? CuboError, want, line: line) }
        }
        try refuses(">f4", .bigEndianNotSupported(">f4"))
        try refuses("<c8", .unsupportedElementType("<c8"))
        try refuses("<f4", fortran: "True", .fortranOrderNotSupported)
        try refuses("<f4", shape: "(2, 2, 2)", .wrongRank(expected: 4, received: 3))
        let short = scratch.appendingPathComponent("r_short.npy")
        try writeRawNPY(dict: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 2, 2), }", payload: 15, to: short)
        XCTAssertThrowsError(try MappedCube(npy: short)) {
            guard case .fileTooShort(_, let needed, let available)? = $0 as? CuboError else { return XCTFail("\($0)") }
            XCTAssertEqual(needed - available, 1)
        }
        let unknown = scratch.appendingPathComponent("r_key.npy")
        try writeRawNPY(dict: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 2, 2), 'x': 1, }", payload: 16, to: unknown)
        XCTAssertThrowsError(try MappedCube(npy: unknown))
        let notNPY = scratch.appendingPathComponent("r_magic.npy")
        try Data("PNG not numpy at all".utf8).write(to: notNPY)
        XCTAssertThrowsError(try MappedCube(npy: notNPY)) {
            guard case .notNPY? = $0 as? CuboError else { return XCTFail("\($0)") }
        }
        XCTAssertThrowsError(try MappedCube(url: Self.fixtures.appendingPathComponent("digits.npy"), dataOffset: 2,
                                            elementType: .float32, shape: try CubeShape([2, 3, 5, 7])))
        XCTAssertThrowsError(try MappedCube(npy: scratch.appendingPathComponent("absent.npy")))
    }

    // MARK: helpers

    func writeRawNPY(dict: String, payload: Int, to url: URL) throws {
        let pad = (64 - (10 + dict.utf8.count + 1) % 64) % 64
        let header = dict + String(repeating: " ", count: pad) + "\n"
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0])
        let n = UInt16(header.utf8.count)
        data.append(contentsOf: [UInt8(n & 0xFF), UInt8(n >> 8)])
        data.append(contentsOf: Array(header.utf8))
        data.append(Data(count: payload))
        try data.write(to: url)
    }

    func writeNPY<T: FixedWidthInteger>(_ raw: [T], descr: String, shape: [Int], to url: URL) throws {
        let dict = "{'descr': '\(descr)', 'fortran_order': False, 'shape': (" + shape.map(String.init)
            .joined(separator: ", ") + "), }"
        try writeRawNPY(dict: dict, payload: 0, to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: raw.map { $0.littleEndian }.withUnsafeBytes { Data($0) })
        try handle.close()
    }

    func assertThrowsAsync(_ body: () async throws -> some Any, line: UInt = #line) async {
        do { _ = try await body(); XCTFail("did not throw", line: line) } catch {}
    }
}
