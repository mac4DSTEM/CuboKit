//
//  CuboCoreTests.swift
//  The cube model (registration 1.1, 3.5, 5.5): shapes, tilings, the eager and lazy cubes, the .npy container and the
//  element conversions.
//

import Foundation
import Testing
import CuboCore

/// A final class for its deinit: each test gets its own scratch directory, removed afterwards.
@Suite final class CuboCoreTests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("CuboVirtualTests/Fixtures")
    let scratch: URL

    init() throws {
        scratch = FileManager.default.temporaryDirectory.appendingPathComponent("CuboCoreTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    deinit { try? FileManager.default.removeItem(at: scratch) }

    // MARK: Shapes and tilings

    @Test func testShapeValidatesAndCounts() throws {
        let s = try CubeShape([2, 3, 5, 7])
        #expect(s.scanCount == 6); #expect(s.detectorCount == 35); #expect(s.elementCount == 210)
        #expect(s.elementsPerScanRow == 105); #expect(s.scanIndex(row: 1, column: 2) == 5)
        #expect(throws: CuboError.invalidShape([2, 0, 5, 7])) { try CubeShape([2, 0, 5, 7]) }
        #expect(throws: CuboError.invalidShape([2, 3, 5])) { try CubeShape([2, 3, 5]) }
        // Two negative extents multiply to a positive count: only the per-extent guard refuses them (mutant M20).
        #expect(throws: CuboError.invalidShape([-2, -3, 5, 7])) { try CubeShape([-2, -3, 5, 7]) }
        #expect(throws: CuboError.invalidShape([Int.max / 2, 4, 4, 4])) { try CubeShape([Int.max / 2, 4, 4, 4]) }
        let decoded = try JSONDecoder().decode(CubeShape.self, from: JSONEncoder().encode(s))
        #expect(decoded == s)
        #expect(throws: CuboError.invalidShape([0, 1, 1, 1])) {
            try JSONDecoder().decode(CubeShape.self, from: Data(
                #"{"scanRows":0,"scanColumns":1,"detectorRows":1,"detectorColumns":1}"#.utf8))
        }
    }

    /// `.automatic` is a function of the shape alone: the largest row count whose Float32 tile is at most 64 MiB.
    @Test func testAutomaticTilingIsAFunctionOfTheShape() throws {
        let almgsi = try CubeShape([330, 330, 64, 64])          // 5 406 720 bytes per scan row
        #expect(try ScanTiling.automatic.rowsPerTile(for: almgsi) == 12)
        let big = try CubeShape([20, 20, 512, 512])             // 20 971 520 bytes per scan row
        #expect(try ScanTiling.automatic.rowsPerTile(for: big) == 3)
        let huge = try CubeShape([4, 64, 512, 512])             // one row is over 64 MiB
        #expect(try ScanTiling.automatic.rowsPerTile(for: huge) == 1)
        let small = try CubeShape([2, 3, 5, 7])
        #expect(try ScanTiling.automatic.rowsPerTile(for: small) == 2)
        #expect(try ScanTiling.rows(4).ranges(for: try CubeShape([10, 1, 1, 1])) == [0..<4, 4..<8, 8..<10])
        #expect(try ScanTiling.rows(99).rowsPerTile(for: small) == 2)
        #expect(throws: CuboError.invalidTiling(0)) { try ScanTiling.rows(0).rowsPerTile(for: small) }
    }

    // MARK: Eager and lazy

    @Test func testEagerCubeReadsAndRefuses() async throws {
        let shape = try CubeShape([2, 3, 5, 7])
        let values = (0..<shape.elementCount).map(Float.init)
        let cube = try MemoryCube(shape: shape, values: values)
        let p = try await cube.readPattern(row: 1, column: 2)
        #expect(p.values == Array(values[175..<210]))
        #expect(p[4, 6] == 209)
        let t = try await cube.readScanTile(rows: 1..<2)
        #expect(t.values == Array(values[105..<210])); #expect(t.positionCount == 3)
        await #expect(throws: CuboError.self) { try await cube.readPattern(row: 2, column: 0) }
        await #expect(throws: CuboError.self) { try await cube.readPattern(row: 0, column: -1) }
        await #expect(throws: CuboError.self) { try await cube.readScanTile(rows: 1..<3) }
        await #expect(throws: CuboError.self) { try await cube.readScanTile(rows: 1..<1) }
        #expect(throws: CuboError.valueCountMismatch(expected: 210, received: 3)) {
            try MemoryCube(shape: shape, values: [1, 2, 3])
        }
    }

    /// The lazy cube reads the same Float32 values as NumPy's astype(np.float32) of the file, for every fixture.
    @Test func testLazyCubeEqualsTheFileForEveryFixture() async throws {
        for name in ["digits", "gaussian", "counts16", "counts32", "thirds", "negative", "boundary"] {
            let url = Self.fixtures.appendingPathComponent("\(name).npy")
            let lazy = try MappedCube(npy: url)
            let (shape, doubles) = try NPYFile.readDoubles(url)
            #expect(lazy.shape.extents == shape, "\(name)")
            let want = doubles.map { Float($0) }       // Float(Double) rounds to nearest even, as astype(np.float32)
            let eager = try await MemoryCube(loading: lazy, tiling: .rows(1))
            #expect(eager.values == want, "\(name)")
            let last = try await lazy.readPattern(row: shape[0] - 1, column: shape[1] - 1)
            #expect(last.values == Array(want.suffix(shape[2] * shape[3])), "\(name)")
        }
    }

    /// thirds holds k/3 in float64: the lazy read rounds each to the nearest Float32 (not truncates).
    @Test func testFloat64RoundsToNearest() async throws {
        let url = Self.fixtures.appendingPathComponent("thirds.npy")
        let lazy = try MappedCube(npy: url)
        #expect(lazy.elementType == .float64)
        let tile = try await lazy.readScanTile(rows: 0..<2)
        let doubles = try NPYFile.readDoubles(url).values
        for (f, d) in zip(tile.values, doubles) {
            #expect(abs(Double(f) - d) <= Double(f.ulp) / 2)
        }
        #expect(zip(tile.values, doubles).contains { Double($0) != $1 }, "k/3 must not all be exact in Float32")
    }

    /// Every element type converts as Swift's Float(_:) does (exact up to 2^24, nearest even above).
    @Test func testEveryElementTypeConverts() async throws {
        func check<T: FixedWidthInteger>(_ type: T.Type, _ descr: String, _ raw: [T]) async throws {
            let url = scratch.appendingPathComponent("\(descr.dropFirst()).npy")
            try writeNPY(raw, descr: descr, shape: [1, 1, 1, raw.count], to: url)
            let cube = try MappedCube(npy: url)
            let p = try await cube.readPattern(row: 0, column: 0)
            #expect(p.values == raw.map { Float($0) }, "\(descr)")
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
        #expect(pf.values == f)
    }

    /// An adapter that forwards to another asynchronous store conforms with `async` reads, and the kit reads through it
    /// the same values (the pattern mac4DSTEM's switch uses: its own array behind `CubeSource`; consumers-0.0.1.md).
    @Test func testAnAsyncAdapterIsACubeSource() async throws {
        actor Forwarding: CubeSource {
            let inner: MappedCube
            nonisolated let shape: CubeShape
            init(_ inner: MappedCube) { self.inner = inner; shape = inner.shape }
            func readPattern(row: Int, column: Int) async throws -> DiffractionPattern {
                try await inner.readPattern(row: row, column: column)
            }
            func readScanTile(rows: Range<Int>) async throws -> ScanTile { try await inner.readScanTile(rows: rows) }
        }
        let lazy = try MappedCube(npy: Self.fixtures.appendingPathComponent("gaussian.npy"))
        let adapter = Forwarding(lazy)
        let viaAdapter = try await MemoryCube(loading: adapter, tiling: .rows(2))
        let direct = try await MemoryCube(loading: lazy)
        #expect(viaAdapter.values == direct.values)
        #expect(try await adapter.readPattern(row: 5, column: 4) == (try await lazy.readPattern(row: 5, column: 4)))
    }

    // MARK: The .npy container

    @Test func testHeaderOfANumPyFile() throws {
        let h = try NPYFile.header(of: Self.fixtures.appendingPathComponent("gaussian.npy"))
        #expect(h.majorVersion == 1); #expect(h.elementType == .float32)
        #expect(h.shape == [6, 5, 16, 16]); #expect(h.dataOffset % 64 == 0)
        let u = try NPYFile.header(of: Self.fixtures.appendingPathComponent("counts16.npy"))
        #expect(u.elementType == .uint16)
    }

    @Test func testWriteRoundTripsAndPadsAsNumPy() async throws {
        let url = scratch.appendingPathComponent("w.npy")
        let values = (0..<(2 * 3 * 4 * 5)).map { Float($0) * 0.25 - 3 }
        try NPYFile.write(values, shape: [2, 3, 4, 5], to: url)
        let h = try NPYFile.header(of: url)
        #expect(h.dataOffset % 64 == 0); #expect(h.shape == [2, 3, 4, 5])
        let eager = try await MemoryCube(loading: try MappedCube(npy: url))
        #expect(eager.values == values)
    }

    @Test func testVersion2HeaderIsRead() async throws {
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
        #expect(cube.shape.extents == [1, 2, 1, 3])
        let eager = try await MemoryCube(loading: cube)
        #expect(eager.values == [1, 2, 3, 40000, 5, 6])

        // A version 2.0 header longer than 64 KiB (what version 2.0 exists for): its third length byte is not zero, so a
        // reader that drops or swaps the high bytes reads the data from the wrong place (mutant M14b, 2026-10-15).
        let long = scratch.appendingPathComponent("v2-long.npy")
        var longHeader = dict + String(repeating: " ", count: 70_000)
        longHeader += String(repeating: " ", count: (64 - (12 + longHeader.utf8.count + 1) % 64) % 64) + "\n"
        var longData = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 2, 0])
        let m = UInt32(longHeader.utf8.count)
        #expect(m > 65_535)
        longData.append(contentsOf: [UInt8(m & 0xFF), UInt8(m >> 8 & 0xFF), UInt8(m >> 16 & 0xFF), UInt8(m >> 24)])
        longData.append(contentsOf: Array(longHeader.utf8))
        for v: UInt16 in [7, 8, 9, 50000, 11, 12] { longData.append(contentsOf: [UInt8(v & 0xFF), UInt8(v >> 8)]) }
        try longData.write(to: long)
        let longHeaderRead = try NPYFile.header(of: long)
        #expect(longHeaderRead.dataOffset == 12 + Int(m))
        let longEager = try await MemoryCube(loading: try MappedCube(npy: long))
        #expect(longEager.values == [7, 8, 9, 50000, 11, 12])
    }

    @Test func testRefusals() throws {
        func refuses(_ descr: String, fortran: String = "False", shape: String = "(1, 1, 2, 2)", payload: Int = 16,
                     _ want: CuboError,
                     sourceLocation: SourceLocation = SourceLocation(fileID: #fileID, filePath: #filePath,
                                                                     line: #line, column: #column)) throws {
            let url = scratch.appendingPathComponent("r\(sourceLocation.line).npy")
            try writeRawNPY(dict: "{'descr': '\(descr)', 'fortran_order': \(fortran), 'shape': \(shape), }",
                            payload: payload, to: url)
            #expect(throws: want, sourceLocation: sourceLocation) { try MappedCube(npy: url) }
        }
        try refuses(">f4", .bigEndianNotSupported(">f4"))
        try refuses("<c8", .unsupportedElementType("<c8"))
        try refuses("<f4", fortran: "True", .fortranOrderNotSupported)
        try refuses("<f4", shape: "(2, 2, 2)", .wrongRank(expected: 4, received: 3))
        let short = scratch.appendingPathComponent("r_short.npy")
        try writeRawNPY(dict: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 2, 2), }", payload: 15, to: short)
        let tooShort = #expect(throws: CuboError.self) { try MappedCube(npy: short) }
        guard case .fileTooShort(_, let needed, let available)? = tooShort else {
            Issue.record("expected fileTooShort, got \(String(describing: tooShort))"); return
        }
        #expect(needed - available == 1)
        let unknown = scratch.appendingPathComponent("r_key.npy")
        try writeRawNPY(dict: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 2, 2), 'x': 1, }", payload: 16, to: unknown)
        #expect(throws: CuboError.notNPY("unknown key 'x'")) { try MappedCube(npy: unknown) }
        let notNPY = scratch.appendingPathComponent("r_magic.npy")
        try Data("PNG not numpy at all".utf8).write(to: notNPY)
        #expect(throws: CuboError.notNPY("no NUMPY magic")) { try MappedCube(npy: notNPY) }
        #expect(throws: CuboError.misalignedOffset(2, elementSize: 4)) {
            try MappedCube(url: Self.fixtures.appendingPathComponent("digits.npy"), dataOffset: 2,
                           elementType: .float32, shape: try CubeShape([2, 3, 5, 7]))
        }
        let absent = #expect(throws: CuboError.self) { try MappedCube(npy: self.scratch.appendingPathComponent("absent.npy")) }
        if case .cannotOpen? = absent {} else { Issue.record("expected cannotOpen, got \(String(describing: absent))") }
    }

    /// Gate B F2 (2026-10-15): a header whose element or byte count overflows is refused, never trapped on.
    @Test(arguments: ["(4294967296, 4294967296, 4294967296, 4294967296)", "(9223372036854775807, 2)",
                      "(3000000000, 3000000000)", "(1152921504606846976,)"])
    func testOverflowingShapesAreRefused(_ shape: String) throws {
        let url = scratch.appendingPathComponent("overflow.npy")
        try writeRawNPY(dict: "{'descr': '<f8', 'fortran_order': False, 'shape': \(shape), }", payload: 64, to: url)
        #expect(throws: CuboError.self) { try NPYFile.readDoubles(url) }
        #expect(throws: CuboError.self) { try NPYFile.header(of: url) }
        #expect(throws: CuboError.self) { try MappedCube(npy: url) }
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

}
