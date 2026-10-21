import CuboCore
import CuboVirtual
import Foundation
import Testing
@testable import CuboBragg

/// Whole-cube detection with failure accounting, progress and cancellation. Peaks of finite patterns must equal the plain
/// `detectAll` exactly (same code path); progress and failure lists are exact.
@Suite struct ScanResultTests {
    static let size = 48

    /// A corner-centred Gaussian kernel.
    static func kernel() -> DetectorImage {
        var v = [Double](repeating: 0, count: size * size)
        for r in 0..<size { for c in 0..<size {
            let dr = Double(min(r, size - r)), dc = Double(min(c, size - c))
            v[r * size + c] = exp(-(dr * dr + dc * dc) / (2 * 2.0 * 2.0))
        } }
        return DetectorImage(detectorRows: size, detectorColumns: size, values: v, path: .cpu)
    }

    /// 4 × 3 scan, disks planted at position-dependent places; pattern `nan` carries a NaN pixel.
    static func cube(nan: Int? = nil) throws -> MemoryCube {
        let rows = 4, cols = 3, n = size * size
        var values = [Float](repeating: 0, count: rows * cols * n)
        for p in 0..<(rows * cols) {
            for (cr, cc) in [(14.0 + Double(p % 5), 16.0), (30.0, 28.0 - Double(p % 4))] {
                for r in 0..<size { for c in 0..<size {
                    let d = (Double(r) - cr) * (Double(r) - cr) + (Double(c) - cc) * (Double(c) - cc)
                    values[p * n + r * size + c] += Float(100 * exp(-d / 8))
                } }
            }
        }
        if let nan { values[nan * n + 5 * size + 5] = .nan }
        return try MemoryCube(shape: CubeShape(scanRows: rows, scanColumns: cols, detectorRows: size, detectorColumns: size), values: values)
    }

    static var options: BraggDiskOptions {
        var o = BraggDiskOptions()
        o.edgeBoundary = 4; o.minimumPeakSpacing = 6; o.subpixel = .parabolic
        return o
    }

    /// `.multicorr` refinement of a NaN correlation is non-finite, so the NaN pattern fails.
    static var multicorr: BraggDiskOptions {
        var o = options; o.subpixel = .multicorr; o.upsampleFactor = 8
        return o
    }

    @Test func healthyScanMatchesPlainDetectAll() async throws {
        let cube = try Self.cube()
        let plain = try await BraggDisks.detectAll(in: cube, kernel: Self.kernel(), options: Self.options, tiling: .rows(1))
        let r = try await BraggDisks.detectAll(in: cube, kernel: Self.kernel(), options: Self.options, tiling: .rows(1),
                                               failurePolicy: .recordEmpty)
        #expect(r.failedPositions.isEmpty && r.firstFailure == nil)
        #expect(r.vectors.peaks == plain.peaks)
        #expect(plain.peaks.allSatisfy { $0.count >= 2 })
    }

    /// A NaN pixel is not a failure in the kit (it propagates to an empty peak list); the failure accounting is exercised
    /// with forced failures at scan positions 4 and 9, across tile boundaries.
    @Test func failurePolicies() async throws {
        let cube = try Self.cube(nan: 7)
        let healthy = try await BraggDisks.detectAll(in: Self.cube(), kernel: Self.kernel(), options: Self.options, tiling: .rows(1))
        let r = try await BraggDisks.scan(in: cube, kernel: Self.kernel(), options: Self.options, tiling: .rows(1),
                                          failurePolicy: .recordEmpty, isCancelled: { false }, progress: { _ in },
                                          forcedFailures: [4, 9])
        #expect(r.failedPositions == [4, 9])
        #expect(r.firstFailure == CuboBraggError.nonFiniteRefinement.localizedDescription)
        for p in 0..<12 where p != 4 && p != 9 && p != 7 { #expect(r.vectors.peaks[p] == healthy.peaks[p]) }
        #expect(r.vectors.peaks[4].isEmpty && r.vectors.peaks[9].isEmpty)
        #expect(r.vectors.peaks[7].isEmpty)   // the NaN pattern propagates to no peaks, not a failure
        await #expect(throws: CuboBraggError.nonFiniteRefinement) {
            _ = try await BraggDisks.scan(in: cube, kernel: Self.kernel(), options: Self.options, tiling: .rows(1),
                                          failurePolicy: .throwOnFirst, isCancelled: { false }, progress: { _ in },
                                          forcedFailures: [9])
        }
    }

    @Test func nanPatternIsEmptyNotFailed() async throws {
        let r = try await BraggDisks.detectAll(in: Self.cube(nan: 4), kernel: Self.kernel(), options: Self.multicorr, tiling: .rows(1),
                                               failurePolicy: .recordEmpty)
        #expect(r.failedPositions.isEmpty && r.vectors.peaks[4].isEmpty)
    }

    @Test func progressIsMonotoneAndEndsAtOne() async throws {
        final class Log: @unchecked Sendable { var v: [Double] = []; let l = NSLock()
            func add(_ x: Double) { l.lock(); v.append(x); l.unlock() } }
        let log = Log()
        _ = try await BraggDisks.detectAll(in: Self.cube(), kernel: Self.kernel(), options: Self.options, tiling: .rows(1),
                                           failurePolicy: .recordEmpty, progress: { log.add($0) })
        #expect(log.v.count == 4)
        #expect(log.v == log.v.sorted())
        #expect(log.v.first! > 0 && log.v.last! == 1)
    }

    @Test func cancellationAndValidation() async throws {
        await #expect(throws: CancellationError.self) {
            _ = try await BraggDisks.detectAll(in: Self.cube(), kernel: Self.kernel(), options: Self.options, tiling: .rows(1),
                                               failurePolicy: .recordEmpty, isCancelled: { true })
        }
        var bad = Self.options
        bad.correlationSigma = -1
        #expect(throws: CuboBraggError.self) { try bad.validate() }
        #expect(throws: Never.self) { try Self.options.validate() }
        await #expect(throws: CuboBraggError.self) {
            _ = try await BraggDisks.detectAll(in: Self.cube(), kernel: Self.kernel(), options: bad, failurePolicy: .recordEmpty)
        }
    }
}
