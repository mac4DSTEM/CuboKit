import Accelerate
import CuboCore
@testable import CuboVirtual
import Foundation
import Testing

/// Where the CPU mean and maximum pattern spend their time (CuboKit#7). Skipped unless `CUBOKIT_MEANMAX_BENCH` is set:
/// `CUBOKIT_MEANMAX_BENCH=1 swift test -c release --filter MeanMaxBench`. On a resident 100 x 100 x 64 x 64 Float32 cube it
/// prints the median and best of 15 calls of the public CPU reducers and of loops that isolate one cost each: reading the
/// cube once (the bandwidth floor), widening plus a plain Double add, the compensated (TwoSum) add, block size (how many
/// cores take part) and one thread. The "4 rows/pass" lines run ``CompensatedSum/addRows``, what the reducers use since #7.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["CUBOKIT_MEANMAX_BENCH"] != nil, "set CUBOKIT_MEANMAX_BENCH"))
struct MeanMaxBench {
    private func time(_ calls: Int = 15, _ body: () async throws -> Void) async rethrows -> (Double, Double) {
        try await body()
        var ts = [Double]()
        for _ in 0..<calls {
            let t0 = DispatchTime.now().uptimeNanoseconds
            try await body()
            ts.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        ts.sort()
        return (ts[ts.count / 2], ts[0])
    }

    private func time(_ calls: Int = 15, _ body: () -> Void) -> (Double, Double) {
        body()
        var ts = [Double]()
        for _ in 0..<calls {
            let t0 = DispatchTime.now().uptimeNanoseconds
            body()
            ts.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        ts.sort()
        return (ts[ts.count / 2], ts[0])
    }

    private func report(_ name: String, _ t: (Double, Double), bytes: Int) {
        print(String(format: "MEANMAXBENCH %-28@ median %6.2f ms  best %6.2f ms  %5.1f GB/s", name as NSString, t.0, t.1,
                     Double(bytes) / (t.0 * 1e6)))
    }

    @Test func typicalCube() async throws {
        let shape = try CubeShape(scanRows: 100, scanColumns: 100, detectorRows: 64, detectorColumns: 64)
        var state: UInt64 = 7
        let values = (0..<shape.elementCount).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / 1024
        }
        let cube = try MemoryCube(shape: shape, values: values)
        let n = shape.detectorCount, positions = shape.scanCount, bytes = shape.elementCount * 4

        report("api mean", try await time { _ = try await VirtualDiffraction.mean(of: cube) }, bytes: bytes)
        report("api max", try await time { _ = try await VirtualDiffraction.max(of: cube) }, bytes: bytes)
        report("api meanAndMax", try await time { _ = try await VirtualDiffraction.meanAndMax(of: cube) }, bytes: bytes)

        values.withUnsafeBufferPointer { vb in
            let v = SharedBuffer(base: UnsafeMutablePointer(mutating: vb.baseAddress!))
            // the floor: every element read once, contiguous chunks of positions on every core (vDSP_sve, Float)
            for chunks in [1, 6, 18] {
                report("read once sve x\(chunks)", time {
                    let per = (positions + chunks - 1) / chunks
                    DispatchQueue.concurrentPerform(iterations: chunks) { c in
                        let lo = c * per, hi = Swift.min(positions, lo + per)
                        guard lo < hi else { return }
                        var s: Float = 0
                        vDSP_sve(v.base + lo * n, 1, &s, vDSP_Length((hi - lo) * n))
                        blackHole(Double(s))
                    }
                }, bytes: bytes)
            }
            for blockSize in [128, 256, 512, 1024, 4096] {
                let blocks = (n + blockSize - 1) / blockSize
                var sum = [Double](repeating: 0, count: n), corr = sum
                var mx = [Float](repeating: -.infinity, count: n)
                sum.withUnsafeMutableBufferPointer { sb in corr.withUnsafeMutableBufferPointer { cb in
                mx.withUnsafeMutableBufferPointer { mb in
                    let s = SharedBuffer(base: sb.baseAddress!), c = SharedBuffer(base: cb.baseAddress!)
                    let m = SharedBuffer(base: mb.baseAddress!)
                    report("plain Double add bs\(blockSize)", time {
                        DispatchQueue.concurrentPerform(iterations: blocks) { b in
                            let lo = b * blockSize, count = Swift.min(n, lo + blockSize) - lo
                            for k in 0..<positions { plainAdd(v.base + k * n + lo, s.base + lo, count) }
                        }
                    }, bytes: bytes)
                    report("TwoSum bs\(blockSize)", time {
                        DispatchQueue.concurrentPerform(iterations: blocks) { b in
                            let lo = b * blockSize, count = Swift.min(n, lo + blockSize) - lo
                            for k in 0..<positions {
                                CompensatedSum.add(v.base + k * n + lo, weight: nil, sum: s.base + lo, correction: c.base + lo, count: count)
                            }
                        }
                    }, bytes: bytes)
                    report("TwoSum 4 rows/pass bs\(blockSize)", time {
                        DispatchQueue.concurrentPerform(iterations: blocks) { b in
                            let lo = b * blockSize, count = Swift.min(n, lo + blockSize) - lo
                            CompensatedSum.addRows(v.base + lo, stride: n, rows: positions, sum: s.base + lo,
                                                   correction: c.base + lo, count: count)
                        }
                    }, bytes: bytes)
                    report("vmax bs\(blockSize)", time {
                        DispatchQueue.concurrentPerform(iterations: blocks) { b in
                            let lo = b * blockSize, count = Swift.min(n, lo + blockSize) - lo
                            for k in 0..<positions {
                                vDSP_vmax(m.base + lo, 1, v.base + k * n + lo, 1, m.base + lo, 1, vDSP_Length(count))
                            }
                        }
                    }, bytes: bytes)
                    if blockSize == 512 {
                        report("TwoSum bs512 one thread", time {
                            for b in 0..<blocks {
                                let lo = b * blockSize, count = Swift.min(n, lo + blockSize) - lo
                                for k in 0..<positions {
                                    CompensatedSum.add(v.base + k * n + lo, weight: nil, sum: s.base + lo, correction: c.base + lo, count: count)
                                }
                            }
                        }, bytes: bytes)
                    }
                } } }
            }
        }
    }

    @inline(never) private func blackHole(_ x: Double) { if x == -42.4242 { print(x) } }

    @inline(__always) private func plainAdd(_ x: UnsafePointer<Float>, _ s: UnsafeMutablePointer<Double>, _ count: Int) {
        let rx = UnsafeRawPointer(x), rs = UnsafeMutableRawPointer(s)
        var i = 0
        while i + 4 <= count {
            let a = rs.loadUnaligned(fromByteOffset: i * 8, as: SIMD4<Double>.self)
            rs.storeBytes(of: a + SIMD4<Double>(rx.loadUnaligned(fromByteOffset: i * 4, as: SIMD4<Float>.self)),
                          toByteOffset: i * 8, as: SIMD4<Double>.self)
            i += 4
        }
    }
}
