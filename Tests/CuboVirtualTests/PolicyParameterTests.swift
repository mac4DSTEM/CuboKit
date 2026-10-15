//
//  PolicyParameterTests.swift
//  The generality pass (0.2.0): the Gaussian filter's radius guard, the Friedel upsampling cap and the median's working
//  memory are parameters whose defaults are the old values (10⁶ px, 1024, 256 MiB); the default path is pinned bit for
//  bit by the slice tests (tier 2).
//

import Foundation
import Testing
import CuboCore
import CuboVirtual

@Suite struct VirtualPolicyTests {
    @Test func testGaussianRadiusGuardIsAParameter() throws {
        #expect(try GaussianFilter(sigma: 10).radius == 40)
        #expect(throws: CuboVirtualError.self) { try GaussianFilter(sigma: 10, maximumRadius: 40) }
        #expect(try GaussianFilter(sigma: 10, maximumRadius: 41).kernel == GaussianFilter(sigma: 10).kernel)
        // refuter 0.2.0, F2: a lifted guard trapped (Int overflow) or exhausted memory; bounded at 10⁸ now
        #expect(throws: CuboVirtualError.self) { try GaussianFilter(sigma: 1e9, maximumRadius: .max) }
        #expect(throws: CuboVirtualError.self) { try GaussianFilter(sigma: 1, maximumRadius: 0) }
    }

    @Test func testUpsamplingCapIsAParameter() throws {
        var p = [Double](repeating: 0, count: 64)
        p[3 * 8 + 3] = 10; p[3 * 8 + 4] = 6; p[4 * 8 + 3] = 6; p[4 * 8 + 4] = 10
        let image = DetectorImage(detectorRows: 8, detectorColumns: 8, values: p, path: .cpu)
        #expect(throws: CuboVirtualError.self) { try FriedelOrigin.origin(of: image, upsampleFactor: 1100) }
        let raised = try FriedelOrigin.origin(of: image, upsampleFactor: 1100, maximumUpsampleFactor: 2048)
        #expect(abs(raised.row - 3.5) <= 1e-6 && abs(raised.column - 3.5) <= 1e-6, "\(raised)")
        #expect(throws: CuboVirtualError.self) { try FriedelOrigin.origin(of: image, upsampleFactor: 16, maximumUpsampleFactor: 8) }
        // refuter 0.2.0, F2 and F3: Int.max trapped; a cap below 3 gave a garbled message
        #expect(throws: CuboVirtualError.self) {
            try FriedelOrigin.origin(of: image, upsampleFactor: .max, maximumUpsampleFactor: .max)
        }
        #expect(throws: CuboVirtualError.invalidInput("the maximum upsampling factor must be from 3 to 4096 (got 2)")) {
            try FriedelOrigin.origin(of: image, upsampleFactor: 3, maximumUpsampleFactor: 2)
        }
    }

    @Test func testMedianWorkingMemoryChangesPassesNotNumbers() async throws {
        let cube = try MappedCube(npy: Fixtures.url("gaussian.npy"))
        let byDefault = try await VirtualDiffraction.median(of: cube)
        let small = try await VirtualDiffraction.pattern(of: cube, statistic: .median, medianWorkingBytes: 64)
        #expect(small.values.map(\.bitPattern) == byDefault.values.map(\.bitPattern))
        await #expect(throws: CuboVirtualError.self) {
            try await VirtualDiffraction.pattern(of: cube, statistic: .median, medianWorkingBytes: 0)
        }
    }
}
