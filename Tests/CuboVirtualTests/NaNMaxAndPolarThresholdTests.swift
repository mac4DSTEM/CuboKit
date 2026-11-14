import Foundation
import Testing
import CuboCore
import CuboVirtual

// CuboKit#92 item 5: the CPU maximum pattern is accumulated with `vDSP_vmax`. Measured on macOS 27 (arm64, 14 Nov 2026):
// vDSP_vmax returns NaN when either operand is NaN, in the vector body and the scalar tail, so the CPU path propagates NaN as
// NumPy's `max` (py4DSTEM `get_dp_max`) does. These tests pin that, whatever position the NaN sits at.
// Item 6: a polar mask threshold of 0 or below made empty cells "valid" with the value 0/0; it is refused.

@Suite struct NaNMaxAndPolarThresholdTests {
    @Test func theCPUMaximumPropagatesNaNAtAnyPosition() async throws {
        let shape = try CubeShape(scanRows: 3, scanColumns: 3, detectorRows: 6, detectorColumns: 7)
        for position in [0, 4, 8] {
            for pixel in [0, 17, 41] {   // the first block's start, its body and the last pixel
                var values = (0..<shape.elementCount).map { Float($0 % 13) + 1 }
                values[position * shape.detectorCount + pixel] = .nan
                let cube = try MemoryCube(shape: shape, values: values)
                for tiling in [ScanTiling.rows(1), .automatic] {
                    let max = try await VirtualDiffraction.max(of: cube, tiling: tiling)
                    #expect(max.values[pixel].isNaN, "NaN at position \(position), pixel \(pixel), \(tiling)")
                    #expect(max.values.enumerated().allSatisfy { $0.offset == pixel || $0.element.isFinite })
                }
            }
        }
    }

    /// Mutation: drop the threshold guard - the init accepts 0, red.
    @Test func aNonPositiveMaskThresholdIsRefused() throws {
        for threshold in [0.0, -0.1] {
            #expect(throws: CuboVirtualError.self, "\(threshold)") {
                _ = try PolarCube(detectorRows: 8, detectorColumns: 8, sampling: PolarSampling(maskThreshold: threshold))
            }
        }
        #expect(throws: Never.self) { _ = try PolarCube(detectorRows: 8, detectorColumns: 8, sampling: PolarSampling(maskThreshold: 1e-6)) }
    }
}
