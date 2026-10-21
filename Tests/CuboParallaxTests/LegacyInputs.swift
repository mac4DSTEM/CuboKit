import Foundation
import CuboCore
@testable import CuboParallax

/// Deterministic inputs shared by the 0.5.0 golden generator and the default-unchanged tests.
enum LegacyInputs {
    static func values(_ count: Int, seed: UInt64, low: Double = 0.5, high: Double = 2) -> [Double] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return low + (high - low) * Double(state >> 11) / Double(1 << 53)
        }
    }
    static func images(_ n: Int, rows: Int, columns: Int, seed: UInt64) -> [[Double]] {
        (0..<n).map { values(rows * columns, seed: seed &+ UInt64($0) &* 7919) }
    }
    static let terms = [ParallaxAberrationTerm(radialOrder: 1, angularOrder: 0, component: 0),
                        ParallaxAberrationTerm(radialOrder: 1, angularOrder: 2, component: 0),
                        ParallaxAberrationTerm(radialOrder: 1, angularOrder: 2, component: 1),
                        ParallaxAberrationTerm(radialOrder: 2, angularOrder: 1, component: 0)]
    static let coefficients = [80.0, 25, -15, 4000]
    static let angles = [ParallaxShift(row: 0.004, column: -0.002), ParallaxShift(row: -0.003, column: 0.005),
                         ParallaxShift(row: 0.001, column: 0.0015)]
}
