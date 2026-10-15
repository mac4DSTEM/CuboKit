//
//  ParallelScores.swift
//  The rotation search's angles evaluated on every core: each angle is independent and written to its own slot, so the
//  result is bit identical to the sequential map. (CuboPhase does not depend on CuboVirtual, which has the same idea.)
//

import Foundation

/// A pointer handed to concurrent workers that each touch only their own elements.
struct SharedScoreBuffer<T>: @unchecked Sendable {
    let base: UnsafeMutablePointer<T>
}

/// `(0..<count).map(body)` on every core (Dispatch's concurrentPerform). `isCancelled` is asked before each element and
/// once more at the end; when it has returned true, nothing is returned and `CancellationError` is thrown. `body` must
/// not share mutable state.
func parallelScores(_ count: Int, isCancelled: (@Sendable () -> Bool)?, _ body: @Sendable (Int) -> Double) throws -> [Double] {
    if isCancelled?() == true { throw CancellationError() }
    guard count > 0 else { return [] }
    var out = [Double](repeating: .nan, count: count)
    var skipped = [UInt8](repeating: 0, count: count)
    out.withUnsafeMutableBufferPointer { o in
        skipped.withUnsafeMutableBufferPointer { s in
            let values = SharedScoreBuffer(base: o.baseAddress!), flags = SharedScoreBuffer(base: s.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: count) { k in
                if isCancelled?() == true { (flags.base + k).pointee = 1; return }
                (values.base + k).pointee = body(k)
            }
        }
    }
    if skipped.contains(1) || isCancelled?() == true { throw CancellationError() }
    return out
}
