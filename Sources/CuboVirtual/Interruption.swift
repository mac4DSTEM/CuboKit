//
//  Interruption.swift
//  How the long reducers learn that the caller wants to stop. The public entry points take `isCancelled:` (a closure, as
//  ``RotationSolver/curve(_:anglesDegrees:criterion:transpose:isCancelled:)`` does) and hand it to the passes below them
//  through a task-local value, so every pass asks it before each tile without a parameter on every internal function.
//

import Foundation

enum Interruption {
    @TaskLocal static var isCancelled: (@Sendable () -> Bool)?
}

/// Throws `CancellationError` when the task is cancelled or the caller's `isCancelled` answers true (asked before each tile).
func checkInterrupted() throws {
    try Task.checkCancellation()
    if Interruption.isCancelled?() == true { throw CancellationError() }
}

/// Runs `body` with `isCancelled` consulted by every pass inside it; an already cancelled caller throws before any work.
func withInterruption<T>(_ isCancelled: (@Sendable () -> Bool)?, _ body: () async throws -> T) async throws -> T {
    guard let isCancelled else { return try await body() }
    if isCancelled() { throw CancellationError() }
    return try await Interruption.$isCancelled.withValue(isCancelled) { try await body() }
}
