/// Throws `CancellationError` when the caller's cancellation closure reports true.
@inline(__always)
package func parallaxCheckCancelled(_ isCancelled: (() -> Bool)?) throws {
    if isCancelled?() == true { throw CancellationError() }
}
