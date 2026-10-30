import Foundation

// PCA, k-means and the diffraction-pattern featurisation behind them. The primary entry points follow scikit-learn 1.4.2;
// the numerics of mac4DSTEM's diffraction-embedding pipeline are separate, explicitly named options.

/// Why an embedding computation was refused.
public enum EmbeddingError: Error, Sendable, Equatable, CustomStringConvertible {
    /// The data length does not equal rows × columns, a dimension is zero, or too few rows were given.
    case invalidShape(String)
    /// The requested component or cluster count is outside what the chosen variant accepts.
    case invalidCount(String)
    /// A value is NaN or infinite.
    case nonFinite
    /// LAPACK reported a failure (its `info` value).
    case solverFailed(Int)

    public var description: String {
        switch self {
        case .invalidShape(let m): return "Invalid shape: \(m)"
        case .invalidCount(let m): return "Invalid count: \(m)"
        case .nonFinite: return "The data contain non-finite values."
        case .solverFailed(let info): return "The eigensolver failed (LAPACK info \(info))."
        }
    }
}
