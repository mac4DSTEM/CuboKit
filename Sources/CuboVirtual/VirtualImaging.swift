//
//  VirtualImaging.swift
//  The public API of CuboVirtual: virtual images, and the mean and maximum pattern, on the path the
//  caller chooses.
//
//  py4DSTEM f050d207: DataCube.get_virtual_image (virtualimage.py 67) with centered=False, calibrated=False,
//  shift_center=False; DataCube.get_dp_mean and get_dp_max (virtualdiffraction.py 337, 308). The kit is held to the
//  float64 evaluation of the same definitions; py4DSTEM's own float32 accumulation is recorded beside it in the pins.
//

import CuboCore

public enum VirtualImaging {
    /// For every scan position, the sum over the detector of pattern × mask, with `geometry`'s mask.
    ///
    /// `progress` receives the fraction of scan rows done after each tile (never decreasing, ending at 1); `isCancelled` is
    /// asked before each tile and throws `CancellationError` when it answers true, returning nothing (as does a cancelled
    /// task). The same two parameters, with the same meaning, end the signatures of the other whole-cube reducers of
    /// ``VirtualDiffraction`` and ``CenterOfMass``.
    public static func image(of source: some CubeSource, geometry: DetectorGeometry, path: ComputePath = .cpu,
                             tiling: ScanTiling = .automatic,
                             progress: (@Sendable (Double) -> Void)? = nil,
                             isCancelled: (@Sendable () -> Bool)? = nil) async throws -> ScanImage {
        let mask = try DetectorMask.make(geometry, shape: source.shape)
        return try await withInterruption(isCancelled) {
            switch path {
            case .cpu: return try await CPUReducers.virtualImage(of: source, mask: mask, tiling: tiling, progress: progress)
            case .metal: return try await MetalReducers.virtualImage(of: source, mask: mask, tiling: tiling, progress: progress)
            }
        }
    }
}

public enum VirtualDiffraction {
    /// The mean pattern over every scan position (py4DSTEM's `get_dp_mean`).
    public static func mean(of source: some CubeSource, path: ComputePath = .cpu, tiling: ScanTiling = .automatic,
                            progress: (@Sendable (Double) -> Void)? = nil,
                            isCancelled: (@Sendable () -> Bool)? = nil) async throws -> DetectorImage {
        try await withInterruption(isCancelled) {
            if path == .cpu { return try await CPUReducers.meanAndMax(of: source, tiling: tiling, wantMax: false, progress: progress).0 }
            return try await meanAndMax(of: source, path: path, tiling: tiling, progress: progress).mean
        }
    }

    /// The maximum pattern over every scan position (py4DSTEM's `get_dp_max`); exact on every path.
    public static func max(of source: some CubeSource, path: ComputePath = .cpu, tiling: ScanTiling = .automatic,
                           progress: (@Sendable (Double) -> Void)? = nil,
                           isCancelled: (@Sendable () -> Bool)? = nil) async throws -> DetectorImage {
        try await withInterruption(isCancelled) {
            if path == .cpu { return try await CPUReducers.meanAndMax(of: source, tiling: tiling, wantMean: false, progress: progress).1 }
            return try await meanAndMax(of: source, path: path, tiling: tiling, progress: progress).max
        }
    }

    /// Both patterns in one pass over the cube.
    public static func meanAndMax(of source: some CubeSource, path: ComputePath = .cpu,
                                  tiling: ScanTiling = .automatic, progress: (@Sendable (Double) -> Void)? = nil,
                                  isCancelled: (@Sendable () -> Bool)? = nil)
        async throws -> (mean: DetectorImage, max: DetectorImage) {
        try await withInterruption(isCancelled) {
            switch path {
            case .cpu: return try await CPUReducers.meanAndMax(of: source, tiling: tiling, progress: progress)
            case .metal: return try await MetalReducers.meanAndMax(of: source, tiling: tiling, progress: progress)
            }
        }
    }

    /// Whether this machine can run the `.metal` path.
    public static var metalIsAvailable: Bool {
        if case .success = MetalContext.shared { return true }
        return false
    }
}
