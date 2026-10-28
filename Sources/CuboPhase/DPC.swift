//
//  DPC.swift
//  Differential phase contrast by py4DSTEM's iterative reconstruction: `DPC.preprocess`'s
//  operators and `DPC.reconstruct` with `_forward`, `_adjoint`, `_update` (process/phase/dpc.py at f050d207), literally,
//  in Double, with its default constraints (no Gaussian and no Butterworth filter). py4DSTEM computes this in float32;
//  the kit is held to the float64 evaluation of the same steps.
//

import CuboCore
import Foundation

public struct DPCReconstruction: Sendable, Equatable {
    public let scanRows: Int
    public let scanColumns: Int
    /// The reconstructed phase on the scan grid, row-major (the scan region of the padded object).
    public let phase: [Double]
    /// The error of every accepted iteration, in order (py4DSTEM's `error_iterations`).
    public let errors: [Double]
    /// Per iteration run, whether it was accepted (false: backtracked, the step halved).
    public let accepted: [Bool]
    public let finalStepSize: Double
}

public enum DPC {
    /// py4DSTEM's `reconstruct` from the corrected centre-of-mass field (`RotationSolution.corrected`).
    public static func reconstruct(_ field: ScanVectorField, scanSampling: (row: Double, column: Double) = (1, 1),
                                   paddingFactor: Double = 2, maxIterations: Int = 64, stepSize: Double = 0.5,
                                   stoppingCriterion: Double = 1e-6, backtrack: Bool = true) throws -> DPCReconstruction {
        let R = field.scanRows, C = field.scanColumns
        guard scanSampling.row > 0, scanSampling.column > 0, scanSampling.row.isFinite, scanSampling.column.isFinite else {
            throw CuboPhaseError.invalidParameter("scan sampling must be positive and finite")
        }
        guard paddingFactor >= 1, paddingFactor.isFinite, maxIterations >= 0, stepSize > 0, stepSize.isFinite else {
            throw CuboPhaseError.invalidParameter("padding ≥ 1, iterations ≥ 0, a positive step")
        }
        guard field.row.allSatisfy(\.isFinite), field.column.allSatisfy(\.isFinite) else {
            throw CuboPhaseError.invalidField("DPC needs a finite field")
        }
        let denominator = NumPySum.mean(zip(field.row, field.column).map { $0 * $0 + $1 * $1 })
        guard denominator > 0 else { throw CuboPhaseError.zeroField }
        // The padded grid: np.round(shape · padding).astype(int), half to even.
        let Rp = Int((Double(R) * paddingFactor).rounded(.toNearestOrEven))
        let Cp = Int((Double(C) * paddingFactor).rounded(.toNearestOrEven))
        let kx = fftfreq(Rp, scanSampling.row), ky = fftfreq(Cp, scanSampling.column)
        // k_den = 1 / (kx² + ky²), 0 at the origin (1/∞); kx_op = −0.25 i kx k_den, so FFT·op = (−b·g, a·g), g = −0.25 kx k_den.
        var gx = [Double](repeating: 0, count: Rp * Cp), gy = gx
        for r in 0..<Rp { for c in 0..<Cp {
            var den = kx[r] * kx[r] + ky[c] * ky[c]
            if r == 0 && c == 0 { den = .infinity }
            let kden = 1 / den
            gx[r * Cp + c] = -0.25 * kx[r] * kden
            gy[r * Cp + c] = -0.25 * ky[c] * kden
        } }
        let fft = FFT2D(rows: Rp, columns: Cp)
        var phi = [Double](repeating: 0, count: Rp * Cp)
        var step = stepSize, error = Double.infinity
        var errors = [Double](), accepted = [Bool]()
        var maskValues = [Double](repeating: 0, count: R * C)
        for _ in 0..<maxIterations {
            if step < stoppingCriterion { break }
            // _forward: obj_dx = (roll(φ, 1, 0) − roll(φ, −1, 0)) / (2 dx), plus the field on the scan region, 0 elsewhere.
            var dx = [Double](repeating: 0, count: Rp * Cp), dy = dx
            for r in 0..<R { for c in 0..<C {
                let up = phi[((r - 1 + Rp) % Rp) * Cp + c], down = phi[((r + 1) % Rp) * Cp + c]
                let left = phi[r * Cp + (c - 1 + Cp) % Cp], right = phi[r * Cp + (c + 1) % Cp]
                dx[r * Cp + c] = (up - down) / (2 * scanSampling.row) + field.row[r * C + c]
                dy[r * Cp + c] = (left - right) / (2 * scanSampling.column) + field.column[r * C + c]
                maskValues[r * C + c] = dx[r * Cp + c] * dx[r * Cp + c] + dy[r * Cp + c] * dy[r * Cp + c]
            } }
            let newError = NumPySum.mean(maskValues) / denominator
            if newError > error && backtrack {
                step /= 2
                accepted.append(false)
                continue
            }
            error = newError
            // _adjoint: Re(ifft2(fft2(dx) · kx_op + fft2(dy) · ky_op)).
            var xi = [Double](repeating: 0, count: Rp * Cp), yi = xi
            fft.transform(&dx, &xi, inverse: false)
            fft.transform(&dy, &yi, inverse: false)
            var ur = [Double](repeating: 0, count: Rp * Cp), ui = ur
            for k in 0..<(Rp * Cp) {
                ur[k] = -xi[k] * gx[k] + -yi[k] * gy[k]
                ui[k] = dx[k] * gx[k] + dy[k] * gy[k]
            }
            fft.transform(&ur, &ui, inverse: true)
            for k in 0..<(Rp * Cp) { phi[k] += step * ur[k] }
            errors.append(error)
            accepted.append(true)
        }
        var phase = [Double](repeating: 0, count: R * C)
        for r in 0..<R { for c in 0..<C { phase[r * C + c] = phi[r * Cp + c] } }
        return DPCReconstruction(scanRows: R, scanColumns: C, phase: phase, errors: errors, accepted: accepted,
                                 finalStepSize: step)
    }

    /// NumPy's fftfreq(n, d): k · (1 / (n · d)), k = 0, 1, …, then the negative frequencies.
    static func fftfreq(_ n: Int, _ d: Double) -> [Double] {
        let scale = 1.0 / (Double(n) * d)
        let positive = (n - 1) / 2 + 1
        return (0..<n).map { Double($0 < positive ? $0 : $0 - n) * scale }
    }
}
