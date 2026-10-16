import Foundation
import CuboCore

/// Boundary assumptions for Fourier integration of a scan-frame vector field.
public enum IDPCBoundary: Sendable, Equatable {
    /// Solve on the native scan grid, with opposite edges adjacent.
    case periodic
    /// Center the field in a zero-valued grid, enlarged by an integer factor.
    case zeroPadded(factor: Int)
}

/// Physical calibration converting detector-pixel CoM shifts to projected phase.
public struct IDPCPhysicalCalibration: Sendable, Equatable {
    /// Scan sampling in Å per scan pixel.
    public let rowSamplingAngstrom: Double, columnSamplingAngstrom: Double
    /// Reciprocal sampling in Å⁻¹ per detector pixel.
    public let reciprocalAngstromPerDetectorPixel: Double
    /// Requires finite positive sampling on all three axes.
    public init(rowSamplingAngstrom: Double, columnSamplingAngstrom: Double,
                reciprocalAngstromPerDetectorPixel: Double) throws {
        guard [rowSamplingAngstrom,columnSamplingAngstrom,reciprocalAngstromPerDetectorPixel].allSatisfy({$0.isFinite && $0 > 0}) else {
            throw CuboPhaseError.invalidParameter("iDPC calibration must be positive and finite")
        }
        self.rowSamplingAngstrom = rowSamplingAngstrom; self.columnSamplingAngstrom = columnSamplingAngstrom
        self.reciprocalAngstromPerDetectorPixel = reciprocalAngstromPerDetectorPixel
    }
}

/// Fourier least-squares integration, distinct from iterative ``DPC``.
public enum IDPC {
    /// Integrates a scan-frame field into a row-major, zero-mean scalar map.
    /// Solves φ̂ = −i(q·F̂)/(2π(q²+εq²max)), with q²max=(0.5/dr)²+(0.5/dc)².
    /// Mean-subtracts each component before padding and the phase after cropping.
    /// Sampling sets spatial units; `gradientScale` converts input values to a gradient.
    /// The field must already be rotated/transposed into the scan frame by the caller.
    public static func integrate(_ field: ScanVectorField, rowSampling: Double = 1,
                                 columnSampling: Double = 1, gradientScale: Double = 1,
                                 regularization: Double = 1e-4,
                                 boundary: IDPCBoundary = .zeroPadded(factor:2)) throws -> [Double] {
        let rows = field.scanRows, columns = field.scanColumns
        guard rows > 1, columns > 1,
              rowSampling.isFinite, rowSampling > 0, columnSampling.isFinite, columnSampling > 0,
              gradientScale.isFinite, regularization.isFinite, regularization >= 0,
              field.row.allSatisfy(\.isFinite), field.column.allSatisfy(\.isFinite) else {
            throw CuboPhaseError.invalidParameter("iDPC needs a finite field, positive sampling and nonnegative regularization")
        }
        let factor: Int
        switch boundary { case .periodic: factor = 1; case .zeroPadded(let value): factor = value }
        guard factor > 0, rows <= Int.max/factor, columns <= Int.max/factor else {
            throw CuboPhaseError.invalidParameter("iDPC padded dimensions overflow or factor is not positive")
        }
        let nr = rows*factor, nc = columns*factor
        guard nr <= Int.max/nc, nr*nc <= Int.max/MemoryLayout<Double>.stride else {
            throw CuboPhaseError.invalidParameter("iDPC padded element count overflows")
        }
        let qmax = pow(0.5/rowSampling,2)+pow(0.5/columnSampling,2)
        let epsilon = regularization*qmax
        guard qmax.isFinite, qmax > 0, epsilon.isFinite else {
            throw CuboPhaseError.invalidParameter("iDPC frequency scale overflows or underflows")
        }
        let ro = (nr-rows)/2, co = (nc-columns)/2
        let row = field.row.map{$0*gradientScale}, column = field.column.map{$0*gradientScale}
        guard row.allSatisfy(\.isFinite), column.allSatisfy(\.isFinite) else {
            throw CuboPhaseError.invalidParameter("iDPC gradient conversion overflow")
        }
        let meanRow = NumPySum.mean(row), meanColumn = NumPySum.mean(column)
        var ar = [Double](repeating:0,count:nr*nc), ai = ar, br = ar, bi = ar
        for r in 0..<rows { for c in 0..<columns {
            let k = (r+ro)*nc+c+co, j = r*columns+c
            ar[k] = row[j]-meanRow; br[k] = column[j]-meanColumn
        } }
        let fft = FFT2D(rows:nr,columns:nc)
        fft.transform(&ar,&ai,inverse:false); fft.transform(&br,&bi,inverse:false)
        for r in 0..<nr { for c in 0..<nc {
            let qr = Double(r < (nr+1)/2 ? r : r-nr)/Double(nr)/rowSampling
            let qc = Double(c < (nc+1)/2 ? c : c-nc)/Double(nc)/columnSampling
            let k = r*nc+c, q2 = qr*qr+qc*qc
            if q2 == 0 { ar[k] = 0; ai[k] = 0; continue }
            let denominator = 2 * Double.pi * (q2+epsilon)
            let re = qr*ar[k]+qc*br[k], im = qr*ai[k]+qc*bi[k]
            ar[k] = im/denominator; ai[k] = -re/denominator
        } }
        fft.transform(&ar,&ai,inverse:true)
        var result = [Double](repeating:0,count:rows*columns)
        for r in 0..<rows { for c in 0..<columns { result[r*columns+c] = ar[(r+ro)*nc+c+co] } }
        let mean = NumPySum.mean(result)
        result = result.map{$0-mean}
        guard result.allSatisfy(\.isFinite) else { throw CuboPhaseError.invalidParameter("iDPC integration overflow") }
        return result
    }

    /// Returns projected phase in radians, using ∇φ=2π·Q_sampling·CoM.
    /// Sampling is in Å per scan pixel and Å⁻¹ per detector pixel.
    public static func integratePhysical(_ field: ScanVectorField, calibration: IDPCPhysicalCalibration,
                                         regularization: Double = 1e-4,
                                         boundary: IDPCBoundary = .zeroPadded(factor:2)) throws -> [Double] {
        try integrate(field,rowSampling:calibration.rowSamplingAngstrom,
                      columnSampling:calibration.columnSamplingAngstrom,
                      gradientScale:2 * .pi * calibration.reciprocalAngstromPerDetectorPixel,
                      regularization:regularization,boundary:boundary)
    }
}
