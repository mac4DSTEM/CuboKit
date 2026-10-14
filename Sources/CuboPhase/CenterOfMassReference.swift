//
//  CenterOfMassReference.swift
//  The reference a measured centre of mass is taken against, and the normalized shift (registration slice 2, 3.2):
//  py4DSTEM's `fit_origin` (process/calibration/origin.py 76) and the normalization at the end of
//  `_calculate_intensities_center_of_mass` (process/phase/phase_base_class.py) at f050d207.
//
//  The plane is the least-squares solution in closed form, the definition; py4DSTEM reaches the same plane with
//  scipy's curve_fit on coordinates scaled to [0, 1], within that solver's convergence.
//

import CuboCore

public enum CenterOfMassReference: Sendable, Equatable {
    /// The least-squares plane a + b·row + c·column of each component over the finite positions (py4DSTEM "plane").
    case planeFit
    /// The mean of each component over every position (py4DSTEM "constant"; NaN if any value is NaN).
    case constant
    /// Zero: the measured centre of mass is the shift.
    case none
    /// A given field, per-position origins measured elsewhere (py4DSTEM `force_com_shifts`).
    case given(ScanVectorField)

    /// The fitted reference and the normalized shift `(measured − fitted)`, NaN → 0 and ±∞ → ±greatest finite (NumPy's
    /// `nan_to_num`), times `reciprocalSampling`.
    public static func normalize(_ measured: ScanVectorField, reference: CenterOfMassReference,
                                 reciprocalSampling: Double = 1) throws -> (fitted: ScanVectorField, normalized: ScanVectorField) {
        let R = measured.scanRows, C = measured.scanColumns
        let fitted: ScanVectorField
        switch reference {
        case .planeFit:
            fitted = try ScanVectorField(scanRows: R, scanColumns: C, row: try plane(measured.row, R, C),
                                         column: try plane(measured.column, R, C))
        case .constant:
            fitted = try ScanVectorField(scanRows: R, scanColumns: C,
                                         row: Array(repeating: NumPySum.mean(measured.row), count: R * C),
                                         column: Array(repeating: NumPySum.mean(measured.column), count: R * C))
        case .none:
            fitted = try ScanVectorField(scanRows: R, scanColumns: C, row: Array(repeating: 0, count: R * C),
                                         column: Array(repeating: 0, count: R * C))
        case .given(let field):
            guard field.scanRows == R, field.scanColumns == C else {
                throw CuboPhaseError.invalidField("the given reference is \(field.scanRows) × \(field.scanColumns), the field \(R) × \(C)")
            }
            fitted = field
        }
        func clean(_ v: Double) -> Double {
            v.isNaN ? 0 : (v == .infinity ? .greatestFiniteMagnitude : (v == -.infinity ? -.greatestFiniteMagnitude : v))
        }
        let nr = zip(measured.row, fitted.row).map { clean($0 - $1) * reciprocalSampling }
        let nc = zip(measured.column, fitted.column).map { clean($0 - $1) * reciprocalSampling }
        return (fitted, try ScanVectorField(scanRows: R, scanColumns: C, row: nr, column: nc))
    }

    /// The least-squares plane over the finite values, from the normal equations on centred coordinates, evaluated
    /// at every position.
    static func plane(_ m: [Double], _ R: Int, _ C: Int) throws -> [Double] {
        let finite = m.indices.filter { m[$0].isFinite }
        guard finite.count >= 3 else { throw CuboPhaseError.planeFitDegenerate }
        let n = Double(finite.count)
        let rMean = finite.reduce(0.0) { $0 + Double($1 / C) } / n
        let cMean = finite.reduce(0.0) { $0 + Double($1 % C) } / n
        var suu = 0.0, suv = 0.0, svv = 0.0, sum = 0.0
        for k in finite {
            let u = Double(k / C) - rMean, v = Double(k % C) - cMean
            suu += u * u; suv += u * v; svv += v * v; sum += m[k]
        }
        let a = sum / n
        var sum_um = 0.0, sum_vm = 0.0
        for k in finite {
            let u = Double(k / C) - rMean, v = Double(k % C) - cMean
            sum_um += u * (m[k] - a); sum_vm += v * (m[k] - a)
        }
        let det = suu * svv - suv * suv
        guard det > 1e-12 * Swift.max(suu * svv, 1) else { throw CuboPhaseError.planeFitDegenerate }
        let b = (svv * sum_um - suv * sum_vm) / det
        let c = (suu * sum_vm - suv * sum_um) / det
        return (0..<(R * C)).map { a + b * (Double($0 / C) - rMean) + c * (Double($0 % C) - cMean) }
    }
}
