import Foundation

/// struct ParallaxPhysicalShift: Equatable, Sendable.
public struct ParallaxPhysicalShift: Equatable, Sendable {
    /// let rowAngstrom: Double.
    public let rowAngstrom: Double
    /// let columnAngstrom: Double.
    public let columnAngstrom: Double

    /// init(rowAngstrom: Double, columnAngstrom: Double).
    public init(rowAngstrom: Double, columnAngstrom: Double) {
        self.rowAngstrom = rowAngstrom
        self.columnAngstrom = columnAngstrom
    }
}

/// struct ParallaxLowOrderAberrationOptions: Equatable, Sendable.
public struct ParallaxLowOrderAberrationOptions: Equatable, Sendable {
    /// init().
    public init() {}

    /// var forceTranspose = false.
    public var forceTranspose = false
    /// var forceRotationAngleDegrees: Double? = nil.
    public var forceRotationAngleDegrees: Double? = nil
}

/// struct ParallaxAberrationFitResult: Equatable, Sendable.
public struct ParallaxAberrationFitResult: Equatable, Sendable {
    /// let measuredShifts: [ParallaxPhysicalShift].
    public let measuredShifts: [ParallaxPhysicalShift]
    /// let fittedShifts: [ParallaxPhysicalShift].
    public let fittedShifts: [ParallaxPhysicalShift]
    /// py4DSTEM's fitted Q-to-R rotation convention.
    public let rotationRad: Double
    /// Defocus and two-fold astigmatism coefficients, in Å.
    public let c1Angstrom: Double
    /// let c12aAngstrom: Double.
    public let c12aAngstrom: Double
    /// let c12bAngstrom: Double.
    public let c12bAngstrom: Double
    /// let rmsResidualAngstrom: Double.
    public let rmsResidualAngstrom: Double
    /// let forceTranspose: Bool.
    public let forceTranspose: Bool
    /// let forcedRotationAngleDegrees: Double?.
    public let forcedRotationAngleDegrees: Double?

    /// init(measuredShifts: [ParallaxPhysicalShift], fittedShifts: [ParallaxPhysicalShift], rotationRad: Double, c1Angstrom: Double, c12aAngstrom: Double, c12bAngstrom: Double, rmsResidualAngstrom: Double, forceTranspose: Bool, forcedRotationAngleDegrees: Double?).
    public init(measuredShifts: [ParallaxPhysicalShift], fittedShifts: [ParallaxPhysicalShift], rotationRad: Double, c1Angstrom: Double, c12aAngstrom: Double, c12bAngstrom: Double, rmsResidualAngstrom: Double, forceTranspose: Bool, forcedRotationAngleDegrees: Double?) {
        self.measuredShifts = measuredShifts
        self.fittedShifts = fittedShifts
        self.rotationRad = rotationRad
        self.c1Angstrom = c1Angstrom
        self.c12aAngstrom = c12aAngstrom
        self.c12bAngstrom = c12bAngstrom
        self.rmsResidualAngstrom = rmsResidualAngstrom
        self.forceTranspose = forceTranspose
        self.forcedRotationAngleDegrees = forcedRotationAngleDegrees
    }
}

/// struct ParallaxAberrationTerm: Equatable, Hashable, Sendable.
public struct ParallaxAberrationTerm: Equatable, Hashable, Sendable {
    /// let radialOrder: Int.
    public let radialOrder: Int
    /// let angularOrder: Int.
    public let angularOrder: Int
    /// 0 is the cosine/radial component; 1 is the sine component.
    public let component: Int

    /// init(radialOrder: Int, angularOrder: Int, component: Int).
    public init(radialOrder: Int, angularOrder: Int, component: Int) {
        self.radialOrder = radialOrder
        self.angularOrder = angularOrder
        self.component = component
    }
}

/// enum ParallaxAberrationFitMethod: String, CaseIterable, Sendable.
public enum ParallaxAberrationFitMethod: String, CaseIterable, Sendable {
    case recursive
    case recursiveExclusive
    case global
}

/// struct ParallaxHigherOrderAberrationOptions: Equatable, Sendable.
public struct ParallaxHigherOrderAberrationOptions: Equatable, Sendable {
    /// init().
    public init() {}

    /// var maxRadialOrder = 3.
    public var maxRadialOrder = 3
    /// var maxAngularOrder = 4.
    public var maxAngularOrder = 4
    /// var minRadialOrder = 2.
    public var minRadialOrder = 2
    /// var minAngularOrder = 0.
    public var minAngularOrder = 0
    /// var fitMethod: ParallaxAberrationFitMethod = .recursive.
    public var fitMethod: ParallaxAberrationFitMethod = .recursive
    /// var initializeWithLowOrder = true.
    public var initializeWithLowOrder = true
    /// Origin convention for the supplied basis angles; see `basisAngles` helper.
    public var basisConvention: ParallaxBasisConvention = .detectorGrid
    /// var lowOrder = ParallaxLowOrderAberrationOptions().
    public var lowOrder = ParallaxLowOrderAberrationOptions()
}

/// struct ParallaxHigherOrderAberrationFitResult: Equatable, Sendable.
public struct ParallaxHigherOrderAberrationFitResult: Equatable, Sendable {
    /// let lowOrder: ParallaxAberrationFitResult.
    public let lowOrder: ParallaxAberrationFitResult
    /// let terms: [ParallaxAberrationTerm].
    public let terms: [ParallaxAberrationTerm]
    /// Cartesian coefficient for each term, in Å.
    public let coefficientsAngstrom: [Double]
    /// let fittedShifts: [ParallaxPhysicalShift].
    public let fittedShifts: [ParallaxPhysicalShift]
    /// let rmsResidualAngstrom: Double.
    public let rmsResidualAngstrom: Double
    /// let fitMethod: ParallaxAberrationFitMethod.
    public let fitMethod: ParallaxAberrationFitMethod

    /// init(lowOrder: ParallaxAberrationFitResult, terms: [ParallaxAberrationTerm], coefficientsAngstrom: [Double], fittedShifts: [ParallaxPhysicalShift], rmsResidualAngstrom: Double, fitMethod: ParallaxAberrationFitMethod).
    public init(lowOrder: ParallaxAberrationFitResult, terms: [ParallaxAberrationTerm], coefficientsAngstrom: [Double], fittedShifts: [ParallaxPhysicalShift], rmsResidualAngstrom: Double, fitMethod: ParallaxAberrationFitMethod) {
        self.lowOrder = lowOrder
        self.terms = terms
        self.coefficientsAngstrom = coefficientsAngstrom
        self.fittedShifts = fittedShifts
        self.rmsResidualAngstrom = rmsResidualAngstrom
        self.fitMethod = fitMethod
    }
}

/// enum ParallaxAberrationFitter.
public enum ParallaxAberrationFitter {
    /// enum FitError: LocalizedError, Equatable.
    public enum FitError: LocalizedError, Equatable {
        case invalidInput(String)
        case singularAngleGeometry
        case singularPolarDecomposition

        /// var errorDescription: String?.
        public var errorDescription: String? {
            switch self {
            case .invalidInput(let detail):
                return "Cannot fit parallax aberrations: \(detail)."
            case .singularAngleGeometry:
                return "The selected probe angles do not span two reciprocal-space directions."
            case .singularPolarDecomposition:
                return "The measured parallax shift transform has a singular polar decomposition."
            }
        }
    }

    /// Port of py4DSTEM `_aberration_fit_polar_decomposition`.
    public static func fitLowOrder(
        probeAngles: [ParallaxShift], pixelShifts: [ParallaxShift],
        scanSampling: ParallaxShift,
        options: ParallaxLowOrderAberrationOptions = .init()
    ) throws -> ParallaxAberrationFitResult {
        let count = probeAngles.count
        guard count >= 2, pixelShifts.count == count else {
            throw FitError.invalidInput("the probe-angle and shift fields differ")
        }
        let sampling = scanSampling.row
        guard sampling.isFinite, sampling > 0, scanSampling.column.isFinite, scanSampling.column > 0 else {
            throw FitError.invalidInput("scan sampling is not positive and finite")
        }
        if let forced = options.forceRotationAngleDegrees, !forced.isFinite {
            throw FitError.invalidInput("the forced rotation is not finite")
        }

        var angles = [(Double, Double)]()
        var measured = [ParallaxPhysicalShift]()
        angles.reserveCapacity(count)
        measured.reserveCapacity(count)
        for (angle, pixelShift) in zip(
            probeAngles, pixelShifts
        ) {
            let rowAngle = angle.row
            let columnAngle = angle.column
            var rowShift = pixelShift.row * sampling
            var columnShift = pixelShift.column * scanSampling.column
            if options.forceTranspose { swap(&rowShift, &columnShift) }
            guard rowAngle.isFinite, columnAngle.isFinite,
                  rowShift.isFinite, columnShift.isFinite else {
                throw FitError.invalidInput("the angle or shift field is non-finite")
            }
            angles.append((rowAngle, columnAngle))
            measured.append(ParallaxPhysicalShift(
                rowAngstrom: rowShift, columnAngstrom: columnShift
            ))
        }

        let transform = try leastSquaresTransform(
            angles: angles, shifts: measured
        )
        let rotation: Matrix2
        var aberration: Matrix2
        let rotationRad: Double
        if let forcedDegrees = options.forceRotationAngleDegrees {
            rotationRad = forcedDegrees * .pi / 180
            let cosine = cos(rotationRad), sine = sin(rotationRad)
            var forcedRotation = Matrix2(
                m00: cosine, m01: -sine,
                m10: sine, m11: cosine
            )
            if options.forceTranspose { forcedRotation = forcedRotation.transposed }
            rotation = forcedRotation
            // This multiplication order deliberately follows py4DSTEM.
            aberration = rotation * transform
        } else {
            let decomposition = try rightPolarDecomposition(transform)
            rotation = options.forceTranspose
                ? decomposition.rotation.transposed
                : decomposition.rotation
            aberration = decomposition.symmetric
            var recovered = -atan2(rotation.m10, rotation.m00)
            if 2 * abs(positiveModulo(recovered + .pi, 2 * .pi) - .pi) > .pi {
                recovered = positiveModulo(recovered, 2 * .pi) - .pi
                aberration = aberration.scaled(by: -1)
            }
            rotationRad = recovered
        }

        let c1 = (aberration.m00 + aberration.m11) / 2
        let c12a = options.forceTranspose
            ? -(aberration.m00 - aberration.m11) / 2
            : (aberration.m00 - aberration.m11) / 2
        let c12b = (aberration.m10 + aberration.m01) / 2
        guard rotationRad.isFinite, c1.isFinite, c12a.isFinite,
              c12b.isFinite else {
            throw FitError.singularPolarDecomposition
        }

        var fitted = [ParallaxPhysicalShift]()
        fitted.reserveCapacity(count)
        var squaredResidual: Double = 0
        for (angle, observed) in zip(angles, measured) {
            let row = angle.0 * transform.m00 + angle.1 * transform.m10
            let column = angle.0 * transform.m01 + angle.1 * transform.m11
            fitted.append(ParallaxPhysicalShift(
                rowAngstrom: row, columnAngstrom: column
            ))
            squaredResidual += pow(observed.rowAngstrom - row, 2)
                + pow(observed.columnAngstrom - column, 2)
        }
        let rms = sqrt(squaredResidual / Double(count))
        return ParallaxAberrationFitResult(
            measuredShifts: measured, fittedShifts: fitted,
            rotationRad: rotationRad,
            c1Angstrom: c1, c12aAngstrom: c12a, c12bAngstrom: c12b,
            rmsResidualAngstrom: rms,
            forceTranspose: options.forceTranspose,
            forcedRotationAngleDegrees: options.forceRotationAngleDegrees
        )
    }

    /// static func defaultTerms(.
    public static func defaultTerms(
        maxRadialOrder: Int = 3,
        maxAngularOrder: Int = 4,
        minRadialOrder: Int = 2,
        minAngularOrder: Int = 0
    ) throws -> [ParallaxAberrationTerm] {
        guard minRadialOrder >= 2, maxRadialOrder >= minRadialOrder, maxRadialOrder <= 8,
              maxAngularOrder <= 8,
              minAngularOrder >= 0, maxAngularOrder >= minAngularOrder else {
            throw FitError.invalidInput("the requested aberration orders are invalid")
        }
        var terms = [ParallaxAberrationTerm]()
        for m in (minRadialOrder - 1)..<maxRadialOrder {
            let maximumN = min(maxAngularOrder, m + 1)
            guard minAngularOrder <= maximumN else { continue }
            for n in minAngularOrder...maximumN where (m + n).isMultiple(of: 2) == false {
                terms.append(ParallaxAberrationTerm(
                    radialOrder: m, angularOrder: n, component: 0
                ))
                if n > 0 {
                    terms.append(ParallaxAberrationTerm(
                        radialOrder: m, angularOrder: n, component: 1
                    ))
                }
            }
        }
        guard !terms.isEmpty else {
            throw FitError.invalidInput("the requested aberration range contains no terms")
        }
        return terms
    }

    /// Fits the requested Cartesian aberration orders, with recursive or global refinement.
    /// Detector-grid basis coordinates follow the pinned reference by default.
    public static func fitHigherOrder(
        probeAngles: [ParallaxShift], pixelShifts: [ParallaxShift],
        scanSampling: ParallaxShift,
        basisAngles: [ParallaxShift],
        options: ParallaxHigherOrderAberrationOptions = .init()
    ) throws -> ParallaxHigherOrderAberrationFitResult {
        let lowOrder = try fitLowOrder(
            probeAngles: probeAngles, pixelShifts: pixelShifts, scanSampling: scanSampling,
            options: options.lowOrder
        )
        let terms = try defaultTerms(
            maxRadialOrder: options.maxRadialOrder,
            maxAngularOrder: options.maxAngularOrder,
            minRadialOrder: options.minRadialOrder,
            minAngularOrder: options.minAngularOrder
        )
        let gradients = try gradientSamples(
            probeAngles: basisAngles,
            rotationRad: lowOrder.rotationRad,
            terms: terms
        )
        let count = lowOrder.measuredShifts.count
        guard basisAngles.count == count else { throw FitError.invalidInput("basis angles differ") }
        var coefficients = [Double](repeating: 0, count: terms.count)
        var fitted = [ParallaxPhysicalShift](
            repeating: ParallaxPhysicalShift(rowAngstrom: 0, columnAngstrom: 0),
            count: count
        )

        if options.initializeWithLowOrder {
            for (index, term) in terms.enumerated() {
                switch (term.radialOrder, term.angularOrder, term.component) {
                case (1, 0, 0): coefficients[index] = lowOrder.c1Angstrom
                case (1, 2, 0): coefficients[index] = lowOrder.c12aAngstrom
                case (1, 2, 1): coefficients[index] = lowOrder.c12bAngstrom
                default: continue
                }
            }
            fitted = fittedShifts(
                gradients: gradients, coefficients: coefficients,
                count: count, termCount: terms.count
            )
        }

        let groups = radialGroups(terms)
        let fitPasses: [[Int]]
        switch options.fitMethod {
        case .recursiveExclusive:
            fitPasses = groups
        case .recursive:
            fitPasses = groups.indices.map { groupIndex in
                Array(groups[0...groupIndex].joined())
            }
        case .global:
            fitPasses = [Array(terms.indices)]
        }

        for indices in fitPasses {
            var matrix = [[Double]](
                repeating: [Double](repeating: 0, count: indices.count),
                count: count * 2
            )
            var residual = [Double](repeating: 0, count: count * 2)
            for sample in 0..<count {
                residual[sample] = lowOrder.measuredShifts[sample].rowAngstrom
                    - fitted[sample].rowAngstrom
                residual[count + sample] = lowOrder.measuredShifts[sample].columnAngstrom
                    - fitted[sample].columnAngstrom
                for (local, term) in indices.enumerated() {
                    matrix[sample][local] = gradients[sample * terms.count + term].0
                    matrix[count + sample][local]
                        = gradients[sample * terms.count + term].1
                }
            }
            let increment = try leastSquares(matrix: matrix, values: residual)
            for (local, term) in indices.enumerated() {
                coefficients[term] += increment[local]
            }
            for sample in 0..<count {
                var row = fitted[sample].rowAngstrom
                var column = fitted[sample].columnAngstrom
                for (local, term) in indices.enumerated() {
                    row += gradients[sample * terms.count + term].0 * increment[local]
                    column += gradients[sample * terms.count + term].1 * increment[local]
                }
                fitted[sample] = ParallaxPhysicalShift(
                    rowAngstrom: row, columnAngstrom: column
                )
            }
        }

        if options.lowOrder.forceTranspose {
            for index in terms.indices
            where terms[index].angularOrder > 0 && terms[index].component == 0 {
                coefficients[index] *= -1
            }
        }
        var squaredResidual: Double = 0
        for (observed, estimate) in zip(lowOrder.measuredShifts, fitted) {
            squaredResidual += pow(observed.rowAngstrom - estimate.rowAngstrom, 2)
                + pow(observed.columnAngstrom - estimate.columnAngstrom, 2)
        }
        return ParallaxHigherOrderAberrationFitResult(
            lowOrder: lowOrder, terms: terms,
            coefficientsAngstrom: coefficients, fittedShifts: fitted,
            rmsResidualAngstrom: sqrt(squaredResidual / Double(count)),
            fitMethod: options.fitMethod
        )
    }

    /// Row-major `[sample, term]` pairs of (dχ/du, dχ/dv). The derivatives
    /// intentionally omit the phase-only global scale, matching py4DSTEM.
    public static func gradientSamples(
        probeAngles: [ParallaxShift],
        rotationRad: Double,
        terms: [ParallaxAberrationTerm]
    ) throws -> [(Double, Double)] {
        guard rotationRad.isFinite, !terms.isEmpty else {
            throw FitError.invalidInput("the gradient basis is empty or non-finite")
        }
        let cosine = cos(-rotationRad), sine = sin(-rotationRad)
        var output = [(Double, Double)]()
        output.reserveCapacity(probeAngles.count * terms.count)
        for angle in probeAngles {
            let initialU = angle.row
            let initialV = angle.column
            let u = initialU * cosine + initialV * sine
            let v = -initialU * sine + initialV * cosine
            let alpha = hypot(u, v)
            let theta = atan2(v, u)
            guard u.isFinite, v.isFinite else {
                throw FitError.invalidInput("the probe-angle field is non-finite")
            }
            for term in terms {
                let m = term.radialOrder
                let n = term.angularOrder
                let radial = pow(alpha, Double(m - 1))
                if n == 0 {
                    output.append((u * radial, v * radial))
                } else if term.component == 0 {
                    let cosN = cos(Double(n) * theta)
                    let sinN = sin(Double(n) * theta)
                    output.append((
                        radial * (Double(m + 1) * u * cosN
                                  + Double(n) * v * sinN) / Double(m + 1),
                        radial * (Double(m + 1) * v * cosN
                                  - Double(n) * u * sinN) / Double(m + 1)
                    ))
                } else {
                    let cosN = cos(Double(n) * theta)
                    let sinN = sin(Double(n) * theta)
                    output.append((
                        radial * (Double(m + 1) * u * sinN
                                  - Double(n) * v * cosN) / Double(m + 1),
                        radial * (Double(m + 1) * v * sinN
                                  + Double(n) * u * cosN) / Double(m + 1)
                    ))
                }
            }
        }
        return output
    }

    private static func radialGroups(
        _ terms: [ParallaxAberrationTerm]
    ) -> [[Int]] {
        var groups = [[Int]]()
        for index in terms.indices {
            if groups.last?.first.map({ terms[$0].radialOrder })
                != terms[index].radialOrder {
                groups.append([])
            }
            groups[groups.count - 1].append(index)
        }
        return groups
    }

    private static func fittedShifts(
        gradients: [(Double, Double)], coefficients: [Double],
        count: Int, termCount: Int
    ) -> [ParallaxPhysicalShift] {
        (0..<count).map { sample in
            var row: Double = 0, column: Double = 0
            for term in 0..<termCount {
                row += gradients[sample * termCount + term].0 * coefficients[term]
                column += gradients[sample * termCount + term].1 * coefficients[term]
            }
            return ParallaxPhysicalShift(
                rowAngstrom: row, columnAngstrom: column
            )
        }
    }

    private static func leastSquares(
        matrix: [[Double]], values: [Double]
    ) throws -> [Double] {
        guard let columnCount = matrix.first?.count, columnCount > 0,
              matrix.count == values.count,
              matrix.allSatisfy({ $0.count == columnCount }) else {
            throw FitError.invalidInput("the aberration least-squares matrix is inconsistent")
        }
        let norms = (0..<columnCount).map { c in sqrt(matrix.reduce(0) { $0 + $1[c] * $1[c] }) }
        guard norms.allSatisfy({ $0.isFinite && $0 > 0 }) else { throw FitError.singularAngleGeometry }
        var normal = [[Double]](repeating: [Double](repeating: 0, count: columnCount), count: columnCount)
        var right = [Double](repeating: 0, count: columnCount)
        for row in matrix.indices {
            for column in 0..<columnCount {
                let v = matrix[row][column] / norms[column]
                right[column] += v * values[row]
                for other in 0..<columnCount { normal[column][other] += v * matrix[row][other] / norms[other] }
            }
        }
        let scale = normal.flatMap { $0 }.map(abs).max() ?? 0
        guard scale.isFinite, scale > 0 else { throw FitError.singularAngleGeometry }
        for pivot in 0..<columnCount {
            let selected = (pivot..<columnCount).max {
                abs(normal[$0][pivot]) < abs(normal[$1][pivot])
            }!
            guard abs(normal[selected][pivot]) > scale * 1e-12 else {
                throw FitError.singularAngleGeometry
            }
            if selected != pivot {
                normal.swapAt(selected, pivot)
                right.swapAt(selected, pivot)
            }
            let divisor = normal[pivot][pivot]
            for column in pivot..<columnCount { normal[pivot][column] /= divisor }
            right[pivot] /= divisor
            for row in 0..<columnCount where row != pivot {
                let factor = normal[row][pivot]
                for column in pivot..<columnCount {
                    normal[row][column] -= factor * normal[pivot][column]
                }
                right[row] -= factor * right[pivot]
            }
        }
        let result = zip(right, norms).map(/)
        guard result.allSatisfy(\.isFinite) else { throw FitError.singularAngleGeometry }
        return result
    }

    private struct Matrix2 {
        let m00: Double
        let m01: Double
        let m10: Double
        let m11: Double

        var determinant: Double { m00 * m11 - m01 * m10 }

        var transposed: Matrix2 {
            Matrix2(m00: m00, m01: m10, m10: m01, m11: m11)
        }

        func scaled(by scale: Double) -> Matrix2 {
            Matrix2(
                m00: m00 * scale, m01: m01 * scale,
                m10: m10 * scale, m11: m11 * scale
            )
        }

        static func * (lhs: Matrix2, rhs: Matrix2) -> Matrix2 {
            Matrix2(
                m00: lhs.m00 * rhs.m00 + lhs.m01 * rhs.m10,
                m01: lhs.m00 * rhs.m01 + lhs.m01 * rhs.m11,
                m10: lhs.m10 * rhs.m00 + lhs.m11 * rhs.m10,
                m11: lhs.m10 * rhs.m01 + lhs.m11 * rhs.m11
            )
        }
    }

    private static func leastSquaresTransform(
        angles: [(Double, Double)], shifts: [ParallaxPhysicalShift]
    ) throws -> Matrix2 {
        var xx: Double = 0, xy: Double = 0, yy: Double = 0
        var xRow: Double = 0, yRow: Double = 0
        var xColumn: Double = 0, yColumn: Double = 0
        for (angle, shift) in zip(angles, shifts) {
            xx += angle.0 * angle.0
            xy += angle.0 * angle.1
            yy += angle.1 * angle.1
            xRow += angle.0 * shift.rowAngstrom
            yRow += angle.1 * shift.rowAngstrom
            xColumn += angle.0 * shift.columnAngstrom
            yColumn += angle.1 * shift.columnAngstrom
        }
        let determinant = xx * yy - xy * xy
        guard determinant.isFinite,
              abs(determinant) > max(1e-30, (xx + yy) * 1e-12) else {
            throw FitError.singularAngleGeometry
        }
        return Matrix2(
            m00: (yy * xRow - xy * yRow) / determinant,
            m01: (yy * xColumn - xy * yColumn) / determinant,
            m10: (-xy * xRow + xx * yRow) / determinant,
            m11: (-xy * xColumn + xx * yColumn) / determinant
        )
    }

    /// For M = QP, P = sqrt(MᵀM) and Q = MP⁻¹. The closed-form square root
    /// is exact for a positive-definite 2×2 symmetric matrix and also retains
    /// the reflection branch used by SciPy when det(M) < 0.
    private static func rightPolarDecomposition(
        _ matrix: Matrix2
    ) throws -> (rotation: Matrix2, symmetric: Matrix2) {
        let x = matrix.m00 * matrix.m00 + matrix.m10 * matrix.m10
        let z = matrix.m00 * matrix.m01 + matrix.m10 * matrix.m11
        let y = matrix.m01 * matrix.m01 + matrix.m11 * matrix.m11
        let determinant = x * y - z * z
        guard determinant.isFinite, determinant > 1e-30 else {
            throw FitError.singularPolarDecomposition
        }
        let rootDeterminant = sqrt(determinant)
        let denominator = sqrt(x + y + 2 * rootDeterminant)
        guard denominator.isFinite, denominator > 0 else {
            throw FitError.singularPolarDecomposition
        }
        let symmetric = Matrix2(
            m00: (x + rootDeterminant) / denominator,
            m01: z / denominator,
            m10: z / denominator,
            m11: (y + rootDeterminant) / denominator
        )
        let inverseDeterminant = symmetric.determinant
        guard inverseDeterminant.isFinite, abs(inverseDeterminant) > 1e-15 else {
            throw FitError.singularPolarDecomposition
        }
        let inverse = Matrix2(
            m00: symmetric.m11 / inverseDeterminant,
            m01: -symmetric.m01 / inverseDeterminant,
            m10: -symmetric.m10 / inverseDeterminant,
            m11: symmetric.m00 / inverseDeterminant
        )
        return (matrix * inverse, symmetric)
    }

    private static func positiveModulo(_ value: Double, _ modulus: Double) -> Double {
        let remainder = value.truncatingRemainder(dividingBy: modulus)
        return remainder >= 0 ? remainder : remainder + modulus
    }
}

/// Basis-coordinate and rotation choice for higher-order aberration fitting.
public enum ParallaxBasisConvention: Sendable, Equatable {
    /// Detector FFT grid centered on integer half-shape; passive reference rotation.
    case detectorGrid
    /// Selected-angle centroid used in the source application.
    case selectedCentroid
}

extension ParallaxAberrationFitter {
    /// Returns angles in radians for the chosen gradient basis origin.
    /// Use `.detectorGrid` (default) with `fitHigherOrder`; `.selectedCentroid`
    /// preserves the source application's higher-order interpretation.
    public static func basisAngles(detectorIndices: [[Int]], detectorRows: Int, detectorColumns: Int,
        reciprocalSampling: ParallaxShift, wavelength: Double,
        convention: ParallaxBasisConvention = .detectorGrid) throws -> [ParallaxShift] {
        guard detectorRows > 0, detectorColumns > 0, !detectorIndices.isEmpty,
            reciprocalSampling.row.isFinite, reciprocalSampling.column.isFinite,
            reciprocalSampling.row > 0, reciprocalSampling.column > 0, wavelength.isFinite, wavelength > 0,
            detectorIndices.allSatisfy({ $0.count == 2 && $0[0] >= 0 && $0[0] < detectorRows && $0[1] >= 0 && $0[1] < detectorColumns }) else { throw ParallaxError.invalidInput }
        let cr = convention == .detectorGrid ? Double(detectorRows/2) : detectorIndices.reduce(0.0) { $0 + Double($1[0]) } / Double(detectorIndices.count)
        let cc = convention == .detectorGrid ? Double(detectorColumns/2) : detectorIndices.reduce(0.0) { $0 + Double($1[1]) } / Double(detectorIndices.count)
        let result = detectorIndices.map { ParallaxShift(row: (Double($0[0])-cr)*reciprocalSampling.row*wavelength,
            column: (Double($0[1])-cc)*reciprocalSampling.column*wavelength) }
        guard result.allSatisfy({ $0.row.isFinite && $0.column.isFinite }) else { throw ParallaxError.nonfiniteValues }
        return result
    }
}
