import Accelerate
import CuboCore
import Foundation

/// The arithmetic of a ptychographic engine: `double` is the kit's reference path; `single` runs the heavy loops in Float32
/// (vDSP's Float DFT, half the memory) and agrees with `double` to 1e-5 relative to the object's scale (D12 Float32).
public enum PtychographyPrecision: Sendable, Equatable {
    case double
    case single
}

/// A zeroed Float buffer owned by hand, so the hot loops use plain pointers.
private struct FloatBuffer {
    let pointer: UnsafeMutablePointer<Float>
    let count: Int
    init(_ count: Int) {
        self.count = count
        pointer = .allocate(capacity: max(count, 1))
        pointer.initialize(repeating: 0, count: max(count, 1))
    }
    init(_ values: [Double]) {
        self.init(values.count)
        for k in values.indices { pointer[k] = Float(values[k]) }
    }
    func free() { pointer.deallocate() }
    var doubles: [Double] { (0..<count).map { Double(pointer[$0]) } }
}

extension SingleslicePtychography {
    /// ``SingleslicePtychography/reconstruct(amplitudes:positions:object:probe:iterations:stepSize:normalizationMinimum:fixProbe:purePhaseObject:constrainObjectAmplitude:fixProbeCenterOfMass:constrainProbeAmplitude:probeAmplitudeRelativeRadius:probeAmplitudeRelativeWidth:fixProbeAperture:probeSupportMask:errorNormalization:positionCorrection:precision:isCancelled:progress:)``
    /// in Float32, for amplitudes that are already Float (no Double copy of the stack is made).
    ///
    /// The same update, constraints and error as the Double path; the forward and adjoint transforms, the shifted probe, the
    /// projection and the accumulations run in Float32 on vDSP's DFT, and the probe constraints (centre of mass, aperture,
    /// amplitude) run in Double once per iteration. The detector size must be one vDSP's DFT takes (`2^n × {1, 3, 5, 15}`,
    /// ≥ 16 on a side), otherwise this throws ``PtychographyError/invalidInput``; position correction is not available here.
    /// `isCancelled` is polled before every fourth pattern and throws `CancellationError`; `progress` receives
    /// `(completedIterations, iterations)` after each iteration. The result is converted to Double.
    public static func reconstructSingle(
        amplitudes: [Float], positions: [PtychographyPosition],
        object: PtychographyArray, probe: PtychographyArray,
        iterations: Int = 8, stepSize: Double = 0.5, normalizationMinimum: Double = 1, fixProbe: Bool = false,
        purePhaseObject: Bool = false, constrainObjectAmplitude: Bool = true,
        fixProbeCenterOfMass: Bool = false, constrainProbeAmplitude: Bool = false,
        probeAmplitudeRelativeRadius: Double = 0.5, probeAmplitudeRelativeWidth: Double = 0.05,
        fixProbeAperture: Bool = false, probeSupportMask: [Bool]? = nil,
        errorNormalization: PtychographyErrorNormalization = .suppliedAmplitudeEnergy,
        isCancelled: (() -> Bool)? = nil, progress: ((_ completed: Int, _ total: Int) -> Void)? = nil
    ) throws -> Result {
        let rows = probe.rows, columns = probe.columns, n = rows * columns, count = positions.count
        let objectRows = object.rows, objectColumns = object.columns
        guard count > 0, count <= Int.max / n, amplitudes.count == count * n,
              amplitudes.allSatisfy({ $0.isFinite && $0 >= 0 }),
              iterations >= 0, stepSize.isFinite, stepSize > 0,
              normalizationMinimum.isFinite, (0...1).contains(normalizationMinimum),
              positionsAreUsable(positions), probeConstraintParametersAreValid(radius: probeAmplitudeRelativeRadius,
                                                                               width: probeAmplitudeRelativeWidth),
              objectRows >= rows, objectColumns >= columns,
              probeSupportMask == nil || probeSupportMask!.count == n,
              let fft = FloatFFT2D(rows: rows, columns: columns) else { throw PtychographyError.invalidInput }
        let divisor: Double
        switch errorNormalization {
        case .meanIntensity(let mean): divisor = mean * Double(count)
        case .suppliedAmplitudeEnergy: divisor = amplitudes.reduce(0) { $0 + Double($1) * Double($1) }
        }
        guard divisor.isFinite, divisor > 0 else { throw PtychographyError.invalidInput }

        let doubleFFT = FFT2D(rows: rows, columns: columns)
        let initialAperture = fixProbeAperture ? fourierAmplitude(probe.real, probe.imaginary, doubleFFT) : []
        let inverseCount = 1 / Float(n)

        // measured amplitudes in the spectrum's transposed layout (columns x rows), once
        let measured = FloatBuffer(count * n); defer { measured.free() }
        amplitudes.withUnsafeBufferPointer { source in
            for pattern in 0..<count {
                vDSP_mtrans(source.baseAddress! + pattern * n, 1, measured.pointer + pattern * n, 1,
                            vDSP_Length(columns), vDSP_Length(rows))
            }
        }

        let objectCount = objectRows * objectColumns
        let or = FloatBuffer(object.real), oi = FloatBuffer(object.imaginary)
        let pr = FloatBuffer(probe.real), pi = FloatBuffer(probe.imaginary)
        let onr = FloatBuffer(objectCount), oni = FloatBuffer(objectCount), pn = FloatBuffer(objectCount)
        let pnr = FloatBuffer(n), pni = FloatBuffer(n), on = FloatBuffer(n)
        let psrT = FloatBuffer(n), psiT = FloatBuffer(n)                    // probe spectrum, transposed
        let ar = FloatBuffer(n), ai = FloatBuffer(n)                        // work A
        let sr = FloatBuffer(n), si = FloatBuffer(n)                        // the shifted probe, natural
        let fr = FloatBuffer(n), fi = FloatBuffer(n)                        // far field, transposed
        defer { for b in [or, oi, pr, pi, onr, oni, pn, pnr, pni, on, psrT, psiT, ar, ai, sr, si, fr, fi] { b.free() } }

        let rowFrequency = (0..<rows).map { frequency($0, rows) }, columnFrequency = (0..<columns).map { frequency($0, columns) }
        var rowCos = [Float](repeating: 0, count: rows), rowSin = rowCos
        var columnCos = [Float](repeating: 0, count: columns), columnSin = columnCos
        var rowBase = [Int](repeating: 0, count: rows), columnIndex = [Int](repeating: 0, count: columns)
        var indices = [Int](repeating: 0, count: n)
        func wrapped(_ v: Int, _ n: Int) -> Int { let m = v % n; return m < 0 ? m + n : m }

        var errors = [Double]()
        let m = normalizationMinimum
        func inverse(_ local: Float, _ maximum: Float) -> Float {
            let l = (1 - m) * Double(local), x = m * Double(maximum)
            return Float(1 / (1e-16 + l * l + x * x).squareRoot())
        }
        for iteration in 0..<iterations {
            try checkCancelled(isCancelled)
            for buffer in [onr, oni, pn, pnr, pni, on] { buffer.pointer.update(repeating: 0, count: buffer.count) }
            // the probe spectrum in Double (once per iteration), kept transposed in Float
            var spectrumReal = pr.doubles, spectrumImaginary = pi.doubles
            doubleFFT.transform(&spectrumReal, &spectrumImaginary, inverse: false)
            for r in 0..<rows { for c in 0..<columns {
                psrT.pointer[c * rows + r] = Float(spectrumReal[r * columns + c])
                psiT.pointer[c * rows + r] = Float(spectrumImaginary[r * columns + c])
            } }
            var error = 0.0
            for pattern in 0..<count {
                if pattern & 3 == 0 { try checkCancelled(isCancelled) }
                let position = positions[pattern]
                let rr = Int(position.row.rounded(.toNearestOrEven)), cc = Int(position.column.rounded(.toNearestOrEven))
                let dr = position.row - Double(rr), dc = position.column - Double(cc)
                for r in 0..<rows {
                    let a = -2 * Double.pi * rowFrequency[r] * dr
                    rowCos[r] = Float(cos(a)); rowSin[r] = Float(sin(a))
                }
                for c in 0..<columns {   // the inverse transform's 1/n rides on the column ramp
                    let a = -2 * Double.pi * columnFrequency[c] * dc
                    columnCos[c] = Float(cos(a)) * inverseCount; columnSin[c] = Float(sin(a)) * inverseCount
                }
                for c in 0..<columns {
                    let cc = columnCos[c], cs = columnSin[c], base = c * rows
                    for r in 0..<rows {
                        let cosine = rowCos[r] * cc - rowSin[r] * cs, sine = rowSin[r] * cc + rowCos[r] * cs
                        let a = psrT.pointer[base + r], b = psiT.pointer[base + r]
                        ar.pointer[base + r] = a * cosine - b * sine; ai.pointer[base + r] = a * sine + b * cosine
                    }
                }
                fft.inverse(aRe: ar.pointer, aIm: ai.pointer, bRe: sr.pointer, bIm: si.pointer)
                for r in 0..<rows {
                    let rowOffset = r < (rows + 1) / 2 ? r : r - rows
                    rowBase[r] = wrapped(rr + rowOffset, objectRows) * objectColumns
                }
                for c in 0..<columns {
                    let colOffset = c < (columns + 1) / 2 ? c : c - columns
                    columnIndex[c] = wrapped(cc + colOffset, objectColumns)
                }
                for r in 0..<rows {
                    let base = r * columns, rowStart = rowBase[r]
                    for c in 0..<columns {
                        let k = base + c, j = rowStart + columnIndex[c]
                        indices[k] = j
                        ar.pointer[k] = sr.pointer[k] * or.pointer[j] - si.pointer[k] * oi.pointer[j]
                        ai.pointer[k] = sr.pointer[k] * oi.pointer[j] + si.pointer[k] * or.pointer[j]
                    }
                }
                fft.forward(aRe: ar.pointer, aIm: ai.pointer, bRe: fr.pointer, bIm: fi.pointer)
                // Fourier-modulus projection, in the transposed layout, with the inverse's 1/n folded in
                let target = measured.pointer + pattern * n
                var patternError: Float = 0
                for t in 0..<n {
                    let a = fr.pointer[t], b = fi.pointer[t], wanted = target[t]
                    let magnitude = (a * a + b * b).squareRoot()
                    let miss = wanted - magnitude
                    patternError += miss * miss
                    if magnitude > 0 {
                        let scale = wanted / magnitude
                        fr.pointer[t] = (scale * a - a) * inverseCount; fi.pointer[t] = (scale * b - b) * inverseCount
                    } else {
                        fr.pointer[t] = wanted * inverseCount; fi.pointer[t] = 0
                    }
                }
                error += Double(patternError)
                fft.inverse(aRe: fr.pointer, aIm: fi.pointer, bRe: ar.pointer, bIm: ai.pointer)
                for k in 0..<n {
                    let j = indices[k]
                    let s0 = sr.pointer[k], s1 = si.pointer[k], f0 = ar.pointer[k], f1 = ai.pointer[k]
                    let o0 = or.pointer[j], o1 = oi.pointer[j]
                    onr.pointer[j] += s0 * f0 + s1 * f1; oni.pointer[j] += s0 * f1 - s1 * f0
                    pn.pointer[j] += s0 * s0 + s1 * s1
                    pnr.pointer[k] += o0 * f0 + o1 * f1; pni.pointer[k] += o0 * f1 - o1 * f0
                    on.pointer[k] += o0 * o0 + o1 * o1
                }
            }
            var maxPN: Float = 0, maxON: Float = 0
            vDSP_maxv(pn.pointer, 1, &maxPN, vDSP_Length(objectCount)); vDSP_maxv(on.pointer, 1, &maxON, vDSP_Length(n))
            for j in 0..<objectCount {
                let s = Float(stepSize) * inverse(pn.pointer[j], maxPN)
                or.pointer[j] += s * onr.pointer[j]; oi.pointer[j] += s * oni.pointer[j]
            }
            if constrainObjectAmplitude || purePhaseObject { for j in 0..<objectCount {
                let a0 = or.pointer[j], a1 = oi.pointer[j]
                let amplitude = (a0 * a0 + a1 * a1).squareRoot()
                let wanted: Float = purePhaseObject ? 1 : min(amplitude, 1)
                if amplitude > 0 { let scale = wanted / amplitude; or.pointer[j] = a0 * scale; oi.pointer[j] = a1 * scale }
                else { or.pointer[j] = wanted; oi.pointer[j] = 0 }
            } }
            if !fixProbe {
                for k in 0..<n {
                    let s = Float(stepSize) * inverse(on.pointer[k], maxON)
                    pr.pointer[k] += s * pnr.pointer[k]; pi.pointer[k] += s * pni.pointer[k]
                }
                if fixProbeCenterOfMass || fixProbeAperture || constrainProbeAmplitude {
                    var re = pr.doubles, im = pi.doubles
                    if fixProbeCenterOfMass { centerOfMassConstraint(&re, &im, doubleFFT) }
                    if fixProbeAperture { apertureConstraint(&re, &im, doubleFFT, aperture: initialAperture) }
                    if constrainProbeAmplitude {
                        amplitudeConstraint(&re, &im, rows: rows, columns: columns,
                                            radius: probeAmplitudeRelativeRadius, width: probeAmplitudeRelativeWidth)
                    }
                    for k in 0..<n { pr.pointer[k] = Float(re[k]); pi.pointer[k] = Float(im[k]) }
                }
            }
            if let mask = probeSupportMask { for k in 0..<n where !mask[k] { pr.pointer[k] = 0; pi.pointer[k] = 0 } }
            errors.append(error / divisor)
            progress?(iteration + 1, iterations)
        }
        return try Result(object: PtychographyArray(rows: objectRows, columns: objectColumns, real: or.doubles, imaginary: oi.doubles),
                          probe: PtychographyArray(rows: rows, columns: columns, real: pr.doubles, imaginary: pi.doubles),
                          errors: errors, positions: positions)
    }

    static func checkCancelled(_ isCancelled: (() -> Bool)?) throws {
        if isCancelled?() == true { throw CancellationError() }
    }
}
