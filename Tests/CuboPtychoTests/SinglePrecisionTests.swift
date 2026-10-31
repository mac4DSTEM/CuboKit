import Foundation
import Testing
import CuboPtycho

/// 0.7.0: the Float32 single-slice engine against the Double engine, on amplitudes that come from a known object (a smooth
/// phase object read through a Gaussian probe, naive DFT), so the data is consistent and the iteration is well conditioned.
/// Tolerances (float32, measured 2.9e-5 object / 1.5e-6 probe after 3 iterations on a unit-modulus object): object 1e-4
/// absolute, probe 1e-5 absolute, error history 1e-5 relative.
@Suite struct SinglePrecisionTests {
    static let d = 32, objectSize = 80, count = 36

    struct Scene {
        let amplitudes: [Double], positions: [PtychographyPosition], object: PtychographyArray, probe: PtychographyArray
    }

    static func scene() throws -> Scene {
        let d = Self.d, size = Self.objectSize, count = Self.count
        var probeRe = [Double](repeating: 0, count: d * d)
        for r in 0..<d { for c in 0..<d {
            let rr = Double(min(r, d - r)), cc = Double(min(c, d - c)); probeRe[r * d + c] = exp(-(rr * rr + cc * cc) / 40)
        } }
        let norm = probeRe.reduce(0) { $0 + $1 * $1 }.squareRoot()
        probeRe = probeRe.map { $0 / norm }
        var truthRe = [Double](repeating: 1, count: size * size), truthIm = [Double](repeating: 0, count: size * size)
        for r in 0..<size { for c in 0..<size {
            let phase = 0.6 * sin(Double(r) / 9) * cos(Double(c) / 7) + 0.3 * sin(Double(r + c) / 5)
            truthRe[r * size + c] = 0.9 * cos(phase); truthIm[r * size + c] = 0.9 * sin(phase)
        } }
        var positions = [PtychographyPosition]()
        for i in 0..<count { positions.append(PtychographyPosition(row: Double(20 + (i / 6) * 8), column: Double(20 + (i % 6) * 8))) }
        let twiddle = (0..<d).map { k in (cos(-2 * Double.pi * Double(k) / Double(d)), sin(-2 * Double.pi * Double(k) / Double(d))) }
        var amplitudes = [Double](repeating: 0, count: count * d * d)
        for p in 0..<count {
            var re = [Double](repeating: 0, count: d * d), im = re
            for r in 0..<d { for c in 0..<d {
                let ro = r < d / 2 ? r : r - d, co = c < d / 2 ? c : c - d
                let j = ((Int(positions[p].row) + ro + size) % size) * size + (Int(positions[p].column) + co + size) % size
                re[r * d + c] = probeRe[r * d + c] * truthRe[j]; im[r * d + c] = probeRe[r * d + c] * truthIm[j]
            } }
            var tr = re, ti = im
            for r in 0..<d { for k in 0..<d {
                var sr = 0.0, si = 0.0
                for c in 0..<d { let w = twiddle[(k * c) % d]; sr += re[r * d + c] * w.0 - im[r * d + c] * w.1; si += re[r * d + c] * w.1 + im[r * d + c] * w.0 }
                tr[r * d + k] = sr; ti[r * d + k] = si
            } }
            for c in 0..<d { for k in 0..<d {
                var sr = 0.0, si = 0.0
                for r in 0..<d { let w = twiddle[(k * r) % d]; sr += tr[r * d + c] * w.0 - ti[r * d + c] * w.1; si += tr[r * d + c] * w.1 + ti[r * d + c] * w.0 }
                amplitudes[p * d * d + k * d + c] = (sr * sr + si * si).squareRoot()
            } }
        }
        return try Scene(amplitudes: amplitudes, positions: positions,
                         object: PtychographyArray(rows: size, columns: size, real: [Double](repeating: 1, count: size * size),
                                                   imaginary: [Double](repeating: 0, count: size * size)),
                         probe: PtychographyArray(rows: d, columns: d, real: probeRe, imaginary: [Double](repeating: 0, count: d * d)))
    }

    @Test func singleAgreesWithDouble() throws {
        let s = try Self.scene()
        let reference = try SingleslicePtychography.reconstruct(amplitudes: s.amplitudes, positions: s.positions, object: s.object,
                                                                probe: s.probe, iterations: 3)
        let single = try SingleslicePtychography.reconstruct(amplitudes: s.amplitudes, positions: s.positions, object: s.object,
                                                             probe: s.probe, iterations: 3, precision: .single)
        func worst(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).map { abs($0 - $1) }.max()! }
        #expect(worst(reference.object.real, single.object.real) <= 1e-4)
        #expect(worst(reference.object.imaginary, single.object.imaginary) <= 1e-4)
        #expect(worst(reference.probe.real, single.probe.real) <= 1e-5)
        #expect(worst(reference.probe.imaginary, single.probe.imaginary) <= 1e-5)
        for (a, b) in zip(reference.errors, single.errors) { #expect(abs(a - b) <= 1e-5 * abs(a)) }
        #expect(single.errors[2] < single.errors[0])      // it converges
        #expect(single.positions.count == s.positions.count)
    }

    @Test func singleHonoursTheOptions() throws {
        let s = try Self.scene()
        let a = try SingleslicePtychography.reconstruct(amplitudes: s.amplitudes, positions: s.positions, object: s.object, probe: s.probe,
                                                        iterations: 2, purePhaseObject: true, fixProbeCenterOfMass: true,
                                                        constrainProbeAmplitude: true)
        let b = try SingleslicePtychography.reconstructSingle(amplitudes: s.amplitudes.map { Float($0) }, positions: s.positions,
                                                              object: s.object, probe: s.probe, iterations: 2, purePhaseObject: true,
                                                              fixProbeCenterOfMass: true, constrainProbeAmplitude: true)
        #expect(zip(a.object.real, b.object.real).map { abs($0 - $1) }.max()! <= 1e-4)
        #expect(zip(a.probe.imaginary, b.probe.imaginary).map { abs($0 - $1) }.max()! <= 1e-4)
        // pure phase: unit modulus
        let modulus = zip(b.object.real, b.object.imaginary).map { ($0 * $0 + $1 * $1).squareRoot() }
        #expect(modulus.allSatisfy { abs($0 - 1) < 1e-5 })
    }

    @Test func cancellationProgressAndRefusals() throws {
        let s = try Self.scene()
        nonisolated(unsafe) var seen = [Int]()
        _ = try SingleslicePtychography.reconstruct(amplitudes: s.amplitudes, positions: s.positions, object: s.object, probe: s.probe,
                                                    iterations: 2, progress: { done, total in #expect(total == 2); seen.append(done) })
        #expect(seen == [1, 2])
        for precision in [PtychographyPrecision.double, .single] {
            nonisolated(unsafe) var polls = 0
            #expect(throws: CancellationError.self) {
                try SingleslicePtychography.reconstruct(amplitudes: s.amplitudes, positions: s.positions, object: s.object, probe: s.probe,
                                                        iterations: 3, precision: precision, isCancelled: { polls += 1; return polls > 3 })
            }
        }
        #expect(throws: PtychographyError.invalidInput) {
            try SingleslicePtychography.reconstruct(amplitudes: s.amplitudes, positions: s.positions, object: s.object, probe: s.probe,
                                                    iterations: 1, positionCorrection: PtychographyPositionCorrection(), precision: .single)
        }
        // a 14-pixel detector is a length vDSP's DFT does not take: since CuboKit#9 it runs (Bluestein), it is not refused
        let small = try PtychographyArray(rows: 14, columns: 14, real: [Double](repeating: 0.1, count: 196), imaginary: [Double](repeating: 0, count: 196))
        let ran = try SingleslicePtychography.reconstruct(amplitudes: [Double](repeating: 1, count: 196), positions: [PtychographyPosition(row: 4, column: 4)],
                                                          object: s.object, probe: small, iterations: 1, precision: .single)
        #expect(ran.errors.count == 1 && ran.errors[0].isFinite)
        // a probe larger than the object is still refused
        let wide = try PtychographyArray(rows: 16, columns: 96, real: [Double](repeating: 0.1, count: 16 * 96), imaginary: [Double](repeating: 0, count: 16 * 96))
        #expect(throws: PtychographyError.invalidInput) {
            try SingleslicePtychography.reconstruct(amplitudes: [Double](repeating: 1, count: 16 * 96), positions: [PtychographyPosition(row: 4, column: 4)],
                                                    object: s.object, probe: wide, iterations: 1, precision: .single)
        }
    }
}

/// Fractional positions on a non-square detector (32 x 48, both lengths vDSP takes): a swapped row/column ramp or a transposed
/// layout error shows here and not on the square, integer-position scene. The amplitudes are the forward model of a known object
/// (naive separable DFT, the probe shifted by the Fourier ramp), so the iteration is well conditioned; the single path must agree
/// with the Double path (object 1e-4, probe 1e-5, errors 1e-4 relative).
@Suite struct SinglePrecisionNonSquareTests {
    /// Naive 2-D DFT, `e^{-+2 pi i ...}`, unnormalised both ways (the caller divides for the inverse).
    static func dft(_ re: [Double], _ im: [Double], rows: Int, columns: Int, sign: Double) -> ([Double], [Double]) {
        var tr = re, ti = im
        for r in 0..<rows { for k in 0..<columns {
            var sr = 0.0, si = 0.0
            for c in 0..<columns {
                let a = sign * 2 * Double.pi * Double(k * c % columns) / Double(columns), w = (cos(a), sin(a))
                sr += re[r * columns + c] * w.0 - im[r * columns + c] * w.1; si += re[r * columns + c] * w.1 + im[r * columns + c] * w.0
            }
            tr[r * columns + k] = sr; ti[r * columns + k] = si
        } }
        var outR = tr, outI = ti
        for c in 0..<columns { for k in 0..<rows {
            var sr = 0.0, si = 0.0
            for r in 0..<rows {
                let a = sign * 2 * Double.pi * Double(k * r % rows) / Double(rows), w = (cos(a), sin(a))
                sr += tr[r * columns + c] * w.0 - ti[r * columns + c] * w.1; si += tr[r * columns + c] * w.1 + ti[r * columns + c] * w.0
            }
            outR[k * columns + c] = sr; outI[k * columns + c] = si
        } }
        return (outR, outI)
    }

    @Test func nonSquareFractionalPositions() throws {
        try Self.agreesWithDouble(rows: 32, columns: 48, size: 96)
    }

    /// CuboKit#9: detector lengths the vDSP DFT does not take (an odd prime, 2 x 11, 3 x 7, the padded parallax and
    /// ptychography size 165 = 3 x 5 x 11) run Bluestein in Float32 and agree with the Double engine at the same tolerances.
    @Test(arguments: [(22, 21), (14, 14), (13, 26), (165, 40), (32, 21)])
    func bluesteinDetectorAgreesWithDouble(rows: Int, columns: Int) throws {
        try Self.agreesWithDouble(rows: rows, columns: columns, size: 2 * max(rows, columns) + 40)
    }

    /// The single path against the Double path on a `rows x columns` detector over a `size x size` object.
    static func agreesWithDouble(rows: Int, columns: Int, size: Int) throws {
        let n = rows * columns
        var probeRe = [Double](repeating: 0, count: n), probeIm = probeRe
        for r in 0..<rows { for c in 0..<columns {
            let rr = Double(min(r, rows - r)), cc = Double(min(c, columns - c)), g = exp(-(rr * rr / 20 + cc * cc / 60))
            probeRe[r * columns + c] = g; probeIm[r * columns + c] = 0.04 * rr * g
        } }
        let norm = zip(probeRe, probeIm).reduce(0) { $0 + $1.0 * $1.0 + $1.1 * $1.1 }.squareRoot()
        probeRe = probeRe.map { $0 / norm }; probeIm = probeIm.map { $0 / norm }
        let probe = try PtychographyArray(rows: rows, columns: columns, real: probeRe, imaginary: probeIm)
        var truthRe = [Double](repeating: 1, count: size * size), truthIm = [Double](repeating: 0, count: size * size)
        for r in 0..<size { for c in 0..<size {
            let p = 0.5 * sin(Double(r) / 6) * cos(Double(c) / 11) + 0.2 * sin(Double(r + 2 * c) / 8)
            truthRe[r * size + c] = 0.9 * cos(p); truthIm[r * size + c] = 0.9 * sin(p)
        } }
        let start = try PtychographyArray(rows: size, columns: size, real: [Double](repeating: 1, count: size * size),
                                          imaginary: [Double](repeating: 0, count: size * size))
        let positions = (0..<12).map { PtychographyPosition(row: 30 + Double($0 / 4) * 7.37, column: 30 + Double($0 % 4) * 9.61) }
        let (spectrumRe, spectrumIm) = Self.dft(probeRe, probeIm, rows: rows, columns: columns, sign: -1)
        var amplitudes = [Double](repeating: 0, count: positions.count * n)
        for (p, position) in positions.enumerated() {
            let rr = Int(position.row.rounded(.toNearestOrEven)), cc = Int(position.column.rounded(.toNearestOrEven))
            let dr = position.row - Double(rr), dc = position.column - Double(cc)
            var rampedRe = spectrumRe, rampedIm = spectrumIm
            for u in 0..<rows { for v in 0..<columns {
                let fu = Double(u < (rows + 1) / 2 ? u : u - rows) / Double(rows), fv = Double(v < (columns + 1) / 2 ? v : v - columns) / Double(columns)
                let a = -2 * Double.pi * (fu * dr + fv * dc), k = u * columns + v
                rampedRe[k] = spectrumRe[k] * cos(a) - spectrumIm[k] * sin(a); rampedIm[k] = spectrumRe[k] * sin(a) + spectrumIm[k] * cos(a)
            } }
            var (sr, si) = Self.dft(rampedRe, rampedIm, rows: rows, columns: columns, sign: 1)
            sr = sr.map { $0 / Double(n) }; si = si.map { $0 / Double(n) }
            var exitRe = sr, exitIm = si
            for u in 0..<rows { for v in 0..<columns {
                let ro = u < (rows + 1) / 2 ? u : u - rows, co = v < (columns + 1) / 2 ? v : v - columns
                let j = ((rr + ro + size) % size) * size + (cc + co + size) % size, k = u * columns + v
                exitRe[k] = sr[k] * truthRe[j] - si[k] * truthIm[j]; exitIm[k] = sr[k] * truthIm[j] + si[k] * truthRe[j]
            } }
            let (fr, fi) = Self.dft(exitRe, exitIm, rows: rows, columns: columns, sign: -1)
            for k in 0..<n { amplitudes[p * n + k] = (fr[k] * fr[k] + fi[k] * fi[k]).squareRoot() }
        }
        let reference = try SingleslicePtychography.reconstruct(amplitudes: amplitudes, positions: positions, object: start, probe: probe, iterations: 3)
        let single = try SingleslicePtychography.reconstruct(amplitudes: amplitudes, positions: positions, object: start, probe: probe,
                                                             iterations: 3, precision: .single)
        func gap(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).map { abs($0 - $1) }.max()! }
        #expect(gap(reference.object.real, single.object.real) <= 1e-4)
        #expect(gap(reference.object.imaginary, single.object.imaginary) <= 1e-4)
        #expect(gap(reference.probe.real, single.probe.real) <= 1e-5)
        #expect(gap(reference.probe.imaginary, single.probe.imaginary) <= 1e-5)
        for (a, b) in zip(reference.errors, single.errors) { #expect(abs(a - b) <= 1e-4 * abs(a)) }
        #expect(reference.errors[2] < reference.errors[0])
    }
}
