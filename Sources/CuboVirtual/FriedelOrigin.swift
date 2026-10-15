//
//  FriedelOrigin.swift
//  Per-position origins from Friedel symmetry (registration slice 4, 3.3): py4DSTEM's get_origin_friedel
//  (process/calibration/origin.py 332 at f050d207) and its DFT upsampling (process/utils/multicorr.py
//  upsampled_correlation 22, dftUpsample 145), CPU, Double. A pattern is centrosymmetric about the unscattered beam, so
//  the correlation of the zero-padded pattern with its point reflection, Re(IFFT2(FFT2(im)²)), peaks at twice the origin.
//  With a mask (true in the pattern, false under a beamstop) py4DSTEM's three-term masked correlation.
//
//  Seeds: mac4DSTEM 7f192da9 Core/Analysis/FriedelOrigin.swift (Float32, NaN filled by a neighbour median, a zero
//  denominator read as 0); the kit follows py4DSTEM in Double and IEEE arithmetic.
//  DEVIATION: the parabola's neighbours wrap at both ends (py4DSTEM wraps x − 1 = −1 and raises IndexError at x + 1 = 2Q).
//

import CuboCore
import Foundation

public enum FriedelOrigin {
    /// The Friedel origin of every pattern. `mask` (detector-sized, row-major) is true in the pattern and false under a
    /// beamstop (the negation of `BeamstopMask.make`'s result); `upsampleFactor` 1 (none) or above 2 (py4DSTEM asserts
    /// > 2 when upsampling).
    public static func measure(of source: some CubeSource, mask: [Bool]? = nil, upsampleFactor: Int = 1,
                               tiling: ScanTiling = .automatic) async throws -> OriginMap {
        let shape = source.shape
        let plan = try Plan(rows: shape.detectorRows, columns: shape.detectorColumns, mask: mask, upsampleFactor: upsampleFactor)
        let n = shape.detectorCount
        var rows = [Double](repeating: 0, count: shape.scanCount), cols = rows
        for range in try tiling.ranges(for: shape) {
            try Task.checkCancellation()
            let tile = try await source.readScanTile(rows: range)
            let first = range.lowerBound * shape.scanColumns
            let count = tile.positionCount, values = tile.values
            let found = parallelMap(count) { k in plan.origin((0..<n).map { Double(values[k * n + $0]) }) }
            for k in 0..<count { rows[first + k] = found[k].0; cols[first + k] = found[k].1 }
        }
        let img = { (v: [Double]) in ScanImage(scanRows: shape.scanRows, scanColumns: shape.scanColumns, values: v, path: .cpu) }
        return try OriginMap(row: img(rows), column: img(cols))
    }

    /// The Friedel origin of one pattern (row, column).
    public static func origin(of pattern: DetectorImage, mask: [Bool]? = nil, upsampleFactor: Int = 1) throws
        -> (row: Double, column: Double) {
        let plan = try Plan(rows: pattern.detectorRows, columns: pattern.detectorColumns, mask: mask,
                            upsampleFactor: upsampleFactor)
        let o = plan.origin(pattern.values)
        return (o.0, o.1)
    }

    /// Everything that depends on the detector and the mask only, shared by every pattern (and across threads).
    struct Plan: Sendable {
        let qy: Int, qx: Int, ny: Int, nx: Int, u: Int
        let fft: FFT2D
        let maskPad: [Double]?                      // the mask at the top-left, 1 in the padding (py4DSTEM pads with ones)
        let mRe: [Double], mIm: [Double]            // M = FFT2(maskPad)
        let m2Re: [Double], m2Im: [Double]          // IFFT2(M²)

        init(rows qy: Int, columns qx: Int, mask: [Bool]?, upsampleFactor u: Int) throws {
            guard u == 1 || u > 2 else {
                throw CuboVirtualError.invalidInput("the upsampling factor must be 1 or above 2 (got \(u))")
            }
            if let mask, mask.count != qy * qx {
                throw CuboVirtualError.maskLengthMismatch(expected: qy * qx, received: mask.count)
            }
            self.qy = qy; self.qx = qx; ny = 2 * qy; nx = 2 * qx; self.u = u
            fft = FFT2D(rows: 2 * qy, columns: 2 * qx)
            if let mask {
                var mp = [Double](repeating: 1, count: 4 * qy * qx)
                for i in 0..<qy { for j in 0..<qx { mp[i * 2 * qx + j] = mask[i * qx + j] ? 1 : 0 } }
                var re = mp, im = [Double](repeating: 0, count: mp.count)
                fft.transform(&re, &im, inverse: false)
                var sr = re, si = im
                Self.square(&sr, &si)
                fft.transform(&sr, &si, inverse: true)
                maskPad = mp; mRe = re; mIm = im; m2Re = sr; m2Im = si
            } else {
                maskPad = nil; mRe = []; mIm = []; m2Re = []; m2Im = []
            }
        }

        static func square(_ re: inout [Double], _ im: inout [Double]) {
            for k in re.indices { let a = re[k], b = im[k]; re[k] = a * a - b * b; im[k] = 2 * a * b }
        }

        /// The definition of 3.3 for one pattern (qy × qx, row-major).
        func origin(_ p: [Double]) -> (Double, Double) {
            let total = ny * nx
            var re = [Double](repeating: 0, count: total), im = re
            for i in 0..<qy { for j in 0..<qx { re[i * nx + j] = p[i * qx + j] } }
            let pad = re
            fft.transform(&re, &im, inverse: false)
            Self.square(&re, &im)
            fft.transform(&re, &im, inverse: true)          // IFFT2(FFT2(im)²)
            var cc = [Double](repeating: 0, count: total)
            if let maskPad {
                var sq = pad.map { $0 * $0 }, si = [Double](repeating: 0, count: total)
                fft.transform(&sq, &si, inverse: false)
                for k in 0..<total {
                    let a = sq[k], b = si[k]
                    sq[k] = a * mRe[k] - b * mIm[k]; si[k] = a * mIm[k] + b * mRe[k]
                }
                fft.transform(&sq, &si, inverse: true)      // IFFT2(FFT2(im²) · M)
                for k in 0..<total {
                    let term1 = re[k] * m2Re[k] - im[k] * m2Im[k]
                    let term3 = pad[k] * maskPad[k]
                    cc[k] = (term1 - term3) / (sq[k] - term3)
                }
            } else {
                cc = re
            }
            let peak = Self.argmax(cc)
            let x = peak / nx, y = peak % nx
            let at = { (r: Int, c: Int) in cc[((r % ny + ny) % ny) * nx + (c % nx + nx) % nx] }
            var sx = Double(x) + Self.parabola(cc[peak], at(x + 1, y), at(x - 1, y))
            var sy = Double(y) + Self.parabola(cc[peak], at(x, y + 1), at(x, y - 1))
            if u > 2 { (sx, sy) = upsampled(cc, sx, sy) }
            return (Self.remainder(sx / 2, Double(qy)), Self.remainder(sy / 2, Double(qx)))
        }

        /// py4DSTEM's upsampled_correlation(FFT2(cc), u, (sx, sy)) (3.3).
        func upsampled(_ cc: [Double], _ x: Double, _ y: Double) -> (Double, Double) {
            let ud = Double(u)
            let sx = (x * ud).rounded(.toNearestOrEven) / ud, sy = (y * ud).rounded(.toNearestOrEven) / ud
            let g = ((1.5 * ud).rounded(.up) / 2).rounded(.towardZero)
            let c0 = g - ud * sx, c1 = g - ud * sy
            let n = Int((1.5 * ud).rounded(.up))
            var cr = cc, ci = [Double](repeating: 0, count: cc.count)
            fft.transform(&cr, &ci, inverse: false)          // C = FFT2(cc); conj(C) below
            let fr = Self.frequencies(ny), fc = Self.frequencies(nx)
            // T = conj(C) · K, ny × n; K[k, b] = exp(−2πi/(nx u) · fc[k] (b − c1))
            let kc = -2 * Double.pi / (Double(nx) * ud), kr = -2 * Double.pi / (Double(ny) * ud)
            var kRe = [Double](repeating: 0, count: nx * n), kIm = kRe
            for k in 0..<nx { for b in 0..<n {
                let a = kc * (fc[k] * (Double(b) - c1)); kRe[k * n + b] = cos(a); kIm[k * n + b] = sin(a)
            } }
            var tRe = [Double](repeating: 0, count: ny * n), tIm = tRe
            for r in 0..<ny { for b in 0..<n {
                var sre = 0.0, sim = 0.0
                for k in 0..<nx {
                    let a = cr[r * nx + k], bb = -ci[r * nx + k], c = kRe[k * n + b], d = kIm[k * n + b]
                    sre += a * c - bb * d; sim += a * d + bb * c
                }
                tRe[r * n + b] = sre; tIm[r * n + b] = sim
            } }
            // U = Re(R · T), n × n; R[a, k] = exp(−2πi/(ny u) · (a − c0) fr[k])
            var U = [Double](repeating: 0, count: n * n)
            for a in 0..<n {
                var rRe = [Double](repeating: 0, count: ny), rIm = rRe
                for k in 0..<ny { let t = kr * ((Double(a) - c0) * fr[k]); rRe[k] = cos(t); rIm[k] = sin(t) }
                for b in 0..<n {
                    var s = 0.0
                    for k in 0..<ny { s += rRe[k] * tRe[k * n + b] - rIm[k] * tIm[k * n + b] }
                    U[a * n + b] = s
                }
            }
            let best = Self.argmax(U)
            let a = best / n, b = best % n
            var ex = 0.0, ey = 0.0
            if a > 0, a < n - 1, b > 0, b < n - 1 {     // py4DSTEM's except branch at the edge: both zero
                ex = Self.parabola(U[best], U[best + n], U[best - n])
                ey = Self.parabola(U[best], U[best + 1], U[best - 1])
            }
            return (sx + ((Double(a) - g) + ex) / ud, sy + ((Double(b) - g) + ey) / ud)
        }

        /// ifftshift(arange(n)) − floor(n / 2): the signed frequency index in FFT order.
        static func frequencies(_ n: Int) -> [Double] {
            (0..<n).map { k in Double((k + n / 2) % n) - Double(n / 2) }
        }

        /// NumPy's argmax: the first maximum in row-major order, or the first NaN.
        static func argmax(_ v: [Double]) -> Int {
            if let nan = v.firstIndex(where: \.isNaN) { return nan }
            var best = 0
            for i in 1..<v.count where v[i] > v[best] { best = i }
            return best
        }

        /// py4DSTEM's `(c₊ − c₋) / (4 c₀ − 2 c₊ − 2 c₋)` in IEEE arithmetic (a zero denominator gives ±∞ or NaN, as NumPy's
        /// float64 scalars).
        static func parabola(_ c0: Double, _ plus: Double, _ minus: Double) -> Double {
            (plus - minus) / (4.0 * c0 - 2.0 * plus - 2.0 * minus)
        }

        /// NumPy's np.remainder (npy_divmod): the result takes the divisor's sign.
        static func remainder(_ a: Double, _ b: Double) -> Double {
            var mod = fmod(a, b)
            if mod != 0 { if (b < 0) != (mod < 0) { mod += b } } else { mod = copysign(0, b) }
            return mod
        }
    }
}
