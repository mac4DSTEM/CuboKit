//
//  MINPACKLeastSquares.swift
//  MINPACK's `lmdif` (Levenberg-Marquardt with a forward-difference Jacobian), the routine behind `scipy.optimize.leastsq` and so
//  behind py4DSTEM's ellipse fits. A transcription of the reference algorithm (`lmdif`, `fdjac2`, `qrfac`, `lmpar`, `qrsolv`,
//  `enorm`) in Double with SciPy's defaults, so that a fit starting from the same point takes the same steps and stops at the same
//  place: `ftol = xtol = 1.49012e-8`, `gtol = 0`, `maxfev = 200 (n + 1)`, `epsfcn = machine epsilon`, `factor = 100`, `mode = 1`
//  (the variables are scaled by the column norms of the first Jacobian).
//

import Foundation

package enum MINPACKLeastSquares {
    /// Why `lmdif` stopped (`info`): 1 to 4 are convergence, 5 to 8 are limits.
    package struct Result: Sendable {
        package let x: [Double]
        package let info: Int
        package let evaluations: Int
        package var converged: Bool { (1...4).contains(info) }
    }

    package enum Failure: Error, Equatable, Sendable {
        /// `m < n`: fewer residuals than parameters (`leastsq` raises a `TypeError`).
        case fewerResidualsThanParameters(residuals: Int, parameters: Int)
    }

    private static let epsmch = Double.ulpOfOne          // dpmpar(1)
    private static let dwarf = Double.leastNormalMagnitude   // dpmpar(2)

    /// `scipy.optimize.leastsq(func, x0)` with the default tolerances. `func` returns the residual vector for a parameter vector;
    /// `shouldStop` is asked before every evaluation and throws `CancellationError` when it says so.
    package static func leastSquares(_ x0: [Double], ftol: Double = 1.49012e-8, xtol: Double = 1.49012e-8, gtol: Double = 0,
                                     maxEvaluations: Int = 0, factor: Double = 100,
                                     shouldStop: (() -> Bool)? = nil,
                                     residuals fcn: ([Double]) -> [Double]) throws -> Result {
        let n = x0.count
        var x = x0
        func evaluate(_ p: [Double]) throws -> [Double] {
            if shouldStop?() == true { throw CancellationError() }
            return fcn(p)
        }
        var fvec = try evaluate(x)
        let m = fvec.count
        guard m >= n else { throw Failure.fewerResidualsThanParameters(residuals: m, parameters: n) }
        let maxfev = maxEvaluations > 0 ? maxEvaluations : 200 * (n + 1)
        var nfev = 1
        var info = 0
        var fjac = [Double](repeating: 0, count: m * n)       // column-major, ldfjac = m
        var ipvt = [Int](repeating: 0, count: n)
        var diag = [Double](repeating: 0, count: n)
        var qtf = [Double](repeating: 0, count: n)
        var wa1 = qtf, wa2 = qtf, wa3 = qtf
        var wa4 = [Double](repeating: 0, count: m)
        var fnorm = enorm(fvec)
        var par = 0.0
        var iter = 1
        var delta = 0.0, xnorm = 0.0
        let eps = max(epsmch, epsmch).squareRoot()            // epsfcn = machine epsilon

        outer: while true {
            // the Jacobian by forward differences (fdjac2)
            for j in 0..<n {
                let temp = x[j]
                var h = eps * abs(temp)
                if h == 0 { h = eps }
                x[j] = temp + h
                let wa = try evaluate(x)
                x[j] = temp
                for i in 0..<m { fjac[i + j * m] = (wa[i] - fvec[i]) / h }
            }
            nfev += n
            qrfac(m, n, &fjac, &ipvt, &wa1, &wa2, &wa3)
            if iter == 1 {
                for j in 0..<n { diag[j] = wa2[j] == 0 ? 1 : wa2[j] }
                for j in 0..<n { wa3[j] = diag[j] * x[j] }
                xnorm = enorm(wa3)
                delta = factor * xnorm
                if delta == 0 { delta = factor }
            }
            // Q^T fvec, first n components in qtf
            wa4 = fvec
            for j in 0..<n {
                if fjac[j + j * m] != 0 {
                    var sum = 0.0
                    for i in j..<m { sum += fjac[i + j * m] * wa4[i] }
                    let temp = -sum / fjac[j + j * m]
                    for i in j..<m { wa4[i] += fjac[i + j * m] * temp }
                }
                fjac[j + j * m] = wa1[j]
                qtf[j] = wa4[j]
            }
            // the norm of the scaled gradient
            var gnorm = 0.0
            if fnorm != 0 {
                for j in 0..<n {
                    let l = ipvt[j]
                    if wa2[l] != 0 {
                        var sum = 0.0
                        for i in 0...j { sum += fjac[i + j * m] * (qtf[i] / fnorm) }
                        gnorm = max(gnorm, abs(sum / wa2[l]))
                    }
                }
            }
            if gnorm <= gtol { info = 4 }
            if info != 0 { break outer }
            for j in 0..<n { diag[j] = max(diag[j], wa2[j]) }
            // the inner loop
            var ratio = 0.0
            repeat {
                lmpar(n, m, &fjac, ipvt, diag, qtf, delta, &par, &wa1, &wa2, &wa3, &wa4)
                for j in 0..<n { wa1[j] = -wa1[j]; wa2[j] = x[j] + wa1[j]; wa3[j] = diag[j] * wa1[j] }
                let pnorm = enorm(wa3)
                if iter == 1 { delta = min(delta, pnorm) }
                wa4 = try evaluate(wa2)
                nfev += 1
                let fnorm1 = enorm(wa4)
                var actred = -1.0
                if 0.1 * fnorm1 < fnorm { actred = 1 - (fnorm1 / fnorm) * (fnorm1 / fnorm) }
                for j in 0..<n { wa3[j] = 0 }
                for j in 0..<n {
                    let l = ipvt[j], temp = wa1[l]
                    for i in 0...j { wa3[i] += fjac[i + j * m] * temp }
                }
                let temp1 = enorm(wa3) / fnorm
                let temp2 = (par.squareRoot() * pnorm) / fnorm
                let prered = temp1 * temp1 + temp2 * temp2 / 0.5
                let dirder = -(temp1 * temp1 + temp2 * temp2)
                ratio = prered != 0 ? actred / prered : 0
                if ratio <= 0.25 {
                    var temp = actred >= 0 ? 0.5 : 0.5 * dirder / (dirder + 0.5 * actred)
                    if 0.1 * fnorm1 >= fnorm || temp < 0.1 { temp = 0.1 }
                    delta = temp * min(delta, pnorm / 0.1)
                    par /= temp
                } else if par == 0 || ratio >= 0.75 {
                    delta = pnorm / 0.5
                    par = 0.5 * par
                }
                if ratio >= 1e-4 {
                    for j in 0..<n { x[j] = wa2[j]; wa2[j] = diag[j] * x[j] }
                    for i in 0..<m { fvec[i] = wa4[i] }
                    xnorm = enorm(wa2)
                    fnorm = fnorm1
                    iter += 1
                }
                if abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1 { info = 1 }
                if delta <= xtol * xnorm { info = 2 }
                if abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1 && info == 2 { info = 3 }
                if info != 0 { break outer }
                if nfev >= maxfev { info = 5 }
                if abs(actred) <= epsmch && prered <= epsmch && 0.5 * ratio <= 1 { info = 6 }
                if delta <= epsmch * xnorm { info = 7 }
                if gnorm <= epsmch { info = 8 }
                if info != 0 { break outer }
            } while ratio < 1e-4
        }
        return Result(x: x, info: info, evaluations: nfev)
    }

    // MARK: enorm

    /// The Euclidean norm with MINPACK's three accumulators (no overflow or underflow for extreme components).
    package static func enorm(_ x: [Double]) -> Double {
        let rdwarf = 3.834e-20, rgiant = 1.304e19
        var s1 = 0.0, s2 = 0.0, s3 = 0.0, x1max = 0.0, x3max = 0.0
        let agiant = rgiant / Double(x.count)
        for v in x {
            let xabs = abs(v)
            if xabs > rdwarf && xabs < agiant {
                s2 += xabs * xabs
            } else if xabs <= rdwarf {
                if xabs <= x3max {
                    if xabs != 0 { s3 += (xabs / x3max) * (xabs / x3max) }
                } else {
                    s3 = 1 + s3 * (x3max / xabs) * (x3max / xabs)
                    x3max = xabs
                }
            } else {
                if xabs <= x1max {
                    s1 += (xabs / x1max) * (xabs / x1max)
                } else {
                    s1 = 1 + s1 * (x1max / xabs) * (x1max / xabs)
                    x1max = xabs
                }
            }
        }
        if s1 != 0 { return x1max * (s1 + (s2 / x1max) / x1max).squareRoot() }
        if s2 != 0 {
            if s2 >= x3max { return (s2 * (1 + (x3max / s2) * (x3max * s3))).squareRoot() }
            return (x3max * ((s2 / x3max) + (x3max * s3))).squareRoot()
        }
        return x3max * s3.squareRoot()
    }

    private static func enorm(_ a: [Double], from start: Int, count: Int) -> Double {
        enorm(Array(a[start..<(start + count)]))
    }

    // MARK: qrfac

    /// QR factorisation with column pivoting of the `m x n` column-major `a`; `rdiag` holds the diagonal of R, `acnorm` the
    /// original column norms, `ipvt` the permutation.
    private static func qrfac(_ m: Int, _ n: Int, _ a: inout [Double], _ ipvt: inout [Int], _ rdiag: inout [Double],
                              _ acnorm: inout [Double], _ wa: inout [Double]) {
        for j in 0..<n {
            acnorm[j] = enorm(a, from: j * m, count: m)
            rdiag[j] = acnorm[j]
            wa[j] = rdiag[j]
            ipvt[j] = j
        }
        for j in 0..<min(m, n) {
            var kmax = j
            for k in j..<n where rdiag[k] > rdiag[kmax] { kmax = k }
            if kmax != j {
                for i in 0..<m { a.swapAt(i + j * m, i + kmax * m) }
                rdiag[kmax] = rdiag[j]
                wa[kmax] = wa[j]
                ipvt.swapAt(j, kmax)
            }
            var ajnorm = enorm(a, from: j + j * m, count: m - j)
            if ajnorm != 0 {
                if a[j + j * m] < 0 { ajnorm = -ajnorm }
                for i in j..<m { a[i + j * m] /= ajnorm }
                a[j + j * m] += 1
                if j + 1 < n {
                    for k in (j + 1)..<n {
                        var sum = 0.0
                        for i in j..<m { sum += a[i + j * m] * a[i + k * m] }
                        let temp = sum / a[j + j * m]
                        for i in j..<m { a[i + k * m] -= temp * a[i + j * m] }
                        if rdiag[k] != 0 {
                            let t = a[j + k * m] / rdiag[k]
                            rdiag[k] *= max(0, 1 - t * t).squareRoot()
                            let ratio = rdiag[k] / wa[k]
                            if 0.05 * ratio * ratio <= epsmch {
                                rdiag[k] = enorm(a, from: j + 1 + k * m, count: m - j - 1)
                                wa[k] = rdiag[k]
                            }
                        }
                    }
                }
            }
            rdiag[j] = -ajnorm
        }
    }

    // MARK: lmpar

    private static func lmpar(_ n: Int, _ ldr: Int, _ r: inout [Double], _ ipvt: [Int], _ diag: [Double], _ qtb: [Double],
                              _ delta: Double, _ par: inout Double, _ x: inout [Double], _ sdiag: inout [Double],
                              _ wa1: inout [Double], _ wa2: inout [Double]) {
        // the Gauss-Newton direction
        var nsing = n
        for j in 0..<n {
            wa1[j] = qtb[j]
            if r[j + j * ldr] == 0 && nsing == n { nsing = j }
            if nsing < n { wa1[j] = 0 }
        }
        if nsing >= 1 {
            for k in 0..<nsing {
                let j = nsing - 1 - k
                wa1[j] /= r[j + j * ldr]
                let temp = wa1[j]
                for i in 0..<j { wa1[i] -= r[i + j * ldr] * temp }
            }
        }
        for j in 0..<n { x[ipvt[j]] = wa1[j] }
        var iter = 0
        for j in 0..<n { wa2[j] = diag[j] * x[j] }
        var dxnorm = enorm(wa2, from: 0, count: n)   // wa2 may be longer than n (lmdif passes an m-vector)
        var fp = dxnorm - delta
        if fp <= 0.1 * delta {
            par = 0
            return
        }
        var parl = 0.0
        if nsing >= n {
            for j in 0..<n {
                let l = ipvt[j]
                wa1[j] = diag[l] * (wa2[l] / dxnorm)
            }
            for j in 0..<n {
                var sum = 0.0
                for i in 0..<j { sum += r[i + j * ldr] * wa1[i] }
                wa1[j] = (wa1[j] - sum) / r[j + j * ldr]
            }
            let temp = enorm(wa1)
            parl = ((fp / delta) / temp) / temp
        }
        for j in 0..<n {
            var sum = 0.0
            for i in 0...j { sum += r[i + j * ldr] * qtb[i] }
            wa1[j] = sum / diag[ipvt[j]]
        }
        let gnorm = enorm(wa1)
        var paru = gnorm / delta
        if paru == 0 { paru = dwarf / min(delta, 0.1) }
        par = max(par, parl)
        par = min(par, paru)
        if par == 0 { par = gnorm / dxnorm }
        while true {
            iter += 1
            if par == 0 { par = max(dwarf, 0.001 * paru) }
            let temp = par.squareRoot()
            for j in 0..<n { wa1[j] = temp * diag[j] }
            qrsolv(n, ldr, &r, ipvt, wa1, qtb, &x, &sdiag, &wa2)
            for j in 0..<n { wa2[j] = diag[j] * x[j] }
            dxnorm = enorm(wa2, from: 0, count: n)
            let prev = fp
            fp = dxnorm - delta
            if abs(fp) <= 0.1 * delta || (parl == 0 && fp <= prev && prev < 0) || iter == 10 { break }
            for j in 0..<n {
                let l = ipvt[j]
                wa1[j] = diag[l] * (wa2[l] / dxnorm)
            }
            for j in 0..<n {
                wa1[j] /= sdiag[j]
                let t = wa1[j]
                if j + 1 < n { for i in (j + 1)..<n { wa1[i] -= r[i + j * ldr] * t } }
            }
            let t2 = enorm(wa1)
            let parc = ((fp / delta) / t2) / t2
            if fp > 0 { parl = max(parl, par) }
            if fp < 0 { paru = min(paru, par) }
            par = max(parl, par + parc)
        }
        if iter == 0 { par = 0 }
    }

    // MARK: qrsolv

    private static func qrsolv(_ n: Int, _ ldr: Int, _ r: inout [Double], _ ipvt: [Int], _ diag: [Double], _ qtb: [Double],
                               _ x: inout [Double], _ sdiag: inout [Double], _ wa: inout [Double]) {
        for j in 0..<n {
            for i in j..<n { r[i + j * ldr] = r[j + i * ldr] }
            x[j] = r[j + j * ldr]
            wa[j] = qtb[j]
        }
        for j in 0..<n {
            let l = ipvt[j]
            if diag[l] != 0 {
                for k in j..<n { sdiag[k] = 0 }
                sdiag[j] = diag[l]
                var qtbpj = 0.0
                for k in j..<n {
                    if sdiag[k] == 0 { continue }
                    let sin: Double, cos: Double
                    if abs(r[k + k * ldr]) < abs(sdiag[k]) {
                        let cotan = r[k + k * ldr] / sdiag[k]
                        sin = 0.5 / (0.25 + 0.25 * cotan * cotan).squareRoot()
                        cos = sin * cotan
                    } else {
                        let tan = sdiag[k] / r[k + k * ldr]
                        cos = 0.5 / (0.25 + 0.25 * tan * tan).squareRoot()
                        sin = cos * tan
                    }
                    r[k + k * ldr] = cos * r[k + k * ldr] + sin * sdiag[k]
                    let temp = cos * wa[k] + sin * qtbpj
                    qtbpj = -sin * wa[k] + cos * qtbpj
                    wa[k] = temp
                    if k + 1 < n {
                        for i in (k + 1)..<n {
                            let t = cos * r[i + k * ldr] + sin * sdiag[i]
                            sdiag[i] = -sin * r[i + k * ldr] + cos * sdiag[i]
                            r[i + k * ldr] = t
                        }
                    }
                }
            }
            sdiag[j] = r[j + j * ldr]
            r[j + j * ldr] = x[j]
        }
        var nsing = n
        for j in 0..<n {
            if sdiag[j] == 0 && nsing == n { nsing = j }
            if nsing < n { wa[j] = 0 }
        }
        if nsing >= 1 {
            for k in 0..<nsing {
                let j = nsing - 1 - k
                var sum = 0.0
                if j + 1 < nsing { for i in (j + 1)..<nsing { sum += r[i + j * ldr] * wa[i] } }
                wa[j] = (wa[j] - sum) / sdiag[j]
            }
        }
        for j in 0..<n { x[ipvt[j]] = wa[j] }
    }
}
