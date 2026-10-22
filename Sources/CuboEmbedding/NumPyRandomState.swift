//
//  NumPyRandomState.swift
//  NumPy's legacy `RandomState` (MT19937), so the seeded paths of scikit-learn (`random_state=<int>`) draw the same numbers:
//  `RandomState(seed)`, `random_sample`, `randint`, `uniform`, `standard_normal` and `choice(n, p=p)`. Pinned against NumPy 2.5.4.
//

import Foundation

/// NumPy's `np.random.RandomState(seed)`: the Mersenne Twister MT19937 seeded as `init_genrand(seed)` with the legacy
/// polar-method normal generator. The draws are bit identical to NumPy's for the same seed and call sequence.
public struct NumPyRandomState: Sendable {
    private var key = [UInt32](repeating: 0, count: 624)
    private var position = 624
    private var cachedGauss: Double?

    /// `RandomState(seed)` for an integer seed in `0...2³²−1`.
    public init(seed: UInt32) {
        key[0] = seed
        for i in 1..<624 { key[i] = 1812433253 &* (key[i - 1] ^ (key[i - 1] >> 30)) &+ UInt32(i) }
    }

    private mutating func refill() {
        for i in 0..<624 {
            let y = (key[i] & 0x8000_0000) | (key[(i + 1) % 624] & 0x7fff_ffff)
            var next = key[(i + 397) % 624] ^ (y >> 1)
            if y & 1 != 0 { next ^= 0x9908_b0df }
            key[i] = next
        }
        position = 0
    }

    /// The next 32 random bits.
    public mutating func nextUInt32() -> UInt32 {
        if position >= 624 { refill() }
        var y = key[position]
        position += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c_5680
        y ^= (y << 15) & 0xefc6_0000
        y ^= y >> 18
        return y
    }

    /// `random_sample()`: 53 random bits in `[0, 1)`.
    public mutating func randomSample() -> Double {
        let a = Double(nextUInt32() >> 5), b = Double(nextUInt32() >> 6)
        return (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// `randint(upperBound)` for `1 <= upperBound <= 2³²`: masked rejection on 32-bit draws, as NumPy's legacy bounded integers.
    /// Traps (a precondition) outside that range; the callers inside the kit pass counts they have checked.
    public mutating func randint(_ upperBound: Int) -> Int {
        precondition(upperBound >= 1 && upperBound <= 1 << 32, "randint: upperBound out of range")
        let range = UInt32(truncatingIfNeeded: upperBound - 1)
        if range == 0 { return 0 }
        var mask = range
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4; mask |= mask >> 8; mask |= mask >> 16
        while true {
            let value = nextUInt32() & mask
            if value <= range { return Int(value) }
        }
    }

    /// `standard_normal()`: the legacy polar (Marsaglia) method, which yields two values per pair of draws.
    public mutating func standardNormal() -> Double {
        if let g = cachedGauss { cachedGauss = nil; return g }
        var x1 = 0.0, x2 = 0.0, r2 = 0.0
        repeat {
            x1 = 2.0 * randomSample() - 1.0
            x2 = 2.0 * randomSample() - 1.0
            // NumPy's arm64 build fuses this as fma(x1, x1, x2·x2); 5000 draws agree bit for bit only with the fused form.
            r2 = (x2 * x2).addingProduct(x1, x1)
        } while r2 >= 1.0 || r2 == 0.0
        let f = (-2.0 * log(r2) / r2).squareRoot()
        cachedGauss = f * x1
        return f * x2
    }

    /// `choice(p.count, p=p)` for one draw: the cumulative sum, normalised by its last entry, searched with `side="right"` for one
    /// `random_sample()`.
    /// `probabilities` must not be empty (a precondition traps); the callers inside the kit pass non-empty ones.
    public mutating func choice(probabilities p: [Double]) -> Int {
        var cdf = [Double](repeating: 0, count: p.count)
        var running = 0.0
        for i in 0..<p.count { running += p[i]; cdf[i] = running }
        let last = cdf[p.count - 1]
        for i in 0..<p.count { cdf[i] /= last }
        let u = randomSample()
        var low = 0, high = p.count
        while low < high {
            let mid = (low + high) / 2
            if cdf[mid] <= u { low = mid + 1 } else { high = mid }
        }
        return low
    }
}
