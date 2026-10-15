# What a consumer changes to import CuboKit 0.2.0

Written 2026-10-15 for mac4DSTEM by the CuboKit session. 0.2.0 is the generality pass announced in
`docs/consumers-0.1.0.md`: the kit's fixed policies become parameters and the package stands on its own for people outside
the suite. It is a minor bump because new parameters appear in public signatures; **every 0.1.0 call compiles unchanged
and returns the same numbers** (every new parameter defaults to the old value, and the full gate pins every fixture bit
for bit at the default).

## What changes

| 0.1.0 | 0.2.0 |
|---|---|
| `OriginFit.fit(_:rows:columns:model:robust:robustSteps:robustThreshold:)`, a column left out below 1e-12 of the largest norm | `…, rankTolerance: Double = 1e-12` |
| `CenterOfMassReference.normalize(_:reference:reciprocalSampling:)`, the plane fit refused below 1e-12 | `…, degeneracyTolerance: Double = 1e-12` |
| `GaussianFilter(sigma:)`, a radius above 10⁶ px refused | `GaussianFilter(sigma:maximumRadius: Int = 1_000_000)` |
| `FriedelOrigin.measure(of:mask:upsampleFactor:tiling:)` and `.origin(of:mask:upsampleFactor:)`, upsampling above 1024 refused | `…, maximumUpsampleFactor: Int = 1024` |
| `VirtualDiffraction.pattern(of:statistic:scan:shift:tiling:)`, the median gathered in 256 MiB blocks | `…, medianWorkingBytes: Int = 256 << 20` (changes the passes, never the result) |
| `platforms: [.macOS("27.0")]` | **`[.macOS(.v15), .iOS(.v18)]`** (owner's decision, 13:58 CEST): an app at macOS 27 builds unchanged; CI's macOS 26 runner can now build and test the kit |
| `CenterOfMass.measure(of:...path: .cpu)` single-threaded (0.77 s on the 330 × 330 × 64 × 64 cube) | positions in parallel, each with its own compensated sums: bit identical (kit ask 3) |
| `RotationSolver.solve` single-threaded, not cancellable (0.767 s on a 330 × 330 field) | the angle grid in parallel, bit identical per angle (0.031 s); `RotationSolver.curve(_:anglesDegrees:criterion:transpose:isCancelled:)`, the curve alone; `isCancelled:` on `solve` too, throwing `CancellationError` (kit ask 2) |
| Comments, README, example, CI and DocC naming the suite and its apps | none of them do; provenance for the licence is in `NOTICE` |
| — | a DocC catalogue per module (one article each); `tools/readme-consumer`: the README's own code, compiled and run |

## Behaviour to know

1. **Nothing to change for mac4DSTEM.** Re-pin `exact: "0.2.0"` and build; the pinned consumer tests must stay green. A
   call that passes `tiling:` to `FriedelOrigin.measure` still compiles (the new parameter sits before it with a default).
2. **The registered claims are unchanged** (`docs/registration-slice-1.md` to `-slice-4.md`). A non-default parameter
   is outside them: a larger upsampling factor, for example, is not refuted on this machine beyond 1024 (slice 4, §9).
3. **Build the kit optimised.** A Debug app compiles its dependencies at `-Onone` too, and the kit then runs tens of
   times slower (measured by the app on the Al–Mg–Si cube: the centre of mass 48 to 58 s, the rotation 186 to 217 s,
   against well under a second at `-O`). Either build the app in release for anything interactive, or set the kit's
   optimisation for Debug in the app's build settings (Xcode: the package dependency's `SWIFT_OPTIMIZATION_LEVEL = -O`;
   SwiftPM: `swift build -c release`). The manifest cannot carry unsafe flags for its consumers (kit ask 4).
4. **Not in 0.2.0: a zero-copy source for a resident cube** (kit ask 1). `CubeSource.readScanTile` returns `[Float]`, so a
   cube already resident in an `MTLBuffer` is copied tile by tile (an interactive virtual image went from 0.006–0.02 s to
   0.08 s in the app). A source that lends an `UnsafeBufferPointer` or a buffer slice changes the protocol's contract and
   comes first in 0.2.1, registered; until then the app keeps its own resident path for interactive drags.
5. **The Metal centre-of-mass bound** (|S| ≥ 1e-3 · Σ|wP|, slice 2) is a claim about where the Metal path agrees with the
   CPU path, not a policy the code applies; it stays in the registration and `docs/parity.md`, not a parameter.
