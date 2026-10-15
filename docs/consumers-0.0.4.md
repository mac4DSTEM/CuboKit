# What a consumer changes to import CuboKit 0.0.4

Written 2026-10-15 for mac4DSTEM by the CuboKit session, read at mac4DSTEM's pushed `7f192da9`. mac4DSTEM's own session
makes the switch. 0.0.4 is 0.0.3 plus slice 4 (`docs/registration-slice-4.md`); nothing 0.0.1 to 0.0.3 returns changes,
and every 0.0.3 call compiles unchanged.

## What slice 4 offers, and what it replaces

| mac4DSTEM today | CuboKit 0.0.4 |
|---|---|
| `tiledDiffraction` over a real-space ROI (mean, max) | `VirtualDiffraction.pattern(of:statistic:scan:shift:)`: py4DSTEM's `get_virtual_diffraction` with `.mean`, `.max` or `.median` over `.all`, `.mask([Bool])` or `.weights([Double])`, optionally centre-shifted; `VirtualDiffraction.median(of:)` |
| — | `OriginMeasurement.measure(of:radius:radiusScale:scanMask:fastCenter:)` and `OriginMap.replacingNaNWithMean()`: `get_origin`'s `mask`, `fast_center`, `remove_NaN` |
| `Core/Analysis/FriedelOrigin.swift` (Float32, NaN filled by a neighbour median, a zero denominator read as 0) | `FriedelOrigin.measure(of:mask:upsampleFactor:)` and `FriedelOrigin.origin(of:mask:upsampleFactor:)`: py4DSTEM's `get_origin_friedel` in Double with IEEE arithmetic, and its DFT upsampling |
| `Core/Analysis/BeamstopMask.swift` (Float32; the threshold index clipped to N − 1) | `BeamstopMask.make(from:threshold:distanceEdge:includeEdges:sigma:scaleRadial:)`: `get_beamstop_mask` on the pattern you pass (mean, or max for `use_max_dp`); `true` under the beamstop |
| `ProbeKernel.measured`, `.synthetic`, `.flat` (Float32; wrapped-coordinate kernels; a cosine shoulder instead of py4DSTEM's vacuum mask) | `VacuumProbe.measure(of:scanMask:align:zeroVacuum:detectorMask:)` (`get_vacuum_probe`), `VacuumProbe.zeroVacuum`, `.dampVacuum`; `ProbeKernel.flat`, `.gaussian`, `.sigmoid(profile: .sineSquared / .logistic)`; `SyntheticProbe.make` — each a corner-centred `DetectorImage` in Double |

## Behaviour to know (tier 1)

1. **Names shared with the app.** CuboKit now has public `FriedelOrigin`, `BeamstopMask` and `ProbeKernel`, as
   mac4DSTEM's `package` types are named. Inside mac4DSTEM's own module its types shadow the kit's; qualify the kit's as
   `CuboVirtual.FriedelOrigin` where both are in scope, or delete the app's copy when switching.
2. **A new error case.** `CuboVirtualError.invalidInput(String)` is new: an exhaustive `switch` over `CuboVirtualError`
   needs a `default` (or the new case).
3. **The Friedel origin is py4DSTEM's, not the app's port.** A NaN pixel makes that pattern's origin NaN (py4DSTEM's FFT
   is NaN-poisoned the same way; the app filled NaN by a neighbour median); a zero parabola denominator gives a
   non-finite origin, not 0. Filter non-finite pixels before the call if the app wants its old behaviour.
4. **On integer cubes py4DSTEM's own masked Friedel origins are wrong** (it squares uint16 patterns in uint16, which
   wraps; up to 30 px on the simulated Au cube). The kit follows the float64 definition; a comparison against py4DSTEM
   must cast the cube to float first.
5. **The beamstop mask refuses** a threshold whose index reaches N (py4DSTEM raises) and a cleaned mask with no pixel
   outside the pattern (scipy measures to a pixel outside the array there); the app clipped the index.
6. **The vacuum probe follows py4DSTEM exactly, quirks included**: the first selected pattern only anchors the first
   alignment (its weight in the mean is 0), and the default damping scales the disk itself by sin(1) ≈ 0.841. Kernels
   built from it are normalised, so the scale does not matter for detection.
7. **Kernels are corner-centred on the native grid**, as the app's; the flat kernel is shifted by Fourier (py4DSTEM's
   default for flat), the Gaussian and sigmoid kernels bilinearly (py4DSTEM's default for those). The origin defaults to
   `ProbeSize.estimate` of the probe.
8. **Everything new runs on the CPU** (Double). A Metal path is a later lane.
