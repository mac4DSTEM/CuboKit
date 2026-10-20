# What a consumer changes to import CuboKit 0.0.2

Written 2026-10-15 for mac4DSTEM, by the CuboKit session, read at mac4DSTEM's pushed `7f192da9`. **mac4DSTEM's own session
makes the switch** (a tier-1 lane at a tag of its choosing). 0.0.2 is slice 1 (`docs/consumers-0.0.1.md`, which still
holds) plus slice 2: centre of mass, probe size, the R–Q rotation and py4DSTEM's iterative DPC. Bumping from 0.0.1 to
0.0.2 changes nothing slice 1 returns.

## 1. The dependency

`.package(url: "https://github.com/mac4DSTEM/CuboKit", exact: "0.0.2")`, and the products `CuboCore`, `CuboVirtual`,
`CuboPhase` (new). `CuboPhase` depends on `CuboCore` only; it takes centre-of-mass maps, so it can also be fed by a
measurement the app makes itself.

## 2. The call sites (mac4DSTEM `7f192da9`)

| mac4DSTEM today | With CuboKit 0.0.2 |
|---|---|
| `VirtualDetector.tiledCenterOfMass(data:descriptor:center:origins:)` (`App/AppState+Calibration.swift:319`; the `centerOfMass` Metal kernel) | `CenterOfMass.measure(of:mask:path:tiling:)`; then `CenterOfMassReference.normalize(_:reference: .given(origins))` or `.none` for the shift |
| `OriginCalibration.probeSize(dp:qy:qx:)` (`Core/Analysis/OriginCalibration.swift:128`, used at 583, 821, 855, `ProbeRingHint.swift:43`, `AppState+DiskDetection.swift:147`) | `ProbeSize.estimate(_:)` on the kit's `DetectorImage` (mean or max) |
| `RotationCalibration.solve(com:width:height:)` (`AppState+Calibration.swift:269`) | `RotationSolver.solve(_:anglesDegrees:criterion:forceRotationDegrees:forceTranspose:)` on the normalized field |
| `DPC.applyRotation(com:rotationRad:transpose:)` (`AppState+DPC.swift:133`) | `RotationSolution.corrected` |
| `DPC.integrateIDPC` / `integratePhysicalIDPC` (direct Fourier integration) | unchanged in the app: CuboKit has py4DSTEM's *iterative* DPC (`DPC.reconstruct`), a different method; the app may offer both |

What stays in mac4DSTEM: per-position origin measurement (py4DSTEM `get_origin`; a calibration slice later), the iDPC
integration, the magnitude, angle and colour-wheel views, physical calibration of the samplings, and the probe-ring hint's
logic around the probe size.

## 3. Behaviour that changes (tier 1)

1. **Negative intensities count.** mac4DSTEM's `centerOfMass` kernel clamps each pixel at 0 before summing; py4DSTEM does
   not, and neither does the kit. The owner's dark-subtracted cubes have pixels down to about −500, so the centre of mass
   changes wherever such pixels lie; the size of that change on each cube is the switch lane's to measure.
2. **A zero sum is NaN.** mac4DSTEM substitutes the reference centre where the summed intensity is not positive; the kit
   returns NaN, as py4DSTEM does, and `CenterOfMassReference.normalize` turns NaN into 0 (`nan_to_num`), as py4DSTEM's
   preprocessing does.
3. **Double, compensated.** mac4DSTEM's kernel sums in Float32; the kit's CPU path agrees with py4DSTEM's float64 definition at
   0.0 px on the owner's cubes and its Metal kernel (renormalised double-float sums) bit for bit on integer cubes and within 1e-9 px where
   |S| ≥ 1e-3 · Σ|wP| (strongly dark-subtracted positions below that: use `.cpu`) (py4DSTEM's own float32 centre of mass differs
   from it by up to 3.6e-5 px, on py4DSTEM's small cube).
4. **The probe size is py4DSTEM's** on the same pattern (asserted exact today; D12: within the written tolerance), and refuses (`probeNotFound`) where py4DSTEM returns
   NaN. Check mac4DSTEM's port against it (D12: bit for bit is a quick check; on a miss, compare within tolerance) on the five cubes before switching; the CuboKit numbers are in its status.
5. **The rotation is py4DSTEM's selection rule** (ties keep the plain orientation and the first angle; the transposed
   orientation only when strictly better). The angles and transposes chosen on the owner's cubes: Al–Mg–Si 83° transposed,
   Au reference 2° transposed, simulated poly-Au 86°, particle 80° transposed, py4DSTEM's small cube −18°.

## 4. How mac4DSTEM checks the switch

`tools/idpc-test` (its iDPC harness) is unaffected. For the centre of mass and the rotation, compare before and after on
the five cubes: the differences are those of section 3 (the clamp, the substitution, Float32 against Double). CuboKit's
`CuboPhaseRealDataTests` hold the kit against py4DSTEM on the same cubes.
