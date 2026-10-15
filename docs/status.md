# Status

What is live, one table; then the hand-off. Rules are in `CLAUDE.md`; the registration in `docs/registration-slice-1.md`;
parity with py4DSTEM in `docs/parity.md`; what mac4DSTEM changes to import the kit in `docs/consumers-0.0.1.md`.

## Live (2026-10-15 early morning, end of the first session)

| Area | State | Evidence |
|---|---|---|
| Name | CuboKit passed the clash check on 2026-10-15: no Apple framework, no Swift Package Index package, no 4D-STEM tool of that name (the nearest are GoCubeKit, a Go `cube-kit`, and a Python `cubo` for satellite data cubes) | the session's searches |
| Registration of slice 1 | Written before any code (`b620ba5`); pins before any Swift (`6833b4e`); two dated amendments (§9), the second after Gate B | `docs/registration-slice-1.md` |
| `CuboCore` | `CubeShape`, `CubeSource` (async reads), `MemoryCube`, memory-mapped `MappedCube` (ten element types to Float32), `NPYFile` | `CuboCoreTests` |
| `CuboVirtual` | `DetectorGeometry` and `DetectorMask` (py4DSTEM's `make_detector` in Double), `VirtualImaging.image`, `VirtualDiffraction.mean/max/meanAndMax`; `.cpu` (Accelerate, Double, compensated mean) and `.metal` (compensated Float32 kernels, Dot2) | `PinsTests`, `TruthTests`, `RefuterRegressionTests` |
| Pins against py4DSTEM `f050d207` | 7 synthetic cubes, 71 geometries: masks identical; CPU at distance 0.0 from py4DSTEM's float64 definition; integer cubes bit for bit against py4DSTEM's own calls | `tools/slice-1/pins.py`, `Fixtures/slice-1-pins.json` |
| Real data | The owner's Al–Mg–Si study cube (330 × 330 × 64 × 64), Au reference, simulated poly-Au, the particle cube, and py4DSTEM's `small_datacube.dm4`: CPU mean and BF, ADF, DF images at distance 0.0 from float64; Metal within 2.4e-13; eager equals lazy; py4DSTEM's own float32 mean of the study cube is 1.9e-5 from its float64 definition | `RealDataTests`, log `test-1b15f21` (session scratchpad), exports in `~/Library/Caches/CuboKit/real-data` |
| Slice 2 registration | Before code (`9e12232`), pins before Swift (`0603136`), two amendments (§9: first run; Gate B) | `docs/registration-slice-2.md` |
| `CuboVirtual` slice 2 | `CenterOfMass` (Neumaier CPU; Metal with renormalised double-float sums), `ProbeSize` (py4DSTEM's `get_probe_size`, NumPy's sums) | `Slice2PinsTests`, `Slice2TruthTests`, `Slice2RefuterRegressionTests` |
| `CuboPhase` | `CenterOfMassReference` (plane, constant, none, given), `RotationSolver` (curl / divergence, py4DSTEM's selection), `DPC.reconstruct` (iterative, Double, Bluestein FFT) | `CuboPhasePinsTests`, `CuboPhaseTruthTests` |
| Slice 2 on real data | The five cubes: CPU CoM 0.0 px from float64, Metal ≤ 1.9e-10 px; rotation and transpose equal to py4DSTEM (Al–Mg–Si 83° T, Au 2° T, sim Au 86°, particle 80° T, small cube −18°); DPC phase ≤ 3.3e-12 of max from the float64 chain (py4DSTEM's own float32: 5e-7 to 5.2e-5); probe size bit for bit | `CuboPhaseRealDataTests`, log `test-s2-gateB` |
| Slice 2 Gate B | One refuter on `84815c9`: Metal CoM fixed (F1; the Metal claim needs \|S\| ≥ 1e-3·Σ\|wP\|), radians as NumPy (F3), py4DSTEM's float32 tie flips recorded (F2), non-finite input refused (F4) | `registration-slice-2.md` §9, `a67805c` |
| Slice 2 break-once proof | 30 mutants on `a67805c`: 29 red, the registered survivor green; slice 1's 40 re-run there as registered | `docs/slice-2-mutants-2026-10-15.md` |
| Slice 3 | Registration `22c3bdd`, pins `d272160`, code `5c82508`, Gate B fixes `63dbfef`: `OriginMeasurement` (get_origin in Double, scipy's filter order), `OriginFit` (four models, robust, lstsq's projection when rank-deficient), centred, calibrated and shifted virtual images, shifted mean and max (rounded, Fourier) | `Slice3Tests`, `Slice3RealDataTests`, `docs/registration-slice-3.md` |
| Slice 3 on real data | The five cubes: origins 0.0 px from float64; py4DSTEM's float32 filter moves 2 of 63 550 Au argmaxes (recorded); integer cubes bit for bit against py4DSTEM's centred, calibrated and rounded images | log `test-s3-final` |
| Slice 3 Gate B and mutants | Refuter on `5c82508`: symmetric ties (F1), traps (F2), rank-deficient fits (F3), NaN (F4) fixed; the Fourier claim restated normwise. 19 mutants on `63dbfef`: 17 red, 2 registered survivors | `docs/slice-3-mutants-2026-10-15.md` |
| Placeholder modules | `CuboStrain`, `CuboPtycho`: a scope statement each | their placeholder tests |
| Tests | Swift Testing (owner, 2026-10-15): 74 tests, 0 failures, release, real data included | `tools/test.sh` |
| Gate | green and stamped at the closing commit | `.build/last-green`, `tools/last-green.sh` |
| Consumer check | `tools/consumer-check`: 38 checks through the public API (slices 1 to 3), Debug, 0 warnings | `tools/consumer-check/run.sh` |
| Break-once proof | `tools/slice-1/mutants.py`, 40 mutants on the fixed commit `1b15f21`: 38 red, the 2 registered survivors green, none unexpected | `docs/slice-1-mutants-2026-10-15.md` |
| Gate B refuter | One refuter on `85318cd`: F1 (Metal tolerance on zero-mean data) and F2 (`.npy` overflow trap) fixed in `1b15f21`; F3, F4 made consistent; three behaviours recorded | `registration-slice-1.md` §9 |

Validation: nothing here is validated on a microscope. Agreement with py4DSTEM and recovery of planted cubes are what is
claimed (`registration-slice-1.md` §8).

## Hand-off

The next session starts from the suite board's `lanes/cubokit` prompt. Before it, the owner pushes `main` and may tag
0.0.2 (commands on the board); mac4DSTEM's own session switches at a tag of its choosing (`docs/consumers-0.0.1.md`,
`docs/consumers-0.0.2.md`).

Slice 4, registered before code: the median pattern and masked virtual diffraction (`get_virtual_diffraction` with a
real-space mask, median), `get_origin`'s real-space mask and `fast_center`, `get_origin_friedel`, the probe kernels
(`get_vacuum_probe`, the sigmoid kernel) that disk detection needs; or strain (`CuboStrain`) if the owner prefers. A Metal
path for the slice-3 operations and DPC's filters are later lanes.

## Unverified

- iPadOS: not built (the maths and `mmap` allow it).
- Metal: Float32 subnormals flush to zero (recorded); the Metal CoM claim covers |S| ≥ 1e-3 · Σ|wP| only.
- Tiles larger than Metal's maximum buffer length, and Float32 overflow in Metal sums of huge finite values: not tested.
- `.npy` files with an unaligned data offset are refused (only hand-made headers have one).
