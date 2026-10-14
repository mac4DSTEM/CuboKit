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
| Placeholder modules | `CuboStrain`, `CuboPtycho`: a scope statement each | their placeholder tests |
| Tests | Swift Testing (owner, 2026-10-15): 37 tests, 0 failures, release | `tools/test.sh` |
| Gate | green and stamped at the closing commit | `.build/last-green`, `tools/last-green.sh` |
| Consumer check | `tools/consumer-check`: 27 checks through the public API, Debug, 0 warnings | `tools/consumer-check/run.sh` |
| Break-once proof | `tools/slice-1/mutants.py`, 40 mutants on the fixed commit `1b15f21`: 38 red, the 2 registered survivors green, none unexpected | `docs/slice-1-mutants-2026-10-15.md` |
| Gate B refuter | One refuter on `85318cd`: F1 (Metal tolerance on zero-mean data) and F2 (`.npy` overflow trap) fixed in `1b15f21`; F3, F4 made consistent; three behaviours recorded | `registration-slice-1.md` §9 |

Validation: nothing here is validated on a microscope. Agreement with py4DSTEM and recovery of planted cubes are what is
claimed (`registration-slice-1.md` §8).

## Hand-off

The next session starts from the suite board's `lanes/cubokit` prompt (slice 2: centre of mass and DPC). Before it:

1. The owner creates the GitHub remote, pushes and tags 0.0.1 (the commands are on the board's waiting list).
2. mac4DSTEM's own session switches to the import at a tag of its choosing (`docs/consumers-0.0.1.md`).

Slice 2, registered before code: centre of mass per position (py4DSTEM's `get_com`-style measurement and its DPC
preprocessing), the probe size (`get_probe_size`), the curl-minimising R–Q rotation and the Fourier-integrated DPC
potential, against py4DSTEM at the same pin, on the owner's cubes and py4DSTEM's test datacube; mac4DSTEM's
`Core/Analysis/DPC.swift`, `VirtualDetector.tiledCenterOfMass`, the `CenterOfMass.metal` kernel and its `idpc-test` harness
are the seeds. The CPU reference and a compensated Metal path, as slice 1.

## Unverified

- iPadOS: not built (the maths and `mmap` allow it).
- Tiles larger than Metal's maximum buffer length, and Float32 overflow in Metal sums of huge finite values: not tested.
- `.npy` files with an unaligned data offset are refused (only hand-made headers have one).
