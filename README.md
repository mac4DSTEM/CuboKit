# CuboKit

The macEM suite's 4D-STEM library for macOS: a Swift package of numbers, never UI, that mac4DSTEM imports by tagged version
and that other people can build their own apps on. It is the Mac-native counterpart of py4DSTEM's `DataCube`: the same
numbers on the same inputs, checked against py4DSTEM at a pinned commit, with Accelerate on the CPU path and Metal on the
accelerated path behind one API. Headless, Swift 6 strict concurrency, one SwiftPM product per module.

| Module | Scope | State |
|---|---|---|
| `CuboCore` | the cube: `CubeShape` (scan rows and columns, detector rows and columns), `CubeSource`, the eager `MemoryCube`, the lazy memory-mapped `MappedCube` (ten element types, converted to Float32 on read), NumPy's `.npy` container | slice 1, `docs/registration-slice-1.md` |
| `CuboVirtual` | py4DSTEM's virtual detectors (point, circle, annulus, rectangle, any mask), virtual bright- and dark-field images, the mean and maximum diffraction pattern, on `.cpu` or `.metal` | slice 1 |
| `CuboVirtual` (slice 2) | the centre of mass per position (`.cpu` and `.metal`), py4DSTEM's probe size | slice 2, `docs/registration-slice-2.md` |
| `CuboPhase` | the centre-of-mass reference (plane fit, constant, given origins), the R–Q rotation by curl or divergence, py4DSTEM's iterative DPC | slice 2 |
| `CuboVirtual` (slice 3) | per-position origins (`get_origin`), centred, calibrated and centre-shifted virtual images and patterns; `CuboCore`'s `OriginFit` (`fit_origin`) | slice 3, `docs/registration-slice-3.md` |
| `CuboStrain` | strain maps from Bragg vectors | placeholder, a scope statement |
| `CuboPtycho` | ptychography and tilt-corrected bright field | placeholder, a scope statement |

Slice 2 (centre of mass, probe size, rotation, DPC) is held to py4DSTEM the same way: the CPU centre of mass at 0.0 px
from the float64 definition on the owner's cubes, the probe size bit for bit, the rotation and transpose equal to
py4DSTEM's choice, the DPC phase within 3.3e-12 of max of the float64 chain.

## The first ten minutes

```swift
import CuboCore
import CuboVirtual

// In Python: np.save("cube.npy", datacube.data)   (py4DSTEM's (R_Nx, R_Ny, Q_Nx, Q_Ny), C order)
let cube = try MappedCube(npy: URL(fileURLWithPath: "cube.npy"))            // lazy: nothing is read yet
let mean = try await VirtualDiffraction.mean(of: cube)                       // get_dp_mean, in Double
let bf = try await VirtualImaging.image(of: cube,
    geometry: .circle(centerRow: 31, centerColumn: 32, radius: 6.4))         // get_virtual_image('circle', ((31, 32), 6.4))
let adf = try await VirtualImaging.image(of: cube,
    geometry: .annulus(centerRow: 31, centerColumn: 32, inner: 18, outer: 30), path: .metal)
```

**Rows and columns, never x and y.** py4DSTEM's first axis of a pair (`Rx`, `Qx`) is the kit's row; its second (`Ry`, `Qy`)
the column. A py4DSTEM geometry passes through unchanged: `((qx0, qy0), r)` is `.circle(centerRow: qx0, centerColumn: qy0,
radius: r)`.

## Numbers and their tolerances

The kit is held to the float64 evaluation of py4DSTEM's definitions, not to NumPy's float32 accumulator (whose own results
are recorded beside it). Measured on the synthetic pins and on five real cubes (`docs/status.md`):

| Quantity | CPU path (`.cpu`, the default) | Metal path (`.metal`) |
|---|---|---|
| Detector masks | identical to py4DSTEM's `make_detector` | the same mask |
| Virtual image | integer cubes bit for bit against py4DSTEM; float cubes within 1e-10 of the float64 definition (measured: identical) | within 1e-5 of the CPU path (compensated kernels; measured: at most 2.4e-13 on the owner's cubes, 1.5e-8 on the Gate B refuter's noise cubes) |
| Mean pattern | integer cubes bit for bit; float cubes within 1e-10 (measured: identical) | within 1e-5 (measured: at most 7.6e-14) |
| Maximum pattern | bit for bit | bit for bit |
| Eager against lazy, any tiling | bit for bit | the same bits for the image; the mean within 1e-5 |

There is no automatic choice of path: a number never depends on the hardware unless the caller asks for `.metal`, and every
result records the path that made it.

## Licence

GPL-3.0 (`LICENSE`, `NOTICE`). **Which parts come from where:**

- The cube model and the reducers are extracted from **mac4DSTEM** (GPL-3.0; `github.com/mac4DSTEM/mac4DSTEM`, commit
  `7448401b`): `FourDDataSource`, `FourDArray`, `ResidentCube`, `VirtualDetector` and the `VirtualMask` and `DPStatistics`
  Metal kernels, rewritten as a public API without the app's types.
- The detector definitions (`DetectorMask`) are a port of **py4DSTEM**'s `make_detector`, and the virtual image, mean and
  maximum pattern follow its `get_virtual_image`, `get_dp_mean` and `get_dp_max` (GPL-3.0; `github.com/py4dstem/py4DSTEM`,
  commit `f050d207`, 0.14.19). py4DSTEM is also the oracle; please cite Savitzky et al., Microscopy and Microanalysis 27,
  712 (2021).
- The `.npy` reader and writer follow NumPy's documented format (BSD-3-Clause); no NumPy code is copied.

The whole package is GPL-3.0, so an app built on it must be GPL-compatible.

## Platform

macOS 27 or later (`Package.swift`), Apple silicon or Intel with Metal for the `.metal` path; the `.cpu` path needs only
Accelerate. Swift 6.2 tools, Swift 6 language mode. iPadOS is planned (the maths and `mmap` allow it) and not yet built.

## Build and test

```sh
swift build                         # builds alone: no setup, no dependency, the Metal kernels compile at run time
tools/test.sh                       # the gate: swift test -c release; stamps .build/last-green on a clean tree
tools/last-green.sh                 # the last green commit and whether HEAD equals it
tools/consumer-check/run.sh         # a separate package using the public API only
```

The oracle is not in the repository: `tools/slice-1/provision-py4dstem.sh` builds the pinned py4DSTEM venv in the gitignored
`References/`, and `tools/slice-1/pins.py` writes the pins (`synthetic`, committed) and the real-data exports (`real`,
outside the repository, in `$CUBOKIT_REAL_DATA`). The real-data tests skip by name when that directory is absent.

## Status

`docs/status.md`; the parity checklist against py4DSTEM's documented API is `docs/parity.md`. No tag yet; the first tag is
the owner's.
