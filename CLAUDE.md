# CuboKit

The 4D-STEM library: mac4DSTEM's processing core, checked against py4DSTEM. Pure Swift, no UI, standalone; the suite's rules are in `../CLAUDE.md`. Start with `README.md`,
`docs/parity.md` and the task.

## Reference
py4DSTEM `f050d207` (0.14.19) in `../References/python/py4dstem` (macEM-suite `tools/setup-python-refs.sh py4dstem`; NumPy 2.5.4, h5py 3.16.0, Python 3.12.14). Real cubes come from `$MACEM_DATA`; their `.npy` exports live in `$CUBOKIT_REAL_DATA` (default `~/Library/Caches/CuboKit/real-data`), never in the repo. Fixtures are generated from it (`tools/*/pins.py`) and committed under `Tests/*/Fixtures/`; a difference inside
the reference's own tolerance is agreement, outside it a bug. Readers, I/O, indices and metadata exact.

## Build and test
```sh
swift build
swift test -c release          # the gate; Debug is many times slower on the numeric loops
tools/consumer-check/run.sh    # a separate package that compiles against the public API
```
Never `swift build --build-tests -c release` (fails on this toolchain). Real-data tests skip when the data is absent.

## Modules
`CuboCore` (cube, sources, mapped cube, `.npy`, FFT plans, hot pixels), `CuboVirtual` (detectors, virtual images, mean/max/median, centre of mass, origins, probes; CPU and Metal), `CuboPhase` (DPC, iDPC), `CuboStrain`, `CuboBragg`, `CuboParallax`, `CuboPtycho`, `CuboEmbedding`.

## Rules of the kit
- Public API only; nothing names an app or an app type; error messages never say "mac4DSTEM" or "macTEM".
- Rows and columns, not x and y. Every accelerated path keeps the CPU float64 path as its reference (D16).
- A new or changed number gets a test against the reference at its upstream tolerance, and a row in `docs/parity.md`.
- No package dependencies. Metal kernels in `Sources/CuboVirtual/Metal/`; the caller asks for `.metal`, the result records it.

## Closeout
`swift test -c release` green; `CHANGELOG.md` entry with a Migration paragraph where a consumer must change something;
version bumped in the entry; one commit on `main` whose last paragraph is the hand-off. Paul tags with
`../macEM-suite/tools/ship.sh` (tag = CHANGELOG top version).
