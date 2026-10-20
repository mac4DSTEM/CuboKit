# CLAUDE.md — start here

CuboKit: the macEM suite's 4D-STEM library, mac4DSTEM's processing core made public and checked against py4DSTEM, imported
by tagged version. GPL-3.0 (README.md says which parts come from where). Modules: `CuboCore` (the cube: shape, sources, the
eager `MemoryCube`, the lazy memory-mapped `MappedCube`, the `.npy` container), `CuboVirtual` (detector geometries and masks,
the mean and maximum pattern, virtual images, the centre of mass, the probe size; CPU and Metal behind one API),
`CuboPhase` (the centre-of-mass reference, the R–Q rotation, py4DSTEM's iterative DPC; slice 2), `CuboStrain` and `CuboPtycho` (placeholders with
their planned scope). Rules only here; state in `docs/status.md`.

## Read first

1. The suite's `../CLAUDE.md` (board, tiers, how we work) and `../ROADMAP.md`, "Kits" (the direction, phase 5, the licence
   rule, the Apple-infrastructure requirements).
2. `docs/registration-slice-1.md`, `-slice-2.md` and `-slice-3.md`: what each slice makes, the definitions, the pins,
   the tolerances per quantity, the gates, the amendments after Gate B.
3. `docs/status.md`: what is live, the hand-off, what is unverified. `docs/parity.md`: the checklist against py4DSTEM.
4. `../SpectroKit/CLAUDE.md`: the model for a kit repo (this file follows it).

## Hard rules

- **mac4DSTEM is read-only from here**: `git -C ../mac4DSTEM --no-optional-locks show origin/main:<path>`, never its working
  tree. Never build, test or write in it or in another kit. Slice 1 was read at mac4DSTEM `7448401b` (its pushed head on
  2026-10-14 night). Switching mac4DSTEM to the import is its own session's lane (`docs/consumers-0.0.1.md`).
- **Registration before code** for anything tier 1 (every number a scientist reads): `docs/registration-<slice>.md`, with the
  definitions, the pins, the planted truth, the tolerances, the gates and the planned mutants, committed before the first
  Swift line; a change after code starts is a dated amendment with its reason.
- **The gate is `tools/test.sh`** (`swift test -c release "$@"`). Never run `swift build --build-tests -c release` (fails on
  this toolchain). The manifest carries no unsafe flags, so a consumer builds the kit optimised itself. It refuses to run
  while the suite's `drive` lock exists.
- **Green stamp**: a full, passing `tools/test.sh` run on a clean tracked tree, with the real data present, writes
  `.build/last-green` (commit, UTC time); `tools/last-green.sh` prints it and says whether HEAD equals it.
- **Hook**: `.claude/settings.json` runs `tools/hooks/no-sleep-poll.sh` before every Bash call; it refuses a sleep of 30 s or
  more and any `while`/`until` loop with a sleep in it. Wait on background-task notifications or Monitor. Proof:
  `tools/hooks/test-no-sleep-poll.sh`.
- **A reference for every number.** Pin: py4DSTEM `f050d207` (0.14.19), mac4DSTEM's lock, in the gitignored `References/`
  (`py4DSTEM-f050d207`, `py4dstem-venv` with mac4DSTEM's `py4dstem-ci-constraints.txt`: NumPy 2.5.0, h5py 3.11.0, Python
  3.12.4; `py4dstem-venv-freeze.txt`), built by `tools/slice-1/provision-py4dstem.sh`. Oracles run as
  `References/py4dstem-venv/bin/python -I -B`. An unpinned copy (the owner's conda env holds 0.14.17) is never a reference.
  Agreement with py4DSTEM is agreement, not truth: truth is a closed form, a hand count or a planted cube.
- **The kit is held to the float64 definition, not to NumPy's float32 accumulator.** py4DSTEM's own float32 results are
  recorded beside the float64 ones in every pin and never asserted against (registration section 2, 3.4).
- **Pins are generated before the Swift code** (`tools/slice-1/pins.py synthetic` writes `Tests/CuboVirtualTests/Fixtures/`);
  a difference is a failure of the port, never a tolerance to widen.
- **Numerical testing policy (D12, Paul 2026-10-19):** readers, I/O, indices and metadata are tested exact. Numerics are
  compared against the pinned Python reference within a tolerance written per function (assert_allclose style: about 1e-7
  relative for float64, 1e-5 for float32), never widened to hide a bug. During an app-to-kit switch, bit for bit is only a
  quick check that nothing moved by accident; if it fails, verify against the Python reference within tolerance and do
  not block. A moved pinned number gets its own commit with the reason. Elsewhere in this repo, "D12" means this rule.
- **Every accelerated path keeps the CPU reference** and a registered tolerance against it (Metal: relative 1e-5; the max is
  exact, D12). There is no automatic path choice: the caller asks for `.metal`, and the result records it.
- **Public API only, no app types.** Nothing an app needs is `package`; nothing here names `AppState`, `FourDArray`,
  `DatasetDescriptor`, `LoadView` or any other app type; error messages never say "mac4DSTEM". `tools/consumer-check` is a
  separate package that compiles only while every symbol it names is `public`.
- **The owner's data** lives in the suite's `data/` folder, read as `$MACEM_DATA` (default `../data`): never copied into this
  repository, not even into a gitignored folder. Its `.npy` exports for the real-data tests live outside the repository in
  `$CUBOKIT_REAL_DATA` (default `~/Library/Caches/CuboKit/real-data`), regenerable by `tools/slice-1/pins.py real`; a
  manifest line in `../data/README.md` per file used.
- **Rigour follows risk** (`../CLAUDE.md`): tier named in every commit; tier 1 is registration, every new test broken once,
  one Gate B refuter on the frozen commit, Gate D when a cause is unproven, a fixture.
- **Names** (the suite's standard): the library is domain + `Kit`, modules carry its prefix: `CuboCore`, `CuboVirtual`,
  `CuboStrain`, `CuboPtycho`. The kit says row and column, never x and y (registration 1.1).
- **Pushing, tagging and creating the remote are the owner's.** One session per repo; `main` only; commit by explicit paths at
  every green checkpoint. At each tag, `docs/consumers-x.y.z.md` says what each consumer changes.
- **Suite board**: this repo's row is `lanes/cubokit`; how a session opens and closes it is in `../CLAUDE.md`.
