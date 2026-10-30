# Contributing to CuboKit

Thanks for your interest. Bug reports, questions and pull requests are welcome.

## Reporting issues

Open an issue and say what you ran, what you expected and what you got. For a wrong number, a small cube (or the code
that makes one), the call you made and, if you have it, py4DSTEM's answer help most. Please say which Mac, macOS and
Swift version you used and whether you used `.cpu` or `.metal`.

## Building and testing

CuboKit needs macOS 15 or later and a Swift 6.2 toolchain.

```sh
swift build
swift test -c release
```

The tests use Swift Testing (`import Testing`, `@Test`, `#expect`). They need nothing outside the package: the small
reference fixtures are committed. Tests that need large real cubes skip themselves when the data is absent.

## Numbers are checked against py4DSTEM

Results are compared with fixtures produced by py4DSTEM at a pinned commit (`f050d207`, 0.14.19), or with a closed form
or a planted cube. A test's tolerance is the one the Python package's own tests use for that function; a result outside it is a bug. A change that moves a fixture says why in the pull request and in the CHANGELOG. Regenerate the fixture in its own commit.

## Regenerating the fixtures

Every fixture under `Tests/*/Fixtures/` names its generator in `Tests/Fixtures/README.md`; the generators are the
scripts under `tools/<subject>/`. From a clone of this repository:

```sh
python3.12 -m venv .venv && .venv/bin/python -m pip install -r tools/requirements.txt
git clone https://github.com/py4dstem/py4DSTEM ../py4DSTEM-f050d207 && git -C ../py4DSTEM-f050d207 checkout f050d207
export PY4DSTEM_SRC=$PWD/../py4DSTEM-f050d207
.venv/bin/python -I -B tools/cubeview/pins.py        # any generator, from any working directory
```

`tools/pinenv.py` is where every generator finds what lives outside this repository, each an environment variable:

| Variable | What | Needed by |
|---|---|---|
| `PY4DSTEM_SRC` | a git checkout of py4DSTEM at `f050d207`; the generators check that the installed package equals it | `bragg`, `cubeview`, `hot-pixels`, `origin`, `parallax`, `ptycho`, `rereference`, `strain` |
| `PY4DSTEM_TEST_DATA` | py4DSTEM's test data folder (`small_datacube.dm4`, `Au_sim.h5`) | `parallax/pins.py`, `strain`, `virtual real` |
| `MAC4DSTEM_REPO` | a clone of mac4DSTEM, for the pins transcribed from the original app at a named commit | `bragg`, `embedding`, `idpc`, `origin`, `parallax`, `strain` |
| `IMAGOKIT_REPO` | a clone of ImagoKit | `bragg` |
| `CUBOKIT_REAL_DATA` | the real-data exports, never in the repository (default `~/Library/Caches/CuboKit/real-data`) | the `real` modes of `virtual`, `com-dpc`, `origin-fit`, `probes` |

The other generators need only the Python environment. `tools/embedding/pins.py` alone needs the older environment of
`tools/embedding/requirements.txt` (scikit-learn 1.4.2, whose PCA output the committed embedding fixtures carry).
Fixtures stamped with the earlier stack (Python 3.12.4, NumPy 2.5.0) regenerate with the same numbers under
`tools/requirements.txt`; only the version stamps change.

## Pull requests

- Keep a change focused, with a test that fails before it and passes after.
- Public API only: a type a user needs must be `public`, and nothing should name an application.
- Say in the pull request what you ran. Please do not edit generated fixtures by hand.
- By contributing you agree that your contribution is licensed under GPL-3.0-only, like the rest of the package.
