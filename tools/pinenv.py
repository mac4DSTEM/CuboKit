"""Where CuboKit's pin generators find their inputs: one place for every script under tools/.

Each generator imports this module and runs from any working directory:

    python -I -B tools/<subject>/<script>.py [arguments]

with the Python environment of tools/requirements.txt (tools/embedding/ has its own, older one). The inputs that live
outside this repository are environment variables, each with a default for the suite layout (this clone beside
References/ and the app and kit repos):

  PY4DSTEM_SRC        a git checkout of py4DSTEM at PY4DSTEM_PIN; the generators compare the installed package with it
                      (default ../References/CuboKit/py4DSTEM-f050d207)
  PY4DSTEM_TEST_DATA  py4DSTEM's test data folder (small_datacube.dm4, Au_sim.h5)
                      (default ../References/CuboKit/py4dstem-test-data)
  MAC4DSTEM_REPO      a clone of mac4DSTEM, for the generators that transcribe the original app at a commit (default ../mac4DSTEM)
  IMAGOKIT_REPO       a clone of ImagoKit (tools/bragg only; default ../ImagoKit)
  CUBOKIT_REAL_DATA   the real-data exports, never in the repository (default ~/Library/Caches/CuboKit/real-data)

From a clone of this repository alone: create the environment from tools/requirements.txt, then
    git clone https://github.com/py4dstem/py4DSTEM "$PY4DSTEM_SRC" && git -C "$PY4DSTEM_SRC" checkout f050d207
CONTRIBUTING.md has the whole recipe. A worktree outside the suite folder finds the suite through its main checkout.
"""
import os
import subprocess
from pathlib import Path

PY4DSTEM_PIN = "f050d207851f8accd3aa550365baaf1f38230428"

TOOLS = Path(__file__).resolve().parent
KIT = TOOLS.parent


def _suite() -> Path:
    """The folder that holds this clone and References/: the clone's parent, or the main checkout's for a worktree."""
    if (KIT.parent / "References").is_dir():
        return KIT.parent
    try:
        common = subprocess.check_output(["git", "-C", str(KIT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        return Path(common).parent.parent
    except (OSError, subprocess.CalledProcessError):
        return KIT.parent


SUITE = _suite()


def _path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable, "")
    return Path(value).expanduser() if value else default


PY4DSTEM_SRC = _path("PY4DSTEM_SRC", SUITE / "References/CuboKit/py4DSTEM-f050d207")
PY4DSTEM_TEST_DATA = _path("PY4DSTEM_TEST_DATA", SUITE / "References/CuboKit/py4dstem-test-data")
MAC4DSTEM_REPO = _path("MAC4DSTEM_REPO", SUITE / "mac4DSTEM")
IMAGOKIT_REPO = _path("IMAGOKIT_REPO", SUITE / "ImagoKit")
REAL = _path("CUBOKIT_REAL_DATA", Path.home() / "Library/Caches/CuboKit/real-data")


def fixtures(target: str) -> Path:
    """`Tests/<target>/Fixtures` of this clone."""
    return KIT / "Tests" / target / "Fixtures"
