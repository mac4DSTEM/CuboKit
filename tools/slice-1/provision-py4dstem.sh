#!/bin/zsh
# CuboKit References: the py4DSTEM oracle, pinned to mac4DSTEM's lock (tools/lib/fetch-py4dstem.sh at 7448401b):
# commit f050d207851f8accd3aa550365baaf1f38230428 (dev, 2026-03-26, version 0.14.19), installed into a venv whose
# numerical stack is mac4DSTEM's tools/lib/py4dstem-ci-constraints.txt (the stack its parity numbers were established on).
# Python: the owner's conda env py4dstem's 3.12 interpreter (py4DSTEM's upstream CI floor), not the 3.14 base.
set -euo pipefail
mkdir -p "$(dirname "$0")/../../References"
REF="$(cd "$(dirname "$0")/../../References" && pwd)"
LOCK="f050d207851f8accd3aa550365baaf1f38230428"
SRC="$REF/py4DSTEM-f050d207"
VENV="$REF/py4dstem-venv"
PY="$HOME/miniconda3/envs/py4dstem/bin/python3.12"
[[ -x "$PY" ]] || PY="$HOME/miniconda3/envs/py4dstem/bin/python"
if [[ ! -d "$SRC/.git" ]]; then
  git clone -q --filter=blob:none https://github.com/py4dstem/py4DSTEM.git "$SRC"
fi
git -C "$SRC" checkout -q "$LOCK"
grep -q '0.14.19' "$SRC/py4DSTEM/version.py" || { echo "version.py is not 0.14.19 at the lock" >&2; exit 1; }
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip
# mac4DSTEM's constraints, read from its pushed HEAD (never its working tree).
git -C "$REF/../../mac4DSTEM" --no-optional-locks show origin/main:tools/lib/py4dstem-ci-constraints.txt > "$REF/py4dstem-ci-constraints.txt"
"$VENV/bin/python" -m pip install -c "$REF/py4dstem-ci-constraints.txt" "$SRC"
"$VENV/bin/python" -m pip freeze > "$REF/py4dstem-venv-freeze.txt"
"$VENV/bin/python" -I -B -c 'import py4DSTEM, numpy, h5py; print("py4DSTEM", py4DSTEM.__version__, py4DSTEM.__file__); print("numpy", numpy.__version__, "h5py", h5py.__version__)'
echo "provision-py4dstem: ready"
