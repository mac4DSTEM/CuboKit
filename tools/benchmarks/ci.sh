#!/bin/bash
# The audit workflow's nightly CPU benchmark (.github/workflows/audit.yml, the suite's template): makes the inputs with
# `bench.py make` (NumPy in a throwaway venv), runs the kit side (BenchmarkHarness, CUBOKIT_BENCH_DIR) and writes the CPU
# timings to $1 as a JSON list of {name, dtype, size, ms: [median]}. Metal rows are left out (the runner's GPU is
# virtual); the py4DSTEM side and the table stay local (bench.py). This run only compares the kit with its own previous night.
set -euo pipefail
out="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
dir="$(mktemp -d)"
python3 -m venv "$dir/venv"
"$dir/venv/bin/pip" install --quiet numpy
"$dir/venv/bin/python" -I -B tools/benchmarks/bench.py make "$dir"
CUBOKIT_BENCH_DIR="$dir" swift test -c release --filter BenchmarkHarness
python3 -I - "$dir/swift.json" "$out" <<'PY'
import json, sys
rows = []
for key, t in sorted(json.load(open(sys.argv[1]))["results"].items()):
    name, size, path = key.split("|")
    if path != "cpu":
        continue
    rows.append({"name": name, "dtype": "float32", "size": size, "ms": [t["median"] * 1000]})
if not rows:
    sys.exit("no cpu rows in swift.json")
json.dump(rows, open(sys.argv[2], "w"), indent=1)
print(f"{len(rows)} cases")
PY
