#!/bin/zsh
# CuboKit test gate: `swift test -c release`.
#
# Release, as SpectroKit's and DiffractoKit's gates: the manifest carries no Debug `-O` (a versioned dependency cannot
# carry unsafe flags) and Debug at -Onone is many times slower on the reducers; a consumer builds the kit optimised
# itself. Never run `swift build --build-tests -c release` (fails on this toolchain); this script is the gate.
#
# Real data: RealDataTests read $CUBOKIT_REAL_DATA (default ~/Library/Caches/CuboKit/real-data, outside the repository,
# written by `tools/slice-1/pins.py real`) and skip by name when it is absent. A run that skipped them says so below
# and is not a full run: it writes no stamp.
#
# Lock: the suite's machine-wide test lock (../CLAUDE.md). A `drive` file means a session is driving an app on screen:
# refused. A kit's `swift test` takes no slot of its own (as SpectroKit's).
#
# Stamp: a full run (no arguments, real data present) that passes, on a tree whose Sources/, Tests/ (untracked files
# included: a new source file nobody committed is not green),
# Package.swift and Package.resolved are clean, writes .build/last-green (HEAD sha, UTC time). tools/last-green.sh prints
# it and whether HEAD still equals it.
set -euo pipefail
cd "${0:A:h}/.."
locks=/private/tmp/claude-501/-Users-paullobpreis-GitHub-mac4DSTEM-Organization-mac4DSTEM/locks
if [[ -e $locks/drive ]]; then
  print -u2 "tools/test.sh: a session is driving an app on screen ($locks/drive): no test run now"
  exit 75
fi
log=$(mktemp -t cubokit-test)
swift test -c release "$@" 2>&1 | tee $log
rc=${pipestatus[1]}
(( rc == 0 )) || exit $rc
skipped=$(grep -c "RealDataTests.*skipped" $log || true)
if (( $# > 0 )); then
  print -u2 "tools/test.sh: green, but arguments were given (not a full run): no stamp"
elif (( skipped > 0 )); then
  print -u2 "tools/test.sh: green, but $skipped real-data test(s) skipped (no real data here): no stamp"
elif [[ -n $(git status --porcelain --untracked-files=all -- Sources Tests Package.swift Package.resolved) ]]; then
  print -u2 "tools/test.sh: green, but Sources/Tests/Package files are not clean: no stamp"
else
  mkdir -p .build
  print -r -- "commit $(git rev-parse HEAD)
time $(date -u +%Y-%m-%dT%H:%M:%SZ)" > .build/last-green
  print -u2 "tools/test.sh: stamped .build/last-green at $(git rev-parse --short HEAD)"
fi
