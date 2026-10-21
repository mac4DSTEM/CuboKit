#!/bin/zsh
# CuboKit test run: `swift test -c release` (release: a versioned dependency cannot carry Debug -O; Debug at -Onone is
# many times slower on the reducers). Never `swift build --build-tests -c release` (fails on this toolchain).
# Real data: RealDataTests read $CUBOKIT_REAL_DATA (default ~/Library/Caches/CuboKit/real-data, written by
# `tools/slice-1/pins.py real`) and skip by name when it is absent; this script says so.
set -euo pipefail
cd "${0:A:h}/.."
log=$(mktemp -t cubokit-test)
swift test -c release "$@" 2>&1 | tee $log
rc=${pipestatus[1]}
(( rc == 0 )) || exit $rc
skipped=$(grep -c "skipped.*real data absent" $log || true)
(( skipped == 0 )) || print -u2 "tools/test.sh: green, but $skipped real-data test(s) skipped (no real data here)"
