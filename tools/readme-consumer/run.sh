#!/bin/zsh
# CuboKit README consumer: prove a stranger can use the kit from its README alone. Build tools/readme-consumer (a
# separate package that depends on CuboKit by path and holds README.md's Swift code verbatim) in DEBUG with no warnings,
# run it in an empty temporary directory, and require exit 0 and its summary line.
#
#   run.sh            (a) every line of every ```swift block in README.md must appear verbatim in the consumer's
#                     main.swift, so the README and the consumer cannot drift apart silently: change the README, and
#                     this fails until main.swift is updated; (b) swift build --jobs 2, zero warnings; (c) run it.
set -euo pipefail
here=${0:A:h}
root=${here:h:h}
die() { print -u2 "readme-consumer: $*"; exit 1 }
main=$here/Sources/readme-consumer/main.swift
readme=$root/README.md

# (a) The drift check: the lines of the README's ```swift blocks, blank lines aside, each found whole in main.swift.
blocks=$(awk '/^```swift[[:space:]]*$/ { inblock = 1; nblocks++; next } /^```/ { inblock = 0; next } inblock && NF { print }' $readme)
nblocks=$(grep -cE '^```swift[[:space:]]*$' $readme || true)
(( nblocks > 0 )) || die "README.md has no \`\`\`swift block"
lines=0
missing=0
while IFS= read -r line; do
  lines=$(( lines + 1 ))
  if ! grep -qxF -- "$line" $main; then
    print -u2 "readme-consumer: README line not in main.swift: $line"
    missing=$(( missing + 1 ))
  fi
done <<< "$blocks"
(( missing == 0 )) || die "$missing of $lines README code line(s) are missing from ${main#$root/}: update the consumer"

# (b) Build.
cd $here
mkdir -p .build
log=.build/readme-consumer-build.log
# --disable-dependency-cache: SwiftPM's shared repository cache can print a 'skipping cache' warning on a CI runner,
# which the warning count would take for the kit's own (DiffractoKit 0.4.1's CI).
if ! swift build --disable-dependency-cache --jobs 2 -c debug >$log 2>&1; then
  grep -E "error" $log | head -20 >&2
  die "BUILD FAILED (full log: $here/$log)"
fi
warnings=$(grep -c "warning:" $log || true)
(( warnings == 0 )) || { grep "warning:" $log | head >&2; die "$warnings warning(s) in the build" }

# (c) Run, in an empty temporary directory.
bin=$(swift build --disable-dependency-cache --jobs 2 -c debug --show-bin-path)
work=$(mktemp -d)
trap 'rm -rf $work' EXIT
out=$here/.build/readme-consumer-run.log
set +e
(cd $work && $bin/readme-consumer) >$out 2>&1
rc=$?
set -e
cat $out
(( rc == 0 )) || die "FAILED (exit $rc)"
grep -q "^readme-consumer summary:" $out || die "no summary line in the output"
head=$(git -C $root --no-optional-locks rev-parse --short HEAD 2>/dev/null || print unknown)
print "readme-consumer: passed — $nblocks swift block(s), $lines README line(s) found verbatim, compiled warning-free in Debug and run, exit 0 (CuboKit $head)"
