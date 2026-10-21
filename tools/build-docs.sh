#!/bin/bash
# Builds the DocC site for every module with a .docc catalog and merges it into one static site.
#   tools/build-docs.sh <output-dir> [<hosting-base-path>]     e.g. tools/build-docs.sh .docs-site /CuboKit
# macOS with Xcode 16 or later. Output is gitignored (.docs-site).
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:?output dir}"
base="${2:-}"
derived="$(mktemp -d)"
trap 'rm -rf "$derived"' EXIT

modules=$(find Sources -maxdepth 2 -name '*.docc' -exec basename {} .docc \; | sort)
archives=()
for m in $modules; do
  xcodebuild docbuild -scheme "$m" -destination 'generic/platform=macOS' -derivedDataPath "$derived" >"$derived/$m.log" 2>&1 \
    || { tail -30 "$derived/$m.log"; echo "docbuild failed for $m"; exit 1; }
  archives+=("$(find "$derived/Build/Products" -name "$m.doccarchive" -maxdepth 3 | head -1)")
done

merged="$derived/CuboKit.doccarchive"
xcrun docc merge "${archives[@]}" --output-path "$merged"
rm -rf "$out"
xcrun docc process-archive transform-for-static-hosting "$merged" --output-path "$out" ${base:+--hosting-base-path "$base"}
# the site root has no page of its own: send it to the merged documentation landing page
printf '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=%s/documentation/">' "${base}" > "$out/index.html"
echo "built $out ($(echo "$modules" | wc -l | tr -d ' ') modules)"
