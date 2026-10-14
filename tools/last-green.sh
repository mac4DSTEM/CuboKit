#!/bin/zsh
# Prints .build/last-green (written by tools/test.sh after a full green run on a clean tree) and whether HEAD equals it.
cd "${0:A:h}/.."
[[ -f .build/last-green ]] || { print "no stamp: tools/test.sh has not passed a full clean run here"; exit 1 }
cat .build/last-green
g=$(sed -n 's/^commit //p' .build/last-green)
if [[ $g == $(git rev-parse HEAD) ]]; then print "HEAD equals the last green commit"; else print "HEAD $(git rev-parse --short HEAD) differs from last green ${g[1,7]}"; exit 1; fi
