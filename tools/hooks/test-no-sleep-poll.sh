#!/bin/bash
# Proof for no-sleep-poll.sh: feeds synthetic hook JSON, prints PASS/FAIL per case, exits non-zero on any failure.
cd "$(dirname "$0")" || exit 1
fail=0
check() { # expected-exit command
  local json; json=$(/usr/bin/python3 -I -c 'import json,sys; print(json.dumps({"tool_name":"Bash","tool_input":{"command":sys.argv[1]}}))' "$2")
  ./no-sleep-poll.sh <<<"$json" 2>/dev/null; local rc=$?
  if [[ $rc == "$1" ]]; then echo "PASS (exit $rc): $2"; else echo "FAIL (exit $rc, wanted $1): $2"; fail=1; fi
}
check 2 'sleep 60'
check 2 'sleep 30'
check 2 'sleep 45 && echo done'
check 2 'echo a; /bin/sleep 120'
check 2 'sleep 1m'
check 2 'sleep 0.5h'
check 2 'while true; do x; sleep 2; done'
check 2 'until grep -q x f; do sleep 1; done'
check 0 'echo "sleep 60"'   # quoted: passes (sleep must follow whitespace or an operator)
check 0 'sleep 1'
check 0 'sleep 5 && ls'
check 0 'sleep 29.9'
check 0 'git log'
check 0 'swift test'
check 0 'tools/test.sh --filter Sleeper'
check 2 'pkill -f swift'
check 2 'pkill -f xctest'
check 2 'killall swift-build'
check 2 'killall -9 xctest'
check 2 'sudo pkill -f foo'
check 0 'kill 12345'
check 0 'kill -9 12345'
check 0 'kill %1'
check 0 'pkill -P 4242'
check 0 'pkill -f /private/tmp/claude-501/x/scratchpad/lane/.build'
check 2 'git log --grep killall'   # kept as in FormatoKit and SpectroKit: any killall word is refused (false positive)
check 0 'grep -n killallowed notes.txt'
exit $fail
