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
exit $fail
