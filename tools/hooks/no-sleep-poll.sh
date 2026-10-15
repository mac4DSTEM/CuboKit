#!/bin/bash
# PreToolUse hook (matcher "Bash"): refuses sleep and poll loops. Reads the hook JSON on stdin, takes .tool_input.command.
# Refused with exit 2 (stderr goes to the agent): (1) any `sleep` of 30 s or more (`sleep 45`, `sleep 1m`, `sleep 0.5h`,
# `/bin/sleep 120`, also inside && / ; chains); (2) a `while` or `until` loop with `sleep` anywhere in the command, any length.
# (3) machine-wide kills by name (2026-10-15, after two on the night of 2026-10-14 stopped other sessions' jobs): `killall`
# in any form, and `pkill` unless it names a parent pid (`-P <pid>`) or its pattern holds this machine's session scratchpad
# path (`/private/tmp/claude-` or `/scratchpad/`); `kill <pid>` passes.
# Everything else passes (`sleep 1`, `sleep 5`, `git log`). Matched as text, not parsed: a sleep right after a
# quote passes, a quoted sleep after a space is refused (accepted false positive).
# Wait on background-task notifications (Bash run_in_background) or the Monitor tool instead.
exec /usr/bin/python3 -I -c '
import json, re, sys
try:
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
except Exception:
    sys.exit(0)
if not isinstance(cmd, str):
    sys.exit(0)
msg = ("Refused by tools/hooks/no-sleep-poll.sh: %s. Do not sleep or poll. Run the job with Bash run_in_background "
       "and wait for its completion notification, or use the Monitor tool (an until-loop inside Monitor is its job).")
sleeps = list(re.finditer(r"(?:^|[\s;&|()`{])(?:/\S*/)?sleep\s+([0-9]*\.?[0-9]+)([smhd]?)(?![\w.])", cmd))
if re.search(r"\b(?:while|until)\b", cmd) and re.search(r"(?:^|[\s;&|()`{])(?:/\S*/)?sleep\b", cmd):
    print(msg % "a while/until loop containing sleep", file=sys.stderr); sys.exit(2)
kill_msg = ("Refused by tools/hooks/no-sleep-poll.sh: %s kills by name on the whole machine (other sessions run here). "
            "Kill by pid (`kill <pid>`, the pid from your own background job), `pkill -P <pid>`, or a pkill pattern holding "
            "your scratchpad path.")
if re.search(r"(?:^|[\s;&|()`{])(?:/\S*/)?killall\b", cmd):
    print(kill_msg % "killall", file=sys.stderr); sys.exit(2)
for m in re.finditer(r"(?:^|[\s;&|()`{])(?:/\S*/)?pkill\b([^;&|]*)", cmd):
    args = m.group(1)
    if not (re.search(r"-P\s*\d+", args) or "/private/tmp/claude-" in args or "/scratchpad/" in args):
        print(kill_msg % "pkill without a parent pid or a scratchpad path", file=sys.stderr); sys.exit(2)
for m in sleeps:
    secs = float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    if secs >= 30:
        print(msg % ("sleep of %g s (limit: under 30 s)" % secs), file=sys.stderr); sys.exit(2)
sys.exit(0)
'
