"""The program behind the smoke's fake `claude.cmd` / `codex.cmd` / `opencode.cmd`
shims: writes what it received (argv, stdin, and the text of every argument that
names an existing file) as JSON to the file named by $ARGV_DUMP_FILE."""

import json
import os
import sys

argv = sys.argv[1:]
stdin = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.buffer.read().decode("utf-8")
files = {}
for a in argv:
    if os.path.isfile(a):
        with open(a, encoding="utf-8") as f:
            files[a] = f.read()
with open(os.environ["ARGV_DUMP_FILE"], "w", encoding="utf-8") as f:
    json.dump({"argv": argv, "stdin": stdin, "files": files}, f)
