"""Windows smoke helper: do the `--llm` CLI backends survive an npm `.cmd` shim?

  uv run --no-project --with numpy python release/smoke-llm-shim.py SORREL_AUTO.py DUMP_DIR

Run with the shim dir (claude.cmd, codex.cmd, opencode.cmd, each
`@python argv_dump.py %*`) first on PATH. For every backend it calls the
sidecar's own `ask_cli` with hostile prompts (quotes, % ^ & | < > ! and
newlines), records the argv and stdin `sorrel_auto` intended (by wrapping
subprocess.run), and compares them with what the shim's program received
(argv_dump.py writes `$ARGV_DUMP_FILE`). Fails on any difference.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sidecar, dump_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(sidecar.parent))
import sorrel_auto as sa  # noqa: E402

SYSTEM = 'Rubric "quoted" 100% ^caret & amp | pipe < lt > gt !bang\nsecond line'
MESSAGE = 'Decide "this" %PATH% ^ & | < > !\nunit 7'

intended: list[dict] = []
real_run = subprocess.run


def spy(argv, **kw):
    intended.append({"argv": [str(a) for a in argv], "stdin": kw.get("input")})
    return real_run(argv, **kw)


sa.subprocess.run = spy
failed = False
for backend in ("claude", "codex", "opencode"):
    intended.clear()
    dump = dump_dir / f"{backend}.json"
    dump.unlink(missing_ok=True)
    os.environ["ARGV_DUMP_FILE"] = str(dump)
    exe = sa.shutil.which(backend)
    assert exe and (os.name != "nt" or exe.lower().endswith((".cmd", ".bat"))), f"{backend} resolves to {exe}, not a shim"
    sa.ask_cli(backend, "smoke-model", SYSTEM, "unit", MESSAGE)  # the shim's reply is not a verdict
    assert len(intended) == 1 and dump.exists(), f"{backend}: shim never ran ({intended})"
    want, got = intended[0], json.loads(dump.read_text(encoding="utf-8"))
    problems = []
    if got["argv"] != want["argv"][1:]:
        problems.append(f"argv\n  intended {want['argv'][1:]!r}\n  received {got['argv']!r}")
    # text-mode pipes turn \n into \r\n on Windows; that is not the shim's doing
    if got["stdin"].replace("\r\n", "\n") != (want["stdin"] or ""):
        problems.append(f"stdin\n  intended {want['stdin']!r}\n  received {got['stdin']!r}")
    texts = [t.replace("\r\n", "\n") for t in got["files"].values()]
    needle = SYSTEM if backend == "claude" else MESSAGE
    # opencode reads a file only when it is a shim (Windows); codex gets its text on stdin
    if backend == "claude" or (backend == "opencode" and os.name == "nt"):
        if not any(needle in t for t in texts):
            problems.append(f"no attached file carries {needle!r}: {got['files']!r}")
    print(f"{backend}: {'FAIL' if problems else 'argv round-trips'}: {got['argv']!r}")
    for p in problems:
        print(f"  {backend} {p}")
    failed |= bool(problems)
sys.exit(1 if failed else 0)
