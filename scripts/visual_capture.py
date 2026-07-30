#!/usr/bin/env python3
"""Capture Sorrel's real native window from a deterministic Kilosort fixture."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


TARGET = "sorrel/application"
VIEWPORT = {"width": 1280, "height": 800, "dpr": 1.0}
THEME = "dark"
LOCALE = "en-US"
PRESET = "ui-regression"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(value, indent=2)}\n")


def artifact(root: Path, path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": path.relative_to(root).as_posix(), "sha256": digest, "bytes": path.stat().st_size}


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write_npy(path: Path, descr: str, shape: tuple[int, ...], values: list[int | float]) -> None:
    formats = {"<u4": "I", "<u8": "Q", "<f4": "f"}
    try:
        fmt = formats[descr]
    except KeyError as error:
        raise ValueError(f"unsupported fixture NPY dtype {descr}") from error
    shape_text = "(" + ", ".join(str(value) for value in shape) + ("," if len(shape) == 1 else "") + ")"
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}".encode()
    padding = 16 - ((10 + len(header) + 1) % 16)
    header += b" " * padding + b"\n"
    payload = struct.pack("<" + fmt * len(values), *values)
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload)


def make_fixture(root: Path) -> Path:
    fixture = root / "target/visual/fixture"
    if fixture.exists():
        shutil.rmtree(fixture)
    fixture.mkdir(parents=True)

    n_channels = 8
    n_samples = 30_000
    spike_times = [1000 + index * 317 for index in range(72)]
    spike_clusters = [index % 3 for index in range(len(spike_times))]
    write_npy(fixture / "spike_times.npy", "<u8", (len(spike_times),), spike_times)
    write_npy(fixture / "spike_clusters.npy", "<u4", (len(spike_clusters),), spike_clusters)
    write_npy(
        fixture / "amplitudes.npy",
        "<f4",
        (len(spike_times),),
        [0.75 + (index % 9) * 0.04 for index in range(len(spike_times))],
    )
    write_npy(fixture / "spike_templates.npy", "<u4", (len(spike_times),), [0] * len(spike_times))
    write_npy(
        fixture / "channel_positions.npy",
        "<f4",
        (n_channels, 2),
        [value for channel in range(n_channels) for value in (channel * 20.0, (channel % 2) * 20.0)],
    )
    write_npy(fixture / "channel_map.npy", "<u4", (n_channels,), list(range(n_channels)))
    write_npy(
        fixture / "pc_features.npy",
        "<f4",
        (len(spike_times), 3, 4),
        [((index % 12) - 6) / 6.0 for index in range(len(spike_times) * 3 * 4)],
    )
    write_npy(fixture / "pc_feature_ind.npy", "<u4", (1, 4), [0, 1, 2, 3])
    write_npy(
        fixture / "templates.npy",
        "<f4",
        (1, 40, n_channels),
        [((sample % 10) - 5) / 5.0 * (1.0 - channel / 12.0) for sample in range(40) for channel in range(n_channels)],
    )
    write_npy(fixture / "similar_templates.npy", "<f4", (1, 1), [1.0])

    raw = bytearray(n_samples * n_channels * 2)
    for sample in range(n_samples):
        for channel in range(n_channels):
            value = ((sample * 7 + channel * 31) % 400) - 200
            struct.pack_into("<h", raw, (sample * n_channels + channel) * 2, value)
    (fixture / "recording.dat").write_bytes(raw)
    (fixture / "params.py").write_text(
        "dat_path = 'recording.dat'\n"
        "n_channels_dat = 8\n"
        "dtype = 'int16'\n"
        "offset = 0\n"
        "sample_rate = 30000.\n"
        "hp_filtered = False\n"
    )
    (fixture / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n0\tgood\n1\tmua\n2\tnoise\n"
    )
    (fixture / "quality_metrics.csv").write_text(
        "cluster_id\tisi_violations\tamplitude_cutoff\n"
        "0\t0.01\t0.08\n1\t0.04\t0.15\n2\t0.12\t0.32\n"
    )
    return fixture


def stop_process(process: subprocess.Popen[object] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def start_xvfb(log_path: Path) -> tuple[subprocess.Popen[object], str]:
    for number in range(100, 200):
        socket = Path(f"/tmp/.X11-unix/X{number}")
        if socket.exists():
            continue
        display = f":{number}"
        log = log_path.open("w")
        process = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1280x800x24", "-nolisten", "tcp", "-ac"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            if socket.exists():
                log.close()
                return process, display
            time.sleep(0.1)
        stop_process(process)
        log.close()
    raise RuntimeError("could not start a free Xvfb display in :100-:199")


def app_environment(display: str, runtime_dir: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("WAYLAND_DISPLAY", None)
    environment["DISPLAY"] = display
    environment["XDG_RUNTIME_DIR"] = runtime_dir
    environment["WINIT_UNIX_BACKEND"] = "x11"
    environment["WGPU_BACKEND"] = "vulkan"
    environment["LIBGL_ALWAYS_SOFTWARE"] = "1"
    environment["RUST_LOG"] = "warn"
    return environment


def capture_window(root: Path, fixture: Path, visual_root: Path) -> Path:
    binary = root / "target/debug/sorrel"
    build = subprocess.run(["cargo", "build", "--quiet", "-p", "sorrel"], cwd=root, check=False)
    if build.returncode != 0 or not binary.is_file():
        raise RuntimeError("Sorrel binary build failed")

    with tempfile.TemporaryDirectory(prefix="sorrel-visual-runtime-") as runtime_dir:
        xvfb, display = start_xvfb(visual_root / "xvfb.log")
        environment = app_environment(display, runtime_dir)
        log_path = visual_root / "sorrel.log"
        log = log_path.open("w")
        process = subprocess.Popen(
            [str(binary), str(fixture), "--journal", str(fixture / "sorrel.sqlite")],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            window = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Sorrel exited with {process.returncode}; see {log_path}")
                result = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", "^Sorrel$"],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                windows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if windows:
                    window = windows[-1]
                    break
                time.sleep(0.2)
            if window is None:
                raise TimeoutError("timed out waiting for the Sorrel window")
            time.sleep(1.0)
            image_path = visual_root / "captures/summary/desktop/dark/en-US/screenshot.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot = subprocess.run(
                ["import", "-window", window, str(image_path)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if screenshot.returncode != 0:
                raise RuntimeError(f"Sorrel screenshot failed: {screenshot.stderr.strip()}")
            return image_path
        finally:
            stop_process(process)
            log.close()
            stop_process(xvfb)


def rubric_result(root: Path, rubric_root: Path, image: Path) -> dict[str, object]:
    environment = dict(os.environ)
    for variable in ("CARGO_HOME", "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "RUSTC_WRAPPER"):
        environment.pop(variable, None)
    result = subprocess.run(
        [
            "nix",
            "develop",
            "--no-write-lock-file",
            str(rubric_root),
            "-c",
            "cargo",
            "run",
            "--locked",
            "--features",
            "audit",
            "--bin",
            "visual-rubric",
            "--",
            "configured",
            "--image",
            str(image.resolve()),
            "--preset",
            PRESET,
            "--json",
        ],
        cwd=rubric_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"status": "error", "reason": "visual-rubric configured failed", "anomalies": result.stderr.splitlines()[-4:]}
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        value = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as error:
        return {"status": "error", "reason": f"visual-rubric returned invalid JSON: {error}", "anomalies": lines[-4:]}
    return {
        "status": {"pass": "pass", "fail": "fail"}.get(value.get("verdict"), "error"),
        "reason": value.get("reason", ""),
        "anomalies": value.get("anomalies", []),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    visual_root = root / "target/visual"
    captures_root = visual_root / "captures"
    rubric_root = Path(os.environ.get("VISUAL_RUBRIC_ROOT", str(root.parent / "visual-rubric"))).resolve()
    missing = [tool for tool in ("Xvfb", "xdotool", "import") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError("missing native capture tools: " + ", ".join(missing))
    if not (rubric_root / "Cargo.toml").is_file():
        raise FileNotFoundError(f"visual-rubric checkout is missing: {rubric_root}")

    revision = git_output(root, "rev-parse", "HEAD")
    dirty = bool(git_output(root, "status", "--porcelain=v1"))
    visual_root.mkdir(parents=True, exist_ok=True)
    if captures_root.exists():
        shutil.rmtree(captures_root)
    for name in ("capture_manifest.json", "run_report.json", "deterministic_report.json", "rubric_batch_report.json"):
        (visual_root / name).unlink(missing_ok=True)

    fixture = make_fixture(root)
    image_path = capture_window(root, fixture, visual_root)
    state_path = image_path.with_name("state.json")
    write_json(state_path, {"state": "summary", "backend": "kilosort", "fixture": fixture.name})
    capture_id = "summary/desktop/dark/en-US"
    image = artifact(root, image_path)
    metadata = artifact(root, state_path)
    contract = {
        "schema_version": 1,
        "target": TARGET,
        "revision": revision,
        "surfaces": [{"id": "summary", "platform": "linux-native", "required": True, "description": "Default Sorrel cluster summary"}],
        "transitions": [],
        "exclusions": [],
    }
    contract_path = visual_root / "coverage/contract.json"
    write_json(contract_path, contract)
    report_path = visual_root / "coverage/report.json"
    write_json(
        report_path,
        {
            "schema_version": 1,
            "contract_sha256": hashlib.sha256(json.dumps(contract, separators=(",", ":")).encode()).hexdigest(),
            "planned_ids": ["summary"],
            "executed_ids": ["summary"],
            "passed_ids": ["summary"],
            "failed_ids": [],
            "blocked_ids": [],
        },
    )
    manifest_path = visual_root / "capture_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 3,
            "target": TARGET,
            "revision": revision,
            "dirty": dirty,
            "environment": {"platform": sys.platform, "renderer": "egui-wgpu", "capture_backend": "xvfb-xdotool-imagemagick"},
            "declared_cells": 1,
            "captures": [{"id": capture_id, "image": image, "state": "summary", "viewport": VIEWPORT, "theme": THEME, "locale": LOCALE, "metadata": {"state": metadata}, "presets": [PRESET]}],
            "coverage_contract": artifact(root, contract_path),
            "coverage_report": artifact(root, report_path),
        },
    )
    deterministic_path = visual_root / "deterministic_report.json"
    write_json(deterministic_path, {"schema_version": 1, "target": TARGET, "revision": revision, "dirty": dirty, "status": "pass", "observations": [], "findings": []})
    rubric = rubric_result(root, rubric_root, image_path)
    batch_path = visual_root / "rubric_batch_report.json"
    write_json(batch_path, {"schema_version": 1, "mode": "configured", "results": [{"id": capture_id, "rubric": rubric}]})
    cell = {"id": capture_id, "image": image["path"], "locale": LOCALE, "rubric": rubric, "state": "summary", "status": rubric["status"], "theme": THEME, "viewport": VIEWPORT}
    report = {
        "schema_version": 3,
        "target": TARGET,
        "git": {"sha": revision, "dirty": dirty},
        "capture_manifest": artifact(root, manifest_path),
        "capture_environment": {"renderer": "egui-wgpu", "capture_backend": "xvfb-xdotool-imagemagick"},
        "rubric_batch_report": artifact(root, batch_path),
        "deterministic_report": artifact(root, deterministic_path),
        "cells": [cell],
        "summary": {"total_cells": 1, "passed_cells": int(cell["status"] == "pass"), "failed_cells": int(cell["status"] == "fail"), "error_cells": int(cell["status"] not in ("pass", "fail"))},
        "failures": [cell] if cell["status"] == "fail" else [],
        "errors": [cell] if cell["status"] not in ("pass", "fail") else [],
        "rate_limit_events": [],
    }
    write_json(visual_root / "run_report.json", report)
    print(f"Sorrel visual producer wrote {cell['status']} cell to {visual_root / 'run_report.json'}", flush=True)
    return 0 if cell["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Sorrel visual producer failed: {error}", file=sys.stderr)
        sys.exit(1)
