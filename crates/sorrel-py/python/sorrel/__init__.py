"""Sorrel — native curation GUI for SpikeInterface sortings.

Public API:

    sorrel.open(recording, sorting, **kwargs)
        Curate a (recording, sorting) pair. Builds a SortingAnalyzer and
        hands it to `open_analyzer`.

    sorrel.open_analyzer(sorting_analyzer, **kwargs)
        Curate a SortingAnalyzer. **No phy dump.** If the analyzer is already
        a `binary_folder` on disk, sorrel launches on that folder directly
        (zero copy). Otherwise it's saved with the lightweight
        `save_as("binary_folder")` into a tempdir before launch.

    sorrel.export_phy(...) / sorrel.open_phy(path)
        Escape hatches for the legacy phy-export workflow.

    sorrel.launch(path, binary=None)
        Launch the sorrel binary on an existing data directory (phy or
        SortingAnalyzer binary_folder — the binary auto-detects).

    sorrel.inspect_phy(path)
        Return (n_clusters, n_samples, sample_rate) for a phy directory.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import _native

__version__ = _native.version()
__all__ = [
    "open",
    "open_analyzer",
    "open_phy",
    "export_phy",
    "launch",
    "inspect_phy",
]


def inspect_phy(path: str | os.PathLike) -> tuple[int, int, float]:
    return _native.inspect_phy(str(path))


def launch(
    path: str | os.PathLike,
    *,
    binary: str | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Spawn the sorrel binary on `path`. Blocks until close."""
    return _native.launch(str(path), binary, extra_args)


def open(  # noqa: A001 — intentionally shadowing built-in for ergonomics
    recording: Any,
    sorting: Any,
    *,
    output_folder: str | os.PathLike | None = None,
    keep_temp: bool = False,
    binary: str | None = None,
    **analyzer_kwargs: Any,
) -> str:
    """Curate `(recording, sorting)` in Sorrel.

    Wraps the inputs in a SpikeInterface `SortingAnalyzer` (in-memory) and
    forwards to `open_analyzer`. `analyzer_kwargs` are passed through to
    `si.create_sorting_analyzer`.
    """
    try:
        import spikeinterface as si
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "sorrel.open() requires spikeinterface; "
            "install with `pip install spikeinterface`"
        ) from e
    analyzer = si.create_sorting_analyzer(
        sorting=sorting, recording=recording, **analyzer_kwargs
    )
    return open_analyzer(
        analyzer,
        output_folder=output_folder,
        keep_temp=keep_temp,
        binary=binary,
    )


def open_analyzer(
    sorting_analyzer: Any,
    *,
    output_folder: str | os.PathLike | None = None,
    keep_temp: bool = False,
    binary: str | None = None,
) -> str:
    """Curate a `SortingAnalyzer` in Sorrel — no phy dump.

    Fast path: if the analyzer is already a `binary_folder` on disk and the
    user didn't ask for a different `output_folder`, launch sorrel directly
    on the existing folder. Otherwise call `analyzer.save_as("binary_folder")`
    into either `output_folder` or a tempdir.

    Returns the directory the GUI loaded from (so callers can re-load it
    with `si.load_sorting_analyzer(...)` to read curation back).
    """
    existing = _on_disk_binary_folder(sorting_analyzer)
    if existing is not None and output_folder is None:
        launch(existing, binary=binary)
        return str(existing)

    if output_folder is None:
        out = Path(tempfile.mkdtemp(prefix="sorrel-sa-"))
        cleanup = not keep_temp
    else:
        out = Path(output_folder)
        cleanup = False

    _save_as_binary_folder(sorting_analyzer, out)

    try:
        launch(out, binary=binary)
    finally:
        if cleanup:
            shutil.rmtree(out, ignore_errors=True)
    return str(out)


def open_phy(
    path: str | os.PathLike,
    *,
    binary: str | None = None,
) -> int:
    """Open an existing phy/Kilosort directory directly."""
    return launch(path, binary=binary)


def export_phy(
    sorting_analyzer: Any,
    output_folder: str | os.PathLike,
    *,
    binary: str | None = None,
    launch_after: bool = True,
    **export_kwargs: Any,
) -> str:
    """Legacy escape hatch: dump a `SortingAnalyzer` via SpikeInterface's
    `export_to_phy` and (optionally) open it. Use only when you specifically
    need the phy directory layout — `open_analyzer` is faster and lighter.
    """
    try:
        from spikeinterface.exporters import export_to_phy
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "sorrel.export_phy() requires spikeinterface[exporters]"
        ) from e
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    export_kwargs.setdefault("copy_binary", True)
    export_kwargs.setdefault("compute_pc_features", True)
    export_to_phy(sorting_analyzer, output_folder=str(out), **export_kwargs)
    if launch_after:
        launch(out, binary=binary)
    return str(out)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _on_disk_binary_folder(analyzer: Any) -> Path | None:
    """Return the on-disk `binary_folder` path for `analyzer`, or None.

    SI's `SortingAnalyzer` exposes both `format` (``'memory' | 'binary_folder'
    | 'zarr'``) and `folder`. We require both: format == 'binary_folder' and a
    real directory at `folder`. Anything else (memory, zarr, missing folder)
    means we have to materialise.
    """
    fmt = getattr(analyzer, "format", None)
    folder = getattr(analyzer, "folder", None)
    if fmt != "binary_folder" or folder is None:
        return None
    p = Path(folder)
    return p if p.is_dir() else None


def _save_as_binary_folder(analyzer: Any, out: Path) -> None:
    """Materialise an in-memory or zarr analyzer to `out` as binary_folder.

    `save_as` requires the destination not to exist; if `out` is empty we
    pass it through, else we save into a fresh subdirectory and shuffle.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"sorrel: output_folder {out} is not empty; refuse to overwrite"
        )
    if out.exists():
        out.rmdir()  # save_as wants the path absent
    analyzer.save_as(format="binary_folder", folder=str(out))
