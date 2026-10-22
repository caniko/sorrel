"""Sorrel — native curation GUI for SpikeInterface sortings.

Two ways in:

In-process (no window), for notebooks and pipelines::

    s = sorrel.Session.open("my_run/phy")      # or a SortingAnalyzer binary_folder
    s.qc()                                     # list of dicts, cluster_qc.tsv columns
    s.suggest()                                # {"merges": [...], "splits": [...]}
    s.set_label(3, "good"); s.merge([4, 5], 6) # journaled, undoable
    s.save()                                   # phy files + curation.json

GUI (subprocess, blocks until the window closes)::

    sorrel.open(recording, sorting)            # builds a SortingAnalyzer
    sorrel.open_analyzer(sorting_analyzer)     # launches on its binary_folder
    sorrel.open_phy(path) / sorrel.launch(path)

Reading curation back into SpikeInterface::

    result_dir = sorrel.open_analyzer(analyzer)
    curated = sc.apply_curation(analyzer, sorrel.read_curation(result_dir))
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import _native

__version__ = _native.version()
__all__ = [
    "Session",
    "export_phy",
    "inspect_phy",
    "launch",
    "open",
    "open_analyzer",
    "open_phy",
    "read_curation",
]


class Session:
    """In-process curation session; see the module docstring.

    Thin wrapper over the native class that decodes its JSON payloads.
    """

    def __init__(self, native: Any):
        self._n = native

    @classmethod
    def open(
        cls,
        path: str | os.PathLike,
        *,
        journal: str | os.PathLike | None = None,
        discard_journal: bool = False,
    ) -> "Session":
        return cls(
            _native.Session.open(
                str(path),
                None if journal is None else str(journal),
                discard_journal,
            )
        )

    # -- read -------------------------------------------------------------
    @property
    def n_clusters(self) -> int:
        return self._n.n_clusters()

    @property
    def sample_rate(self) -> float:
        return self._n.sample_rate()

    @property
    def version(self) -> int:
        """Monotonic curation-state version; changes on every edit."""
        return self._n.version()

    def labels(self) -> list[str]:
        return self._n.labels()

    def spike_times(self, cluster: int) -> list[int]:
        return self._n.spike_times(cluster)

    def qc(self) -> list[dict[str, Any]]:
        """One dict per cluster with the ``cluster_qc.tsv`` columns (NaN → None)."""
        return json.loads(self._n.qc_json())

    def suggest(self) -> dict[str, list[dict[str, Any]]]:
        return json.loads(self._n.suggest_json())

    def curation(self) -> dict[str, Any]:
        """SpikeInterface curation document for the current state."""
        return json.loads(self._n.curation_json())

    # -- edit (journaled, undoable) --------------------------------------
    def set_label(self, cluster: int, label: str) -> int:
        return self._n.set_label(cluster, label)

    def merge(self, sources: list[int], target: int) -> int:
        return self._n.merge(list(sources), target)

    def split(self, cluster: int, spike_idx: list[int]) -> int:
        """Move ``spike_idx`` (positions in the cluster's time-sorted spikes)
        into a new cluster; returns the new cluster id."""
        return self._n.split(cluster, list(spike_idx))

    def undo(self) -> int:
        return self._n.undo()

    def redo(self) -> int:
        return self._n.redo()

    def save(self) -> str:
        """Write ``spike_clusters.npy``, ``cluster_group.tsv`` and
        ``curation.json``; returns the directory."""
        return self._n.save()


def read_curation(path: str | os.PathLike) -> dict[str, Any]:
    """Load ``curation.json`` from a dataset directory (or the file itself),
    ready for ``spikeinterface.curation.apply_curation``."""
    p = Path(path)
    if p.is_dir():
        p = p / "curation.json"
    with p.open() as f:
        return json.load(f)


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


def open(
    recording: Any,
    sorting: Any,
    *,
    output_folder: str | os.PathLike | None = None,
    keep_temp: bool = True,
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
    keep_temp: bool = True,
    binary: str | None = None,
) -> str:
    """Curate a `SortingAnalyzer` in Sorrel — no phy dump.

    Fast path: if the analyzer is already a `binary_folder` on disk and the
    user didn't ask for a different `output_folder`, launch sorrel directly
    on the existing folder. Otherwise call `analyzer.save_as("binary_folder")`
    into either `output_folder` or a tempdir.

    Returns the directory the GUI loaded from. It holds the curation
    journal and, after a save in the GUI, ``curation.json``; feed that to
    ``read_curation`` + ``apply_curation`` to get the curated analyzer.
    The tempdir is kept by default for exactly that reason; pass
    ``keep_temp=False`` to delete it (and the curation) on exit.
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
