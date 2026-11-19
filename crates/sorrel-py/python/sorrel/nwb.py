"""Write a curated Sorrel session into an NWB file as its Units table.

Needs PyNWB: ``pip install "sorrel-ephys[nwb]"``. Pair with a recording's
NeuroConv interface for raw traces, electrodes and recording metadata.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from hdmf.common import VectorData, VectorIndex
from pynwb.misc import Units

if TYPE_CHECKING:
    from pathlib import Path

    from pynwb import NWBFile

    from . import Session

__all__ = ["add_units_to_nwbfile"]


def add_units_to_nwbfile(
    session: Session | str | Path,
    nwbfile: NWBFile,
    *,
    starting_time: float = 0.0,
    description: str | None = None,
) -> Units:
    """Add the curated units of ``session`` to ``nwbfile`` as ``nwbfile.units``.

    One row per cluster that still has spikes, with its backend unit id.
    Columns include spike times, quality, origin, parents, tags, QC metrics
    and the sorter's unit names when they differ from the integer ids.
    The table description records curation provenance, and ``was_generated_by``
    gets a Sorrel entry.

    Times in seconds are ``sample_index / exact_sample_rate + starting_time``;
    resolution is ``1 / exact_sample_rate``. Use the recording's first-sample
    offset as ``starting_time``. Multi-segment SortingAnalyzers are rejected.
    Refuses an existing Units table rather than merging into it.
    """
    from . import Session

    if not isinstance(session, Session):
        session = Session.open(session)
    if nwbfile.units is not None:
        raise ValueError("nwbfile already has a units table; sorrel will not merge into it")

    summary, resolution, ids, specs, generated_by = session._n._nwb_units(starting_time)
    columns = []
    for name, doc, dtype, values, index in specs:
        # Explicit types also cover ragged columns whose rows are all empty.
        data = VectorData(name=name, description=doc, data=np.asarray(values, dtype=dtype))
        columns.append(data)
        if index is not None:
            columns.append(VectorIndex(name=f"{name}_index", data=np.asarray(index, dtype=np.uint64),
                                       target=data))

    units = Units(name="units", description=f"{description} {summary}" if description else summary,
                  resolution=resolution, id=ids, columns=columns)
    nwbfile.units = units

    entry = tuple(generated_by)
    existing = [tuple(e) for e in nwbfile.was_generated_by or ()]
    if entry not in existing:
        # The attribute is write-once and may be a tuple; replace the field.
        nwbfile.fields["was_generated_by"] = [*existing, entry]
    return units
