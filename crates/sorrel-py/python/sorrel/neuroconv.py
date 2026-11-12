"""NeuroConv data interface for a curated Sorrel session.

Needs NeuroConv: ``pip install "sorrel-ephys[neuroconv]"``. Use it next to
the recording's interface in a ``-to-nwb`` converter::

    from neuroconv import ConverterPipe
    from neuroconv.datainterfaces import SpikeGLXRecordingInterface
    from sorrel.neuroconv import SorrelSortingInterface

    converter = ConverterPipe(data_interfaces={
        "recording": SpikeGLXRecordingInterface(file_path=...),
        "sorting": SorrelSortingInterface(folder_path="my_run/phy"),
    })
"""
from __future__ import annotations

from neuroconv import BaseDataInterface
from pydantic import DirectoryPath
from pynwb import NWBFile

from . import Session
from .nwb import add_units_to_nwbfile

__all__ = ["SorrelSortingInterface"]


class SorrelSortingInterface(BaseDataInterface):
    """Curated units, with Sorrel's labels, lineage and provenance.

    Parameters
    ----------
    folder_path : DirectoryPath
        The Sorrel dataset: a phy directory or a SortingAnalyzer
        binary_folder. Opening it takes the curation journal's writer lock,
        so close the Sorrel window on it first.
    verbose : bool, default: False
    """

    display_name = "Sorrel Sorting"
    keywords = ("extracellular electrophysiology", "spike sorting", "curation")
    associated_suffixes = (".npy", ".tsv", ".json")
    info = "Interface for units curated in Sorrel (phy directory or SortingAnalyzer folder)."

    def __init__(self, folder_path: DirectoryPath, verbose: bool = False):
        super().__init__(folder_path=folder_path, verbose=verbose)
        self._starting_time = 0.0

    def set_aligned_starting_time(self, aligned_starting_time: float) -> None:
        """Offset, in seconds, of the sorted recording's first sample from the
        NWB session start; added to every spike time."""
        self._starting_time = float(aligned_starting_time)

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        units_description: str | None = None,
    ) -> None:
        """Write ``nwbfile.units`` with ``sorrel.nwb.add_units_to_nwbfile``;
        ``units_description`` prefixes Sorrel's provenance summary."""
        add_units_to_nwbfile(
            Session.open(self.source_data["folder_path"]),
            nwbfile,
            starting_time=self._starting_time,
            description=units_description,
        )
