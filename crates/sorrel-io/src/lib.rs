//! sorrel-io: backend-specific data providers exposed as concrete structs.
//!
//! The [`DataProvider`] trait is a *generic bound*, never a `dyn` object: the
//! `sorrel-data` core takes a `P: DataProvider` so the compiler emits a
//! bespoke, fully-inlined session for each backend.

pub mod cbin;
pub mod curation_json;
pub mod detect;
pub mod events;
pub mod extras;
pub mod kilosort;
pub mod ks4_rez;
pub mod mda;
pub mod npy;
pub mod nwb;
pub mod open_ephys;
pub mod params;
pub mod probeinterface;
pub mod provider;
pub mod sorting_analyzer;
pub mod spikeglx;

pub use cbin::CbinReader;
pub use curation_json::{read_curation, read_curation_labels, CurationImport};
pub use detect::{detect_backend, detect_sorter, Backend};
pub use events::Events;
pub use extras::{
    HasAmplitudes, HasGeometry, HasPcFeatures, HasQualityMetrics, HasSpikeTemplates,
    HasTemplateWaveforms, QualityMetrics,
};
pub use kilosort::KilosortProvider;
pub use mda::MdaHeader;
pub use open_ephys::{OebinMeta, OebinStream};
pub use params::PhyParams;
pub use probeinterface::ProbeGeometry;
pub use provider::{
    ChannelId, ClusterId, DataProvider, OptionalRow, RawFile, SampleIndex, TraceDtype,
    TraceSamples, TraceSlice, UnitAnnotation,
};
pub use sorting_analyzer::SortingAnalyzerProvider;
pub use spikeglx::SpikeGlxMeta;

#[cfg(feature = "hdf5")]
pub use ks4_rez::{Ks4RezOpenParams, Ks4RezProvider};
#[cfg(feature = "hdf5")]
pub use nwb::NwbProvider;
