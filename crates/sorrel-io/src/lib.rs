//! sorrel-io: backend-specific data providers exposed as concrete structs.
//!
//! The [`DataProvider`] trait is a *generic bound*, never a `dyn` object: the
//! `sorrel-data` core takes a `P: DataProvider` so the compiler emits a
//! bespoke, fully-inlined session for each backend.

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

pub use extras::{
    HasAmplitudes, HasGeometry, HasPcFeatures, HasQualityMetrics, HasSpikeTemplates,
    HasTemplateWaveforms,
};
pub use kilosort::KilosortProvider;
pub use mda::MdaHeader;
pub use open_ephys::{OebinMeta, OebinStream};
pub use params::PhyParams;
pub use probeinterface::ProbeGeometry;
pub use provider::{
    ChannelId, ClusterId, DataProvider, SampleIndex, TraceDtype, TraceSamples, TraceSlice,
};
pub use sorting_analyzer::SortingAnalyzerProvider;
pub use spikeglx::SpikeGlxMeta;

#[cfg(feature = "hdf5")]
pub use ks4_rez::{Ks4RezOpenParams, Ks4RezProvider};
#[cfg(feature = "hdf5")]
pub use nwb::NwbProvider;
