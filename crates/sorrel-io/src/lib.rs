//! sorrel-io: backend-specific data providers exposed as concrete structs.
//!
//! The [`DataProvider`] trait is a *generic bound*, never a `dyn` object: the
//! `sorrel-data` core takes a `P: DataProvider` so the compiler emits a
//! bespoke, fully-inlined session for each backend.

pub mod kilosort;
pub mod npy;
pub mod provider;

pub use kilosort::KilosortProvider;
pub use provider::{ChannelId, ClusterId, DataProvider, SampleIndex, TraceSlice};
