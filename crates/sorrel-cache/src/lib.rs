//! Derived artifact cache for expensive Sorrel computations.
//!
//! This cache is derivable; deleting it must never lose user data. It is for
//! artifacts that can be recomputed from provider content, algorithm
//! parameters, an explicit algorithm version, and the journal head. Curation
//! state remains load-bearing in the journal, which uses its own `baseline_hash`
//! sealing and MessagePack format. Any observable change to an artifact's shape
//! or computation must bump the `algo_version` included in the cache key.
//!
//! Archives are written with rkyv's bytechecked, little-endian, 64-bit pointer
//! layout. Sorrel does not support 32-bit hosts, and the cache should be
//! treated as a local rebuildable optimization rather than a portable exchange
//! format.

mod gc;
mod key;
mod store;

pub use gc::{GcConfig, GcSummary};
pub use key::{CacheKey, Fingerprint, FingerprintBuilder};
pub use store::{CacheError, CacheRead, CacheStore};

#[cfg(test)]
mod tests;
