//! The learned curator foundation.
//!
//! [`gbdt`] is a dependency-free, deterministic gradient-boosted decision
//! tree trainer and predictor; [`bundle`] packages one or more fitted models
//! ("heads") with their feature schemas and provenance into a single portable
//! `.sorrel-model` file; [`table`] is the on-disk feature/target dataset
//! format; [`features`] extracts those tables from a session; and [`cv`]/
//! [`metrics`] run grouped cross-validation and score it. The CLI/MCP
//! surfaces build on these pieces in later work.

pub mod bundle;
pub mod cv;
pub mod features;
pub mod gbdt;
pub mod metrics;
pub mod shard;
pub mod table;

/// Version stamped into every extracted table and into a head's feature
/// schema. Bump it whenever an input-only column is added, removed, renamed or
/// its definition changes: a model trained on another version must refuse to
/// score silently. Target-only changes bump [`TARGET_VERSION`] instead, so a
/// retarget can reuse the expensive input-only shard.
pub const EXTRACTOR_VERSION: &str = "learn-v2";

/// Version of the truth-derived `y_*` columns. It is independent of
/// [`EXTRACTOR_VERSION`] so changing a target definition (or the truth behind
/// it) invalidates only the cheap target shard, never the input-only features.
pub const TARGET_VERSION: &str = "targets-v1";

#[cfg(test)]
pub(crate) mod test_support {
    //! Deterministic test data helpers. The workspace has no `rand`
    //! dependency, and tests need the same numbers on every platform, so a
    //! tiny splitmix64 PRNG is all that is required.

    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Self(seed)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        pub fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }

        pub fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n.max(1) as u64) as usize
        }

        pub fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }
}
