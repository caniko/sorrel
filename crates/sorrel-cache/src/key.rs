use std::fmt;

/// A typed cache key for one kind of derived artifact.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CacheKey {
    /// Artifact class, e.g. `"amplitudes"` or `"pc_subspace"`.
    pub kind: &'static str,
    /// Bump when the artifact shape or computation changes observably.
    pub algo_version: u32,
    /// BLAKE3 digest over input identity, parameters, algorithm version, and
    /// journal head.
    pub fingerprint: Fingerprint,
}

impl CacheKey {
    /// Builds a key from an already-finished fingerprint.
    ///
    /// Callers should include `algo_version` in the [`FingerprintBuilder`] via
    /// [`FingerprintBuilder::add_algo_version`]. The field is repeated here so
    /// logs and future index tooling can inspect keys without parsing digest
    /// inputs.
    pub const fn new(kind: &'static str, algo_version: u32, fingerprint: Fingerprint) -> Self {
        Self {
            kind,
            algo_version,
            fingerprint,
        }
    }
}

/// A 32-byte BLAKE3 fingerprint.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Fingerprint([u8; 32]);

impl Fingerprint {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    pub fn builder() -> FingerprintBuilder {
        FingerprintBuilder::new()
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Fingerprint").field(&self.hex()).finish()
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

/// Domain-separated BLAKE3 fingerprint builder.
#[derive(Clone)]
pub struct FingerprintBuilder {
    hasher: blake3::Hasher,
}

impl FingerprintBuilder {
    pub fn new() -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"sorrel-cache-fingerprint-v1");
        Self { hasher }
    }

    pub fn add_provider_identity(mut self, bytes: &[u8]) -> Self {
        self.add_segment(b"provider_identity", bytes);
        self
    }

    pub fn add_algo_version(mut self, version: u32) -> Self {
        self.add_segment(b"algo_version", &version.to_le_bytes());
        self
    }

    pub fn add_journal_head(mut self, head: u64) -> Self {
        self.add_segment(b"journal_head", &head.to_le_bytes());
        self
    }

    pub fn add_params<T: bytemuck::NoUninit>(mut self, params: &T) -> Self {
        self.add_segment(
            std::any::type_name::<T>().as_bytes(),
            bytemuck::bytes_of(params),
        );
        self
    }

    pub fn add_param_bytes(mut self, label: &'static str, bytes: &[u8]) -> Self {
        self.add_segment(label.as_bytes(), bytes);
        self
    }

    pub fn finish(self) -> Fingerprint {
        Fingerprint(*self.hasher.finalize().as_bytes())
    }

    fn add_segment(&mut self, label: &[u8], bytes: &[u8]) {
        self.hasher.update(&(label.len() as u32).to_le_bytes());
        self.hasher.update(label);
        self.hasher.update(&(bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
    }
}

impl Default for FingerprintBuilder {
    fn default() -> Self {
        Self::new()
    }
}
