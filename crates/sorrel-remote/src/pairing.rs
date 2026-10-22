//! Pairing codes: six BIP-39 words that both machines turn into libp2p
//! identities.
//!
//! `seed = argon2id(code)` is slow on purpose: the host PeerId is public (it
//! is what the guest dials), so an attacker who sees it could try codes
//! offline. 66 bits behind argon2id (64 MiB, t=3) is out of reach. From the
//! seed both sides derive the host keypair and the guest keypair, so the
//! Noise handshake itself proves that each peer knows the code; the host
//! additionally allow-lists only the derived guest PeerId.

use anyhow::{bail, Context, Result};
use libp2p::identity::Keypair;
use libp2p::PeerId;

const WORDLIST: &str = include_str!("../wordlist.txt");
const WORDS: usize = 6;
const SALT: &[u8] = b"sorrel-remote-v1";
const GUEST_CONTEXT: &str = "sorrel-remote v1 guest";

fn words() -> Vec<&'static str> {
    WORDLIST
        .lines()
        .map(str::trim)
        .filter(|w| !w.is_empty())
        .collect()
}

/// A normalised pairing code: six lowercase words separated by single spaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairingCode(String);

impl PairingCode {
    /// Six random words from the 2048-word list (66 bits).
    pub fn generate() -> Result<Self> {
        let list = words();
        let mut buf = [0u8; 2 * WORDS];
        getrandom::fill(&mut buf).context("random pairing code")?;
        let code = buf
            .chunks(2)
            .map(|c| list[(u16::from_le_bytes([c[0], c[1]]) as usize) % list.len()])
            .collect::<Vec<_>>()
            .join(" ");
        Ok(Self(code))
    }

    /// Accepts any mix of case, hyphens, and extra whitespace between words;
    /// rejects anything that is not exactly six list words.
    pub fn parse(s: &str) -> Result<Self> {
        let list = words();
        let parts: Vec<String> = s
            .split(|c: char| c.is_whitespace() || c == '-')
            .filter(|p| !p.is_empty())
            .map(|p| p.to_ascii_lowercase())
            .collect();
        if parts.len() != WORDS {
            bail!("pairing code must be {WORDS} words, got {}", parts.len());
        }
        if let Some(bad) = parts.iter().find(|p| !list.contains(&p.as_str())) {
            bail!("'{bad}' is not a pairing-code word");
        }
        Ok(Self(parts.join(" ")))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PairingCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Both identities derived from one code. Each side builds this, uses its own
/// keypair, and checks the other side's PeerId.
#[derive(Clone)]
pub struct Pairing {
    pub host: Keypair,
    pub guest: Keypair,
}

impl Pairing {
    pub fn derive(code: &PairingCode) -> Result<Self> {
        use argon2::{Algorithm, Argon2, Params, Version};
        // 64 MiB, 3 passes, 1 lane: ~0.3 s on a laptop, memory-hard for GPUs.
        let params = Params::new(64 * 1024, 3, 1, Some(32))
            .map_err(|e| anyhow::anyhow!("argon2 params: {e}"))?;
        let mut seed = [0u8; 32];
        Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
            .hash_password_into(code.as_str().as_bytes(), SALT, &mut seed)
            .map_err(|e| anyhow::anyhow!("argon2: {e}"))?;
        let mut guest_seed = blake3::derive_key(GUEST_CONTEXT, &seed);
        let host = Keypair::ed25519_from_bytes(seed).context("host keypair")?;
        let guest = Keypair::ed25519_from_bytes(guest_seed).context("guest keypair")?;
        guest_seed.fill(0);
        Ok(Self { host, guest })
    }

    pub fn host_peer(&self) -> PeerId {
        self.host.public().to_peer_id()
    }

    pub fn guest_peer(&self) -> PeerId {
        self.guest.public().to_peer_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wordlist_is_bip39_sized_and_sorted() {
        let w = words();
        assert_eq!(w.len(), 2048);
        assert!(w.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    fn parse_normalises_and_rejects() {
        let c = PairingCode::parse("  Abandon-ability ABLE about above\tabsent ").unwrap();
        assert_eq!(c.as_str(), "abandon ability able about above absent");
        assert!(PairingCode::parse("abandon ability able about above").is_err());
        assert!(PairingCode::parse("abandon ability able about above zzzz").is_err());
    }

    #[test]
    fn pairing_is_deterministic_and_roles_differ() {
        let code = PairingCode::parse("abandon ability able about above absent").unwrap();
        let a = Pairing::derive(&code).unwrap();
        let b = Pairing::derive(&code).unwrap();
        assert_eq!(a.host_peer(), b.host_peer());
        assert_eq!(a.guest_peer(), b.guest_peer());
        assert_ne!(a.host_peer(), a.guest_peer());
        let other = Pairing::derive(&PairingCode::generate().unwrap()).unwrap();
        assert_ne!(other.host_peer(), a.host_peer());
    }
}
