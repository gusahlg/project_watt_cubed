//! Multiplayer: an authoritative headless [`server`] and the thin [`client`]
//! [`Connection`](client::Connection) the game talks to, speaking the compact
//! binary [`protocol`].
//!
//! The whole design leans on one fact from [`world`](crate::world): terrain is
//! *procedural*, regenerated identically from a seed, so the network never carries
//! voxel data. A join transfers only the seed and the sparse overlay of player
//! *edits* — the same portable, name-keyed form [`save`](crate::save) already uses.
//! Everything else on the wire is small and frequent (positions, chat), which is
//! what the framing and interest management here are tuned for.
//!
//! **Trust:** the server is authoritative and never trusts a client. Frames are
//! length-capped, joins are password-gated, and every edit and move is validated
//! and rate-limited server-side ([`server`]). Communities add extra rules through
//! the [`hooks`] seam (`ServerMod`) without changing the wire.
pub(crate) mod client;
pub(crate) mod hooks;
pub(crate) mod protocol;

// Deny a bare `.unwrap()` on production paths; server.rs's `lock_recover()`
// is the one sanctioned recovery point. The tests module carries its own
// `#![allow]` since a setup unwrap there is a loud, desired test failure.
#[deny(clippy::unwrap_used)]
pub mod server;

/// Shared QUIC transport setup for [`client`] and [`server`]. One place mints the
/// server's self-signed cert, the accept-any-cert client verifier, and the common
/// transport tuning, and installs the process-wide rustls crypto provider exactly
/// once. Encryption is TLS 1.3; server identity is NOT authenticated (self-signed
/// cert, client accepts any) — the real gate stays the app-level password in
/// [`Hello`](protocol::ClientMessage::Hello), carried over TLS 1.3.
pub(crate) mod quic {
    use std::io;
    use std::sync::{Arc, Once};
    use std::time::Duration;

    use quinn::{ClientConfig, ServerConfig, TransportConfig, VarInt};
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::crypto::CryptoProvider;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
    use rustls::{DigitallySignedStruct, SignatureScheme};

    /// Idle connections are reaped after this long (mirrors the server's
    /// `IDLE_TIMEOUT`); a keep-alive well under it holds a quiet-but-live link open.
    const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
    const KEEP_ALIVE: Duration = Duration::from_secs(10);

    /// rustls 0.23 requires a provider installed before any TLS config is built;
    /// both endpoints call this, so `Once` makes concurrent callers safe.
    pub(crate) fn install_crypto() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let _ = rustls::crypto::ring::default_provider().install_default();
        });
    }

    /// Caps concurrent bidi streams at 1: all app traffic multiplexes onto one stream.
    fn transport() -> Arc<TransportConfig> {
        let mut t = TransportConfig::default();
        t.max_idle_timeout(Some(IDLE_TIMEOUT.try_into().expect("30s fits a QUIC VarInt")));
        t.keep_alive_interval(Some(KEEP_ALIVE));
        t.max_concurrent_bidi_streams(VarInt::from_u32(1));
        Arc::new(t)
    }

    /// Fresh self-signed cert per process — no files, no CA.
    pub(crate) fn server_config() -> io::Result<ServerConfig> {
        install_crypto();
        let rcgen::CertifiedKey { cert, signing_key } =
            rcgen::generate_simple_self_signed(vec!["watt".to_string()]).map_err(io::Error::other)?;
        let key = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(signing_key.serialize_der()));
        let mut cfg =
            ServerConfig::with_single_cert(vec![cert.der().clone()], key).map_err(io::Error::other)?;
        cfg.transport_config(transport());
        Ok(cfg)
    }

    /// The client endpoint config: encrypts, but accepts ANY server cert (see the
    /// module note) — confidentiality without server authentication.
    pub(crate) fn client_config() -> ClientConfig {
        install_crypto();
        let rustls_cfg = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(SkipServerVerification::new())
            .with_no_client_auth();
        let quic = quinn::crypto::rustls::QuicClientConfig::try_from(rustls_cfg)
            .expect("a default rustls ClientConfig is TLS 1.3 capable");
        let mut cfg = ClientConfig::new(Arc::new(quic));
        cfg.transport_config(transport());
        cfg
    }

    /// A verifier that accepts every server certificate. The password in `Hello`
    /// is the real authentication; TLS here buys confidentiality, not identity.
    #[derive(Debug)]
    struct SkipServerVerification(Arc<CryptoProvider>);

    impl SkipServerVerification {
        fn new() -> Arc<Self> {
            Arc::new(Self(Arc::new(rustls::crypto::ring::default_provider())))
        }
    }

    impl ServerCertVerifier for SkipServerVerification {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }

        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            self.0.signature_verification_algorithms.supported_schemes()
        }
    }
}

/// Wire revision. Client and server must match exactly at join. Bump on any
/// incompatible frame change; history is `documentation/notes/protocol-history.md`.
pub(crate) const PROTOCOL_VERSION: u32 = 9;

pub const DEFAULT_PORT: u16 = 5555;

/// Hard cap on a single wire frame (bytes). A frame claiming more is rejected
/// before a byte of its body is read, so a hostile peer can't force a huge alloc.
pub(crate) const MAX_FRAME: usize = 64 * 1024;

pub(crate) const MAX_NAME: usize = 24;
pub(crate) const MAX_CHAT: usize = 256;
pub(crate) const MAX_SPEC: usize = 256;

/// Largest accepted voice payload (bytes): one 20 ms opus frame at up to
/// ~64 kbps with margin. `net` owns its own copy of the cap rather than
/// depending on the `audio` crate: the two modules fan out in parallel and net
/// must compile without it. The codec rejects any inbound voice frame past this
/// cap, and callers guard outbound.
pub(crate) const MAX_VOICE_PAYLOAD: usize = 400;

/// A stable 64-bit digest of everything that determines what a seed GENERATES:
/// the worldgen version, the element table, and the full compiled placement
/// palette (canonical portable spec per block). Seed-only multiplayer never
/// ships voxels, so two builds whose generation differs in ANY of these would
/// silently build different worlds from one seed — the handshake compares
/// fingerprints and rejects the join instead. Protocol changes are versioned
/// separately by [`PROTOCOL_VERSION`].
///
/// Uses FNV hash (not the std hasher) so the value is identical across
/// platforms, architectures, and Rust releases. The placement compile is
/// seed-invariant, so one fingerprint speaks for every world a build can generate.
pub(crate) fn content_fingerprint() -> u64 {
    content_fingerprint_kind(crate::world::generation::WorldgenKind::Classic)
}

pub(crate) fn content_fingerprint_kind(kind: crate::world::generation::WorldgenKind) -> u64 {
    content_fingerprint_kind_cfg(kind, crate::world::diffusion::DiffusionCfg::default())
}

pub(crate) fn content_fingerprint_kind_cfg(
    kind: crate::world::generation::WorldgenKind,
    cfg: crate::world::diffusion::DiffusionCfg,
) -> u64 {
    let mut registry = crate::block::BlockRegistry::with_builtins();
    let _ = crate::world::placement::builtin().compile(&mut registry);
    fingerprint_kind_cfg(&registry, kind, cfg)
}

/// The fingerprint of an already-compiled registry — the server hashes the
/// one it built for spawn heights instead of compiling twice.
pub(crate) fn fingerprint_of(registry: &crate::block::BlockRegistry) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    eat(&crate::world::placement::WORLDGEN_VERSION.to_le_bytes());
    let elements = registry.elements();
    eat(&(elements.len() as u32).to_le_bytes());
    for i in 0..elements.len() {
        let element = elements.get(crate::block::ElementId(i as u16));
        eat(element.name.as_bytes());
        eat(&[0]); // name terminator so "ab"+"c" != "a"+"bc"
        eat(&crate::block::element::Core::from(element.core).0);
        eat(&[element.color.r, element.color.g, element.color.b, element.color.a]);
        eat(&(element.specials.len() as u32).to_le_bytes());
        for special in &element.specials {
            eat(&[special.kind() as u8, special.strength()]);
        }
    }
    eat(&(registry.block_count() as u32).to_le_bytes());
    for i in 0..registry.block_count() {
        let block = registry.block(crate::block::BlockId(i as u16));
        eat(crate::save::block_spec(registry, crate::block::BlockId(i as u16)).as_bytes());
        eat(&[0]);
        eat(&crate::block::element::Core::from(block.core).0);
        eat(&(block.specials.len() as u32).to_le_bytes());
        for &(kind, strength) in &block.specials {
            eat(&[kind as u8, strength]);
        }
        eat(&(block.reactions.len() as u32).to_le_bytes());
        for reaction in &block.reactions {
            eat(reaction.name.as_bytes());
            eat(&[0, reaction.strength]);
            match reaction.effect {
                crate::block::reaction::ReactionEffect::CoreBonus(core) => {
                    eat(&[0]);
                    eat(&crate::block::element::Core::from(core).0);
                }
                crate::block::reaction::ReactionEffect::Emergent(kind, strength) => {
                    eat(&[1, kind as u8, strength]);
                }
            }
        }
    }
    hash
}

/// Same as [`fingerprint_of`], plus worldgen kind and diffusion knobs so two
/// worlds that would generate different terrain cannot silently desync.
pub(crate) fn fingerprint_kind_cfg(
    registry: &crate::block::BlockRegistry,
    kind: crate::world::generation::WorldgenKind,
    cfg: crate::world::diffusion::DiffusionCfg,
) -> u64 {
    let mut hash = fingerprint_of(registry);
    if kind != crate::world::generation::WorldgenKind::Classic {
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                hash ^= b as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        };
        eat(kind.id().as_bytes());
        eat(&cfg.tile.to_le_bytes());
        eat(&cfg.stride.to_le_bytes());
        eat(&cfg.phases.to_le_bytes());
        eat(&cfg.relief.to_bits().to_le_bytes());
    }
    hash
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;
    use crate::world::diffusion::DiffusionCfg;
    use crate::world::generation::WorldgenKind;

    #[test]
    fn classic_fingerprint_ignores_diffusion_knobs() {
        let a = content_fingerprint();
        let b = content_fingerprint_kind(WorldgenKind::Classic);
        let cfg = DiffusionCfg { tile: 64, ..Default::default() };
        let c = content_fingerprint_kind_cfg(WorldgenKind::Classic, cfg);
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn diffusion_fingerprint_differs_and_mixes_knobs() {
        let classic = content_fingerprint();
        let diff = content_fingerprint_kind(WorldgenKind::Diffusion);
        assert_ne!(classic, diff);
        let cfg = DiffusionCfg { phases: 8, ..Default::default() };
        assert_ne!(diff, content_fingerprint_kind_cfg(WorldgenKind::Diffusion, cfg));
    }
}

/// Chat channels. Local is proximity-limited; global reaches everyone.
pub(crate) mod chat {
    pub const LOCAL: u8 = 0;
    pub const GLOBAL: u8 = 1;
    /// How far local (proximity) chat carries.
    pub const RADIUS: f64 = 48.0 * crate::math::PER_METER;
}
