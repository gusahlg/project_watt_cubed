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
mod pair;
pub(crate) mod persist;
pub(crate) mod protocol;

pub use pair::ChannelPair;

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

    /// Idle connections are reaped after this long. A keep-alive well under it
    /// holds a quiet-but-live link open. The client's own silence watch gives up
    /// on the same interval.
    const IDLE_TIMEOUT: Duration = Duration::from_secs(12);
    const KEEP_ALIVE: Duration = Duration::from_secs(3);

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
        t.max_idle_timeout(Some(IDLE_TIMEOUT.try_into().expect("12s fits a QUIC VarInt")));
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

    /// UDP socket bound to `::` with `IPV6_V6ONLY` off, so one port takes IPv4 and IPv6.
    /// Cleared before `bind`: `std` follows the process default, and a host may set that to v6-only.
    pub(crate) fn bind_dual_stack(port: u16) -> io::Result<std::net::UdpSocket> {
        use socket2::{Domain, Protocol, Socket, Type};
        let socket = Socket::new(Domain::IPV6, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_only_v6(false)?;
        let addr = std::net::SocketAddr::from((std::net::Ipv6Addr::UNSPECIFIED, port));
        socket.bind(&addr.into())?;
        socket.set_nonblocking(true)?;
        Ok(socket.into())
    }
}

/// Wire revision. Client and server must match exactly at join. Bump on any
/// incompatible frame change; history is `documentation/notes/protocol-history.md`.
pub(crate) const PROTOCOL_VERSION: u32 = 16;

pub const DEFAULT_PORT: u16 = 5555;

/// Hard cap on a single wire frame (bytes). A frame claiming more is rejected
/// before a byte of its body is read, so a hostile peer can't force a huge alloc.
pub(crate) const MAX_FRAME: usize = 64 * 1024;

pub const MAX_NAME: usize = 24;
pub(crate) const MAX_CHAT: usize = 256;
/// Longest block spec on the wire: `c:` plus the hex of a full configuration
/// (`1 + CAPACITY * D` bytes). A shorter cap rejects blocks the game can place.
pub(crate) const MAX_SPEC: usize = 2 + 2 * (1 + material::CAPACITY * material::D);

/// Largest accepted mod-channel payload (bytes). One 20 ms opus frame at up to
/// ~64 kbps fits, with margin, and the same cap bounds every channel. `net`
/// owns its own copy rather than depending on `audio`. The codec rejects a
/// longer frame before the bytes are trusted.
pub(crate) const MAX_MOD_BYTES: usize = 400;

/// What a join must share: generator version, gravity, material law, palette.
/// Worldgen kind and terrain knobs are not in here — [`Welcome`](protocol::ServerMessage::Welcome)
/// carries them, and the joiner adopts the server's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContentId {
    pub worldgen: u32,
    pub gravity: u64,
    pub law: u64,
    pub palette: u64,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_feed(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= b as u64;
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// Code and content identity of `registry`. Interning a block does not change it: the
/// palette is a function of the law, not of which configurations have been seen.
pub(crate) fn content_id(registry: &crate::block::BlockRegistry) -> ContentId {
    let mut gravity = FNV_OFFSET;
    for word in crate::gravity::law_digest() {
        fnv_feed(&mut gravity, &word.to_le_bytes());
    }
    let mut palette = FNV_OFFSET;
    for entry in crate::world::terrain::palette::of(registry.law()) {
        fnv_feed(&mut palette, entry.config.encode().as_bytes());
    }
    ContentId {
        worldgen: u32::from(crate::world::terrain::WORLDGEN_VERSION),
        gravity,
        law: registry.law().fingerprint(),
        palette,
    }
}

/// `None` when `client` may share a world with `server`. Otherwise a reason that names the
/// first part that differs and contains `content`, so a refusal says what diverged.
pub(crate) fn content_mismatch(server: ContentId, client: ContentId) -> Option<String> {
    if server == client {
        return None;
    }
    let detail = if server.worldgen != client.worldgen {
        format!(
            "generator version differs: server v{}, client v{}",
            server.worldgen, client.worldgen
        )
    } else if server.gravity != client.gravity {
        "gravity law differs".to_string()
    } else if server.law != client.law {
        "material law differs".to_string()
    } else {
        "palette differs".to_string()
    };
    Some(format!("world content mismatch: {detail}"))
}

#[cfg(test)]
pub(crate) fn content_fingerprint() -> u64 {
    content_fingerprint_kind(crate::world::generation::WorldgenKind::Diffusion)
}

#[cfg(test)]
pub(crate) fn content_fingerprint_kind(kind: crate::world::generation::WorldgenKind) -> u64 {
    content_fingerprint_kind_cfg(kind, crate::world::terrain::TerrainCfg::default())
}

/// A stable 64-bit digest of everything that determines what a seed GENERATES and how a body
/// falls, including the worldgen kind and its knobs. The handshake no longer sends this —
/// kind and knobs arrive in Welcome — the tests keep it so the mix stays characterised.
#[cfg(test)]
pub(crate) fn content_fingerprint_kind_cfg(
    kind: crate::world::generation::WorldgenKind,
    cfg: crate::world::terrain::TerrainCfg,
) -> u64 {
    fingerprint_kind_cfg(&crate::block::BlockRegistry::with_builtins(), kind, cfg)
}

/// The fingerprint of the gravity law, a registry's material law, and the generator's palette.
#[cfg(test)]
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
    eat(&crate::world::terrain::WORLDGEN_VERSION.to_le_bytes());
    for word in crate::gravity::law_digest() {
        eat(&word.to_le_bytes());
    }
    eat(&registry.law().fingerprint().to_le_bytes());
    for entry in crate::world::terrain::palette::of(registry.law()) {
        eat(entry.config.encode().as_bytes());
    }
    hash
}

/// Same as [`fingerprint_of`], plus the worldgen kind and its knobs.
#[cfg(test)]
pub(crate) fn fingerprint_kind_cfg(
    registry: &crate::block::BlockRegistry,
    kind: crate::world::generation::WorldgenKind,
    cfg: crate::world::terrain::TerrainCfg,
) -> u64 {
    let mut hash = fingerprint_of(registry);
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    eat(kind.id().as_bytes());
    if kind == crate::world::generation::WorldgenKind::Diffusion {
        for v in cfg.clamp().to_wire() {
            eat(&v.to_le_bytes());
        }
    }
    hash
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;
    use crate::world::generation::WorldgenKind;
    use crate::world::terrain::TerrainCfg;

    #[test]
    fn flat_fingerprint_ignores_knobs_and_differs_from_diffusion() {
        let flat = content_fingerprint_kind(WorldgenKind::Flat);
        let cfg = TerrainCfg { relief: 150, ..Default::default() };
        assert_eq!(flat, content_fingerprint_kind_cfg(WorldgenKind::Flat, cfg));
        assert_ne!(flat, content_fingerprint());
    }

    #[test]
    fn diffusion_fingerprint_mixes_knobs() {
        let diff = content_fingerprint_kind(WorldgenKind::Diffusion);
        for cfg in [
            TerrainCfg { relief: 150, ..Default::default() },
            TerrainCfg { caves: 50, ..Default::default() },
            TerrainCfg { mines: 0, ..Default::default() },
            TerrainCfg { space: 200, ..Default::default() },
            TerrainCfg { variety: 50, ..Default::default() },
            TerrainCfg { features: 0, ..Default::default() },
            TerrainCfg { structures: 175, ..Default::default() },
            TerrainCfg { deep: 25, ..Default::default() },
        ] {
            assert_ne!(diff, content_fingerprint_kind_cfg(WorldgenKind::Diffusion, cfg));
        }
    }

    #[test]
    fn fingerprint_survives_a_reaction_the_client_never_saw() {
        use crate::block::BlockRegistry;
        use material::{Configuration, Element};

        let mut server = BlockRegistry::with_builtins();
        crate::world::terrain::Materials::intern(&mut server);
        let before = fingerprint_of(&server);
        let novel = Configuration::new(vec![Element::new([3, 9, 27, 81]), Element::new([4, 16, 64, 1])]).unwrap();
        let sid = server.intern(&novel).unwrap();
        let spec = server.spec(sid);
        assert_eq!(fingerprint_of(&server), before, "interning does not change the handshake");

        let mut client = BlockRegistry::with_builtins();
        crate::world::terrain::Materials::intern(&mut client);
        assert!(client.lookup(&novel).is_none(), "client has not seen the product");
        let cid = client.parse_spec(&spec).unwrap();
        assert_eq!(client.configuration(cid), server.configuration(sid));
        assert_eq!(fingerprint_of(&client), before);
    }

    #[test]
    fn content_id_ignores_interns_and_names_a_mismatch() {
        use crate::block::BlockRegistry;
        use material::{Configuration, Element};

        let bare = BlockRegistry::with_builtins();
        let id = content_id(&bare);
        assert_eq!(id.worldgen, u32::from(crate::world::terrain::WORLDGEN_VERSION));
        let mut interned = BlockRegistry::with_builtins();
        crate::world::terrain::Materials::intern(&mut interned);
        assert_eq!(content_id(&interned), id, "generation interns do not change the join id");
        let novel = Configuration::new(vec![Element::new([3, 9, 27, 81])]).unwrap();
        interned.intern(&novel).unwrap();
        assert_eq!(content_id(&interned), id, "a novel block does not change the join id");

        assert!(content_mismatch(id, id).is_none());
        let drifted = ContentId { worldgen: id.worldgen + 1, ..id };
        let why = content_mismatch(id, drifted).unwrap();
        assert!(why.contains("content"), "{why}");
        assert!(why.contains(&format!("server v{}", id.worldgen)), "{why}");
        assert!(why.contains(&format!("client v{}", drifted.worldgen)), "{why}");
        assert!(content_mismatch(id, ContentId { gravity: id.gravity ^ 1, ..id }).unwrap().contains("gravity"));
        assert!(content_mismatch(id, ContentId { law: id.law ^ 1, ..id }).unwrap().contains("law"));
        assert!(content_mismatch(id, ContentId { palette: id.palette ^ 1, ..id }).unwrap().contains("palette"));
    }

    /// Peers compare this id at join: a change refuses every older client.
    #[test]
    fn content_id_is_pinned() {
        let id = content_id(&crate::block::BlockRegistry::with_builtins());
        assert_eq!(
            id,
            ContentId { worldgen: 10, gravity: 0xbda9_b2eb_c25d_c4d0, law: 0x04ce_0caa_d622_c6eb, palette: 0x95c9_349d_7e12_74b8 }
        );
    }
}

/// Chat channels. Local is proximity-limited; global reaches everyone.
pub(crate) mod chat {
    pub const LOCAL: u8 = 0;
    pub const GLOBAL: u8 = 1;
    /// How far local (proximity) chat carries.
    pub const RADIUS: f64 = 48.0 * crate::math::PER_METER;
}
