# Remote Work

Share a Kilosort/phy dataset with a collaborator on another machine. The
host keeps the data; the guest curates in a normal Sorrel window and every
edit lands in the host's journal as it happens.

```text
host:   sorrel share <DIR> [--port N]                 # or: --relay MULTIADDR
guest:  sorrel join "<six words>" --direct MULTIADDR  # or: --relay MULTIADDR
```

## What moves over the wire

- **phy artefacts** (`spike_times.npy`, `spike_clusters.npy`, templates,
  PC features, …) are copied once into the guest's `--into` directory and
  verified with BLAKE3. Re-running `join` with the same directory skips
  files that already match.
- **The raw recording is never copied.** The guest keeps a sparse mirror
  under `.sorrel/remote/recording.dat` and fetches the byte ranges the trace
  and waveform views ask for. Windows render grey until they arrive;
  neighbouring windows are prefetched.
- **Curation commands** stream back to the host as journal records. Each
  carries the `(baseline, record count)` cursor it was applied against; the
  host refuses a record that does not fast-forward its journal, and the
  guest's own journal keeps everything so a later `join` catches up.
- The derived cache (`.sorrel/cache`) is not shared; the guest rebuilds it.
- A guest's **Export phy** writes `<into>_phy` with `params.py` pointing at
  the sparse mirror, so traces only cover what has been fetched. Export on
  the host for a phy directory with the complete recording. `--copy-raw` on
  a guest has no separate guard: it reads the trace through the mirror, and
  the export fails ("came back incomplete") at the first window not fetched
  yet, rather than writing the gaps.

Only one writer exists while a share is active: the host runs headless and
only applies what the guest sends. Open the host directory normally after
the share ends.

Current limits of protocol v3:

- The guest's edits reach the host as journal records, so they appear in
  the host's History with the actor `guest`.
- An undo, redo or save the guest made while disconnected cannot be pushed
  to the host; `join` refuses and asks for a fresh join directory.
- `sorrel_original/` travels with the other files, so the guest replays the
  same history the host has, saves included.

## Pairing

`share` prints six words. Both machines derive their network identities
from those words (argon2id, then ed25519), so the encrypted handshake itself
proves each side knows the code and the host accepts exactly one guest
identity. Treat the words like a password for that session and send them
over a channel you trust.

## Modes

**Direct** (the default): the host listens on QUIC and TCP (`--port`, default 7373) and prints `sorrel join … --direct /ip4/…` lines. On the same LAN the
guest can omit `--direct`; mDNS finds the host. Across the internet you need
a reachable address (port forward, VPN). Nothing sits between the two
machines.

**Relay** (`--relay MULTIADDR`): for guests behind firewalls that only allow
outbound HTTPS. Both sides dial a `sorrel-relay` server over WebSocket/TLS
on port 443; the relay forwards encrypted bytes and sees neither the code
nor the data. The relay address has the form
`/dns4/relay.example.org/tcp/443/wss/p2p/<RelayId>`; the server prints its id
on start.

Limitation: explicit HTTP `CONNECT` proxies (the `HTTPS_PROXY` kind) are not
supported yet; the relay path works where outbound TLS on 443 is allowed.

## Running a relay

```text
sorrel-relay --listen /ip4/0.0.0.0/tcp/4001/ws --identity /var/lib/sorrel-relay/relay.key
```

Terminate TLS in front of it (caddy, nginx, or a Cloudflare-proxied
hostname with WebSockets enabled) and publish
`/dns4/<hostname>/tcp/443/wss/p2p/<RelayId>`. The identity file keeps the
id stable across restarts. Circuit byte and duration limits are disabled;
rate limits are off, so keep the relay for people you know.
