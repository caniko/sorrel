//! The single GUI instance per user: `sorrel open` hands a dataset to the
//! running window (it becomes a new tab) instead of starting a second one.
//! `<user_dir>/gui.json` names the loopback port and token, the same
//! scheme as the per-dataset MCP listener in [`crate::serve`].

use crate::serve::{
    connect_authed, handshake, random_token, read_port_token, shutdown_listener, user_dir,
    write_private_json,
};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

/// How long either side waits for the UI thread's reply.
const REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// One request line to the GUI instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstanceRequest {
    /// Open a dataset as a tab. `args` are `sorrel open`'s arguments
    /// (DATA_DIR and flags), relative to `cwd`.
    Open { cwd: PathBuf, args: Vec<String> },
    /// Raise the window.
    Focus,
    /// Reply with `{"pid", "tabs": [{"dataset", "dirty", "active"}]}`.
    List,
}

/// A request plus the channel the UI thread answers on.
pub type Pending = (InstanceRequest, Sender<Value>);

/// Keeps the instance listener alive; dropping it removes `gui.json` (only
/// if this process wrote it) and stops accepting.
pub struct InstanceHandle {
    port: u16,
    path: PathBuf,
    nonce: String,
    stop: Arc<AtomicBool>,
}

impl Drop for InstanceHandle {
    fn drop(&mut self) {
        shutdown_listener(&self.stop, &self.path, &self.nonce, self.port);
    }
}

/// Register this process as the GUI instance. `wake` is called after each
/// request is queued so the UI drains the receiver promptly.
pub fn serve_instance(
    wake: Arc<dyn Fn() + Send + Sync>,
) -> std::io::Result<(InstanceHandle, Receiver<Pending>)> {
    serve_instance_in(&user_dir()?, wake)
}

fn serve_instance_in(
    dir: &Path,
    wake: Arc<dyn Fn() + Send + Sync>,
) -> std::io::Result<(InstanceHandle, Receiver<Pending>)> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let (token, nonce) = (random_token(), random_token());
    let path = dir.join("gui.json");
    write_private_json(
        &path,
        &json!({"port": port, "token": token, "pid": std::process::id(), "nonce": nonce}),
    )?;
    let stop = Arc::new(AtomicBool::new(false));
    let handle = InstanceHandle {
        port,
        path,
        nonce,
        stop: stop.clone(),
    };
    let (queue, rx) = channel::<Pending>();
    // ponytail: one thread per connection, uncapped; every connection must
    // present the token first and gui.json is 0600 in a 0700 directory. Add
    // serve.rs's MAX_HANDSHAKES cap if this ever listens beyond one user.
    std::thread::Builder::new()
        .name("sorrel-instance".into())
        .spawn(move || {
            for stream in listener.incoming() {
                if stop.load(SeqCst) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                let (queue, wake, token) = (queue.clone(), wake.clone(), token.clone());
                std::thread::spawn(move || {
                    if let Err(e) = serve_one(stream, &token, &queue, &*wake) {
                        log::debug!("instance client ended: {e}");
                    }
                });
            }
        })?;
    Ok((handle, rx))
}

/// Token handshake, one request line in, one reply line out.
fn serve_one(
    stream: std::net::TcpStream,
    token: &str,
    queue: &Sender<Pending>,
    wake: &(dyn Fn() + Send + Sync),
) -> std::io::Result<()> {
    let Some((mut reader, mut writer)) = handshake(stream, token)? else {
        return Ok(());
    };
    writer.set_read_timeout(Some(REPLY_TIMEOUT))?;
    let mut line = String::new();
    reader.by_ref().take(1 << 20).read_line(&mut line)?;
    let reply = match serde_json::from_str::<InstanceRequest>(&line) {
        Err(e) => json!({"error": format!("bad request: {e}")}),
        Ok(req) => {
            let (tx, rx) = channel();
            if queue.send((req, tx)).is_err() {
                json!({"error": "instance is shutting down"})
            } else {
                wake();
                // Timeout or a dropped sender: close without a reply.
                let Ok(v) = rx.recv_timeout(REPLY_TIMEOUT) else {
                    return Ok(());
                };
                v
            }
        }
    };
    writeln!(writer, "{reply}")?;
    writer.flush()
}

/// Send `req` to the running GUI instance. `Ok(None)` means there is none
/// (no `gui.json`, or the connection is refused: a stale file).
pub fn send_instance(req: &InstanceRequest) -> Result<Option<Value>> {
    send_instance_in(&user_dir()?, req)
}

fn send_instance_in(dir: &Path, req: &InstanceRequest) -> Result<Option<Value>> {
    let Some((port, token)) = read_port_token(&dir.join("gui.json")) else {
        return Ok(None);
    };
    let Some((mut stream, mut reader)) = connect_authed(port, &token)? else {
        return Ok(None);
    };
    stream.set_read_timeout(Some(REPLY_TIMEOUT))?;
    writeln!(stream, "{}", serde_json::to_string(req)?)?;
    stream.flush()?;
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(n) if n > 0 => Ok(Some(serde_json::from_str(&line)?)),
        Ok(_) => bail!("sorrel instance on port {port} closed without a reply"),
        Err(e) => bail!("sorrel instance on port {port} did not reply: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_and_list_round_trip_then_none_after_drop() {
        let dir = tempfile::tempdir().unwrap();
        let gui = dir.path().join("gui.json");
        let (handle, rx) = serve_instance_in(dir.path(), Arc::new(|| {})).unwrap();
        let ui = std::thread::spawn(move || {
            while let Ok((req, tx)) = rx.recv() {
                let _ = tx.send(match req {
                    InstanceRequest::List => json!({"tabs": []}),
                    _ => json!({"ok": true}),
                });
            }
        });
        let open = InstanceRequest::Open {
            cwd: "/x".into(),
            args: vec!["d".into()],
        };
        let send = |r| send_instance_in(dir.path(), r).unwrap();
        assert_eq!(send(&InstanceRequest::List), Some(json!({"tabs": []})));
        assert_eq!(send(&open), Some(json!({"ok": true})));

        // A wrong token means a live host: an error, never "no instance".
        let (port, _) = read_port_token(&gui).unwrap();
        let body = |token| json!({"port": port, "token": token, "pid": 0, "nonce": "x"});
        write_private_json(&gui, &body("wrong")).unwrap();
        assert!(send_instance_in(dir.path(), &open).is_err());

        // Not our file any more (pid mismatch): drop leaves it alone.
        drop(handle);
        assert!(gui.exists());
        ui.join().unwrap();
        assert_eq!(send(&open), None, "stale file: connection refused");

        // Our own file is removed on drop.
        let (handle, _rx) = serve_instance_in(dir.path(), Arc::new(|| {})).unwrap();
        drop(handle);
        assert!(!gui.exists());
        assert_eq!(send(&open), None);
    }
}
