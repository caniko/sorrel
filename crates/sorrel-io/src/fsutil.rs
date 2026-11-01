//! Windows-aware file helpers shared by every writer: one place for the
//! "something holds this file open" hint and for hiding `\\?\` prefixes.

use anyhow::Result;
use std::path::Path;

pub use dunce::simplified;

/// `path` for a human or a JSON document: a verbatim `\\?\C:\x` prefix
/// (what `canonicalize` returns on Windows) is dropped when the plain path
/// means the same (dunce keeps it when stripping would change the path).
pub fn display_path(path: &Path) -> std::path::Display<'_> {
    dunce::simplified(path).display()
}

/// Windows error codes for a file another process (or a live mapping) holds:
/// 5 access denied, 32 sharing violation, 1224 user-mapped section.
fn busy_code(code: Option<i32>) -> bool {
    matches!(code, Some(5 | 32 | 1224))
}

/// `e` says the file is in use by another program. Never true off Windows,
/// where a rename over an open or mapped file succeeds.
pub fn is_busy(e: &std::io::Error) -> bool {
    cfg!(windows) && busy_code(e.raw_os_error())
}

/// Rename `from` over `to`. On Windows a busy `to` (phy, Python, an Explorer
/// preview) gets a hint to close what holds it; elsewhere this is a rename.
pub fn replace_file(from: &Path, to: &Path) -> Result<()> {
    std::fs::rename(from, to).map_err(|e| {
        let busy = is_busy(&e);
        let err =
            anyhow::Error::new(e).context(format!("rename {} -> {}", from.display(), to.display()));
        if busy {
            err.context(format!(
                "close programs holding {} (phy, Python, Explorer preview) and try again",
                to.display()
            ))
        } else {
            err
        }
    })
}

/// [`replace_file`] for callers that return `io::Result`; the hint stays in
/// the message.
pub fn replace_file_io(from: &Path, to: &Path) -> std::io::Result<()> {
    replace_file(from, to).map_err(|e| std::io::Error::other(format!("{e:#}")))
}

/// Drop the inherited ACEs of `path` and grant only the current user, like
/// Unix mode 0600. Shells out to `icacls`, which ships with Windows.
#[cfg(windows)]
pub fn restrict_to_current_user(path: &Path) -> Result<()> {
    use anyhow::Context;
    let user = std::env::var("USERNAME").context("USERNAME is not set")?;
    let out = std::process::Command::new("icacls")
        .arg(path)
        .args(["/inheritance:r", "/grant:r"])
        .arg(format!("{user}:F"))
        .output()
        .context("run icacls")?;
    anyhow::ensure!(
        out.status.success(),
        "icacls {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stdout).trim()
    );
    Ok(())
}

/// Create `path` (which must not exist) readable by the current user only
/// (mode 0600; on Windows an ACL, set before the bytes are written) and write
/// `bytes`. A Windows ACL failure is a warning, not an error.
pub fn write_private_new(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    let mut f = opts.open(path)?;
    #[cfg(windows)]
    if let Err(e) = restrict_to_current_user(path) {
        log::warn!(
            "{} is not ACL-restricted ({e:#}); anyone who can read it can use its contents",
            path.display()
        );
    }
    f.write_all(bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file holds one explicit ACE (the current user) and inherits none.
    #[test]
    fn private_file_has_no_inherited_aces() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("relay.key");
        write_private_new(&p, b"secret").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"secret");
        assert!(write_private_new(&p, b"x").is_err(), "must not overwrite");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&p).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        #[cfg(windows)]
        {
            let out = std::process::Command::new("icacls")
                .arg(&p)
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&out.stdout);
            // `path  DOMAIN\user:(F)`, then one line per further ACE; `(I)` marks inherited.
            let aces: Vec<&str> = text.lines().filter(|l| l.contains(":(")).collect();
            assert!(!text.contains("(I)"), "inherited ACEs remain:\n{text}");
            assert_eq!(aces.len(), 1, "expected one ACE:\n{text}");
            let user = std::env::var("USERNAME").unwrap().to_lowercase();
            assert!(
                aces[0].to_lowercase().contains(&user),
                "not the current user:\n{text}"
            );
            assert!(aces[0].contains("(F)"), "{text}");
        }
    }

    #[test]
    fn busy_codes_and_plain_rename() {
        assert!([5, 32, 1224].iter().all(|&c| busy_code(Some(c))));
        assert!(!busy_code(Some(2)) && !busy_code(None));
        let d = std::env::temp_dir().join(format!("sorrel-fsutil-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let (a, b) = (d.join("a"), d.join("b"));
        std::fs::write(&a, "1").unwrap();
        std::fs::write(&b, "0").unwrap();
        replace_file(&a, &b).unwrap();
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "1");
        assert!(replace_file(&a, &b).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn display_path_strips_only_drive_verbatim() {
        // dunce is a no-op off Windows, so these hold everywhere.
        let p = Path::new("/a/b");
        assert_eq!(display_path(p).to_string(), "/a/b");
        #[cfg(windows)]
        {
            assert_eq!(display_path(Path::new(r"\\?\C:\x")).to_string(), r"C:\x");
            // Verbatim UNC stays as it is.
            assert_eq!(
                display_path(Path::new(r"\\?\UNC\h\s")).to_string(),
                r"\\?\UNC\h\s"
            );
        }
    }
}
