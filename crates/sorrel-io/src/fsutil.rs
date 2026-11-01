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

#[cfg(test)]
mod tests {
    use super::*;

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
            assert_eq!(
                display_path(Path::new(r"\\?\UNC\h\s")).to_string(),
                r"\\h\s"
            );
        }
    }
}
