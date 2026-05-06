use anyhow::{Context, Result};
use rusqlite::{params, Connection, OpenFlags};
use std::path::Path;

use crate::command::CurationCommand;

/// Concrete SQLite-backed journal. No trait object: callers hold a
/// `SqliteJournal` directly.
pub struct SqliteJournal {
    conn: Connection,
}

impl SqliteJournal {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )
        .with_context(|| format!("open journal {}", path.display()))?;

        // Synchronous=FULL: the call returns only after fsync, satisfying
        // the "durably recorded before UI updates" requirement.
        conn.pragma_update(None, "journal_mode", &"WAL")?;
        conn.pragma_update(None, "synchronous", &"FULL")?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS operations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_unix_us  INTEGER NOT NULL,
                payload     BLOB    NOT NULL
            );
            "#,
        )?;
        Ok(Self { conn })
    }

    /// Append the command. Blocks until the row is on disk.
    pub fn append(&self, cmd: &CurationCommand) -> Result<i64> {
        let payload = bincode::serialize(cmd)?;
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
        self.conn.execute(
            "INSERT INTO operations (ts_unix_us, payload) VALUES (?1, ?2)",
            params![now_us, payload],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Replay every committed op in insertion order, oldest first.
    pub fn replay(&self) -> Result<Vec<CurationCommand>> {
        let mut stmt = self
            .conn
            .prepare("SELECT payload FROM operations ORDER BY id ASC")?;
        let rows = stmt.query_map([], |r| r.get::<_, Vec<u8>>(0))?;
        let mut out = Vec::new();
        for row in rows {
            let bytes = row?;
            out.push(bincode::deserialize(&bytes)?);
        }
        Ok(out)
    }
}
