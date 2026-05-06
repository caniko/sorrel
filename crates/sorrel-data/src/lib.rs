//! sorrel-data: the monomorphised core. `Session<P>` is generic over a
//! `DataProvider`, so the entire state machine specialises per backend.

pub mod command;
pub mod journal;
pub mod session;

pub use command::{CurationCommand, PhyLabelOp};
pub use journal::SqliteJournal;
pub use session::Session;
