# Workspace Layout

Sorrel is split into focused Rust crates:

| Crate | Role |
|-------|------|
| `sorrel-io` | Concrete data backends and the `DataProvider` trait bound. |
| `sorrel-compute` | Generic kernels such as LTTB and histograms. |
| `sorrel-data` | Generic `Session<P>`, `CurationCommand`, and SQLite journal. |
| `sorrel-render` | Concrete vertex types and monomorphised buffer builders. |
| `sorrel-ui` | egui widgets generic over `P: DataProvider`. |
| `sorrel` | Binary crate that detects the backend and instantiates `SorrelApp<P>`. |

The workspace keeps backend IO, session state, compute kernels, render buffers, and UI concerns in separate crates while preserving static dispatch through the core data path.
