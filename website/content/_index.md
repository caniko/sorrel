+++
title = "Sorrel"

[extra]
tagline = "Native spike-sorting curation"
subtitle = "A high-performance GUI for manual curation of spike-sorted electrophysiology data, built around monomorphised Rust data paths and durable SQLite journaling."
install = "cargo build --release -p sorrel"

[[extra.features]]
title = "Kilosort / phy2 V1"
description = "Loads Kilosort arrays, phy2 labels, and raw recording traces for cluster-oriented curation."

[[extra.features]]
title = "Static Dispatch Core"
description = "The session, UI, render, and compute paths are generic over DataProvider instead of runtime trait objects."

[[extra.features]]
title = "Memory-Mapped Inputs"
description = "Spike arrays and raw trace data are memory-mapped so large recordings stay practical to inspect."

[[extra.features]]
title = "Durable Curation"
description = "Label operations are written to SQLite before in-memory state is updated."

[[extra.features]]
title = "Native GUI"
description = "egui and wgpu provide a responsive desktop interface for cluster navigation and trace inspection."

[[extra.features]]
title = "Focused Workspace"
description = "IO, data state, compute kernels, rendering buffers, UI, and the binary entry point live in separate crates."
+++
