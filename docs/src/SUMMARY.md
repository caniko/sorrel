# Summary

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [CLI Reference](./getting-started/cli.md)

# Usage

- [Backends & Inputs](./usage/backends.md)
- [Kilosort / phy2](./usage/kilosort-inputs.md)
- [Integrations](./usage/integrations.md)
- [Views](./usage/views.md)
- [Keyboard & Mouse](./usage/keyboard-controls.md)
- [Saving & QC Export](./usage/saving.md)

# Architecture

- [Workspace Layout](./architecture/workspace-layout.md)
- [Data Flow](./architecture/data-flow.md)
- [Derived Cache](./architecture/cache.md)
- [GPU Compute](./architecture/gpu.md)

# Planning

- [Sorrel improvement research](./planning/sorrel-improvement-research.md)
- [Sorrel scientific correctness research](./planning/sorrel-science-research.md)
- [rkyv derived-data cache](./planning/rkyv-derived-cache/README.md)
  - [01 — Cache infrastructure](./planning/rkyv-derived-cache/01-cache-infrastructure.md)
  - [02 — First consumer](./planning/rkyv-derived-cache/02-first-consumer.md)
  - [03 — Expensive artifacts](./planning/rkyv-derived-cache/03-expensive-artifacts/README.md)
    - [03/01 — PC subspaces](./planning/rkyv-derived-cache/03-expensive-artifacts/sub-01-pc-subspaces.md)
    - [03/02 — Isolation metrics](./planning/rkyv-derived-cache/03-expensive-artifacts/sub-02-isolation-metrics.md)
    - [03/03 — Correlograms](./planning/rkyv-derived-cache/03-expensive-artifacts/sub-03-correlograms.md)
  - [04 — Invalidation, GC, docs](./planning/rkyv-derived-cache/04-invalidation-gc-docs.md)
