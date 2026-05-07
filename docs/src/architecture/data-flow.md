# Data Flow

Sorrel's core data path is generic over `P: DataProvider`.

The binary detects a supported backend, constructs the concrete provider, opens the SQLite journal, and creates a `Session<P>`. The UI receives that concrete session type, so calls into provider, session, render, and compute paths remain statically dispatched.

For the Kilosort backend:

1. `spike_times.npy` and `spike_clusters.npy` are memory-mapped.
2. Spike times are bucketed per cluster during load.
3. `cluster_group.tsv` initializes one compact label per cluster.
4. Curation commands append to the SQLite journal.
5. After the journal write succeeds, the in-memory session applies the command.

This ordering keeps the current V1 curation state recoverable from the journal.
