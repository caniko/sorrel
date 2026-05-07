# Views

The central panel is a tab strip. Each tab consumes the active selection
and renders into the available area; switching tabs preserves selection,
trace window, and per-view caches.

| Tab | What it shows |
|-----|---------------|
| **Summary** | Cluster table + selection summary. The cluster table sorts by id, spike count, amplitude, ISI-violation rate, composite quality score, or label, with text and noise/empty filters. |
| **Waveforms** | Mean ± std waveform per selected cluster, in probe layout when geometry is available, otherwise single-channel. |
| **Templates** | Template traces (when `templates.npy` is present). |
| **Amplitudes** | Per-spike amplitude vs time scatter for each selected cluster. |
| **ISI** | Inter-spike interval histogram with a refractory-period overlay. |
| **CCG** | Auto- and cross-correlograms across the selected clusters. |
| **Features** | PC-feature scatter (uses `pc_features.npy` / `pc_feature_ind.npy`). |
| **Raster** | Spike raster across selected clusters, time-aligned with the trace window. |
| **Rate** | Sliding-window firing rate. |
| **Probe** | Channel layout from probe geometry. |
| **Similar** | Similarity ranking — `similar_templates.npy` when present, otherwise computed from templates. |
| **Stats** | Per-cluster numeric breakdown: refractory contamination, ISI stats, amplitude IQR, etc. |
| **Quality** | Composite quality breakdown (refractory, presence, isolation). |
| **Suggest** | Ranked merge and split suggestions for the active selection. |
| **DriftMap** | Spike amplitude vs depth over time. |
| **ContamTime** | Sliding refractory contamination over the recording. |

The trace view is always visible above the tabs. It uses LTTB
downsampling on a configurable trace window (`H` / `L` to pan), with
optional CMR + high-pass filtering applied before downsampling.
