# Views

Every view is a dockable tile. The default layout puts three tab groups
side by side — _shape_ (Waveforms, Templates, Probe, Summary),
_comparison_ (CCG, Features, Amplitudes, ISI, Rate, Similar) and
_assessment_ (Quality, Suggest, Stats, Raster, Drift, Contam·time); PSTH sits in the comparison group — above
a full-width Trace, so a waveform overlay, a correlogram grid and a feature
scatter of the same selection are visible at once, as in phy.

- Drag a tab title to move it into another group, or to the edge of a
  tile to split it. Drag the borders to resize.
- `×` on a tab closes the view; the `+` at the right of a tab bar or the
  **View** menu in the top bar reopens it. **View → Reset layout** restores
  the default.
- The layout, table sort/filter, trace settings and last selected cluster
  are saved per dataset in `<dataset>/.sorrel/ui.json` and restored on the
  next open.

The cluster table stays in the left panel. Clusters in the selection are
coloured by their position in it (first = blue, second = orange, …), and
every view uses the same colour for the same cluster.

| View              | What it shows                                                                                                                                                                                                                                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Trace**         | Raw recording for a configurable window, one row per channel, with channel numbers, a time axis and a tick on each selected cluster's peak channel for every spike in view. LTTB downsampling with optional CMR + high-pass. Drag or wheel to pan, Ctrl-wheel to zoom.                                           |
| **Trace image**   | phy v2.1's TraceImageView: the trace window as an image (blue negative, red positive), following the Trace window, filters and origin.                                                                                                                                                                           |
| **Spike data**    | phy's Spike*Views: any `spike_<name>.npy` against time (1-D) or as x vs y (2-D), Ctrl-click lasso and `K` to split.                                                                                                                                                                                              |
| **Summary**       | Selection summary: cluster ids, spike counts and labels.                                                                                                                                                                                                                                                         |
| **Waveforms**     | Spike overlays plus the mean waveform per selected cluster, in probe layout when geometry is available, otherwise on the peak channel. Snippets go through the Trace pane's HP/CMR toggles (with a settling margin before each snippet), so they match the trace.                                                |
| **Templates**     | phy's TemplateView: every cluster's template on its peak channel, in table order, the selection outlined; click selects, Ctrl/Shift-click adds.                                                                                                                                                                  |
| **Amplitudes**    | phy's AmplitudeView: template, feature or raw amplitude over time (`A` cycles), grey background spikes on the peak channel, a per-cluster histogram, Ctrl-click lasso and Alt-right-drag threshold, `K` splits; Alt-click seeks the trace.                                                                       |
| **ISI**           | Inter-spike interval histogram per cluster with the refractory period shaded.                                                                                                                                                                                                                                    |
| **CCG**           | Auto- and cross-correlogram grid across the selected clusters (memoised per selection).                                                                                                                                                                                                                          |
| **Features**      | phy's FeatureView: a 4×4 grid of PC1/PC2 on the selection's two best channels with time on the diagonal and grey background spikes. Uses `pc_features.npy` / `pc_feature_ind.npy` when present, otherwise PCs fitted on raw snippets. Ctrl-click a polygon lasso and press `K` to split; `C` fixes the channels. |
| **Raster**        | Every spike of every cluster, time on x, cluster row on y. Selected rows are highlighted and the current trace window is outlined; click to seek.                                                                                                                                                                |
| **Rate**          | Firing rate over the recording, in Hz, one line per selected cluster; bin count (`:frn`, `:frb`) and time unit (s, min, h) in the header.                                                                                                                                                                        |
| **Probe**         | Channel layout from probe geometry with a ring on each selected cluster's peak channel.                                                                                                                                                                                                                          |
| **Similar**       | Clusters whose template is most similar to the first selected one. Click a row to compare, Cmd/Ctrl-click to add it to the selection.                                                                                                                                                                            |
| **Stats**         | Histograms of spike count, mean amplitude and ISI violations across all clusters.                                                                                                                                                                                                                                |
| **Quality**       | Composite quality breakdown (refractory, presence, coverage, SNR, completeness, stability, isolation). Computed on a background thread; a spinner shows while it runs.                                                                                                                                           |
| **Suggest**       | Ranked merge and split suggestions with predicted outcomes. Ranked on a background thread; curation stays responsive while it runs.                                                                                                                                                                              |
| **Drift**         | Spike depth (µm, PC-weighted centre of mass with template-peak fallback) over time with a regression line per cluster; amplitude over time when the backend has no depth estimate.                                                                                                                               |
| **Contam·time**   | Sliding refractory contamination over the recording.                                                                                                                                                                                                                                                             |
| **Tmpl features** | phy's TemplateFeatureView: with two clusters selected, each spike projected on both clusters' templates (`template_features.npy`), Ctrl-click lasso, `K` splits.                                                                                                                                                 |
| **Scatter**       | phy v2.1's ClusterScatterView: one point per cluster, x / y / size from amp, depth, fr, n_spikes or ch; click selects, Shift-click adds, Ctrl-click lasso selects.                                                                                                                                               |
| **PSTH**          | Event-aligned firing rate per selected cluster plus a per-trial raster of the first one. Needs `events.csv` (`time,label`, seconds) or `events.npy` next to the dataset, an NWB file's `/intervals/trials` table, or `--events FILE`; pick the label, pre/post window and bins in the toolbar.                   |

The Trace toolbar offers **per shank** CMR when the probe has more than one
shank (`channel_shanks.npy`), referencing each shank by its own median.

Plots follow the egui theme, so light mode renders light plot backgrounds.
