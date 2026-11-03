//! Session-level checks for the statistical proposers (k-means split,
//! template-filtered merges, rejection memory, attention). The pure
//! proposer maths is unit-tested in `src/proposers.rs`; here we only check
//! the wiring against a provider that carries PC features and templates.
//!
//! Layout: cluster 0 = two units concatenated (interleaved in time, so the
//! parent is heavily contaminated and each unit is clean), clusters 1 and 2
//! = one neuron split in two (perfect refractory dip, similar templates),
//! cluster 3 = one clean unit, template-wise unrelated.

use sorrel_data::{
    journal::Journal, rank_merge_candidates, rank_split_candidates, Session, SuggestConfig,
};
use sorrel_io::{
    ClusterId, DataProvider, HasAmplitudes, HasPcFeatures, HasSpikeTemplates, HasTemplateWaveforms,
    SampleIndex, TraceSamples, TraceSlice,
};

const N_TEMPLATES: usize = 4;

struct Prov {
    times: Vec<Vec<SampleIndex>>,
    amps: Vec<Vec<f32>>,
    templates: Vec<Vec<u32>>,
    pc_rows: Vec<Vec<u32>>,
    pc: Vec<f32>,
    waves: Vec<f32>,
    sim: Vec<f32>,
    tags: Vec<Vec<String>>,
}

impl DataProvider for Prov {
    type Label = u8;
    fn sample_rate(&self) -> f32 {
        30_000.0
    }
    fn n_channels(&self) -> u32 {
        3
    }
    fn n_samples(&self) -> SampleIndex {
        SampleIndex(250_000)
    }
    fn n_clusters(&self) -> u32 {
        self.times.len() as u32
    }
    fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
        &self.times[c.idx()]
    }
    fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
        TraceSlice {
            start: SampleIndex(0),
            n_channels: 1,
            samples: TraceSamples::I16(&[]),
        }
    }
    fn initial_labels(&self) -> Vec<u8> {
        vec![0; self.times.len()]
    }
    fn initial_tags(&self) -> Vec<Vec<String>> {
        self.tags.clone()
    }
}
impl HasAmplitudes for Prov {
    fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
        &self.amps[c.idx()]
    }
}
impl HasSpikeTemplates for Prov {
    fn spike_templates(&self, c: ClusterId) -> &[u32] {
        &self.templates[c.idx()]
    }
}
impl HasPcFeatures for Prov {
    fn pc_features(&self) -> &[f32] {
        &self.pc
    }
    fn pc_shape(&self) -> (usize, usize) {
        (3, 3)
    }
    fn pc_feature_ind(&self) -> &[u32] {
        &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    }
    fn spike_pc_indices(&self, c: ClusterId) -> &[u32] {
        &self.pc_rows[c.idx()]
    }
}
impl HasTemplateWaveforms for Prov {
    fn template_waveforms(&self) -> &[f32] {
        &self.waves
    }
    fn template_shape(&self) -> (usize, usize, usize) {
        (N_TEMPLATES, 4, 3)
    }
    fn similar_templates(&self) -> &[f32] {
        &self.sim
    }
}

fn gaussian(s: &mut u32) -> f32 {
    let mut u = || {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        (*s as f64 / u32::MAX as f64).clamp(1e-9, 1.0)
    };
    let (u1, u2) = (u(), u());
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Adjusted Rand index for binary labelings.
fn ari(a: &[u8], b: &[u8]) -> f64 {
    let c2 = |x: f64| x * (x - 1.0) / 2.0;
    let mut t = [[0.0_f64; 2]; 2];
    for (&x, &y) in a.iter().zip(b) {
        t[x as usize][y as usize] += 1.0;
    }
    let sum: f64 = t.iter().flatten().map(|&v| c2(v)).sum();
    let ra: f64 = t.iter().map(|r| c2(r[0] + r[1])).sum();
    let rb: f64 = (0..2).map(|j| c2(t[0][j] + t[1][j])).sum();
    let exp = ra * rb / c2(a.len() as f64);
    (sum - exp) / (0.5 * (ra + rb) - exp)
}

/// Build the provider; `tags` is per-cluster. Returns the truth labels of
/// cluster 0 (0 = unit A, 1 = unit B) in local order.
fn build(tags: Vec<Vec<String>>) -> (Prov, Vec<u8>) {
    let mut s = 0xBEEF_u32;
    let mut p = Prov {
        times: vec![],
        amps: vec![],
        templates: vec![],
        pc_rows: vec![],
        pc: vec![],
        waves: vec![1.0; N_TEMPLATES * 4 * 3],
        sim: vec![0.1; N_TEMPLATES * N_TEMPLATES],
        tags,
    };
    for t in 0..N_TEMPLATES {
        p.sim[t * N_TEMPLATES + t] = 1.0;
    }
    p.sim[N_TEMPLATES + 2] = 0.9; // templates 1 and 2 are near-identical
    p.sim[2 * N_TEMPLATES + 1] = 0.9;

    let mut truth = Vec::new();
    // (template, times, unit-B shift on PC dims) per cluster.
    for c in 0..4_usize {
        let mut times = Vec::new();
        let mut unit = Vec::new();
        match c {
            0 => (0..300).for_each(|i| {
                times.extend([20 * i, 20 * i + 7].map(SampleIndex));
                unit.extend([0u8, 1]);
            }),
            1 => (0..500).for_each(|i| times.push(SampleIndex(i * 400))),
            2 => (0..500).for_each(|i| times.push(SampleIndex(i * 400 + 200))),
            _ => (0..500).for_each(|i| times.push(SampleIndex(i * 400 + 100))),
        }
        unit.resize(times.len(), 0);
        let first_row = (p.pc.len() / 9) as u32;
        p.pc_rows
            .push((first_row..first_row + times.len() as u32).collect());
        for &u in &unit {
            for k in 0..9 {
                let shift = if u == 1 && k % 3 == 0 && k < 6 {
                    5.0
                } else {
                    0.0
                };
                p.pc.push(gaussian(&mut s) + shift);
            }
        }
        p.amps.push(
            (0..times.len())
                .map(|_| 5.0 + 0.1 * gaussian(&mut s))
                .collect(),
        );
        p.templates.push(vec![c as u32; times.len()]);
        p.times.push(times);
        if c == 0 {
            truth = unit;
        }
    }
    (p, truth)
}

fn session(tags: Vec<Vec<String>>) -> (Session<Prov>, Vec<u8>, tempfile::TempDir) {
    let (p, truth) = build(tags);
    let dir = tempfile::tempdir().unwrap();
    let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
    let mut s = Session::new(p, journal);
    s.seed_amplitudes();
    s.seed_templates();
    s.seed_pc_features();
    s.seed_template_waveforms();
    (s, truth, dir)
}

#[test]
fn kmeans_split_recovers_concatenated_units_and_spares_single_unit() {
    let (s, truth, _d) = session(vec![]);
    let splits = rank_split_candidates(&s, &SuggestConfig::default());
    let c0 = splits
        .iter()
        .find(|c| c.cluster == ClusterId(0))
        .expect("cluster 0 flagged");
    assert_eq!(c0.method, "kmeans");
    let mut pred = vec![0u8; truth.len()];
    c0.bipartition_minor_idx
        .iter()
        .for_each(|&i| pred[i as usize] = 1);
    let score = ari(&truth, &pred);
    assert!(score > 0.9, "ARI {score}");
    assert!(
        splits
            .iter()
            .all(|c| c.cluster == ClusterId(0) || c.method != "kmeans"),
        "single units must not get a k-means split"
    );
}

#[test]
fn notsplit_tag_suppresses_split() {
    let (s, _, _d) = session(vec![vec!["notsplit".into()]]);
    let splits = rank_split_candidates(&s, &SuggestConfig::default());
    assert!(splits.iter().all(|c| c.cluster != ClusterId(0)));
}

#[test]
fn merge_prefilter_and_notmerge_memory() {
    let (s, _, _d) = session(vec![]);
    let merges = rank_merge_candidates(&s, &SuggestConfig::default());
    assert!(
        merges
            .iter()
            .any(|m| (m.a, m.b) == (ClusterId(1), ClusterId(2))),
        "{merges:?}"
    );
    assert!(
        merges
            .iter()
            .all(|m| m.a.0 >= 1 && m.b.0 >= 1 && (m.a.0, m.b.0) == (1, 2)),
        "template pre-filter should leave only the similar pair: {merges:?}"
    );

    let (s, _, _d) = session(vec![vec![], vec!["notmerge:2".into()]]);
    let merges = rank_merge_candidates(&s, &SuggestConfig::default());
    assert!(merges.is_empty(), "notmerge must hide the pair: {merges:?}");
}

#[test]
fn attention_ranks_flagged_clusters_first() {
    let (s, _, _d) = session(vec![vec![], vec![], vec![], vec!["review:check".into()]]);
    let all = s.attention_all();
    let get = |c: u32| all.iter().find(|a| a.0 == ClusterId(c)).unwrap();
    assert!(get(0).2.contains(&"split"), "{:?}", get(0));
    assert_eq!(get(3).1, 1.0);
    assert!(get(3).2.contains(&"review_tag"));
    assert!(get(1).2.contains(&"merge"), "{:?}", get(1));
    assert!(
        all.windows(2).all(|w| w[0].1 >= w[1].1),
        "sorted descending"
    );
    assert_eq!(s.attention(ClusterId(3)).0, 1.0);
}

impl sorrel_data::session::ApplyPhyLabel for Prov {
    fn label_from_op(_: sorrel_data::PhyLabelOp) -> u8 {
        0
    }
}
impl sorrel_io::HasGeometry for Prov {
    fn channel_positions(&self) -> &[[f32; 2]] {
        &[]
    }
}

/// The MCP attention queue uses the shared scorer, so a cluster whose only
/// problem is PC-space outliers is listed with the "outliers" reason.
#[test]
fn mcp_attention_queue_includes_pc_outliers() {
    let (mut p, _) = build(vec![]);
    // 4% of the clean cluster 3 far out, in varying directions.
    for (j, &row) in p.pc_rows[3].clone().iter().step_by(25).enumerate() {
        for k in 0..9 {
            let bit = ((j as u32 + 1).wrapping_mul(2_654_435_761) >> (k + 3)) & 1;
            p.pc[row as usize * 9 + k] = if bit == 0 { 15.0 } else { -15.0 };
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
    let mut s = Session::new(p, journal);
    s.seed_amplitudes();
    s.seed_templates();
    s.seed_pc_features();
    s.seed_template_waveforms();
    let want = s.attention(ClusterId(3));
    assert!(want.1.contains(&"outliers"), "{want:?}");
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"attention_queue","arguments":{"limit":50}}}"#;
    let mut out = Vec::new();
    sorrel_data::serve(
        &sorrel_data::SessionHost::new(s),
        dir.path(),
        false,
        |_: &u8| "unsorted",
        std::io::Cursor::new(format!("{req}\n")),
        &mut out,
    )
    .unwrap();
    let reply: serde_json::Value = serde_json::from_slice(&out).unwrap();
    let rows: serde_json::Value =
        serde_json::from_str(reply["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    let row = rows
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["cluster"] == 3)
        .expect("outlier cluster listed");
    assert!(
        row["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r == "outliers"),
        "{row}"
    );
}

/// Provider of `clusters` of `(time, flagged)` spikes. PC dims 0..3 of a
/// flagged spike are `N(mu, sd)`, everything else `N(0, 1)`.
fn flagged_prov(clusters: Vec<Vec<(u64, bool)>>, mu: f32, sd: f32) -> (Prov, Vec<bool>) {
    let mut s = 0xC0DE_u32;
    let mut p = Prov {
        times: vec![],
        amps: vec![],
        templates: vec![],
        pc_rows: vec![],
        pc: vec![],
        waves: vec![1.0; N_TEMPLATES * 4 * 3],
        sim: vec![0.1; N_TEMPLATES * N_TEMPLATES],
        tags: vec![],
    };
    let mut truth = Vec::new();
    for (c, spikes) in clusters.into_iter().enumerate() {
        let first_row = (p.pc.len() / 9) as u32;
        p.pc_rows
            .push((first_row..first_row + spikes.len() as u32).collect());
        for &(_, flagged) in &spikes {
            for k in 0..9 {
                let (mu, sd) = if flagged { (mu, sd) } else { (0.0, 1.0) };
                p.pc.push(mu * (k < 3) as u8 as f32 + sd * gaussian(&mut s));
            }
        }
        p.amps.push(vec![5.0; spikes.len()]);
        p.templates.push(vec![c as u32; spikes.len()]);
        p.times
            .push(spikes.iter().map(|&(t, _)| SampleIndex(t)).collect());
        if c == 0 {
            truth = spikes.iter().map(|&(_, f)| f).collect();
        }
    }
    (p, truth)
}

/// Cluster 0 = a broad contaminated cloud (triples 8 samples apart, 97%)
/// around a tight refractory core (3%, offset in PC space); cluster 1 = one
/// clean unimodal unit. Returns the provider and cluster 0's core truth.
fn core_prov() -> (Prov, Vec<bool>) {
    let mut host: Vec<(u64, bool)> = (0..1293_u64)
        .flat_map(|j| [0, 8, 16].map(|o| (190 * j + 5 + o, false)))
        .collect();
    host.extend((0..120_u64).map(|i| (2000 * i + 1000, true)));
    host.sort_unstable();
    let clean: Vec<(u64, bool)> = (0..500_u64).map(|i| (i * 400 + 100, false)).collect();
    flagged_prov(vec![host, clean], 4.0, 0.3)
}

/// Cluster 0 = 75% a clean refractory unit (spikes 100 samples apart) plus 25%
/// a broad contaminated cloud (triples 8 samples apart, offset in PC space);
/// cluster 1 = one clean unit. Truth is true for the cloud.
fn cleanup_prov() -> (Prov, Vec<bool>) {
    let mut host: Vec<(u64, bool)> = (0..2400_u64).map(|i| (100 * i + 17, false)).collect();
    host.extend((0..267_u64).flat_map(|j| [0, 8, 16].map(|o| (900 * j + 50 + o, true))));
    host.sort_unstable();
    let clean: Vec<(u64, bool)> = (0..500_u64).map(|i| (i * 400 + 100, false)).collect();
    flagged_prov(vec![host, clean], 5.0, 1.5)
}

fn seeded(p: Prov) -> (Session<Prov>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new(p, Journal::open(&dir.path().join("j.sqlite")).unwrap());
    s.seed_amplitudes();
    s.seed_templates();
    s.seed_pc_features();
    s.seed_template_waveforms();
    (s, dir)
}

fn jaccard(got: &[u32], want: impl Iterator<Item = u32>) -> f64 {
    let got: std::collections::HashSet<u32> = got.iter().copied().collect();
    let want: std::collections::HashSet<u32> = want.collect();
    got.intersection(&want).count() as f64 / got.union(&want).count() as f64
}

#[test]
fn core_split_isolates_the_refractory_core_in_a_contaminated_cluster() {
    let (p, truth) = core_prov();
    let (s, dir) = seeded(p);

    let splits = rank_split_candidates(&s, &SuggestConfig::default());
    let c0 = splits
        .iter()
        .find(|c| c.cluster == ClusterId(0))
        .expect("host flagged");
    assert_eq!(c0.method, "core", "{c0:?}");
    let want = (0..truth.len() as u32).filter(|&i| truth[i as usize]);
    let j = jaccard(&c0.bipartition_minor_idx, want);
    assert!(
        j >= 0.9,
        "Jaccard {j}, {} spikes",
        c0.bipartition_minor_idx.len()
    );
    assert!(
        splits.iter().all(|c| c.cluster != ClusterId(1)),
        "a clean unimodal unit gets no proposal"
    );

    let line = probe_line(dir.path(), "split 0 by=core", s);
    assert!(
        line.contains("s0.1 core") && line.contains("parts 3879/120"),
        "{line}"
    );
}

fn probe_line(dir: &std::path::Path, q: &str, s: Session<Prov>) -> String {
    let req = serde_json::json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
        "params":{"name":"probe","arguments":{"q":q}}});
    let mut out = Vec::new();
    sorrel_data::serve(
        &sorrel_data::SessionHost::new(s),
        dir,
        false,
        |_: &u8| "unsorted",
        std::io::Cursor::new(format!("{req}\n")),
        &mut out,
    )
    .unwrap();
    let reply: serde_json::Value = serde_json::from_slice(&out).unwrap();
    reply["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn cleanup_split_drops_the_contaminated_minority_and_keeps_the_clean_majority() {
    let (p, truth) = cleanup_prov();
    let (s, dir) = seeded(p);

    // k-means outranks the core proposer in the queue and also finds this
    // well-separated cloud, so call the core/cleanup proposer directly.
    let globals = sorrel_data::proposers::cluster_globals(&s, ClusterId(0));
    let c0 = sorrel_data::proposers::core_split(&s, ClusterId(0), &globals).expect("cleanup");
    assert_eq!(c0.method, "cleanup", "{c0:?}");
    let n = truth.len() as u32;
    // The kept part is the clean unit; the minor part is the dropped cloud.
    let minor: std::collections::HashSet<u32> = c0.minor_idx.iter().copied().collect();
    let kept: Vec<u32> = (0..n).filter(|i| !minor.contains(i)).collect();
    let j = jaccard(&kept, (0..n).filter(|&i| !truth[i as usize]));
    assert!(j >= 0.9, "kept Jaccard {j}, {} dropped", minor.len());
    assert!(
        minor.len() * 2 < n as usize,
        "the dropped part is the minority"
    );
    assert!(
        sorrel_data::proposers::core_split(
            &s,
            ClusterId(1),
            &sorrel_data::proposers::cluster_globals(&s, ClusterId(1))
        )
        .is_none(),
        "a clean unimodal unit gets no proposal"
    );

    let line = probe_line(dir.path(), "split 0 by=core", s);
    assert!(line.contains("s0.1 cleanup"), "{line}");
}

#[test]
fn core_split_declines_a_clean_or_unstructured_cluster() {
    use sorrel_data::proposers::{core_split_proposal, whiten};
    let mut s = 7_u32;
    let raw: Vec<f32> = (0..4000 * 9).map(|_| gaussian(&mut s)).collect();
    let (feats, d) = whiten(&raw, 9).unwrap();
    // Contaminated but featureless: BIC keeps one component.
    assert!(core_split_proposal(&feats, d, |_| 0.5, |_| 100).is_none());
    // Clean whole: never tried.
    assert!(core_split_proposal(&feats, d, |_| 0.0, |_| 0).is_none());
}
