//! Counterfactual evidence: the metrics a merge or split *would* produce,
//! computed without applying it, plus the handles (`s12.1`, `m3`) that let an
//! agent act on a suggestion without transcribing spike indices.
//!
//! `sync_z` / `dip_z` are z-scores of the cross-correlogram (1 ms bins,
//! +-50 ms) of two trains: `dip_z = -(shoulder mean - centre mean) / shoulder
//! std` over the +-2 ms centre (shoulders: the next 8 bins each side), so a
//! refractory dip is NEGATIVE (one neuron). `sync_z` is the same statistic
//! over a +-3 ms centre (12 shoulder bins) clipped at 0: positive means the
//! centre sits ABOVE the shoulders, two neurons firing together, so do not
//! merge. Both are clamped to +-99; NaN (`-`) when a train is empty or the
//! shoulders average under [`MIN_SHOULDER`] counts per bin (the z is then
//! noise, e.g. 99 from a std floor).

use crate::brief::{quantise, thresholds};
use crate::proposals::spike_basis;
use crate::session::Session;
use crate::suggest::{MergeCandidate, SplitCandidate};
use serde::Serialize;
use sorrel_compute::{
    analyse_refractory_dip, cross_correlogram, isi_violation_rate, ks_two_sample, mean_amplitude,
    quality_breakdown, sliding_refractory_argmin,
};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

const PRESENCE_BINS: usize = 20;
/// Pairwise CCGs are O(k^2); past this many parts only the totals are given.
const MAX_PAIR_PARTS: usize = 6;
pub const STALE: &str = "call brief level 3 again";

/// The unit metrics `cluster_qc.tsv` reports, for one spike train.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct UnitStats {
    pub n: usize,
    pub rp_contamination: f32,
    pub isi_violation_ratio: f32,
    pub presence_ratio: f32,
    pub amplitude_cutoff: f32,
    pub drift_correlation: f32,
}

/// CCG evidence between part `i` and part `j` of an evaluation.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PairStats {
    pub i: usize,
    pub j: usize,
    pub dip_z: f32,
    pub sync_z: f32,
    pub amp_ks: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct MergeEval {
    pub ids: Vec<u32>,
    pub parts: Vec<UnitStats>,
    pub merged: UnitStats,
    pub pairs: Vec<PairStats>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SplitEval {
    pub cluster: u32,
    /// `[major, minor]`.
    pub parts: [UnitStats; 2],
    pub mean_amp: [f32; 2],
    pub dip_z: f32,
    pub sync_z: f32,
}

pub fn unit_stats(times: &[SampleIndex], amps: &[f32], sr: f32, total: u64) -> UnitStats {
    let has = !amps.is_empty() && amps.len() == times.len();
    let (c, rp) = sliding_refractory_argmin(
        times,
        (sr * 0.0005) as u64,
        (sr * 0.010) as u64,
        20,
        total,
        sr,
    );
    let refractory = (sr * 0.0015).round() as u64;
    let q = quality_breakdown(
        times,
        if has { amps } else { &[] },
        refractory,
        total,
        sr,
        PRESENCE_BINS,
    );
    let or_nan = |v: f32, ok: bool| if ok { v } else { f32::NAN };
    UnitStats {
        n: times.len(),
        rp_contamination: or_nan(c, rp > 0),
        isi_violation_ratio: isi_violation_rate(times, refractory, sr),
        presence_ratio: q.raw_presence_ratio,
        amplitude_cutoff: or_nan(q.raw_amp_cutoff, has),
        drift_correlation: or_nan(q.raw_drift_corr, has),
    }
}

/// Fewest mean shoulder counts per CCG bin for a z-score to mean anything.
const MIN_SHOULDER: f32 = 2.0;

/// `(dip_z, sync_z)` of the CCG of `a` against `b` (see the module doc).
pub fn ccg_z(a: &[SampleIndex], b: &[SampleIndex], sr: f32) -> (f32, f32) {
    let max_lag = (sr * 0.050).round() as u64;
    if a.is_empty() || b.is_empty() || max_lag == 0 {
        return (f32::NAN, f32::NAN);
    }
    let h = cross_correlogram(a, b, max_lag, 100);
    let z = |r, s| {
        let a = analyse_refractory_dip(&h, r, s);
        // std is floored at 1e-6 by the analysis: a floored std is zero variance.
        match a.shoulder_mean < MIN_SHOULDER || a.shoulder_std <= 1e-5 {
            true => f32::NAN,
            false => -a.z,
        }
    };
    let (dip, sync) = (z(2, 8), z(3, 12));
    // f32::max(NaN, 0) is 0: keep NaN a NaN
    let sync = if sync.is_nan() { sync } else { sync.max(0.0) };
    (dip.clamp(-99.0, 99.0), sync.clamp(-99.0, 99.0))
}

/// Merge of `parts` (time-sorted trains with their amplitudes) evaluated
/// without ids; [`eval_merge`] fills them in.
pub fn eval_trains(parts: &[(&[SampleIndex], &[f32])], sr: f32, total: u64) -> MergeEval {
    let stats: Vec<UnitStats> = parts
        .iter()
        .map(|&(t, a)| unit_stats(t, a, sr, total))
        .collect();
    // ponytail: concat + stable sort, O(n log n); a k-way heap merge only
    // matters past ~10^7 spikes.
    let amps_ok = parts.iter().all(|(t, a)| t.len() == a.len());
    let mut all: Vec<(SampleIndex, f32)> = parts
        .iter()
        .flat_map(|&(t, a)| {
            t.iter()
                .enumerate()
                .map(move |(i, &t)| (t, a.get(i).copied().unwrap_or(f32::NAN)))
        })
        .collect();
    all.sort_by_key(|p| p.0 .0);
    let times: Vec<SampleIndex> = all.iter().map(|p| p.0).collect();
    let amps: Vec<f32> = match amps_ok {
        true => all.iter().map(|p| p.1).collect(),
        false => Vec::new(),
    };
    let mut pairs = Vec::new();
    if parts.len() <= MAX_PAIR_PARTS {
        for i in 0..parts.len() {
            for j in i + 1..parts.len() {
                let (dip_z, sync_z) = ccg_z(parts[i].0, parts[j].0, sr);
                let (a, b) = (parts[i].1, parts[j].1);
                let amp_ks = match a.is_empty() || b.is_empty() {
                    true => f32::NAN,
                    false => ks_two_sample(a, b),
                };
                pairs.push(PairStats {
                    i,
                    j,
                    dip_z,
                    sync_z,
                    amp_ks,
                });
            }
        }
    }
    MergeEval {
        ids: Vec::new(),
        merged: unit_stats(&times, &amps, sr, total),
        parts: stats,
        pairs,
    }
}

pub fn eval_merge<P: DataProvider>(session: &Session<P>, ids: &[ClusterId]) -> MergeEval {
    let parts: Vec<(&[SampleIndex], &[f32])> = ids
        .iter()
        .map(|&c| (session.spike_times(c), session.spike_amplitudes(c)))
        .collect();
    let mut ev = eval_trains(
        &parts,
        session.provider.sample_rate().max(1.0),
        session.provider.n_samples().0,
    );
    ev.ids = ids.iter().map(|c| c.0).collect();
    ev
}

/// Split of one train by local spike positions `minor` (ascending or not).
pub fn eval_split_trains(
    times: &[SampleIndex],
    amps: &[f32],
    minor: &[u32],
    sr: f32,
    total: u64,
) -> SplitEval {
    let is_minor: BTreeSet<usize> = minor.iter().map(|&i| i as usize).collect();
    let pick = |want: bool| -> (Vec<SampleIndex>, Vec<f32>) {
        let keep = |i: &usize| is_minor.contains(i) == want;
        let t = (0..times.len()).filter(keep).map(|i| times[i]).collect();
        let a = match amps.len() == times.len() {
            true => (0..times.len()).filter(keep).map(|i| amps[i]).collect(),
            false => Vec::new(),
        };
        (t, a)
    };
    let (major, minor) = (pick(false), pick(true));
    let mean = |a: &[f32]| match a.is_empty() {
        true => f32::NAN,
        false => mean_amplitude(a),
    };
    let (dip_z, sync_z) = ccg_z(&major.0, &minor.0, sr);
    SplitEval {
        cluster: 0,
        parts: [
            unit_stats(&major.0, &major.1, sr, total),
            unit_stats(&minor.0, &minor.1, sr, total),
        ],
        mean_amp: [mean(&major.1), mean(&minor.1)],
        dip_z,
        sync_z,
    }
}

pub fn eval_split<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    minor_idx: &[u32],
) -> SplitEval {
    let mut ev = eval_split_trains(
        session.spike_times(cluster),
        session.spike_amplitudes(cluster),
        minor_idx,
        session.provider.sample_rate().max(1.0),
        session.provider.n_samples().0,
    );
    ev.cluster = cluster.0;
    ev
}

type Col = (&'static str, &'static str, fn(&UnitStats) -> f32);

fn short(s: String) -> String {
    if let Some(r) = s.strip_prefix("0.") {
        format!(".{r}")
    } else if let Some(r) = s.strip_prefix("-0.") {
        format!("-.{r}")
    } else {
        s
    }
}
pub(crate) fn qm(v: f32, metric: &str) -> String {
    short(quantise(v, &thresholds(metric)))
}
pub(crate) fn qn(v: f32) -> String {
    short(quantise(v, &[]))
}

/// `m2 {4,9}: rpc .03,.02→.04 pres .52,.47→.98 cut .2,.1→.05 | 4-9 dip_z -4.1
/// sync_z .3 ks .08 sim .93 d 12um pc .8`. With candidate `edges`, only the
/// pairs that are edges (best four) are listed, with their score causes.
pub fn render_merge(label: &str, ev: &MergeEval, edges: &[MergeCandidate]) -> String {
    let set: Vec<String> = ev.ids.iter().map(u32::to_string).collect();
    let mut s = format!("{label} {{{}}}:", set.join(","));
    let cols: [Col; 3] = [
        ("rpc", "rp_contamination", |u| u.rp_contamination),
        ("pres", "presence_ratio", |u| u.presence_ratio),
        ("cut", "amplitude_cutoff", |u| u.amplitude_cutoff),
    ];
    for (name, metric, f) in cols {
        let parts: Vec<String> = ev.parts.iter().map(|u| qm(f(u), metric)).collect();
        let _ = write!(
            s,
            " {name} {}→{}",
            parts.join(","),
            qm(f(&ev.merged), metric)
        );
    }
    let edge = |p: &PairStats| {
        let key = (ev.ids[p.i], ev.ids[p.j]);
        edges
            .iter()
            .find(|e| (e.a.0, e.b.0) == key || (e.b.0, e.a.0) == key)
    };
    let mut pairs: Vec<(&PairStats, Option<&MergeCandidate>)> =
        ev.pairs.iter().map(|p| (p, edge(p))).collect();
    if !edges.is_empty() {
        pairs.retain(|p| p.1.is_some());
        let score = |p: &(&PairStats, Option<&MergeCandidate>)| p.1.map_or(0.0, |e| e.score);
        pairs.sort_by(|a, b| score(b).total_cmp(&score(a)));
        pairs.truncate(4);
    }
    for (k, (p, e)) in pairs.iter().enumerate() {
        s.push_str(if k == 0 { " | " } else { " ; " });
        let _ = write!(
            s,
            "{}-{} dip_z {} sync_z {} ks {}",
            ev.ids[p.i],
            ev.ids[p.j],
            qn(p.dip_z),
            qn(p.sync_z),
            qn(p.amp_ks)
        );
        if let Some(e) = e {
            if e.template_sim.is_finite() {
                let _ = write!(s, " sim {}", qn(e.template_sim));
            }
            if e.centroid_um.is_finite() {
                let _ = write!(s, " d {}um", qn(e.centroid_um));
            }
            let _ = write!(s, " pc {}", qn(e.presence_complement));
        }
    }
    s
}

/// `s12.1 kmeans sc .62: parts 1200/300 amp 80,140 rpc .02,.3 dip_z -3.9 bic 41`.
pub fn render_split(handle: &str, method: &str, score: f32, bic: f32, ev: &SplitEval) -> String {
    let two = |f: &dyn Fn(usize) -> String| format!("{},{}", f(0), f(1));
    format!(
        "{handle} {method} sc {}: parts {}/{} amp {} rpc {} dip_z {} sync_z {} bic {}",
        qn(score),
        ev.parts[0].n,
        ev.parts[1].n,
        two(&|i| qn(ev.mean_amp[i])),
        two(&|i| qm(ev.parts[i].rp_contamination, "rp_contamination")),
        qn(ev.dip_z),
        qn(ev.sync_z),
        qn(bic)
    )
}

struct SplitHandle {
    cluster: ClusterId,
    minor: Vec<u32>,
    basis: String,
    method: &'static str,
    score: f32,
    bic: f32,
}
struct CompHandle {
    members: Vec<ClusterId>,
    bases: Vec<String>,
}

/// A relation for the brief / `suggest`: the handle, the clusters it
/// touches and its one-line counterfactual.
#[derive(Clone, Debug, Serialize)]
pub struct Relation {
    pub handle: String,
    pub clusters: Vec<u32>,
    pub line: String,
}

/// Per-connection handle table. A split handle `s<cluster>.<k>` resolves to
/// that cluster's minor spike positions and the [`spike_basis`] they were
/// proposed against; a merge handle `m<k>` to a component's clusters. The
/// table is rebuilt (never reset) when `Session::version()` changes: a handle
/// from an earlier build keeps resolving while every cluster it names still
/// has the basis it had at creation; an edit to any of them makes it stale.
#[derive(Default)]
pub struct Handles {
    version: Option<u64>,
    next_m: u32,
    splits: BTreeMap<String, SplitHandle>,
    comps: BTreeMap<String, CompHandle>,
    /// Ids issued by the latest rebuild (the ones worth listing).
    current: BTreeSet<String>,
}

fn find(p: &mut BTreeMap<u32, u32>, x: u32) -> u32 {
    let px = *p.entry(x).or_insert(x);
    if px == x {
        return x;
    }
    let r = find(p, px);
    p.insert(x, r);
    r
}

/// Weakest candidate edge that may join a merge component: below it the pairs
/// chain unrelated units into one blob (0.15 gave a 31-unit component).
/// Left-out pairs stay reachable with `eval {merge:[..]}`.
const COMPONENT_MIN_SCORE: f32 = 0.4;
/// Most units in one merge component.
const COMPONENT_MAX: usize = 4;

/// Components of `(a, b, score)` edges: strongest edges first, an edge below
/// [`COMPONENT_MIN_SCORE`] or that would grow a component past
/// [`COMPONENT_MAX`] is skipped. Members ascending, components sorted.
// ponytail: greedy size cap, not a clustering; use average-linkage if the cap
// splits a true chain badly.
fn components(edges: &[(u32, u32, f32)]) -> Vec<Vec<u32>> {
    let mut es: Vec<_> = edges
        .iter()
        .filter(|e| e.2 >= COMPONENT_MIN_SCORE)
        .collect();
    es.sort_by(|x, y| y.2.total_cmp(&x.2).then((x.0, x.1).cmp(&(y.0, y.1))));
    let mut parent = BTreeMap::new();
    let mut groups: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &&(a, b, _) in &es {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra == rb {
            continue;
        }
        let mut g = groups.remove(&ra).unwrap_or_else(|| vec![a]);
        let h = groups.remove(&rb).unwrap_or_else(|| vec![b]);
        if g.len() + h.len() > COMPONENT_MAX {
            groups.insert(ra, g);
            groups.insert(rb, h);
            continue;
        }
        g.extend(h);
        parent.insert(ra, rb);
        groups.insert(rb, g);
    }
    let mut comps: Vec<Vec<u32>> = groups.into_values().filter(|g| g.len() >= 2).collect();
    comps.iter_mut().for_each(|g| g.sort_unstable());
    comps.sort();
    comps
}

impl Handles {
    /// Rebuild from the current suggestions if the session changed.
    // ponytail: spike_basis scans every spike per handle; cache per cluster
    // if a recording with thousands of candidates makes refresh slow.
    pub fn refresh<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        merges: &[MergeCandidate],
        splits: &[SplitCandidate],
    ) {
        if self.version == Some(session.version()) {
            return;
        }
        self.version = Some(session.version());
        self.current.clear();
        // Stale split handles stay (resolving refuses them by basis), so
        // their ids are never reissued. ponytail: grows by one entry per
        // retired candidate; fine for a session's lifetime.
        self.comps.retain(|_, c| {
            (c.members.iter().zip(&c.bases)).all(|(m, b)| spike_basis(session, *m) == *b)
        });
        for s in splits
            .iter()
            .filter(|s| !s.bipartition_minor_idx.is_empty())
        {
            let basis = spike_basis(session, s.cluster);
            let id = self.split_id(s.cluster, &s.bipartition_minor_idx, &basis);
            self.current.insert(id.clone());
            self.splits.insert(
                id,
                SplitHandle {
                    cluster: s.cluster,
                    minor: s.bipartition_minor_idx.clone(),
                    basis,
                    method: s.method,
                    score: s.score,
                    bic: s.gmm_bic_delta,
                },
            );
        }
        let edges: Vec<(u32, u32, f32)> = merges.iter().map(|m| (m.a.0, m.b.0, m.score)).collect();
        let comps = components(&edges);
        for members in comps {
            let members: Vec<ClusterId> = members.into_iter().map(ClusterId).collect();
            let existing = self.comps.iter().find(|(_, c)| c.members == members);
            let id = match existing {
                Some((id, _)) => id.clone(),
                None => {
                    self.next_m += 1;
                    format!("m{}", self.next_m)
                }
            };
            let bases = members.iter().map(|&c| spike_basis(session, c)).collect();
            self.current.insert(id.clone());
            self.comps.insert(id, CompHandle { members, bases });
        }
    }

    /// The id for splitting `minor` off `cluster`: the handle already naming
    /// exactly that split, else the next free `s<cluster>.<k>`. A live id is
    /// never reissued for different spikes, so a handle the LLM evaluated
    /// always splits what it evaluated.
    fn split_id(&self, cluster: ClusterId, minor: &[u32], basis: &str) -> String {
        let same = |h: &SplitHandle| h.cluster == cluster && h.minor == minor && h.basis == basis;
        if let Some((id, _)) = self.splits.iter().find(|(_, h)| same(h)) {
            return id.clone();
        }
        let prefix = format!("s{}.", cluster.0);
        let k = (self.splits.keys())
            .filter_map(|id| id.strip_prefix(&prefix)?.parse::<u32>().ok())
            .max()
            .unwrap_or(0)
            + 1;
        format!("{prefix}{k}")
    }

    /// Register a split of `cluster` (local positions `minor`) from `probe`
    /// (see [`Self::split_id`]); it resolves like a ranker handle.
    pub fn add_split<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        cluster: ClusterId,
        minor: Vec<u32>,
        method: &'static str,
        score: f32,
        bic: f32,
    ) -> String {
        let basis = spike_basis(session, cluster);
        let id = self.split_id(cluster, &minor, &basis);
        let h = SplitHandle {
            cluster,
            minor,
            basis,
            method,
            score,
            bic,
        };
        self.splits.insert(id.clone(), h);
        id
    }

    fn stale(id: &str) -> String {
        format!("stale handle {id}; {STALE}")
    }

    /// `(cluster, minor spike positions, basis)` of split handle `id`, if
    /// its cluster is unchanged.
    pub fn resolve_split<P: DataProvider>(
        &self,
        session: &Session<P>,
        id: &str,
    ) -> Result<(ClusterId, Vec<u32>, String), String> {
        let h = self.splits.get(id).ok_or_else(|| Self::stale(id))?;
        match spike_basis(session, h.cluster) == h.basis {
            true => Ok((h.cluster, h.minor.clone(), h.basis.clone())),
            false => Err(Self::stale(id)),
        }
    }

    /// The clusters of merge handle `id`, if none has changed.
    pub fn resolve_merge<P: DataProvider>(
        &self,
        session: &Session<P>,
        id: &str,
    ) -> Result<Vec<ClusterId>, String> {
        let c = self.comps.get(id).ok_or_else(|| Self::stale(id))?;
        match (c.members.iter().zip(&c.bases)).all(|(m, b)| spike_basis(session, *m) == *b) {
            true => Ok(c.members.clone()),
            false => Err(Self::stale(id)),
        }
    }

    /// One-line counterfactual of a (resolvable) handle.
    pub fn line<P: DataProvider>(
        &self,
        session: &Session<P>,
        id: &str,
        merges: &[MergeCandidate],
    ) -> Result<String, String> {
        if id.starts_with('s') {
            let (cluster, minor, _) = self.resolve_split(session, id)?;
            let h = &self.splits[id];
            let ev = eval_split(session, cluster, &minor);
            return Ok(render_split(id, h.method, h.score, h.bic, &ev));
        }
        let members = self.resolve_merge(session, id)?;
        let edges = merge_edges(merges, &members);
        Ok(render_merge(id, &eval_merge(session, &members), &edges))
    }

    /// Split handle (and its line) the latest rebuild issued for `cluster`.
    pub fn split_for<P: DataProvider>(
        &self,
        session: &Session<P>,
        cluster: ClusterId,
    ) -> Option<(String, String)> {
        let id = self
            .current
            .iter()
            .find(|id| self.splits.get(*id).is_some_and(|h| h.cluster == cluster))?;
        Some((id.clone(), self.line(session, id, &[]).ok()?))
    }

    /// Relations issued by the latest rebuild, components then splits, only
    /// those touching `filter` when given.
    pub fn relations<P: DataProvider>(
        &self,
        session: &Session<P>,
        merges: &[MergeCandidate],
        filter: Option<&[u32]>,
    ) -> Vec<Relation> {
        let touches = |cs: &[u32]| filter.is_none_or(|f| cs.iter().any(|c| f.contains(c)));
        let mut out = Vec::new();
        for id in &self.current {
            let clusters: Vec<u32> = match (self.comps.get(id), self.splits.get(id)) {
                (Some(c), _) => c.members.iter().map(|c| c.0).collect(),
                (_, Some(h)) => vec![h.cluster.0],
                _ => continue,
            };
            if !touches(&clusters) {
                continue;
            }
            if let Ok(line) = self.line(session, id, merges) {
                out.push(Relation {
                    handle: id.clone(),
                    clusters,
                    line,
                });
            }
        }
        // components (m) before splits (s), each by number
        out.sort_by_key(|r| {
            let n = r.handle[1..]
                .split('.')
                .map(|x| x.parse::<u32>().unwrap_or(0))
                .collect::<Vec<_>>();
            (r.handle.starts_with('s'), n)
        });
        out
    }
}

/// The candidate edges with both ends in `members`.
pub fn merge_edges(merges: &[MergeCandidate], members: &[ClusterId]) -> Vec<MergeCandidate> {
    merges
        .iter()
        .filter(|m| members.contains(&m.a) && members.contains(&m.b))
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 30_000.0;

    fn rng(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Poisson train (~19 Hz) with a 2 ms absolute refractory period.
    fn train(seed: u64, n: usize) -> (Vec<SampleIndex>, Vec<f32>) {
        let mut s = seed;
        let mut t = 0u64;
        let times = (0..n)
            .map(|_| {
                t += 60 + (-rng(&mut s).max(1e-6).ln() * 1500.0) as u64;
                SampleIndex(t)
            })
            .collect();
        (times, (0..n).map(|_| 1.0 + rng(&mut s) * 0.1).collect())
    }

    fn halves(t: &[SampleIndex], a: &[f32], rem: usize) -> (Vec<SampleIndex>, Vec<f32>) {
        let keep = |i: &usize| i % 2 == rem;
        (
            (0..t.len()).filter(keep).map(|i| t[i]).collect(),
            (0..t.len()).filter(keep).map(|i| a[i]).collect(),
        )
    }

    #[test]
    fn halves_of_one_refractory_train_merge_clean_with_a_dip() {
        let (t, a) = train(0x9e37_79b9_7f4a_7c15, 11_000);
        let total = t.last().unwrap().0 + 1000;
        let (h0, h1) = (halves(&t, &a, 0), halves(&t, &a, 1));
        let ev = eval_trains(&[(&h0.0, &h0.1), (&h1.0, &h1.1)], SR, total);
        assert!(ev.merged.rp_contamination < 0.05, "{ev:?}");
        assert!(ev.pairs[0].dip_z < -5.0, "{ev:?}");
        assert_eq!(ev.pairs[0].sync_z, 0.0);
    }

    #[test]
    fn components_never_exceed_the_cap_on_a_chain() {
        // 1-2-3-...-12 chain, every edge strong, plus a weak pair left out.
        let mut e: Vec<(u32, u32, f32)> =
            (1..12).map(|i| (i, i + 1, 0.9 - i as f32 * 0.01)).collect();
        e.push((20, 21, 0.3));
        let comps = components(&e);
        assert!(
            comps.iter().all(|c| (2..=COMPONENT_MAX).contains(&c.len())),
            "{comps:?}"
        );
        assert!(
            comps.iter().flatten().all(|&u| u <= 12),
            "weak edge joined: {comps:?}"
        );
        let mut seen: Vec<u32> = comps.concat();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), comps.concat().len(), "units in two components");
        assert_eq!(comps[0], vec![1, 2, 3, 4]);
    }

    #[test]
    fn sparse_ccg_gives_nan_not_99() {
        let a: Vec<SampleIndex> = (0..20).map(|i| SampleIndex(i * 90_000)).collect();
        let b: Vec<SampleIndex> = (0..20).map(|i| SampleIndex(i * 90_000 + 50_000)).collect();
        let (d, s) = ccg_z(&a, &b, SR);
        assert!(d.is_nan() && s.is_nan(), "{d} {s}");
    }

    #[test]
    fn independent_trains_show_no_dip() {
        let (t0, a0) = train(0x1234_5678_9abc_def1, 11_000);
        let (t1, a1) = train(0xdead_beef_cafe_f00d, 11_000);
        let total = t0.last().unwrap().0.max(t1.last().unwrap().0) + 1000;
        let ev = eval_trains(&[(&t0, &a0), (&t1, &a1)], SR, total);
        assert!(ev.pairs[0].dip_z > -3.0, "{ev:?}");
        let line = render_merge(
            "merge",
            &MergeEval {
                ids: vec![4, 9],
                ..ev
            },
            &[],
        );
        assert!(
            line.starts_with("merge {4,9}: rpc ") && line.contains("| 4-9 dip_z"),
            "{line}"
        );
    }

    #[test]
    fn split_of_one_refractory_train_shows_the_dip_between_parts() {
        let (t, a) = train(0x0bad_cafe_1234_5678, 11_000);
        let total = t.last().unwrap().0 + 1000;
        let minor: Vec<u32> = (0..t.len() as u32).filter(|i| i % 2 == 1).collect();
        let ev = eval_split_trains(&t, &a, &minor, SR, total);
        assert!(ev.dip_z < -5.0, "{ev:?}");
        assert_eq!((ev.parts[0].n, ev.parts[1].n), (5500, 5500));
        assert!(render_split("s1.1", "gmm", 0.62, 12.0, &ev)
            .starts_with("s1.1 gmm sc .62: parts 5500/5500"));
    }
}
