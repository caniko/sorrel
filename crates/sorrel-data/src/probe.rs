//! `probe`: a closed grammar of read-only computations for hypotheses the
//! precomputed brief did not anticipate (the `probe` MCP tool). No arbitrary
//! code: [`parse`] turns one query line into a [`Query`], [`run`] answers it
//! in one or a few compact, quantised lines. `split` registers a new split
//! handle (`s<C>.<k>`) so `split {candidate}`, `decide` and `eval` act on it
//! like on a ranker handle.

use crate::brief::ACG_EDGES;
use crate::cache::correlograms::{acg_or_compute, ccg_or_compute};
use crate::evaluate::{
    ccg_z, eval_merge, merge_edges, qm, qn, render_merge, unit_stats, Handles, UnitStats,
};
use crate::proposers::{cluster_feature_space, cluster_globals, kmeans_split};
use crate::session::Session;
use crate::suggest::MergeCandidate;
use sorrel_compute::{
    burst_index, gmm_split_proposal, log_bins, mean_amplitude, peak_channel, sax,
    sliding_refractory_argmin, waveform_landmarks,
};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};

pub const GRAMMAR: &str = "probe grammar (ms for acg/ccg, seconds for window/split; [..] optional):
  acg C [win=50] [bin=1]
  ccg A B [win=50] [bin=1]
  compare A B
  window C T0 T1
  split C by=amp|time|pc|kmeans [at=X]   (new handle; minor part = spikes above/after X)
  eval merge A B ..   |   eval s<C>.<k>   |   eval m<k>";

/// Most queries in one `probe` call.
pub const MAX_QUERIES: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum By {
    Amp,
    Time,
    Pc,
    Kmeans,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Query {
    Acg { c: u32, win: f32, bin: f32 },
    Ccg { a: u32, b: u32, win: f32, bin: f32 },
    Compare { a: u32, b: u32 },
    Window { c: u32, t0: f32, t1: f32 },
    Split { c: u32, by: By, at: Option<f32> },
    Merge(Vec<u32>),
    Handle(String),
}

/// One query line. Errors say what is wrong; the caller adds [`GRAMMAR`].
pub fn parse(q: &str) -> Result<Query, String> {
    let mut words = q.split_whitespace();
    let op = words.next().ok_or("empty query")?;
    let (mut pos, mut kv) = (Vec::new(), Vec::new());
    for w in words {
        match w.split_once('=') {
            Some(p) => kv.push(p),
            None => pos.push(w),
        }
    }
    let num = |s: &str, what: &str| {
        (s.parse::<f32>().ok().filter(|v| v.is_finite()))
            .ok_or(format!("bad number {s:?} for {what}"))
    };
    let id = |s: &str| {
        s.parse::<u32>()
            .map_err(|_| format!("bad cluster id {s:?}"))
    };
    let want = |n: usize, keys: &[&str]| {
        if pos.len() != n {
            return Err(format!("{op} takes {n} arguments, got {}", pos.len()));
        }
        match kv.iter().find(|(k, _)| !keys.contains(k)) {
            Some((k, _)) => Err(format!("unknown option {k:?} for {op}")),
            None => Ok(()),
        }
    };
    let opt = |key: &str| {
        (kv.iter().rev().find(|(k, _)| *k == key))
            .map(|(_, v)| num(v, key))
            .transpose()
    };
    let span = || -> Result<(f32, f32), String> {
        let (win, bin) = (opt("win")?.unwrap_or(50.0), opt("bin")?.unwrap_or(1.0));
        match win > 0.0 && bin > 0.0 && bin <= win {
            true => Ok((win, bin)),
            false => Err("need 0 < bin <= win".into()),
        }
    };
    match op {
        "acg" => {
            want(1, &["win", "bin"])?;
            let (win, bin) = span()?;
            Ok(Query::Acg {
                c: id(pos[0])?,
                win,
                bin,
            })
        }
        "ccg" => {
            want(2, &["win", "bin"])?;
            let (win, bin) = span()?;
            Ok(Query::Ccg {
                a: id(pos[0])?,
                b: id(pos[1])?,
                win,
                bin,
            })
        }
        "compare" => {
            want(2, &[])?;
            Ok(Query::Compare {
                a: id(pos[0])?,
                b: id(pos[1])?,
            })
        }
        "window" => {
            want(3, &[])?;
            let (t0, t1) = (num(pos[1], "T0")?, num(pos[2], "T1")?);
            match t1 > t0 && t0 >= 0.0 {
                true => Ok(Query::Window {
                    c: id(pos[0])?,
                    t0,
                    t1,
                }),
                false => Err("need 0 <= T0 < T1".into()),
            }
        }
        "split" => {
            want(1, &["by", "at"])?;
            let by = match kv.iter().find(|(k, _)| *k == "by").map(|p| p.1) {
                Some("amp") => By::Amp,
                Some("time") => By::Time,
                Some("pc") => By::Pc,
                Some("kmeans") => By::Kmeans,
                Some(o) => return Err(format!("unknown by={o}")),
                None => return Err("split needs by=amp|time|pc|kmeans".into()),
            };
            Ok(Query::Split {
                c: id(pos[0])?,
                by,
                at: opt("at")?,
            })
        }
        "eval" if pos.first() == Some(&"merge") => {
            want(pos.len(), &[])?;
            let ids = pos[1..]
                .iter()
                .map(|s| id(s))
                .collect::<Result<Vec<_>, _>>()?;
            match ids.len() >= 2 {
                true => Ok(Query::Merge(ids)),
                false => Err("eval merge needs at least two clusters".into()),
            }
        }
        "eval" => {
            want(1, &[])?;
            let h = pos[0];
            let ok = h.starts_with(['s', 'm']) && h.len() > 1;
            ok.then(|| Query::Handle(h.into()))
                .ok_or(format!("bad handle {h:?}"))
        }
        _ => Err(format!("unknown op {op:?}")),
    }
}

fn cl<P: DataProvider>(s: &Session<P>, c: u32) -> Result<ClusterId, String> {
    match c < s.n_clusters() && !s.spike_times(ClusterId(c)).is_empty() {
        true => Ok(ClusterId(c)),
        false => Err(format!("cluster {c} does not exist or has no spikes")),
    }
}

/// The `eval` of a merge of `ids` (the `eval {merge}` tool line).
pub fn merge_line<P: DataProvider>(
    session: &Session<P>,
    merges: &[MergeCandidate],
    ids: &[u32],
) -> Result<String, String> {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() < 2 {
        return Err("merge needs at least two clusters".into());
    }
    let ids = ids
        .into_iter()
        .map(|c| cl(session, c))
        .collect::<Result<Vec<_>, _>>()?;
    let ev = eval_merge(session, &ids);
    Ok(render_merge("merge", &ev, &merge_edges(merges, &ids)))
}

/// Answer `q`. `split` adds a handle to `handles`.
pub fn run<P: DataProvider>(
    session: &Session<P>,
    handles: &mut Handles,
    merges: &[MergeCandidate],
    positions: Option<&[[f32; 2]]>,
    q: &Query,
) -> Result<String, String> {
    let sr = session.provider.sample_rate().max(1.0);
    let total = session.provider.n_samples().0.max(1);
    let times = |c| session.spike_times(c);
    let samples = |ms: f32| (ms * sr / 1000.0).round() as u32;
    match q {
        Query::Acg { c, win, bin } => {
            let c = cl(session, *c)?;
            let (b, w) = (samples(*bin).max(1), samples(*win));
            let h = acg_or_compute(session, c, b, w);
            Ok(acg_text(
                c.0,
                times(c),
                sr,
                total,
                &h,
                b as f32 * 1000.0 / sr,
                *win,
            ))
        }
        Query::Ccg { a, b, win, bin } => {
            let (a, b) = (cl(session, *a)?, cl(session, *b)?);
            let (bs, w) = (samples(*bin).max(1), samples(*win));
            let h = ccg_or_compute(session, a, b, bs, w);
            Ok(ccg_text((a.0, b.0), (times(a), times(b)), sr, &h, *win))
        }
        Query::Compare { a, b } => {
            let (a, b) = (cl(session, *a)?, cl(session, *b)?);
            let tpl = |c| {
                session
                    .primary_template(c)
                    .and_then(|t| session.template_waveform(t))
            };
            let (_, ns, nc) = session.template_shape();
            let (wa, wb) = tpl(a)
                .zip(tpl(b))
                .ok_or("no template waveforms for these clusters")?;
            let r = compare_templates(wa, wb, ns, nc, positions, sr)
                .ok_or("flat or mis-sized template")?;
            Ok(format!("compare {} {}: {}", a.0, b.0, r.line()))
        }
        Query::Window { c, t0, t1 } => {
            let c = cl(session, *c)?;
            window_text(
                c.0,
                times(c),
                session.spike_amplitudes(c),
                sr,
                total,
                *t0,
                *t1,
            )
        }
        Query::Split { c, by, at } => {
            let c = cl(session, *c)?;
            let (minor, method, bic, used) = split_minor(session, c, *by, *at)?;
            let n = times(c).len();
            if minor.is_empty() || minor.len() >= n {
                return Err(format!(
                    "split leaves an empty part ({} of {n} spikes would move)",
                    minor.len()
                ));
            }
            let id = handles.add_split(session, c, minor, method, f32::NAN, bic);
            let line = handles.line(session, &id, &[])?;
            Ok(used.map_or(line.clone(), |x| format!("{line} [at {}]", qn(x))))
        }
        Query::Merge(ids) => merge_line(session, merges, ids),
        Query::Handle(h) => handles.line(session, h, merges),
    }
}

fn rates(counts: &[u32], n: usize, bin_ms: f32) -> String {
    let k = n as f32 * bin_ms / 1000.0;
    counts
        .iter()
        .map(|&c| qn(c as f32 / k))
        .collect::<Vec<_>>()
        .join(",")
}

/// ACG detail from the centred two-sided histogram `h`.
pub fn acg_text(
    c: u32,
    times: &[SampleIndex],
    sr: f32,
    total: u64,
    h: &[u32],
    bin_ms: f32,
    win_ms: f32,
) -> String {
    let (cont, rp) = sliding_refractory_argmin(
        times,
        (sr * 0.0005) as u64,
        (sr * 0.010) as u64,
        20,
        total,
        sr,
    );
    let rp = match rp {
        0 => "-".into(),
        r => format!("{:.1}ms", r as f32 * 1000.0 / sr),
    };
    let edges: Vec<f32> = ACG_EDGES.iter().copied().filter(|&e| e <= win_ms).collect();
    let letters = match edges.len() >= 3 && !h.is_empty() {
        true => sax(&log_bins(h, bin_ms, &edges), 12, 5),
        false => "-".into(),
    };
    let first = &h[(h.len() / 2).min(h.len())..][..6.min(h.len() - h.len() / 2)];
    format!(
        "acg {c} n {} win {win_ms}ms bin {bin_ms:.2}ms: rp {rp} rpc {} burst {}\nsax(log lag 0-{}ms) {letters}\nrate Hz bins 0..{}: {}",
        times.len(),
        qm(if rp == "-" { f32::NAN } else { cont }, "rp_contamination"),
        qn(burst_index(h, bin_ms)),
        edges.last().copied().unwrap_or(0.0),
        first.len(),
        rates(first, times.len(), bin_ms)
    )
}

/// CCG z-scores (always on the standard 1 ms, +-50 ms CCG), the +/- lag
/// asymmetry and the SAX of the centred histogram `h`. Lag = t_b - t_a.
pub fn ccg_text(
    ids: (u32, u32),
    trains: (&[SampleIndex], &[SampleIndex]),
    sr: f32,
    h: &[u32],
    win_ms: f32,
) -> String {
    let (dip, sync) = ccg_z(trains.0, trains.1, sr);
    let half = h.len() / 2;
    let (neg, pos) = (
        h[..half].iter().map(|&x| u64::from(x)).sum::<u64>(),
        h[half..].iter().map(|&x| u64::from(x)).sum::<u64>(),
    );
    let z = (pos as f32 - neg as f32) / ((pos + neg) as f32).sqrt();
    let letters = sax(&h.iter().map(|&x| x as f32).collect::<Vec<_>>(), 12, 5);
    format!(
        "ccg {} {} n {}/{} win {win_ms}ms: dip_z {} sync_z {} asym +{pos}/-{neg} z {}\nsax(+-{win_ms}ms) {letters}",
        ids.0, ids.1, trains.0.len(), trains.1.len(), qn(dip), qn(sync), qn(z)
    )
}

/// Slice `[t0, t1)` s of a train against the whole unit.
pub fn window_text(
    c: u32,
    times: &[SampleIndex],
    amps: &[f32],
    sr: f32,
    total: u64,
    t0: f32,
    t1: f32,
) -> Result<String, String> {
    let (lo, hi) = ((t0 * sr) as u64, ((t1 * sr) as u64).min(total));
    if lo >= hi {
        return Err(format!(
            "window is outside the recording ({:.1}s)",
            total as f32 / sr
        ));
    }
    let (a, b) = (
        times.partition_point(|t| t.0 < lo),
        times.partition_point(|t| t.0 < hi),
    );
    let shifted: Vec<SampleIndex> = times[a..b].iter().map(|t| SampleIndex(t.0 - lo)).collect();
    let has = amps.len() == times.len();
    let part_amps = if has { &amps[a..b] } else { &[] };
    let (part, whole) = (
        unit_stats(&shifted, part_amps, sr, hi - lo),
        unit_stats(times, if has { amps } else { &[] }, sr, total),
    );
    let mean = |v: &[f32]| {
        if v.is_empty() {
            f32::NAN
        } else {
            mean_amplitude(v)
        }
    };
    let pair = |x: f32, y: f32| format!("{}/{}", qn(x), qn(y));
    let both = |f: fn(&UnitStats) -> f32| pair(f(&part), f(&whole));
    Ok(format!(
        "window {c} [{},{}]s (slice/whole): n {}/{} rate {} Hz rpc {} amp {} pres {}",
        qn(t0),
        qn(t1),
        b - a,
        times.len(),
        pair(
            (b - a) as f32 / ((hi - lo) as f32 / sr),
            times.len() as f32 / (total as f32 / sr)
        ),
        both(|s| s.rp_contamination),
        pair(mean(part_amps), mean(amps)),
        both(|s| s.presence_ratio),
    ))
}

pub struct Compare {
    pub cosine: f32,
    pub peak_ch: (usize, usize),
    pub dist_um: Option<f32>,
    pub ttp_ms: (f32, f32),
    pub half_width_ms: (f32, f32),
    pub amp_ratio: f32,
    pub shared_ch: usize,
}

impl Compare {
    fn line(&self) -> String {
        let d = |p: (f32, f32)| qn(p.1 - p.0);
        format!(
            "cos {} peak_ch {}/{} dist {}um dttp {}ms dhw {}ms amp x{} ch {}",
            qn(self.cosine),
            self.peak_ch.0,
            self.peak_ch.1,
            self.dist_um.map_or("-".into(), qn),
            d(self.ttp_ms),
            d(self.half_width_ms),
            qn(self.amp_ratio),
            self.shared_ch
        )
    }
}

/// Peak-to-peak per channel of a row-major `(ns, nc)` template.
fn ptp(w: &[f32], ns: usize, nc: usize) -> Vec<f32> {
    (0..nc)
        .map(|c| {
            let (lo, hi) = (0..ns)
                .map(|s| w[s * nc + c])
                .fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)));
            hi - lo
        })
        .collect()
}

/// Shape similarity of two row-major templates: cosine over the union of
/// their active channels (ptp >= 20% of own max), peak-channel distance,
/// landmark deltas and amplitude ratio (b over a). `None` for flat or
/// mis-sized templates.
pub fn compare_templates(
    wa: &[f32],
    wb: &[f32],
    ns: usize,
    nc: usize,
    positions: Option<&[[f32; 2]]>,
    sr: f32,
) -> Option<Compare> {
    if ns == 0 || nc == 0 || wa.len() != ns * nc || wb.len() != ns * nc {
        return None;
    }
    let (pa, pb) = (ptp(wa, ns, nc), ptp(wb, ns, nc));
    let max = |p: &[f32]| p.iter().copied().fold(0.0, f32::max);
    let (ma, mb) = (max(&pa), max(&pb));
    let (ka, kb) = (peak_channel(wa, ns, nc)?, peak_channel(wb, ns, nc)?);
    let active: Vec<usize> = (0..nc)
        .filter(|&c| pa[c] >= 0.2 * ma || pb[c] >= 0.2 * mb)
        .collect();
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for s in 0..ns {
        for &c in &active {
            let (x, y) = (f64::from(wa[s * nc + c]), f64::from(wb[s * nc + c]));
            (dot, na, nb) = (dot + x * y, na + x * x, nb + y * y);
        }
    }
    let lm = |w: &[f32], k: usize| {
        let trace: Vec<f32> = (0..ns).map(|s| w[s * nc + k]).collect();
        waveform_landmarks(&trace, sr)
    };
    let nan = f32::NAN;
    let (la, lb) = (lm(wa, ka), lm(wb, kb));
    let get =
        |l: &Option<_>, f: fn(&sorrel_compute::WaveformLandmarks) -> f32| l.as_ref().map_or(nan, f);
    Some(Compare {
        cosine: (dot / (na.sqrt() * nb.sqrt())) as f32,
        peak_ch: (ka, kb),
        dist_um: positions.filter(|p| p.len() == nc).map(|p| {
            let (u, v) = (p[ka], p[kb]);
            (u[0] - v[0]).hypot(u[1] - v[1])
        }),
        ttp_ms: (
            get(&la, |l| l.trough_to_peak_ms),
            get(&lb, |l| l.trough_to_peak_ms),
        ),
        half_width_ms: (get(&la, |l| l.half_width_ms), get(&lb, |l| l.half_width_ms)),
        amp_ratio: mb / ma,
        shared_ch: active.len(),
    })
}

/// First principal axis of row-major whitened `feats` (`d` columns) by power
/// iteration (d <= 9, so 50 sweeps are exact enough).
fn pc1(feats: &[f32], d: usize) -> Vec<f32> {
    let mut cov = vec![0.0f32; d * d];
    for r in feats.chunks_exact(d) {
        for i in 0..d {
            for j in 0..d {
                cov[i * d + j] += r[i] * r[j];
            }
        }
    }
    let mut v: Vec<f32> = (0..d).map(|i| 1.0 + 0.1 * i as f32).collect();
    for _ in 0..50 {
        let w: Vec<f32> = (0..d)
            .map(|i| (0..d).map(|j| cov[i * d + j] * v[j]).sum())
            .collect();
        let norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-12 {
            break;
        }
        v = w.iter().map(|x| x / norm).collect();
    }
    v
}

/// Local indices of spikes with `key > at`, ascending.
fn above(keys: &[f32], at: f32) -> Vec<u32> {
    (0..keys.len() as u32)
        .filter(|&i| keys[i as usize] > at)
        .collect()
}

/// `(minor local spike positions, handle method, bic, threshold used)`.
/// ponytail: by=time without at= scans 20 coarse bin edges (each side >= 5% of
/// the spikes, at least 10) for the largest mean-amplitude jump; a change-point
/// test would be sharper.
#[allow(clippy::type_complexity)]
fn split_minor<P: DataProvider>(
    session: &Session<P>,
    c: ClusterId,
    by: By,
    at: Option<f32>,
) -> Result<(Vec<u32>, &'static str, f32, Option<f32>), String> {
    let (times, amps) = (session.spike_times(c), session.spike_amplitudes(c));
    let sr = session.provider.sample_rate().max(1.0);
    let n = times.len();
    let need_amps = || {
        (amps.len() == n)
            .then_some(())
            .ok_or("cluster has no amplitudes")
    };
    match by {
        By::Amp => {
            need_amps()?;
            match at {
                Some(x) => Ok((above(amps, x), "amp", f32::NAN, None)),
                None => {
                    let p = gmm_split_proposal(amps)
                        .ok_or("no amplitude mixture found (unimodal); give at=")?;
                    Ok((p.minor_spike_idx, "gmm", p.bic_delta, None))
                }
            }
        }
        By::Time => {
            let secs: Vec<f32> = times.iter().map(|t| t.0 as f32 / sr).collect();
            let x = match at {
                Some(x) => x,
                None => {
                    need_amps()?;
                    let (first, last) = (secs[0], secs[n - 1]);
                    let min = (n / 20).max(10);
                    let mut pre = vec![0.0f64];
                    pre.extend(amps.iter().scan(0.0f64, |s, &a| {
                        *s += f64::from(a);
                        Some(*s)
                    }));
                    (1..20)
                        .map(|b| first + (last - first) * b as f32 / 20.0)
                        .filter_map(|x| {
                            let k = secs.partition_point(|&t| t <= x);
                            if k < min || n - k < min {
                                return None;
                            }
                            let gap =
                                (pre[k] / k as f64 - (pre[n] - pre[k]) / (n - k) as f64).abs();
                            Some((x, gap))
                        })
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|p| p.0)
                        .ok_or("too few spikes for a time scan; give at=")?
                }
            };
            Ok((above(&secs, x), "time", f32::NAN, Some(x)))
        }
        By::Pc | By::Kmeans => {
            let globals = cluster_globals(session, c);
            if by == By::Kmeans {
                let km = kmeans_split(session, c, &globals)
                    .ok_or("no k-means split (needs PC features, silhouette >= .25, d' >= 2)")?;
                return Ok((km.minor_idx, "kmeans", f32::NAN, None));
            }
            let (feats, d) = cluster_feature_space(session, c, &globals).ok_or("no PC features")?;
            let axis = pc1(&feats, d);
            let mut proj: Vec<f32> = feats
                .chunks_exact(d)
                .map(|r| r.iter().zip(&axis).map(|(x, a)| x * a).sum())
                .collect();
            let x = at.unwrap_or(0.0);
            // the axis sign is arbitrary: without at= make the minor side the smaller one
            if at.is_none() && above(&proj, 0.0).len() * 2 > n {
                proj.iter_mut().for_each(|p| *p = -*p);
            }
            Ok((above(&proj, x), "pc", f32::NAN, Some(x)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_compute::{auto_correlogram, cross_correlogram};

    const SR: f32 = 30_000.0;

    /// Poisson train (~19 Hz) with a 2 ms absolute refractory period.
    fn train(n: usize) -> Vec<SampleIndex> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut t = 0u64;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 40) as f32 / (1u64 << 24) as f32).max(1e-6);
                t += 60 + (-u.ln() * 1500.0) as u64;
                SampleIndex(t)
            })
            .collect()
    }

    /// The number after `key ` in `text`; NaN for `-`.
    fn field(text: &str, key: &str) -> f32 {
        let rest = text.split(&format!("{key} ")).nth(1).unwrap();
        rest.split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap_or(f32::NAN)
    }

    #[test]
    fn parser_accepts_and_rejects() {
        use Query::*;
        assert_eq!(
            parse("acg 3"),
            Ok(Acg {
                c: 3,
                win: 50.0,
                bin: 1.0
            })
        );
        assert_eq!(
            parse("ccg 1 2 win=20 bin=0.5"),
            Ok(Ccg {
                a: 1,
                b: 2,
                win: 20.0,
                bin: 0.5
            })
        );
        assert_eq!(parse("compare 81 84"), Ok(Compare { a: 81, b: 84 }));
        assert_eq!(
            parse("window 5 0 100.5"),
            Ok(Window {
                c: 5,
                t0: 0.0,
                t1: 100.5
            })
        );
        assert_eq!(
            parse("split 7 by=time at=12"),
            Ok(Split {
                c: 7,
                by: By::Time,
                at: Some(12.0)
            })
        );
        assert_eq!(parse("eval merge 1 2 3"), Ok(Merge(vec![1, 2, 3])));
        assert_eq!(parse("eval s12.1"), Ok(Handle("s12.1".into())));
        for bad in [
            "",
            "frob 1",
            "acg",
            "acg x",
            "acg 1 2",
            "acg 1 win=abc",
            "acg 1 foo=2",
            "acg 1 win=5 bin=9",
            "ccg 1",
            "window 1 5 2",
            "split 1",
            "split 1 by=magic",
            "eval merge 4",
            "eval 12",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn acg_of_a_refractory_train_has_no_contamination() {
        let t = train(5_000);
        let total = t.last().unwrap().0 + 1000;
        let h = auto_correlogram(&t, 1500, 100);
        let text = acg_text(3, &t, SR, total, &h, 1.0 * 30.0 / 30.0, 50.0);
        assert!(field(&text, "rpc") < 0.05, "{text}");
        assert!(
            text.contains("\nsax(log lag 0-50ms) ") && text.contains("\nrate Hz bins 0..6: "),
            "{text}"
        );
    }

    #[test]
    fn ccg_of_two_halves_of_one_train_shows_a_dip() {
        let t = train(11_000);
        let (a, b): (Vec<_>, Vec<_>) = (
            t.iter().step_by(2).copied().collect(),
            t.iter().skip(1).step_by(2).copied().collect(),
        );
        let h = cross_correlogram(&a, &b, 1500, 100);
        let text = ccg_text((1, 2), (&a, &b), SR, &h, 50.0);
        assert!(field(&text, "dip_z") < 0.0, "{text}");
    }

    #[test]
    fn a_template_is_identical_to_itself() {
        let (ns, nc) = (20, 4);
        let w: Vec<f32> = (0..ns * nc)
            .map(|i| {
                let (s, c) = (i / nc, i % nc);
                -((s as f32 - 8.0) / 3.0).powi(2).exp().recip() * (4 - c) as f32
            })
            .collect();
        let pos = [[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]];
        let r = compare_templates(&w, &w, ns, nc, Some(&pos), SR).unwrap();
        assert!(
            (r.cosine - 1.0).abs() < 1e-5 && r.dist_um == Some(0.0) && r.amp_ratio == 1.0,
            "{}",
            r.line()
        );
        assert!(compare_templates(&w, &w[1..], ns, nc, None, SR).is_none());
    }

    #[test]
    fn window_compares_slice_with_whole() {
        let t = train(2_000);
        let total = t.last().unwrap().0 + 1000;
        let amps = vec![2.0; t.len()];
        let text = window_text(1, &t, &amps, SR, total, 0.0, 10.0).unwrap();
        assert!(
            text.starts_with("window 1 [0,10]s (slice/whole): n "),
            "{text}"
        );
        assert!(window_text(1, &t, &amps, SR, total, 1e6, 2e6).is_err());
    }
}
