//! `brief`: a token-lean text encoding of a spike sort for an LLM reviewer,
//! the `brief` MCP tool (`sorrel call DIR brief`). Levels are progressive: L0 header
//! (population quantiles + flag legend), L1 ledger (flagged units, the rest
//! run-length coded), L2 unit cards (every metric plus SAX strings).
//!
//! Decision-lossless: every number is quantised by [`quantise`], which widens
//! the precision until it sits on the same side of every flag threshold as
//! the raw value, so a flag evaluated on what the reader sees equals the flag
//! on the raw metric. Output is a pure function of the session (sorted ids,
//! fixed formatting).

use crate::cache::correlograms::acg_or_compute;
use crate::evaluate::Relation;
use crate::qc_export::QcRow;
use crate::rules::cond_holds;
use crate::serve::{load_evidence, unit_evidence};
use crate::session::Session;
use crate::suggest::{MergeCandidate, SplitCandidate};
use serde_json::Value;
use sorrel_compute::{log_bins, sax};
use sorrel_io::{ClusterId, DataProvider};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::Path;

/// A named predicate over one metric. An `op` starting with `|` compares the
/// absolute value (`"|>"` is `|metric| > value`). `NaN` never fires.
pub struct Flag {
    pub code: &'static str,
    pub metric: &'static str,
    pub op: &'static str,
    pub value: f32,
    pub meaning: &'static str,
}

const fn flag(
    code: &'static str,
    metric: &'static str,
    op: &'static str,
    value: f32,
    meaning: &'static str,
) -> Flag {
    Flag {
        code,
        metric,
        op,
        value,
        meaning,
    }
}

/// Default thresholds, declared in every header so the reader sees them.
/// `isi_violation_ratio` is violations per second of spike-train span (not a
/// fraction), so 0.5/s is a unit that breaks its 1.5 ms refractory period
/// about every other second. `amplitude_dip` is Hartigan's dip in [0, 0.25]
/// on <= 5000 subsampled amplitudes: noise sits near 0.01, 0.05 is a clear
/// second mode. `amplitude_snr` is the mean/sd of per-spike amplitude
/// scalings (not a waveform SNR); 2.0 is where that ratio stops separating
/// units on real sorts (3.0 fired on most of them).
pub static FLAGS: &[Flag] = &[
    flag(
        "RPV",
        "rp_contamination",
        ">",
        0.1,
        "refractory-period contamination",
    ),
    flag(
        "ISI",
        "isi_violation_ratio",
        ">",
        0.5,
        "ISI violations per second",
    ),
    flag(
        "CUT",
        "amplitude_cutoff",
        ">",
        0.1,
        "amplitudes cut off by the detection threshold",
    ),
    flag(
        "PRES",
        "presence_ratio",
        "<",
        0.9,
        "absent from part of the recording",
    ),
    flag(
        "GAP",
        "longest_silent_gap",
        ">",
        0.2,
        "silent gap, fraction of recording",
    ),
    flag(
        "DRF",
        "drift_correlation",
        "|>",
        0.5,
        "amplitude drifts with time",
    ),
    flag(
        "LOWSNR",
        "amplitude_snr",
        "<",
        2.0,
        "low amplitude mean/sd (per-spike scalings, not waveform SNR)",
    ),
    flag("FEW", "n_spikes", "<", 300.0, "too few spikes to judge"),
    flag("MPK", "n_extrema", ">=", 3.0, "multi-peak waveform"),
    flag(
        "NARROW",
        "trough_to_peak_ms",
        "<",
        0.15,
        "trough-to-peak narrower than a spike",
    ),
    flag(
        "LONG",
        "trough_to_peak_ms",
        ">",
        1.5,
        "trough-to-peak too slow for a soma",
    ),
    flag(
        "1CH",
        "n_active_channels",
        "<=",
        1.0,
        "footprint on a single channel",
    ),
    flag(
        "WIDE",
        "footprint_r80_um",
        ">",
        120.0,
        "80% energy radius, um",
    ),
    flag(
        "AXON",
        "propagation_ms_per_100um",
        "|>",
        0.25,
        "waveform propagates along the probe",
    ),
    flag(
        "BIMOD",
        "amplitude_dip",
        ">",
        0.05,
        "bimodal amplitudes (Hartigan dip)",
    ),
    flag(
        "ISO",
        "l_ratio",
        ">",
        0.1,
        "poor PC-space isolation (L-ratio)",
    ),
];

/// Non-metric flags: (code, meaning).
const EXTRA_FLAGS: [(&str, &str); 3] = [
    ("MERGE?", "in a merge candidate (merge_candidates)"),
    ("SPLIT?", "has a split candidate (split_proposals)"),
    ("REV", "tagged review:* (waiting for a decision)"),
];

/// Ledger shows units with a flag or composite quality below this.
const Q_CLEAN: f32 = 0.7;
/// ACG lag edges (ms): whole ms (the ACG has 1 ms bins), finer near the
/// refractory period; see the header for the reader's view.
pub(crate) const ACG_EDGES: [f32; 13] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0, 20.0, 30.0, 40.0, 50.0,
];
const AMP_BINS: usize = 10;
const ALPHABET: u8 = 5;

/// Every threshold a reader may compare `metric` against: flag values (and
/// their negatives for `|` ops) and the ledger's quality cutoff.
pub fn thresholds(metric: &str) -> Vec<f32> {
    thresholds_with(metric, &[])
}

/// [`thresholds`] plus `extra` (metric, threshold) pairs, e.g. a rule set's
/// [`crate::rules::RuleSet::thresholds`], so rule decisions survive quantising.
pub fn thresholds_with(metric: &str, extra: &[(String, f32)]) -> Vec<f32> {
    let mut t: Vec<f32> = FLAGS
        .iter()
        .filter(|f| f.metric == metric)
        .flat_map(|f| match f.op.starts_with('|') {
            true => vec![f.value, -f.value],
            false => vec![f.value],
        })
        .collect();
    if metric == "composite_quality" {
        t.push(Q_CLEAN);
    }
    t.extend(extra.iter().filter(|e| e.0 == metric).map(|e| e.1));
    t
}

/// Codes of the metric flags that fire; `get` yields a metric by name.
pub fn metric_flags(get: impl Fn(&str) -> f32) -> Vec<&'static str> {
    FLAGS
        .iter()
        .filter(|f| cond_holds(f.op, get(f.metric), f.value))
        .map(|f| f.code)
        .collect()
}

/// Mode-relative flag coding (MDL): the flags firing on more than half of
/// the units, with their counts, in `all` order. Rows then carry only a
/// [`delta`], which is lossless because the header states the mode.
pub fn mode_of(sets: &[Vec<&'static str>]) -> Vec<(&'static str, usize)> {
    let mut n: BTreeMap<&str, usize> = BTreeMap::new();
    sets.iter()
        .flatten()
        .for_each(|c| *n.entry(c).or_default() += 1);
    FLAGS
        .iter()
        .map(|f| f.code)
        .chain(EXTRA_FLAGS.iter().map(|e| e.0))
        .filter_map(|c| n.get(c).map(|&k| (c, k)))
        .filter(|&(_, k)| 2 * k > sets.len())
        .collect()
}

/// `flags` against the mode: `+X` fires but is not in the mode, `-X` is in
/// the mode but does not fire, `=` for no difference.
pub fn delta(flags: &[&str], mode: &[(&str, usize)]) -> String {
    let plus = flags
        .iter()
        .filter(|f| !mode.iter().any(|m| m.0 == **f))
        .map(|f| format!("+{f}"));
    let minus = mode
        .iter()
        .filter(|m| !flags.contains(&m.0))
        .map(|m| format!("-{}", m.0));
    let d: Vec<String> = plus.chain(minus).collect();
    match d.is_empty() {
        true => "=".into(),
        false => d.join(","),
    }
}

fn plain(r: f32) -> String {
    if r != 0.0 && r.abs() < 1e-3 {
        format!("{r:e}")
    } else {
        format!("{r}")
    }
}

/// `v` at 2 significant figures, widened (up to 7, then exact) until its
/// printed value orders against every threshold exactly as `v` does.
/// `NaN` is `-`. Integers round like everything else (667 -> 670), so the
/// printed order of values always matches their true order.
pub fn quantise(v: f32, thresholds: &[f32]) -> String {
    if v.is_nan() {
        return "-".into();
    }
    if !v.is_finite() {
        return v.to_string();
    }
    for sig in 2..=7 {
        let s = plain(format!("{:.*e}", sig - 1, v).parse().unwrap_or(v));
        if let Ok(p) = s.parse::<f32>() {
            if thresholds
                .iter()
                .all(|t| p.partial_cmp(t) == v.partial_cmp(t))
            {
                return s;
            }
        }
    }
    plain(v)
}

/// Ids as `1-3,7,9-10`; `ids` ascending and unique.
pub fn rle(ids: &[u32]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < ids.len() {
        let mut j = i;
        while j + 1 < ids.len() && ids[j + 1] == ids[j] + 1 {
            j += 1;
        }
        if !out.is_empty() {
            out.push(',');
        }
        let _ = match j > i {
            true => write!(out, "{}-{}", ids[i], ids[j]),
            false => write!(out, "{}", ids[i]),
        };
        i = j + 1;
    }
    out
}

pub struct BriefOpts {
    /// 0 header, 1 + ledger, 2 + cards for flagged units, 3 + relations
    /// (merge components and split candidates with counterfactuals).
    pub level: u8,
    /// Cards for exactly these units (any level).
    pub clusters: Option<Vec<u32>>,
    /// Estimated tokens (bytes / 3.5); `None` = unbounded.
    pub budget_tokens: Option<usize>,
    /// Further (metric, threshold) pairs the quantiser must not cross,
    /// normally the active rule set's.
    pub extra: Vec<(String, f32)>,
}

impl Default for BriefOpts {
    fn default() -> Self {
        Self {
            level: 1,
            clusters: None,
            budget_tokens: None,
            extra: Vec::new(),
        }
    }
}

struct Unit<'a> {
    id: u32,
    row: &'a QcRow,
    flags: Vec<&'static str>,
    label: &'static str,
}

fn header<P: DataProvider>(
    session: &Session<P>,
    units: &[Unit],
    mode: &[(&'static str, usize)],
    extra: &[(String, f32)],
) -> String {
    let sr = session.provider.sample_rate();
    let mut h = format!(
        "# sorrel brief v1\nrec: {:.1} s, {} Hz, {} ch, {} live clusters\n\
         amp: template units (uV only if the input says so); a missing key is NaN\n",
        session.provider.n_samples().0 as f32 / sr.max(1.0),
        sr,
        session.provider.n_channels(),
        units.len()
    );
    let mut labels: BTreeMap<&str, usize> = BTreeMap::new();
    for u in units {
        *labels.entry(u.label).or_default() += 1;
    }
    let labels: Vec<String> = labels.iter().map(|(l, n)| format!("{l}={n}")).collect();
    let _ = writeln!(h, "labels: {}", labels.join(" "));
    let m: Vec<String> = mode
        .iter()
        .map(|(c, k)| format!("{c} {k}/{}", units.len()))
        .collect();
    let _ = writeln!(
        h,
        "mode: {}",
        if m.is_empty() {
            "-".into()
        } else {
            m.join(",")
        }
    );
    h.push_str("pop (p10/p50/p90 over live units):\n");
    for m in QcRow::METRICS {
        let th = thresholds_with(m, extra);
        let mut v: Vec<f32> = units
            .iter()
            .filter_map(|u| u.row.get(m))
            .filter(|v| v.is_finite())
            .collect();
        if v.is_empty() {
            continue; // nothing measured (e.g. no templates): no row
        }
        v.sort_by(f32::total_cmp);
        let q = |p: f32| quantise(v[((v.len() - 1) as f32 * p).round() as usize], &th);
        let _ = writeln!(h, "{m} {}/{}/{}", q(0.1), q(0.5), q(0.9));
    }
    h.push_str("flags (fire when true, NaN never):\n");
    for f in FLAGS {
        let _ = match f.op.strip_prefix('|') {
            Some(op) => writeln!(
                h,
                "{}: |{}| {op} {} - {}",
                f.code, f.metric, f.value, f.meaning
            ),
            None => writeln!(
                h,
                "{}: {} {} {} - {}",
                f.code, f.metric, f.op, f.value, f.meaning
            ),
        };
    }
    for (c, m) in EXTRA_FLAGS {
        let _ = writeln!(h, "{c}: {m}");
    }
    let _ =
        writeln!(
        h,
        "quantise: 2 significant figures, widened until every comparison with a flag threshold \
         (and q vs {Q_CLEAN}) matches the raw value\n\
         ledger: id n fr lab q flags; units with a flag or q < {Q_CLEAN}; clean(N) lists the rest \
         (no flag, q >= {Q_CLEAN})\n\
         flags in rows and cards are a delta vs `mode`, the flags firing on >50% of units: \
         +X fires and is not in the mode, -X is in the mode but does not fire, = same as the mode\n\
         cards: one row per unit, columns named once; acg = {} SAX symbols of the ACG rate over lags 0,{}..50 ms; \
         amp_t = {AMP_BINS} SAX symbols of mean amplitude in equal time bins, _ = no spikes; \
         ev = bc bombcell, ur unitrefine/prob, llm label/confidence\n\
         SAX: z-normalised per series, a..e = lowest..highest; compare shape, not level\n\
         drill: brief{{clusters:[..]}} cards; metrics, evidence, merge_candidates, \
         split_proposals, render(SVG) take {{cluster}}",
        ACG_EDGES.len() - 1,
        ACG_EDGES[1..12].iter().map(f32::to_string).collect::<Vec<_>>().join(",")
    );
    h
}

fn ledger_line(u: &Unit, mode: &[(&str, usize)], extra: &[(String, f32)]) -> String {
    let r = u.row;
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\n",
        u.id,
        r.n_spikes,
        quantise(r.firing_rate_hz, &[]),
        u.label,
        quantise(
            r.composite_quality,
            &thresholds_with("composite_quality", extra)
        ),
        delta(&u.flags, mode)
    )
}

/// The card table's header: fixed columns, then `cols`, then the series.
fn card_head(cols: &[&str]) -> String {
    format!(
        "cards:\nid\tlab\tflags\ttags\t{}\tacg\tamp_t\tev\n",
        cols.join("\t")
    )
}

/// One card row: the columns of [`card_head`], `-` for missing values.
fn card<P: DataProvider>(
    session: &Session<P>,
    u: &Unit,
    mode: &[(&str, usize)],
    cols: &[&str],
    evidence: Option<&Value>,
    extra: &[(String, f32)],
) -> String {
    let c = ClusterId(u.id);
    let tags = session.tags(c);
    let mut s = format!(
        "{}\t{}\t{}\t{}",
        u.id,
        u.label,
        delta(&u.flags, mode),
        if tags.is_empty() {
            "-".into()
        } else {
            tags.join(",")
        }
    );
    for m in cols {
        let v = u.row.get(m).unwrap_or(f32::NAN);
        let _ = write!(s, "\t{}", quantise(v, &thresholds_with(m, extra)));
    }
    let sr = session.provider.sample_rate().max(1.0);
    let bin = ((0.001 * sr) as u32).max(1);
    let acg = acg_or_compute(session, c, bin, (0.050 * sr) as u32);
    let _ = write!(
        s,
        "\t{}",
        sax(
            &log_bins(&acg, bin as f32 * 1000.0 / sr, &ACG_EDGES),
            12,
            ALPHABET
        )
    );
    let (times, amps) = (session.spike_times(c), session.spike_amplitudes(c));
    let total = session.provider.n_samples().0.max(1);
    let mut amp_t = "-".to_string();
    if amps.len() == times.len() && !times.is_empty() {
        let mut acc = [(0.0f32, 0u32); AMP_BINS];
        for (t, &a) in times.iter().zip(amps) {
            let b = ((t.0 as u128 * AMP_BINS as u128 / total as u128) as usize).min(AMP_BINS - 1);
            acc[b] = (acc[b].0 + a, acc[b].1 + 1);
        }
        let mean = amps.iter().sum::<f32>() / amps.len() as f32;
        let series: Vec<f32> = acc
            .iter()
            .map(|&(sum, n)| if n > 0 { sum / n as f32 } else { mean })
            .collect();
        amp_t = sax(&series, AMP_BINS, ALPHABET)
            .chars()
            .zip(acc)
            .map(|(ch, (_, n))| if n > 0 { ch } else { '_' })
            .collect();
    }
    let _ = write!(s, "\t{amp_t}");
    let mut ev = Vec::new();
    if let Some(unit) =
        evidence.and_then(|d| unit_evidence(d, u64::from(session.provider.unit_id(c))).0)
    {
        if let Some(b) = unit["bombcell"].as_str() {
            ev.push(format!("bc:{b}"));
        }
        if let Some(l) = unit["unitrefine"].as_str() {
            let p = unit["unitrefine_probability"]
                .as_f64()
                .map(|p| format!("/{}", quantise(p as f32, &[0.85])));
            ev.push(format!("ur:{l}{}", p.unwrap_or_default()));
        }
        if let Some(l) = unit["llm"]["label"].as_str() {
            let p = unit["llm"]["confidence"]
                .as_f64()
                .map(|p| format!("/{}", quantise(p as f32, &[])));
            ev.push(format!("llm:{l}{}", p.unwrap_or_default()));
        }
    }
    let ev = if ev.is_empty() {
        "-".into()
    } else {
        ev.join(",")
    };
    let _ = writeln!(s, "\t{ev}");
    s
}

const RELATIONS_LEGEND: &str =
    "relations (level 3), strongest first: mK = merge component (<= 4 units), \
sI.K = split candidate (sc = score); act by handle, test other sets with eval {merge:[..]}.\n\
merge: rpc/pres/cut = rp_contamination/presence_ratio/amplitude_cutoff of each part -> merged; \
per pair dip_z < 0 = refractory dip (one neuron), sync_z > 0 = fire together (two neurons: do not \
merge), ks amplitude KS, sim template similarity, d um apart, pc presence complement (drift); \
- = too few CCG counts to say.\n\
split: parts = major/minor spikes, amp/rpc per part, dip_z/sync_z between parts (dip = one \
neuron: do not split), bic.\n";

/// Render the brief. `rows` as from `collect_qc_rows_with`; the candidate
/// lists as from the suggestion rankers (any listed unit gets MERGE?/SPLIT?).
#[allow(clippy::too_many_arguments)] // ponytail: one render entry point; bundle into a struct if it grows again
pub fn render_brief<P: DataProvider>(
    session: &Session<P>,
    rows: &[QcRow],
    merges: &[MergeCandidate],
    splits: &[SplitCandidate],
    relations: &[Relation],
    label_str: impl Fn(&P::Label) -> &'static str,
    save_dir: &Path,
    opts: &BriefOpts,
) -> String {
    let in_merge: BTreeSet<u32> = merges.iter().flat_map(|m| [m.a.0, m.b.0]).collect();
    let in_split: BTreeSet<u32> = splits.iter().map(|s| s.cluster.0).collect();
    let units: Vec<Unit> = rows
        .iter()
        .filter(|r| r.n_spikes > 0)
        .map(|r| {
            let mut flags = metric_flags(|m| r.get(m).unwrap_or(f32::NAN));
            if in_merge.contains(&r.cluster_id) {
                flags.push("MERGE?");
            }
            if in_split.contains(&r.cluster_id) {
                flags.push("SPLIT?");
            }
            if session
                .tags(ClusterId(r.cluster_id))
                .iter()
                .any(|t| t.starts_with("review:"))
            {
                flags.push("REV");
            }
            Unit {
                id: r.cluster_id,
                row: r,
                flags,
                label: session
                    .label(ClusterId(r.cluster_id))
                    .map_or("?", |l| label_str(&l)),
            }
        })
        .collect();

    let mode = mode_of(&units.iter().map(|u| u.flags.clone()).collect::<Vec<_>>());
    let mut out = header(session, &units, &mode, &opts.extra);
    if opts.level >= 3 {
        out.push_str(RELATIONS_LEGEND);
    }
    if opts.level == 0 && opts.clusters.is_none() {
        return out;
    }
    // ponytail: bytes/3.5 is a calibration-free token estimate; tune against count_tokens.
    let mut left = opts.budget_tokens.map(|t| (t as f32 * 3.5) as usize);
    if let Some(l) = left.as_mut() {
        *l = l.saturating_sub(out.len()); // the header is never cut
    }
    let mut fit = |s: &str| match left.as_mut() {
        None => true,
        Some(l) if s.len() <= *l => {
            *l -= s.len();
            true
        }
        Some(_) => false,
    };

    // Priority order: most flags first, then id.
    let mut order: Vec<&Unit> = units.iter().collect();
    order.sort_by_key(|u| (std::cmp::Reverse(u.flags.len()), u.id));
    let listed = |u: &&&Unit| !u.flags.is_empty() || u.row.composite_quality < Q_CLEAN;

    let mut omitted = Vec::new();
    let ledger_head = "id\tn\tfr\tlab\tq\tflags\n";
    let mut kept: Vec<(u32, String)> = Vec::new();
    let mut ledger_cut = Vec::new();
    let mut ledger_ids = BTreeSet::new();
    if opts.level >= 1 {
        fit(ledger_head);
        let mut full = false;
        for u in order.iter().filter(listed) {
            ledger_ids.insert(u.id);
            let line = ledger_line(u, &mode, &opts.extra);
            if !full && fit(&line) {
                kept.push((u.id, line));
            } else {
                full = true;
                ledger_cut.push(u.id);
            }
        }
    }

    let mut cards = String::new();
    let mut card_cut = Vec::new();
    let wanted: Vec<u32> = match (&opts.clusters, opts.level) {
        (Some(c), _) => c
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        (None, 2..) => order
            .iter()
            .filter(|u| !u.flags.is_empty())
            .map(|u| u.id)
            .collect(),
        _ => Vec::new(),
    };
    if !wanted.is_empty() {
        let evidence = load_evidence(save_dir);
        let by_id: BTreeMap<u32, &Unit> = units.iter().map(|u| (u.id, u)).collect();
        // Columns measured for at least one wanted unit; the header goes once.
        let cols: Vec<&str> = QcRow::METRICS
            .into_iter()
            .filter(|m| {
                wanted
                    .iter()
                    .filter_map(|id| by_id.get(id))
                    .any(|u| u.row.get(m).is_some_and(|v| !v.is_nan()))
            })
            .collect();
        let head = card_head(&cols);
        let mut full = !fit(&head);
        if !full {
            cards.push_str(&head);
        }
        let mut dead = Vec::new();
        for id in &wanted {
            let Some(u) = by_id.get(id) else {
                dead.push(*id);
                continue;
            };
            let text = card(session, u, &mode, &cols, evidence.as_ref(), &opts.extra);
            if !full && fit(&text) {
                cards.push_str(&text);
            } else {
                full = true;
                card_cut.push(*id);
            }
        }
        if !dead.is_empty() {
            dead.sort_unstable();
            omitted.push(format!("not live: {}", rle(&dead)));
        }
    }

    let mut clean: Vec<u32> = units
        .iter()
        .filter(|u| !ledger_ids.contains(&u.id))
        .map(|u| u.id)
        .collect();
    clean.sort_unstable();
    let clean_line = format!("clean({}): {}\n", clean.len(), rle(&clean));
    let clean_cut = opts.level >= 1 && !clean.is_empty() && !fit(&clean_line);

    if opts.level >= 1 {
        out.push_str(ledger_head);
        kept.sort_unstable();
        kept.iter().for_each(|(_, l)| out.push_str(l));
        if !clean.is_empty() && !clean_cut {
            out.push_str(&clean_line);
        }
    }
    out.push_str(&cards);
    if opts.level >= 3 && !relations.is_empty() {
        out.push_str("relations:\n");
        let mut cut = 0;
        for r in relations {
            let line = format!("{}\n", r.line);
            if fit(&line) {
                out.push_str(&line);
            } else {
                cut += 1;
            }
        }
        if cut > 0 {
            omitted.push(format!("{cut} relations"));
        }
    }
    ledger_cut.sort_unstable();
    card_cut.sort_unstable();
    if !ledger_cut.is_empty() {
        omitted.push(format!("ledger rows {}", rle(&ledger_cut)));
    }
    if !card_cut.is_empty() {
        omitted.push(format!("cards {}", rle(&card_cut)));
    }
    if clean_cut {
        omitted.push(format!("clean list ({} units)", clean.len()));
    }
    if !omitted.is_empty() {
        let _ = writeln!(
            out,
            "omitted: {}; raise budget, or brief{{clusters:[..]}} / metrics{{cluster}}",
            omitted.join("; ")
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift, deterministic.
    fn rng(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Values on both sides of every threshold, 1e-6..0.1 away, plus the
    /// threshold itself and its f32 neighbours.
    fn near(t: f32, s: &mut u64) -> Vec<f32> {
        let mut v = vec![
            t,
            f32::from_bits(t.to_bits() + 1),
            f32::from_bits(t.to_bits() - 1),
        ];
        for k in 0..400 {
            let d = 10f32.powf(-6.0 + 5.0 * rng(s)) * (1.0 + k as f32 / 400.0);
            v.extend([t * (1.0 + d), t * (1.0 - d), t + d, t - d]);
        }
        v
    }

    #[test]
    fn quantise_never_changes_a_threshold_comparison() {
        let mut s = 0x9e37_79b9_7f4a_7c15;
        for m in QcRow::METRICS {
            let th = thresholds(m);
            for &t in &th {
                for v in near(t, &mut s) {
                    let q = quantise(v, &th);
                    let p: f32 = q.parse().unwrap();
                    for u in &th {
                        assert_eq!(p.partial_cmp(u), v.partial_cmp(u), "{m}: {v} -> {q} vs {u}");
                    }
                }
            }
        }
        assert_eq!(quantise(f32::NAN, &[]), "-");
        assert_eq!(quantise(1234.0, &[]), "1200");
        assert_eq!(quantise(300.0, &[300.0]), "300");
        assert_eq!(quantise(0.123456, &[]), "0.12");
        assert_eq!(quantise(0.099999, &[0.1]), "0.099999");
        assert_eq!(quantise(0.0004321, &[]), "4.3e-4");
    }

    #[test]
    fn rule_thresholds_are_respected_by_quantise() {
        let set = crate::rules::RuleSet::parse(
            "[[rule]]\nmetric = \"firing_rate\"\nop = \">\"\nvalue = 1.234\nlabel = \"good\"\n",
        )
        .unwrap();
        let v = 1.2341;
        let plain: f32 = quantise(v, &thresholds("firing_rate")).parse().unwrap();
        assert!(plain < 1.234, "without the rule the value rounds below it");
        let th = thresholds_with("firing_rate", &set.thresholds());
        let kept: f32 = quantise(v, &th).parse().unwrap();
        assert!(kept > 1.234, "{kept}");
    }

    #[test]
    fn flags_agree_on_raw_and_quantised_metrics() {
        let mut s = 0xdead_beef_cafe_f00d;
        for _ in 0..2000 {
            // Each metric independently at a threshold neighbourhood or random.
            let vals: BTreeMap<&str, f32> = QcRow::METRICS
                .iter()
                .map(|&m| {
                    let th = thresholds(m);
                    let v = match th.is_empty() {
                        true => rng(&mut s) * 10.0,
                        false => {
                            let t = th[(rng(&mut s) * th.len() as f32) as usize % th.len()];
                            let v = near(t, &mut s);
                            v[(rng(&mut s) * v.len() as f32) as usize % v.len()]
                        }
                    };
                    (m, if rng(&mut s) < 0.05 { f32::NAN } else { v })
                })
                .collect();
            let raw = metric_flags(|m| vals[m]);
            let quant = metric_flags(|m| {
                quantise(vals[m], &thresholds(m))
                    .parse()
                    .unwrap_or(f32::NAN)
            });
            assert_eq!(raw, quant, "{vals:?}");
        }
    }

    #[test]
    fn flag_table_is_wired_to_real_metrics() {
        for f in FLAGS {
            assert!(QcRow::METRICS.contains(&f.metric), "{}", f.code);
            assert!(
                crate::rules::compare(f.op.trim_start_matches('|'), 0.0, 0.0).is_ok(),
                "{}",
                f.code
            );
        }
        let get = |m: &str| match m {
            "drift_correlation" => -0.6,
            "n_extrema" => 3.0,
            "n_spikes" => f32::NAN,
            _ => 0.5,
        };
        let f = metric_flags(get);
        assert!(
            f.contains(&"DRF") && f.contains(&"MPK") && !f.contains(&"FEW"),
            "{f:?}"
        );
    }

    #[test]
    fn mode_is_stated_once_and_rows_carry_only_the_delta() {
        let sets = vec![
            vec!["RPV", "CUT"],
            vec!["RPV", "MPK"],
            vec!["RPV", "CUT"],
            vec![],
        ];
        // RPV 3/4 and CUT 2/4 (not > half): only RPV is in the mode.
        let mode = mode_of(&sets);
        assert_eq!(mode, vec![("RPV", 3)]);
        assert_eq!(delta(&sets[0], &mode), "+CUT");
        assert_eq!(delta(&sets[1], &mode), "+MPK");
        assert_eq!(delta(&sets[3], &mode), "-RPV");
        assert_eq!(delta(&["RPV"], &mode), "=");
        let rows = [QcRow {
            cluster_id: 7,
            n_spikes: 9,
            ..QcRow::default()
        }];
        let u = Unit {
            id: 7,
            row: &rows[0],
            flags: vec!["RPV", "MPK"],
            label: "good",
        };
        let line = ledger_line(&u, &mode, &[]);
        assert!(
            line.trim_end().ends_with("+MPK") && !line.contains("RPV"),
            "{line}"
        );
        // all units share a flag: it is in the mode, rows never repeat it.
        let all = vec![vec!["ISO"], vec!["ISO", "FEW"], vec!["ISO"]];
        assert_eq!(mode_of(&all), vec![("ISO", 3)]);
        assert_eq!(delta(&all[0], &mode_of(&all)), "=");
    }

    #[test]
    fn rle_collapses_runs() {
        assert_eq!(rle(&[1, 2, 3, 7, 9, 10]), "1-3,7,9-10");
        assert_eq!(rle(&[]), "");
    }
}
