//! Compare two label assignments (`sorrel diff a.tsv b.tsv`): the
//! agreement table two curators or two rule sets produce for the same
//! sorting. Clusters absent from a file count as `unsorted`, matching how
//! phy reads `cluster_group.tsv`.
//!
//! [`diff_spikes`] (`sorrel diff TRUTH TEST --root INPUT`) compares two
//! curations of one sort spike by spike, scored against the root clusters
//! both started from, so the misses are keyed by ids every arm shares.

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sorrel_io::kilosort::{read_cluster_group_tsv, PhyLabel};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

const LABELS: [PhyLabel; 4] = [
    PhyLabel::Unsorted,
    PhyLabel::Good,
    PhyLabel::Mua,
    PhyLabel::Noise,
];

#[derive(Clone, Debug, Serialize)]
pub struct LabelDiff {
    /// Row = label in `a`, column = label in `b`; order unsorted, good, mua, noise.
    pub confusion: [[u32; 4]; 4],
    pub n_clusters: u32,
    pub n_agree: u32,
    /// Cohen's kappa over the four labels; `NaN` when chance agreement is 1.
    pub kappa: f32,
    /// `(cluster, label in a, label in b)` for every disagreement.
    pub disagreements: Vec<(u32, &'static str, &'static str)>,
}

/// `a` and `b` are `(cluster_id, label)` rows as read from the TSVs.
pub fn diff_labels(a: &[(u32, PhyLabel)], b: &[(u32, PhyLabel)]) -> LabelDiff {
    let n = a.iter().chain(b).map(|&(id, _)| id + 1).max().unwrap_or(0);
    let dense = |rows: &[(u32, PhyLabel)]| {
        let mut v = vec![PhyLabel::Unsorted; n as usize];
        for &(id, l) in rows {
            v[id as usize] = l;
        }
        v
    };
    let (la, lb) = (dense(a), dense(b));

    let mut confusion = [[0u32; 4]; 4];
    let mut disagreements = Vec::new();
    for (id, (&x, &y)) in la.iter().zip(&lb).enumerate() {
        confusion[x as usize][y as usize] += 1;
        if x != y {
            disagreements.push((id as u32, x.as_str(), y.as_str()));
        }
    }
    let n_agree = (0..4).map(|i| confusion[i][i]).sum::<u32>();
    let total = n as f64;
    let observed = n_agree as f64 / total.max(1.0);
    let expected = LABELS
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let row: u32 = confusion[i].iter().sum();
            let col: u32 = (0..4).map(|j| confusion[j][i]).sum();
            (row as f64 / total.max(1.0)) * (col as f64 / total.max(1.0))
        })
        .sum::<f64>();
    let kappa = if (1.0 - expected).abs() < f64::EPSILON {
        f32::NAN
    } else {
        ((observed - expected) / (1.0 - expected)) as f32
    };
    LabelDiff {
        confusion,
        n_clusters: n,
        n_agree,
        kappa,
        disagreements,
    }
}

impl LabelDiff {
    /// Human-readable rendering: confusion table, kappa, disagreements.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str("a \\ b\t");
        for l in LABELS {
            s.push_str(l.as_str());
            s.push('\t');
        }
        s.push('\n');
        for (i, l) in LABELS.iter().enumerate() {
            s.push_str(l.as_str());
            for n in self.confusion[i] {
                s.push_str(&format!("\t{n}"));
            }
            s.push('\n');
        }
        s.push_str(&format!(
            "agree {}/{} (kappa {:.3})\n",
            self.n_agree, self.n_clusters, self.kappa
        ));
        for (id, x, y) in &self.disagreements {
            s.push_str(&format!("{id}\t{x}\t{y}\n"));
        }
        s
    }
}

/// One curation of a sort: the unit of every spike and the label of each unit
/// (a unit absent from `labels` is `unsorted`).
pub struct Curation<'a> {
    pub clusters: &'a [u32],
    pub labels: &'a [(u32, PhyLabel)],
}

#[derive(Clone, Debug)]
pub struct SpikeDiffOpts {
    /// Minimum Jaccard overlap (and the same label) for a test unit to
    /// recover a truth unit.
    pub match_jaccard: f64,
    /// Root clusters with fewer spikes are left out of the merge and split
    /// metrics; a split piece needs at least this many spikes too.
    pub min_spikes: usize,
    /// A split piece must hold at least this fraction of its root cluster's
    /// kept (non-noise) spikes.
    pub split_share: f64,
    /// Score good against not-good: every non-good label becomes noise. `None`
    /// is automatic: on when the truth's labels, over units with spikes, are
    /// all good or noise (a curator who never uses mua).
    pub binary: Option<bool>,
    /// Root cluster ids dropped from scoring (every one of their spikes).
    pub disputed: HashSet<u32>,
}

impl Default for SpikeDiffOpts {
    fn default() -> Self {
        Self {
            match_jaccard: 0.8,
            min_spikes: 10,
            split_share: 0.1,
            binary: None,
            disputed: HashSet::new(),
        }
    }
}

/// Precision and recall of a set of claims; `1.0` when there are none to
/// score (nothing to find, nothing wrongly found).
#[derive(Clone, Debug, Serialize)]
pub struct Pr {
    pub n_truth: u64,
    pub n_test: u64,
    /// Truth items the test found / test items the truth confirms.
    pub hit_truth: u64,
    pub hit_test: u64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl Pr {
    fn new(n_truth: u64, n_test: u64, hit_truth: u64, hit_test: u64) -> Self {
        let ratio = |n: u64, d: u64| if d == 0 { 1.0 } else { n as f64 / d as f64 };
        let (precision, recall) = (ratio(hit_test, n_test), ratio(hit_truth, n_truth));
        let f1 = match precision + recall {
            s if s > 0.0 => 2.0 * precision * recall / s,
            _ => 0.0,
        };
        Self {
            n_truth,
            n_test,
            hit_truth,
            hit_test,
            precision,
            recall,
            f1,
        }
    }
}

/// One place the test curation departs from the truth, keyed by root cluster.
#[derive(Clone, Debug, Serialize)]
pub struct Miss {
    /// `missed_merge`, `false_merge`, `missed_split`, `false_split`, `label`
    /// or `exclusion`.
    pub category: &'static str,
    /// The root clusters involved (a pair for the merges).
    pub roots: Vec<u32>,
    /// Truth and test label of the root's majority unit (`label`, `exclusion`).
    pub truth: Option<&'static str>,
    pub test: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpikeDiff {
    pub n_spikes: u64,
    pub n_root_clusters: u32,
    /// Root clusters dropped as disputed.
    pub n_disputed: u32,
    pub match_jaccard: f64,
    pub min_spikes: u64,
    pub split_share: f64,
    /// Whether labels were scored good against not-good (see
    /// `SpikeDiffOpts::binary`).
    pub binary: bool,
    /// Truth units recovered by a test unit (recall) and test units confirmed
    /// by a truth unit (precision), over the kept (non-noise) units.
    pub units: Pr,
    /// The same over `good` units only.
    pub good_units: Pr,
    /// Majority label of each root cluster, truth against test.
    pub labels: LabelDiff,
    /// Pairs of root clusters (kept in both curations) that share a unit.
    pub merges: Pr,
    /// Root clusters divided into at least two substantial kept units (see
    /// `split_share`).
    pub splits: Pr,
    /// Spikes in noise units.
    pub exclusion: Pr,
    /// Adjusted Rand index of the two spike partitions.
    pub ari: f64,
    pub misses: Vec<Miss>,
}

fn label_map(rows: &[(u32, PhyLabel)]) -> HashMap<u32, PhyLabel> {
    rows.iter().copied().collect()
}

/// Every unit of `clusters` that is not good becomes noise (an unlabelled
/// unit is unsorted, hence noise). Applied once, so every metric (units,
/// labels, exclusion, kept) sees the same two-label vocabulary.
fn fold_binary(m: &mut HashMap<u32, PhyLabel>, clusters: &[u32]) {
    for u in clusters.iter().collect::<BTreeSet<_>>() {
        let l = m.get(u).copied().unwrap_or_default();
        m.insert(
            *u,
            if l == PhyLabel::Good {
                l
            } else {
                PhyLabel::Noise
            },
        );
    }
}

/// The truth's label map and whether it is scored binary (`forced`, else on
/// when every unit is good or noise), folded when it is.
fn truth_labels(truth: &Curation, forced: Option<bool>) -> (HashMap<u32, PhyLabel>, bool) {
    let mut lab = label_map(truth.labels);
    let units: BTreeSet<u32> = truth.clusters.iter().copied().collect();
    let binary = forced.unwrap_or_else(|| {
        units.iter().all(|u| {
            matches!(
                lab.get(u).copied().unwrap_or_default(),
                PhyLabel::Good | PhyLabel::Noise
            )
        })
    });
    if binary {
        fold_binary(&mut lab, truth.clusters);
    }
    (lab, binary)
}

/// Spike counts of `parts`' (root, unit) cells, per root and the majority
/// unit of each root (ties to the lowest id).
struct Cells {
    by_root: BTreeMap<u32, Vec<(u32, u32)>>,
}

impl Cells {
    fn new(cells: &HashMap<(u32, u32), u32>) -> Self {
        let mut by_root: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
        for (&(r, u), &n) in cells {
            by_root.entry(r).or_default().push((u, n));
        }
        by_root
            .values_mut()
            .for_each(|v| v.sort_unstable_by_key(|&(u, n)| (std::cmp::Reverse(n), u)));
        Self { by_root }
    }
    fn home(&self, root: u32) -> u32 {
        self.by_root[&root][0].0
    }
    /// Spike counts of the root's units that are not noise.
    fn kept(&self, root: u32, lab: &HashMap<u32, PhyLabel>) -> Vec<u32> {
        self.by_root[&root]
            .iter()
            .filter(|&&(u, _)| lab.get(&u).copied().unwrap_or_default() != PhyLabel::Noise)
            .map(|&(_, n)| n)
            .collect()
    }
}

/// Unit recovery over the units `keep` admits: a unit is recovered when some
/// unit of the other curation has Jaccard >= `thr` and the same label.
/// ponytail: matching is per side, not one-to-one; below 0.5 a test unit may
/// recover two truth units (the default 0.8 cannot).
fn recovery(
    overlap: &HashMap<(u32, u32), u32>,
    (size_t, size_s): (&HashMap<u32, u32>, &HashMap<u32, u32>),
    (lab_t, lab_s): (&HashMap<u32, PhyLabel>, &HashMap<u32, PhyLabel>),
    thr: f64,
    keep: impl Fn(PhyLabel) -> bool,
) -> Pr {
    let label = |m: &HashMap<u32, PhyLabel>, u: u32| m.get(&u).copied().unwrap_or_default();
    let kept =
        |sizes: &HashMap<u32, u32>, m| sizes.keys().filter(|&&u| keep(label(m, u))).count() as u64;
    let (mut hit_t, mut hit_s) = (HashSet::new(), HashSet::new());
    for (&(t, s), &n) in overlap {
        let (lt, ls) = (label(lab_t, t), label(lab_s, s));
        if lt != ls || !keep(lt) {
            continue;
        }
        let union = size_t[&t] + size_s[&s] - n;
        if n as f64 / union as f64 >= thr {
            hit_t.insert(t);
            hit_s.insert(s);
        }
    }
    Pr::new(
        kept(size_t, lab_t),
        kept(size_s, lab_s),
        hit_t.len() as u64,
        hit_s.len() as u64,
    )
}

fn pairs(n: f64) -> f64 {
    n * (n - 1.0) / 2.0
}

/// Compare `test` against `truth`, both curations of the sort `root` (the
/// root cluster of every spike). All three slices are in spike order.
pub fn diff_spikes(
    root: &[u32],
    truth: &Curation,
    test: &Curation,
    opts: &SpikeDiffOpts,
) -> Result<SpikeDiff> {
    ensure!(
        root.len() == truth.clusters.len() && root.len() == test.clusters.len(),
        "spike counts differ: root {}, truth {}, test {}",
        root.len(),
        truth.clusters.len(),
        test.clusters.len()
    );
    let (lab_t, binary) = truth_labels(truth, opts.binary);
    let mut lab_s = label_map(test.labels);
    if binary {
        fold_binary(&mut lab_s, test.clusters);
    }
    let label = |m: &HashMap<u32, PhyLabel>, u: u32| m.get(&u).copied().unwrap_or_default();

    // One pass: the sparse (root, truth, test) table everything derives from.
    // A disputed root's spikes are left out entirely (unit sizes shrink too).
    let mut triples: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut disputed_seen = HashSet::new();
    for ((&r, &t), &s) in root.iter().zip(truth.clusters).zip(test.clusters) {
        if opts.disputed.contains(&r) {
            disputed_seen.insert(r);
        } else {
            *triples.entry((r, t, s)).or_insert(0) += 1;
        }
    }
    let n_spikes: u64 = triples.values().map(|&n| n as u64).sum();
    let (mut rt, mut rs, mut ts) = (HashMap::new(), HashMap::new(), HashMap::new());
    let (mut size_r, mut size_t, mut size_s) = (HashMap::new(), HashMap::new(), HashMap::new());
    for (&(r, t, s), &n) in &triples {
        *rt.entry((r, t)).or_insert(0u32) += n;
        *rs.entry((r, s)).or_insert(0u32) += n;
        *ts.entry((t, s)).or_insert(0u32) += n;
        *size_r.entry(r).or_insert(0u32) += n;
        *size_t.entry(t).or_insert(0u32) += n;
        *size_s.entry(s).or_insert(0u32) += n;
    }
    let (cells_t, cells_s) = (Cells::new(&rt), Cells::new(&rs));

    let keep_all = |l: PhyLabel| l != PhyLabel::Noise;
    let units = recovery(
        &ts,
        (&size_t, &size_s),
        (&lab_t, &lab_s),
        opts.match_jaccard,
        keep_all,
    );
    let good_units = recovery(
        &ts,
        (&size_t, &size_s),
        (&lab_t, &lab_s),
        opts.match_jaccard,
        |l| l == PhyLabel::Good,
    );

    // Majority label per root cluster, on dense ids so a root with no spikes
    // (or a disputed one) is not an agreeing "unsorted" row.
    let roots: Vec<u32> = size_r
        .keys()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let home_label = |cells: &Cells, m: &HashMap<u32, PhyLabel>, r| label(m, cells.home(r));
    let rows = |cells: &Cells, m: &HashMap<u32, PhyLabel>| -> Vec<(u32, PhyLabel)> {
        roots
            .iter()
            .enumerate()
            .map(|(i, &r)| (i as u32, home_label(cells, m, r)))
            .collect()
    };
    let mut labels = diff_labels(&rows(&cells_t, &lab_t), &rows(&cells_s, &lab_s));
    // Report root ids, not dense indices.
    let sparse: Vec<(u32, &'static str, &'static str)> = labels
        .disagreements
        .iter()
        .map(|&(i, a, b)| (roots[i as usize], a, b))
        .collect();
    labels.disagreements = sparse;

    let mut misses = Vec::new();
    let noise =
        |cells: &Cells, m: &HashMap<u32, PhyLabel>, r| home_label(cells, m, r) == PhyLabel::Noise;
    for &r in &roots {
        let (lt, ls) = (
            home_label(&cells_t, &lab_t, r),
            home_label(&cells_s, &lab_s, r),
        );
        let category = match (lt == PhyLabel::Noise, ls == PhyLabel::Noise) {
            (false, false) if lt != ls => "label",
            (a, b) if a != b => "exclusion",
            _ => continue,
        };
        misses.push(Miss {
            category,
            roots: vec![r],
            truth: Some(lt.as_str()),
            test: Some(ls.as_str()),
        });
    }

    // Merge claims are scored over roots kept in both curations and big
    // enough, so an exclusion is reported once and not as a lost merge.
    let scored: Vec<u32> = roots
        .iter()
        .copied()
        .filter(|&r| {
            size_r[&r] as usize >= opts.min_spikes
                && !noise(&cells_t, &lab_t, r)
                && !noise(&cells_s, &lab_s, r)
        })
        .collect();
    let same_unit = |cells: &Cells| -> HashSet<(u32, u32)> {
        let mut by_unit: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for &r in &scored {
            by_unit.entry(cells.home(r)).or_default().push(r);
        }
        by_unit
            .values()
            .flat_map(|g| {
                g.iter()
                    .enumerate()
                    .flat_map(|(i, &a)| g[i + 1..].iter().map(move |&b| (a, b)))
            })
            .collect()
    };
    let (mt, ms) = (same_unit(&cells_t), same_unit(&cells_s));
    let both = mt.intersection(&ms).count() as u64;
    let merges = Pr::new(mt.len() as u64, ms.len() as u64, both, both);
    let mut pair_misses = |from: &HashSet<(u32, u32)>, not_in: &HashSet<(u32, u32)>, category| {
        let mut v: Vec<_> = from.difference(not_in).copied().collect();
        v.sort_unstable();
        misses.extend(v.into_iter().map(|(a, b)| Miss {
            category,
            roots: vec![a, b],
            truth: None,
            test: None,
        }));
    };
    pair_misses(&mt, &ms, "missed_merge");
    pair_misses(&ms, &mt, "false_merge");

    // A split is judged on a root's kept (non-noise) spikes in each curation:
    // curators carve a few clean cores out of a big template and call the rest
    // noise, so the cores are tiny against the whole root. Not gated on
    // `scored`: a truth that keeps only cores must still be judged, and a test
    // that throws the whole root away has missed the split (its lost core
    // spikes also count against exclusion precision).
    let (mut st, mut ss) = (Vec::new(), Vec::new());
    let is_split = |kept: &[u32]| {
        let total = kept.iter().sum::<u32>() as f64;
        let big =
            |&&n: &&u32| n as f64 >= opts.split_share * total && n as usize >= opts.min_spikes;
        kept.iter().filter(big).count() >= 2
    };
    for &r in roots
        .iter()
        .filter(|r| size_r[r] as usize >= opts.min_spikes)
    {
        let (kt, ks) = (cells_t.kept(r, &lab_t), cells_s.kept(r, &lab_s));
        if is_split(&kt) {
            st.push(r);
        }
        if is_split(&ks) {
            ss.push(r);
        }
    }
    let both = st.iter().filter(|r| ss.contains(r)).count() as u64;
    let splits = Pr::new(st.len() as u64, ss.len() as u64, both, both);
    for (from, other, category) in [(&st, &ss, "missed_split"), (&ss, &st, "false_split")] {
        misses.extend(from.iter().filter(|r| !other.contains(r)).map(|&r| Miss {
            category,
            roots: vec![r],
            truth: None,
            test: None,
        }));
    }

    // Exclusion over spikes: noise in truth against noise in test.
    let (mut n_t, mut n_s, mut n_both) = (0u64, 0u64, 0u64);
    for (&(_, t, s), &n) in &triples {
        let (nt, ns) = (
            label(&lab_t, t) == PhyLabel::Noise,
            label(&lab_s, s) == PhyLabel::Noise,
        );
        n_t += (nt as u64) * n as u64;
        n_s += (ns as u64) * n as u64;
        n_both += (nt && ns) as u64 * n as u64;
    }
    let exclusion = Pr::new(n_t, n_s, n_both, n_both);

    // ARI from the truth x test table.
    let (sum_ij, sum_t, sum_s) = (
        ts.values().map(|&n| pairs(n as f64)).sum::<f64>(),
        size_t.values().map(|&n| pairs(n as f64)).sum::<f64>(),
        size_s.values().map(|&n| pairs(n as f64)).sum::<f64>(),
    );
    let expected = sum_t * sum_s / pairs(n_spikes as f64).max(1.0);
    let max = (sum_t + sum_s) / 2.0;
    let ari = if (max - expected).abs() < f64::EPSILON {
        1.0 // both partitions trivial: identical
    } else {
        (sum_ij - expected) / (max - expected)
    };

    let order = |c: &str| {
        [
            "missed_merge",
            "false_merge",
            "missed_split",
            "false_split",
            "label",
            "exclusion",
        ]
        .iter()
        .position(|x| *x == c)
        .unwrap_or(usize::MAX)
    };
    misses.sort_by(|a, b| (order(a.category), &a.roots).cmp(&(order(b.category), &b.roots)));

    Ok(SpikeDiff {
        n_spikes,
        n_root_clusters: roots.len() as u32,
        n_disputed: disputed_seen.len() as u32,
        match_jaccard: opts.match_jaccard,
        min_spikes: opts.min_spikes as u64,
        split_share: opts.split_share,
        binary,
        units,
        good_units,
        labels,
        merges,
        splits,
        exclusion,
        ari,
        misses,
    })
}

/// What the truth curation did with one root cluster (the dev DB's `truth_roots`).
#[derive(Clone, Debug, PartialEq)]
pub struct RootTruth {
    pub root: u32,
    pub n_spikes: u32,
    /// The first that holds of: `noise` (every unit is noise), `kept` (one
    /// good unit holds at least 90% of the root), `cores` (good units hold
    /// under 50% and the rest is noise: cores carved out of a noise
    /// remainder), `cleanup` (good units hold at least 50% and noise at least
    /// 10%: the majority kept, a contaminated tail dropped), `other`.
    pub verdict: &'static str,
    /// Good units in the root with at least `min_spikes` spikes here; a
    /// smaller piece of a good unit counts as noise.
    pub n_good: u32,
    /// Good spikes over the root's spikes.
    pub good_share: f64,
    /// The good units, largest first: `(share of the root's spikes, spikes)`.
    pub cores: Vec<(f64, u32)>,
}

/// [`RootTruth`] of every root cluster with spikes, ascending. Labels are
/// folded exactly as [`diff_spikes`] folds the truth's; a unit that is
/// neither good nor noise (mua, unsorted, in a truth that uses them) is
/// neither in the shares, so such a root is `other`.
pub fn root_truths(root: &[u32], truth: &Curation, opts: &SpikeDiffOpts) -> Result<Vec<RootTruth>> {
    ensure!(
        root.len() == truth.clusters.len(),
        "spike counts differ: root {}, truth {}",
        root.len(),
        truth.clusters.len()
    );
    let (lab, _) = truth_labels(truth, opts.binary);
    let mut cells: HashMap<(u32, u32), u32> = HashMap::new();
    for (&r, &t) in root.iter().zip(truth.clusters) {
        *cells.entry((r, t)).or_insert(0) += 1;
    }
    let cells = Cells::new(&cells);
    Ok(cells
        .by_root
        .iter()
        .map(|(&r, units)| {
            let n: u32 = units.iter().map(|&(_, n)| n).sum();
            let big = opts.min_spikes.min(n as usize) as u32;
            let (mut good, mut noise) = (Vec::new(), 0u32);
            for &(u, c) in units {
                match lab.get(&u).copied().unwrap_or_default() {
                    PhyLabel::Good if c >= big => good.push(c),
                    PhyLabel::Good | PhyLabel::Noise => noise += c,
                    _ => {}
                }
            }
            let good_n: u32 = good.iter().sum();
            let share = |c: u32| f64::from(c) / f64::from(n);
            let verdict = if noise == n {
                "noise"
            } else if good.first().is_some_and(|&c| share(c) >= 0.9) {
                "kept"
            } else if good_n > 0 && share(good_n) < 0.5 && good_n + noise == n {
                "cores"
            } else if share(good_n) >= 0.5 && share(noise) >= 0.1 && good_n + noise == n {
                "cleanup"
            } else {
                "other"
            };
            RootTruth {
                root: r,
                n_spikes: n,
                verdict,
                n_good: good.len() as u32,
                good_share: share(good_n),
                cores: good.iter().map(|&c| (share(c), c)).collect(),
            }
        })
        .collect())
}

/// A disputed-roots file: one root cluster id per line, optional
/// `cluster_id` header, blank lines ignored.
pub fn read_disputed(path: &Path) -> Result<HashSet<u32>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && *l != "cluster_id")
        .map(|l| {
            l.parse()
                .with_context(|| format!("{}: bad cluster id {l:?}", path.display()))
        })
        .collect()
}

/// Spike units and `(unit, label)` rows of a saved curation.
type Loaded = (Vec<u32>, Vec<(u32, PhyLabel)>);

/// Load a curation saved as phy files (`spike_clusters.npy`, optional
/// `cluster_group.tsv`) from `dir`.
fn load_curation(dir: &Path) -> Result<Loaded> {
    let clusters = sorrel_io::npy::read_1d_u32(&dir.join("spike_clusters.npy"))
        .with_context(|| format!("read {}/spike_clusters.npy", dir.display()))?;
    let tsv = dir.join("cluster_group.tsv");
    let labels = match tsv.exists() {
        true => read_cluster_group_tsv(&tsv).with_context(|| format!("read {}", tsv.display()))?,
        false => vec![],
    };
    Ok((clusters, labels))
}

/// [`diff_spikes`] over saved directories: `truth` and `test` hold the two
/// curations, `root` the sort both started from (its root files, so a
/// curated phy directory works as well as a blind input).
pub fn diff_dirs(
    truth: &Path,
    test: &Path,
    root: &Path,
    opts: &SpikeDiffOpts,
) -> Result<SpikeDiff> {
    let root_path = sorrel_io::RootView::open(root)?.path("spike_clusters.npy");
    let root_clusters = sorrel_io::npy::read_1d_u32(&root_path)
        .with_context(|| format!("read {}", root_path.display()))?;
    let ((tc, tl), (sc, sl)) = (load_curation(truth)?, load_curation(test)?);
    diff_spikes(
        &root_clusters,
        &Curation {
            clusters: &tc,
            labels: &tl,
        },
        &Curation {
            clusters: &sc,
            labels: &sl,
        },
        opts,
    )
}

/// [`root_truths`] over a saved truth directory and the sort `root` it was
/// curated from; also whether the truth is scored binary.
pub fn root_truths_dirs(
    truth: &Path,
    root: &Path,
    opts: &SpikeDiffOpts,
) -> Result<(bool, Vec<RootTruth>)> {
    let root_path = sorrel_io::RootView::open(root)?.path("spike_clusters.npy");
    let root_clusters = sorrel_io::npy::read_1d_u32(&root_path)
        .with_context(|| format!("read {}", root_path.display()))?;
    let (clusters, labels) = load_curation(truth)?;
    let c = Curation {
        clusters: &clusters,
        labels: &labels,
    };
    Ok((
        truth_labels(&c, opts.binary).1,
        root_truths(&root_clusters, &c, opts)?,
    ))
}

impl SpikeDiff {
    /// Short human-readable summary.
    pub fn render(&self) -> String {
        let pr = |name: &str, p: &Pr| {
            format!(
                "{name}: P {:.3} R {:.3} F1 {:.3} ({}/{} truth, {}/{} test)\n",
                p.precision, p.recall, p.f1, p.hit_truth, p.n_truth, p.hit_test, p.n_test
            )
        };
        let mut s = format!(
            "{} spikes, {} root clusters ({} disputed), match {}\n",
            self.n_spikes, self.n_root_clusters, self.n_disputed, self.match_jaccard
        );
        if self.binary {
            s += "binary: good against not-good (mua and unsorted count as noise)\n";
        }
        s += &pr("units", &self.units);
        s += &pr("good units", &self.good_units);
        s += &pr("merge pairs", &self.merges);
        s += &pr("splits", &self.splits);
        s += &pr("noise spikes", &self.exclusion);
        s += &format!(
            "labels: agree {}/{} kappa {:.3}\nARI {:.3}\n",
            self.labels.n_agree, self.labels.n_clusters, self.labels.kappa, self.ari
        );
        for m in &self.misses {
            let ids: Vec<String> = m.roots.iter().map(u32::to_string).collect();
            s += &format!("{}\t{}", m.category, ids.join(","));
            if let (Some(t), Some(x)) = (m.truth, m.test) {
                s += &format!("\t{t} -> {x}");
            }
            s.push('\n');
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confusion_kappa_and_disagreements() {
        // a: 0 good, 1 good, 2 mua, 3 noise, 4 (absent → unsorted)
        // b: 0 good, 1 mua,  2 mua, 3 noise, 4 noise
        let a = [
            (0, PhyLabel::Good),
            (1, PhyLabel::Good),
            (2, PhyLabel::Mua),
            (3, PhyLabel::Noise),
        ];
        let b = [
            (0, PhyLabel::Good),
            (1, PhyLabel::Mua),
            (2, PhyLabel::Mua),
            (3, PhyLabel::Noise),
            (4, PhyLabel::Noise),
        ];
        let d = diff_labels(&a, &b);
        assert_eq!(d.n_clusters, 5);
        assert_eq!(d.n_agree, 3);
        assert_eq!(d.confusion[1][1], 1, "good/good");
        assert_eq!(d.confusion[1][2], 1, "good in a, mua in b");
        assert_eq!(d.confusion[0][3], 1, "absent in a, noise in b");
        assert_eq!(
            d.disagreements,
            vec![(1, "good", "mua"), (4, "unsorted", "noise")]
        );
        // po = 3/5 = 0.6; pe = Σ row_i*col_i / 25 with rows (1,2,1,1), cols (0,1,2,2)
        //    = (0 + 2 + 2 + 2) / 25 = 0.24 → kappa = (0.6 - 0.24) / 0.76
        assert!((d.kappa - 0.4737).abs() < 1e-3, "kappa = {}", d.kappa);
        assert!(d.render().contains("agree 3/5"));
    }

    /// Root 0 (20 spikes), 1 (20), 2 (20), 3 (20), 4 (20): the truth merges
    /// 1 into 0, splits 2 in half, labels 3 mua and calls 4 noise.
    fn scenario() -> (Vec<u32>, Vec<u32>, Vec<(u32, PhyLabel)>) {
        let root: Vec<u32> = (0..5u32).flat_map(|r| std::iter::repeat_n(r, 20)).collect();
        let truth: Vec<u32> = root
            .iter()
            .enumerate()
            .map(|(i, &r)| match r {
                1 => 0,
                2 if i % 2 == 0 => 5,
                r => r,
            })
            .collect();
        let labels = vec![
            (0, PhyLabel::Good),
            (2, PhyLabel::Good),
            (3, PhyLabel::Mua),
            (4, PhyLabel::Noise),
            (5, PhyLabel::Good),
        ];
        (root, truth, labels)
    }

    #[test]
    fn identical_curations_are_perfect() {
        let (root, truth, labels) = scenario();
        let c = Curation {
            clusters: &truth,
            labels: &labels,
        };
        let d = diff_spikes(&root, &c, &c, &SpikeDiffOpts::default()).unwrap();
        assert!(d.misses.is_empty(), "{:?}", d.misses);
        assert_eq!(
            (d.units.f1, d.good_units.f1, d.merges.f1, d.splits.f1),
            (1.0, 1.0, 1.0, 1.0)
        );
        assert_eq!(
            (d.exclusion.f1, d.ari, d.labels.n_agree),
            (1.0, 1.0, d.labels.n_clusters)
        );
        assert_eq!((d.merges.n_truth, d.splits.n_truth), (1, 1));
    }

    #[test]
    fn the_root_sort_misses_exactly_what_the_curator_did() {
        let (root, truth, labels) = scenario();
        let d = diff_spikes(
            &root,
            &Curation {
                clusters: &truth,
                labels: &labels,
            },
            &Curation {
                clusters: &root,
                labels: &[],
            },
            &SpikeDiffOpts::default(),
        )
        .unwrap();
        let got: Vec<_> = d
            .misses
            .iter()
            .map(|m| (m.category, m.roots.clone()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("missed_merge", vec![0, 1]),
                ("missed_split", vec![2]),
                ("label", vec![0]),
                ("label", vec![1]),
                ("label", vec![2]),
                ("label", vec![3]),
                ("exclusion", vec![4]),
            ]
        );
        assert_eq!((d.merges.recall, d.merges.precision), (0.0, 1.0));
        assert_eq!(d.exclusion.recall, 0.0);
        assert!(d.ari < 1.0);
    }

    /// One root of 1000 spikes; the truth carves cores of 50, 52 and 48 spikes
    /// (units 1..=3, good) out of it and leaves 850 as noise (unit 0).
    fn cores_and_noise() -> (Vec<u32>, Vec<u32>, Vec<(u32, PhyLabel)>) {
        let truth: Vec<u32> = (0..1000u32)
            .map(|i| match i {
                0..50 => 1,
                50..102 => 2,
                102..150 => 3,
                _ => 0,
            })
            .collect();
        let labels = vec![
            (0, PhyLabel::Noise),
            (1, PhyLabel::Good),
            (2, PhyLabel::Good),
            (3, PhyLabel::Good),
        ];
        (vec![0; 1000], truth, labels)
    }

    #[test]
    fn split_share_is_over_kept_spikes() {
        let (root, truth, labels) = cores_and_noise();
        let t = Curation {
            clusters: &truth,
            labels: &labels,
        };
        let opts = SpikeDiffOpts::default();
        // Untouched root: one unit, labelled good (any label but noise).
        let whole = vec![0; 1000];
        let untouched = Curation {
            clusters: &whole,
            labels: &[(0, PhyLabel::Good)],
        };
        let d = diff_spikes(&root, &t, &untouched, &opts).unwrap();
        let got: Vec<_> = d.misses.iter().map(|m| m.category).collect();
        assert!(got.contains(&"missed_split"), "{got:?}");
        assert_eq!((d.splits.n_truth, d.splits.recall), (1, 0.0));

        // The same truth reproduced: perfect split score, nothing missed.
        let d = diff_spikes(&root, &t, &t, &opts).unwrap();
        assert_eq!((d.splits.n_truth, d.splits.f1), (1, 1.0));
        assert!(d.misses.is_empty(), "{:?}", d.misses);

        // A test that calls the whole root noise has missed the split too:
        // throwing the root away must not score a vacuous split recall of 1.
        let all_noise = Curation {
            clusters: &whole,
            labels: &[(0, PhyLabel::Noise)],
        };
        let d = diff_spikes(&root, &t, &all_noise, &opts).unwrap();
        assert_eq!((d.splits.n_truth, d.splits.recall), (1, 0.0));
        assert!(d.misses.iter().any(|m| m.category == "missed_split"));

        // Each core is about a third of the kept spikes, so 0.4 leaves no split.
        let strict = SpikeDiffOpts {
            split_share: 0.4,
            ..opts
        };
        assert_eq!(
            diff_spikes(&root, &t, &t, &strict).unwrap().splits.n_truth,
            0
        );
    }

    /// Roots 0 (good), 1 and 2 (noise) in the truth; the test calls 1 mua and
    /// leaves 2 unlabelled.
    fn binary_pair() -> (Loaded, Loaded) {
        let root: Vec<u32> = (0..3u32).flat_map(|r| std::iter::repeat_n(r, 20)).collect();
        let truth = vec![
            (0, PhyLabel::Good),
            (1, PhyLabel::Noise),
            (2, PhyLabel::Noise),
        ];
        let test = vec![(0, PhyLabel::Good), (1, PhyLabel::Mua)];
        ((root.clone(), truth), (root, test))
    }

    #[test]
    fn a_truth_without_mua_is_scored_good_against_not_good() {
        let ((root, tl), (_, sl)) = binary_pair();
        let score = |binary| {
            let (t, s) = (
                Curation {
                    clusters: &root,
                    labels: &tl,
                },
                Curation {
                    clusters: &root,
                    labels: &sl,
                },
            );
            let opts = SpikeDiffOpts {
                binary,
                ..Default::default()
            };
            diff_spikes(&root, &t, &s, &opts).unwrap()
        };
        let d = score(None);
        assert!(d.binary && d.misses.is_empty(), "{:?}", d.misses);
        assert_eq!((d.exclusion.f1, d.labels.n_agree), (1.0, 3));
        assert!(d.render().contains("binary"));

        let d = score(Some(false));
        let got: Vec<_> = d
            .misses
            .iter()
            .map(|m| (m.category, m.roots.clone()))
            .collect();
        assert!(!d.binary);
        assert_eq!(got, vec![("exclusion", vec![1]), ("exclusion", vec![2])]);
    }

    #[test]
    fn a_truth_with_mua_is_not_binary() {
        let (root, truth, labels) = scenario();
        let c = Curation {
            clusters: &truth,
            labels: &labels,
        };
        let d = diff_spikes(&root, &c, &c, &SpikeDiffOpts::default()).unwrap();
        assert!(!d.binary);
        assert!(!d.render().contains("binary"));
        // Forced on, the mua root folds into noise in both curations.
        let forced = SpikeDiffOpts {
            binary: Some(true),
            ..Default::default()
        };
        let d = diff_spikes(&root, &c, &c, &forced).unwrap();
        assert!(d.binary && d.misses.is_empty());
        assert_eq!(d.exclusion.n_truth, 40);
    }

    #[test]
    fn disputed_roots_are_not_scored() {
        let (root, truth, labels) = scenario();
        let opts = SpikeDiffOpts {
            disputed: [0, 1, 2, 3, 4, 99].into(),
            ..Default::default()
        };
        let d = diff_spikes(
            &root,
            &Curation {
                clusters: &truth,
                labels: &labels,
            },
            &Curation {
                clusters: &root,
                labels: &[],
            },
            &opts,
        )
        .unwrap();
        assert!(d.misses.is_empty() && d.n_spikes == 0 && d.ari == 1.0);
        assert_eq!(d.n_disputed, 5, "an id absent from the root is not counted");
    }

    #[test]
    fn root_truths_tell_noise_kept_cores_and_cleanup() {
        // root 0: all noise; 1: one good unit; 2: a 20-spike good core in a
        // noise remainder; 3: two good halves; 4: 70 good, 30 noise.
        let mut root = Vec::new();
        let mut truth = Vec::new();
        let cells = [
            (0, 10, 50),
            (1, 11, 50),
            (2, 12, 80),
            (2, 13, 20),
            (3, 14, 30),
            (3, 15, 30),
            (4, 16, 70),
            (4, 17, 30),
        ];
        for (r, t, n) in cells {
            root.extend(std::iter::repeat_n(r, n));
            truth.extend(std::iter::repeat_n(t, n));
        }
        let noise = [10, 12, 17].map(|u| (u, PhyLabel::Noise));
        let good = [11, 13, 14, 15, 16].map(|u| (u, PhyLabel::Good));
        let labels = [noise.as_slice(), good.as_slice()].concat();
        let c = Curation {
            clusters: &truth,
            labels: &labels,
        };
        let got = root_truths(&root, &c, &SpikeDiffOpts::default()).unwrap();
        let v: Vec<_> = got.iter().map(|r| (r.root, r.verdict, r.n_good)).collect();
        assert_eq!(
            v,
            [
                (0, "noise", 0),
                (1, "kept", 1),
                (2, "cores", 1),
                (3, "other", 2),
                (4, "cleanup", 1)
            ]
        );
        assert_eq!(got[2].cores, [(0.2, 20)]);
        assert_eq!((got[4].good_share, got[0].good_share), (0.7, 0.0));
    }

    #[test]
    fn a_root_with_mua_left_over_is_not_cleanup() {
        // 70 good, 20 noise, 10 mua: the mua is neither share, so not cleanup.
        let root = vec![0u32; 100];
        let truth: Vec<u32> = [(1, 70), (2, 20), (3, 10)]
            .into_iter()
            .flat_map(|(u, n)| std::iter::repeat_n(u, n))
            .collect();
        let labels = [
            (1, PhyLabel::Good),
            (2, PhyLabel::Noise),
            (3, PhyLabel::Mua),
        ];
        let c = Curation {
            clusters: &truth,
            labels: &labels,
        };
        let got = root_truths(&root, &c, &SpikeDiffOpts::default()).unwrap();
        assert_eq!(got[0].verdict, "other");
    }
}
