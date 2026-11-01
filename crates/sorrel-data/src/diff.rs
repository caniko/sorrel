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
    /// metrics.
    pub min_spikes: usize,
    /// A split piece must hold at least this fraction of its root cluster.
    pub split_fraction: f64,
    /// Root cluster ids dropped from scoring (every one of their spikes).
    pub disputed: HashSet<u32>,
}

impl Default for SpikeDiffOpts {
    fn default() -> Self {
        Self {
            match_jaccard: 0.8,
            min_spikes: 10,
            split_fraction: 0.1,
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
    /// Truth units recovered by a test unit (recall) and test units confirmed
    /// by a truth unit (precision), over the kept (non-noise) units.
    pub units: Pr,
    /// The same over `good` units only.
    pub good_units: Pr,
    /// Majority label of each root cluster, truth against test.
    pub labels: LabelDiff,
    /// Pairs of root clusters (kept in both curations) that share a unit.
    pub merges: Pr,
    /// Root clusters (kept in both) divided into at least two substantial units.
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
    /// Units holding at least `frac` of the root's `total` spikes.
    fn substantial(&self, root: u32, total: u32, frac: f64) -> usize {
        self.by_root[&root]
            .iter()
            .filter(|&&(_, n)| n as f64 >= frac * total as f64)
            .count()
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
    let (lab_t, lab_s) = (label_map(truth.labels), label_map(test.labels));
    let label = |m: &HashMap<u32, PhyLabel>, u: u32| m.get(&u).copied().unwrap_or_default();

    // One pass: the sparse (root, truth, test) table everything derives from.
    // A disputed root's spikes are left out entirely (unit sizes shrink too).
    let mut triples: HashMap<(u32, u32, u32), u32> = HashMap::new();
    for ((&r, &t), &s) in root.iter().zip(truth.clusters).zip(test.clusters) {
        if !opts.disputed.contains(&r) {
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

    // Merge and split claims are scored over roots kept in both curations and
    // big enough, so an exclusion is reported once and not as a lost merge.
    // ponytail: a truth split of a root the test calls noise is an exclusion
    // miss only; split noise out of a kept root is scored.
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

    let is_split = |cells: &Cells, r| cells.substantial(r, size_r[&r], opts.split_fraction) >= 2;
    let (st, ss): (Vec<u32>, Vec<u32>) = (
        scored
            .iter()
            .copied()
            .filter(|&r| is_split(&cells_t, r))
            .collect(),
        scored
            .iter()
            .copied()
            .filter(|&r| is_split(&cells_s, r))
            .collect(),
    );
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
        n_disputed: opts.disputed.len() as u32,
        match_jaccard: opts.match_jaccard,
        min_spikes: opts.min_spikes as u64,
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

    #[test]
    fn disputed_roots_are_not_scored() {
        let (root, truth, labels) = scenario();
        let opts = SpikeDiffOpts {
            disputed: [0, 1, 2, 3, 4].into(),
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
    }
}
