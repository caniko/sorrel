//! Compare two label assignments (`sorrel diff a.tsv b.tsv`): the
//! agreement table two curators or two rule sets produce for the same
//! sorting. Clusters absent from a file count as `unsorted`, matching how
//! phy reads `cluster_group.tsv`.

use serde::Serialize;
use sorrel_io::kilosort::PhyLabel;

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
}
