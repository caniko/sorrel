//! Lead-only dev diagnostics. Proposer boundaries precede truth reads. The
//! separate feature-oracle mode deliberately fits human training labels and
//! must never be deployed as an input-only proposer or curation performance.

use super::*;
use sorrel_data::proposers::all_cluster_globals;
use sorrel_data::robust_cleanup::robust_cleanup;
use sorrel_data::{Journal, Session};
use sorrel_io::kilosort::{read_cluster_group_tsv, KilosortProvider, PhyLabel};
use sorrel_io::ClusterId;

struct CoverageNode {
    members: Vec<u32>,
    details: Value,
}

pub(super) fn dev_sessions(map: &SessionMap, limit: usize) -> Result<Vec<&str>> {
    ensure!(limit > 0, "--dev-limit must be positive");
    let ids: Vec<_> = map
        .sessions
        .iter()
        .filter(|(_, s)| s.split == Split::Dev)
        .take(limit)
        .map(|(sid, _)| sid.as_str())
        .collect();
    ensure!(!ids.is_empty(), "no dev sessions for coverage");
    Ok(ids)
}

fn jaccard(intersection: usize, predicted: usize, truth: usize) -> f64 {
    let union = predicted + truth - intersection;
    if union > 0 {
        intersection as f64 / union as f64
    } else {
        0.0
    }
}

/// Session globals are time-sorted with cluster-ID tie breaking; NPY rows
/// need not be. Never silently compare these two index spaces directly.
pub(super) fn aligned_truth(
    rows: &[Option<u32>],
    truth: &[u32],
    input_roots: &[u32],
    session_roots: &[u32],
) -> Result<Vec<u32>> {
    ensure!(
        rows.len() == truth.len()
            && input_roots.len() == truth.len()
            && session_roots.len() == truth.len(),
        "truth row-map lengths differ"
    );
    let mut seen = vec![false; truth.len()];
    let mut aligned = Vec::with_capacity(truth.len());
    for (global, row) in rows.iter().enumerate() {
        let row = row.context("truth needs a complete original NPY-row map")? as usize;
        ensure!(
            row < truth.len() && !seen[row],
            "truth row map is not a permutation"
        );
        ensure!(
            input_roots[row] == session_roots[global],
            "truth row map disagrees with the input cluster"
        );
        seen[row] = true;
        aligned.push(truth[row]);
    }
    Ok(aligned)
}

impl Bench {
    pub(super) fn coverage(&self, limit: usize, method: CoverageMethod) -> Result<()> {
        let mut reports = Vec::new();
        let mut totals = [0usize; 3];
        // No split flag or arbitrary path: the native map must prove dev
        // before any input or truth is opened. Holdout paths are never touched.
        for sid in dev_sessions(&self.map, limit)? {
            let temp = tempfile::Builder::new()
                .prefix("coverage-")
                .tempdir_in(&self.root)?;
            let data = temp.path().join("input");
            self.copy_input(sid, &data)?;
            let provider = KilosortProvider::open(&data, Default::default())?;
            let mut session = Session::new(
                provider,
                Journal::open(&temp.path().join("diagnostic.journal"))?,
            );
            session.seed_pc_features();
            session.seed_templates();
            if matches!(method, CoverageMethod::FeatureOracle | CoverageMethod::Axis) {
                session.seed_amplitudes();
            }
            let globals = all_cluster_globals(&session);
            // All candidate construction precedes truth loading. No human
            // unit count, label, index, or overlap can steer model fitting.
            let proposals: Vec<_> = globals
                .iter()
                .enumerate()
                .map(|(c, g)| {
                    if method == CoverageMethod::Robust {
                        robust_cleanup(&session, ClusterId(c as u32), g)
                    } else {
                        None
                    }
                })
                .collect();
            let trees: Vec<Vec<CoverageNode>> = globals
                .iter()
                .enumerate()
                .map(|(c, g)| {
                    if method == CoverageMethod::Density {
                        sorrel_data::density_tree::density_candidates(
                            &session,
                            ClusterId(c as u32),
                            g,
                        ).into_iter().map(|n|CoverageNode{details:json!({"depth":n.depth,
                            "supported":n.supported,"rpc":n.contamination,"bic_gain":n.bic_gain}),members:n.members}).collect()
                    } else if method==CoverageMethod::Axis {
                        sorrel_data::axis_split::axis_candidates(&session,ClusterId(c as u32),g)
                            .into_iter().map(|n|CoverageNode{details:json!({"feature":n.feature,"at":n.at,"above":n.above,
                                "supported":n.supported,"rpc":n.contamination,"bic_gain":n.bic_gain}),members:n.members}).collect()
                    } else {
                        vec![]
                    }
                })
                .collect();
            let truth_dir = self.session(sid).join("truth");
            let truth_rows = sorrel_io::npy::read_1d_u32(&truth_dir.join("spike_clusters.npy"))?;
            let input_roots = sorrel_io::npy::read_1d_u32(&data.join("spike_clusters.npy"))?;
            let rows: Vec<_> = session
                .cluster_index()
                .spike_pc_indices()
                .iter()
                .map(|row| row.get())
                .collect();
            let session_roots: Vec<_> = session
                .spike_clusters_global()
                .iter()
                .map(|c| c.0)
                .collect();
            let truth = aligned_truth(&rows, &truth_rows, &input_roots, &session_roots)?;
            ensure!(
                truth.len() == session.spike_clusters_global().len(),
                "truth and input spike counts differ"
            );
            let labels = read_cluster_group_tsv(&truth_dir.join("cluster_group.tsv"))
                .context("dev truth labels missing or unreadable")?;
            let good: BTreeSet<_> = labels
                .iter()
                .filter(|(_, l)| *l == PhyLabel::Good)
                .map(|(id, _)| *id)
                .collect();
            let mut sizes: BTreeMap<u32, usize> = BTreeMap::new();
            for &id in &truth {
                if good.contains(&id) {
                    *sizes.entry(id).or_default() += 1;
                }
            }
            if method == CoverageMethod::FeatureOracle {
                let mut truth_totals: BTreeMap<u32, [usize; 3]> =
                    sizes.keys().map(|&id| (id, [0; 3])).collect();
                for (global, &id) in truth.iter().enumerate() {
                    if let Some(total) = truth_totals.get_mut(&id) {
                        total[global % 2] += 1;
                        total[2] += 1;
                    }
                }
                let mut audit_rows = Vec::new();
                for (root, g) in globals.iter().enumerate() {
                    let local_truth: Vec<_> = g.iter().map(|&i| truth[i as usize]).collect();
                    let targets: BTreeSet<_> = local_truth
                        .iter()
                        .copied()
                        .filter(|id| good.contains(id))
                        .collect();
                    if targets.is_empty() {
                        continue;
                    }
                    let c = ClusterId(root as u32);
                    let training: Vec<_> = g.iter().map(|&i| i % 2 == 0).collect();
                    let mut columns = Vec::new();
                    if let Some((feats, d)) =
                        sorrel_data::proposers::cluster_feature_space(&session, c, g)
                    {
                        for j in 0..d {
                            columns.push((
                                format!("pc_whitened_{j}"),
                                feats.chunks_exact(d).map(|r| r[j]).collect::<Vec<_>>(),
                            ));
                        }
                    }
                    let amps = session.spike_amplitudes(c);
                    if amps.len() == g.len() {
                        columns.push(("amplitude".into(), amps.to_vec()));
                    }
                    columns.push((
                        "time_sample".into(),
                        session.spike_times(c).iter().map(|t| t.0 as f32).collect(),
                    ));
                    for id in targets {
                        let target: Vec<_> = local_truth.iter().map(|&v| v == id).collect();
                        let whole =
                            jaccard(target.iter().filter(|&&v| v).count(), g.len(), sizes[&id]);
                        let mut axes = Vec::new();
                        for (name, values) in &columns {
                            if let Some(a) = sorrel_data::feature_oracle::audit_axis(
                                values,
                                &target,
                                &training,
                                truth_totals[&id],
                            ) {
                                axes.push(json!({"feature":name,"at":a.at,"above":a.above,
                                    "train_jaccard":a.train_jaccard,"validation_jaccard":a.validation_jaccard,
                                    "full_jaccard":a.full_jaccard,"sampled_training":a.sampled_training,
                                    "validation_predicted":a.predicted_validation,"validation_hits":a.hit_validation}));
                            }
                        }
                        // Feature selection uses training objective ONLY. Validation
                        // labels cannot select the dimension, direction or threshold.
                        axes.sort_by(|a, b| {
                            b["train_jaccard"]
                                .as_f64()
                                .unwrap()
                                .total_cmp(&a["train_jaccard"].as_f64().unwrap())
                        });
                        axes.truncate(3);
                        audit_rows.push(json!({"root":root,"unit":id,"root_n":g.len(),"truth_n":sizes[&id],
                            "truth_train":truth_totals[&id][0],"truth_validation":truth_totals[&id][1],
                            "whole_jaccard":whole,"axes_ranked_by_training":axes}));
                    }
                }
                totals[0] += sizes.len();
                println!("dev {sid}: {} good units; truth-fitted feature audit, NOT curation performance",sizes.len());
                reports.push(json!({"sid":sid,"n_good":sizes.len(),"rows":audit_rows,
                    "input_hash":sorrel_data::devdb::truth_hash(&data)?,"truth_hash":sorrel_data::devdb::truth_hash(&truth_dir)?}));
                continue;
            }
            let mut rows = Vec::new();
            let mut best: BTreeMap<u32, (f64, f64)> =
                sizes.keys().map(|&id| (id, (0.0, 0.0))).collect();
            for (root, g) in globals.iter().enumerate() {
                if matches!(method, CoverageMethod::Density | CoverageMethod::Axis) {
                    let mut whole_hits: BTreeMap<u32, usize> = BTreeMap::new();
                    for &index in g {
                        let id = truth[index as usize];
                        if good.contains(&id) {
                            *whole_hits.entry(id).or_default() += 1;
                        }
                    }
                    let mut overlaps: BTreeMap<u32, Vec<Value>> = BTreeMap::new();
                    for (candidate, node) in trees[root].iter().enumerate() {
                        let mut hits: BTreeMap<u32, usize> = BTreeMap::new();
                        for &local in &node.members {
                            let id = truth[g[local as usize] as usize];
                            if good.contains(&id) {
                                *hits.entry(id).or_default() += 1;
                            }
                        }
                        for (id, hit) in hits {
                            let mut detail = node.details.clone();
                            detail["candidate"] = json!(candidate);
                            detail["size"] = json!(node.members.len());
                            detail["jaccard"] = json!(jaccard(hit, node.members.len(), sizes[&id]));
                            overlaps.entry(id).or_default().push(detail);
                        }
                    }
                    for (id, hit) in whole_hits {
                        let whole = jaccard(hit, g.len(), sizes[&id]);
                        let candidates = overlaps.remove(&id).unwrap_or_default();
                        let supported = candidates
                            .iter()
                            .filter(|c| c["supported"] == true)
                            .filter_map(|c| c["jaccard"].as_f64())
                            .fold(whole, f64::max);
                        let b = best.get_mut(&id).unwrap();
                        b.0 = b.0.max(whole);
                        b.1 = b.1.max(supported);
                        rows.push(json!({"root":root,"unit":id,"truth_n":sizes[&id],"root_n":g.len(),
                            "whole_jaccard":whole,"supported_best_jaccard":supported,"candidates":candidates}));
                    }
                    continue;
                }
                let p = &proposals[root];
                let mut intersections: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
                for (i, &index) in g.iter().enumerate() {
                    let id = truth[index as usize];
                    if !good.contains(&id) {
                        continue;
                    }
                    let counts = intersections.entry(id).or_default();
                    counts.0 += 1;
                    if p.as_ref()
                        .is_some_and(|p| p.minor_idx.binary_search(&(i as u32)).is_ok())
                    {
                        counts.1 += 1;
                    }
                }
                for (id, (whole_hits, minor_hits)) in intersections {
                    let whole = jaccard(whole_hits, g.len(), sizes[&id]);
                    let kept = p.as_ref().map(|p| {
                        jaccard(
                            whole_hits - minor_hits,
                            g.len() - p.minor_idx.len(),
                            sizes[&id],
                        )
                    });
                    let dropped = p
                        .as_ref()
                        .map(|p| jaccard(minor_hits, p.minor_idx.len(), sizes[&id]));
                    let supported = p.as_ref().is_some_and(|p| p.supported);
                    let reachable = if supported {
                        whole.max(kept.unwrap()).max(dropped.unwrap())
                    } else {
                        whole
                    };
                    let b = best.get_mut(&id).unwrap();
                    b.0 = b.0.max(whole);
                    b.1 = b.1.max(reachable);
                    rows.push(json!({"root":root,"unit":id,"truth_n":sizes[&id],"root_n":g.len(),
                        "whole_jaccard":whole,"kept_jaccard":kept,"dropped_jaccard":dropped,
                        "supported":supported,"dropped_n":p.as_ref().map(|p|p.minor_idx.len()),
                        "bic_gain":p.as_ref().map(|p|p.bic_gain),"clean_weight":p.as_ref().map(|p|p.clean_weight),
                        "inflation":p.as_ref().map(|p|p.inflation),
                        "whole_rpc":p.as_ref().map(|p|p.whole_contamination),
                        "kept_rpc":p.as_ref().map(|p|p.kept_contamination)}));
                }
            }
            let whole = best.values().filter(|b| b.0 >= 0.8).count();
            let augmented = best.values().filter(|b| b.1 >= 0.8).count();
            totals[0] += sizes.len();
            totals[1] += whole;
            totals[2] += augmented;
            println!("dev {sid}: {} good units; whole-root oracle {whole}, whole+supported-{} oracle {augmented}",sizes.len(),method.name());
            reports.push(
                json!({"sid":sid,"n_good":sizes.len(),"whole_reachable":whole,
                "with_proposer_reachable":augmented,"rows":rows,
                "input_hash":sorrel_data::devdb::truth_hash(&data)?,
                "truth_hash":sorrel_data::devdb::truth_hash(&truth_dir)?}),
            );
        }
        let exe = std::env::current_exe()?;
        let report = if method == CoverageMethod::FeatureOracle {
            json!({"diagnostic":"DEV human-truth-fitted feature audit, NOT a deployable proposer or curation performance",
                "binary_blake3":super::runner::file_hash(&exe)?,"method":method.name(),
                "alignment":"original NPY rows through validated permutation and input-root crosscheck",
                "fitting":"even session-global spikes; class-weighted <=2500 samples per class per axis; feature selected by training Jaccard",
                "validation":"odd spikes of the SAME dev recording; correlated spikes, NOT leave-session-out validation",
                "limitation":"sampled axis search is not a proof of non-separability; do not copy fitted thresholds into production",
                "n_good":totals[0],"sessions":reports})
        } else {
            json!({"diagnostic":"dev-only oracle candidate coverage, NOT curation performance",
            "binary_blake3":super::runner::file_hash(&exe)?,"match_jaccard":0.8,
            "alignment":"original NPY rows through validated permutation and input-root crosscheck",
            "method":method.name(),"n_good":totals[0],"whole_reachable":totals[1],"with_proposer_reachable":totals[2],"sessions":reports})
        };
        let out =
            self.root
                .join("reports")
                .join(format!("coverage-{}-{}.json", now(), &opaque()[..8]));
        write_json(&out, &report)?;
        println!("{}", out.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truth_alignment_preserves_original_rows_at_simultaneous_spike_ties() {
        let roots = [1, 0, 1, 0];
        let truth = [10, 20, 30, 40];
        let globals = [0, 0, 1, 1];
        let rows = [Some(1), Some(3), Some(0), Some(2)];
        assert_eq!(
            aligned_truth(&rows, &truth, &roots, &globals).unwrap(),
            vec![20, 40, 10, 30]
        );
        assert!(aligned_truth(
            &[Some(0), Some(1), Some(2), Some(3)],
            &truth,
            &roots,
            &globals
        )
        .is_err());
        assert!(aligned_truth(
            &[Some(1), Some(1), Some(0), Some(2)],
            &truth,
            &roots,
            &globals
        )
        .is_err());
        assert!(
            aligned_truth(&[Some(1), None, Some(0), Some(2)], &truth, &roots, &globals).is_err()
        );
        assert!(aligned_truth(
            &[Some(1), Some(4), Some(0), Some(2)],
            &truth,
            &roots,
            &globals
        )
        .is_err());
    }
    #[test]
    fn selection_cannot_read_holdout_and_jaccard_counts_foreign_root_spikes() {
        let mut map = SessionMap::default();
        for (sid, split) in [
            ("a", Split::Holdout),
            ("b", Split::Dev),
            ("c", Split::Holdout),
            ("d", Split::Dev),
        ] {
            map.sessions.insert(
                sid.into(),
                SessionEntry {
                    path: "unreadable".into(),
                    split,
                    forced_dev: false,
                    pure_manual: false,
                    keep: vec![],
                    fingerprint: String::new(),
                    recuration_of: None,
                },
            );
        }
        assert_eq!(dev_sessions(&map, 1).unwrap(), vec!["b"]);
        assert_eq!(dev_sessions(&map, 8).unwrap(), vec!["b", "d"]);
        assert!(dev_sessions(&map, 0).is_err());
        assert_eq!(jaccard(80, 100, 120), 80.0 / 140.0);
        assert_eq!(jaccard(0, 0, 0), 0.0);
    }
}
