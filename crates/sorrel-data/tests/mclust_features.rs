//! MClust behavior across the real phy reader, journal, metrics, and rules.
#[path = "support/mclust_phy.rs"]
mod mclust_phy;

use std::{fs, path::Path};

use mclust_phy::{write_phy, DUPLICATE_TIMES, SPIKES_PER_UNIT};
use sorrel_data::rules::RuleSet;
use sorrel_data::{
    apply_rules, collect_qc_rows, eval_merge, open_phy, pairwise_cluster_separation, save_session,
    seed_phy, Actor, CurationCommand, JournalOpts, PhyLabelOp, Session,
};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
use sorrel_io::{ClusterId, SampleIndex};

const A: ClusterId = ClusterId(0);
const B: ClusterId = ClusterId(1);
const CHILD: ClusterId = ClusterId(2);

fn open(root: &Path) -> Session<KilosortProvider> {
    let mut session =
        open_phy(root, KilosortOpenParams::default(), &JournalOpts::default()).unwrap();
    seed_phy(&mut session).unwrap();
    session
}

#[test]
fn dedup_save_replay_and_undo_redo_preserve_membership_and_annotations() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("phy");
    write_phy(&root, 20.0);
    let mut session = open(&root);
    session
        .dispatch(CurationCommand::Tag {
            cluster: A,
            tags: ["src:sorter", "review:split"].map(String::from).into(),
        })
        .unwrap();
    let original = session.spike_clusters_global().to_vec();
    let indices = sorrel_compute::duplicate_spike_indices(
        session.spike_times(A),
        session.spike_amplitudes(A),
        6,
    );
    assert_eq!(indices, [0, 3]);
    let actor = Actor::Agent {
        client: "test".into(),
    };
    let commands = session.dedup_split_commands(A, indices, CHILD, &actor);
    session
        .dispatch_as(actor, CurationCommand::batch(commands.into()).unwrap())
        .unwrap();
    let mut expected = original.clone();
    // The two weaker spikes are original NPY rows 0 and 4. The stronger
    // partners at rows 1 and 3 must stay in the parent.
    expected[0] = CHILD;
    expected[4] = CHILD;
    assert_eq!(session.spike_clusters_global(), expected);
    save_session(&mut session, &root, true, |label| label.as_str()).unwrap();
    assert_eq!(
        sorrel_io::npy::read_1d_u32(&root.join("spike_clusters.npy")).unwrap(),
        expected.iter().map(|id| id.0).collect::<Vec<_>>()
    );
    drop(session);

    let check_split = |session: &Session<KilosortProvider>| {
        assert_eq!(session.spike_clusters_global(), expected);
        assert_eq!(session.spike_times(A).len(), SPIKES_PER_UNIT);
        assert_eq!(session.spike_times(CHILD), DUPLICATE_TIMES.map(SampleIndex));
        assert_eq!(session.spike_amplitudes(CHILD), [0.25, 0.5]);
        assert_eq!(session.label(CHILD), Some(PhyLabel::Noise));
        assert_eq!(
            session.tags(CHILD),
            ["src:sorter", "review:split", "src:agent", "duplicate"]
        );
        let qc = collect_qc_rows(session);
        assert_eq!(
            qc.iter()
                .find(|r| r.cluster_id == A.0)
                .unwrap()
                .duplicate_spikes,
            0
        );
    };
    let mut reopened = open(&root);
    check_split(&reopened);
    reopened.dispatch(CurationCommand::Undo).unwrap();
    assert_eq!(reopened.n_clusters(), 2);
    assert_eq!(reopened.spike_clusters_global(), original);
    assert_eq!(reopened.tags(A), ["src:sorter", "review:split"]);
    reopened.dispatch(CurationCommand::Redo).unwrap();
    check_split(&reopened);
}

#[test]
fn duplicate_spikes_rule_matches_only_the_unit_with_duplicates() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("phy");
    write_phy(&root, 20.0);
    let session = open(&root);
    let rules = tmp.path().join("rules.toml");
    fs::write(
        &rules,
        r#"[[rule]]
metric = "duplicate_spikes"
op = ">"
value = 0
label = "mua"
"#,
    )
    .unwrap();
    let commands = apply_rules(
        &RuleSet::from_path(&rules).unwrap(),
        &collect_qc_rows(&session),
        |id| session.label(id).unwrap().as_str(),
    )
    .unwrap();
    assert!(
        matches!(
            commands.as_slice(),
            [CurationCommand::Relabel {
                cluster: A,
                op: PhyLabelOp::SetMua
            }]
        ),
        "{commands:?}"
    );
}

#[test]
fn merge_separation_aligns_pc_slots_on_physical_channels() {
    const KEYS: [&str; 4] = [
        "feature_d_prime",
        "feature_bhattacharyya",
        "feature_overlap_est",
        "feature_l_ratio",
    ];
    let tmp = tempfile::tempdir().unwrap();
    for (name, shift) in [("separate", 20.0), ("overlap", 0.0), ("no-pc", 0.0)] {
        let root = tmp.path().join(name);
        write_phy(&root, shift);
        if name == "no-pc" {
            fs::remove_file(root.join("pc_features.npy")).unwrap();
        }
        let session = open(&root);
        let sep = pairwise_cluster_separation(&session, A, B);
        let eval = eval_merge(&session, &[A, B]);
        assert_eq!(eval.pairs.len(), 1);
        let json = serde_json::to_value(eval).unwrap();
        let pair = &json["pairs"][0];
        if name == "no-pc" {
            assert!(sep.is_none());
            for key in KEYS {
                assert!(pair.get(key).is_none(), "{key}: {pair}");
            }
            continue;
        }
        let sep = sep.unwrap();
        assert_eq!(
            (sep.n_a as usize, sep.n_b as usize),
            (SPIKES_PER_UNIT + DUPLICATE_TIMES.len(), SPIKES_PER_UNIT)
        );
        if name == "separate" {
            assert!(sep.d_prime > 5.0 && sep.overlap_est < 0.01, "{sep:?}");
        } else {
            assert!(sep.d_prime < 0.2 && sep.overlap_est > 0.98, "{sep:?}");
        }
        for (key, expected) in KEYS.into_iter().zip([
            sep.d_prime,
            sep.bhattacharyya,
            sep.overlap_est,
            sep.l_ratio_ab,
        ]) {
            assert!(expected.is_finite(), "{key}: {sep:?}");
            assert_eq!(pair[key].as_f64(), Some(f64::from(expected)), "{key}");
        }
    }
}
