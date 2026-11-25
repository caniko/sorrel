use sorrel_data::{
    baseline_hash, open_journal, save_session, CurationCommand, JournalOpts, Method, PhyLabelOp,
    Session,
};
use sorrel_io::kilosort::PhyLabel;
use sorrel_io::neuralynx::{write_ntt, NttFixture};
use sorrel_io::{ClusterId, DataProvider, NttProvider, SampleIndex};
use std::path::Path;

fn open(path: &Path, journal: &Path) -> anyhow::Result<Session<NttProvider>> {
    let p = NttProvider::open(path)?;
    let seal = baseline_hash(&p.identity_bytes());
    let j = open_journal(&JournalOpts::default(), journal, seal, p.n_clusters())?;
    let mut s = Session::new(p, j);
    s.seed_amplitudes();
    s.replay_journal()?;
    Ok(s)
}

#[test]
fn pool_is_cut_saved_and_replayed_against_the_original_content() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("TT1.ntt");
    let journal = dir.path().join("TT1.ntt.sorrel.journal");
    let artifacts = dir.path().join("TT1.ntt.sorrel");
    std::fs::create_dir(&artifacts).unwrap();
    let mut fixture = NttFixture {
        spikes: (0..4)
            .map(|i| {
                let mut wave = [[0; 4]; 32];
                wave[7][0] = i as i16 + 1;
                (1_000 + i * 1_000, wave)
            })
            .collect(),
        ..Default::default()
    };
    write_ntt(&path, &fixture).unwrap();
    let original = std::fs::read(&path).unwrap();
    let pool = ClusterId(0);
    let cut = ClusterId(1);
    {
        let mut s = open(&path, &journal).unwrap();
        assert_eq!(s.n_clusters(), 1);
        assert_eq!(s.label(pool), Some(PhyLabel::Unsorted));
        assert_eq!(s.spike_times(pool), [7, 39, 71, 103].map(SampleIndex));
        assert_eq!(s.spike_amplitudes(pool), [1., 2., 3., 4.]);
        let err = sorrel_data::raw_waveform::measure(&s, pool).err().unwrap();
        assert!(
            err.contains("need continuous data") && err.contains("32-sample"),
            "{err}"
        );

        s.dispatch(CurationCommand::Split {
            cluster: pool,
            spike_idx: vec![1, 3],
            new_cluster: pool, // assigned by dispatch
            by: Method::Manual,
        })
        .unwrap();
        s.dispatch(CurationCommand::Relabel {
            cluster: cut,
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        assert_eq!(s.n_clusters(), 2);
        assert_eq!(s.spike_times(pool), [7, 71].map(SampleIndex));
        assert_eq!(s.spike_times(cut), [39, 103].map(SampleIndex));
        let mut snippet = [0.; 128];
        assert!(s
            .provider
            .stored_snippet(s.spike_times(cut)[1], &mut snippet));
        assert_eq!(snippet[28], -4.);
        save_session(&mut s, &artifacts, false, |label| label.as_str()).unwrap();
        let assignments =
            sorrel_io::npy::read_1d_u32(&artifacts.join("spike_clusters.npy")).unwrap();
        assert_eq!(assignments, [0, 1, 0, 1]);
        assert!(!dir.path().join("spike_clusters.npy").exists());
    }
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let copy = dir.path().join("renamed.ntt");
    std::fs::copy(&path, &copy).unwrap();
    {
        let s = open(&copy, &journal).unwrap();
        assert_eq!(s.history_len(), 2);
        assert_eq!(s.spike_times(pool), [7, 71].map(SampleIndex));
        assert_eq!(s.spike_amplitudes(cut), [2., 4.]);
        assert_eq!(s.label(cut), Some(PhyLabel::Good));
    }
    fixture.spikes[0].1[7][0] += 1;
    write_ntt(&copy, &fixture).unwrap();
    let err = open(&copy, &journal).err().unwrap().to_string();
    assert!(err.contains("sealed against different data"), "{err}");
}
