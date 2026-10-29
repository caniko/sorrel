//! `brief` on the committed tiny phy fixture: size bound for L0+L1.

use sorrel_data::{
    collect_qc_rows_with, open_phy, rank_merge_candidates, rank_split_candidates, render_brief,
    seed_phy, BriefOpts, JournalOpts, SuggestConfig,
};
use sorrel_io::kilosort::KilosortOpenParams;
use std::path::Path;

#[test]
fn tiny_phy_brief_is_small_and_deterministic() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/tiny-phy");
    if !src.exists() {
        eprintln!("skipping: fixture not in this source tree");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    for f in std::fs::read_dir(&src).unwrap() {
        let f = f.unwrap();
        std::fs::copy(f.path(), dir.path().join(f.file_name())).unwrap();
    }
    let mut s = open_phy(
        dir.path(),
        KilosortOpenParams::default(),
        &JournalOpts::default(),
    )
    .unwrap();
    seed_phy(&mut s).unwrap();
    let rows = collect_qc_rows_with(&s, None);
    let cfg = SuggestConfig::default();
    let (m, sp) = (
        rank_merge_candidates(&s, &cfg),
        rank_split_candidates(&s, &cfg),
    );
    let brief = |level| {
        let o = BriefOpts {
            level,
            ..Default::default()
        };
        render_brief(&s, &rows, &m, &sp, &[], |l| l.as_str(), dir.path(), &o)
    };
    let l1 = brief(1);
    eprintln!("tiny-phy L0+L1: {} bytes", l1.len());
    assert!(l1.len() < 4000, "{}", l1.len());
    assert!(brief(0).len() < l1.len());
    assert_eq!(l1, brief(1));
}
