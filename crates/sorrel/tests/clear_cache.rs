use std::process::Command;

#[test]
fn clear_cache_removes_dataset_cache_and_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join(".sorrel").join("cache").join("kind");
    std::fs::create_dir_all(&cache_dir).unwrap();
    std::fs::write(
        cache_dir.join(format!("{}.rkyv", "a".repeat(64))),
        b"cached",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sorrel"))
        .arg("--clear-cache")
        .arg(dir.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "sorrel --clear-cache failed: status={:?}, stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!dir.path().join(".sorrel").join("cache").exists());
}
