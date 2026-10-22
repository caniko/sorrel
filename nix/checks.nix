{
  craneLib,
  commonArgs,
  cargoArtifacts,
  sorrel,
}: {
  default = sorrel;

  clippy = craneLib.cargoClippy (commonArgs
    // {
      inherit cargoArtifacts;
      cargoClippyExtraArgs = "--workspace --all-targets -- --deny warnings";
    });
}
