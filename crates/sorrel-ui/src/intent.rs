use sorrel_data::PhyLabelOp;
use sorrel_io::ClusterId;

/// Curation intents. Keyboard defaults and descriptions live in `shortcuts`;
/// buttons, snippets and agents can invoke the same intents without keys.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Intent {
    SelectCluster(ClusterId),
    ToggleCluster(ClusterId),
    ExtendCluster(ClusterId),
    Move(MoveTarget, PhyLabelOp),
    Undo,
    Redo,
    Save,
    NextCluster,
    PrevCluster,
    FirstCluster,
    LastCluster,
    NextSimilar,
    PrevSimilar,
    SelectFirstSimilar,
    UnselectSimilar,
    ResetWizard,
    PageBack,
    PageForward,
    TraceStart,
    TraceEnd,
    TraceNarrow,
    TraceWiden,
    ToggleSpikeReorder,
    ToggleMergeMode,
    TraceOrigin,
    TraceAllSpikes,
    TraceLabels,
    TraceGain(i32),
    NextSpike,
    PrevSpike,
    MergeSelection,
    Cancel,
    ToggleHelp,
    ToggleMaximize,
    Split,
    SplitInit,
    ToggleFeatureChannels,
    CycleAmplitudes(i32),
    Snippet(&'static str),
    Quit,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MoveTarget {
    Best,
    Similar,
    All,
}
