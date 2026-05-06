use sorrel_data::PhyLabelOp;
use sorrel_io::ClusterId;

/// User intents — a closed enum, dispatched in a single `match`. No heap, no
/// virtual calls in the input path.
#[derive(Copy, Clone, Debug)]
pub enum Intent {
    SelectCluster(ClusterId),
    Relabel(ClusterId, PhyLabelOp),
    Undo,
    Redo,
    NextCluster,
    PrevCluster,
    PageBack,
    PageForward,
}

pub fn from_key(key: egui::Key, modifiers: egui::Modifiers, selected: ClusterId) -> Option<Intent> {
    use egui::Key::*;
    Some(match key {
        G => Intent::Relabel(selected, PhyLabelOp::SetGood),
        M => Intent::Relabel(selected, PhyLabelOp::SetMua),
        N => Intent::Relabel(selected, PhyLabelOp::SetNoise),
        U => Intent::Relabel(selected, PhyLabelOp::SetUnsorted),
        ArrowDown | J => Intent::NextCluster,
        ArrowUp | K => Intent::PrevCluster,
        ArrowLeft | H => Intent::PageBack,
        ArrowRight | L => Intent::PageForward,
        Z if modifiers.command && modifiers.shift => Intent::Redo,
        Z if modifiers.command => Intent::Undo,
        _ => return None,
    })
}
