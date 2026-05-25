use rkyv::{Archive, Deserialize, Serialize};

pub const KIND: &str = "amplitudes";
pub const ALGO_VERSION: u32 = 1;

#[derive(Archive, Serialize, Deserialize, Debug, PartialEq)]
pub struct CachedAmplitudes {
    pub per_cluster: Vec<Vec<f32>>,
    pub algo_version: u32,
}
