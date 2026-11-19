//! The typed Units payload shared by the HDF5 and PyNWB serializers.

#[cfg(feature = "hdf5")]
pub(crate) const STARTING_TIME_ATTR: &str = "sorrel_starting_time";

pub struct Units {
    pub description: String,
    pub resolution: f64,
    pub starting_time: f64,
    pub id: Vec<i64>,
    pub columns: Vec<Column>,
    pub generated_by: [&'static str; 2],
}

pub struct Column {
    pub name: String,
    pub description: String,
    pub data: Values,
    /// Exclusive end offsets for a ragged column, including empty rows.
    pub index: Option<Vec<u64>>,
}

pub enum Values {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Text(Vec<String>),
}
