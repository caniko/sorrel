//! Event / trial times for peri-event views.
//!
//! Sidecar conventions (first found wins when auto-detecting in a dataset
//! directory):
//!
//! - `events.csv`: header `time,label`; `time` in **seconds**, `label`
//!   free text (optional column; one unnamed label when absent). A tab or
//!   semicolon separator is accepted as for the metrics CSV.
//! - `events.npy`: 1-D `f64` (or `f32`) seconds, one unnamed label.
//!
//! An NWB file (`hdf5` feature) contributes `/intervals/trials/start_time`,
//! labelled by the first of `trial_type`, `stimulus`, `condition` that is a
//! string column. It is read by [`Events::read`] only, never auto-detected.

use crate::npy::read_1d_f64;
use crate::provider::SampleIndex;
use anyhow::{bail, Context, Result};
use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Events {
    /// Ascending event times in seconds.
    pub times_s: Vec<f64>,
    /// Index into `names`, parallel to `times_s`.
    pub label: Vec<u16>,
    pub names: Vec<String>,
}

impl Events {
    pub fn len(&self) -> usize {
        self.times_s.len()
    }
    pub fn is_empty(&self) -> bool {
        self.times_s.is_empty()
    }

    /// Event times as sample indices at `sample_rate`, for one label (or
    /// every label with `None`). Ascending.
    pub fn samples(&self, sample_rate: f32, label: Option<u16>) -> Vec<SampleIndex> {
        self.times_s
            .iter()
            .zip(&self.label)
            .filter(|(_, &l)| label.is_none_or(|want| want == l))
            .map(|(&t, _)| SampleIndex((t * sample_rate as f64).round().max(0.0) as u64))
            .collect()
    }

    /// `<root>/events.csv` or `<root>/events.npy`, if either exists.
    pub fn detect(root: &Path) -> Result<Option<Self>> {
        for name in ["events.csv", "events.npy"] {
            let p = root.join(name);
            if p.exists() {
                return Self::read(&p).map(Some);
            }
        }
        Ok(None)
    }

    pub fn read(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut ev = match ext.as_str() {
            "npy" => {
                let times_s = read_1d_f64(path)?;
                let n = times_s.len();
                Self {
                    times_s,
                    label: vec![0; n],
                    names: vec!["event".into()],
                }
            }
            "csv" | "tsv" | "txt" => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("read {}", path.display()))?;
                Self::parse_csv(&text)?
            }
            #[cfg(feature = "hdf5")]
            "nwb" => Self::read_nwb_trials(path)?,
            other => bail!("{}: unsupported events file type {other:?}", path.display()),
        };
        // Keep (time, label) pairs together while sorting by time.
        let mut order: Vec<usize> = (0..ev.len()).collect();
        order.sort_by(|&a, &b| ev.times_s[a].total_cmp(&ev.times_s[b]));
        ev.times_s = order.iter().map(|&i| ev.times_s[i]).collect();
        ev.label = order.iter().map(|&i| ev.label[i]).collect();
        Ok(ev)
    }

    fn parse_csv(text: &str) -> Result<Self> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().context("events file is empty")?;
        let sep = ['\t', ';', ',']
            .into_iter()
            .find(|&c| header.contains(c))
            .unwrap_or(',');
        let cols: Vec<&str> = header.split(sep).map(str::trim).collect();
        let t_col = cols
            .iter()
            .position(|c| matches!(c.to_ascii_lowercase().as_str(), "time" | "time_s" | "t"))
            .context("events file needs a `time` column (seconds)")?;
        let l_col = cols
            .iter()
            .position(|c| matches!(c.to_ascii_lowercase().as_str(), "label" | "event" | "type"));
        let mut ev = Self::default();
        for (n, line) in lines.enumerate() {
            let cells: Vec<&str> = line.split(sep).map(str::trim).collect();
            let t: f64 = cells
                .get(t_col)
                .and_then(|c| c.parse().ok())
                .with_context(|| format!("events line {}: bad time {:?}", n + 2, line))?;
            let name = l_col.and_then(|c| cells.get(c)).copied().unwrap_or("event");
            ev.push(t, name);
        }
        Ok(ev)
    }

    fn push(&mut self, t: f64, name: &str) {
        let idx = match self.names.iter().position(|x| x == name) {
            Some(i) => i,
            None => {
                self.names.push(name.to_string());
                self.names.len() - 1
            }
        };
        self.times_s.push(t);
        self.label.push(idx as u16);
    }

    #[cfg(feature = "hdf5")]
    fn read_nwb_trials(path: &Path) -> Result<Self> {
        use hdf5_metno::types::{VarLenAscii, VarLenUnicode};
        let f =
            hdf5_metno::File::open(path).with_context(|| format!("open NWB {}", path.display()))?;
        let g = f
            .group("intervals/trials")
            .with_context(|| format!("{}: no /intervals/trials table", path.display()))?;
        let starts: Vec<f64> = g.dataset("start_time")?.read_raw()?;
        let names: Option<Vec<String>> = ["trial_type", "stimulus", "condition"]
            .iter()
            .filter_map(|c| g.dataset(c).ok())
            .find_map(|ds| {
                ds.read_raw::<VarLenUnicode>()
                    .map(|v| v.iter().map(|s| s.to_string()).collect())
                    .or_else(|_| {
                        ds.read_raw::<VarLenAscii>()
                            .map(|v| v.iter().map(|s| s.to_string()).collect())
                    })
                    .ok()
            })
            .filter(|n: &Vec<String>| n.len() == starts.len());
        let mut ev = Self::default();
        for (i, &t) in starts.iter().enumerate() {
            ev.push(t, names.as_ref().map_or("trial", |n| n[i].as_str()));
        }
        Ok(ev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_with_labels_sorts_and_maps_to_samples() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("events.csv");
        std::fs::write(&p, "time,label\n2.0,go\n0.5,cue\n1.0,go\n").unwrap();
        let ev = Events::detect(dir.path()).unwrap().unwrap();
        assert_eq!(ev.times_s, vec![0.5, 1.0, 2.0]);
        assert_eq!(ev.names, vec!["go", "cue"]);
        assert_eq!(ev.label, vec![1, 0, 0]);
        assert_eq!(
            ev.samples(1000.0, Some(0)),
            vec![SampleIndex(1000), SampleIndex(2000)]
        );
        assert_eq!(ev.samples(1000.0, None).len(), 3);

        let bare = Events::parse_csv("t\n3\n").unwrap();
        assert_eq!(bare.names, vec!["event"]);
        assert!(Events::parse_csv("x,y\n1,2\n").is_err());
        assert!(Events::detect(&dir.path().join("missing"))
            .unwrap()
            .is_none());
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn nwb_trials_with_trial_type() {
        use hdf5_metno::types::VarLenUnicode;
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("s.nwb");
        {
            let f = hdf5_metno::File::create(&p).unwrap();
            let g = f.create_group("intervals/trials").unwrap();
            g.new_dataset_builder()
                .with_data(&[1.5f64, 0.5])
                .create("start_time")
                .unwrap();
            let kinds: Vec<VarLenUnicode> = ["go", "nogo"].map(|s| s.parse().unwrap()).into();
            g.new_dataset_builder()
                .with_data(&kinds)
                .create("trial_type")
                .unwrap();
        }
        let ev = Events::read(&p).unwrap();
        assert_eq!(ev.times_s, vec![0.5, 1.5]);
        assert_eq!(ev.names, vec!["go", "nogo"]);
        assert_eq!(ev.label, vec![1, 0]);
    }
}
