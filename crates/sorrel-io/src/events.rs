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
//! NWB `/intervals/trials` is the natural third source and belongs in the
//! `hdf5`-gated reader; it is not wired yet.

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
            let idx = match ev.names.iter().position(|x| x == name) {
                Some(i) => i,
                None => {
                    ev.names.push(name.to_string());
                    ev.names.len() - 1
                }
            };
            ev.times_s.push(t);
            ev.label.push(idx as u16);
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
}
