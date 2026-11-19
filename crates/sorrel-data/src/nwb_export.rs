//! Build the same curated Units payload for the CLI and PyNWB.

use crate::{collect_qc_rows_with, provenance, Session};
use anyhow::{ensure, Result};
use sorrel_io::curation_json::unit_id_text;
use sorrel_io::extras::QualityMetrics;
use sorrel_io::kilosort::PhyLabel;
use sorrel_io::nwb_units::{Column, Units, Values};
use sorrel_io::{ClusterId, DataProvider, HasGeometry};
use std::collections::BTreeMap;

/// `rate` and `starting_time` place the recording's sample indices on the NWB time axis.
pub fn collect_units<P>(
    session: &Session<P>,
    rate: f64,
    starting_time: f64,
    stored_metrics: Option<&QualityMetrics>,
) -> Result<Units>
where
    P: DataProvider<Label = PhyLabel> + HasGeometry,
{
    ensure!(
        rate.is_finite() && rate > 0.0 && (1.0 / rate).is_finite(),
        "invalid sample rate {rate}"
    );
    ensure!(starting_time.is_finite(), "starting_time must be finite");
    let clusters: Vec<_> = (0..session.n_clusters())
        .map(ClusterId)
        .filter(|&c| !session.spike_times(c).is_empty())
        .collect();
    ensure!(!clusters.is_empty(), "the session has no units with spikes");
    let provider = &session.provider;
    let id: Vec<_> = clusters
        .iter()
        .map(|&c| i64::from(provider.unit_id(c)))
        .collect();
    let times = ragged(clusters.iter().map(|&c| {
        session
            .spike_times(c)
            .iter()
            .map(|s| s.0 as f64 / rate + starting_time)
    }));
    ensure!(
        times.0.iter().all(|t| t.is_finite()),
        "spike times overflow the NWB time axis"
    );
    let parents = ragged(clusters.iter().map(|&c| {
        session
            .parents(c)
            .iter()
            .map(|&p| i64::from(provider.unit_id(p)))
            .chain(session.missing_parent_unit_ids(c).iter().copied())
    }));
    let tags = ragged(clusters.iter().map(|&c| session.tags(c).iter().cloned()));
    let mut columns = vec![
        column("spike_times", "Spike times in seconds: sample_index / sample_rate + starting_time.", Values::Float(times.0), Some(times.1)),
        column("quality", "Sorrel curation label: good, mua, noise or unsorted.", Values::Text(clusters.iter().map(|c| session.labels()[c.idx()].as_str().to_owned()).collect()), None),
        column("origin", "How the unit was made: Auto[<sorter>] if the sorter produced it, Sorrel[<method>] if a Sorrel split created it (Manual, Algo:<name>, LLM:<name>).", Values::Text(clusters.iter().map(|&c| session.origin_label(c)).collect()), None),
        column("parent_unit_ids", "Unit ids this unit came from in Sorrel: the unit it was split from, or the units merged into it. A parent with no spikes left has no row in this table.", Values::Int(parents.0), Some(parents.1)),
        column("tags", "Sorrel tags: src:* is who last touched the unit, review:* asks for review, split/tuned mark structural edits.", Values::Text(tags.0), Some(tags.1)),
    ];

    let names: Vec<_> = clusters.iter().map(|&c| unit_name(provider, c)).collect();
    if names.iter().zip(&id).any(|(n, id)| *n != id.to_string()) {
        columns.push(column(
            "unit_name",
            "The sorter's unit id (SpikeInterface unit_ids) when it is not an integer.",
            Values::Text(names),
            None,
        ));
    }

    let qc = std::cell::OnceCell::new();
    let computed = |c: ClusterId, name: &str| {
        let rows =
            qc.get_or_init(|| collect_qc_rows_with(session, Some(provider.channel_positions())));
        rows[c.idx()]
            .get(name)
            .filter(|v| v.is_finite())
            .unwrap_or(f32::NAN)
    };
    if let Some(stored) = stored_metrics.filter(|m| !m.is_empty()) {
        for name in stored.names() {
            if name == "id"
                || columns.iter().any(|c| {
                    c.name == *name
                        || (c.index.is_some() && name.strip_suffix("_index") == Some(&c.name))
                })
            {
                continue;
            }
            // Imported metrics describe the original trains. Keep them for
            // unchanged rows; recompute what we can for edited trains.
            let original = stored.get(name).unwrap_or_default();
            let values = clusters
                .iter()
                .map(|&c| {
                    let v = if session.spike_times(c) == provider.spike_times(c) {
                        original.get(c.idx()).copied().unwrap_or(f32::NAN)
                    } else {
                        computed(c, name)
                    };
                    f64::from(v)
                })
                .collect();
            let doc = metric_docs().find(|(key, _)| *key == name).map_or(
                "Metric imported from the source Units table.",
                |(_, doc)| doc,
            );
            columns.push(column(
                name,
                format!("Sorrel QC metric. {doc}"),
                Values::Float(values),
                None,
            ));
        }
    } else {
        for (name, doc) in metric_docs() {
            let values: Vec<_> = clusters
                .iter()
                .map(|&c| f64::from(computed(c, name)))
                .collect();
            if values.iter().any(|v| !v.is_nan()) {
                let data = match name {
                    "n_spikes" | "isi_violations" => {
                        Values::Int(values.into_iter().map(|v| v as i64).collect())
                    }
                    _ => Values::Float(values),
                };
                columns.push(column(name, format!("Sorrel QC metric. {doc}"), data, None));
            }
        }
    }

    let p = provenance(session)?;
    let version = env!("CARGO_PKG_VERSION");
    let description = format!(
        "Units from {} spike sorting, curated in Sorrel {version}. Curation provenance: \
         journal_xxh3={}, head_sealed={}, edits on the applied path={} of {} in the history; \
         by actor: {}; by method: {}; pure_manual={}, machine_anchored={}.",
        session.sorter(),
        p.journal_xxh3,
        p.sealed,
        p.n_applied,
        p.n_nodes,
        counts(&p.by_actor),
        counts(&p.by_method),
        p.pure_manual,
        p.anchored,
    );
    Ok(Units {
        description,
        resolution: 1.0 / rate,
        starting_time,
        id,
        columns,
        generated_by: ["sorrel-ephys", version],
    })
}

pub fn unit_name<P: DataProvider>(provider: &P, cluster: ClusterId) -> String {
    let id = provider.unit_id_json(cluster);
    unit_id_text(&id).unwrap_or_else(|| id.to_string())
}

fn ragged<T>(rows: impl Iterator<Item = impl IntoIterator<Item = T>>) -> (Vec<T>, Vec<u64>) {
    let mut data = Vec::new();
    let index = rows
        .map(|row| {
            data.extend(row);
            data.len() as u64
        })
        .collect();
    (data, index)
}

fn column(
    name: &str,
    description: impl Into<String>,
    data: Values,
    index: Option<Vec<u64>>,
) -> Column {
    Column {
        name: name.to_owned(),
        description: description.into(),
        data,
        index,
    }
}

fn metric_docs() -> impl Iterator<Item = (&'static str, &'static str)> {
    include_str!("nwb_metrics.tsv")
        .lines()
        .filter_map(|line| line.split_once('\t'))
}

fn counts(map: &BTreeMap<String, u32>) -> String {
    if map.is_empty() {
        return "none".to_owned();
    }
    map.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    #[test]
    fn documents_all_qc_metrics() {
        assert_eq!(
            super::metric_docs()
                .map(|(name, _)| name)
                .collect::<Vec<_>>(),
            crate::QcRow::METRICS
        );
    }
}
