//! ElectricalSeries ancestry from cached specifications in either NWB backend.

use super::storage::Group;
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

fn collect_parents(v: &Value, add: &mut impl FnMut(&str, &str)) {
    match v {
        Value::Object(m) => {
            if let (Some(def), Some(inc)) = (
                m.get("neurodata_type_def").and_then(Value::as_str),
                m.get("neurodata_type_inc").and_then(Value::as_str),
            ) {
                add(def, inc);
            }
            m.values().for_each(|v| collect_parents(v, add));
        }
        Value::Array(a) => a.iter().for_each(|v| collect_parents(v, add)),
        _ => {}
    }
}

fn read_parents(f: &Group, legacy: bool, mut add: impl FnMut(&str, &str)) -> Result<()> {
    let Some(specs) = f.spec_group(legacy)? else {
        return Ok(());
    };
    for (name, ns) in specs.spec_groups()? {
        for (version_name, version) in ns.spec_groups()? {
            for (source, ds) in version.spec_datasets()? {
                let doc = ds
                    .spec_text(legacy)
                    .and_then(|t| Ok(serde_json::from_str(&t)?));
                if legacy && doc.is_err() {
                    continue;
                }
                collect_parents(
                    &doc.with_context(|| {
                        format!(
                            "unreadable cached NWB specification {name}/{version_name}/{source}"
                        )
                    })?,
                    &mut add,
                );
            }
        }
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct TypeSpecs(HashMap<String, BTreeSet<String>>);

impl TypeSpecs {
    pub(super) fn read(f: &Group) -> Result<Self> {
        let mut out = Self::default();
        read_parents(f, false, |def, inc| {
            out.0
                .entry(def.to_owned())
                .or_default()
                .insert(inc.to_owned());
        })?;
        Ok(out)
    }

    pub(super) fn is_series(&self, ty: &str, refuse_conflicts: bool) -> Result<bool> {
        let mut ty = ty;
        // Bounded walk: a malformed spec could loop. Check the parent set
        // before accepting the terminal name, even for a direct ElectricalSeries.
        for _ in 0..32 {
            let parents = self.0.get(ty);
            if let Some(parents) = parents.filter(|p| p.len() > 1) {
                if !refuse_conflicts {
                    return Ok(false);
                }
                let parents = parents
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!("NWB type {ty} is defined with conflicting parents {parents} in the cached specifications; Sorrel cannot tell which applies");
            }
            if ty == "ElectricalSeries" {
                return Ok(true);
            }
            let Some(parent) = parents.and_then(BTreeSet::first) else {
                return Ok(false);
            };
            ty = parent;
        }
        Ok(false)
    }
}

/// Only for the legacy journal seal: reproduce trunk's last-wins walk.
pub(super) fn legacy_type_parents(f: &Group) -> Result<HashMap<String, String>> {
    let mut parents = HashMap::new();
    read_parents(f, true, |def, inc| {
        parents.insert(def.to_owned(), inc.to_owned());
    })?;
    Ok(parents)
}

pub(super) fn legacy_is_series(ty: &str, parents: &HashMap<String, String>) -> bool {
    let mut ty = ty;
    for _ in 0..32 {
        if ty == "ElectricalSeries" {
            return true;
        }
        let Some(parent) = parents.get(ty) else {
            return false;
        };
        ty = parent;
    }
    false
}
