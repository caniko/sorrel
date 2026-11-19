//! ElectricalSeries ancestry in the file's own namespace catalog.

use super::storage::Group;
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

pub(super) fn core_series(ns: &str, ty: &str) -> Option<bool> {
    if ns != "core" {
        return None;
    }
    match ty {
        "ElectricalSeries" => Some(true),
        "TimeSeries" | "ProcessingModule" | "LFP" | "FilteredEphys" | "NWBDataInterface"
        | "NWBContainer" => Some(false),
        _ => None,
    }
}

fn visit_objects(
    v: &Value,
    visit: &mut impl FnMut(&serde_json::Map<String, Value>) -> Result<()>,
) -> Result<()> {
    match v {
        Value::Object(m) => {
            visit(m)?;
            for child in m.values() {
                visit_objects(child, visit)?;
            }
        }
        Value::Array(a) => {
            for child in a {
                visit_objects(child, visit)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Only for the legacy journal seal: reproduce trunk's unqualified walk.
pub(super) fn legacy_type_parents(f: &Group) -> Result<HashMap<String, String>> {
    let mut parents = HashMap::new();
    let Some(specs) = f.spec_group(true)? else {
        return Ok(parents);
    };
    for (_, ns) in specs.spec_groups()? {
        for (_, version) in ns.spec_groups()? {
            for (_, ds) in version.spec_datasets()? {
                let Ok(spec) = ds
                    .spec_text(true)
                    .and_then(|t| Ok(serde_json::from_str(&t)?))
                else {
                    continue;
                };
                visit_objects(&spec, &mut |m| {
                    if let (Some(def), Some(inc)) = (
                        m.get("neurodata_type_def").and_then(Value::as_str),
                        m.get("neurodata_type_inc").and_then(Value::as_str),
                    ) {
                        parents.insert(def.to_owned(), inc.to_owned());
                    }
                    Ok(())
                })?;
            }
        }
    }
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

#[derive(Deserialize)]
struct NamespaceDocument {
    namespaces: Vec<Namespace>,
}

#[derive(Deserialize)]
struct Namespace {
    name: String,
    schema: Vec<SchemaEntry>,
}

#[derive(Deserialize)]
struct SchemaEntry {
    namespace: Option<String>,
    source: Option<String>,
    #[serde(rename = "neurodata_types", alias = "data_types")]
    types: Option<HashSet<String>>,
}

#[derive(Default, PartialEq, Eq)]
struct NamespaceSpec {
    parents: HashMap<String, Option<String>>,
    includes: HashMap<String, Option<HashSet<String>>>,
}

#[derive(Default)]
pub(super) struct TypeSpecs(HashMap<String, NamespaceSpec>);

impl TypeSpecs {
    pub(super) fn read(f: &Group) -> Result<Self> {
        let mut out = Self::default();
        let Some(specs) = f.spec_group(false)? else {
            return Ok(out);
        };
        for (name, ns) in specs.spec_groups()? {
            for (version_name, version) in ns.spec_groups()? {
                let mut docs = HashMap::<_, Value>::new();
                for (source, ds) in version.spec_datasets()? {
                    let doc = ds.spec_text(false).and_then(|t| Ok(serde_json::from_str(&t)?))
                        .with_context(|| format!("unreadable cached NWB specification {name}/{version_name}/{source}"))?;
                    docs.insert(source, doc);
                }
                let doc: NamespaceDocument =
                    serde_json::from_value(docs.remove("namespace").with_context(|| {
                        format!("missing cached namespace declaration for {name:?}")
                    })?)?;
                let [namespace] = doc.namespaces.as_slice() else {
                    bail!("expected one cached namespace declaration for {name:?}");
                };
                ensure!(
                    namespace.name == name,
                    "cached namespace name disagrees with {name:?}"
                );
                let spec = NamespaceSpec::from_schema(&namespace.schema, &docs)
                    .with_context(|| format!("invalid cached namespace {name}/{version_name}"))?;
                ensure!(
                    out.0.get(&name).is_none_or(|old| *old == spec),
                    "conflicting cached versions of namespace {name:?}"
                );
                out.0.insert(name.clone(), spec);
            }
        }
        Ok(out)
    }

    pub(super) fn is_series(&self, ns: Option<&str>, ty: &str) -> Result<bool> {
        let mut ty = ty;
        let mut ns = match ns {
            Some(ns) => ns,
            None => {
                // a557886 ignored namespace tags. Keep bare core names and
                // unqualified cached names, but never guess between owners.
                let mut owner = core_series("core", ty).map(|_| "core");
                for (ns, spec) in &self.0 {
                    if spec.parents.contains_key(ty) {
                        choose_owner(&mut owner, ns, ty)?;
                    }
                }
                owner.with_context(|| format!("no cached definition of {ty:?}"))?
            }
        };
        for _ in 0..32 {
            if let Some(answer) = core_series(ns, ty) {
                return Ok(answer);
            }
            let parent = self
                .0
                .get(ns)
                .and_then(|spec| spec.parents.get(ty))
                .with_context(|| format!("no cached definition of {ns}:{ty}"))?;
            let Some(parent) = parent else {
                return Ok(false);
            };
            ns = self.resolve_parent(ns, parent)?;
            ty = parent;
        }
        bail!("cyclic or too-deep cached ancestry at {ns}:{ty}")
    }

    fn resolve_parent<'a>(&'a self, ns: &'a str, ty: &str) -> Result<&'a str> {
        let mut pending = vec![ns];
        let mut seen = HashSet::new();
        let mut owner = None;
        while let Some(ns) = pending.pop() {
            if !seen.insert(ns) {
                continue;
            }
            let spec = self.0.get(ns);
            if core_series(ns, ty).is_some() || spec.is_some_and(|s| s.parents.contains_key(ty)) {
                // A local definition binds in its own namespace. HDMF's
                // cache writer can widen selected imports to whole namespaces.
                choose_owner(&mut owner, ns, ty)?;
            } else {
                let spec = spec
                    .with_context(|| format!("missing cached namespace {ns:?} resolving {ty:?}"))?;
                for (ns, types) in &spec.includes {
                    if types.as_ref().is_none_or(|types| types.contains(ty)) {
                        pending.push(ns);
                    }
                }
            }
        }
        owner.with_context(|| {
            format!("no cached definition of parent {ty:?} in {ns:?} or its includes")
        })
    }
}

fn choose_owner<'a>(owner: &mut Option<&'a str>, ns: &'a str, ty: &str) -> Result<()> {
    if let Some(other) = *owner {
        ensure!(
            other == ns,
            "ambiguous cached type {ty:?} in namespaces {other:?} and {ns:?}"
        );
    }
    *owner = Some(ns);
    Ok(())
}

impl NamespaceSpec {
    fn from_schema(schema: &[SchemaEntry], docs: &HashMap<String, Value>) -> Result<Self> {
        let mut spec = Self::default();
        for entry in schema {
            match (&entry.namespace, &entry.source) {
                (Some(ns), None) => {
                    ensure!(
                        spec.includes
                            .insert(ns.clone(), entry.types.clone())
                            .is_none(),
                        "duplicate import of {ns:?}"
                    );
                }
                (None, Some(source)) => {
                    let doc = docs
                        .get(source)
                        .with_context(|| format!("missing cached source {source:?}"))?;
                    ensure!(doc.is_object(), "invalid cached source {source:?}");
                    for kind in ["groups", "datasets"] {
                        if let Some(roots) = doc.get(kind) {
                            for root in roots
                                .as_array()
                                .context("invalid cached type definitions")?
                            {
                                let def = root
                                    .get("neurodata_type_def")
                                    .or_else(|| root.get("data_type_def"))
                                    .and_then(Value::as_str)
                                    .context("missing cached type name")?;
                                if entry.types.as_ref().is_none_or(|types| types.contains(def)) {
                                    spec.collect_parents(root)?;
                                }
                            }
                        }
                    }
                }
                _ => bail!("cached schema entry must specify either a source or a namespace"),
            }
        }
        Ok(spec)
    }

    fn collect_parents(&mut self, spec: &Value) -> Result<()> {
        visit_objects(spec, &mut |m| {
            if let Some(def) = m
                .get("neurodata_type_def")
                .or_else(|| m.get("data_type_def"))
            {
                let def = def.as_str().context("invalid cached type name")?;
                let inc = m
                    .get("neurodata_type_inc")
                    .or_else(|| m.get("data_type_inc"))
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .context("invalid cached parent type")
                    })
                    .transpose()?;
                ensure!(
                    self.parents.get(def).is_none_or(|old| *old == inc),
                    "conflicting cached definitions of {def:?}"
                );
                self.parents.insert(def.to_owned(), inc);
            }
            Ok(())
        })
    }
}
