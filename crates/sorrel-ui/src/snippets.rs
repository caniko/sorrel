//! phy snippets: `:` opens a one-line command bar (`l` opens it with
//! `:l ` typed). The text is parsed here and run by the app.

use sorrel_io::ClusterId;

#[derive(Clone, Debug, PartialEq)]
pub enum Snippet {
    /// `:g` merge the selection.
    Merge,
    /// `:k` split off the lassoed spikes.
    Split,
    /// `:c 3 7 10-12` select these clusters.
    Select(Vec<ClusterId>),
    /// `:l name value` set label `name` on the selection (`group` is the
    /// good/mua/noise/unsorted label).
    Label { name: String, value: String },
    /// `:note text` set the note of every selected cluster (empty clears).
    Note(String),
    /// `:f expr` filter the cluster table.
    Filter(String),
    /// `:s column` sort the cluster table.
    Sort(String),
    /// `:tg 12.5` move the trace to this time (s).
    TraceGoto(f32),
    /// `:ts 0.2` set the trace window length (s).
    TraceWindow(f32),
    /// `:csx amp`, `:csy depth`, `:css fr`: ClusterScatter x / y / size.
    ScatterAxis(usize, String),
    /// `:km`, `:km 3`, `:km auto`: FeatureView k-means lasso (optionally
    /// setting k; `auto` picks it by silhouette).
    KMeans(Option<Option<usize>>),
    /// `:frn 200`: FiringRate bin count.
    RateBins(usize),
    /// `:frb 10`: FiringRate bin size in seconds.
    RateBinSize(f32),
    /// `:cf group`: phy colour field for unselected clusters.
    ColourField(String),
    /// `:cm name`: phy colormap.
    Colormap(String),
}

/// Parse one snippet; the leading `:` is optional. Errors are shown to the
/// user as is.
pub fn parse(text: &str) -> Result<Snippet, String> {
    let text = text.trim().trim_start_matches(':').trim_start();
    let (head, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    let rest = rest.trim();
    let number = |what: &str| -> Result<f32, String> {
        rest.parse::<f32>()
            .ok()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .ok_or_else(|| format!(":{head} needs {what}"))
    };
    Ok(match head {
        "g" => Snippet::Merge,
        "k" => Snippet::Split,
        "c" => Snippet::Select(cluster_list(rest)?),
        "l" => {
            let (name, value) = rest
                .split_once(char::is_whitespace)
                .ok_or(":l needs a label name and a value, e.g. :l group good")?;
            Snippet::Label {
                name: name.to_string(),
                value: value.trim().to_string(),
            }
        }
        "note" => Snippet::Note(rest.to_string()),
        "f" => Snippet::Filter(rest.to_string()),
        "s" if !rest.is_empty() => Snippet::Sort(rest.to_string()),
        "tg" => Snippet::TraceGoto(number("a time in seconds")?),
        "ts" => Snippet::TraceWindow(number("a window length in seconds")?),
        "km" => Snippet::KMeans(match rest {
            "" => None,
            "auto" => Some(None),
            n => Some(Some(
                n.parse()
                    .ok()
                    .filter(|k| (2..=8).contains(k))
                    .ok_or(":km takes k from 2 to 8, or auto")?,
            )),
        }),
        "frb" => Snippet::RateBinSize(number("a bin size in seconds")?),
        "frn" => Snippet::RateBins(
            rest.parse()
                .ok()
                .filter(|n| (2..=5000).contains(n))
                .ok_or(":frn needs a bin count between 2 and 5000")?,
        ),
        "cf" if !rest.is_empty() => Snippet::ColourField(rest.to_string()),
        "cm" if !rest.is_empty() => Snippet::Colormap(rest.to_string()),
        "csx" | "csy" | "css" if !rest.is_empty() => Snippet::ScatterAxis(
            ["csx", "csy", "css"]
                .iter()
                .position(|&h| h == head)
                .unwrap_or(0),
            rest.to_string(),
        ),
        "" => return Err("type a snippet, e.g. :c 3 7".into()),
        other => return Err(format!("unknown snippet :{other}")),
    })
}

/// `3 7, 10-12` → clusters 3, 7, 10, 11, 12 (phy accepts spaces, commas
/// and inclusive ranges).
fn cluster_list(s: &str) -> Result<Vec<ClusterId>, String> {
    let mut out = Vec::new();
    for tok in s
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
    {
        let id = |t: &str| {
            t.parse::<u32>()
                .map_err(|_| format!("not a cluster id: {t}"))
        };
        match tok.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (id(a)?, id(b)?);
                if b < a || b - a > 10_000 {
                    return Err(format!("bad range {tok}"));
                }
                out.extend((a..=b).map(ClusterId));
            }
            None => out.push(ClusterId(id(tok)?)),
        }
    }
    if out.is_empty() {
        return Err(":c needs cluster ids, e.g. :c 3 7".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_phy_snippets() {
        assert_eq!(parse(":g"), Ok(Snippet::Merge));
        assert_eq!(parse("k"), Ok(Snippet::Split));
        assert_eq!(
            parse(":c 3, 7 10-12"),
            Ok(Snippet::Select([3, 7, 10, 11, 12].map(ClusterId).to_vec()))
        );
        assert_eq!(
            parse(":l group  good"),
            Ok(Snippet::Label {
                name: "group".into(),
                value: "good".into()
            })
        );
        assert_eq!(
            parse(":f n_spikes > 1000 && depth < 500"),
            Ok(Snippet::Filter("n_spikes > 1000 && depth < 500".into()))
        );
        assert_eq!(
            parse(":note  needs a look "),
            Ok(Snippet::Note("needs a look".into()))
        );
        assert_eq!(parse(":note"), Ok(Snippet::Note(String::new())));
        assert_eq!(parse(":s n_spikes"), Ok(Snippet::Sort("n_spikes".into())));
        assert_eq!(parse(":tg 12.5"), Ok(Snippet::TraceGoto(12.5)));
        assert_eq!(parse(":cf group"), Ok(Snippet::ColourField("group".into())));
        assert_eq!(parse(":frn 50"), Ok(Snippet::RateBins(50)));
        assert_eq!(parse(":km"), Ok(Snippet::KMeans(None)));
        assert_eq!(parse(":km 3"), Ok(Snippet::KMeans(Some(Some(3)))));
        assert_eq!(parse(":km auto"), Ok(Snippet::KMeans(Some(None))));
        assert!(parse(":km 1").is_err());
        assert_eq!(parse(":frb 2.5"), Ok(Snippet::RateBinSize(2.5)));
        assert!(parse(":frn 1").is_err());
        assert_eq!(
            parse(":css n_spikes"),
            Ok(Snippet::ScatterAxis(2, "n_spikes".into()))
        );
        assert!(parse(":c 5-2").is_err());
        assert!(parse(":c x").is_err());
        assert!(parse(":l group").is_err());
        assert!(parse(":tg -1").is_err());
        assert!(parse(":zz").is_err());
    }
}
