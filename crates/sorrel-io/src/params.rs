//! Minimal `params.py` parser. phy's `params.py` is Python source, but the
//! subset Kilosort emits is a flat list of `key = value` assignments. We parse
//! only what we need; anything fancier (lists, expressions) is ignored.
//!
//! Supported value forms:
//! - integers and floats (`30000`, `30000.0`, `3e4`)
//! - bools (`True`/`False`)
//! - strings: `'..'`, `".."`, raw strings `r'..'` / `r".."`

use crate::provider::TraceDtype;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct PhyParams {
    pub dat_path: Option<PathBuf>,
    pub n_channels_dat: Option<u32>,
    pub dtype: Option<TraceDtype>,
    pub offset: Option<u64>,
    pub sample_rate: Option<f32>,
    pub hp_filtered: Option<bool>,
}

impl PhyParams {
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut out = Self::default();
        for (lineno, raw) in text.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim().trim_end_matches(';').trim();
            match key {
                "dat_path" => {
                    out.dat_path = parse_string(value).map(PathBuf::from);
                }
                "n_channels_dat" => {
                    out.n_channels_dat = Some(
                        parse_int(value)
                            .with_context(|| format!("params.py:{}: n_channels_dat", lineno + 1))?
                            as u32,
                    );
                }
                "dtype" => {
                    let s = parse_string(value).with_context(|| {
                        format!("params.py:{}: dtype must be a string", lineno + 1)
                    })?;
                    out.dtype = Some(TraceDtype::from_phy_name(&s).with_context(|| {
                        format!("params.py:{}: unsupported dtype {s:?}", lineno + 1)
                    })?);
                }
                "offset" => {
                    out.offset = Some(
                        parse_int(value)
                            .with_context(|| format!("params.py:{}: offset", lineno + 1))?
                            as u64,
                    );
                }
                "sample_rate" => {
                    out.sample_rate = Some(parse_float(value).with_context(|| {
                        format!("params.py:{}: sample_rate", lineno + 1)
                    })? as f32);
                }
                "hp_filtered" => {
                    out.hp_filtered = match value {
                        "True" => Some(true),
                        "False" => Some(false),
                        _ => bail!("params.py:{}: hp_filtered must be True/False", lineno + 1),
                    };
                }
                _ => { /* unknown keys are ignored — phy plugins may add their own */ }
            }
        }
        Ok(out)
    }
}

fn strip_comment(s: &str) -> &str {
    // Naive: comments only outside strings. params.py never has `#` inside the
    // values we care about, so this is fine in practice.
    match s.find('#') {
        Some(i) => &s[..i],
        None => s,
    }
}

fn parse_string(value: &str) -> Option<String> {
    let v = value.trim_start_matches('r').trim();
    let bytes = v.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let q = bytes[0];
    if (q != b'\'' && q != b'"') || bytes[bytes.len() - 1] != q {
        return None;
    }
    Some(v[1..v.len() - 1].to_string())
}

fn parse_int(value: &str) -> Result<i64> {
    Ok(value.parse::<i64>()?)
}

fn parse_float(value: &str) -> Result<f64> {
    // Allow trailing `.` (Python literal `30000.`).
    let v = if value.ends_with('.') {
        format!("{value}0")
    } else {
        value.to_string()
    };
    Ok(v.parse::<f64>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_kilosort_params() {
        let src = r#"
dat_path = r'recording.dat'
n_channels_dat = 384
dtype = 'int16'
offset = 0
sample_rate = 30000.
hp_filtered = False
"#;
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.dat_path, Some(PathBuf::from("recording.dat")));
        assert_eq!(p.n_channels_dat, Some(384));
        assert_eq!(p.dtype, Some(TraceDtype::I16));
        assert_eq!(p.offset, Some(0));
        assert_eq!(p.sample_rate, Some(30000.0));
        assert_eq!(p.hp_filtered, Some(false));
    }

    #[test]
    fn parses_quoted_variants_and_comments() {
        let src = r#"
# leading comment
dat_path = "rec.bin"   # trailing comment
sample_rate = 3e4
dtype = 'float32'
"#;
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.dat_path, Some(PathBuf::from("rec.bin")));
        assert_eq!(p.sample_rate, Some(30000.0));
        assert_eq!(p.dtype, Some(TraceDtype::F32));
    }

    #[test]
    fn rejects_unknown_dtype() {
        let src = "dtype = 'complex64'\n";
        assert!(PhyParams::parse(src).is_err());
    }

    #[test]
    fn ignores_unknown_keys() {
        let src = "n_channels_dat = 32\nfoo = 'bar'\n";
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.n_channels_dat, Some(32));
    }

    #[test]
    fn parses_crlf_line_endings() {
        let src = "n_channels_dat = 64\r\nsample_rate = 30000.\r\n";
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.n_channels_dat, Some(64));
        assert_eq!(p.sample_rate, Some(30_000.0));
    }

    #[test]
    fn parses_zero_offset() {
        let p = PhyParams::parse("offset = 0\n").unwrap();
        assert_eq!(p.offset, Some(0));
    }

    #[test]
    fn parses_scientific_notation_and_decimal_zero() {
        let p = PhyParams::parse("sample_rate = 3.0e4\n").unwrap();
        assert_eq!(p.sample_rate, Some(30_000.0));
        let p = PhyParams::parse("sample_rate = 30000.0\n").unwrap();
        assert_eq!(p.sample_rate, Some(30_000.0));
    }

    #[test]
    fn empty_lines_and_whitespace_only_lines_are_ignored() {
        let src = "\n   \n\n  n_channels_dat = 4\n   \n";
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.n_channels_dat, Some(4));
    }

    #[test]
    fn assigns_with_extra_internal_whitespace() {
        let src = "n_channels_dat   =   16\nsample_rate=10000.\n";
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.n_channels_dat, Some(16));
        assert_eq!(p.sample_rate, Some(10_000.0));
    }

    #[test]
    fn rejects_malformed_numeric_value() {
        let src = "n_channels_dat = not-a-number\n";
        assert!(PhyParams::parse(src).is_err());
    }

    #[test]
    fn rejects_malformed_hp_filtered() {
        let src = "hp_filtered = perhaps\n";
        assert!(PhyParams::parse(src).is_err());
    }

    #[test]
    fn handles_trailing_semicolons() {
        let src = "n_channels_dat = 4;\nsample_rate = 1000.;\n";
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.n_channels_dat, Some(4));
        assert_eq!(p.sample_rate, Some(1000.0));
    }

    #[test]
    fn lines_without_equals_are_silently_ignored() {
        let src = "n_channels_dat = 4\nthis line has no equals\nsample_rate = 1000.\n";
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.n_channels_dat, Some(4));
        assert_eq!(p.sample_rate, Some(1000.0));
    }

    #[test]
    fn raw_string_path_with_backslashes_is_preserved_verbatim() {
        // Real Windows phy projects use raw strings with backslashes.
        let src = "dat_path = r'C:\\data\\rec.bin'\n";
        let p = PhyParams::parse(src).unwrap();
        assert_eq!(p.dat_path, Some(PathBuf::from("C:\\data\\rec.bin")));
    }
}
