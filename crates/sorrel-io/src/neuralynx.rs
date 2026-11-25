//! Neuralynx tetrode spike files. Cheetah stores a timestamp and 32 samples
//! per channel, but neither the trace between spikes nor probe geometry.

use crate::kilosort::PhyLabel;
use crate::{
    ClusterId, DataProvider, HasAmplitudes, HasGeometry, SampleIndex, TraceSamples, TraceSlice,
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use memmap2::Mmap;
use serde::Serialize;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

const HEADER_SIZE: usize = 16_384;
const RECORD_SIZE: usize = 304;
const SAMPLE_OFFSET: usize = 48;
const CHANNELS: usize = 4;
const WAVEFORM_LEN: usize = 32;

#[derive(Debug, Serialize)]
#[serde(transparent)]
pub struct NlxHeader(Vec<(String, String)>);

impl NlxHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let bytes = bytes.get(..HEADER_SIZE).context("truncated header")?;
        let text = bytes.split(|&byte| byte == 0).next().unwrap_or_default();
        let mut entries = Vec::new();
        for line in text.split(|&byte| byte == b'\n') {
            // Some Cheetah versions use UTF-8 for µ in otherwise Latin-1 headers.
            let line = match std::str::from_utf8(line) {
                Ok(s) => std::borrow::Cow::Borrowed(s),
                Err(_) => std::borrow::Cow::Owned(line.iter().copied().map(char::from).collect()),
            };
            let Some(entry) = line.trim().strip_prefix('-') else {
                continue;
            };
            let (key, value) = entry.split_once(char::is_whitespace).unwrap_or((entry, ""));
            let key = key.trim_end_matches(':');
            let value = value.trim();
            let value = value
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or(value);
            entries.push((key.to_owned(), value.to_owned()));
        }
        Ok(Self(entries))
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    fn parse_value<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: FromStr,
        T::Err: Display,
    {
        self.get(key)
            .map(|s| s.parse().map_err(|err| anyhow!("invalid {key}: {err}")))
            .transpose()
    }

    fn validate_layout(&self) -> Result<()> {
        for (key, expected) in [
            ("RecordSize", RECORD_SIZE),
            ("NumADChannels", CHANNELS),
            ("WaveformLength", WAVEFORM_LEN),
        ] {
            if let Some(actual) = self.parse_value::<usize>(key)? {
                ensure!(
                    actual == expected,
                    "{key} is {actual}; .ntt requires {expected}"
                );
            }
        }
        Ok(())
    }

    pub fn sample_rate(&self) -> Result<f64> {
        let fs: f64 = self
            .parse_value("SamplingFrequency")?
            .context("missing SamplingFrequency")?;
        ensure!(
            (fs as f32).is_finite() && (fs as f32) > 0.0,
            "SamplingFrequency is outside the supported range"
        );
        Ok(fs)
    }

    pub fn ad_bit_volts(&self) -> Result<[f64; CHANNELS]> {
        let values = self.get("ADBitVolts").context("missing ADBitVolts")?;
        let values: Vec<f64> = values
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()
            .context("invalid ADBitVolts")?;
        let volts = match values.as_slice() {
            [x] => [*x; CHANNELS],
            [a, b, c, d] => [*a, *b, *c, *d],
            _ => bail!("ADBitVolts must have one or four values"),
        };
        ensure!(
            volts
                .iter()
                .all(|&v| v > 0.0 && ((v * 1e6 * f64::from(u16::MAX)) as f32).is_finite()),
            "ADBitVolts is outside the supported range"
        );
        Ok(volts)
    }

    pub fn input_inverted(&self) -> Result<bool> {
        match self.get("InputInverted") {
            None => Ok(false),
            Some(s) if s.eq_ignore_ascii_case("true") => Ok(true),
            Some(s) if s.eq_ignore_ascii_case("false") => Ok(false),
            Some(s) => bail!("invalid InputInverted: {s}"),
        }
    }

    pub fn alignment_pt(&self) -> Result<u32> {
        let alignment = self.parse_value("AlignmentPt")?.unwrap_or_else(|| {
            log::info!("missing AlignmentPt; assuming 8 for .ntt");
            8
        });
        ensure!(
            (1..=WAVEFORM_LEN as u32).contains(&alignment),
            "invalid AlignmentPt: {alignment}"
        );
        Ok(alignment)
    }

    pub fn ad_channels(&self) -> Option<&str> {
        self.get("ADChannel")
    }
}

#[derive(Debug, Serialize)]
struct Source {
    source_file: String,
    header: NlxHeader,
    t0_us: u64,
    sample_rate: f64,
    pre: u32,
    post: u32,
    record_count: u32,
    units: &'static str,
    input_inverted: bool,
    polarity: &'static str,
    row_order: &'static str,
    bumped_times: u32,
    max_shift_samples: u64,
}

#[derive(Debug)]
pub struct NttProvider {
    mmap: Mmap,
    order: Vec<u32>,
    times: Vec<SampleIndex>,
    amplitudes: Vec<f32>,
    n_samples: SampleIndex,
    scale_uv: [f64; CHANNELS],
    identity: blake3::Hash,
    source: Source,
}

impl NttProvider {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_inner(path).with_context(|| format!("{}: Neuralynx .ntt", path.display()))
    }

    fn open_inner(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        ensure!(
            file.metadata()?.len() >= HEADER_SIZE as u64,
            "truncated header"
        );
        // Like the other mapped backends, the input must not change while open.
        let mmap = unsafe { Mmap::map(&file)? };
        let header = NlxHeader::parse(&mmap)?;
        let payload = (mmap.len() - HEADER_SIZE) as u64;
        let count =
            u32::try_from(payload / RECORD_SIZE as u64).context("too many spike records")?;
        let tail = payload % RECORD_SIZE as u64;
        if tail != 0 {
            log::warn!(
                "{}: discarding {tail} trailing bytes after complete .ntt records",
                path.display()
            );
        }
        ensure!(count != 0, "holds no spike records (header only)");
        header.validate_layout()?;
        let sample_rate = header.sample_rate()?;
        let input_inverted = header.input_inverted()?;
        let scale_uv = header
            .ad_bit_volts()?
            .map(|v| v * if input_inverted { -1e6 } else { 1e6 });
        let alignment = header.alignment_pt()?;
        let (pre, post) = (alignment - 1, WAVEFORM_LEN as u32 - alignment);
        if let Some(disabled) = header.get("DisabledChannels").filter(|s| !s.is_empty()) {
            log::info!(
                "{}: DisabledChannels {disabled}; retaining all four slots",
                path.display()
            );
        }

        let mut order: Vec<u32> = (0..count).collect();
        let timestamp = |i| timestamp_us(record(&mmap, i));
        let sorted = order.windows(2).all(|w| timestamp(w[0]) <= timestamp(w[1]));
        if !sorted {
            log::warn!(
                "{}: .ntt records are out of order; sorting by timestamp",
                path.display()
            );
            order.sort_by_key(|&i| timestamp(i));
        }
        let t0_us = timestamp(order[0]);
        let (mut bumped_times, mut max_shift_samples) = (0, 0);
        let mut times = Vec::with_capacity(count as usize);
        let mut amplitudes = Vec::with_capacity(count as usize);
        let mut next = 0;
        for &i in &order {
            let delta = ((timestamp(i) - t0_us) as f64 * (sample_rate / 1e6)).round();
            ensure!(delta < u64::MAX as f64, "spike sample index overflows");
            let nominal = (delta as u64)
                .checked_add(u64::from(pre))
                .context("spike sample index overflows")?;
            let sample = next.max(nominal);
            if sample != nominal {
                bumped_times += 1;
                max_shift_samples = max_shift_samples.max(sample - nominal);
            }
            times.push(SampleIndex(sample));
            next = sample
                .checked_add(1)
                .context("spike sample index overflows")?;
            amplitudes.push(amplitude(record(&mmap, i), &scale_uv));
        }
        let n_samples = SampleIndex(
            next.checked_add(u64::from(post))
                .context("sample count overflows")?,
        );
        if bumped_times != 0 {
            log::warn!(
                "{}: bumped {bumped_times} .ntt spike times to keep sample indices unique (maximum shift {max_shift_samples} samples)",
                path.display()
            );
        }
        let mut hash = blake3::Hasher::new();
        hash.update(b"sorrel-provider-identity-v1/neuralynx-ntt/uV");
        hash.update(&mmap);
        Ok(Self {
            mmap,
            order,
            times,
            amplitudes,
            n_samples,
            scale_uv,
            identity: hash.finalize(),
            source: Source {
                source_file: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                header,
                t0_us,
                sample_rate,
                pre,
                post,
                record_count: count,
                units: "uV",
                input_inverted,
                polarity: if input_inverted {
                    "negated"
                } else {
                    "unchanged"
                },
                row_order: if sorted { "file" } else { "time" },
                bumped_times,
                max_shift_samples,
            },
        })
    }

    pub fn write_source(&self, path: &Path) -> Result<()> {
        let write = || -> Result<()> {
            let mut bytes = serde_json::to_vec_pretty(&self.source)?;
            bytes.push(b'\n');
            match std::fs::read(path) {
                Ok(existing) if existing == bytes => return Ok(()),
                Ok(_) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(err.into()),
            }
            replace_source(path, &bytes)?;
            Ok(())
        };
        write().with_context(|| format!("writing {}", path.display()))
    }
}

fn replace_source(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::fs::{self, OpenOptions};
    use std::io::ErrorKind;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let (temp, mut file) = loop {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let temp = path.with_file_name(format!(".ntt-source-{}-{id}.tmp", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => break (temp, file),
            Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    };
    let written = file.write_all(bytes);
    drop(file);
    let result = written.and_then(|()| fs::rename(&temp, path));
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn record(bytes: &[u8], index: u32) -> &[u8] {
    let start = HEADER_SIZE + index as usize * RECORD_SIZE;
    &bytes[start..start + RECORD_SIZE]
}

fn timestamp_us(record: &[u8]) -> u64 {
    u64::from_le_bytes(record[..8].try_into().unwrap())
}

fn counts(record: &[u8]) -> impl Iterator<Item = i16> + '_ {
    record[SAMPLE_OFFSET..]
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(i16::from_le_bytes)
}

fn amplitude(record: &[u8], scale: &[f64; CHANNELS]) -> f32 {
    let mut lo = [i16::MAX; CHANNELS];
    let mut hi = [i16::MIN; CHANNELS];
    for (i, v) in counts(record).enumerate() {
        let c = i % CHANNELS;
        lo[c] = lo[c].min(v);
        hi[c] = hi[c].max(v);
    }
    (0..CHANNELS)
        .map(|c| (f64::from(i32::from(hi[c]) - i32::from(lo[c])) * scale[c].abs()) as f32)
        .fold(0.0, f32::max)
}

impl DataProvider for NttProvider {
    type Label = PhyLabel;

    fn sample_rate(&self) -> f32 {
        self.source.sample_rate as f32
    }
    fn exact_sample_rate(&self) -> Result<f64> {
        Ok(self.source.sample_rate)
    }
    fn n_channels(&self) -> u32 {
        CHANNELS as u32
    }
    fn n_samples(&self) -> SampleIndex {
        self.n_samples
    }
    fn n_clusters(&self) -> u32 {
        1
    }
    fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
        if cluster.0 == 0 {
            &self.times
        } else {
            &[]
        }
    }
    fn trace(&self, start: SampleIndex, _len: u32) -> TraceSlice<'_> {
        TraceSlice {
            start,
            n_channels: self.n_channels(),
            samples: TraceSamples::F32(&[]),
        }
    }
    fn initial_labels(&self) -> Vec<Self::Label> {
        vec![PhyLabel::default()]
    }
    fn identity_bytes(&self) -> Vec<u8> {
        self.identity.as_bytes().to_vec()
    }
    fn stored_snippet_window(&self) -> Option<(u32, u32)> {
        Some((self.source.pre, self.source.post))
    }
    fn stored_snippet(&self, t: SampleIndex, out: &mut [f32]) -> bool {
        if out.len() != WAVEFORM_LEN * CHANNELS {
            return false;
        }
        let Ok(row) = self.times.binary_search(&t) else {
            return false;
        };
        for (i, (dst, v)) in out
            .iter_mut()
            .zip(counts(record(&self.mmap, self.order[row])))
            .enumerate()
        {
            *dst = (f64::from(v) * self.scale_uv[i % CHANNELS]) as f32;
        }
        true
    }
}

impl HasGeometry for NttProvider {
    fn channel_positions(&self) -> &[[f32; 2]] {
        &[]
    }
}

impl HasAmplitudes for NttProvider {
    fn spike_amplitudes(&self, cluster: ClusterId) -> &[f32] {
        if cluster.0 == 0 {
            &self.amplitudes
        } else {
            &[]
        }
    }
}

/// Synthetic fixture shared by the workspace tests; `header` excludes padding.
#[doc(hidden)]
pub struct NttFixture {
    pub header: Vec<u8>,
    pub spikes: Vec<(u64, [[i16; CHANNELS]; WAVEFORM_LEN])>,
}

impl Default for NttFixture {
    fn default() -> Self {
        let header = b"######## Neuralynx Data File Header\r\n-RecordSize 304\r\n-NumADChannels 4\r\n-WaveformLength 32\r\n-SamplingFrequency 32000\r\n-ADBitVolts 0.000001\r\n-InputInverted True\r\n-AlignmentPt 8\r\n";
        Self {
            header: header.to_vec(),
            spikes: Vec::new(),
        }
    }
}

#[doc(hidden)]
pub fn write_ntt(path: &Path, fixture: &NttFixture) -> Result<()> {
    ensure!(
        fixture.header.len() <= HEADER_SIZE,
        "fixture header is too long"
    );
    let mut file = BufWriter::new(File::create(path)?);
    let mut header = fixture.header.clone();
    header.resize(HEADER_SIZE, 0);
    file.write_all(&header)?;
    for (ts, samples) in &fixture.spikes {
        file.write_all(&ts.to_le_bytes())?;
        file.write_all(&[0; SAMPLE_OFFSET - 8])?;
        for &count in samples.iter().flatten() {
            file.write_all(&count.to_le_bytes())?;
        }
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_inverted_samples_use_the_declared_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TT5.ntt");
        let mut samples = [[0; 4]; 32];
        samples[0] = [-8, 5, -3, 4];
        write_ntt(
            &path,
            &NttFixture {
                header: b"-SamplingFrequency 32000\n-ADBitVolts 1e-6 2e-6 3e-6 4e-6\n-InputInverted False\n-AlignmentPt 12"
                    .to_vec(),
                spikes: vec![(1_000, samples), (2_000, samples)],
            },
        )
        .unwrap();

        let provider = NttProvider::open(&path).unwrap();
        assert_eq!(provider.stored_snippet_window(), Some((11, 20)));
        let first = provider.spike_times(ClusterId(0))[0];
        assert_eq!(first, SampleIndex(11));
        assert_eq!(provider.n_samples(), SampleIndex(64));
        let mut out = [0.0; 128];
        assert!(provider.stored_snippet(first, &mut out));
        assert_eq!(out[..4], [-8., 10., -9., 16.]);
    }

    #[test]
    fn source_sidecar_preserves_unchanged_file_and_replaces_stale_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let ntt = dir.path().join("TT5.ntt");
        write_ntt(
            &ntt,
            &NttFixture {
                spikes: vec![(0, [[0; 4]; 32])],
                ..Default::default()
            },
        )
        .unwrap();
        let provider = NttProvider::open(&ntt).unwrap();
        let sidecar = dir.path().join("ntt_source.json");
        provider.write_source(&sidecar).unwrap();
        let expected = std::fs::read(&sidecar).unwrap();
        let old = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
        File::options()
            .write(true)
            .open(&sidecar)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let modified = || std::fs::metadata(&sidecar).unwrap().modified().unwrap();
        let before = modified();
        provider.write_source(&sidecar).unwrap();
        assert_eq!(modified(), before);

        std::fs::write(&sidecar, b"stale").unwrap();
        provider.write_source(&sidecar).unwrap();
        assert_eq!(std::fs::read(&sidecar).unwrap(), expected);
    }

    #[test]
    fn preserves_order_repeated_and_empty_values_and_decodes_both_micro_signs() {
        let mut bytes = b"# comment\r\n  -SamplingFrequency: \"32000\"\r\n-ADBitVolts 1e-8\n\
            -Feature Peak 0\r\n-Feature Valley 0\r\n-ProbeName \r\n\
            -DspFilterDelay_\xb5s 1\r\n-DspFilterDelay_\xc2\xb5s 2\r\n"
            .to_vec();
        bytes.resize(HEADER_SIZE, 0);
        let header = NlxHeader::parse(&bytes).unwrap();
        assert_eq!(header.sample_rate().unwrap(), 32_000.0);
        assert_eq!(header.ad_bit_volts().unwrap(), [1e-8; 4]);
        assert_eq!(header.alignment_pt().unwrap(), 8);
        assert_eq!(header.get("Feature"), Some("Peak 0"));
        assert_eq!(header.get("ProbeName"), Some(""));
        assert_eq!(header.0[3], ("Feature".into(), "Valley 0".into()));
        assert_eq!(header.0[5].0, "DspFilterDelay_µs");
        assert_eq!(header.0[6].0, "DspFilterDelay_µs");
        assert!(NlxHeader::parse(&bytes[..HEADER_SIZE - 1]).is_err());
    }

    #[test]
    fn rejects_incomplete_files_and_invalid_headers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TT1.ntt");
        let error = || format!("{:#}", NttProvider::open(&path).unwrap_err());
        std::fs::write(&path, [0; 10]).unwrap();
        assert!(error().contains("truncated header"));
        write_ntt(&path, &NttFixture::default()).unwrap();
        assert!(error().contains("holds no spike records (header only)"));
        for (header, message) in [
            ("-ADBitVolts 1e-8", "missing SamplingFrequency"),
            ("-SamplingFrequency 32000", "missing ADBitVolts"),
            ("-RecordSize 303", "RecordSize is 303"),
            ("-NumADChannels 3", "NumADChannels is 3"),
            ("-WaveformLength 31", "WaveformLength is 31"),
            (
                "-SamplingFrequency 32000\n-ADBitVolts 1e-8\n-AlignmentPt 33",
                "invalid AlignmentPt",
            ),
        ] {
            write_ntt(
                &path,
                &NttFixture {
                    header: header.as_bytes().to_vec(),
                    spikes: vec![(0, [[0; 4]; 32])],
                },
            )
            .unwrap();
            assert!(error().contains(message), "{}", error());
        }
    }

    #[test]
    fn decodes_calibrates_and_stably_orders_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TT1.ntt");
        let mut first = [[0; 4]; 32];
        first[0] = [i16::MIN, 0, 40, -10];
        first[1] = [3, 0, -25, 20];
        let marker = |value| [[value, 0, 0, 0]; 32];
        write_ntt(&path, &NttFixture {
            header: b"-SamplingFrequency 32000\n-ADBitVolts 1e-6 2e-6 3e-6 4e-6\n-InputInverted True\n-AlignmentPt 8\n-DisabledChannels 1".to_vec(),
            spikes: vec![(2_000, marker(4)), (1_000, first), (1_000, marker(2)),
                (1_000, marker(3)), (1_032, marker(5))],
        }).unwrap();
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(&[1, 2, 3]).unwrap();
        drop(f);

        let p = NttProvider::open(&path).unwrap();
        assert_eq!(
            p.spike_times(ClusterId(0)),
            [7, 8, 9, 10, 39].map(SampleIndex)
        );
        assert_eq!(p.n_samples(), SampleIndex(64));
        assert_eq!(p.n_channels(), 4);
        assert_eq!(p.stored_snippet_window(), Some((7, 24)));
        assert_eq!(p.source.t0_us, 1000);
        assert_eq!(p.source.row_order, "time");
        assert_eq!(p.source.bumped_times, 3);
        assert_eq!(p.source.max_shift_samples, 2);
        let mut out = [0.0; 128];
        assert!(p.stored_snippet(SampleIndex(7), &mut out));
        assert_eq!(out[..8], [32768., 0., -120., 40., -3., 0., 75., -80.]);
        assert!(out.iter().skip(1).step_by(4).all(|&v| v == 0.0));
        assert!(!p.stored_snippet(SampleIndex(7), &mut out[..127]));
        for (time, value) in [(8, -2.), (9, -3.), (10, -5.), (39, -4.)] {
            assert!(p.stored_snippet(SampleIndex(time), &mut out));
            assert_eq!(out[0], value);
        }
        assert!(!p.stored_snippet(SampleIndex(11), &mut out));
        assert_eq!(p.spike_amplitudes(ClusterId(0)), [32771., 0., 0., 0., 0.]);
        assert!(p.trace(SampleIndex(0), 1).samples.is_empty());
        assert!(p.channel_positions().is_empty());

        let copy = dir.path().join("copy.ntt");
        std::fs::copy(&path, &copy).unwrap();
        let id = p.identity_bytes();
        assert_eq!(id, NttProvider::open(&copy).unwrap().identity_bytes());
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&copy)
            .unwrap();
        f.write_all(&[4]).unwrap();
        drop(f);
        assert_ne!(id, NttProvider::open(&copy).unwrap().identity_bytes());
    }

    #[test]
    #[ignore = "requires a local copy of the real TT1.ntt fixture"]
    fn real_tt1_fixture() {
        let path = std::env::var_os("SORREL_NTT_FIXTURE").expect("set SORREL_NTT_FIXTURE");
        let p = NttProvider::open(Path::new(&path)).unwrap();
        assert_eq!(p.times.len(), 38_635);
        assert_eq!(p.exact_sample_rate().unwrap(), 32_000.0);
        assert_eq!(p.stored_snippet_window(), Some((7, 24)));
        assert_eq!(
            p.source.header.ad_bit_volts().unwrap(),
            [1.068115234375e-8; 4]
        );
        assert!(p.source.input_inverted);
        assert_eq!(p.source.header.get("DisabledChannels"), Some("1"));
        assert_eq!(p.source.bumped_times, 0);
        assert_eq!(p.source.row_order, "file");
        let timestamps: Vec<_> = p
            .order
            .iter()
            .map(|&i| {
                let record = record(&p.mmap, i);
                assert_eq!(record[12..16], [0; 4]);
                timestamp_us(record)
            })
            .collect();
        assert_eq!(timestamps.windows(2).map(|w| w[1] - w[0]).min(), Some(781));
        let mut out = [0.0; 128];
        for &t in &p.times {
            assert!(p.stored_snippet(t, &mut out));
            assert!(out.iter().skip(1).step_by(4).all(|&v| v == 0.0));
        }
        let mut amplitudes = p.amplitudes.clone();
        amplitudes.sort_by(f32::total_cmp);
        let median = amplitudes[amplitudes.len() / 2];
        assert!((56.0..60.0).contains(&median), "{median} µV");
        println!(
            "TT1: {} spikes at {} Hz; median amplitude {median:.2} µV",
            p.times.len(),
            p.sample_rate()
        );
    }
}
