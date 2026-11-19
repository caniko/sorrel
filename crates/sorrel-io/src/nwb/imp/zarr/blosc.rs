//! Decode c-Blosc 1 chunks (header format version 2):
//! https://github.com/Blosc/c-blosc/blob/v1.21.6/README_CHUNK_FORMAT.rst

use std::{
    borrow::Cow,
    io::Read,
    sync::{Arc, Once},
};

use zarrs::array::codec::api::{
    register_codec_v2, ArrayBytesRaw, BytesRepresentation, BytesToBytesCodecTraits, Codec,
    CodecError, CodecMetadataOptions, CodecOptions, CodecRuntimePluginV2, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs::metadata::Configuration;
use zarrs::plugin::{ExtensionName, ZarrVersion};

pub(super) fn register() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // Array::open uses the V2 plugin directly, without a V3 metadata conversion.
        register_codec_v2(CodecRuntimePluginV2::new(
            |name| name == "blosc",
            |metadata| {
                Ok(Codec::BytesToBytes(Arc::new(Blosc(
                    metadata.configuration().clone(),
                ))))
            },
        ));
    });
}

#[derive(Debug)]
struct Blosc(Configuration);

impl ExtensionName for Blosc {
    fn name(&self, version: ZarrVersion) -> Option<Cow<'static, str>> {
        (version == ZarrVersion::V2).then_some(Cow::Borrowed("blosc"))
    }
}

impl CodecTraits for Blosc {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        version: ZarrVersion,
        _: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        (version == ZarrVersion::V2).then(|| self.0.clone())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl BytesToBytesCodecTraits for Blosc {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }

    fn recommended_concurrency(
        &self,
        _: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        _: ArrayBytesRaw<'a>,
        _: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        Err(CodecError::from("Sorrel's Blosc codec is decode-only"))
    }

    fn decode<'a>(
        &self,
        encoded: ArrayBytesRaw<'a>,
        decoded: &BytesRepresentation,
        _: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        decode_frame(&encoded, *decoded).map(Cow::Owned)
    }

    fn encoded_representation(&self, _: &BytesRepresentation) -> BytesRepresentation {
        BytesRepresentation::UnboundedSize
    }
}

const HEADER: usize = 16;
const SHUFFLE: u8 = 1;
const MEMCPY: u8 = 2;
const BITSHUFFLE: u8 = 4;
const DONT_SPLIT: u8 = 16;
const FILTERS: u8 = SHUFFLE | BITSHUFFLE;

fn malformed() -> CodecError {
    CodecError::from("invalid Blosc frame or block")
}

fn u32_le(bytes: &[u8]) -> Result<usize, CodecError> {
    let (word, _) = bytes.split_first_chunk().ok_or_else(malformed)?;
    // c-Blosc stores these as nonnegative int32_t fields.
    usize::try_from(i32::from_le_bytes(*word)).map_err(|_| malformed())
}

fn decode_frame(src: &[u8], decoded: BytesRepresentation) -> Result<Vec<u8>, CodecError> {
    let (header, body) = src.split_first_chunk::<HEADER>().ok_or_else(malformed)?;
    let [version, codec_version, flags, typesize, ..] = *header;
    // ponytail: only the c-Blosc 1 v2 wire format; add a separate parser if
    // older Blosc headers or Blosc2 extended chunks are needed.
    if version != 2 || codec_version != 1 {
        return Err(CodecError::from(format!(
            "unsupported Blosc frame/codec version {version}/{codec_version}"
        )));
    }
    let compressor = flags >> 5;
    if compressor > 4 {
        return Err(CodecError::from(format!(
            "unsupported Blosc compressor code {compressor}"
        )));
    }
    let nbytes = u32_le(&header[4..])?;
    let blocksize = u32_le(&header[8..])?;
    let width = usize::from(typesize);
    if u32_le(&header[12..])? != src.len()
        || nbytes > i32::MAX as usize - HEADER
        || width == 0
        || blocksize == 0
        || blocksize > (i32::MAX as usize - 255 * 4) / 3
        || blocksize > nbytes.max(1)
        || flags & 8 != 0
        || flags & FILTERS == FILTERS
        || matches!(decoded, BytesRepresentation::FixedSize(size) if nbytes as u64 != size)
        || matches!(decoded, BytesRepresentation::BoundedSize(size) if nbytes as u64 > size)
    {
        return Err(malformed());
    }
    if flags & MEMCPY != 0 {
        return (body.len() == nbytes)
            .then(|| body.to_vec())
            .ok_or_else(malformed);
    }

    let table_len = nbytes
        .div_ceil(blocksize)
        .checked_mul(4)
        .ok_or_else(malformed)?;
    let table = body.get(..table_len).ok_or_else(malformed)?;
    // Parallel c-Blosc writes can put blocks out of table order. Sort in
    // physical order to find each block's end, then decode in logical order.
    let mut physical = Vec::new();
    physical
        .try_reserve_exact(table_len / 4)
        .map_err(|e| CodecError::from(e.to_string()))?;
    for (index, entry) in table.as_chunks::<4>().0.iter().enumerate() {
        let offset = u32_le(entry)?;
        if !(HEADER + table_len..src.len()).contains(&offset) {
            return Err(malformed());
        }
        physical.push((offset, 0, index));
    }
    physical.sort_unstable();
    let mut end = src.len();
    for (start, stop, _) in physical.iter_mut().rev() {
        if *start == end {
            return Err(malformed());
        }
        *stop = end;
        end = *start;
    }
    physical.sort_unstable_by_key(|(_, _, index)| *index);
    let mut output = Vec::new();
    output
        .try_reserve_exact(nbytes)
        .map_err(|e| CodecError::from(e.to_string()))?;
    let mut shuffled = Vec::new();
    for (offset, end, index) in physical {
        let mut streams = &src[offset..end];
        let start = index * blocksize;
        let len = blocksize.min(nbytes - start);
        let count =
            if flags & DONT_SPLIT == 0 && width <= 16 && len == blocksize && len / width >= 128 {
                width
            } else {
                1
            };
        if !len.is_multiple_of(count) {
            return Err(malformed());
        }
        output.resize(start + len, 0);
        let block = &mut output[start..];
        let filtered = (flags & SHUFFLE != 0 && width > 1)
            || (flags & BITSHUFFLE != 0 && len / width >= 8 && (len / width).is_multiple_of(8));
        let target = if filtered {
            shuffled
                .try_reserve(len.saturating_sub(shuffled.len()))
                .map_err(|e| CodecError::from(e.to_string()))?;
            shuffled.resize(len, 0);
            &mut shuffled[..]
        } else {
            &mut *block
        };
        for stream in target.chunks_exact_mut(len / count) {
            let size = u32_le(streams)?;
            let (data, rest) = streams[4..].split_at_checked(size).ok_or_else(malformed)?;
            streams = rest;
            decode_stream(compressor, data, stream)?;
        }
        if filtered {
            unshuffle(&shuffled, block, width, flags & BITSHUFFLE != 0);
        }
    }
    Ok(output)
}

fn decode_stream(compressor: u8, src: &[u8], dst: &mut [u8]) -> Result<(), CodecError> {
    if src.len() == dst.len() {
        dst.copy_from_slice(src);
        return Ok(());
    }
    let valid = match compressor {
        0 => blosclz(src, dst).is_some(),
        1 => lz4_flex::block::decompress_into(src, dst).ok() == Some(dst.len()), // LZ4 and LZ4HC
        2 => snap::raw::Decoder::new().decompress(src, dst).ok() == Some(dst.len()),
        3 => {
            let mut decoder = flate2::Decompress::new(true);
            matches!(
                decoder.decompress(src, dst, flate2::FlushDecompress::Finish),
                Ok(flate2::Status::StreamEnd)
            ) && decoder.total_in() == src.len() as u64
                && decoder.total_out() == dst.len() as u64
        }
        4 => {
            // c-Blosc emits one Zstd frame per stream. Bound writes (and check
            // the end marker) even if its embedded size disagrees with Blosc.
            let mut decoder =
                ruzstd::decoding::StreamingDecoder::new_with_max_window_size(src, dst.len() as u64)
                    .map_err(|e| CodecError::from(format!("invalid Blosc Zstd stream: {e}")))?;
            decoder.read_exact(dst).is_ok()
                && matches!(decoder.read(&mut [0]), Ok(0))
                && decoder.get_ref().is_empty()
                && decoder
                    .decoder
                    .get_checksum_from_data()
                    .is_none_or(|checksum| {
                        decoder.decoder.get_calculated_checksum() == Some(checksum)
                    })
        }
        _ => unreachable!("validated compressor code"),
    };
    valid.then_some(()).ok_or_else(|| {
        CodecError::from(format!(
            "invalid Blosc stream (compressor code {compressor})"
        ))
    })
}

fn unshuffle(src: &[u8], dst: &mut [u8], width: usize, bits: bool) {
    let count = src.len() / width;
    let full = count * width;
    if bits {
        // c-Blosc bitunshuffles the whole block only when its element count
        // is a multiple of eight. Each byte lane has eight LSB-first planes.
        let plane_len = count / 8;
        for byte in 0..width {
            for group in 0..plane_len {
                for item in 0..8 {
                    let mut value = 0;
                    for bit in 0..8 {
                        let plane = src[(byte * 8 + bit) * plane_len + group];
                        value |= ((plane >> item) & 1) << bit;
                    }
                    dst[(group * 8 + item) * width + byte] = value;
                }
            }
        }
    } else {
        for (i, element) in dst[..full].chunks_exact_mut(width).enumerate() {
            for (byte, value) in element.iter_mut().enumerate() {
                *value = src[byte * count + i];
            }
        }
    }
    dst[full..].copy_from_slice(&src[full..]);
}

// Wire rules from https://github.com/Blosc/c-blosc/blob/v1.21.6/blosc/blosclz.c
fn blosclz(mut src: &[u8], dst: &mut [u8]) -> Option<()> {
    fn byte(src: &mut &[u8]) -> Option<usize> {
        let (&value, rest) = src.split_first()?;
        *src = rest;
        Some(value.into())
    }
    let mut control = byte(&mut src)? & 31; // first token is always a literal
    let mut pos = 0usize;
    loop {
        if control < 32 {
            let len = control + 1;
            let (literal, rest) = src.split_at_checked(len)?;
            dst.get_mut(pos..pos + len)?.copy_from_slice(literal);
            src = rest;
            pos += len;
        } else {
            let mut len = (control >> 5) + 2;
            if len == 9 {
                loop {
                    let extra = byte(&mut src)?;
                    len = len.checked_add(extra)?;
                    if len > dst.len() - pos {
                        return None;
                    }
                    if extra != 255 {
                        break;
                    }
                }
            }
            let high = control & 31;
            let low = byte(&mut src)?;
            let distance = if high == 31 && low == 255 {
                8192 + ((byte(&mut src)? << 8) | byte(&mut src)?)
            } else {
                ((high << 8) | low) + 1
            };
            // Like c-Blosc, a match must be followed by another token.
            if src.is_empty() || len > dst.len() - pos {
                return None;
            }
            let from = pos.checked_sub(distance)?;
            if distance == 1 {
                let value = dst[from];
                dst[pos..pos + len].fill(value);
            } else {
                // Forward copy is required for self-overlapping LZ matches.
                for i in 0..len {
                    dst[pos + i] = dst[from + i];
                }
            }
            pos += len;
        }
        if src.is_empty() {
            return (pos == dst.len()).then_some(());
        }
        control = byte(&mut src)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn vectors(name: &str) -> Vec<u8> {
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../release/fixtures/zarr-blosc")
                .join(format!("{name}.bin")),
        )
        .unwrap()
    }

    fn check(frames: &mut &[u8], count: usize, width: usize) {
        let len = u32_le(&frames[12..]).unwrap();
        let (frame, rest) = frames.split_at(len);
        *frames = rest;
        assert_eq!(usize::from(frame[3]), width);
        let expected: Vec<_> = (0..count)
            .flat_map(|i| {
                (0..width).map(move |j| {
                    let k = i / 16 % 128;
                    (k * 11 + k / 17 * 37 + j * 53) as u8
                })
            })
            .collect();
        assert_eq!(
            decode_frame(frame, BytesRepresentation::FixedSize(expected.len() as u64)).unwrap(),
            expected
        );
    }

    #[test]
    fn compressor_shuffle_typesize_matrix() {
        for name in ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"] {
            let vectors = vectors(name);
            let mut frames = vectors.as_slice();
            for _shuffle in 0..3 {
                for width in [1, 2, 4, 8] {
                    check(&mut frames, 1024, width);
                }
            }
            assert!(frames.is_empty());
        }
    }

    #[test]
    fn multiblock_split_tail_and_memcpy() {
        let vectors = vectors("edge");
        let mut frames = vectors.as_slice();
        assert_eq!(frames[2] & (DONT_SPLIT | MEMCPY), 0);
        assert_eq!(u32_le(&frames[8..]).unwrap(), 65536);
        check(&mut frames, 2 * 16384 + 135, 4);
        assert_ne!(frames[2] & MEMCPY, 0);
        check(&mut frames, 112, 2);
        assert!(frames.is_empty());
    }

    #[test]
    fn block_extents_follow_physical_offsets() {
        // Two raw 8-byte streams, physically in reverse logical order.
        let frame: Vec<u8> = [2, 1, DONT_SPLIT, 1]
            .into_iter()
            .chain(
                [16u32, 8, 48, 36, 24, 8]
                    .into_iter()
                    .flat_map(u32::to_le_bytes),
            )
            .chain([2; 8])
            .chain(8u32.to_le_bytes())
            .chain([1; 8])
            .collect();
        let decode = |src: &[u8]| decode_frame(src, BytesRepresentation::FixedSize(16));
        assert_eq!(decode(&frame).unwrap(), [[1; 8], [2; 8]].concat());

        // Pointing inside another block can otherwise decode a valid stream.
        let mut overlap = frame.clone();
        overlap[16..20].copy_from_slice(&32u32.to_le_bytes());
        overlap[32..36].copy_from_slice(&8u32.to_le_bytes());
        assert!(decode(&overlap).is_err(), "overlapping blocks");

        for (field, value) in [
            (16, 24u32),
            (16, 16),
            (16, 48),
            (16, u32::MAX),
            (4, 0),
            (8, 0),
            (8, 17),
        ] {
            let mut bad = frame.clone();
            bad[field..field + 4].copy_from_slice(&value.to_le_bytes());
            assert!(decode(&bad).is_err(), "field {field} = {value}");
        }
        for len in HEADER..frame.len() {
            let mut truncated = frame[..len].to_vec();
            truncated[12..16].copy_from_slice(&(len as u32).to_le_bytes());
            assert!(decode(&truncated).is_err(), "truncated to {len} bytes");
        }
    }

    #[test]
    fn rejects_invalid_frame_headers_and_sizes() {
        let src = vectors("lz4");
        let frame = &src[..u32_le(&src[12..]).unwrap()];
        let decode = |bytes: &[u8]| decode_frame(bytes, BytesRepresentation::FixedSize(1024));
        for len in 0..frame.len() {
            assert!(decode(&frame[..len]).is_err());
        }
        for (offset, byte, error) in [
            (0, 3, "unsupported Blosc frame/codec version"),
            (1, 2, "unsupported Blosc frame/codec version"),
            (2, 7 << 5, "unsupported Blosc compressor code 7"),
            (3, 0, "invalid Blosc frame"),
            (5, 0xff, "invalid Blosc frame"),
            (16, 0, "invalid Blosc frame"),
            (20, 0xff, "invalid Blosc"),
        ] {
            let mut bad = frame.to_vec();
            bad[offset] = byte;
            assert!(decode(&bad).unwrap_err().to_string().contains(error));
        }
    }
}
