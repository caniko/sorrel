# /// script
# requires-python = "==3.12.*"
# dependencies = ["numcodecs==0.15.1", "numpy==2.2.6", "cramjam==2.10.0"]
# ///
"""nix shell nixpkgs#uv -c uv run --python-preference only-managed release/fixtures/zarr-blosc/generate.py"""

import struct
from pathlib import Path

import cramjam
import numpy as np
from numcodecs.blosc import Blosc, set_nthreads


def samples(count, width):
    return bytes((i // 16 % 128 * 11 + i // 16 % 128 // 17 * 37 + j * 53) % 256
                 for i in range(count) for j in range(width))


def frame(cname, shuffle, width, raw, **kwargs):
    array = np.frombuffer(raw, dtype=f"<u{width}")
    codec = Blosc(cname=cname, shuffle=shuffle, **kwargs)
    encoded = codec.encode(array)
    assert bytes(codec.decode(encoded)) == raw
    return encoded


def snappy(shuffle, width, raw):
    # PyPI's numcodecs wheel lacks Blosc/Snappy. Transcode a single, unsplit
    # Blosc/Zstd block, retaining numcodecs' header, offsets and shuffle.
    base = bytearray(frame("zstd", shuffle, width, raw))
    nbytes, blocksize, _, offset, stream_size = struct.unpack_from("<5I", base, 4)
    assert nbytes == blocksize and base[2] & 0x12 == 0x10
    assert offset == 20 and stream_size == len(base) - 24 and stream_size != nbytes
    shuffled = cramjam.zstd.decompress(base[24:])
    payload = bytes(cramjam.snappy.compress_raw(shuffled))
    assert len(payload) != nbytes and bytes(cramjam.snappy.decompress_raw(payload)) == bytes(shuffled)
    base[2] = (base[2] & 31) | (2 << 5)
    struct.pack_into("<I", base, 12, 24 + len(payload))
    struct.pack_into("<I", base, 20, len(payload))
    return base[:24] + payload


def main():
    set_nthreads(1)
    out = Path(__file__).parent
    for cname, code in [("blosclz", 0), ("lz4", 1), ("lz4hc", 1), ("snappy", 2), ("zlib", 3), ("zstd", 4)]:
        frames = []
        for shuffle in range(3):
            for width in (1, 2, 4, 8):
                raw = samples(1024, width)
                encoded = snappy(shuffle, width, raw) if cname == "snappy" else frame(cname, shuffle, width, raw)
                assert encoded[2] >> 5 == code and not encoded[2] & 2, (cname, shuffle, width)
                assert encoded[2] & 5 == (0, 1, 4)[shuffle], (cname, shuffle, width)
                frames.append(encoded)
        (out / f"{cname}.bin").write_bytes(b"".join(frames))

    # Four split streams per full block; a short, unsplit final block whose
    # element count is not divisible by eight, so bitshuffle leaves it alone.
    split = frame("zlib", 2, 4, samples(2 * 16384 + 135, 4), blocksize=512)
    nbytes, blocksize = struct.unpack_from("<2I", split, 4)
    assert split[2] & 0x12 == 0 and 128 * 4 <= blocksize < nbytes
    memcpy = frame("lz4", 1, 2, samples(112, 2), clevel=0)
    assert memcpy[2] & 2
    (out / "edge.bin").write_bytes(split + memcpy)
    (out / "edited-spikes.bin").write_bytes(frame(
        "lz4", 1, 8, struct.pack("<5d", 0.61, 0.7, 0.75, 0.55, 0.79)))


if __name__ == "__main__":
    main()
