"""
Byte-Oriented range Asymmetric Numeral Systems (rANS) Source Codec.

Mathematical Framework & Theoretical Formulation:
--------------------------------------------------
Range Asymmetric Numeral Systems (rANS) is an entropy coding algorithm introduced
by Jarek Duda (2013) that achieves compression rates arbitrarily close to the
Shannon entropy limit while processing symbols at the granularity of native machine
words (bytes), circumventing the bit-level bottlenecks of arithmetic coding and the
integer-bit inefficiencies of Huffman coding.

1. Probability Model & Normalization (Hare-Niemeyer / Largest Remainder):
   Given an alphabet of bytes S subset of {0, ..., 255} with empirical frequencies
   c_s and total length N = sum(c_s):
   Target scale M = 2^B (default B = 14, M = 16384).
   For every symbol present in the input (c_s > 0), a strictly positive integer
   frequency f_s >= 1 is allocated such that:
       sum_{s in S} f_s = M = 2^B
   The cumulative frequencies are defined as:
       C_s = sum_{j < s, j in S} f_j, with C_0 = 0.
   Each symbol s maps to a unique semi-open interval [C_s, C_s + f_s) in [0, M).

2. rANS State Invariant & Working Range:
   The encoder maintains a single integer state x >= L, where L = 2^23 = 8,388,608.
   The state is constrained to the interval [L, b * L - 1], where the base b = 256 (1 byte).
   With B = 14 and L = 2^23, x never exceeds 2^31 - 1, fitting within a 32-bit integer.

3. Encoder Renormalization & State Transition:
   To encode symbol s with normalized frequency f_s, the state must not exceed:
       x_max(s) = floor(L / M) * b * f_s = ((L >> B) << 8) * f_s
   - Renormalization: While x >= x_max(s), the least significant byte is flushed:
         emit(x mod 256),  x <- floor(x / 256)
   - State Update:
         x <- floor(x / f_s) * M + C_s + (x mod f_s)
   Symbols are encoded in reverse order (s_{N-1}, ..., s_0) because rANS behaves as a LIFO stack.
   At completion, the final state x_final is written as a 4-byte little-endian uint32,
   followed by the reversed stream of emitted bytes.

4. Decoder Renormalization & Inversion:
   The decoder reads the initial 4-byte state x = x_final and consumes the stream forward.
   For each symbol to reconstruct (i = 0, ..., N-1):
   - Frequency Slot:
         m = x mod M = x & (M - 1)
   - Symbol Lookup (O(1) direct lookup table):
         s = LUT[m], where LUT[C_s .. C_s + f_s - 1] = s
   - State Inversion:
         x <- f_s * floor(x / M) + (m - C_s) = f_s * (x >> B) + (m - C_s)
   - Renormalization:
         While x < L and stream has remaining bytes:
             x <- (x << 8) | read_byte()
   Upon decoding all N symbols, the state returns precisely to x = L.

References:
- Duda, J. (2013). Asymmetric numeral systems: entropy coding combining speed of
  Huffman coding with compression rate of arithmetic coding. arXiv:1311.2540.
- Giesen, F. (2014). Interleaved range Asymmetric Numeral Systems. Technical Report.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANS_MAGIC: bytes = b"RANS"
RANS_VERSION: int = 1
DEFAULT_SCALE_BITS: int = 14  # M = 2^14 = 16,384
RANS_L: int = 1 << 23  # Lower bound L = 8,388,608

# Minimum container header length:
# magic (4) + version (1) + scale_bits (1) + uncompressed_len (8) + symbol_count (2) + payload_len (4) = 20 bytes
MIN_CONTAINER_HEADER_SIZE: int = 20


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------
class RansError(Exception):
    """Base exception for all rANS compression and decompression errors."""
    pass


class CompressionError(RansError):
    """Raised when compression fails due to invalid parameters or state."""
    pass


class DecompressionError(RansError):
    """Raised when decompression fails due to malformed containers or streams."""
    pass


class CorruptedDataError(DecompressionError):
    """Raised when container headers, checksums, or stream invariants are corrupted."""
    pass


# ---------------------------------------------------------------------------
# Frequency Table Model
# ---------------------------------------------------------------------------
class FrequencyTable:
    """
    Deterministic static byte-frequency distribution normalized to sum = 2^scale_bits.

    Uses the Hare-Niemeyer (largest remainder) method to guarantee:
    1. Every symbol present in the source has f_s >= 1.
    2. Sum of all frequencies strictly equals M = 2^scale_bits.
    3. Perfectly reproducible across platforms without floating-point drift.
    """

    def __init__(
        self,
        frequencies: Dict[int, int],
        scale_bits: int = DEFAULT_SCALE_BITS,
    ) -> None:
        if not (8 <= scale_bits <= 16):
            raise ValueError(f"scale_bits must be between 8 and 16, got {scale_bits}")

        self.scale_bits = scale_bits
        self.M = 1 << scale_bits
        self.frequencies: Dict[int, int] = {}
        self.cumulative: Dict[int, int] = {}
        self.symbols: List[int] = sorted(frequencies.keys())

        # Validate and populate
        total = 0
        for s in self.symbols:
            if not (0 <= s <= 255):
                raise ValueError(f"Symbol {s} is outside valid byte range [0, 255]")
            f = frequencies[s]
            if f <= 0:
                raise ValueError(f"Symbol {s} must have strictly positive frequency, got {f}")
            self.frequencies[s] = f
            self.cumulative[s] = total
            total += f

        if total != self.M:
            raise ValueError(
                f"Frequency sum ({total}) does not match required total M ({self.M})"
            )

    @classmethod
    def from_counts(
        cls,
        counts: Sequence[int],
        total_symbols: int,
        scale_bits: int = DEFAULT_SCALE_BITS,
    ) -> "FrequencyTable":
        """
        Normalize raw symbol counts into an exact 2^scale_bits frequency table.

        Parameters
        ----------
        counts : Sequence[int]
            Length 256 sequence of counts for each byte value.
        total_symbols : int
            Total number of symbols (sum of counts).
        scale_bits : int, default=14
            Scale precision bits (M = 2^scale_bits).

        Returns
        -------
        FrequencyTable
        """
        if total_symbols <= 0:
            raise ValueError("total_symbols must be positive to build a frequency table")

        M = 1 << scale_bits
        present = [i for i in range(256) if counts[i] > 0]
        K = len(present)

        if K == 0:
            raise ValueError("No present symbols found in counts")

        if K > M:
            raise ValueError(
                f"Number of distinct symbols ({K}) exceeds total frequency space M ({M})"
            )

        # Hare-Niemeyer (Largest Remainder) Normalization:
        # 1. Allocate minimum frequency of 1 to every present symbol
        remaining_units = M - K
        preliminary: Dict[int, int] = {s: 1 for s in present}
        remainders: List[Tuple[float, int]] = []

        for s in present:
            exact_share = (counts[s] / total_symbols) * remaining_units
            integer_share = int(exact_share)
            preliminary[s] += integer_share
            remainders.append((exact_share - integer_share, s))

        # 2. Distribute the remaining deficit to symbols with largest remainders
        deficit = M - sum(preliminary.values())
        # Break ties deterministically by symbol byte value
        remainders.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        for i in range(deficit):
            sym = remainders[i][1]
            preliminary[sym] += 1

        return cls(preliminary, scale_bits=scale_bits)

    @classmethod
    def from_data(
        cls,
        data: Union[bytes, bytearray],
        scale_bits: int = DEFAULT_SCALE_BITS,
    ) -> "FrequencyTable":
        """Construct a normalized frequency table directly from input bytes."""
        if len(data) == 0:
            raise ValueError("Cannot construct frequency table from empty data")

        counts = [0] * 256
        for b in data:
            counts[b] += 1

        return cls.from_counts(counts, len(data), scale_bits=scale_bits)

    def serialize(self) -> bytes:
        """
        Serialize frequency table into bytes:
        [symbol_count (uint16 LE)] + K * [symbol (uint8), freq (uint16 LE)]
        """
        buf = bytearray()
        buf.extend(struct.pack("<H", len(self.symbols)))
        for s in self.symbols:
            buf.append(s)
            buf.extend(struct.pack("<H", self.frequencies[s]))
        return bytes(buf)

    @classmethod
    def deserialize(
        cls,
        buffer: Union[bytes, bytearray],
        offset: int,
        scale_bits: int = DEFAULT_SCALE_BITS,
    ) -> Tuple["FrequencyTable", int]:
        """
        Deserialize a frequency table from a buffer at a given offset.

        Returns
        -------
        Tuple[FrequencyTable, int]
            (Deserialized FrequencyTable, bytes_consumed)
        """
        if len(buffer) < offset + 2:
            raise DecompressionError("Buffer truncated: cannot read frequency table symbol count")

        symbol_count = struct.unpack_from("<H", buffer, offset)[0]
        offset_idx = offset + 2

        if symbol_count == 0 or symbol_count > 256:
            raise CorruptedDataError(
                f"Invalid symbol count in frequency table: {symbol_count} (must be in 1..256)"
            )

        expected_entry_bytes = symbol_count * 3
        if len(buffer) < offset_idx + expected_entry_bytes:
            raise DecompressionError(
                f"Buffer truncated: expected {expected_entry_bytes} bytes for {symbol_count} table entries"
            )

        freqs: Dict[int, int] = {}
        for _ in range(symbol_count):
            sym = buffer[offset_idx]
            freq = struct.unpack_from("<H", buffer, offset_idx + 1)[0]
            offset_idx += 3

            if sym in freqs:
                raise CorruptedDataError(f"Duplicate symbol {sym} in serialized frequency table")
            if freq <= 0 or freq > (1 << scale_bits):
                raise CorruptedDataError(
                    f"Invalid frequency {freq} for symbol {sym} (scale_bits={scale_bits})"
                )
            freqs[sym] = freq

        table = cls(freqs, scale_bits=scale_bits)
        bytes_consumed = offset_idx - offset
        return table, bytes_consumed

    def build_lookup_table(self) -> bytearray:
        """
        Construct a direct O(1) bytearray mapping cumulative slots m in [0, M)
        to their corresponding decoded symbol s in [0, 255].
        """
        lut = bytearray(self.M)
        for s in self.symbols:
            start = self.cumulative[s]
            end = start + self.frequencies[s]
            lut[start:end] = bytes([s]) * (end - start)
        return lut


# ---------------------------------------------------------------------------
# Binary Container Format
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RansContainer:
    """
    Validated binary container for rANS-compressed data.

    Layout:
    [0..3]    Magic: b'RANS'
    [4]       Format Version: 1
    [5]       Scale Bits (B): uint8 (e.g. 14)
    [6..13]   Uncompressed Length: uint64 LE
    [14..15]  Symbol Count K: uint16 LE
    [16..16+3*K-1] Frequency Table Entries: [symbol (uint8), freq (uint16 LE)] * K
    [16+3*K..19+3*K] Payload Length: uint32 LE
    [20+3*K..]       Compressed rANS Payload Bytes
    """
    version: int
    scale_bits: int
    uncompressed_length: int
    freq_table: Optional[FrequencyTable]
    payload: bytes

    @property
    def header_size(self) -> int:
        table_bytes = (2 + len(self.freq_table.symbols) * 3) if self.freq_table else 2
        return 4 + 1 + 1 + 8 + table_bytes + 4

    @property
    def payload_size(self) -> int:
        return len(self.payload)

    @property
    def container_size(self) -> int:
        return self.header_size + self.payload_size

    def to_bytes(self) -> bytes:
        """Serialize container to binary byte string."""
        buf = bytearray()
        buf.extend(RANS_MAGIC)
        buf.append(self.version)
        buf.append(self.scale_bits)
        buf.extend(struct.pack("<Q", self.uncompressed_length))

        if self.freq_table is not None and self.uncompressed_length > 0:
            buf.extend(self.freq_table.serialize())
        else:
            buf.extend(struct.pack("<H", 0))

        buf.extend(struct.pack("<I", len(self.payload)))
        buf.extend(self.payload)
        return bytes(buf)

    @classmethod
    def from_bytes(cls, container_bytes: Union[bytes, bytearray]) -> "RansContainer":
        """
        Parse and strictly validate a binary container from bytes.

        Raises
        ------
        DecompressionError
            On truncated data or unsupported format versions.
        CorruptedDataError
            On invalid magic bytes, checksum/table mismatches, or corrupted fields.
        """
        if not isinstance(container_bytes, (bytes, bytearray)):
            raise TypeError(
                f"Expected bytes or bytearray, got {type(container_bytes).__name__}"
            )

        if len(container_bytes) < MIN_CONTAINER_HEADER_SIZE:
            raise DecompressionError(
                f"Container too short: expected at least {MIN_CONTAINER_HEADER_SIZE} bytes, "
                f"got {len(container_bytes)}"
            )

        # 1. Magic
        magic = bytes(container_bytes[:4])
        if magic != RANS_MAGIC:
            raise CorruptedDataError(
                f"Invalid magic identifier: expected {RANS_MAGIC!r}, got {magic!r}"
            )

        # 2. Version
        version = container_bytes[4]
        if version != RANS_VERSION:
            raise DecompressionError(
                f"Unsupported rANS format version: {version} (expected {RANS_VERSION})"
            )

        # 3. Scale bits
        scale_bits = container_bytes[5]
        if not (8 <= scale_bits <= 16):
            raise CorruptedDataError(f"Invalid scale bits in container: {scale_bits}")

        # 4. Uncompressed length
        uncompressed_length = struct.unpack_from("<Q", container_bytes, 6)[0]

        # 5. Frequency Table
        offset = 14
        if len(container_bytes) < offset + 2:
            raise DecompressionError("Truncated container: cannot read symbol count")

        symbol_count = struct.unpack_from("<H", container_bytes, offset)[0]

        freq_table: Optional[FrequencyTable] = None
        if uncompressed_length == 0:
            if symbol_count != 0:
                raise CorruptedDataError(
                    f"Inconsistent container: symbol_count={symbol_count} for empty uncompressed length"
                )
            offset += 2
        else:
            if symbol_count == 0:
                raise CorruptedDataError(
                    "Inconsistent container: symbol_count=0 but uncompressed_length > 0"
                )
            freq_table, table_bytes = FrequencyTable.deserialize(
                container_bytes, offset, scale_bits=scale_bits
            )
            offset += table_bytes

        # 6. Payload Length & Payload
        if len(container_bytes) < offset + 4:
            raise DecompressionError("Truncated container: cannot read payload length")

        payload_length = struct.unpack_from("<I", container_bytes, offset)[0]
        offset += 4

        payload = bytes(container_bytes[offset : offset + payload_length])
        if len(payload) != payload_length:
            raise DecompressionError(
                f"Truncated container payload: expected {payload_length} bytes, "
                f"found {len(payload)}"
            )

        # Reject trailing garbage for container integrity
        if offset + payload_length != len(container_bytes):
            raise CorruptedDataError(
                f"Container has {len(container_bytes) - (offset + payload_length)} unexpected trailing bytes"
            )

        return cls(
            version=version,
            scale_bits=scale_bits,
            uncompressed_length=uncompressed_length,
            freq_table=freq_table,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# rANS Encoder
# ---------------------------------------------------------------------------
class RansEncoder:
    """
    Byte-oriented range Asymmetric Numeral Systems (rANS) Encoder.
    """

    def __init__(
        self,
        scale_bits: int = DEFAULT_SCALE_BITS,
        L: int = RANS_L,
    ) -> None:
        self.scale_bits = scale_bits
        self.L = L

    def encode_to_container(self, data: Union[bytes, bytearray]) -> RansContainer:
        """
        Encode an arbitrary byte sequence into a validated RansContainer.

        Parameters
        ----------
        data : bytes or bytearray
            Input uncompressed byte stream.

        Returns
        -------
        RansContainer
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes or bytearray, got {type(data).__name__}")

        n_bytes = len(data)

        # Empty input edge case
        if n_bytes == 0:
            return RansContainer(
                version=RANS_VERSION,
                scale_bits=self.scale_bits,
                uncompressed_length=0,
                freq_table=None,
                payload=b"",
            )

        freq_table = FrequencyTable.from_data(data, scale_bits=self.scale_bits)
        M = freq_table.M
        scale_bits = self.scale_bits
        L = self.L

        # Precompute lookup arrays for high-speed encoding
        f_table = [0] * 256
        c_table = [0] * 256
        x_max_table = [0] * 256

        shift_factor = (L >> scale_bits) << 8
        for s in freq_table.symbols:
            f = freq_table.frequencies[s]
            f_table[s] = f
            c_table[s] = freq_table.cumulative[s]
            x_max_table[s] = shift_factor * f

        # Encode backwards (LIFO stack behavior of rANS)
        x = L
        out = bytearray()
        out_append = out.append

        for byte_val in reversed(data):
            f = f_table[byte_val]
            c = c_table[byte_val]
            x_max = x_max_table[byte_val]

            # Renormalization: flush low-order bytes until x < x_max(s)
            while x >= x_max:
                out_append(x & 0xFF)
                x >>= 8

            # State transition: x <- floor(x / f_s) * M + C_s + (x mod f_s)
            x = (x // f) * M + c + (x % f)

        # Payload assembly: [x_final (uint32 LE)] + [reversed emitted bytes]
        payload = bytearray(x.to_bytes(4, "little"))
        out.reverse()
        payload.extend(out)

        return RansContainer(
            version=RANS_VERSION,
            scale_bits=self.scale_bits,
            uncompressed_length=n_bytes,
            freq_table=freq_table,
            payload=bytes(payload),
        )

    def encode(self, data: Union[bytes, bytearray]) -> bytes:
        """Encode bytes and return serialized container bytes."""
        container = self.encode_to_container(data)
        return container.to_bytes()


# ---------------------------------------------------------------------------
# rANS Decoder
# ---------------------------------------------------------------------------
class RansDecoder:
    """
    Byte-oriented range Asymmetric Numeral Systems (rANS) Decoder.
    """

    def __init__(self, L: int = RANS_L) -> None:
        self.L = L

    def decode_container(self, container: RansContainer) -> bytes:
        """
        Decompress a validated RansContainer back into original bytes.

        Parameters
        ----------
        container : RansContainer
            The validated container.

        Returns
        -------
        bytes
            Exact reconstructed uncompressed bytes.
        """
        n_bytes = container.uncompressed_length

        # Empty data edge case
        if n_bytes == 0:
            if len(container.payload) != 0:
                raise CorruptedDataError("Non-empty payload for 0-length uncompressed data")
            return b""

        if container.freq_table is None:
            raise CorruptedDataError("Missing frequency table for non-empty uncompressed data")

        payload = container.payload
        if len(payload) < 4:
            raise DecompressionError(
                f"Payload too short for rANS state: expected >= 4 bytes, got {len(payload)}"
            )

        freq_table = container.freq_table
        scale_bits = container.scale_bits
        M = 1 << scale_bits
        mask_m = M - 1
        L = self.L

        # Precompute tables
        f_table = [0] * 256
        c_table = [0] * 256
        for s in freq_table.symbols:
            f_table[s] = freq_table.frequencies[s]
            c_table[s] = freq_table.cumulative[s]

        lut = freq_table.build_lookup_table()

        # Initialize decoder state from leading 4 bytes of payload
        x = int.from_bytes(payload[:4], "little")
        ptr = 4
        payload_len = len(payload)

        decoded = bytearray(n_bytes)

        # Decoding loop
        for i in range(n_bytes):
            # Frequency slot in [0, M)
            m = x & mask_m
            # O(1) symbol lookup
            s = lut[m]
            decoded[i] = s

            f = f_table[s]
            c = c_table[s]

            # Invert state transition: x <- f * (x >> scale_bits) + (m - C_s)
            x = f * (x >> scale_bits) + (m - c)

            # Renormalization: reload bytes while x < L
            while x < L and ptr < payload_len:
                x = (x << 8) | payload[ptr]
                ptr += 1

        # Strict invariant verification:
        # Upon decoding exactly n_bytes symbols, state must equal initial state L
        if x != L:
            raise CorruptedDataError(
                f"rANS state invariant violation: decoder ended with state {x} (expected {L})"
            )

        if ptr != payload_len:
            raise CorruptedDataError(
                f"rANS payload has {payload_len - ptr} unused bytes remaining after decoding"
            )

        return bytes(decoded)

    def decode(self, container_bytes: Union[bytes, bytearray]) -> bytes:
        """Parse container bytes and decompress back to original bytes."""
        container = RansContainer.from_bytes(container_bytes)
        return self.decode_container(container)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------
def rans_compress(
    data: Union[bytes, bytearray],
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> bytes:
    """Convenience helper to compress bytes into a rANS binary container."""
    encoder = RansEncoder(scale_bits=scale_bits)
    return encoder.encode(data)


def rans_decompress(container_bytes: Union[bytes, bytearray]) -> bytes:
    """Convenience helper to decompress a rANS binary container into bytes."""
    decoder = RansDecoder()
    return decoder.decode(container_bytes)


# ---------------------------------------------------------------------------
# CLI Demonstration & Benchmarking
# ---------------------------------------------------------------------------
def run_rans_demo(
    file_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    scale_bits: int = DEFAULT_SCALE_BITS,
    verbose: bool = True,
) -> Dict[str, Union[int, float, str, bool]]:
    """
    Demonstrate rANS compression and decompression on a real file.

    Compares SHA-256 hashes, measures encoding/decoding latency, calculates
    overall container and payload compression ratios, and returns summary metrics.
    """
    input_p = Path(file_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Input file not found: {input_p}")

    original_bytes = input_p.read_bytes()
    orig_size = len(original_bytes)
    orig_sha256 = hashlib.sha256(original_bytes).hexdigest()

    encoder = RansEncoder(scale_bits=scale_bits)
    decoder = RansDecoder()

    # Measure encoding
    t_enc_start = time.perf_counter()
    container = encoder.encode_to_container(original_bytes)
    container_bytes = container.to_bytes()
    t_enc_end = time.perf_counter()
    enc_time_sec = t_enc_end - t_enc_start

    container_size = len(container_bytes)
    payload_size = container.payload_size
    header_size = container.header_size

    # Optionally write to output file
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(container_bytes)

    # Measure decoding
    t_dec_start = time.perf_counter()
    reconstructed_bytes = decoder.decode(container_bytes)
    t_dec_end = time.perf_counter()
    dec_time_sec = t_dec_end - t_dec_start

    rec_sha256 = hashlib.sha256(reconstructed_bytes).hexdigest()
    exact_match = (original_bytes == reconstructed_bytes)
    sha256_match = (orig_sha256 == rec_sha256)

    # Ratios
    overall_cr = (orig_size / container_size) if container_size > 0 else 0.0
    payload_cr = (orig_size / payload_size) if payload_size > 0 else 0.0
    space_saving_pct = (1.0 - container_size / orig_size) * 100.0 if orig_size > 0 else 0.0

    enc_throughput_mb_s = (orig_size / (1024 * 1024)) / enc_time_sec if enc_time_sec > 0 else 0.0
    dec_throughput_mb_s = (orig_size / (1024 * 1024)) / dec_time_sec if dec_time_sec > 0 else 0.0

    results = {
        "file_name": input_p.name,
        "original_size_bytes": orig_size,
        "container_size_bytes": container_size,
        "payload_size_bytes": payload_size,
        "header_size_bytes": header_size,
        "overall_compression_ratio": overall_cr,
        "payload_compression_ratio": payload_cr,
        "space_saving_percent": space_saving_pct,
        "exact_reconstruction": exact_match,
        "sha256_verified": sha256_match,
        "original_sha256": orig_sha256,
        "reconstructed_sha256": rec_sha256,
        "encoding_time_ms": enc_time_sec * 1000.0,
        "decoding_time_ms": dec_time_sec * 1000.0,
        "encoding_throughput_mb_s": enc_throughput_mb_s,
        "decoding_throughput_mb_s": dec_throughput_mb_s,
    }

    if verbose:
        print("=" * 70)
        print("rANS Compression Demonstration & Verification")
        print("=" * 70)
        print(f"File Name                  : {input_p.name}")
        print(f"Original Size              : {orig_size:,} bytes")
        print(f"Compressed Container Size  : {container_size:,} bytes (incl. headers & table)")
        print(f"Payload Size (rANS only)   : {payload_size:,} bytes")
        print(f"Header/Metadata Overhead   : {header_size:,} bytes")
        print(f"Overall Compression Ratio  : {overall_cr:.4f}:1")
        print(f"Payload Compression Ratio  : {payload_cr:.4f}:1")
        print(f"Space Saving               : {space_saving_pct:.2f}%")
        print(f"Original SHA-256           : {orig_sha256}")
        print(f"Reconstructed SHA-256      : {rec_sha256}")
        print(f"Exact Reconstruction       : {'PASS' if exact_match else 'FAIL'}")
        print(f"Encoding Latency           : {enc_time_sec*1000.0:.2f} ms ({enc_throughput_mb_s:.2f} MB/s)")
        print(f"Decoding Latency           : {dec_time_sec*1000.0:.2f} ms ({dec_throughput_mb_s:.2f} MB/s)")
        print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="rANS Byte-Oriented Source Encoder and Decoder Demonstration"
    )
    parser.add_argument("input_file", help="Path to text or binary input file")
    parser.add_argument(
        "-o", "--output", help="Optional output path for compressed container file"
    )
    parser.add_argument(
        "-b", "--scale-bits", type=int, default=DEFAULT_SCALE_BITS,
        help=f"Precision scale bits B (default: {DEFAULT_SCALE_BITS})"
    )

    args = parser.parse_args()
    try:
        res = run_rans_demo(
            args.input_file,
            output_path=args.output,
            scale_bits=args.scale_bits,
            verbose=True,
        )
        sys.exit(0 if res["exact_reconstruction"] else 1)
    except Exception as e:
        print(f"Error during rANS execution: {e}", file=sys.stderr)
        sys.exit(2)
