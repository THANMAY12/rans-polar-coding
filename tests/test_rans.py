"""
Comprehensive test suite for range Asymmetric Numeral Systems (rANS) source codec.

Tests include:
- Empty input, single-byte and short sequences
- Repeated symbols and highly skewed distributions
- Full 256-byte alphabet coverage
- Unicode UTF-8 multi-byte strings and emojis
- Deterministic seeded randomized roundtrips
- Large input files (>100 KB)
- Binary container format and malformed/corrupted data validation
- CLI demonstration runner and metric integrity
"""

import hashlib
from pathlib import Path
import struct
import numpy as np
import pytest

from src.source_coding.rans_codec import (
    CorruptedDataError,
    DecompressionError,
    DEFAULT_SCALE_BITS,
    FrequencyTable,
    RANS_L,
    RANS_MAGIC,
    RANS_VERSION,
    RansContainer,
    RansDecoder,
    RansEncoder,
    rans_compress,
    rans_decompress,
    run_rans_demo,
)
from src.utils.seed import get_rng


# ---------------------------------------------------------------------------
# 1. Edge Cases: Empty, Single-Byte, and Short Inputs
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_input(self):
        empty_data = b""
        compressed = rans_compress(empty_data)
        assert len(compressed) >= 20  # Minimum container header size
        recovered = rans_decompress(compressed)
        assert recovered == empty_data

    @pytest.mark.parametrize("char", [b"A", b"\x00", b"\xff", b"~", b"\n"])
    def test_single_byte_input(self, char):
        compressed = rans_compress(char)
        recovered = rans_decompress(compressed)
        assert recovered == char

    @pytest.mark.parametrize("short_seq", [
        b"ab",
        b"xyz",
        b"1234",
        b"short!",
        b"\x00\x01\x00\x02",
    ])
    def test_short_inputs(self, short_seq):
        compressed = rans_compress(short_seq)
        recovered = rans_decompress(compressed)
        assert recovered == short_seq


# ---------------------------------------------------------------------------
# 2. Repeated Symbols and Skewed Distributions
# ---------------------------------------------------------------------------
class TestRepeatedAndSkewedInputs:
    @pytest.mark.parametrize("length", [10, 100, 1000, 10000])
    def test_identical_symbols(self, length):
        data = b"A" * length
        compressed = rans_compress(data)
        recovered = rans_decompress(compressed)
        assert recovered == data
        # For completely uniform repeated symbols, rANS payload should be tiny (4 bytes for state)
        container = RansContainer.from_bytes(compressed)
        assert container.payload_size == 4
        # Compression ratio exceeds 1.0 once input length exceeds container overhead
        if length >= 100:
            assert len(data) / len(compressed) > 1.0

    def test_highly_skewed_distribution(self):
        # 99.5% 'a', 0.5% 'b'
        data = (b"a" * 1990) + (b"b" * 10)
        compressed = rans_compress(data)
        recovered = rans_decompress(compressed)
        assert recovered == data
        # Should achieve strong compression
        assert len(compressed) < len(data) / 3


# ---------------------------------------------------------------------------
# 3. All 256 Byte Values
# ---------------------------------------------------------------------------
class TestAll256ByteValues:
    def test_full_byte_range_forward(self):
        data = bytes(range(256))
        compressed = rans_compress(data)
        recovered = rans_decompress(compressed)
        assert recovered == data

    def test_full_byte_range_reversed(self):
        data = bytes(reversed(range(256)))
        compressed = rans_compress(data)
        recovered = rans_decompress(compressed)
        assert recovered == data

    def test_full_byte_range_repeated_blocks(self):
        data = bytes(range(256)) * 8
        compressed = rans_compress(data)
        recovered = rans_decompress(compressed)
        assert recovered == data


# ---------------------------------------------------------------------------
# 4. Unicode UTF-8 Text
# ---------------------------------------------------------------------------
class TestUnicodeUtf8:
    @pytest.mark.parametrize("text", [
        "ASCII plain text with punctuation and numbers: 12345, !@#$%^&*()_+.",
        "French accents: L'entropie d'une source d'information caractérise son contenu.",
        "German umlauts: Übergrößenträger, Maßstäbe, zwölf Käfer.",
        "Greek symbols: H(X) = - \\sum P(x) \\log_2 P(x), \\alpha, \\beta, \\gamma, \\lambda.",
        "Japanese script: 情報理論と符号化 (Information Theory and Coding).",
        "Emojis and modern Unicode: 🚀 📡 🛰️ 💻 🔬 📊.",
        "Mixed technical document:\nLine 1: Hello World!\nLine 2: π ≈ 3.1415926535\nLine 3: ✓ Done.",
    ])
    def test_utf8_strings_roundtrip(self, text):
        raw_bytes = text.encode("utf-8")
        compressed = rans_compress(raw_bytes)
        recovered_bytes = rans_decompress(compressed)
        assert recovered_bytes == raw_bytes
        assert recovered_bytes.decode("utf-8") == text


# ---------------------------------------------------------------------------
# 5. Deterministic Randomized Roundtrips
# ---------------------------------------------------------------------------
class TestRandomizedRoundtrips:
    def test_seeded_pseudo_random_corpora(self):
        rng = get_rng(2026)

        # Diverse alphabet sizes and lengths
        test_cases = [
            (2, 500),      # Binary alphabet {65, 66}
            (4, 1000),     # 4-symbol DNA-like {A, C, G, T}
            (26, 4000),    # English alphabet {A-Z}
            (256, 10000),  # Full byte alphabet
        ]

        for num_symbols, length in test_cases:
            symbols = rng.choice(256, size=num_symbols, replace=False)
            random_bytes = bytes(rng.choice(symbols, size=length))

            compressed = rans_compress(random_bytes)
            decompressed = rans_decompress(compressed)

            assert decompressed == random_bytes
            assert hashlib.sha256(decompressed).digest() == hashlib.sha256(random_bytes).digest()


# ---------------------------------------------------------------------------
# 6. Large Input Files
# ---------------------------------------------------------------------------
class TestLargeInput:
    def test_large_structured_corpus(self):
        rng = get_rng(42)
        # Generate 150 KB of realistic structured text
        words = [b"information", b"theory", b"rans", b"polar", b"entropy", b"channel", b"coding", b"text"]
        paragraph = b" ".join(rng.choice(words, size=30)) + b"\n"
        large_data = paragraph * 800  # ~150 KB

        encoder = RansEncoder()
        decoder = RansDecoder()

        container_bytes = encoder.encode(large_data)
        assert len(container_bytes) < len(large_data)  # Guaranteed compression

        reconstructed = decoder.decode(container_bytes)
        assert reconstructed == large_data
        assert hashlib.sha256(reconstructed).hexdigest() == hashlib.sha256(large_data).hexdigest()


# ---------------------------------------------------------------------------
# 7. Container Validation and Malformed Inputs
# ---------------------------------------------------------------------------
class TestContainerValidationAndErrorHandling:
    def test_non_bytes_input(self):
        with pytest.raises(TypeError):
            rans_compress("not bytes")  # type: ignore

        with pytest.raises(TypeError):
            rans_decompress("not bytes")  # type: ignore

    def test_container_too_short(self):
        with pytest.raises(DecompressionError, match="Container too short"):
            rans_decompress(b"RANS\x01")

    def test_invalid_magic_identifier(self):
        valid = rans_compress(b"valid test payload")
        corrupted = b"XXXX" + valid[4:]
        with pytest.raises(CorruptedDataError, match="Invalid magic identifier"):
            rans_decompress(corrupted)

    def test_unsupported_version(self):
        valid = rans_compress(b"valid test payload")
        # Change version byte at index 4 to 99
        corrupted = valid[:4] + bytes([99]) + valid[5:]
        with pytest.raises(DecompressionError, match="Unsupported rANS format version"):
            rans_decompress(corrupted)

    def test_invalid_scale_bits(self):
        valid = rans_compress(b"valid test payload")
        # Change scale bits byte at index 5 to 5 (invalid < 8)
        corrupted = valid[:5] + bytes([5]) + valid[6:]
        with pytest.raises(CorruptedDataError, match="Invalid scale bits"):
            rans_decompress(corrupted)

    def test_truncated_payload(self):
        valid = rans_compress(b"This is a longer payload to ensure multiple bytes")
        truncated = valid[:-5]
        with pytest.raises(DecompressionError, match="Truncated container payload|Container truncated"):
            rans_decompress(truncated)

    def test_extra_trailing_bytes(self):
        valid = rans_compress(b"payload")
        corrupted = valid + b"EXTRA_GARBAGE"
        with pytest.raises(CorruptedDataError, match="unexpected trailing bytes"):
            rans_decompress(corrupted)

    def test_corrupted_payload_bytes_detected(self):
        valid = rans_compress(b"A sufficiently long message that creates a multi-byte rANS state stream")
        # Invert bytes inside payload
        corrupted = bytearray(valid)
        corrupted[-3] ^= 0xFF
        with pytest.raises(CorruptedDataError):
            rans_decompress(corrupted)

    def test_inconsistent_container_headers(self):
        # Empty length but non-zero symbol count
        buf = bytearray(RANS_MAGIC)
        buf.append(RANS_VERSION)
        buf.append(DEFAULT_SCALE_BITS)
        buf.extend(struct.pack("<Q", 0))  # 0 uncompressed length
        buf.extend(struct.pack("<H", 5))  # says 5 symbols (inconsistent!)
        buf.extend(struct.pack("<I", 0))
        with pytest.raises(CorruptedDataError, match="Inconsistent container"):
            RansContainer.from_bytes(buf)


# ---------------------------------------------------------------------------
# 8. Frequency Table Mathematical Invariants
# ---------------------------------------------------------------------------
class TestFrequencyTableInvariants:
    def test_hare_niemeyer_exact_sum(self):
        data = b"Arbitrary source text for testing normalization exactness!"
        table = FrequencyTable.from_data(data, scale_bits=14)
        assert table.M == 16384
        assert sum(table.frequencies.values()) == 16384
        for s in table.symbols:
            assert table.frequencies[s] >= 1

    def test_cumulative_monotonicity(self):
        data = b"Testing cumulative frequencies."
        table = FrequencyTable.from_data(data, scale_bits=12)
        assert table.M == 4096
        current_cum = 0
        for s in table.symbols:
            assert table.cumulative[s] == current_cum
            current_cum += table.frequencies[s]
        assert current_cum == table.M

    def test_table_serialization_roundtrip(self):
        data = b"Serialization verification stream 12345"
        table1 = FrequencyTable.from_data(data, scale_bits=14)
        serialized = table1.serialize()

        table2, consumed = FrequencyTable.deserialize(serialized, 0, scale_bits=14)
        assert consumed == len(serialized)
        assert table1.frequencies == table2.frequencies
        assert table1.cumulative == table2.cumulative


# ---------------------------------------------------------------------------
# 9. CLI Demonstration Runner
# ---------------------------------------------------------------------------
class TestCliDemoRunner:
    def test_run_rans_demo(self, tmp_path: Path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("rANS CLI Demonstration and SHA-256 validation test.\n" * 20, encoding="utf-8")
        out_file = tmp_path / "sample.rans"

        results = run_rans_demo(test_file, output_path=out_file, verbose=False)

        assert results["exact_reconstruction"] is True
        assert results["sha256_verified"] is True
        assert results["original_size_bytes"] == test_file.stat().st_size
        assert results["container_size_bytes"] == out_file.stat().st_size
        assert results["overall_compression_ratio"] > 1.0
        assert results["encoding_time_ms"] > 0
        assert results["decoding_time_ms"] > 0
