"""
Unit tests for bit packing, unpacking, padding, and unpadding utilities.
"""

import numpy as np
import pytest

from src.utils.bit_utils import (
    bits_to_bytes,
    bytes_to_bits,
    pad_bits,
    unpad_bits,
)


class TestBytesToBits:
    def test_empty_bytes(self):
        bits = bytes_to_bits(b"")
        assert isinstance(bits, np.ndarray)
        assert bits.size == 0
        assert bits.dtype == np.uint8

    def test_single_byte_msb(self):
        # 0b10100101 = 165
        data = bytes([0b10100101])
        bits = bytes_to_bits(data, msb_first=True)
        expected = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(bits, expected)

    def test_single_byte_lsb(self):
        # 0b10100101: LSB first means bit 0 is at index 0
        data = bytes([0b10100101])
        bits = bytes_to_bits(data, msb_first=False)
        expected = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        # Note: 0b10100101 has bit 0 = 1, bit 1 = 0, bit 2 = 1, etc.
        np.testing.assert_array_equal(bits, expected)

    def test_type_error(self):
        with pytest.raises(TypeError):
            bytes_to_bits("not bytes")  # type: ignore


class TestBitsToBytes:
    def test_empty_bits(self):
        assert bits_to_bytes(np.empty(0, dtype=np.uint8)) == b""
        assert bits_to_bytes([]) == b""

    def test_single_byte_msb(self):
        bits = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        assert bits_to_bytes(bits, msb_first=True) == bytes([0b10100101])

    def test_non_multiple_of_eight_raises_value_error(self):
        with pytest.raises(ValueError, match="must be an exact multiple of 8"):
            bits_to_bytes(np.array([1, 0, 1], dtype=np.uint8))

    def test_non_binary_raises_value_error(self):
        with pytest.raises(ValueError, match="must contain only 0s and 1s"):
            bits_to_bytes(np.array([1, 0, 2, 0, 0, 0, 0, 0], dtype=np.uint8))

    def test_multidimensional_raises_value_error(self):
        with pytest.raises(ValueError, match="must be 1D"):
            bits_to_bytes(np.zeros((2, 8), dtype=np.uint8))


class TestBitPackingRoundTrip:
    @pytest.mark.parametrize("text", [
        "Hello, World!",
        "ITC Project: rANS & Polar Codes",
        "Deterministic 1234567890 !@#$%^&*()",
        "Multi-byte UTF-8: äöüß—λπθ—🚀",
    ])
    def test_utf8_text_roundtrip(self, text):
        original_bytes = text.encode("utf-8")
        bits = bytes_to_bits(original_bytes, msb_first=True)
        reconstructed_bytes = bits_to_bytes(bits, msb_first=True)
        assert reconstructed_bytes == original_bytes
        assert reconstructed_bytes.decode("utf-8") == text

    def test_random_binary_bytes_roundtrip(self):
        rng = np.random.default_rng(42)
        random_bytes = rng.integers(0, 256, size=512, dtype=np.uint8).tobytes()
        bits = bytes_to_bits(random_bytes, msb_first=True)
        recovered = bits_to_bytes(bits, msb_first=True)
        assert recovered == random_bytes


class TestPadding:
    def test_no_padding_needed(self):
        bits = np.ones(16, dtype=np.uint8)
        padded, pad_count = pad_bits(bits, block_size=8)
        assert pad_count == 0
        np.testing.assert_array_equal(padded, bits)

    def test_padding_applied(self):
        bits = np.array([1, 1, 0, 1, 0], dtype=np.uint8)  # length 5
        padded, pad_count = pad_bits(bits, block_size=8, pad_value=0)
        assert pad_count == 3
        assert padded.size == 8
        np.testing.assert_array_equal(
            padded, np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
        )

    def test_unpad(self):
        original = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        padded, pad_count = pad_bits(original, block_size=64, pad_value=0)
        assert padded.size == 64
        assert pad_count == 59
        recovered = unpad_bits(padded, pad_count)
        np.testing.assert_array_equal(recovered, original)

    def test_invalid_padding_params(self):
        bits = np.ones(8, dtype=np.uint8)
        with pytest.raises(ValueError, match="block_size must be positive"):
            pad_bits(bits, block_size=0)
        with pytest.raises(ValueError, match="pad_value must be 0 or 1"):
            pad_bits(bits, block_size=8, pad_value=2)
        with pytest.raises(ValueError, match="pad_count cannot be negative"):
            unpad_bits(bits, pad_count=-1)
        with pytest.raises(ValueError, match="Cannot remove"):
            unpad_bits(bits, pad_count=10)
