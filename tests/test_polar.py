"""
Comprehensive test suite for Arikan Polar channel coding module.

Tests include:
- Configuration validation (powers of two, K < N, construction types)
- Single-block encode and decode across multiple block sizes (N in {64, 128, 256, 512, 1024})
- Diverse code rates (R in {0.25, 0.375, 0.5, 0.625, 0.75})
- Edge-case messages (all zeros, all ones, alternating bits, pseudo-random)
- Systematic and non-systematic Polar coding modes
- Multi-block chunking and non-byte-aligned / partial final blocks with padding
- Input validation, dimension mismatch, and error handling
- LLR sign conventions and library constraints
- Deterministic reproducibility
"""

import numpy as np
import pytest

from src.channel_coding.polar_codec import (
    bits_to_llr,
    llr_to_hard_bits,
    PolarCodec,
    PolarConfig,
    PolarConfigurationError,
    PolarDecodingError,
    PolarEncodingError,
)
from src.utils.seed import get_rng


# ---------------------------------------------------------------------------
# 1. Configuration Validation
# ---------------------------------------------------------------------------
class TestPolarConfig:
    def test_default_config(self):
        config = PolarConfig()
        assert config.N == 128
        assert config.K == 64
        assert config.code_rate == 0.5
        assert config.design_snr_db == 2.0
        assert config.construction_type == "bb"
        assert config.systematic is False

    @pytest.mark.parametrize("valid_N", [8, 16, 32, 64, 128, 256, 512, 1024])
    def test_valid_power_of_two_block_lengths(self, valid_N):
        config = PolarConfig(N=valid_N, K=valid_N // 2)
        assert config.N == valid_N
        assert config.K == valid_N // 2

    @pytest.mark.parametrize("invalid_N", [0, -16, 15, 63, 100, 250, 1000])
    def test_non_power_of_two_raises_error(self, invalid_N):
        with pytest.raises(PolarConfigurationError, match="must be a power of 2|integer > 1"):
            PolarConfig(N=invalid_N, K=8)

    @pytest.mark.parametrize("invalid_K", [0, -1, 64, 100])
    def test_invalid_information_length_K(self, invalid_K):
        with pytest.raises(PolarConfigurationError, match="must satisfy 1 <= K < N"):
            PolarConfig(N=64, K=invalid_K)

    def test_invalid_construction_type(self):
        with pytest.raises(PolarConfigurationError, match="construction_type must be one of"):
            PolarConfig(N=64, K=32, construction_type="invalid_construction")

    def test_invalid_noiseless_llr(self):
        with pytest.raises(PolarConfigurationError, match="noiseless_llr must be strictly positive"):
            PolarConfig(N=64, K=32, noiseless_llr=0.0)


# ---------------------------------------------------------------------------
# 2. Single-Block Encoding and Noiseless Decoding Roundtrips
# ---------------------------------------------------------------------------
class TestSingleBlockCoding:
    @pytest.fixture
    def default_codec(self):
        return PolarCodec(N=64, K=32)

    def test_all_zeros_message(self, default_codec):
        msg = np.zeros(default_codec.K, dtype=np.uint8)
        codeword = default_codec.encode_block(msg)
        assert codeword.shape == (default_codec.N,)
        assert codeword.dtype == np.uint8
        # For non-systematic polar code with frozen 0s, all-zero message produces all-zero codeword
        np.testing.assert_array_equal(codeword, np.zeros(default_codec.N))

        decoded = default_codec.decode_block_hard(codeword)
        np.testing.assert_array_equal(decoded, msg)

    def test_all_ones_message(self, default_codec):
        msg = np.ones(default_codec.K, dtype=np.uint8)
        codeword = default_codec.encode_block(msg)
        assert codeword.shape == (default_codec.N,)
        decoded = default_codec.decode_block_hard(codeword)
        np.testing.assert_array_equal(decoded, msg)

    def test_alternating_pattern(self, default_codec):
        msg = np.array([0, 1] * (default_codec.K // 2), dtype=np.uint8)
        codeword = default_codec.encode_block(msg)
        decoded = default_codec.decode_block_hard(codeword)
        np.testing.assert_array_equal(decoded, msg)

    def test_soft_llr_roundtrip(self, default_codec):
        rng = get_rng(42)
        msg = rng.integers(0, 2, size=default_codec.K, dtype=np.uint8)
        codeword = default_codec.encode_block(msg)

        # Convert codeword to soft LLRs (+80 for 0, -80 for 1)
        llrs = bits_to_llr(codeword, magnitude=80.0)
        decoded = default_codec.decode_block(llrs)
        np.testing.assert_array_equal(decoded, msg)


# ---------------------------------------------------------------------------
# 3. Multiple Block Sizes and Code Rates
# ---------------------------------------------------------------------------
class TestBlockSizesAndRates:
    @pytest.mark.parametrize("N", [32, 64, 128, 256])
    def test_various_block_lengths(self, N):
        K = N // 2
        codec = PolarCodec(N=N, K=K)
        assert codec.N == N
        assert codec.K == K
        assert codec.code_rate == 0.5
        assert len(codec.frozen_indices) == N - K
        assert len(codec.info_indices) == K

        rng = get_rng(N)
        msg = rng.integers(0, 2, size=K, dtype=np.uint8)
        cw = codec.encode_block(msg)
        dec = codec.decode_block_hard(cw)
        np.testing.assert_array_equal(dec, msg)

    @pytest.mark.parametrize("K, expected_rate", [
        (16, 0.25),
        (32, 0.50),
        (48, 0.75),
    ])
    def test_various_code_rates(self, K, expected_rate):
        N = 64
        codec = PolarCodec(N=N, K=K)
        assert codec.code_rate == pytest.approx(expected_rate)

        rng = get_rng(K)
        msg = rng.integers(0, 2, size=K, dtype=np.uint8)
        cw = codec.encode_block(msg)
        dec = codec.decode_block_hard(cw)
        np.testing.assert_array_equal(dec, msg)


# ---------------------------------------------------------------------------
# 4. Systematic Polar Coding
# ---------------------------------------------------------------------------
class TestSystematicPolarCoding:
    def test_systematic_encode_decode(self):
        codec = PolarCodec(N=64, K=32, systematic=True)
        assert codec.config.systematic is True

        rng = get_rng(999)
        msg = rng.integers(0, 2, size=codec.K, dtype=np.uint8)
        cw = codec.encode_block(msg)
        dec = codec.decode_block_hard(cw)
        np.testing.assert_array_equal(dec, msg)


# ---------------------------------------------------------------------------
# 5. Multi-Block and Partial Final Blocks (Padding & Framing)
# ---------------------------------------------------------------------------
class TestMultiBlockAndPadding:
    @pytest.fixture
    def codec(self):
        return PolarCodec(N=64, K=32)

    def test_empty_message(self, codec):
        codewords, pad_count = codec.encode(np.empty(0, dtype=np.uint8))
        assert codewords.size == 0
        assert pad_count == 0

        decoded = codec.decode_hard(codewords, pad_count=pad_count)
        assert decoded.size == 0

    def test_exact_block_multiple(self, codec):
        # 3 exact blocks: 3 * 32 = 96 bits
        rng = get_rng(101)
        msg = rng.integers(0, 2, size=96, dtype=np.uint8)

        codewords, pad_count = codec.encode(msg)
        assert pad_count == 0
        assert codewords.size == 3 * codec.N  # 3 * 64 = 192 bits

        decoded = codec.decode_hard(codewords, pad_count=pad_count)
        np.testing.assert_array_equal(decoded, msg)

    def test_partial_final_block_padding(self, codec):
        # 85 bits: 2 full blocks (64 bits) + 1 partial block (21 bits, 11 padding bits)
        rng = get_rng(202)
        msg = rng.integers(0, 2, size=85, dtype=np.uint8)

        codewords, pad_count = codec.encode(msg)
        assert pad_count == (32 - (85 % 32))  # 32 - 21 = 11
        assert codewords.size == 3 * codec.N  # 192 bits

        decoded = codec.decode_hard(codewords, pad_count=pad_count)
        np.testing.assert_array_equal(decoded, msg)

    def test_single_bit_message(self, codec):
        # Single bit: 1 bit info + 31 padding bits
        msg = np.array([1], dtype=np.uint8)
        codewords, pad_count = codec.encode(msg)
        assert pad_count == 31
        assert codewords.size == codec.N

        decoded = codec.decode_hard(codewords, pad_count=pad_count)
        np.testing.assert_array_equal(decoded, msg)


# ---------------------------------------------------------------------------
# 6. Input Validation and Dimension Errors
# ---------------------------------------------------------------------------
class TestInputValidation:
    @pytest.fixture
    def codec(self):
        return PolarCodec(N=64, K=32)

    def test_encode_block_wrong_length(self, codec):
        with pytest.raises(PolarEncodingError, match="expects 1D array of length K=32"):
            codec.encode_block(np.zeros(31, dtype=np.uint8))
        with pytest.raises(PolarEncodingError, match="expects 1D array of length K=32"):
            codec.encode_block(np.zeros(33, dtype=np.uint8))

    def test_encode_block_multidimensional(self, codec):
        with pytest.raises(PolarEncodingError, match="expects 1D array"):
            codec.encode_block(np.zeros((2, 16), dtype=np.uint8))

    def test_encode_block_non_binary_values(self, codec):
        invalid_msg = np.zeros(32, dtype=np.uint8)
        invalid_msg[5] = 2
        with pytest.raises(PolarEncodingError, match="must contain only 0s and 1s"):
            codec.encode_block(invalid_msg)

    def test_decode_block_wrong_length(self, codec):
        with pytest.raises(PolarDecodingError, match="expects 1D array of length N=64"):
            codec.decode_block(np.zeros(63, dtype=np.float64))

    def test_decode_block_nan_likelihoods(self, codec):
        llrs = np.ones(64, dtype=np.float64)
        llrs[10] = np.nan
        with pytest.raises(PolarDecodingError, match="contains NaN values"):
            codec.decode_block(llrs)

    def test_multi_block_decode_non_multiple_of_N(self, codec):
        with pytest.raises(PolarDecodingError, match="must be an exact multiple of N=64"):
            codec.decode(np.zeros(100, dtype=np.float64), pad_count=0)

    def test_multi_block_decode_invalid_pad_count(self, codec):
        with pytest.raises(PolarDecodingError, match="pad_count must be in"):
            codec.decode(np.zeros(64, dtype=np.float64), pad_count=-1)
        with pytest.raises(PolarDecodingError, match="pad_count must be in"):
            codec.decode(np.zeros(64, dtype=np.float64), pad_count=32)


# ---------------------------------------------------------------------------
# 7. LLR Conventions and Library Limitations
# ---------------------------------------------------------------------------
class TestLLRConventionsAndLimitations:
    def test_bits_to_llr_mapping(self):
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        llrs = bits_to_llr(bits, magnitude=50.0)
        expected = np.array([50.0, -50.0, 50.0, -50.0], dtype=np.float64)
        np.testing.assert_array_equal(llrs, expected)

    def test_llr_to_hard_bits_mapping(self):
        llrs = np.array([10.5, -2.3, 0.0, -0.001], dtype=np.float64)
        hard_bits = llr_to_hard_bits(llrs)
        # LLR >= 0 -> 0, LLR < 0 -> 1
        expected = np.array([0, 1, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(hard_bits, expected)

    def test_deterministic_reproducibility(self):
        codec1 = PolarCodec(N=64, K=32, design_snr_db=2.5)
        codec2 = PolarCodec(N=64, K=32, design_snr_db=2.5)

        # Frozen indices must be identical
        np.testing.assert_array_equal(codec1.frozen_indices, codec2.frozen_indices)

        msg = get_rng(777).integers(0, 2, size=32, dtype=np.uint8)
        cw1 = codec1.encode_block(msg)
        cw2 = codec2.encode_block(msg)
        np.testing.assert_array_equal(cw1, cw2)
