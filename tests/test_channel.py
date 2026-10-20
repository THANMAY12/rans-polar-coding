"""
Comprehensive test suite for Wired Binary Symmetric Channel (BSC) simulation model.

Tests include:
- Ideal noiseless mode (p = 0): zero bit flips, exact identity, noiseless LLRs
- Deterministic random seed reproducibility and generator injection
- Different seeds behavior
- Empirical bit error rate statistical validation (within confidence interval)
- Boundary conditions (p = 0.0, p = 0.5) and rejection of invalid probabilities (p < 0, p > 0.5)
- LLR mathematical derivation, signs, and magnitude consistency
- Input dimension and value validations (empty, 2D, non-binary)
- Guaranteed in-order delivery and zero packet loss
"""

import math
import numpy as np
import pytest

from src.channel.binary_symmetric_channel import (
    BinarySymmetricChannel,
    ChannelResult,
    InvalidChannelInputError,
    InvalidChannelParameterError,
    WiredChannel,
)
from src.channel_coding.polar_codec import llr_to_hard_bits
from src.utils.seed import get_rng


# ---------------------------------------------------------------------------
# 1. Ideal Channel Mode (p = 0.0)
# ---------------------------------------------------------------------------
class TestIdealChannel:
    @pytest.fixture
    def ideal_channel(self):
        return BinarySymmetricChannel(p=0.0, noiseless_llr=100.0)

    def test_ideal_property(self, ideal_channel):
        assert ideal_channel.is_ideal is True
        assert ideal_channel.p == 0.0
        assert ideal_channel.noiseless_llr == 100.0

    @pytest.mark.parametrize("input_pattern", [
        np.zeros(64, dtype=np.uint8),
        np.ones(64, dtype=np.uint8),
        np.array([0, 1] * 32, dtype=np.uint8),
        np.array([1, 1, 0, 0, 1, 0, 1, 0] * 8, dtype=np.uint8),
    ])
    def test_zero_bit_flips_on_deterministic_patterns(self, ideal_channel, input_pattern):
        res = ideal_channel.transmit(input_pattern)
        assert isinstance(res, ChannelResult)
        assert res.error_count == 0
        assert res.empirical_ber == 0.0
        assert res.total_bits == input_pattern.size
        assert not np.any(res.flip_mask)
        np.testing.assert_array_equal(res.received_bits, input_pattern)

    def test_noiseless_llr_values(self, ideal_channel):
        bits = np.array([0, 1, 0, 0, 1], dtype=np.uint8)
        res = ideal_channel.transmit(bits)

        expected_llrs = np.array([100.0, -100.0, 100.0, 100.0, -100.0], dtype=np.float64)
        np.testing.assert_array_equal(res.llrs, expected_llrs)

        # LLR hard decision slices back to received bits
        hard_decisions = llr_to_hard_bits(res.llrs)
        np.testing.assert_array_equal(hard_decisions, bits)


# ---------------------------------------------------------------------------
# 2. Random Seed Reproducibility and Determinism
# ---------------------------------------------------------------------------
class TestSeedDeterminism:
    def test_identical_seed_produces_identical_transmissions(self):
        rng = get_rng(42)
        test_bits = rng.integers(0, 2, size=500, dtype=np.uint8)

        ch1 = BinarySymmetricChannel(p=0.1, seed=12345)
        ch2 = BinarySymmetricChannel(p=0.1, seed=12345)

        r1 = ch1.transmit(test_bits)
        r2 = ch2.transmit(test_bits)

        np.testing.assert_array_equal(r1.received_bits, r2.received_bits)
        np.testing.assert_array_equal(r1.flip_mask, r2.flip_mask)
        assert r1.error_count == r2.error_count
        np.testing.assert_array_equal(r1.llrs, r2.llrs)

    def test_reset_seed_restores_identical_sequence(self):
        test_bits = np.array([0, 1] * 200, dtype=np.uint8)
        channel = BinarySymmetricChannel(p=0.15, seed=999)

        r1 = channel.transmit(test_bits)
        channel.reset_seed(999)
        r2 = channel.transmit(test_bits)

        np.testing.assert_array_equal(r1.received_bits, r2.received_bits)
        np.testing.assert_array_equal(r1.flip_mask, r2.flip_mask)

    def test_custom_rng_injection(self):
        custom_rng = np.random.default_rng(777)
        channel = BinarySymmetricChannel(p=0.2, rng=custom_rng)
        assert channel.rng is custom_rng

        bits = np.zeros(100, dtype=np.uint8)
        res = channel.transmit(bits)
        assert res.total_bits == 100

    def test_different_seeds_produce_valid_noisy_simulations(self):
        rng = get_rng(100)
        test_bits = rng.integers(0, 2, size=2000, dtype=np.uint8)

        ch1 = BinarySymmetricChannel(p=0.1, seed=10)
        ch2 = BinarySymmetricChannel(p=0.1, seed=20)

        r1 = ch1.transmit(test_bits)
        r2 = ch2.transmit(test_bits)

        # Both simulations must have reasonable error counts
        assert 100 < r1.error_count < 300
        assert 100 < r2.error_count < 300
        # Flips should not be 100% identical
        assert not np.array_equal(r1.flip_mask, r2.flip_mask)


# ---------------------------------------------------------------------------
# 3. Empirical Error Rate Statistical Validation
# ---------------------------------------------------------------------------
class TestEmpiricalErrorRate:
    @pytest.mark.parametrize("p", [0.01, 0.05, 0.10, 0.25])
    def test_ber_convergence_within_confidence_bounds(self, p):
        # Sample size N = 100,000 bits
        N = 100_000
        channel = BinarySymmetricChannel(p=p, seed=2026)

        tx_bits = np.zeros(N, dtype=np.uint8)
        res = channel.transmit(tx_bits)

        # Expected flips: mu = N * p
        # Standard deviation: sigma = sqrt(N * p * (1 - p))
        mu = N * p
        sigma = math.sqrt(N * p * (1.0 - p))

        # Check empirical flip count falls within 4 standard deviations (p < 0.0001 failure rate)
        diff = abs(res.error_count - mu)
        assert diff <= 4.0 * sigma, f"Observed {res.error_count} flips, expected ~{mu} (diff={diff}, 4*sigma={4*sigma:.1f})"

        # Verify empirical BER calculation
        assert res.empirical_ber == pytest.approx(res.error_count / N)
        assert res.empirical_ber == pytest.approx(p, rel=0.10)


# ---------------------------------------------------------------------------
# 4. Boundary Cases and Parameter Validation
# ---------------------------------------------------------------------------
class TestBoundaryAndValidation:
    def test_p_half_maximum_uncertainty(self):
        # When p = 0.5, mutual information is 0 and LLR = ln(0.5 / 0.5) = 0.0
        channel = BinarySymmetricChannel(p=0.5, seed=42)
        bits = np.array([0, 1, 0, 1] * 25, dtype=np.uint8)

        res = channel.transmit(bits)
        assert res.p == 0.5
        # All LLRs must strictly be 0.0
        np.testing.assert_array_equal(res.llrs, np.zeros(bits.size, dtype=np.float64))

    @pytest.mark.parametrize("invalid_p", [-0.001, -0.5, 0.5001, 0.75, 1.0, 2.0, float("nan")])
    def test_invalid_probability_rejected(self, invalid_p):
        with pytest.raises(InvalidChannelParameterError, match="must be in the range \\[0.0, 0.5\\]|cannot be NaN"):
            BinarySymmetricChannel(p=invalid_p)

    def test_invalid_p_type(self):
        with pytest.raises(InvalidChannelParameterError, match="must be numeric"):
            BinarySymmetricChannel(p="not a float")  # type: ignore

    @pytest.mark.parametrize("invalid_llr", [0.0, -10.0, -0.001])
    def test_invalid_noiseless_llr(self, invalid_llr):
        with pytest.raises(InvalidChannelParameterError, match="noiseless_llr must be strictly positive"):
            BinarySymmetricChannel(p=0.0, noiseless_llr=invalid_llr)

    def test_invalid_rng_type(self):
        with pytest.raises(TypeError, match="must be a numpy.random.Generator"):
            BinarySymmetricChannel(p=0.1, rng="invalid_rng")  # type: ignore


# ---------------------------------------------------------------------------
# 5. Input Validation and Dimension Handling
# ---------------------------------------------------------------------------
class TestInputValidation:
    @pytest.fixture
    def channel(self):
        return BinarySymmetricChannel(p=0.05)

    def test_empty_input_transmission(self, channel):
        empty = np.empty(0, dtype=np.uint8)
        res = channel.transmit(empty)

        assert res.total_bits == 0
        assert res.error_count == 0
        assert res.empirical_ber == 0.0
        assert res.transmitted_bits.size == 0
        assert res.received_bits.size == 0
        assert res.llrs.size == 0

    def test_multidimensional_input_rejected(self, channel):
        with pytest.raises(InvalidChannelInputError, match="must be 1D"):
            channel.transmit(np.zeros((2, 10), dtype=np.uint8))

    def test_non_binary_values_rejected(self, channel):
        with pytest.raises(InvalidChannelInputError, match="must contain only binary values 0 and 1"):
            channel.transmit(np.array([0, 1, 2, 0], dtype=np.uint8))
        with pytest.raises(InvalidChannelInputError, match="must contain only binary values 0 and 1"):
            channel.transmit(np.array([0, -1, 1], dtype=np.int8))

    def test_compute_llr_validation(self, channel):
        with pytest.raises(InvalidChannelInputError, match="must be 1D"):
            channel.compute_llr(np.zeros((3, 3), dtype=np.uint8))
        with pytest.raises(InvalidChannelInputError, match="must contain only binary values 0 and 1"):
            channel.compute_llr(np.array([0, 3, 1]))


# ---------------------------------------------------------------------------
# 6. LLR Theoretical Derivation and Sign Conventions
# ---------------------------------------------------------------------------
class TestLLRDerivationAndConventions:
    @pytest.mark.parametrize("p, expected_weight", [
        (0.01, math.log(0.99 / 0.01)),   # ln(99) ≈ 4.5951
        (0.10, math.log(0.90 / 0.10)),   # ln(9)  ≈ 2.1972
        (0.20, math.log(0.80 / 0.20)),   # ln(4)  ≈ 1.3863
        (0.33, math.log(0.67 / 0.33)),   # ln(67/33) ≈ 0.7082
    ])
    def test_exact_bsc_llr_formula(self, p, expected_weight):
        channel = BinarySymmetricChannel(p=p)
        bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        llrs = channel.compute_llr(bits)

        expected = np.array([
            expected_weight,
            -expected_weight,
            expected_weight,
            -expected_weight,
            -expected_weight,
            expected_weight,
        ], dtype=np.float64)

        np.testing.assert_allclose(llrs, expected, rtol=1e-7)

    def test_llr_sign_convention_compatibility_with_polar(self):
        channel = BinarySymmetricChannel(p=0.08, seed=42)
        test_bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

        res = channel.transmit(test_bits)

        # Confirm: positive LLR -> bit 0, negative LLR -> bit 1
        for rx_bit, llr in zip(res.received_bits, res.llrs):
            if rx_bit == 0:
                assert llr > 0.0
            else:
                assert llr < 0.0

        # Confirm hard decision directly recovers received_bits
        sliced_bits = llr_to_hard_bits(res.llrs)
        np.testing.assert_array_equal(sliced_bits, res.received_bits)


# ---------------------------------------------------------------------------
# 7. No Packet Loss & In-Order Delivery Invariant
# ---------------------------------------------------------------------------
class TestNoPacketLossInvariant:
    def test_exact_sequence_length_preservation(self):
        channel = BinarySymmetricChannel(p=0.1, seed=123)

        for size in [1, 7, 64, 128, 513, 2048]:
            tx = np.zeros(size, dtype=np.uint8)
            res = channel.transmit(tx)
            assert res.received_bits.size == size
            assert res.llrs.size == size
            assert res.flip_mask.size == size

    def test_semantic_alias(self):
        assert WiredChannel is BinarySymmetricChannel
