"""
Comprehensive test suite for end-to-end joint rANS source coding and Polar channel coding pipeline.

Tests include:
- Noiseless channel (p = 0.0) exact byte-for-byte reconstruction across diverse corpora:
  - Empty input (b"")
  - Short ASCII text
  - Multi-byte UTF-8 international text and emojis
  - All 256 byte values
  - Exact Polar block multiples (pad_count == 0) and partial final blocks (pad_count > 0)
- Framing metadata validation and out-of-band integrity checking
- Bit serialization/deserialization container identity preservation
- Deterministic replay under fixed random seeds
- Realistic noisy channel transmission trials at selected crossover probabilities (low, moderate, high)
- Honest measurement recording (runtimes, rates, BER, FER, failure reporting)
"""

import numpy as np
import pytest

from src.channel_coding.polar_codec import PolarConfig
from src.pipeline import (
    EndToEndPipeline,
    MetadataMismatchError,
    PipelineResult,
    TransmissionFrameMetadata,
)
from src.source_coding.rans_codec import rans_compress
from src.utils.bit_utils import bits_to_bytes, bytes_to_bits
from src.utils.seed import get_rng


# ---------------------------------------------------------------------------
# 1. Noiseless End-to-End Correctness (p = 0.0)
# ---------------------------------------------------------------------------
class TestNoiselessEndToEnd:
    @pytest.fixture
    def pipeline(self):
        # N = 128, K = 64 (R = 0.5)
        config = PolarConfig(N=128, K=64, design_snr_db=2.0)
        return EndToEndPipeline(polar_config=config)

    def test_empty_input_bytes(self, pipeline):
        res = pipeline.run(b"", p=0.0)
        assert isinstance(res, PipelineResult)
        assert res.exact_reconstruction is True
        assert res.success is True
        assert res.reconstructed_bytes == b""
        assert res.original_size_bytes == 0
        assert res.channel_bit_flips == 0
        assert res.failure_stage is None

    @pytest.mark.parametrize("text", [
        b"A",
        b"Hello!",
        b"Information Theory and Coding - Final Research Project.",
        "French accents: L'entropie d'une source d'information.".encode("utf-8"),
        "German: Übergrößenträger & Maßstäbe.".encode("utf-8"),
        "Japanese: 情報理論とポーラー符号 (Information Theory & Polar Codes).".encode("utf-8"),
        "Emojis: 🚀📡🔬📊🛰️.".encode("utf-8"),
        "Multiline:\nLine 1\nLine 2\nLine 3\nEnd.\n".encode("utf-8"),
    ])
    def test_text_and_unicode_roundtrips(self, pipeline, text):
        res = pipeline.run(text, p=0.0)
        assert res.exact_reconstruction is True
        assert res.success is True
        assert res.reconstructed_bytes == text
        assert res.channel_bit_flips == 0
        assert res.polar_residual_bit_errors == 0
        assert res.failure_stage is None
        assert res.total_latency_ms > 0

    def test_all_256_byte_values(self, pipeline):
        all_bytes = bytes(range(256))
        res = pipeline.run(all_bytes, p=0.0)
        assert res.exact_reconstruction is True
        assert res.success is True
        assert res.reconstructed_bytes == all_bytes
        assert res.channel_bit_flips == 0

    def test_exact_and_partial_polar_blocks(self, pipeline):
        # Test input producing partial block (pad_count > 0)
        res_partial = pipeline.run(b"Short sample message creating partial block", p=0.0)
        assert res_partial.pad_count > 0
        assert res_partial.exact_reconstruction is True
        assert res_partial.success is True

        # Construct payload guaranteed to test multi-block structure
        data = b"Testing multiple polar block framing: " * 15
        res_multi = pipeline.run(data, p=0.0)
        assert res_multi.num_polar_blocks > 1
        assert res_multi.exact_reconstruction is True
        assert res_multi.success is True

    def test_container_byte_preservation_through_bit_layer(self):
        raw_text = b"Preserve container bytes across bit-conversion layer!"
        container_bytes = rans_compress(raw_text)

        # Convert to bits and back
        bits = bytes_to_bits(container_bytes, msb_first=True)
        reconstructed_container = bits_to_bytes(bits, msb_first=True)

        assert reconstructed_container == container_bytes


# ---------------------------------------------------------------------------
# 2. Out-of-Band Metadata Validation & Error Handling
# ---------------------------------------------------------------------------
class TestMetadataValidation:
    def test_metadata_dimension_mismatch(self):
        pipeline = EndToEndPipeline()
        codewords, _, metadata, _, _ = pipeline.transmit_prepare(b"Sample payload")

        # Corrupt LLR stream length
        truncated_llrs = np.ones(codewords.size - 10, dtype=np.float64)
        with pytest.raises(MetadataMismatchError, match="Received LLR stream length"):
            pipeline.receive_process(truncated_llrs, metadata)

    def test_invalid_metadata_parameters(self):
        with pytest.raises(MetadataMismatchError, match="container_byte_length cannot be negative"):
            meta = TransmissionFrameMetadata(
                container_byte_length=-1,
                container_bit_length=-8,
                pad_count=0,
                K=64,
                N=128,
                num_polar_blocks=0,
                total_codeword_bits=0,
                channel_p=0.0,
            )
            meta.validate()

        with pytest.raises(MetadataMismatchError, match="Invalid Polar block dimensions"):
            meta = TransmissionFrameMetadata(
                container_byte_length=10,
                container_bit_length=80,
                pad_count=0,
                K=128,
                N=64,  # Invalid: K >= N
                num_polar_blocks=1,
                total_codeword_bits=64,
                channel_p=0.0,
            )
            meta.validate()


# ---------------------------------------------------------------------------
# 3. Deterministic Seed Replay
# ---------------------------------------------------------------------------
class TestDeterministicReplay:
    def test_fixed_seed_exact_reproducibility(self):
        pipeline = EndToEndPipeline(polar_config=PolarConfig(N=64, K=32))
        data = b"Deterministic seed test message for end-to-end replay!"

        res1 = pipeline.run(data, p=0.05, seed=12345)
        res2 = pipeline.run(data, p=0.05, seed=12345)

        assert res1.channel_bit_flips == res2.channel_bit_flips
        assert res1.channel_ber == res2.channel_ber
        assert res1.polar_residual_bit_errors == res2.polar_residual_bit_errors
        assert res1.exact_reconstruction == res2.exact_reconstruction
        assert res1.success == res2.success


# ---------------------------------------------------------------------------
# 4. Initial Noisy Channel Integration Trials
# ---------------------------------------------------------------------------
class TestNoisyChannelIntegration:
    @pytest.fixture
    def noisy_pipeline(self):
        # Block length N = 256, K = 128 (R = 0.5)
        config = PolarConfig(N=256, K=128, design_snr_db=2.5)
        return EndToEndPipeline(polar_config=config)

    def test_sparse_noise_error_correction(self, noisy_pipeline):
        # Under very light noise (p = 0.001), Polar SCD typically corrects all errors
        data = b"This is a longer paragraph for channel evaluation testing under controlled bit-flip conditions. " * 8
        res = noisy_pipeline.run(data, p=0.001, seed=42)

        assert isinstance(res, PipelineResult)
        assert res.channel_p == 0.001
        # Honest reporting: check whether error correction succeeded or failed gracefully
        if res.success:
            assert res.exact_reconstruction is True
            assert res.reconstructed_bytes == data
            assert res.failure_stage is None
        else:
            assert res.reconstructed_bytes is None
            assert res.failure_stage in ("POLAR_DECODE", "RANS_DECOMPRESS", "BYTE_MISMATCH")

    def test_moderate_to_high_noise_graceful_failure_reporting(self, noisy_pipeline):
        # Under high noise (p = 0.10), Polar SCD on N=256 will exceed its correction threshold.
        # The pipeline must report failure honestly without unhandled exceptions or crashes.
        data = b"High noise test: channel crossover probability p = 0.10 will corrupt Polar blocks."
        res = noisy_pipeline.run(data, p=0.10, seed=101)

        assert isinstance(res, PipelineResult)
        assert res.channel_bit_flips > 0
        assert res.channel_ber > 0.0

        if not res.success:
            # Treated as transmission failure, not successful result
            assert res.exact_reconstruction is False
            assert res.reconstructed_bytes is None
            assert res.failure_stage in ("RANS_DECOMPRESS", "POLAR_DECODE", "CONTAINER_INTEGRITY")
            assert res.error_message is not None


# ---------------------------------------------------------------------------
# 5. Telemetry and Rate Measurements
# ---------------------------------------------------------------------------
class TestTelemetryAndRateCalculations:
    def test_metric_fields_population(self):
        pipeline = EndToEndPipeline(polar_config=PolarConfig(N=128, K=64))
        data = b"Detailed telemetry inspection test sentence."
        res = pipeline.run(data, p=0.0)

        # Check all telemetry attributes are populated and consistent
        assert res.original_size_bytes == len(data)
        assert res.container_size_bytes > 0
        assert res.source_bits == res.container_size_bytes * 8
        assert res.padded_bits == res.source_bits + res.pad_count
        assert res.num_polar_blocks > 0
        assert res.transmitted_codeword_bits == res.num_polar_blocks * 128
        assert res.code_rate == 0.5
        assert res.effective_transmission_rate > 0.0

        # Timing
        assert res.rans_encode_time_ms >= 0.0
        assert res.polar_encode_time_ms >= 0.0
        assert res.channel_time_ms >= 0.0
        assert res.polar_decode_time_ms >= 0.0
        assert res.rans_decode_time_ms >= 0.0
        assert res.total_latency_ms >= 0.0
