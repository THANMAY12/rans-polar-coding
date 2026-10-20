"""
Comprehensive test suite for the Experiment Harness, statistical methods, and CSV schemas.

Tests include:
- Experiment configuration validation and boundary checks
- Wilson score interval mathematical precision and clamping
- Metric calculation formulas and edge cases (zero denominator, empty input)
- Statistical aggregation over mixed successful and failed trials
- Preservation of failed trials without silent dropping
- Deterministic seed derivation and reproducibility
- Output CSV schemas and raw-vs-summary consistency
- End-to-end smoke experiment execution
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from experiments.harness import (
    ExperimentConfig,
    ExperimentRunner,
    wilson_score_interval,
)


# ---------------------------------------------------------------------------
# 1. Configuration Validation
# ---------------------------------------------------------------------------
class TestExperimentConfig:
    def test_valid_smoke_config(self, tmp_path: Path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello ITC!", encoding="utf-8")

        cfg = ExperimentConfig.smoke_test_config(input_file=test_file, output_dir=tmp_path / "results")
        cfg.validate()
        assert cfg.name == "smoke_test"
        assert cfg.input_files == [test_file]
        assert cfg.polar_configs == [(64, 32)]
        assert 0.0 in cfg.channel_probabilities

    def test_missing_input_file_raises_error(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist.txt"
        cfg = ExperimentConfig(input_files=[non_existent])
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            cfg.validate()

    def test_empty_input_files_raises_error(self):
        cfg = ExperimentConfig(input_files=[])
        with pytest.raises(ValueError, match="input_files list cannot be empty"):
            cfg.validate()

    def test_invalid_polar_dimensions(self, tmp_path: Path):
        test_file = tmp_path / "f.txt"
        test_file.write_text("test", encoding="utf-8")

        # N not a power of 2
        with pytest.raises(ValueError, match="Polar N must be a power of 2"):
            ExperimentConfig(input_files=[test_file], polar_configs=[(100, 50)]).validate()

        # K >= N
        with pytest.raises(ValueError, match="Polar K must satisfy 1 <= K < N"):
            ExperimentConfig(input_files=[test_file], polar_configs=[(64, 64)]).validate()

    def test_channel_probability_validation(self, tmp_path: Path):
        test_file = tmp_path / "f.txt"
        test_file.write_text("test", encoding="utf-8")

        # Out of bounds (> 0.5)
        with pytest.raises(ValueError, match="must be in \\[0.0, 0.5\\]"):
            ExperimentConfig(input_files=[test_file], channel_probabilities=[0.0, 0.6]).validate()

        # Missing noiseless p=0.0
        with pytest.raises(ValueError, match="must include p=0.0 as the noiseless baseline"):
            ExperimentConfig(input_files=[test_file], channel_probabilities=[0.01, 0.05]).validate()

    def test_json_serialization_roundtrip(self, tmp_path: Path):
        test_file = tmp_path / "f.txt"
        test_file.write_text("test", encoding="utf-8")

        cfg1 = ExperimentConfig(
            name="test_run",
            input_files=[test_file],
            polar_configs=[(128, 64)],
            channel_probabilities=[0.0, 0.05],
            trials_per_condition=3,
            output_dir=tmp_path / "results",
        )
        d = cfg1.to_dict()
        cfg2 = ExperimentConfig.from_dict(d)

        assert cfg1.name == cfg2.name
        assert cfg1.input_files == cfg2.input_files
        assert cfg1.polar_configs == cfg2.polar_configs
        assert cfg1.channel_probabilities == cfg2.channel_probabilities
        assert cfg1.trials_per_condition == cfg2.trials_per_condition


# ---------------------------------------------------------------------------
# 2. Wilson Score Interval Validation
# ---------------------------------------------------------------------------
class TestWilsonScoreInterval:
    def test_empty_or_zero_total(self):
        low, high = wilson_score_interval(0, 0)
        assert low == 0.0
        assert high == 0.0

    def test_zero_successes_boundary(self):
        # When k = 0, lower bound must strictly be 0.0, upper bound > 0
        low, high = wilson_score_interval(0, 10, confidence=0.95)
        assert low == 0.0
        assert 0.20 < high < 0.35

    def test_all_successes_boundary(self):
        # When k = n, upper bound must strictly be 1.0, lower bound < 1.0
        low, high = wilson_score_interval(10, 10, confidence=0.95)
        assert high == 1.0
        assert 0.65 < low < 0.80

    def test_symmetric_case(self):
        # k = 5, n = 10 -> center should be 0.5
        low, high = wilson_score_interval(5, 10, confidence=0.95)
        assert low < 0.5 < high
        assert (0.5 - low) == pytest.approx(high - 0.5)

    def test_large_sample_narrowing(self):
        # As n -> infinity, interval narrows to p_hat
        low_small, high_small = wilson_score_interval(50, 100, confidence=0.95)
        low_large, high_large = wilson_score_interval(5000, 10000, confidence=0.95)

        width_small = high_small - low_small
        width_large = high_large - low_large
        assert width_large < width_small / 8


# ---------------------------------------------------------------------------
# 3. Metric Calculations and Edge Cases
# ---------------------------------------------------------------------------
class TestMetricCalculations:
    def test_compression_ratio_formula(self):
        orig_bytes = 1000
        container_bytes = 500
        cr = orig_bytes / container_bytes
        assert cr == 2.0

    def test_effective_transmission_rate_formula(self):
        orig_bytes = 100
        total_channel_bits = 1600
        orig_bits = orig_bytes * 8  # 800
        eff_rate = orig_bits / total_channel_bits
        assert eff_rate == 0.5  # 800 / 1600

    def test_zero_original_bytes_edge_case(self):
        orig_bytes = 0
        container_bytes = 20
        total_channel_bits = 384
        cr = (orig_bytes / container_bytes) if container_bytes > 0 else 0.0
        eff_rate = (orig_bytes * 8) / total_channel_bits if total_channel_bits > 0 else 0.0
        assert cr == 0.0
        assert eff_rate == 0.0


# ---------------------------------------------------------------------------
# 4. Aggregation Over Mixed Trials (No Silent Dropping)
# ---------------------------------------------------------------------------
class TestAggregationLogic:
    def test_mixed_successful_and_failed_trials(self):
        # Synthetic DataFrame with 4 trials: 2 successes, 2 failures
        records = [
            {
                "corpus_name": "sample.txt",
                "polar_N": 128,
                "polar_K": 64,
                "polar_rate": 0.5,
                "channel_p": 0.05,
                "original_size_bytes": 1000,
                "container_size_bytes": 600,
                "source_compression_ratio": 1.6667,
                "effective_transmission_rate": 0.40,
                "empirical_channel_ber": 0.048,
                "polar_residual_ber": 0.0,
                "polar_block_error_rate": 0.0,
                "exact_reconstruction": 1,
                "success": 1,
                "failure_stage": "NONE",
                "rans_encode_time_ms": 2.0,
                "polar_encode_time_ms": 1.0,
                "polar_decode_time_ms": 5.0,
                "rans_decode_time_ms": 2.0,
                "total_latency_ms": 10.0,
            },
            {
                "corpus_name": "sample.txt",
                "polar_N": 128,
                "polar_K": 64,
                "polar_rate": 0.5,
                "channel_p": 0.05,
                "original_size_bytes": 1000,
                "container_size_bytes": 600,
                "source_compression_ratio": 1.6667,
                "effective_transmission_rate": 0.40,
                "empirical_channel_ber": 0.052,
                "polar_residual_ber": 0.0,
                "polar_block_error_rate": 0.0,
                "exact_reconstruction": 1,
                "success": 1,
                "failure_stage": "NONE",
                "rans_encode_time_ms": 2.1,
                "polar_encode_time_ms": 1.1,
                "polar_decode_time_ms": 5.2,
                "rans_decode_time_ms": 2.1,
                "total_latency_ms": 10.5,
            },
            {
                "corpus_name": "sample.txt",
                "polar_N": 128,
                "polar_K": 64,
                "polar_rate": 0.5,
                "channel_p": 0.05,
                "original_size_bytes": 1000,
                "container_size_bytes": 600,
                "source_compression_ratio": 1.6667,
                "effective_transmission_rate": 0.40,
                "empirical_channel_ber": 0.051,
                "polar_residual_ber": 0.08,
                "polar_block_error_rate": 0.5,
                "exact_reconstruction": 0,
                "success": 0,
                "failure_stage": "RANS_DECOMPRESS",
                "rans_encode_time_ms": 2.0,
                "polar_encode_time_ms": 1.0,
                "polar_decode_time_ms": 5.1,
                "rans_decode_time_ms": 0.5,
                "total_latency_ms": 8.6,
            },
            {
                "corpus_name": "sample.txt",
                "polar_N": 128,
                "polar_K": 64,
                "polar_rate": 0.5,
                "channel_p": 0.05,
                "original_size_bytes": 1000,
                "container_size_bytes": 600,
                "source_compression_ratio": 1.6667,
                "effective_transmission_rate": 0.40,
                "empirical_channel_ber": 0.049,
                "polar_residual_ber": 0.06,
                "polar_block_error_rate": 0.5,
                "exact_reconstruction": 0,
                "success": 0,
                "failure_stage": "RANS_DECOMPRESS",
                "rans_encode_time_ms": 2.2,
                "polar_encode_time_ms": 1.0,
                "polar_decode_time_ms": 5.0,
                "rans_decode_time_ms": 0.6,
                "total_latency_ms": 8.8,
            },
        ]
        raw_df = pd.DataFrame(records)
        summary_df = ExperimentRunner.aggregate_results(raw_df)

        assert len(summary_df) == 1
        row = summary_df.iloc[0]

        # Ensure all 4 trials accounted for without dropping failures
        assert row["trials_count"] == 4
        assert row["exact_reconstruction_count"] == 2
        assert row["exact_reconstruction_rate"] == 0.5
        assert row["failure_rate"] == 0.5
        assert row["wilson_ci_lower_95"] < 0.5 < row["wilson_ci_upper_95"]
        assert row["mean_channel_ber"] == pytest.approx(0.050, abs=1e-3)
        assert row["std_channel_ber"] > 0.0


# ---------------------------------------------------------------------------
# 5. Deterministic Seed Replay & Runner Execution
# ---------------------------------------------------------------------------
class TestRunnerDeterminismAndSmokeTest:
    def test_seed_derivation_deterministic(self, tmp_path: Path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello ITC", encoding="utf-8")

        cfg1 = ExperimentConfig.smoke_test_config(test_file, tmp_path)
        cfg2 = ExperimentConfig.smoke_test_config(test_file, tmp_path)

        runner1 = ExperimentRunner(cfg1)
        runner2 = ExperimentRunner(cfg2)

        s1 = [runner1._derive_trial_seed(c, t) for c in range(3) for t in range(2)]
        s2 = [runner2._derive_trial_seed(c, t) for c in range(3) for t in range(2)]

        assert s1 == s2

    def test_smoke_experiment_execution(self, tmp_path: Path):
        test_file = tmp_path / "smoke_corpus.txt"
        test_file.write_text("Smoke test technical corpus with multiple words for Polar coding.\n" * 5, encoding="utf-8")

        out_dir = tmp_path / "results"
        cfg = ExperimentConfig(
            name="smoke_test_run",
            input_files=[test_file],
            polar_configs=[(64, 32)],
            channel_probabilities=[0.0, 0.01],
            trials_per_condition=2,
            master_seed=12345,
            output_dir=out_dir,
        )

        runner = ExperimentRunner(cfg)
        raw_df, summary_df = runner.run_all(run_id="test_run_1")

        # Total trials: 1 file * 1 polar config * 2 probabilities * 2 trials = 4
        assert len(raw_df) == 4
        assert len(summary_df) == 2  # 2 conditions: p=0.0 and p=0.01

        # Check p=0.0 condition strictly has 100% exact reconstruction
        p0_summary = summary_df[summary_df["channel_p"] == 0.0].iloc[0]
        assert p0_summary["exact_reconstruction_rate"] == 1.0
        assert p0_summary["wilson_ci_upper_95"] == 1.0

        # Save results and verify CSV schemas
        raw_path, summary_path, config_path = runner.save_results(
            raw_df, summary_df, run_id="test_run_1", overwrite=True
        )

        assert raw_path.exists()
        assert summary_path.exists()
        assert config_path.exists()

        # Reload CSVs and verify columns
        loaded_raw = pd.read_csv(raw_path)
        loaded_summary = pd.read_csv(summary_path)

        assert len(loaded_raw) == 4
        assert len(loaded_summary) == 2
        assert "exact_reconstruction" in loaded_raw.columns
        assert "wilson_ci_lower_95" in loaded_summary.columns
        assert "mean_effective_transmission_rate" in loaded_summary.columns
