"""
Unit and Integration Tests for Experimental Analysis and Plotting Modules.

Tests:
1. Information-theoretic formulas:
   - Zero-order Shannon entropy H0(X).
   - Theoretical Shannon compression ratio bound 8.0 / H0(X).
   - BSC Shannon channel capacity C(p) = 1 - H2(p).
2. Statistical intervals (Wilson score confidence interval).
3. Metric calculations:
   - Compression ratio with container metadata.
   - Effective transmission rate.
   - Pre-FEC BER and Post-FEC residual BER.
   - Polar Block Frame Error Rate (FER).
4. Edge cases:
   - Empty input bytes, zero denominators, missing files.
   - Perfect success (k = n) and total failure (k = 0).
5. Analysis tables generation and schema consistency.
6. Headless figure generation (all 6 publication figures).
"""

import math
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from experiments.analyze_results import (
    analyze_channel_reliability,
    analyze_corpus_compression,
    analyze_failure_diagnostics,
    analyze_polar_configuration_tradeoffs,
    bsc_channel_capacity,
    compute_ideal_compression_ratio,
    compute_zero_order_entropy,
    find_results_files,
    load_latest_dataset,
    load_raw_dataset,
    run_full_analysis,
)
from experiments.harness import wilson_score_interval
from experiments.plot_results import (
    generate_all_figures,
    plot_compression_ratio_by_corpus,
    plot_effective_rate_vs_code_rate,
    plot_pre_fec_ber_vs_p,
    plot_reconstruction_rate_vs_p,
    plot_residual_ber_and_fer_vs_p,
    plot_runtime_breakdown,
)


# ---------------------------------------------------------------------------
# 1. Information-Theoretic Reference Tests
# ---------------------------------------------------------------------------
class TestInformationTheoreticFormulas:
    """Validate mathematical properties of entropy and channel capacity."""

    def test_entropy_empty_input(self):
        assert compute_zero_order_entropy(b"") == 0.0

    def test_entropy_single_repeated_symbol(self):
        # A deterministic sequence has zero entropy
        data = b"AAAAAAAAAA"
        assert compute_zero_order_entropy(data) == 0.0

    def test_entropy_binary_uniform(self):
        # 50% 0, 50% 1 -> 1.0 bit/byte
        data = bytes([0, 1] * 500)
        assert pytest.approx(compute_zero_order_entropy(data), abs=1e-5) == 1.0

    def test_entropy_256_uniform(self):
        # All 256 bytes equiprobable -> 8.0 bits/byte
        data = bytes(range(256)) * 10
        assert pytest.approx(compute_zero_order_entropy(data), abs=1e-5) == 8.0

    def test_ideal_compression_ratio(self):
        assert compute_ideal_compression_ratio(0.0) == 0.0
        assert compute_ideal_compression_ratio(8.0) == 1.0
        assert pytest.approx(compute_ideal_compression_ratio(4.0), abs=1e-5) == 2.0
        assert pytest.approx(compute_ideal_compression_ratio(2.0), abs=1e-5) == 4.0

    def test_bsc_capacity_boundary_values(self):
        # Ideal channel p = 0 -> C = 1.0
        assert bsc_channel_capacity(0.0) == 1.0
        # Completely noisy channel p = 0.5 -> C = 0.0
        assert bsc_channel_capacity(0.5) == 0.0
        # Out-of-bounds safety
        assert bsc_channel_capacity(-0.1) == 1.0
        assert bsc_channel_capacity(1.0) == 0.0

    def test_bsc_capacity_known_values(self):
        # At p = 0.01, H2(0.01) ≈ 0.08079 -> C ≈ 0.9192
        cap_01 = bsc_channel_capacity(0.01)
        assert 0.918 < cap_01 < 0.921

        # At p = 0.10, H2(0.10) ≈ 0.4690 -> C ≈ 0.5310
        cap_10 = bsc_channel_capacity(0.10)
        assert 0.530 < cap_10 < 0.532

    def test_bsc_capacity_monotonicity(self):
        # Capacity should decrease monotonically from p=0 to p=0.5
        probs = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        capacities = [bsc_channel_capacity(p) for p in probs]
        for i in range(len(capacities) - 1):
            assert capacities[i] >= capacities[i + 1]


# ---------------------------------------------------------------------------
# 2. Wilson Score Confidence Interval Tests
# ---------------------------------------------------------------------------
class TestWilsonIntervals:
    """Verify statistical properties of the Wilson score interval."""

    def test_zero_total_trials(self):
        assert wilson_score_interval(0, 0) == (0.0, 0.0)

    def test_zero_successes_lower_bound(self):
        # Lower bound must be exactly 0.0 when k = 0, but upper bound > 0.0
        low, high = wilson_score_interval(0, 20, confidence=0.95)
        assert low == 0.0
        assert high > 0.0
        assert high < 0.25

    def test_all_successes_upper_bound(self):
        # Upper bound must be exactly 1.0 when k = n, but lower bound < 1.0
        low, high = wilson_score_interval(20, 20, confidence=0.95)
        assert high == 1.0
        assert low < 1.0
        assert low > 0.75

    def test_symmetry(self):
        # Interval for 5/20 should mirror 15/20
        l1, u1 = wilson_score_interval(5, 20)
        l2, u2 = wilson_score_interval(15, 20)
        assert pytest.approx(l1, abs=1e-5) == 1.0 - u2
        assert pytest.approx(u1, abs=1e-5) == 1.0 - l2


# ---------------------------------------------------------------------------
# 3. Synthetic Data Analysis Tests
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_raw_dataframe() -> pd.DataFrame:
    """Create a realistic synthetic raw trials DataFrame."""
    rows = []
    corpora = ["text_sample.txt", "code_sample.txt"]
    configs = [(64, 32), (128, 64)]
    p_levels = [0.0, 0.01, 0.05]
    trials = 3

    trial_id = 0
    for corpus in corpora:
        orig_size = 1000 if corpus == "text_sample.txt" else 800
        comp_size = 650 if corpus == "text_sample.txt" else 600
        cr = orig_size / comp_size

        for N, K in configs:
            rate = K / N
            num_blocks = math.ceil((comp_size * 8) / K)
            total_bits = num_blocks * N
            eff_rate = (orig_size * 8) / total_bits

            for p in p_levels:
                for t in range(trials):
                    is_p0 = (p == 0.0)
                    is_noisy = (p > 0.02)
                    success = 1 if is_p0 or (p == 0.01 and t < 2) else 0

                    flips = int(total_bits * p)
                    res_ber = 0.0 if success else p * 0.5
                    res_fer = 0.0 if success else min(1.0, p * 4.0)

                    rows.append({
                        "trial_id": trial_id,
                        "run_id": "test_run",
                        "corpus_name": corpus,
                        "original_size_bytes": orig_size,
                        "container_size_bytes": comp_size,
                        "source_compression_ratio": cr,
                        "polar_N": N,
                        "polar_K": K,
                        "polar_rate": rate,
                        "num_polar_blocks": num_blocks,
                        "pad_count": (num_blocks * K) - (comp_size * 8),
                        "total_transmitted_channel_bits": total_bits,
                        "effective_transmission_rate": eff_rate,
                        "channel_p": p,
                        "trial_seed": 1000 + trial_id,
                        "channel_bit_flips": flips,
                        "empirical_channel_ber": flips / total_bits if total_bits > 0 else 0.0,
                        "polar_residual_bit_errors": int(comp_size * 8 * res_ber),
                        "polar_residual_ber": res_ber,
                        "polar_block_error_count": int(num_blocks * res_fer),
                        "polar_block_error_rate": res_fer,
                        "exact_reconstruction": success,
                        "success": success,
                        "failure_stage": "NONE" if success else "RANS_DECOMPRESS",
                        "error_message": "" if success else "Container validation failure",
                        "rans_encode_time_ms": 2.5,
                        "polar_encode_time_ms": 15.0,
                        "channel_time_ms": 0.5,
                        "polar_decode_time_ms": 120.0,
                        "rans_decode_time_ms": 3.0,
                        "total_latency_ms": 141.0,
                    })
                    trial_id += 1

    return pd.DataFrame(rows)


class TestAnalysisCalculations:
    """Test analysis aggregation and summarization logic."""

    def test_corpus_compression_summary(self, synthetic_raw_dataframe, tmp_path):
        # Create dummy corpora files in tmp_path
        (tmp_path / "text_sample.txt").write_bytes(b"hello world natural language text " * 30)
        (tmp_path / "code_sample.txt").write_bytes(b"def func(x):\n    return x + 1\n" * 30)

        df = analyze_corpus_compression(synthetic_raw_dataframe, data_dir=tmp_path)
        assert not df.empty
        assert len(df) == 2
        assert "zero_order_entropy_bits_per_byte" in df.columns
        assert "ideal_shannon_compression_ratio" in df.columns
        assert "compression_efficiency_pct" in df.columns

        # Actual CR must be positive
        assert (df["compression_ratio"] > 1.0).all()
        # Efficiency must be <= 100%
        assert (df["compression_efficiency_pct"] <= 100.0).all()

    def test_channel_reliability_summary(self, synthetic_raw_dataframe):
        df = analyze_channel_reliability(synthetic_raw_dataframe)
        assert not df.empty
        assert "exact_reconstruction_rate" in df.columns
        assert "wilson_ci_lower_95" in df.columns
        assert "wilson_ci_upper_95" in df.columns
        assert "bsc_capacity" in df.columns

        # Verify noiseless conditions (p=0.0) have 100% reconstruction rate
        noiseless = df[df["channel_p"] == 0.0]
        assert (noiseless["exact_reconstruction_rate"] == 1.0).all()

        # CI bounds must be within [0, 1]
        assert (df["wilson_ci_lower_95"] >= 0.0).all()
        assert (df["wilson_ci_upper_95"] <= 1.0).all()
        assert (df["wilson_ci_lower_95"] <= df["wilson_ci_upper_95"]).all()

    def test_polar_tradeoffs_summary(self, synthetic_raw_dataframe):
        df = analyze_polar_configuration_tradeoffs(synthetic_raw_dataframe)
        assert not df.empty
        assert "mean_polar_decode_ms" in df.columns
        assert "mean_block_fer" in df.columns
        assert "mean_effective_rate" in df.columns

    def test_failure_diagnostics(self, synthetic_raw_dataframe):
        df = analyze_failure_diagnostics(synthetic_raw_dataframe)
        assert not df.empty
        assert "failure_stage" in df.columns
        assert "percentage" in df.columns
        assert df["failure_stage"].iloc[0] == "RANS_DECOMPRESS"

    def test_empty_dataframe_handling(self):
        empty_df = pd.DataFrame()
        assert analyze_corpus_compression(empty_df).empty
        assert analyze_channel_reliability(empty_df).empty
        assert analyze_polar_configuration_tradeoffs(empty_df).empty
        assert analyze_failure_diagnostics(empty_df).empty


# ---------------------------------------------------------------------------
# 4. Figure Generation Integration Tests
# ---------------------------------------------------------------------------
class TestVisualizationGeneration:
    """Verify headless generation of all 6 publication figures."""

    def test_all_six_figures_generate(self, synthetic_raw_dataframe, tmp_path):
        fig_dir = tmp_path / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "text_sample.txt").write_bytes(b"test data " * 50)
        (data_dir / "code_sample.txt").write_bytes(b"test code " * 50)

        # Plot 1
        p1 = plot_reconstruction_rate_vs_p(synthetic_raw_dataframe, fig_dir)
        assert p1.exists() and p1.stat().st_size > 0

        # Plot 2
        p2 = plot_pre_fec_ber_vs_p(synthetic_raw_dataframe, fig_dir)
        assert p2.exists() and p2.stat().st_size > 0

        # Plot 3
        p3 = plot_residual_ber_and_fer_vs_p(synthetic_raw_dataframe, fig_dir)
        assert p3.exists() and p3.stat().st_size > 0

        # Plot 4
        p4 = plot_effective_rate_vs_code_rate(synthetic_raw_dataframe, fig_dir)
        assert p4.exists() and p4.stat().st_size > 0

        # Plot 5
        p5 = plot_compression_ratio_by_corpus(synthetic_raw_dataframe, fig_dir, data_dir=data_dir)
        assert p5.exists() and p5.stat().st_size > 0

        # Plot 6
        p6 = plot_runtime_breakdown(synthetic_raw_dataframe, fig_dir)
        assert p6.exists() and p6.stat().st_size > 0
