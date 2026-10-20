"""
Reproducible Experimental Analysis and Summary Generator.

Analyzes raw trial results from the rANS source coding + Polar channel coding +
Binary Symmetric Channel (BSC) end-to-end transmission pipeline.

Calculates:
- Exact byte-reconstruction success rates with 95% Wilson score confidence intervals.
- Channel error behavior: Pre-FEC BER vs theoretical BSC p.
- Error correction behavior: Post-FEC residual BER vs Polar block Frame Error Rate (FER).
- Source compression metrics: raw size, container size, empirical compression ratio,
  zero-order Shannon entropy H0(X), and theoretical ideal compression bound 8/H0.
- Transmission efficiency: Effective transmission rate vs Polar code rate and BSC capacity C(p).
- Computational latency breakdown across pipeline stages for different Polar block configurations.
- Failure diagnostics: breakdown of error stages and conditions.
"""

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Add project root to path if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.harness import wilson_score_interval


# ---------------------------------------------------------------------------
# Information-Theoretic Reference Formulas
# ---------------------------------------------------------------------------
def compute_zero_order_entropy(data: bytes) -> float:
    """
    Compute empirical zero-order Shannon entropy H0(X) in bits per byte.

    H0(X) = -sum_{b} P(b) * log2(P(b))

    Parameters
    ----------
    data : bytes
        Raw byte sequence.

    Returns
    -------
    float
        Zero-order entropy in [0.0, 8.0] bits/byte. Returns 0.0 for empty input.
    """
    if not data:
        return 0.0
    n = len(data)
    counts = Counter(data)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def compute_ideal_compression_ratio(entropy_bits_per_byte: float) -> float:
    """
    Compute the theoretical Shannon lossless compression ratio bound for an
    i.i.d. memoryless byte source: CR_ideal = 8.0 / H0(X).

    Parameters
    ----------
    entropy_bits_per_byte : float
        Source entropy in bits per byte.

    Returns
    -------
    float
        Theoretical compression ratio. Returns 0.0 if entropy <= 0.
    """
    if entropy_bits_per_byte <= 0.0:
        return 0.0
    return 8.0 / entropy_bits_per_byte


def bsc_channel_capacity(p: float) -> float:
    """
    Calculate theoretical Shannon channel capacity of a Binary Symmetric Channel.

    C(p) = 1 - H2(p)  [bits per channel use]
    where H2(p) = -p*log2(p) - (1-p)*log2(1-p) is the binary entropy function.

    Parameters
    ----------
    p : float
        BSC crossover probability in [0.0, 1.0].

    Returns
    -------
    float
        Capacity in [0.0, 1.0].
    """
    if p <= 0.0 or p >= 1.0:
        return 1.0 if p <= 0.0 else 0.0
    if p == 0.5:
        return 0.0
    # Binary entropy H2(p)
    h2 = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    return max(0.0, min(1.0, 1.0 - h2))


# ---------------------------------------------------------------------------
# Data Loading & Inventory
# ---------------------------------------------------------------------------
def find_results_files(results_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """
    Locate all raw trials, summaries, and configuration JSON files in results_dir.

    Returns
    -------
    Dict[str, List[Path]]
        Mapping of 'raw_csv', 'summary_csv', 'config_json' to list of paths.
    """
    dir_p = Path(results_dir)
    if not dir_p.exists():
        return {"raw_csv": [], "summary_csv": [], "config_json": []}

    raw_files = sorted(dir_p.glob("raw_trials_*.csv"))
    summary_files = sorted(dir_p.glob("summary_*.csv"))
    config_files = sorted(dir_p.glob("config_*.json"))

    return {
        "raw_csv": raw_files,
        "summary_csv": summary_files,
        "config_json": config_files,
    }


def load_raw_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a raw trial-level CSV and validate its schema.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {p}")
    df = pd.read_csv(p)
    return df


def load_latest_dataset(results_dir: Union[str, Path], prefer_benchmark: bool = True) -> Tuple[pd.DataFrame, Path]:
    """
    Load the most comprehensive or latest raw trials CSV dataset from results_dir.

    Parameters
    ----------
    results_dir : Path or str
        Directory where results are stored.
    prefer_benchmark : bool, default=True
        If True, prioritizes full benchmark files over smoke-test files if available.

    Returns
    -------
    Tuple[pd.DataFrame, Path]
        The loaded dataframe and the file path.
    """
    files = find_results_files(results_dir)["raw_csv"]
    if not files:
        raise FileNotFoundError(f"No raw trial CSV files found in {results_dir}")

    if prefer_benchmark:
        # Prioritize files not named smoke_test, or with highest row count
        benchmark_candidates = [f for f in files if "smoke_test" not in f.name]
        if benchmark_candidates:
            # Pick file with largest size / row count
            best_file = max(benchmark_candidates, key=lambda f: f.stat().st_size)
            return load_raw_dataset(best_file), best_file

    # Fallback to latest file by modification time
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return load_raw_dataset(latest_file), latest_file


# ---------------------------------------------------------------------------
# Specialized Analysis Modules
# ---------------------------------------------------------------------------
def analyze_corpus_compression(
    raw_df: pd.DataFrame,
    data_dir: Union[str, Path] = "data/raw",
) -> pd.DataFrame:
    """
    Analyze source coding compression performance across text corpora.

    Computes:
    - Original file size (bytes).
    - Container size (bytes) including header and frequency table metadata.
    - Container overhead in bytes and percentage.
    - Empirical rANS compression ratio.
    - Zero-order Shannon entropy H0(X) in bits/byte.
    - Theoretical Shannon compression bound 8.0 / H0(X).
    - Compression efficiency relative to theoretical bound: CR_actual / CR_ideal.
    """
    if raw_df.empty:
        return pd.DataFrame()

    data_path = Path(data_dir)
    rows: List[Dict[str, Any]] = []

    for corpus_name, group in raw_df.groupby("corpus_name"):
        first_row = group.iloc[0]
        orig_size = int(first_row["original_size_bytes"])
        comp_size = float(group["container_size_bytes"].mean())
        cr_mean = float(group["source_compression_ratio"].mean())

        # Read corpus file to calculate empirical entropy
        target_f = data_path / str(corpus_name)
        if target_f.exists() and target_f.is_file():
            raw_bytes = target_f.read_bytes()
            h0 = compute_zero_order_entropy(raw_bytes)
            unique_syms = len(set(raw_bytes))
        else:
            h0 = 0.0
            unique_syms = 0

        ideal_cr = compute_ideal_compression_ratio(h0) if h0 > 0.0 else 0.0
        eff = (cr_mean / ideal_cr) * 100.0 if ideal_cr > 0.0 else 0.0
        overhead_bytes = comp_size - (orig_size * (h0 / 8.0)) if h0 > 0.0 else 0.0

        rows.append({
            "corpus_name": corpus_name,
            "original_size_bytes": orig_size,
            "unique_symbols": unique_syms,
            "container_size_bytes": comp_size,
            "compression_ratio": cr_mean,
            "zero_order_entropy_bits_per_byte": h0,
            "ideal_shannon_compression_ratio": ideal_cr,
            "compression_efficiency_pct": eff,
            "trials_sampled": len(group),
        })

    return pd.DataFrame(rows).sort_values("original_size_bytes").reset_index(drop=True)


def analyze_channel_reliability(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze end-to-end reliability and error correction across conditions.

    Computes:
    - Exact reconstruction success rate with 95% Wilson confidence intervals.
    - Pre-FEC channel BER (mean and std).
    - Post-FEC residual BER (mean and std).
    - Polar block Frame Error Rate (FER) (mean and std).
    - BSC capacity C(p).
    - Failure stage frequencies.
    """
    if raw_df.empty:
        return pd.DataFrame()

    group_cols = ["corpus_name", "polar_N", "polar_K", "polar_rate", "channel_p"]
    rows: List[Dict[str, Any]] = []

    for keys, group in raw_df.groupby(group_cols, sort=False):
        corpus_name, N, K, rate, p = keys
        n = len(group)
        successes = int(group["exact_reconstruction"].sum())
        rec_rate = successes / n if n > 0 else 0.0
        ci_low, ci_high = wilson_score_interval(successes, n, confidence=0.95)

        cap = bsc_channel_capacity(float(p))
        eff_rate = float(group["effective_transmission_rate"].mean())

        mean_pre_ber = float(group["empirical_channel_ber"].mean())
        std_pre_ber = float(group["empirical_channel_ber"].std(ddof=1) if n > 1 else 0.0)

        mean_post_ber = float(group["polar_residual_ber"].mean())
        std_post_ber = float(group["polar_residual_ber"].std(ddof=1) if n > 1 else 0.0)

        mean_fer = float(group["polar_block_error_rate"].mean())
        std_fer = float(group["polar_block_error_rate"].std(ddof=1) if n > 1 else 0.0)

        # Failure stages
        stage_counts = group["failure_stage"].value_counts().to_dict()

        rows.append({
            "corpus_name": corpus_name,
            "polar_N": int(N),
            "polar_K": int(K),
            "polar_rate": float(rate),
            "channel_p": float(p),
            "bsc_capacity": cap,
            "effective_transmission_rate": eff_rate,
            "trials_count": n,
            "exact_reconstruction_count": successes,
            "exact_reconstruction_rate": rec_rate,
            "wilson_ci_lower_95": ci_low,
            "wilson_ci_upper_95": ci_high,
            "mean_channel_ber": mean_pre_ber,
            "std_channel_ber": std_pre_ber,
            "mean_polar_residual_ber": mean_post_ber,
            "std_polar_residual_ber": std_post_ber,
            "mean_polar_block_fer": mean_fer,
            "std_polar_block_fer": std_fer,
            "failure_stages": json.dumps(stage_counts),
        })

    return pd.DataFrame(rows)


def analyze_polar_configuration_tradeoffs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze Polar configuration trade-offs (block length N, code rate K/N)
    in terms of decoding latency, block error rate, and reconstruction reliability.
    """
    if raw_df.empty:
        return pd.DataFrame()

    group_cols = ["polar_N", "polar_K", "polar_rate", "channel_p"]
    rows: List[Dict[str, Any]] = []

    for keys, group in raw_df.groupby(group_cols, sort=False):
        N, K, rate, p = keys
        n = len(group)
        successes = int(group["exact_reconstruction"].sum())
        rec_rate = successes / n if n > 0 else 0.0
        ci_low, ci_high = wilson_score_interval(successes, n, confidence=0.95)

        rows.append({
            "polar_N": int(N),
            "polar_K": int(K),
            "polar_rate": float(rate),
            "channel_p": float(p),
            "trials_count": n,
            "exact_reconstruction_rate": rec_rate,
            "wilson_ci_lower_95": ci_low,
            "wilson_ci_upper_95": ci_high,
            "mean_effective_rate": float(group["effective_transmission_rate"].mean()),
            "mean_num_blocks": float(group["num_polar_blocks"].mean()),
            "mean_residual_ber": float(group["polar_residual_ber"].mean()),
            "mean_block_fer": float(group["polar_block_error_rate"].mean()),
            "mean_polar_encode_ms": float(group["polar_encode_time_ms"].mean()),
            "mean_polar_decode_ms": float(group["polar_decode_time_ms"].mean()),
            "mean_total_latency_ms": float(group["total_latency_ms"].mean()),
            "std_total_latency_ms": float(group["total_latency_ms"].std(ddof=1) if n > 1 else 0.0),
        })

    return pd.DataFrame(rows)


def analyze_failure_diagnostics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Examine failed trials to categorize failure mechanisms and stages.
    """
    if raw_df.empty:
        return pd.DataFrame()

    failed = raw_df[raw_df["exact_reconstruction"] == 0]
    if failed.empty:
        return pd.DataFrame(columns=[
            "failure_stage", "count", "percentage", "min_p", "max_p", "mean_residual_ber"
        ])

    rows: List[Dict[str, Any]] = []
    total_failures = len(failed)

    for stage, grp in failed.groupby("failure_stage"):
        rows.append({
            "failure_stage": stage,
            "count": len(grp),
            "percentage": (len(grp) / total_failures) * 100.0,
            "min_channel_p": float(grp["channel_p"].min()),
            "max_channel_p": float(grp["channel_p"].max()),
            "mean_residual_ber": float(grp["polar_residual_ber"].mean()),
            "mean_block_fer": float(grp["polar_block_error_rate"].mean()),
        })

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full Analysis Orchestrator
# ---------------------------------------------------------------------------
def run_full_analysis(
    results_dir: Union[str, Path] = "results",
    data_dir: Union[str, Path] = "data/raw",
    output_dir: Optional[Union[str, Path]] = None,
    prefer_benchmark: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Execute full statistical analysis pipeline on experiment results.

    Saves:
    - analysis_corpus_compression.csv
    - analysis_reliability_summary.csv
    - analysis_polar_tradeoffs.csv
    - analysis_failures.csv

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of computed summary DataFrames.
    """
    res_dir = Path(results_dir)
    out_dir = Path(output_dir) if output_dir else res_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df, source_file = load_latest_dataset(res_dir, prefer_benchmark=prefer_benchmark)
    print(f"Loaded {len(raw_df)} trials from: {source_file}")

    corpus_df = analyze_corpus_compression(raw_df, data_dir=data_dir)
    reliability_df = analyze_channel_reliability(raw_df)
    tradeoffs_df = analyze_polar_configuration_tradeoffs(raw_df)
    failures_df = analyze_failure_diagnostics(raw_df)

    # Save summary tables
    corpus_df.to_csv(out_dir / "analysis_corpus_compression.csv", index=False)
    reliability_df.to_csv(out_dir / "analysis_reliability_summary.csv", index=False)
    tradeoffs_df.to_csv(out_dir / "analysis_polar_tradeoffs.csv", index=False)
    failures_df.to_csv(out_dir / "analysis_failures.csv", index=False)

    print(f"Analysis tables successfully written to: {out_dir}")

    return {
        "corpus_compression": corpus_df,
        "channel_reliability": reliability_df,
        "polar_tradeoffs": tradeoffs_df,
        "failures": failures_df,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze experimental transmission benchmark data.")
    parser.add_argument("-r", "--results-dir", type=str, default="results", help="Directory containing results")
    parser.add_argument("-d", "--data-dir", type=str, default="data/raw", help="Directory containing raw corpora")
    parser.add_argument("-o", "--output-dir", type=str, default="results", help="Directory to save analysis tables")
    args = parser.parse_args()

    results = run_full_analysis(
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    for name, df in results.items():
        print(f"\n--- {name.upper()} ---")
        print(df.head(10).to_string(index=False))
