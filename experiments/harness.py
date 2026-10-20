"""
Automated Experiment Harness and Parameter Sweep Framework.

Provides:
1. Configuration-driven experiment sweeps over text corpora, Polar parameters (N, K),
   and BSC crossover probabilities p.
2. Deterministic per-trial seed derivation from a single master seed.
3. Telemetry capture: compression ratio, effective rate, channel BER, residual BER,
   block error rate (FER), exact reconstruction status, and stage latencies.
4. Statistical aggregation across trials with 95% Wilson score confidence intervals
   for binomial exact reconstruction success rates.
5. Export to timestamped, structured CSV and JSON metadata in results/.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.channel_coding.polar_codec import PolarConfig
from src.pipeline import EndToEndPipeline, PipelineResult
from src.utils.seed import get_rng


# ---------------------------------------------------------------------------
# Statistical Helpers: Wilson Score Interval
# ---------------------------------------------------------------------------
def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate the two-sided Wilson score confidence interval for a binomial proportion.

    Unlike the Wald (normal approximation) interval, the Wilson interval does not
    degenerate to zero width at p=0 or p=1 and maintains nominal coverage even
    for small sample sizes.

    Parameters
    ----------
    successes : int
        Number of successful outcomes (exact reconstructions).
    total : int
        Total number of trials.
    confidence : float, default=0.95
        Confidence level (default: 0.95, z ≈ 1.95996).

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) clamped to [0.0, 1.0].
    """
    if total <= 0:
        return 0.0, 0.0

    k = max(0, min(successes, total))
    n = total
    p_hat = k / n

    # Standard normal quantile: z ≈ 1.95996 for 95% confidence
    if confidence == 0.95:
        z = 1.959963984540054
    elif confidence == 0.99:
        z = 2.5758293035489004
    elif confidence == 0.90:
        z = 1.6448536269514722
    else:
        # Fallback approximation for standard normal inverse CDF
        z = 1.959963984540054

    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))
    )

    lower = 0.0 if k == 0 else max(0.0, center - half_width)
    upper = 1.0 if k == n else min(1.0, center + half_width)

    return lower, upper


# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """
    Configuration specification for an automated experiment sweep.
    """
    name: str = "itc_benchmark_sweep"
    input_files: List[Path] = field(default_factory=list)
    polar_configs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(128, 64), (256, 128)]
    )
    channel_probabilities: List[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.01, 0.05, 0.10]
    )
    trials_per_condition: int = 5
    master_seed: int = 20261020
    scale_bits: int = 14
    design_snr_db: float = 2.0
    systematic: bool = False
    output_dir: Path = field(default_factory=lambda: Path("results"))

    def validate(self) -> None:
        """Validate experiment parameters and file paths."""
        if not self.input_files:
            raise ValueError("input_files list cannot be empty")

        for f in self.input_files:
            p = Path(f)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Input file not found: {p}")

        if not self.polar_configs:
            raise ValueError("polar_configs list cannot be empty")

        for N, K in self.polar_configs:
            if N <= 1 or not (N & (N - 1) == 0):
                raise ValueError(f"Polar N must be a power of 2, got {N}")
            if K <= 0 or K >= N:
                raise ValueError(f"Polar K must satisfy 1 <= K < N, got K={K}, N={N}")

        if not self.channel_probabilities:
            raise ValueError("channel_probabilities list cannot be empty")

        for p in self.channel_probabilities:
            if p < 0.0 or p > 0.5:
                raise ValueError(f"Channel crossover probability p must be in [0.0, 0.5], got {p}")

        if 0.0 not in self.channel_probabilities:
            raise ValueError("channel_probabilities must include p=0.0 as the noiseless baseline")

        if self.trials_per_condition < 1:
            raise ValueError(f"trials_per_condition must be >= 1, got {self.trials_per_condition}")

        if self.master_seed < 0:
            raise ValueError(f"master_seed must be non-negative, got {self.master_seed}")

    @classmethod
    def smoke_test_config(
        cls,
        input_file: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "ExperimentConfig":
        """Pre-configured fast smoke test configuration for quick verification."""
        if input_file is None:
            input_file = Path("data/raw/sample_itc_corpus.txt")
        out = output_dir or Path("results")
        return cls(
            name="smoke_test",
            input_files=[input_file],
            polar_configs=[(64, 32)],
            channel_probabilities=[0.0, 0.01],
            trials_per_condition=2,
            master_seed=42,
            output_dir=out,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        d = asdict(self)
        d["input_files"] = [str(f) for f in self.input_files]
        d["polar_configs"] = [list(cfg) for cfg in self.polar_configs]
        d["output_dir"] = str(self.output_dir)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Construct configuration from dictionary."""
        cfg = cls(
            name=data.get("name", "experiment"),
            input_files=[Path(f) for f in data["input_files"]],
            polar_configs=[(int(c[0]), int(c[1])) for c in data["polar_configs"]],
            channel_probabilities=[float(p) for p in data["channel_probabilities"]],
            trials_per_condition=int(data.get("trials_per_condition", 5)),
            master_seed=int(data.get("master_seed", 20261020)),
            scale_bits=int(data.get("scale_bits", 14)),
            design_snr_db=float(data.get("design_snr_db", 2.0)),
            systematic=bool(data.get("systematic", False)),
            output_dir=Path(data.get("output_dir", "results")),
        )
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# Experiment Runner & Aggregator
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """
    Executes parameter sweeps, records trial metrics, computes statistical summaries,
    and exports reproducible CSV/JSON artifacts.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        config.validate()
        self.config = config

    def _derive_trial_seed(self, condition_idx: int, trial_idx: int) -> int:
        """
        Derive an independent deterministic seed for each trial from the master seed.
        """
        # Linear congruential / hashing combination
        seed = (self.config.master_seed + condition_idx * 100_007 + trial_idx * 997) & 0x7FFFFFFF
        return int(seed)

    def run_single_trial(
        self,
        trial_id: int,
        file_path: Path,
        N: int,
        K: int,
        p: float,
        trial_seed: int,
        run_id: str,
    ) -> Dict[str, Any]:
        """Execute a single end-to-end trial and extract all telemetry fields."""
        data = file_path.read_bytes()
        orig_size = len(data)

        polar_cfg = PolarConfig(
            N=N,
            K=K,
            design_snr_db=self.config.design_snr_db,
            systematic=self.config.systematic,
        )
        pipeline = EndToEndPipeline(
            polar_config=polar_cfg,
            scale_bits=self.config.scale_bits,
        )

        res: PipelineResult = pipeline.run(data, p=p, seed=trial_seed)

        return {
            "trial_id": trial_id,
            "run_id": run_id,
            "corpus_name": file_path.name,
            "original_size_bytes": orig_size,
            "container_size_bytes": res.container_size_bytes,
            "source_compression_ratio": res.source_compression_ratio,
            "polar_N": N,
            "polar_K": K,
            "polar_rate": res.code_rate,
            "num_polar_blocks": res.num_polar_blocks,
            "pad_count": res.pad_count,
            "total_transmitted_channel_bits": res.transmitted_codeword_bits,
            "effective_transmission_rate": res.effective_transmission_rate,
            "channel_p": p,
            "trial_seed": trial_seed,
            "channel_bit_flips": res.channel_bit_flips,
            "empirical_channel_ber": res.channel_ber,
            "polar_residual_bit_errors": res.polar_residual_bit_errors,
            "polar_residual_ber": res.polar_residual_ber,
            "polar_block_error_count": res.polar_block_error_count,
            "polar_block_error_rate": res.polar_block_error_rate,
            "exact_reconstruction": int(res.exact_reconstruction),
            "success": int(res.success),
            "failure_stage": res.failure_stage or "NONE",
            "error_message": res.error_message or "",
            "rans_encode_time_ms": res.rans_encode_time_ms,
            "polar_encode_time_ms": res.polar_encode_time_ms,
            "channel_time_ms": res.channel_time_ms,
            "polar_decode_time_ms": res.polar_decode_time_ms,
            "rans_decode_time_ms": res.rans_decode_time_ms,
            "total_latency_ms": res.total_latency_ms,
        }

    def run_all(
        self,
        run_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full combinatorial parameter sweep across all files, (N, K), p, and trials.

        Returns
        -------
        Tuple[raw_trials_df, summary_df]
        """
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{self.config.name}_{timestamp}"

        # Calculate total trials
        total_conditions = (
            len(self.config.input_files)
            * len(self.config.polar_configs)
            * len(self.config.channel_probabilities)
        )
        total_trials = total_conditions * self.config.trials_per_condition

        rows: List[Dict[str, Any]] = []
        trial_counter = 0
        condition_idx = 0

        for file_path in self.config.input_files:
            for N, K in self.config.polar_configs:
                for p in self.config.channel_probabilities:
                    condition_idx += 1
                    for t in range(self.config.trials_per_condition):
                        trial_seed = self._derive_trial_seed(condition_idx, t)
                        if progress_callback is not None:
                            status_msg = f"{file_path.name} | N={N}, K={K} | p={p} | trial {t+1}/{self.config.trials_per_condition}"
                            progress_callback(trial_counter, total_trials, status_msg)

                        record = self.run_single_trial(
                            trial_id=trial_counter,
                            file_path=file_path,
                            N=N,
                            K=K,
                            p=p,
                            trial_seed=trial_seed,
                            run_id=run_id,
                        )
                        rows.append(record)
                        trial_counter += 1

        raw_df = pd.DataFrame(rows)
        summary_df = self.aggregate_results(raw_df)

        return raw_df, summary_df

    @staticmethod
    def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw trial-level measurements by condition (corpus_name, polar_N, polar_K, channel_p).
        Computes means, standard deviations, exact reconstruction rates, and 95% Wilson intervals.
        """
        if raw_df.empty:
            return pd.DataFrame()

        group_cols = ["corpus_name", "polar_N", "polar_K", "polar_rate", "channel_p"]
        summary_rows: List[Dict[str, Any]] = []

        for group_keys, group in raw_df.groupby(group_cols, sort=False):
            corpus_name, N, K, rate, p = group_keys
            n = len(group)
            successes = int(group["exact_reconstruction"].sum())
            reconstruction_rate = successes / n if n > 0 else 0.0

            ci_low, ci_high = wilson_score_interval(successes, n, confidence=0.95)

            summary_rows.append({
                "corpus_name": corpus_name,
                "polar_N": int(N),
                "polar_K": int(K),
                "polar_rate": float(rate),
                "channel_p": float(p),
                "trials_count": n,
                "exact_reconstruction_count": successes,
                "exact_reconstruction_rate": reconstruction_rate,
                "wilson_ci_lower_95": ci_low,
                "wilson_ci_upper_95": ci_high,
                "failure_rate": 1.0 - reconstruction_rate,
                "mean_original_size_bytes": float(group["original_size_bytes"].mean()),
                "mean_container_size_bytes": float(group["container_size_bytes"].mean()),
                "mean_compression_ratio": float(group["source_compression_ratio"].mean()),
                "mean_effective_transmission_rate": float(group["effective_transmission_rate"].mean()),
                "mean_channel_ber": float(group["empirical_channel_ber"].mean()),
                "std_channel_ber": float(group["empirical_channel_ber"].std(ddof=1) if n > 1 else 0.0),
                "mean_polar_residual_ber": float(group["polar_residual_ber"].mean()),
                "std_polar_residual_ber": float(group["polar_residual_ber"].std(ddof=1) if n > 1 else 0.0),
                "mean_polar_block_fer": float(group["polar_block_error_rate"].mean()),
                "std_polar_block_fer": float(group["polar_block_error_rate"].std(ddof=1) if n > 1 else 0.0),
                "mean_rans_encode_ms": float(group["rans_encode_time_ms"].mean()),
                "mean_polar_encode_ms": float(group["polar_encode_time_ms"].mean()),
                "mean_polar_decode_ms": float(group["polar_decode_time_ms"].mean()),
                "mean_rans_decode_ms": float(group["rans_decode_time_ms"].mean()),
                "mean_total_latency_ms": float(group["total_latency_ms"].mean()),
                "std_total_latency_ms": float(group["total_latency_ms"].std(ddof=1) if n > 1 else 0.0),
            })

        return pd.DataFrame(summary_rows)

    def save_results(
        self,
        raw_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        run_id: str,
        overwrite: bool = False,
    ) -> Tuple[Path, Path, Path]:
        """
        Save raw trials CSV, summary CSV, and configuration JSON to output_dir.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Trial-level dataframe.
        summary_df : pd.DataFrame
            Aggregated summary dataframe.
        run_id : str
            Unique run identifier.
        overwrite : bool, default=False
            If False, appends counter suffix to prevent overwriting existing files.

        Returns
        -------
        Tuple[raw_path, summary_path, config_path]
        """
        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_p = out_dir / f"raw_trials_{run_id}.csv"
        summary_p = out_dir / f"summary_{run_id}.csv"
        config_p = out_dir / f"config_{run_id}.json"

        if not overwrite:
            counter = 1
            base_run_id = run_id
            while raw_p.exists() or summary_p.exists() or config_p.exists():
                run_id = f"{base_run_id}_{counter}"
                raw_p = out_dir / f"raw_trials_{run_id}.csv"
                summary_p = out_dir / f"summary_{run_id}.csv"
                config_p = out_dir / f"config_{run_id}.json"
                counter += 1

        raw_df.to_csv(raw_p, index=False)
        summary_df.to_csv(summary_p, index=False)

        config_dict = self.config.to_dict()
        config_dict["run_id"] = run_id
        config_dict["completed_at"] = datetime.now().isoformat()
        config_p.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

        return raw_p, summary_p, config_p
