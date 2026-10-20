"""
Command-Line Runner for Automated Information Theory & Coding Benchmark Sweeps.

Usage:
    # Run fast smoke test
    python experiments/run_experiments.py --smoke-test

    # Run full multi-corpus sweep
    python experiments/run_experiments.py --full-sweep

    # Run from JSON config file
    python experiments/run_experiments.py --config experiments/config.json
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.harness import ExperimentConfig, ExperimentRunner


def print_progress(current: int, total: int, status: str) -> None:
    pct = (current / total) * 100.0 if total > 0 else 0.0
    print(f"[{pct:5.1f}%] [{current}/{total}] {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Run automated rANS + Polar + BSC benchmark sweeps and statistical summaries."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute a minimal 4-trial smoke test to verify execution integrity.",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Execute the full multi-corpus benchmark sweep across (N, K) and crossover probabilities.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an existing JSON experiment configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="results",
        help="Directory where CSV and JSON results will be saved (default: results).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override number of trials per condition.",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Error: configuration file not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        import json
        config_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(config_dict)
    elif args.smoke_test:
        sample_file = Path("data/raw/sample_itc_corpus.txt")
        if not sample_file.exists():
            # Generate default sample if missing
            from experiments.demonstrate_rans import SAMPLE_TEXT
            sample_file.parent.mkdir(parents=True, exist_ok=True)
            sample_file.write_text(SAMPLE_TEXT * 10, encoding="utf-8")
        config = ExperimentConfig.smoke_test_config(input_file=sample_file, output_dir=out_dir)
    elif args.full_sweep:
        # Full sweep across all available text corpora in data/raw/
        raw_files = [
            f for f in Path("data/raw").glob("*.txt")
            if f.is_file() and f.name != "README.md"
        ]
        if not raw_files:
            print("Error: No text files found in data/raw/", file=sys.stderr)
            sys.exit(1)
        config = ExperimentConfig(
            name="full_itc_sweep",
            input_files=raw_files,
            polar_configs=[(64, 32), (128, 64), (256, 128)],
            channel_probabilities=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10],
            trials_per_condition=5,
            master_seed=20261020,
            output_dir=out_dir,
        )
    else:
        # Default behavior: run smoke test
        print("No mode selected, defaulting to --smoke-test.")
        sample_file = Path("data/raw/sample_itc_corpus.txt")
        if not sample_file.exists():
            from experiments.demonstrate_rans import SAMPLE_TEXT
            sample_file.parent.mkdir(parents=True, exist_ok=True)
            sample_file.write_text(SAMPLE_TEXT * 10, encoding="utf-8")
        config = ExperimentConfig.smoke_test_config(input_file=sample_file, output_dir=out_dir)

    if args.trials is not None:
        config.trials_per_condition = args.trials

    print("=" * 70)
    print(f"Launching Experiment Sweep: {config.name}")
    print(f"Input Files           : {[f.name for f in config.input_files]}")
    print(f"Polar Configurations  : {config.polar_configs}")
    print(f"Crossover Probabilities (p): {config.channel_probabilities}")
    print(f"Trials Per Condition  : {config.trials_per_condition}")
    print(f"Master Seed           : {config.master_seed}")
    print(f"Output Directory      : {config.output_dir}")
    print("=" * 70)

    runner = ExperimentRunner(config)
    t_start = time.perf_counter()
    raw_df, summary_df = runner.run_all(progress_callback=print_progress)
    t_elapsed = time.perf_counter() - t_start

    raw_path, summary_path, config_path = runner.save_results(
        raw_df, summary_df, run_id=f"{config.name}_{int(time.time())}"
    )

    print("\n" + "=" * 70)
    print("Sweep Completed Successfully!")
    print(f"Total Trials Executed : {len(raw_df)}")
    print(f"Elapsed Time          : {t_elapsed:.2f} seconds")
    print(f"Raw Trials CSV        : {raw_path}")
    print(f"Summary CSV           : {summary_path}")
    print(f"Run Config JSON       : {config_path}")
    print("=" * 70)

    # Print summary table preview
    cols_to_print = [
        "corpus_name", "polar_N", "polar_K", "channel_p",
        "trials_count", "exact_reconstruction_rate", "wilson_ci_lower_95", "wilson_ci_upper_95",
        "mean_compression_ratio", "mean_effective_transmission_rate"
    ]
    available_cols = [c for c in cols_to_print if c in summary_df.columns]
    print("\nSummary Results Preview:")
    print(summary_df[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
