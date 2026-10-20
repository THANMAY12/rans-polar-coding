# Experiment Harness & Benchmark Guide

This directory provides a configuration-driven framework to run reproducible, automated benchmark sweeps evaluating joint rANS source coding and Polar channel coding over a controlled Binary Symmetric Channel (BSC).

---

## 1. Quickstart

### Running a Fast Smoke Test
The smoke test runs a minimal 4-trial sweep in `< 2 seconds` to verify environment and pipeline health:
```powershell
.\.venv\Scripts\python experiments/run_experiments.py --smoke-test
```

### Running a Full Benchmark Sweep
The full sweep evaluates all available text corpora in `data/raw/` across multiple Polar block lengths $(N \in \{64, 128, 256\})$, code rates, and noise levels ($p \in [0.0, 0.10]$):
```powershell
.\.venv\Scripts\python experiments/run_experiments.py --full-sweep --trials 5
```

### Running Custom Parameter Configurations
You can specify a JSON configuration file:
```powershell
.\.venv\Scripts\python experiments/run_experiments.py --config path/to/config.json
```

---

## 2. Methodology & Statistical Rigor

### Independent Seed Derivation
To prevent global state contamination and ensure 100% deterministic reproducibility across platforms:
- A single `master_seed` (e.g. `20261020`) generates independent per-trial seeds via deterministic hashing:
  $$\text{trial\_seed} = (\text{master\_seed} + \text{condition\_idx} \times 100007 + \text{trial\_idx} \times 997) \pmod{2^{31} - 1}$$
- Every trial can be re-run in isolation by initializing with its logged `trial_seed`.

### Wilson Score Confidence Intervals (95%)
Reconstruction success rate is a binomial proportion $\hat{p} = k / n$. Normal approximations (Wald intervals) fail at boundary values ($k = 0$ or $k = n$). We compute two-sided **Wilson score intervals**:
$$\tilde{n} = n + z^2, \quad \tilde{p} = \frac{k + \frac{z^2}{2}}{\tilde{n}}$$
$$\text{CI}_{95\%} = \tilde{p} \pm \frac{z}{\tilde{n}} \sqrt{\frac{k(n - k)}{n} + \frac{z^2}{4}}, \quad \text{with } z = 1.95996$$
*Note: Confidence intervals quantify sampling uncertainty across the finite trial batch and do not represent universal channel capacity guarantees.*

---

## 3. Metrics Dictionary

### Raw Trials CSV (`results/raw_trials_<run_id>.csv`)
- `trial_id`: Sequential trial integer.
- `run_id`: Unique timestamped experiment run identifier.
- `corpus_name`: Source file name from `data/raw/`.
- `original_size_bytes`: Raw uncompressed text size in bytes.
- `container_size_bytes`: Size of rANS binary container (including header and frequency table).
- `source_compression_ratio`: $\text{Size}_{\text{original}} / \text{Size}_{\text{container}}$.
- `polar_N`, `polar_K`, `polar_rate`: Polar codeword length, info length, and code rate $K/N$.
- `num_polar_blocks`: Number of Polar blocks transmitted.
- `pad_count`: Number of zero-padding bits appended to the final block.
- `total_transmitted_channel_bits`: Total channel bits ($M \times N$).
- `effective_transmission_rate`: Information bits received per transmitted channel bit ($\frac{8 \times \text{Size}_{\text{original}}}{\text{total\_transmitted\_bits}}$).
- `channel_p`: BSC crossover probability.
- `channel_bit_flips`: Exact number of bits inverted by the BSC.
- `empirical_channel_ber`: Observed pre-FEC BER ($\text{flips} / \text{channel\_bits}$).
- `polar_residual_bit_errors`: Post-FEC residual bit errors in the container bitstream.
- `polar_residual_ber`: Observed post-FEC bit error rate.
- `polar_block_error_count`, `polar_block_error_rate`: Number and fraction of Polar blocks with $\ge 1$ error (Frame Error Rate, FER).
- `exact_reconstruction`: Binary flag (1 if byte-for-byte identical, 0 otherwise).
- `success`: Binary flag (1 if exact reconstruction and zero exceptions, 0 otherwise).
- `failure_stage`: Stage where failure occurred (`NONE`, `POLAR_DECODE`, `RANS_DECOMPRESS`, `CONTAINER_INTEGRITY`, `BYTE_MISMATCH`).
- `rans_encode_time_ms`, `polar_encode_time_ms`, `channel_time_ms`, `polar_decode_time_ms`, `rans_decode_time_ms`, `total_latency_ms`: Stage latencies in milliseconds.

### Summary CSV (`results/summary_<run_id>.csv`)
Aggregates trials by `(corpus_name, polar_N, polar_K, channel_p)`:
- `trials_count`: Total trials conducted ($n$).
- `exact_reconstruction_count`: Total successful trials ($k$).
- `exact_reconstruction_rate`: Observed success rate ($k / n$).
- `wilson_ci_lower_95`, `wilson_ci_upper_95`: 95% Wilson score confidence interval bounds.
- `failure_rate`: $1 - (k / n)$.
- `mean_compression_ratio`, `mean_effective_transmission_rate`: Mean source efficiency and throughput.
- `mean_channel_ber`, `std_channel_ber`: Mean and sample standard deviation of channel error rate.
- `mean_polar_residual_ber`, `std_polar_residual_ber`: Mean and std dev of post-FEC error rate.
- `mean_polar_block_fer`, `std_polar_block_fer`: Mean and std dev of Frame Error Rate.
- `mean_total_latency_ms`, `std_total_latency_ms`: Mean and std dev of overall execution latency.
