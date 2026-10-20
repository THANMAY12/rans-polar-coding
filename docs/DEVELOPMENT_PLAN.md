# Research Engineering Development Plan (Phases 1–8)

**Project Title**: End-to-End Evaluation of rANS Source Coding and Polar Channel Coding for Reliable Text Transmission over a No-Packet-Loss Wired Channel Model  
**Author**: Lead Python Research Engineer  
**Date**: October 2026  

---

## Strict Exclusions Reference
Per project specification, the following algorithms are **strictly excluded** from all phases:
- Shannon–Fano
- Shannon–Fano–Elias
- Huffman & Minimum-Variance Huffman
- Lempel–Ziv–Welch (LZW)
- Low-Density Parity-Check (LDPC) codes

---

## Phase Breakdown and Acceptance Criteria

### Phase 1: Project Setup, Audit, and Baseline Infrastructure
- **Objective**: Establish a clean, reproducible research baseline, directory structure, dependency management, seed control, and fundamental bit manipulation utilities.
- **Key Deliverables**:
  - Validated directory layout (`src/`, `experiments/`, `tests/`, `data/raw/`, `data/processed/`, `results/`, `docs/`).
  - `requirements.txt` and `pyproject.toml` with verified packages (`numpy`, `pandas`, `matplotlib`, `pytest`, `py-polar-codes`).
  - `.gitignore` configured for virtual environments, Python caches, generated artifacts, and secrets.
  - Bit packing/unpacking and padding utilities (`src/utils/bit_utils.py`).
  - Deterministic random seed management (`src/utils/seed.py`).
  - Verification of Polar code library API and license.
  - Unit tests covering imports, bit conversions, padding, and seed repeatability.
- **Acceptance Criteria**:
  - All Phase 1 tests pass with 100% success rate.
  - No rANS or Polar coding algorithms implemented in this phase.
  - `git diff` confirms zero untracked environments, credentials, or generated files.

---

### Phase 2: Source Coding (range Asymmetric Numeral Systems)
- **Objective**: Implement a byte-level rANS entropy encoder and decoder optimized for UTF-8 text streams.
- **Key Deliverables**:
  - Byte frequency estimation and integer-scaled cumulative distribution table generation.
  - Precision parameterization with power-of-two state normalization (e.g., state bounds $[L, 2L-1]$).
  - Stream encoder with forward/reverse byte buffering.
  - Stream decoder with symbol lookup and state inversion.
  - Roundtrip unit tests on ASCII, multi-byte UTF-8, and varied entropy distributions.
- **Acceptance Criteria**:
  - Lossless reconstruction: $\text{decode}(\text{encode}(\mathbf{x})) \equiv \mathbf{x}$ for all input byte sequences.
  - Achieved compression ratio consistently outperforms zero-order empirical entropy within 1–2%.
  - Execution profiling benchmarks demonstrating linear time complexity $\mathcal{O}(M)$ with symbol count $M$.
  - Strict compliance with exclusions (no Huffman, Shannon-Fano, or LZW).

---

### Phase 3: Channel Coding (Polar Codes)
- **Objective**: Implement a typed, modular Polar coding interface around the verified `py-polar-codes` engine.
- **Key Deliverables**:
  - Mothercode parameterization with block lengths $N = 2^n$ and information block lengths $K$.
  - Construction of frozen bit set using Bhattacharyya bounds (`construction_type='bb'`).
  - Polar encoder generating systematic and non-systematic codewords.
  - Successive Cancellation Decoder (SCD) taking channel Log-Likelihood Ratios (LLRs).
  - Block partitioning and chunking utilities for messages exceeding $K$ bits.
- **Acceptance Criteria**:
  - Valid generator matrix multiplication and code construction verified.
  - Zero-error decoding on noiseless channel inputs.
  - Expected error correction thresholds achieved over simulated AWGN/BSC likelihood inputs.
  - Strict compliance with exclusions (no LDPC).

---

### Phase 4: Wired Channel Simulation Models
- **Objective**: Build a physical-layer wired channel simulator operating with zero packet loss.
- **Key Deliverables**:
  - **Ideal Mode**: Deterministic error-free transmission ($P_e = 0$).
  - **Controlled Bit-Flip Mode**: Binary Symmetric Channel (BSC) applying independent bit flips with probability $p \in [0, 0.5]$ using isolated, seeded PRNGs.
  - Soft-decision LLR computer converting binary channel outputs into likelihood inputs for the Polar SCD decoder:
    $$\text{LLR}(y) = \ln \left( \frac{1-p}{p} \right) \cdot (1 - 2y)$$
  - Channel telemetry logging (exact bit flip counts, locations, and instantaneous error rates).
- **Acceptance Criteria**:
  - Ideal mode produces 0 bit flips.
  - BSC mode produces empirical bit error rate within $\pm 3\sigma$ confidence interval of specified parameter $p$.
  - Zero packet loss, truncation, or reordering under all test conditions.

---

### Phase 5: End-to-End System Integration
- **Objective**: Interlock source coding, bitstream framing, channel coding, and channel simulation into a unified transmission pipeline.
- **Key Deliverables**:
  - Transmitter pipeline: UTF-8 Text $\to$ rANS Encoder $\to$ Bit Packing & Framing $\to$ Polar Encoder.
  - Receiver pipeline: Channel $\to$ Polar SCD Decoder $\to$ Deframing & Bit Unpacking $\to$ rANS Decoder $\to$ Text.
  - Metadata serialization (compressed length, padding count, frequency table header).
  - Robust exception handling for unrecoverable transmission frames.
- **Acceptance Criteria**:
  - Exact end-to-end string recovery on Ideal Channel for arbitrary UTF-8 text corpora.
  - Graceful reporting of frame decoding failures when channel error exceeds channel capacity or Polar error-correction capability.
  - Complete integration test suite verifying end-to-end data integrity.

---

### Phase 6: Experiment Harness and Metric Collection
- **Objective**: Implement an automated, reproducible benchmark suite executing parametric sweeps.
- **Key Deliverables**:
  - Experiment parameter matrix sweeping over:
    - Code lengths: $N \in \{64, 128, 256, 512, 1024\}$
    - Code rates: $R = K/N \in \{1/4, 1/3, 1/2, 2/3, 3/4\}$
    - Channel error rates: $p \in [10^{-4}, 0.2]$
    - Diverse text corpora (literature, source code, structured JSON).
  - Deterministic batch runner recording metrics:
    - Compression Ratio ($\text{CR} = \text{Size}_{\text{raw}} / \text{Size}_{\text{compressed}}$)
    - Channel Bit Error Rate (Pre-FEC BER)
    - Source Bit Error Rate (Post-FEC BER)
    - Exact Reconstruction Rate ($\%$)
    - Effective Transmission Rate (payload bits / channel uses)
    - Encoder / Decoder latency (ms)
    - Frame Error Rate (FER)
  - CSV output logging to `results/`.
- **Acceptance Criteria**:
  - Experiments run completely unattended from a single CLI command.
  - Zero synthetic or fabricated data; all metrics computed strictly from real pipeline runs.
  - Repeatable results under identical seeds.

---

### Phase 7: Visualization and Comparative Performance Analysis
- **Objective**: Generate publication-ready figures and statistical analysis of experimental data.
- **Key Deliverables**:
  - BER / FER vs. Channel Error Probability $p$ (waterfall curves).
  - Effective Transmission Rate vs. Code Rate across varying noise regimes.
  - Source coding efficiency and rANS compression ratio comparison against Shannon entropy.
  - Throughput and latency trade-off plots.
  - High-resolution SVG/PNG figure export to `results/`.
- **Acceptance Criteria**:
  - All figures render headlessly without display server dependency.
  - Proper axis labels, error bars / confidence intervals, and publication typography.
  - Automated script generating all paper figures from saved CSV logs.

---

### Phase 8: Research Paper and Final Reproducibility Audit
- **Objective**: Synthesize theoretical foundations, mathematical formulations, system design, and experimental outcomes into a complete academic paper.
- **Key Deliverables**:
  - Full research manuscript (Markdown / LaTeX) formatted with:
    1. Introduction & Problem Statement
    2. Information-Theoretic Foundations (rANS & Polarization Theory)
    3. System Architecture & Channel Model
    4. Experimental Methodology & Benchmark Setup
    5. Results & Discussion (Waterfall curves, Throughput, Reconstruction)
    6. Conclusion & Future Outlook
  - Final repository audit ensuring clean working tree, complete documentation, and 100% test passing rate.
- **Acceptance Criteria**:
  - Publication-ready manuscript with verified mathematical formulas and citations.
  - Automated reproducibility script (`run_all.py` or equivalent) reproducing all paper figures and tables from scratch.
  - Final review passing all project constraints and exclusions.
