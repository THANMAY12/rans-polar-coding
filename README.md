# End-to-End Evaluation of rANS Source Coding and Polar Channel Coding for Reliable Text Transmission over a No-Packet-Loss Wired Channel Model

A reproducible Information Theory and Coding (ITC) research framework designed to evaluate the joint performance of range Asymmetric Numeral Systems (rANS) source coding and Polar channel coding over a controlled, no-packet-loss wired channel model.

---

## 1. Project Objectives

The primary goal of this research project is to implement, test, and empirically evaluate an end-to-end digital communications pipeline for plain-text data:
1. **Source Coding**: range Asymmetric Numeral Systems (rANS) operating at the byte level to achieve near-optimal entropy compression on UTF-8 text streams.
2. **Channel Coding**: Arikan Polar codes with Successive Cancellation Decoding (SCD) for forward error correction (FEC).
3. **Input Modality**: UTF-8 encoded text corpora processed strictly as raw bytes without lossy pre-transforms.
4. **Channel Simulation**: A physical-layer wired channel abstraction operating under two deterministic regimes:
   - *Ideal Mode*: Error-free transmission ($P_e = 0$).
   - *Controlled Bit-Flip Mode*: Binary Symmetric Channel (BSC) with parameterized bit-error probability $p \in [0, 0.5]$ without packet loss or jitter.
5. **Evaluation Metrics**:
   - Source Compression Ratio (CR) & Bit Rate (bits/symbol)
   - Pre-FEC Channel Bit Error Rate (Channel BER)
   - Post-FEC Residual Bit Error Rate (Source BER)
   - Exact Reconstruction Rate (Symbol / Byte Match Rate)
   - Effective Transmission Rate (Information bits successfully received per transmitted channel use)
   - Encoding and Decoding Runtime Latency (Wall-clock and throughput in MB/s)
   - Decoding Failure Rate / Frame Error Rate (FER)
6. **Research Artifacts**: CSV experiment logs, Matplotlib visualization figures, unit and integration test suites, exhaustive documentation, and a publication-ready research paper.

---

## 2. Strict Exclusions

Per experimental design constraints, the following legacy or alternate coding techniques are **strictly excluded** from this repository:
- **Excluded Source Coding**:
  - Shannon–Fano coding
  - Shannon–Fano–Elias coding
  - Huffman coding (classical and minimum-variance variants)
  - Lempel–Ziv–Welch (LZW)
- **Excluded Channel Coding**:
  - Low-Density Parity-Check (LDPC) codes

---

## 3. System Architecture

```
+-----------------------------------------------------------------------------------------+
|                                    TRANSMITTER                                          |
|                                                                                         |
|  +--------------------+      +--------------------+      +---------------------------+  |
|  |   UTF-8 Text File  | ---> | rANS Source Encoder| ---> | Bit Stream Serialization  |  |
|  |     (Raw Bytes)    |      | (Frequency Table)  |      |     & Padding (K bits)    |  |
|  +--------------------+      +--------------------+      +---------------------------+  |
|                                                                    |                    |
|                                                                    v                    |
|                                                          +--------------------+         |
|                                                          |   Polar Encoder    |         |
|                                                          |   (Mothercode N)   |         |
|                                                          +--------------------+         |
+--------------------------------------------------------------------|--------------------+
                                                                     | Codewords
                                                                     v
+-----------------------------------------------------------------------------------------+
|                                NO-PACKET-LOSS CHANNEL                                   |
|                                                                                         |
|      Mode A: Error-Free Channel (Ideal)                                                 |
|      Mode B: Controlled Bit-Flip Channel (BSC with Error Probability p)                 |
+-----------------------------------------------------------------------------------------+
                                                                     |
                                                                     v Received Bits / LLRs
+-----------------------------------------------------------------------------------------+
|                                     RECEIVER                                            |
|                                                                                         |
|  +--------------------+      +--------------------+      +---------------------------+  |
|  | Reconstructed Text | <--- | rANS Source Decoder| <--- | Unpadding & Deserializing |  |
|  |  (UTF-8 Validation)|      | (Symbol Recovery)  |      |   (Information bits K)    |  |
|  +--------------------+      +--------------------+      +---------------------------+  |
|                                                                    ^                    |
|                                                                    |                    |
|                                                          +--------------------+         |
|                                                          |  Polar SCD Decoder |         |
|                                                          | (Likelihood/LLRs)  |         |
|                                                          +--------------------+         |
+-----------------------------------------------------------------------------------------+
                                     |
                                     v
                       +---------------------------+
                       |   Metrics & Evaluation    |
                       |  - Channel & Source BER   |
                       |  - Compression Ratio      |
                       |  - Exact Reconstruction   |
                       |  - Execution Profiling    |
                       +---------------------------+
```

---

## 4. Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── source_coding/         # rANS entropy encoder and decoder modules
│   │   └── __init__.py
│   ├── channel_coding/        # Polar code wrapper, construction, and SC decoding
│   │   └── __init__.py
│   ├── channel/               # Ideal and bit-flip channel models
│   │   └── __init__.py
│   └── utils/                 # Bit packing, unpadding, and seed utilities
│       ├── __init__.py
│       ├── bit_utils.py
│       └── seed.py
├── experiments/               # Reproducible benchmark harnesses and runners
│   └── __init__.py
├── tests/                     # Unit, roundtrip, and integration test suite
│   ├── __init__.py
│   ├── test_imports.py
│   ├── test_bit_utils.py
│   └── test_seed.py
├── data/
│   ├── raw/                   # Input UTF-8 corpora (with documentation)
│   │   └── README.md
│   └── processed/             # Formatted benchmark sets (tracked with .gitkeep)
│       └── .gitkeep
├── results/                   # Generated evaluation CSVs, metrics, and plots
│   └── .gitkeep
├── docs/
│   └── DEVELOPMENT_PLAN.md    # Multi-phase engineering roadmap & acceptance criteria
├── pyproject.toml             # Project package configuration and test runner settings
├── requirements.txt           # Verified package dependencies
└── README.md
```

---

## 5. Setup and Installation

### Prerequisites
- Python 3.11 or later
- Git

### Installation Steps
1. Clone or navigate to the repository directory:
   ```bash
   cd "d:/sem 7/itc/Project"
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install verified dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the test suite:
   ```bash
   pytest
   ```

---

## 6. Dependency Selection & Verification

| Dependency | Target Purpose | Verified Package & License |
|---|---|---|
| **NumPy** (`>=1.24.0`) | Fast bit vector operations, LLR calculations | `numpy` (BSD-3-Clause) |
| **pandas** (`>=2.0.0`) | Experiment telemetry logging and tabular aggregations | `pandas` (BSD-3-Clause) |
| **Matplotlib** (`>=3.7.0`) | Publication-quality BER, FER, and rate plots | `matplotlib` (PSF / BSD compatible) |
| **pytest** (`>=7.0.0`) | Test automation and regression assertions | `pytest` (MIT) |
| **Polar Codes** (`>=1.2.2`) | Arikan Polar mothercode construction, encoding & SCD | `py-polar-codes` (MIT, Brendon McBain) |

---

## 7. Development Roadmap

The project is structured into eight sequential phases:
- **Phase 1: Project Setup and Audit** (Current phase: baseline structure, dependencies, seed management, bit packing tests)
- **Phase 2: Source Coding (rANS)** (Byte-level rANS frequency modeling, encoder, decoder, roundtrip validation)
- **Phase 3: Channel Coding (Polar Codes)** (Block-level Polar encoder, frozen bit selection via Bhattacharyya bounds, SCD decoder)
- **Phase 4: Channel Models** (Ideal error-free channel and controlled BSC bit-flip simulation)
- **Phase 5: End-to-End Integration Pipeline** (Coupling rANS and Polar code blocks with padding and serialization)
- **Phase 6: Evaluation & Experiment Harness** (Automated parametric sweeps across block lengths, code rates, and bit-error rates)
- **Phase 7: Visualization & Comparative Analysis** (Publication-ready plots for BER/FER vs SNR/p, Compression Ratio, and Throughput)
- **Phase 8: Research Paper & Final Verification** (Academic manuscript, experimental validation, reproducibility audit)

Refer to [docs/DEVELOPMENT_PLAN.md](file:///d:/sem%207/itc/Project/docs/DEVELOPMENT_PLAN.md) for full phase-by-phase acceptance criteria.
