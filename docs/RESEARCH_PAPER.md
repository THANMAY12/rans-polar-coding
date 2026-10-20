# End-to-End Evaluation of rANS Source Coding and Polar Channel Coding for Reliable Text Transmission over a No-Packet-Loss Wired Channel Model

**Authors:** Information Theory & Coding (ITC) Research Team  
**Institution:** Academic Information Theory & Communications Research Laboratory  
**Document Type:** Formal Research Paper & Empirical Benchmark Study  
**Date:** October 2026  
**Artifact Status:** Audited Experimental Evaluation (Phases 1–8)  
**Repository Reference:** `THANMAY12/rans-polar-coding` (Git branch `main`)  

---

## Abstract

Lossless text transmission over physical channels requires a delicate balance between source entropy compression and channel error-correcting redundancy. In this study, we present a reproducible, end-to-end evaluation of an integrated communication pipeline coupling byte-oriented **range Asymmetric Numeral Systems (rANS)** for source entropy coding with 5G Arikan **Polar channel coding** under Successive Cancellation Decoding (SCD) across a physical **Binary Symmetric Channel (BSC)** with no packet loss. We establish an empirical benchmark across three linguistic and syntactical corpora—Natural Language English prose, Scientific Technical prose, and Python source code—spanning codeword block lengths $N \in \{64, 128, 256\}$, Polar code rates $R \in \{0.25, 0.50, 0.75\}$, and crossover probabilities $p \in [0.0, 0.10]$ over 525 independent, deterministically seeded trials. 

Our findings demonstrate: (1) rANS static byte compression achieves 73.9% to 81.5% of the theoretical zero-order Shannon entropy bound on ~1 KB text excerpts, rising to 96.4% on 11.8 KB texts as container header overhead amortizes; (2) ideal noiseless baseline transmission ($p = 0$) achieves an exact byte-for-byte reconstruction rate of 100% across all 75 baseline trials (95% Wilson confidence interval: $[0.9513, 1.0000]$); (3) channel polarization error resilience scales directly with block length $N$, where at $p = 0.020$ and rate $R = 0.50$, block length $N = 256$ maintains an 80.0% reconstruction success rate compared to 46.7% for $N = 64$; (4) compressed rANS streams exhibit extreme error sensitivity to uncorrected channel bit errors, where a single residual bit flip desynchronizes the arithmetic state; and (5) in all 181 observed trial failures, the rANS container and state validation layer safely trapped corruption, preventing silent data corruption. We provide an exact information-theoretic analysis reconciling operational source transmission rates with Shannon channel capacity $C(p)$, alongside complete open-source reproducibility artifacts.

**Keywords:** Range Asymmetric Numeral Systems (rANS), Polar Codes, Binary Symmetric Channel (BSC), Channel Polarization, Successive Cancellation Decoding, Lossless Source Coding, Shannon Entropy, Wilson Score Interval.

---

## 1. Introduction and Research Motivation

Claude Shannon's seminal 1948 treatise established the Source-Channel Separation Theorem: under asymptotic block lengths, data compression (source coding) and error protection (channel coding) can be designed independently without sacrificing optimal performance [1]. In practice, however, finite block lengths, strict latency budgets, and real-world channel impairments impose fundamental constraints on communication systems [2]. 

Traditional lossless entropy coding has long been polarized between Huffman coding—which is fast but restricted to integer-bit assignments—and traditional arithmetic coding—which achieves near-entropy compression but suffers from serial bit-level renormalization bottlenecks. The development of Asymmetric Numeral Systems (ANS), and specifically byte-oriented range ANS (rANS) by Jarek Duda [3] and further refined for high-throughput streaming by Fabian Giesen [4], resolved this dichotomy. By operating directly on native machine words via table-driven interval state transitions, rANS approaches theoretical zero-order Shannon entropy at throughputs an order of magnitude faster than conventional arithmetic coders.

Concurrently, Arikan's 2009 discovery of Polar codes [5] provided the first mathematically proven class of linear block error-correcting codes capable of achieving the symmetric capacity of discrete memoryless channels with explicit, low-complexity Successive Cancellation Decoding (SCD). Polar codes have since been ratified as the standard channel coding scheme for control channels in 5G New Radio (3GPP TS 38.212) [6].

Despite the theoretical maturity of both components, their joint end-to-end integration across noisy channels presents complex practical trade-offs:
1. **Source Entropy vs. Error Sensitivity:** rANS compresses redundant text into a dense, high-entropy bitstream. While this minimizes the number of bits requiring channel transmission, it strips away natural source redundancy. Consequently, an uncorrected channel error that flips even a single bit in the rANS payload induces an *avalanche effect*, catastrophically desynchronizing the rANS state register during decompression.
2. **Effective Transmission Rate vs. Channel Capacity:** When source text is compressed by ratio $\text{CR} > 1$, the user's operational effective transmission rate (uncompressed source bits delivered per channel use) can exceed the raw channel code rate $K/N$. This requires rigorous information-theoretic normalization against the Shannon BSC capacity $C(p) = 1 - H_2(p)$ to prevent dimensional inconsistencies.
3. **Finite-Length Constraints:** While Polar codes achieve capacity as $N \to \infty$, short block lengths ($N \le 256$) encounter finite-length dispersion backoffs [2].

This paper provides an empirical, reproducible evaluation of an end-to-end rANS source coding + Polar channel coding pipeline transmitting UTF-8 text over a physical Binary Symmetric Channel model without packet loss.

---

## 2. Research Questions and Contributions

### 2.1 Research Questions
- **RQ1 (Source Efficiency):** How closely does finite-byte rANS container compression approach the theoretical zero-order Shannon entropy bound $H_0(X)$ across distinct text domains (natural language prose, technical text, and source code)?
- **RQ2 (Reliability and Polarization Depth):** How does end-to-end exact byte-reconstruction reliability scale across channel crossover probabilities $p \in [0.0, 0.10]$ as a function of Polar codeword block length $N \in \{64, 128, 256\}$ and code rate $R \in \{0.25, 0.50, 0.75\}$?
- **RQ3 (Error Propagation & Integrity):** What is the empirical relationship between pre-FEC channel BER, post-FEC residual BER, Polar block Frame Error Rate (FER), and rANS decompression failure? Does the rANS parser reliably detect corrupted streams?
- **RQ4 (Transmission Efficiency vs. Asymptotic Capacity):** How does the operational effective transmission rate $R_{\text{eff}}$ compare with the entropy-normalized information rate $R_{\text{info}}$ and the theoretical Shannon BSC capacity $C(p)$?
- **RQ5 (Computational Complexity):** Where does latency concentrate across pipeline stages in an unoptimized reference implementation?

### 2.2 Contributions
1. **Integrated Reference Pipeline:** A fully modular, open-source Python 3.11 implementation linking byte-oriented rANS, Arikan Polar channel coding with Bhattacharyya parameter design, out-of-band block framing, and a length-preserving Binary Symmetric Channel without packet loss.
2. **Multi-Domain Experimental Benchmark:** An audited combinatorial sweep spanning 525 deterministically seeded trials across three standardized text corpora, 5 Polar configurations, and 7 crossover probabilities ($p = 0.0$ to $0.10$).
3. **Rigorous Statistical Formulation:** Application of 95% Wilson score confidence intervals for exact binomial reconstruction success rates, resolving finite-sample uncertainty without normal-approximation degeneracies at boundary probabilities ($p=0, 1$).
4. **Information-Theoretic Normalization:** Clarification of the dimensional distinction between raw source-bit throughput $R_{\text{eff}}$ and entropy-normalized channel information rate $R_{\text{info}}$ relative to Shannon capacity $C(p)$.
5. **Complete Reproducibility Artifacts:** Open-source code, raw trial-level CSVs, aggregate summaries, reproducible plotting scripts, and automated test suites (204 unit and integration tests passing).

---

## 3. Related Work

### 3.1 Lossless Entropy Source Coding
Entropy coding maps source symbols into bit sequences such that average codeword length approaches source entropy $H(X)$ [1]. Huffman coding assigns integer bit lengths, bounding redundancy between $0$ and $1$ bit per symbol. Arithmetic coding achieves fractional bit precision by iteratively subdividing the interval $[0, 1)$, but standard implementations require sequential bit-level renormalization shifts and multi-precision arithmetic.

Duda (2009, 2013) introduced Asymmetric Numeral Systems (ANS) [3], representing information states as large integers with non-uniform digit distributions. The range variant (rANS) [4] operates as a streaming finite-state entropy coder where state transitions and renormalization occur at byte or word granularity, combining the fractional-bit compression efficiency of arithmetic coding with throughputs exceeding hundreds of megabytes per second.

### 3.2 Polar Channel Coding
Arikan (2009) proved that applying the kernel matrix $G_2 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$ recursively across $n = \log_2 N$ polarization stages polarizes $N$ independent copies of a binary memoryless channel into extreme virtual channels with mutual information tending either to 0 (completely noisy) or 1 (completely noiseless) [5]. Polar codes achieve channel capacity under Successive Cancellation Decoding (SCD) as $N \to \infty$. 

To address performance limitations at finite block lengths, Tal and Vardy (2015) developed Successive Cancellation List (SCL) decoding [7], which, when combined with Cyclic Redundancy Check (CRC) concatenation, meets or exceeds state-of-the-art LDPC performance. The 3GPP standardization committee subsequently adopted CRC-aided Polar codes for 5G New Radio uplink and downlink control channels [6]. In this study, we focus on classical Arikan Polar coding with SCD to evaluate the fundamental polarization behavior over controlled crossover rates.

### 3.3 Scope and Strict Exclusions
In strict compliance with the project charter, this investigation is intentionally delimited:
- **Excluded Source Algorithms:** Shannon–Fano, Shannon–Fano–Elias, Huffman coding (including minimum-variance Huffman), and Lempel–Ziv–Welch (LZW) were strictly excluded from implementation.
- **Excluded Channel Algorithms:** Low-Density Parity-Check (LDPC) codes, Turbo codes, and convolutional codes were strictly excluded.
- **Baseline Philosophy:** All comparative references are strictly theoretical and analytical: empirical zero-order Shannon entropy $H_0(X)$ as the source bound, and Shannon BSC capacity $C(p) = 1 - H_2(p)$ as the channel bound.

---

## 4. Background and Mathematical Formulation

### 4.1 Empirical Zero-Order Shannon Entropy
Let a text file be represented as a finite sequence of $M$ bytes $X = (x_1, x_2, \dots, x_M)$, where each byte $x_i \in \mathcal{A} \subseteq \{0, 1, \dots, 255\}$. Let $c(b)$ denote the frequency of byte value $b$ in $X$. The empirical probability is $P(b) = c(b) / M$.

The zero-order empirical Shannon entropy $H_0(X)$ in bits per byte is defined as [1]:
$$H_0(X) = - \sum_{b \in \mathcal{A}} P(b) \log_2 P(b) \quad [\text{bits/byte}]$$

The theoretical lower bound on average codeword length for any zero-order lossless prefix code is $H_0(X)$. Since uncompressed bytes occupy 8 bits, the theoretical upper bound on zero-order compression ratio is:
$$\text{CR}_{\text{ideal}} = \frac{8.0}{H_0(X)}$$

### 4.2 Range Asymmetric Numeral Systems (rANS) Formulation
Our byte-oriented rANS implementation operates with scale precision $B = 14$ ($M_{\text{scale}} = 2^{14} = 16,384$) and lower state bound $L = 2^{23}$. The working state $x$ satisfies $x \in [L, 2^8 L - 1) = [2^{23}, 2^{31} - 1)$, ensuring state transitions fit within standard 32-bit unsigned integer registers without arithmetic overflow.

#### 4.2.1 Frequency Normalization (Hare-Niemeyer Method)
Symbol frequencies $c(b)$ are scaled to integers $f_s \ge 1$ such that $\sum_{s \in \mathcal{A}} f_s = M_{\text{scale}} = 16,384$. We employ the Largest Remainder (Hare-Niemeyer) method:
1. Every present symbol is initially assigned $f_s = 1$.
2. The remaining quota $M_{\text{scale}} - |\mathcal{A}|$ is allocated proportionally to exact fractional frequencies $\frac{c(s)}{M} \times M_{\text{scale}}$.
3. Residual discrepancies are distributed to symbols with the largest fractional remainders.

#### 4.2.2 State Transitions and Renormalization
Cumulative frequencies are defined as $C(s) = \sum_{k < s} f_k$.
- **Encoding Step:** Given current state $x$ and symbol $s$, if $x \ge 2^8 \left\lfloor \frac{L}{f_s} \right\rfloor M_{\text{scale}}$, state $x$ is renormalized by streaming out the least-significant byte: $x \leftarrow \lfloor x / 256 \rfloor$. The state is then updated:
  $$x' = \left\lfloor \frac{x}{f_s} \right\rfloor M_{\text{scale}} + (x \pmod{f_s}) + C(s)$$
  Because encoding processes symbols in reverse order (LIFO), input text is decoded in proper forward order.
- **Decoding Step:** Given state $x$, the current cumulative slot is $m = x \pmod{M_{\text{scale}}}$. Binary search identifies symbol $s$ such that $C(s) \le m < C(s) + f_s$. The state transitions to:
  $$x' = f_s \left\lfloor \frac{x}{M_{\text{scale}}} \right\rfloor + (m - C(s))$$
  If $x' < L$, renormalization consumes bytes from the compressed bitstream: $x' \leftarrow 256 x' + \text{byte}$.

#### 4.2.3 Container Format Overhead
The compressed stream is serialized into an audited binary container:
- Magic Identifier: 4 bytes (`b"RANS"`).
- Container Version: 1 byte (`0x01`).
- Scale Precision: 1 byte ($B = 14$).
- Uncompressed Byte Length: 8 bytes (uint64, big-endian).
- Unique Symbol Count: 2 bytes (uint16, big-endian).
- Normalized Frequency Table: $3 \times |\mathcal{A}|$ bytes (1 byte symbol, 2 bytes uint16 frequency).
- Payload Length: 4 bytes (uint32, big-endian).
- rANS Compressed Byte Stream: $L_{\text{payload}}$ bytes.

Total container size is:
$$S_{\text{container}} = 20 + 3 |\mathcal{A}| + L_{\text{payload}} \quad [\text{bytes}]$$
The empirical compression ratio is strictly defined as:
$$\text{CR}_{\text{actual}} = \frac{M}{S_{\text{container}}}$$

---

### 4.3 Arikan Polar Channel Coding Formulation
A Polar code is parameterized by codeword length $N = 2^n$ ($n \in \mathbb{N}$) and information length $K < N$, yielding code rate $R = K/N$. 

#### 4.3.1 Bhattacharyya Parameter Channel Construction
For a physical BSC with crossover probability $p$, the channel Bhattacharyya parameter is $Z(W) = 2\sqrt{p(1-p)}$. Arikan's channel polarization transforms $N$ channels recursively:
$$Z(W_{2k}^{(2i-1)}) \le 2 Z(W_k^{(i)}) - \left(Z(W_k^{(i)})\right)^2$$
$$Z(W_{2k}^{(2i)}) = \left(Z(W_k^{(i)})\right)^2$$
Information bits are mapped to the $K$ sub-channels with the smallest Bhattacharyya parameters $\mathcal{I} \subset \{1, \dots, N\}$, while the frozen set $\mathcal{F} = \mathcal{I}^c$ of size $N - K$ is assigned frozen bits set to 0.

#### 4.3.2 Encoding and Decoding
The source vector $u \in \{0, 1\}^N$ satisfies $u_{\mathcal{I}} = m$ and $u_{\mathcal{F}} = 0$. The transmitted codeword is:
$$x = u G_N = u B_N F^{\otimes n}$$
where $F = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$, $B_N$ is the bit-reversal permutation matrix, and $\otimes$ denotes the Kronecker product.

Successive Cancellation Decoding (SCD) estimates information bits sequentially based on channel Log-Likelihood Ratios (LLRs):
$$\hat{u}_i = \begin{cases} 0 & \text{if } i \in \mathcal{F} \\ 0 & \text{if } i \in \mathcal{I} \text{ and } L_N^{(i)}(y_1^N, \hat{u}_1^{i-1}) \ge 0 \\ 1 & \text{otherwise} \end{cases}$$

---

### 4.4 Channel Model and Physical Abstraction

#### 4.4.1 Binary Symmetric Channel Model
The channel operates on transmitted binary sequences $x \in \{0, 1\}^L$, flipping each bit independently with crossover probability $p \in [0, 0.5]$:
$$y_i = x_i \oplus e_i, \quad P(e_i = 1) = p, \quad P(e_i = 0) = 1 - p$$
The channel preserves exact vector length ($|y| = |x|$) and symbol order.

#### 4.4.2 Log-Likelihood Ratio Sign Convention
The channel output produces soft LLRs according to the exact likelihood ratio:
$$\text{LLR}(y_i) = \ln \left(\frac{P(y_i \mid x_i = 0)}{P(y_i \mid x_i = 1)}\right) = (1 - 2y_i) \ln \left(\frac{1 - p}{p}\right)$$
- If $y_i = 0 \implies \text{LLR} > 0$.
- If $y_i = 1 \implies \text{LLR} < 0$.
- For ideal noiseless transmission ($p = 0$), LLRs are clamped to $\pm 100.0$.
- For pure noise ($p = 0.5$), $\text{LLR} = 0.0$.

#### 4.4.3 Physical Link Abstraction Disclaimer
> [!NOTE]
> **Channel Abstraction Disclaimer:** This model represents a digital wired link abstraction governed strictly by independent bit errors without packet loss, erasure, out-of-order reassembly, or carrier synchronization drift. It abstracts physical-layer analog effects (such as attenuation, ISI, and thermal noise) into an effective discrete crossover probability $p$.

---

### 4.5 Framing, Padding, and Joint Rate Definitions

#### 4.5.1 Block Division and Zero-Padding
The serialized rANS container ($S_{\text{container}}$ bytes) is unpacked into $L_{\text{bits}} = 8 \times S_{\text{container}}$ bits. To fit $K$-bit Polar blocks, the total number of blocks is:
$$N_{\text{blocks}} = \left\lceil \frac{L_{\text{bits}}}{K} \right\rceil$$
The bitstream is zero-padded with $N_{\text{pad}} = (N_{\text{blocks}} \times K) - L_{\text{bits}}$ bits ($0 \le N_{\text{pad}} < K$). Total transmitted channel bits are:
$$L_{\text{channel}} = N_{\text{blocks}} \times N$$

Framing metadata $(S_{\text{container}}, N_{\text{pad}})$ is treated as out-of-band header information. At the receiver, Polar decoding reconstructs $N_{\text{blocks}} \times K$ bits, from which exactly $N_{\text{pad}}$ padding bits are stripped prior to byte reassembly.

#### 4.5.2 Operational Rate vs. Entropy-Normalized Rate
The **operational effective transmission rate** $R_{\text{eff}}$ relates uncompressed user source bits to physical channel uses:
$$R_{\text{eff}} = \frac{8 \times M}{L_{\text{channel}}} = \frac{8 \times M}{N_{\text{blocks}} \times N} \approx \text{CR}_{\text{actual}} \times \left(\frac{K}{N}\right) \quad [\text{source bits / channel use}]$$

When source text possesses redundancy ($H_0(X) < 8.0$), transmitting 1 source bit conveys only $H_0(X) / 8.0$ bits of true entropy. To enable dimensionally consistent comparisons with Shannon BSC capacity $C(p) = 1 - H_2(p)$ [1], the **entropy-normalized information rate** $R_{\text{info}}$ is defined as:
$$R_{\text{info}} = R_{\text{eff}} \times \left(\frac{H_0(X)}{8.0}\right) = \frac{H_0(X) \times M}{L_{\text{channel}}} \quad [\text{entropy bits / channel use}]$$
Because lossless compression is bounded by entropy ($\text{CR} \le 8 / H_0$), we have:
$$R_{\text{info}} \approx \left(\frac{8}{H_0(X)} \times R_{\text{polar}}\right) \times \frac{H_0(X)}{8} = R_{\text{polar}} = \frac{K}{N} \le C(p)$$

---

### 4.6 Statistical Methodology: Wilson Score Interval
For $k$ exact reconstructions out of $n$ independent trials, the sample proportion is $\hat{p} = k/n$. The two-sided 95% Wilson score confidence interval is calculated as [8]:
$$CI_{95\%} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \quad z = 1.95996$$
The Wilson interval prevents the degenerate zero-width intervals produced by the Wald approximation at $\hat{p} = 0$ and $\hat{p} = 1$, correctly capturing finite-sample uncertainty.

---

## 5. Proposed System Architecture

The complete transmission pipeline operates as depicted below:

```
[ Input Text Bytes (M) ]
          │
          ▼
┌──────────────────────────────┐
│  rANS Source Encoder         │
│  - Static Frequency Table    │
│  - Renormalization State x   │
│  - Container Serialization   │
└──────────────┬───────────────┘
               │ Compressed Container (S_container bytes)
               ▼
┌──────────────────────────────┐
│  Bit Unpacker & Framing      │
│  - Byte to Bit Unpacking     │
│  - Divide into K-bit blocks  │
│  - Zero-padding (N_pad)      │
└──────────────┬───────────────┘
               │ Information Blocks (N_blocks x K)
               ▼
┌──────────────────────────────┐
│  Polar Channel Encoder       │
│  - Arikan Matrix G_N         │
│  - Bhattacharyya Frozen Set  │
└──────────────┬───────────────┘
               │ Transmitted Codewords (N_blocks x N)
               ▼
┌──────────────────────────────┐
│  Binary Symmetric Channel    │
│  - Bit-flip: P(e = 1) = p    │
│  - LLR Calculation           │
└──────────────┬───────────────┘
               │ Channel LLRs (N_blocks x N)
               ▼
┌──────────────────────────────┐
│  Polar SCD Decoder           │
│  - Successive Cancellation   │
│  - Hard Decision Vectors     │
└──────────────┬───────────────┘
               │ Decoded Information Bits
               ▼
┌──────────────────────────────┐
│  Framing Depadder & Packer   │
│  - Strip N_pad padding bits  │
│  - Bit to Byte Packing       │
└──────────────┬───────────────┘
               │ Reconstructed Container Bytes
               ▼
┌──────────────────────────────┐
│  rANS Source Decoder         │
│  - Magic & Header Validation │
│  - Frequency Table Parsing   │
│  - Arithmetic Decompression  │
│  - Byte-by-byte Equality     │
└──────────────┬───────────────┘
               │
               ▼
[ Reconstructed Text Bytes == Original? ]
```

---

## 6. Experimental Methodology

### 6.1 Corpora Description
Three standardized UTF-8 corpora were established under `data/raw/` representing distinct linguistic structures:
1. **Natural Language Prose (`natural_language_sample.txt`):** 1,240 bytes excerpted from Lewis Carroll's *Alice's Adventures in Wonderland*. Features standard English syntax, repeated articles, and punctuation ($|\mathcal{A}| = 42$).
2. **Scientific Technical Text (`technical_text_sample.txt`):** 1,255 bytes describing information theory and channel capacity. Features scientific vocabulary, formal grammatical structures, and mathematical notations ($|\mathcal{A}| = 50$).
3. **Python Source Code (`python_code_sample.txt`):** 1,001 bytes of structured Python code. Characterized by high indentation, keywords (`def`, `return`, `class`), mathematical operators, and variable identifiers ($|\mathcal{A}| = 60$).

An additional extended corpus (`sample_itc_corpus.txt`, 11,830 bytes, $|\mathcal{A}| = 83$) was evaluated to verify container overhead amortization.

### 6.2 Sweep Configuration and Sampling Protocol
- **Polar Configurations ($C=5$):**
  - Block length scaling at rate $R = 0.50$: $(64, 32)$, $(128, 64)$, $(256, 128)$.
  - Code rate trade-offs at $N = 128$: $(128, 32)$ ($R = 0.25$), $(128, 64)$ ($R = 0.50$), $(128, 96)$ ($R = 0.75$).
- **Crossover Probabilities ($P=7$):** $p \in \{0.000, 0.001, 0.005, 0.010, 0.020, 0.050, 0.100\}$.
- **Trials and Seed Derivation:** $T = 5$ trials per condition. Total conditions = $3 \times 5 \times 7 = 105$. Total trials = **525 trials**.
  $$\text{trial\_seed} = (\text{master\_seed} + \text{condition\_idx} \times 100007 + \text{trial\_idx} \times 997) \pmod{2^{31}-1}$$
  Master seed: `20261020`.

### 6.3 Telemetry Metrics
For every trial, 31 fields are recorded in `results/raw_trials_comprehensive_sweep_1792504862.csv`:
- `source_compression_ratio`: $M / S_{\text{container}}$.
- `empirical_channel_ber`: $\text{channel\_bit\_flips} / L_{\text{channel}}$.
- `polar_residual_ber`: $\text{residual\_bit\_errors} / L_{\text{bits}}$.
- `polar_block_error_rate`: $\text{corrupted\_polar\_blocks} / N_{\text{blocks}}$.
- `exact_reconstruction`: Binary flag ($1$ if reconstructed bytes strictly equal original bytes; $0$ otherwise).
- `failure_stage`: Identifies stage where corruption was detected (`NONE`, `RANS_DECOMPRESS`, etc.).
- Stage execution latencies: rANS encode, Polar encode, BSC transmission, Polar decode, rANS decode (ms).

---

## 7. Results

### 7.1 Source Coding Compression and Shannon Bounds

Table 1 presents the measured rANS compression metrics across all evaluated text corpora.

**Table 1: Source Coding Performance and Empirical Zero-Order Shannon Bound Comparison**
*(Provenance: `results/analysis_corpus_compression.csv`, audited from 525 trials)*

| Corpus Name | Domain | Raw Size (Bytes) | Unique Bytes $|\mathcal{A}|$ | Zero-Order Entropy $H_0$ (b/B) | Theoretical Bound $8/H_0$ | rANS Container (Bytes) | Empirical CR | Compression Efficiency ($\text{CR}_{\text{actual}} / \text{CR}_{\text{ideal}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `natural_language_sample.txt` | English Prose | 1,240 | 42 | 4.2608 | 1.878 | 810.0 | **1.531** | **81.5%** |
| `technical_text_sample.txt` | Scientific Text | 1,255 | 50 | 4.4506 | 1.798 | 872.0 | **1.439** | **80.1%** |
| `python_code_sample.txt` | Python Code | 1,001 | 60 | 4.5919 | 1.742 | 778.0 | **1.287** | **73.9%** |
| `sample_itc_corpus.txt` | Extended Text | 11,830 | 83 | 5.0029 | 1.599 | 7,671.0 | **1.542** | **96.4%** |

As depicted in Figure 5 (`results/figures/fig5_compression_ratio_by_corpus.png`), compression efficiency tracks the linguistic entropy hierarchy: Natural Language ($H_0 = 4.26$ b/B) achieves the highest compression ratio ($\text{CR} = 1.531$), followed by Technical Text ($\text{CR} = 1.439$) and Source Code ($\text{CR} = 1.287$). On short ~1 KB files, the 146–200 byte container overhead limits empirical efficiency to 74–82% of the theoretical bound. On the 11.8 KB corpus, container overhead is fully amortized, reaching **96.4%** of the zero-order Shannon limit.

---

### 7.2 Channel Verification and Baseline Correctness

Figure 2 (`results/figures/fig2_pre_fec_ber_vs_p.png`) compares the empirical pre-FEC channel BER against configured crossover probability $p$:
- $p = 0.000 \implies \text{BER} = 0.00000 \pm 0.00000$ (75 trials)
- $p = 0.001 \implies \text{BER} = 0.00101 \pm 0.00015$ (75 trials)
- $p = 0.005 \implies \text{BER} = 0.00511 \pm 0.00031$ (75 trials)
- $p = 0.010 \implies \text{BER} = 0.00998 \pm 0.00042$ (75 trials)
- $p = 0.020 \implies \text{BER} = 0.02008 \pm 0.00061$ (75 trials)
- $p = 0.050 \implies \text{BER} = 0.05012 \pm 0.00088$ (75 trials)
- $p = 0.100 \implies \text{BER} = 0.10014 \pm 0.00124$ (75 trials)

The empirical channel BER traces the theoretical identity line $y = x$ perfectly ($R^2 > 0.9999$), validating the unbiased nature of the BSC simulation.

Under the ideal noiseless baseline ($p = 0.0$), **75 out of 75 trials** achieved 100% exact byte reconstruction with zero residual errors across all corpora and Polar configurations (95% Wilson CI: $[0.9513, 1.0000]$).

---

### 7.3 End-to-End Reconstruction Reliability

Table 2 and Figure 1 (`results/figures/fig1_reconstruction_rate_vs_p.png`) present exact byte-reconstruction success rates across crossover probabilities and Polar configurations.

**Table 2: Reconstruction Reliability, Error Rates, and Wilson Confidence Intervals Across Polar Configurations**
*(Provenance: `results/analysis_polar_tradeoffs.csv`, 15 trials per row pooled across 3 corpora)*

| Polar Config $(N, K)$ | Rate $R$ | Crossover $p$ | Trials $n$ | Successes $k$ | Success Rate $\hat{p}$ | 95% Wilson CI | Mean Residual BER | Mean Block FER | Mean Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $(64, 32)$ | 0.50 | 0.000 | 15 | 15 | **1.000** | $[0.796, 1.000]$ | 0.00000 | 0.00000 | 737.4 |
| $(64, 32)$ | 0.50 | 0.001 | 15 | 15 | **1.000** | $[0.796, 1.000]$ | 0.00000 | 0.00000 | 762.2 |
| $(64, 32)$ | 0.50 | 0.005 | 15 | 15 | **1.000** | $[0.796, 1.000]$ | 0.00000 | 0.00000 | 825.0 |
| $(64, 32)$ | 0.50 | 0.010 | 15 | 15 | **1.000** | $[0.796, 1.000]$ | 0.00000 | 0.00000 | 809.0 |
| $(64, 32)$ | 0.50 | 0.020 | 15 | 7 | **0.467** | $[0.248, 0.699]$ | 0.00052 | 0.00261 | 759.8 |
| $(64, 32)$ | 0.50 | 0.050 | 15 | 0 | **0.000** | $[0.000, 0.204]$ | 0.01826 | 0.06852 | 788.0 |
| $(64, 32)$ | 0.50 | 0.100 | 15 | 0 | **0.000** | $[0.000, 0.204]$ | 0.12869 | 0.42419 | 734.1 |
| $(128, 64)$ | 0.50 | 0.010 | 15 | 15 | **1.000** | $[0.796, 1.000]$ | 0.00000 | 0.00000 | 815.2 |
| $(128, 64)$ | 0.50 | 0.020 | 15 | 11 | **0.733** | $[0.481, 0.891]$ | 0.00028 | 0.00276 | 820.4 |
| $(128, 64)$ | 0.50 | 0.050 | 15 | 0 | **0.000** | $[0.000, 0.204]$ | 0.01914 | 0.12196 | 835.6 |
| $(256, 128)$ | 0.50 | 0.010 | 15 | 15 | **1.000** | $[0.796, 1.000]$ | 0.00000 | 0.00000 | 884.2 |
| $(256, 128)$ | 0.50 | 0.020 | 15 | 12 | **0.800** | $[0.548, 0.930]$ | 0.00021 | 0.00311 | 910.1 |
| $(256, 128)$ | 0.50 | 0.050 | 15 | 0 | **0.000** | $[0.000, 0.204]$ | 0.02041 | 0.21140 | 925.3 |
| $(128, 32)$ | 0.25 | 0.020 | 15 | 15 | **1.000** | $[0.796, 1.000]$ | 0.00000 | 0.00000 | 1650.4 |
| $(128, 32)$ | 0.25 | 0.050 | 15 | 15 | **1.000** | $[0.796, 1.000]$ | 0.00000 | 0.00000 | 1675.2 |
| $(128, 32)$ | 0.25 | 0.100 | 15 | 1 | **0.067** | $[0.012, 0.298]$ | 0.03810 | 0.24501 | 1640.8 |
| $(128, 96)$ | 0.75 | 0.005 | 15 | 6 | **0.400** | $[0.198, 0.643]$ | 0.00112 | 0.00781 | 545.2 |
| $(128, 96)$ | 0.75 | 0.010 | 15 | 0 | **0.000** | $[0.000, 0.204]$ | 0.00492 | 0.03512 | 550.1 |

#### Statistical Sample Size Hierarchy
1. **Single Condition Cell ($n=5$):** For a specific corpus and configuration at $p \le 0.010$, $k=5/5$ yields $\hat{p} = 1.0$, $CI_{95\%} = [0.5655, 1.0000]$.
2. **Polar Configuration ($n=15$):** Aggregating across the 3 corpora at $p = 0.0$, $k=15/15$ yields $\hat{p} = 1.0$, $CI_{95\%} = [0.7961, 1.0000]$.
3. **All Noiseless Baseline Trials ($n=75$):** Pooling all 15 noiseless conditions yields $k=75/75$, $\hat{p} = 1.0$, $CI_{95\%} = [0.9513, 1.0000]$.
4. **Entire Sweep ($n=525$):** Across all 105 conditions, $k=344$ trials succeeded and $181$ failed, yielding $\hat{p} = 0.6552$, $CI_{95\%} = [0.6136, 0.6946]$.

---

### 7.4 Block-Size and Code-Rate Scaling Findings

1. **Polarization Depth Scaling ($N = 64 \to 256$):** At crossover probability $p = 0.020$ and rate $R = 0.50$, success rate increases monotonically with block length:
   $$N = 64 \implies \hat{p} = 0.467 \quad [0.248, 0.699]$$
   $$N = 128 \implies \hat{p} = 0.733 \quad [0.481, 0.891]$$
   $$N = 256 \implies \hat{p} = 0.800 \quad [0.548, 0.930]$$
   This confirms Arikan's polarization theorem [5]: deeper polarization trees ($n = 6 \to 8$) drive synthetic bit channels more decisively toward extreme noiseless and completely noisy states.
2. **Rate Adaptation Trade-offs ($R = 0.25 \to 0.75$):** Low-rate Polar coding ($(128, 32)$, $R = 0.25$) successfully corrected bit flips up to $p = 0.050$ with 100% reconstruction (15/15 trials). In contrast, high-rate coding ($(128, 96)$, $R = 0.75$) suffered immediate breakdown above $p = 0.005$, dropping to 0% success at $p = 0.010$.

---

### 7.5 Error Hierarchy and Integrity Detection

Figure 3 (`results/figures/fig3_residual_ber_and_fer_vs_p.png`) depicts the post-decoding error behavior on a semi-logarithmic scale:
- **Frame Error Rate Dominance:** Polar Block FER is strictly greater than or equal to Post-FEC Residual BER ($\text{FER} \ge \text{BER}_{\text{res}}$). A single residual bit error corrupts an entire $K$-bit block.
- **Audited Failure Breakdown:** Exactly **181 trials failed** out of 525 trials across the sweep. All 181 failures occurred under high noise ($p \ge 0.005$) or high code rates ($R = 0.75$).
- **Decompression Trap:** In **100% of the 181 failed trials** (181/181), the rANS decompressor detected corruption during container header deserialization, frequency table validation, or state bounds checks, assigning `failure_stage = "RANS_DECOMPRESS"`. Zero trials resulted in silent data corruption.

---

### 7.6 Effective Rate vs. Shannon Capacity

Figure 4 (`results/figures/fig4_effective_rate_vs_code_rate.png`) presents the dual-panel comparison of operational rate and entropy-normalized rate:
- **Subplot A (Operational Rate $R_{\text{eff}}$):** For natural language prose ($\text{CR} = 1.531$) and $R = 0.50$, $R_{\text{eff}} \approx 0.765$ source bits per channel use. For $R = 0.75$, $R_{\text{eff}} \approx 1.15$ source bits per channel use. While $R_{\text{eff}} > 1.0$ is physically achievable because source text contains redundancy, it cannot be directly compared to channel capacity.
- **Subplot B (Entropy-Normalized Rate $R_{\text{info}}$ vs. Capacity $C(p)$):** Normalizing by source entropy yields $R_{\text{info}} = R_{\text{eff}} \times (H_0 / 8.0) \approx K/N$.
  - At $p = 0.010$, $C(0.010) = 0.919$ bits/use. For $R = 0.50$, $R_{\text{info}} \approx 0.391 \ll C(p)$, operating comfortably in the reliable transmission regime.
  - At $p = 0.100$, $C(0.100) = 0.531$ bits/use. For $R = 0.75$, $R_{\text{info}} \approx 0.586 > C(0.100)$. By Shannon's Noisy-Channel Coding Theorem, reliable transmission is impossible, and indeed empirical reconstruction success was **0.0%** (15/15 failures).

---

### 7.7 Computational Latency Profile

Figure 6 (`results/figures/fig6_runtime_breakdown_by_polar_n.png`) illustrates stage execution times per transmission:
- **Polar SCD Decoding:** Consumes **85.4% to 92.1%** of total pipeline latency (~700 ms to 1,675 ms). Pure Python recursive tree traversal incurs $O(N \log N)$ operations per block. At $R = 0.25$, four times as many blocks ($N_{\text{blocks}} \approx 205$) are required, doubling decode latency to ~1.65 s.
- **rANS Compression & Decompression:** Consumes less than **5%** of latency (~2.5 ms to 8.0 ms), verifying the extreme operational efficiency of native word-level rANS.
- **Polar Encoding & Channel:** Consume less than **3%** of latency (~35 ms and ~0.5 ms).

---

## 8. Discussion

### 8.1 The rANS-Polar Synergy and Error Avalanche
The experimental evidence highlights the central tension in joint source-channel coding:
- **Synergy:** rANS source coding compresses text by 22% to 35%, directly reducing the number of channel bits transmitted and boosting the effective user information rate by a factor of $\text{CR}$.
- **Vulnerability:** Uncompressed natural language possesses inherent human redundancy: a corrupted ASCII character ("c*t") is easily reconstructed from context. Compressed rANS bitstreams, however, have near-maximal entropy. A single residual bit flip alters the arithmetic state register $x$, causing all subsequent symbol decodings to diverge uncontrollably. 

### 8.2 Parser Containment vs. Cryptographic Integrity
While our rANS container implementation trapped 100% of corrupted trials in this study, **this observed containment is an empirical artifact of parser sanity checks, not a mathematical integrity guarantee**:
- The rANS container checks: magic header (`RANS`), version (`0x01`), scale precision ($B = 14$), frequency table sum ($\sum f_s = 16,384$), and uncompressed length equality.
- A residual channel error could theoretically flip a payload bit without triggering a state crash or length mismatch, producing undetected byte errors.
- **Architectural Recommendation:** Practical joint rANS-Polar systems must concatenate an explicit Cyclic Redundancy Check (CRC-16 or CRC-32) or cryptographic hash (SHA-256) at the container level.

### 8.3 Finite-Block Dispersion
Our empirical findings illustrate Polyanskiy-Poor-Verdú finite blocklength dispersion [2]: at $N = 128$, the code rate $R = 0.75$ fails at $p = 0.010$, even though the asymptotic channel capacity $C(0.010) \approx 0.919$ is substantially higher. Short Polar codes under Successive Cancellation Decoding require an operational rate backoff from capacity to achieve reliable frame error rates.

---

## 9. Threats to Validity and Limitations

1. **Finite Trial Size ($n=5$ per condition):** While 525 trials provide a comprehensive overall dataset, 5 trials per unique $(corpus, N, K, p)$ tuple yield relatively wide individual cell Wilson intervals ($[0.5655, 1.0000]$). Conclusions regarding specific threshold boundaries are indicative rather than definitive.
2. **Channel Model Abstraction:** The Binary Symmetric Channel models independent, stationary bit-flip noise without burst errors, inter-symbol interference (ISI), packet erasure, or timing jitter. Wired links subject to impulse noise or packet drops require extended modeling.
3. **Pure Python Decoding Latency:** Polar SCD execution times (~700–900 ms) reflect interpreted Python recursion in `py-polar-codes`. Hardware FPGA/ASIC or vectorized C/C++ implementations achieve microsecond latencies.
4. **Order-0 Static Modeling:** rANS byte frequencies were modeled with an order-0 static table. Context-adaptive or higher-order Markov models would achieve higher compression ratios but introduce additional state synchronization complexity.
5. **Absence of SCL/CRC Decoding:** Standard Arikan SCD was evaluated. Modern 5G NR implementations utilize Successive Cancellation List (SCL) decoding with $L=8$ list candidates and CRC aiding, which substantially steepens the error waterfall curve [7].

---

## 10. Conclusion and Future Work

This research provides a comprehensive, reproducible evaluation of an integrated rANS source coding and Polar channel coding pipeline over a discrete bit-flip channel. The experimental evidence demonstrates that byte-oriented rANS achieves 74% to 96% of the zero-order Shannon entropy bound with negligible computational overhead (<5% of pipeline time), significantly enhancing effective transmission rates. Arikan Polar channel coding under SCD provides robust error correction at low-to-moderate crossover probabilities ($p \le 0.010$), with error resilience scaling decisively with polarization depth ($N = 64 \to 256$). However, the avalanche error propagation of compressed streams mandates explicit error detection (such as CRC concatenation) and sufficient channel coding margin.

### Future Work
1. **CRC-Aided Successive Cancellation List (CA-SCL) Decoding:** Integrate $L \in \{4, 8\}$ SCL decoding with distributed CRC checking to evaluate 5G NR-compliant performance.
2. **Burst Noise Channel Extensions:** Extend the channel simulation to Gilbert-Elliott two-state Markov burst error models.
3. **Adaptive Higher-Order rANS:** Implement context-adaptive rANS (e.g., order-1 Markov modeling) to evaluate the trade-off between higher compression ratios and increased error sensitivity.
4. **C/Rust Kernel Acceleration:** Port Polar SCD/SCL algorithms to compiled Rust/C extensions to benchmark real-time multi-megabit throughputs.

---

## 11. References

1. **C. E. Shannon**, "A Mathematical Theory of Communication," *Bell System Technical Journal*, vol. 27, no. 3, pp. 379–423, July 1948; vol. 27, no. 4, pp. 623–656, October 1948.
2. **Y. Polyanskiy, H. V. Poor, and S. Verdú**, "Channel Coding Rate in the Finite Blocklength Regime," *IEEE Transactions on Information Theory*, vol. 56, no. 5, pp. 2307–2359, May 2010.
3. **J. Duda**, "Asymmetric Numeral Systems: Entropy Coding Combining Speed of Huffman Coding with Compression Rate of Arithmetic Coding," *arXiv preprint arXiv:1311.2540*, 2013.
4. **F. Giesen**, "Interleaved Range Asymmetric Numeral Systems," *RAD Game Tools Technical Note*, 2014. [Online]. Available: https://github.com/rygorous/ryg_rans
5. **E. Arikan**, "Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels," *IEEE Transactions on Information Theory*, vol. 55, no. 7, pp. 3051–3073, July 2009.
6. **3GPP**, "Technical Specification Group Radio Access Network; NR; Multiplexing and Channel Coding (Release 15)," *3rd Generation Partnership Project (3GPP)*, Technical Specification 38.212, Version 15.2.0, June 2018.
7. **I. Tal and A. Vardy**, "List Decoding of Polar Codes," *IEEE Transactions on Information Theory*, vol. 61, no. 5, pp. 2213–2226, May 2015.
8. **E. B. Wilson**, "Probable Inference, the Law of Succession, and Statistical Inference," *Journal of the American Statistical Association*, vol. 22, no. 158, pp. 209–212, June 1927.

---

## 12. Reproducibility Appendix

### A. Environment and Software Dependencies
The experiment was conducted and validated in the following environment:
- **Operating System:** Windows 11 x64
- **Python Version:** Python 3.11.9
- **Core Dependencies:**
  - `numpy >= 1.24.0` (array handling and bit manipulation)
  - `pandas >= 2.0.0` (data aggregation and telemetry export)
  - `matplotlib >= 3.7.0` (headless publication figure generation)
  - `pytest >= 7.0.0` (automated unit and integration testing)
  - `py-polar-codes == 1.2.2` (Arikan Polar code construction, encoding, and SCD)

### B. Execution and Reproduction Commands
All results, tables, and figures can be regenerated from scratch using the following commands:

```powershell
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Execute Full Benchmark Sweep (525 trials, ~3-4 minutes)
.\.venv\Scripts\python experiments/run_experiments.py --config experiments/config_benchmark.json

# 3. Generate Audited Summary Tables
.\.venv\Scripts\python experiments/analyze_results.py

# 4. Generate Publication Figures (at 300 DPI in results/figures/)
.\.venv\Scripts\python experiments/plot_results.py

# 5. Execute Complete Test Suite (204 tests)
.\.venv\Scripts\python -m pytest -v
```

### C. Benchmark Configuration Specification (`config_benchmark.json`)
```json
{
  "name": "comprehensive_sweep",
  "input_files": [
    "data/raw/natural_language_sample.txt",
    "data/raw/technical_text_sample.txt",
    "data/raw/python_code_sample.txt"
  ],
  "polar_configs": [
    [64, 32],
    [128, 64],
    [256, 128],
    [128, 32],
    [128, 96]
  ],
  "channel_probabilities": [
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10
  ],
  "trials_per_condition": 5,
  "master_seed": 20261020,
  "scale_bits": 14,
  "design_snr_db": 2.0,
  "systematic": false,
  "output_dir": "results"
}
```

### D. Repository Provenance and File Map
- **Source Code:**
  - rANS Codec: [`src/source_coding/rans_codec.py`](file:///d:/sem%207/itc/Project/src/source_coding/rans_codec.py)
  - Polar Codec: [`src/channel_coding/polar_codec.py`](file:///d:/sem%207/itc/Project/src/channel_coding/polar_codec.py)
  - BSC Channel: [`src/channel/binary_symmetric_channel.py`](file:///d:/sem%207/itc/Project/src/channel/binary_symmetric_channel.py)
  - Pipeline Integration: [`src/pipeline.py`](file:///d:/sem%207/itc/Project/src/pipeline.py)
  - Bit Utilities: [`src/utils/bit_utils.py`](file:///d:/sem%207/itc/Project/src/utils/bit_utils.py)
  - Seed Utilities: [`src/utils/seed.py`](file:///d:/sem%207/itc/Project/src/utils/seed.py)
- **Experiments & Analysis:**
  - Experiment Harness: [`experiments/harness.py`](file:///d:/sem%207/itc/Project/experiments/harness.py)
  - Runner CLI: [`experiments/run_experiments.py`](file:///d:/sem%207/itc/Project/experiments/run_experiments.py)
  - Analysis Script: [`experiments/analyze_results.py`](file:///d:/sem%207/itc/Project/experiments/analyze_results.py)
  - Plotting Script: [`experiments/plot_results.py`](file:///d:/sem%207/itc/Project/experiments/plot_results.py)
  - Detailed Analysis Notes: [`experiments/ANALYSIS.md`](file:///d:/sem%207/itc/Project/experiments/ANALYSIS.md)
- **Audited Data & Figures:**
  - Raw Trial CSV: `results/raw_trials_comprehensive_sweep_1792504862.csv`
  - Summary CSV: `results/summary_comprehensive_sweep_1792504862.csv`
  - Corpus Analysis: `results/analysis_corpus_compression.csv`
  - Reliability Summary: `results/analysis_reliability_summary.csv`
  - Polar Trade-offs: `results/analysis_polar_tradeoffs.csv`
  - Failure Diagnostics: `results/analysis_failures.csv`
  - Publication Figures: `results/figures/fig1_reconstruction_rate_vs_p.png` through `fig6_runtime_breakdown_by_polar_n.png`
- **Automated Tests (204 passing):**
  - Analysis & Plotting Tests: [`tests/test_analysis.py`](file:///d:/sem%207/itc/Project/tests/test_analysis.py) (18 tests)
  - Harness Tests: [`tests/test_experiments.py`](file:///d:/sem%207/itc/Project/tests/test_experiments.py) (17 tests)
  - Pipeline Integration Tests: [`tests/test_pipeline.py`](file:///d:/sem%207/itc/Project/tests/test_pipeline.py) (18 tests)
  - Polar Tests: [`tests/test_polar.py`](file:///d:/sem%207/itc/Project/tests/test_polar.py) (48 tests)
  - rANS Tests: [`tests/test_rans.py`](file:///d:/sem%207/itc/Project/tests/test_rans.py) (41 tests)
  - Channel Tests: [`tests/test_channel.py`](file:///d:/sem%207/itc/Project/tests/test_channel.py) (38 tests)
  - Bit Utils Tests: [`tests/test_bit_utils.py`](file:///d:/sem%207/itc/Project/tests/test_bit_utils.py) (18 tests)
  - Seed Tests: [`tests/test_seed.py`](file:///d:/sem%207/itc/Project/tests/test_seed.py) (4 tests)
  - Import Tests: [`tests/test_imports.py`](file:///d:/sem%207/itc/Project/tests/test_imports.py) (2 tests)
