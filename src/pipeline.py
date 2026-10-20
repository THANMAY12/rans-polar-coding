"""
End-to-End Joint Source-Channel Transmission Pipeline.

Integrates:
1. Source Coding: range Asymmetric Numeral Systems (rANS) byte-oriented compression.
2. Channel Coding: 5G Arikan Polar Coding with Successive Cancellation Decoding (SCD).
3. Physical Channel Model: Wired Binary Symmetric Channel (BSC) with no packet loss.

Transmitter Architecture:
-------------------------
[Input Text Bytes] -> [rANS Source Encoder] -> [Validated rANS Container Bytes]
                   -> [Bit Serialization (MSB-first)] -> [K-bit Block Partitioning & Padding]
                   -> [Polar Encoder (K -> N)] -> [Transmitted Codewords] -> [BSC Channel]

Receiver Architecture:
----------------------
[Channel LLRs / Bits] -> [Polar SCD Decoder (N -> K)] -> [Padding Removal (pad_count)]
                     -> [Bit-to-Byte Deserialization] -> [rANS Container Parsing]
                     -> [rANS Source Decoder] -> [Reconstructed Text Bytes]

Framing & Metadata:
--------------------
To decouple physical channel data transmission from control-plane signaling (as in 5G NR
PDCCH/DCI control channels), framing parameters (container byte length, padding count,
block dimensions K and N) are maintained in a structured `TransmissionFrameMetadata` object
labeled explicitly as out-of-band metadata.
"""

from dataclasses import dataclass
import time
from typing import Optional, Sequence, Tuple, Union
import numpy as np

from src.channel.binary_symmetric_channel import BinarySymmetricChannel, ChannelResult
from src.channel_coding.polar_codec import PolarCodec, PolarConfig
from src.source_coding.rans_codec import (
    CorruptedDataError,
    DEFAULT_SCALE_BITS,
    DecompressionError,
    RansError,
    rans_compress,
    rans_decompress,
)
from src.utils.bit_utils import bits_to_bytes, bytes_to_bits, pad_bits, unpad_bits


# ---------------------------------------------------------------------------
# Pipeline Exceptions
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """Base exception for end-to-end pipeline errors."""
    pass


class PipelineEncodingError(PipelineError):
    """Raised when transmitter-side encoding fails."""
    pass


class PipelineDecodingError(PipelineError):
    """Raised when receiver-side decoding fails."""
    pass


class MetadataMismatchError(PipelineError, ValueError):
    """Raised when out-of-band metadata is inconsistent with received stream."""
    pass


# ---------------------------------------------------------------------------
# Out-of-Band Transmission Metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransmissionFrameMetadata:
    """
    Out-of-band transmission metadata preserving framing parameters.

    In real communication systems (e.g. 5G NR), transport block size, rate-matching
    parameters, and padding counts are signaled out-of-band over heavily coded control
    channels (PDCCH). This dataclass explicitly isolates these signaling parameters.

    Attributes
    ----------
    container_byte_length : int
        Exact byte length of the compressed rANS container.
    container_bit_length : int
        Exact bit length of the container (container_byte_length * 8).
    pad_count : int
        Number of zero-padding bits added to align container bits with Polar block size K.
    K : int
        Polar code information block length.
    N : int
        Polar code codeword block length.
    num_polar_blocks : int
        Total number of Polar code blocks: (container_bit_length + pad_count) // K.
    total_codeword_bits : int
        Total number of transmitted channel codeword bits: num_polar_blocks * N.
    channel_p : float
        Simulated channel crossover probability.
    """
    container_byte_length: int
    container_bit_length: int
    pad_count: int
    K: int
    N: int
    num_polar_blocks: int
    total_codeword_bits: int
    channel_p: float

    def validate(self) -> None:
        """Validate internal consistency of framing parameters."""
        if self.container_byte_length < 0:
            raise MetadataMismatchError(
                f"container_byte_length cannot be negative, got {self.container_byte_length}"
            )
        if self.container_bit_length != self.container_byte_length * 8:
            raise MetadataMismatchError(
                f"container_bit_length ({self.container_bit_length}) != container_byte_length * 8 "
                f"({self.container_byte_length * 8})"
            )
        if self.K <= 0 or self.N <= 0 or self.K >= self.N:
            raise MetadataMismatchError(f"Invalid Polar block dimensions K={self.K}, N={self.N}")
        if self.pad_count < 0 or (self.container_bit_length > 0 and self.pad_count >= self.K):
            raise MetadataMismatchError(
                f"pad_count ({self.pad_count}) must be in [0, K-1]=[0, {self.K-1}]"
            )
        expected_blocks = (
            (self.container_bit_length + self.pad_count) // self.K
            if self.container_bit_length > 0 else 0
        )
        if self.num_polar_blocks != expected_blocks:
            raise MetadataMismatchError(
                f"num_polar_blocks ({self.num_polar_blocks}) does not match expected ({expected_blocks})"
            )
        if self.total_codeword_bits != self.num_polar_blocks * self.N:
            raise MetadataMismatchError(
                f"total_codeword_bits ({self.total_codeword_bits}) != num_polar_blocks * N "
                f"({self.num_polar_blocks * self.N})"
            )


# ---------------------------------------------------------------------------
# Pipeline Result & Telemetry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineResult:
    """
    Complete telemetry record of an end-to-end transmission trial.
    """
    # Size metrics
    original_size_bytes: int
    container_size_bytes: int
    source_compression_ratio: float
    source_bits: int
    padded_bits: int
    pad_count: int
    num_polar_blocks: int
    transmitted_codeword_bits: int

    # Channel & Coding metrics
    code_rate: float
    effective_transmission_rate: float
    channel_p: float
    channel_bit_flips: int
    channel_ber: float
    polar_residual_bit_errors: int
    polar_residual_ber: float
    polar_block_error_count: int
    polar_block_error_rate: float

    # Outcome
    exact_reconstruction: bool
    success: bool
    reconstructed_bytes: Optional[bytes]
    failure_stage: Optional[str]
    error_message: Optional[str]
    metadata: TransmissionFrameMetadata

    # Timing profiling (milliseconds)
    rans_encode_time_ms: float
    polar_encode_time_ms: float
    channel_time_ms: float
    polar_decode_time_ms: float
    rans_decode_time_ms: float
    total_latency_ms: float


# ---------------------------------------------------------------------------
# End-to-End Pipeline Implementation
# ---------------------------------------------------------------------------
class EndToEndPipeline:
    """
    Orchestrator for rANS Source Coding, Polar Channel Coding, and BSC Transmission.

    Parameters
    ----------
    polar_config : Optional[PolarConfig]
        Configuration for the Polar codec (N, K, design SNR, etc.).
    scale_bits : int, default=14
        Precision scale bits B for rANS frequency table normalization.
    noiseless_llr : float, default=100.0
        LLR magnitude used for noiseless channel representation.
    """

    def __init__(
        self,
        polar_config: Optional[PolarConfig] = None,
        scale_bits: int = DEFAULT_SCALE_BITS,
        noiseless_llr: float = 100.0,
    ) -> None:
        self.polar_config = polar_config or PolarConfig(N=128, K=64)
        self.polar_codec = PolarCodec(config=self.polar_config)
        self.scale_bits = scale_bits
        self.noiseless_llr = noiseless_llr

    # -----------------------------------------------------------------------
    # Transmitter
    # -----------------------------------------------------------------------
    def transmit_prepare(
        self,
        data: Union[bytes, bytearray],
        channel_p: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, TransmissionFrameMetadata, float, float]:
        """
        Transmitter pipeline:
        1. rANS compress input bytes into a validated binary container.
        2. Convert container bytes into a 1D binary bit sequence (MSB-first).
        3. Partition container bits into K-bit Polar blocks with zero-padding.
        4. Polar-encode each block into N-bit codewords.
        5. Build out-of-band TransmissionFrameMetadata.

        Returns
        -------
        Tuple[transmitted_codewords, container_bits, metadata, rans_time_ms, polar_time_ms]
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Input data must be bytes or bytearray, got {type(data).__name__}")

        raw_bytes = bytes(data)

        # Step 1: rANS Source Compression
        t0 = time.perf_counter()
        container_bytes = rans_compress(raw_bytes, scale_bits=self.scale_bits)
        t1 = time.perf_counter()
        rans_enc_ms = (t1 - t0) * 1000.0

        container_size = len(container_bytes)

        # Step 2: Bit Serialization (MSB-first)
        container_bits = bytes_to_bits(container_bytes, msb_first=True)
        container_bit_len = container_bits.size

        # Step 3 & 4: Polar Encoding with K-bit block framing and padding
        t2 = time.perf_counter()
        codewords, pad_count = self.polar_codec.encode(container_bits)
        t3 = time.perf_counter()
        polar_enc_ms = (t3 - t2) * 1000.0

        num_blocks = codewords.size // self.polar_config.N if codewords.size > 0 else 0

        # Step 5: Construct Out-of-band Transmission Metadata
        metadata = TransmissionFrameMetadata(
            container_byte_length=container_size,
            container_bit_length=container_bit_len,
            pad_count=pad_count,
            K=self.polar_config.K,
            N=self.polar_config.N,
            num_polar_blocks=num_blocks,
            total_codeword_bits=codewords.size,
            channel_p=channel_p,
        )
        metadata.validate()

        return codewords, container_bits, metadata, rans_enc_ms, polar_enc_ms

    # -----------------------------------------------------------------------
    # Receiver
    # -----------------------------------------------------------------------
    def receive_process(
        self,
        llrs: np.ndarray,
        metadata: TransmissionFrameMetadata,
    ) -> Tuple[Optional[bytes], np.ndarray, Optional[str], Optional[str], float, float]:
        """
        Receiver pipeline:
        1. Validate received LLR length against metadata.
        2. Decode Polar blocks using Successive Cancellation Decoding (SCD).
        3. Remove framing padding bits (pad_count) to isolate container bits.
        4. Convert decoded container bits back to container bytes (MSB-first).
        5. Decompress container bytes using rANS decoder.

        Returns
        -------
        Tuple[reconstructed_bytes, decoded_container_bits, failure_stage, error_message, polar_dec_ms, rans_dec_ms]
        """
        metadata.validate()

        if llrs.size != metadata.total_codeword_bits:
            raise MetadataMismatchError(
                f"Received LLR stream length ({llrs.size}) != metadata total_codeword_bits ({metadata.total_codeword_bits})"
            )

        if metadata.container_byte_length == 0:
            return b"", np.empty(0, dtype=np.uint8), None, None, 0.0, 0.0

        # Step 2 & 3: Polar Decoding and Padding Removal
        t0 = time.perf_counter()
        try:
            decoded_container_bits = self.polar_codec.decode(
                llrs, pad_count=metadata.pad_count
            )
        except Exception as e:
            t1 = time.perf_counter()
            return None, np.empty(0, dtype=np.uint8), "POLAR_DECODE", str(e), (t1 - t0) * 1000.0, 0.0

        t1 = time.perf_counter()
        polar_dec_ms = (t1 - t0) * 1000.0

        if decoded_container_bits.size != metadata.container_bit_length:
            return (
                None,
                decoded_container_bits,
                "POLAR_DECODE",
                f"Decoded bit length ({decoded_container_bits.size}) != metadata bit length ({metadata.container_bit_length})",
                polar_dec_ms,
                0.0,
            )

        # Step 4: Bit-to-Byte Deserialization
        try:
            reconstructed_container = bits_to_bytes(decoded_container_bits, msb_first=True)
        except Exception as e:
            return None, decoded_container_bits, "CONTAINER_INTEGRITY", str(e), polar_dec_ms, 0.0

        # Step 5: rANS Source Decompression
        t2 = time.perf_counter()
        try:
            reconstructed_bytes = rans_decompress(reconstructed_container)
            t3 = time.perf_counter()
            rans_dec_ms = (t3 - t2) * 1000.0
            return reconstructed_bytes, decoded_container_bits, None, None, polar_dec_ms, rans_dec_ms
        except (RansError, DecompressionError, CorruptedDataError, Exception) as e:
            t3 = time.perf_counter()
            rans_dec_ms = (t3 - t2) * 1000.0
            return None, decoded_container_bits, "RANS_DECOMPRESS", str(e), polar_dec_ms, rans_dec_ms

    # -----------------------------------------------------------------------
    # End-to-End Simulation Run
    # -----------------------------------------------------------------------
    def run(
        self,
        data: Union[bytes, bytearray],
        p: float = 0.0,
        seed: Optional[int] = None,
    ) -> PipelineResult:
        """
        Execute full end-to-end simulation: Transmitter -> BSC Channel -> Receiver.

        Parameters
        ----------
        data : bytes or bytearray
            Input source bytes to transmit.
        p : float, default=0.0
            Binary Symmetric Channel bit-flip probability.
        seed : int, optional
            Random seed for channel simulation reproducibility.

        Returns
        -------
        PipelineResult
            Comprehensive telemetry and verification result.
        """
        t_start = time.perf_counter()
        orig_bytes = bytes(data)
        orig_size = len(orig_bytes)

        # 1. Transmitter
        (
            codewords,
            tx_container_bits,
            metadata,
            rans_enc_ms,
            polar_enc_ms,
        ) = self.transmit_prepare(orig_bytes, channel_p=p)

        # 2. Channel Simulation
        channel = BinarySymmetricChannel(
            p=p,
            seed=seed,
            noiseless_llr=self.noiseless_llr,
        )
        t_ch0 = time.perf_counter()
        channel_result = channel.transmit(codewords)
        t_ch1 = time.perf_counter()
        channel_ms = (t_ch1 - t_ch0) * 1000.0

        # 3. Receiver
        (
            reconstructed_bytes,
            rx_container_bits,
            failure_stage,
            error_msg,
            polar_dec_ms,
            rans_dec_ms,
        ) = self.receive_process(channel_result.llrs, metadata)

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0

        # 4. Telemetry Computations
        exact_match = (
            reconstructed_bytes is not None and reconstructed_bytes == orig_bytes
        )
        success = exact_match and (failure_stage is None)

        if not success and failure_stage is None:
            failure_stage = "BYTE_MISMATCH"
            error_msg = "Reconstructed bytes do not match original bytes"

        # Residual Post-FEC bit errors in the container bitstream
        if rx_container_bits.size == tx_container_bits.size:
            polar_residual_errors = int(np.count_nonzero(tx_container_bits != rx_container_bits))
            polar_residual_ber = (
                polar_residual_errors / tx_container_bits.size if tx_container_bits.size > 0 else 0.0
            )
        else:
            polar_residual_errors = tx_container_bits.size
            polar_residual_ber = 1.0

        # Block error estimation (how many Polar blocks had residual errors)
        block_err_count = 0
        if metadata.num_polar_blocks > 0 and rx_container_bits.size == tx_container_bits.size:
            K = metadata.K
            pad = metadata.pad_count
            # Compare padded blocks
            tx_pad, _ = pad_bits(tx_container_bits, block_size=K, pad_value=0)
            rx_pad, _ = pad_bits(rx_container_bits, block_size=K, pad_value=0)
            for b in range(metadata.num_polar_blocks):
                if not np.array_equal(
                    tx_pad[b * K : (b + 1) * K], rx_pad[b * K : (b + 1) * K]
                ):
                    block_err_count += 1
            block_fer = block_err_count / metadata.num_polar_blocks
        else:
            block_fer = 1.0 if not success else 0.0

        # Rates
        cr = (orig_size / metadata.container_byte_length) if metadata.container_byte_length > 0 else 0.0
        code_rate = self.polar_config.code_rate
        eff_rate = (
            (orig_size * 8.0) / metadata.total_codeword_bits
            if metadata.total_codeword_bits > 0 else 0.0
        )

        return PipelineResult(
            original_size_bytes=orig_size,
            container_size_bytes=metadata.container_byte_length,
            source_compression_ratio=cr,
            source_bits=metadata.container_bit_length,
            padded_bits=metadata.container_bit_length + metadata.pad_count,
            pad_count=metadata.pad_count,
            num_polar_blocks=metadata.num_polar_blocks,
            transmitted_codeword_bits=metadata.total_codeword_bits,
            code_rate=code_rate,
            effective_transmission_rate=eff_rate,
            channel_p=p,
            channel_bit_flips=channel_result.error_count,
            channel_ber=channel_result.empirical_ber,
            polar_residual_bit_errors=polar_residual_errors,
            polar_residual_ber=polar_residual_ber,
            polar_block_error_count=block_err_count,
            polar_block_error_rate=block_fer,
            exact_reconstruction=exact_match,
            success=success,
            reconstructed_bytes=reconstructed_bytes,
            failure_stage=failure_stage,
            error_message=error_msg,
            metadata=metadata,
            rans_encode_time_ms=rans_enc_ms,
            polar_encode_time_ms=polar_enc_ms,
            channel_time_ms=channel_ms,
            polar_decode_time_ms=polar_dec_ms,
            rans_decode_time_ms=rans_dec_ms,
            total_latency_ms=total_latency_ms,
        )
