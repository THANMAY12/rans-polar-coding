"""
Wired Binary Symmetric Channel (BSC) Simulation Model without Packet Loss.

Mathematical Formulation & Physical Abstraction:
------------------------------------------------
This module provides a physical-layer abstraction of a point-to-point wired
communication channel operating under two deterministic or controlled regimes:
1. Ideal Mode: Error-free transmission (crossover probability p = 0).
2. Controlled Bit-Flip Mode: Binary Symmetric Channel (BSC) with crossover
   probability p in (0, 0.5].

Channel Properties:
- In-Order Delivery: Bits are transmitted and received in exact sequential order.
- Zero Packet Loss: No bit dropping, truncation, insertion, or reordering occurs.
  len(received_bits) == len(transmitted_bits) strictly.
- Memoryless Property: Each bit flip occurs independently via a Bernoulli(p) process.

Log-Likelihood Ratio (LLR) Derivation:
--------------------------------------
For a memoryless BSC with crossover probability p:
    P(y = 0 | x = 0) = 1 - p,    P(y = 0 | x = 1) = p
    P(y = 1 | x = 0) = p,        P(y = 1 | x = 1) = 1 - p

The Log-Likelihood Ratio for received bit y in {0, 1} is:
    LLR(y) = ln( P(y | x = 0) / P(y | x = 1) )

Case 1: Received y = 0
    LLR(0) = ln( (1 - p) / p ) > 0  (for 0 < p < 0.5)

Case 2: Received y = 1
    LLR(1) = ln( p / (1 - p) ) = -ln( (1 - p) / p ) < 0

Unified formula:
    LLR(y) = (1 - 2y) * ln( (1 - p) / p )

Boundary Conditions:
- Ideal / Noiseless (p = 0.0):
  ln((1-p)/p) = ln(1/0) = +infinity.
  To prevent floating-point overflow or NaN propagation in numerical decoders,
  a configurable finite magnitude `noiseless_llr` (default: 100.0) is assigned:
      LLR(0) = +noiseless_llr,   LLR(1) = -noiseless_llr
- Fully Random Channel (p = 0.5):
  ln((1 - 0.5) / 0.5) = ln(1.0) = 0.0.
  Both bit 0 and bit 1 yield LLR = 0.0 (maximum uncertainty, mutual information I(X; Y) = 0).
- Invalid range: p < 0.0 or p > 0.5 is rejected with InvalidChannelParameterError.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Union
import numpy as np

from src.utils.seed import get_rng


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class ChannelError(Exception):
    """Base exception for all channel simulation errors."""
    pass


class InvalidChannelParameterError(ChannelError, ValueError):
    """Raised when channel parameters (e.g. crossover probability p) are invalid."""
    pass


class InvalidChannelInputError(ChannelError, ValueError):
    """Raised when input bitstreams violate binary or dimension constraints."""
    pass


# ---------------------------------------------------------------------------
# Transmission Result Dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelResult:
    """
    Immutable telemetry record of a channel transmission.

    Attributes
    ----------
    transmitted_bits : np.ndarray
        The input bitstream sent into the channel (dtype uint8).
    received_bits : np.ndarray
        The output bitstream received from the channel (dtype uint8).
    flip_mask : np.ndarray
        Boolean array where True indicates a bit flip (transmitted != received).
    error_count : int
        Total number of bit errors introduced during transmission.
    empirical_ber : float
        Observed Bit Error Rate: error_count / total_bits (0.0 if empty).
    llrs : np.ndarray
        Computed Log-Likelihood Ratios for the received bits (dtype float64).
    p : float
        Channel crossover probability parameter used for simulation.
    """
    transmitted_bits: np.ndarray
    received_bits: np.ndarray
    flip_mask: np.ndarray
    error_count: int
    empirical_ber: float
    llrs: np.ndarray
    p: float

    @property
    def total_bits(self) -> int:
        return self.transmitted_bits.size


# ---------------------------------------------------------------------------
# Binary Symmetric Channel Implementation
# ---------------------------------------------------------------------------
class BinarySymmetricChannel:
    """
    Configurable Binary Symmetric Channel (BSC) with independent bit flips.

    Parameters
    ----------
    p : float, default=0.0
        Bit flip probability (crossover probability). Must be in [0.0, 0.5].
        p = 0.0 corresponds to an ideal noiseless channel.
    seed : int, optional
        Seed for the internal NumPy random generator to ensure reproducibility.
    rng : np.random.Generator, optional
        Injected isolated NumPy Generator. If provided, `seed` is ignored.
    noiseless_llr : float, default=100.0
        Magnitude of soft LLR used when p = 0.0 (+100 for bit 0, -100 for bit 1).
    """

    def __init__(
        self,
        p: float = 0.0,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        noiseless_llr: float = 100.0,
    ) -> None:
        self._validate_p(p)
        if noiseless_llr <= 0:
            raise InvalidChannelParameterError(
                f"noiseless_llr must be strictly positive, got {noiseless_llr}"
            )

        self._p = float(p)
        self._noiseless_llr = float(noiseless_llr)

        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
            self._rng = rng
        else:
            self._rng = get_rng(seed)

    @staticmethod
    def _validate_p(p: float) -> None:
        """Ensure crossover probability p is within supported physical bounds [0.0, 0.5]."""
        if not isinstance(p, (int, float, np.floating, np.integer)):
            raise InvalidChannelParameterError(f"Crossover probability p must be numeric, got {type(p).__name__}")
        if math.isnan(p):
            raise InvalidChannelParameterError("Crossover probability p cannot be NaN")
        if p < 0.0 or p > 0.5:
            raise InvalidChannelParameterError(
                f"Crossover probability p must be in the range [0.0, 0.5], got {p}"
            )

    @property
    def p(self) -> float:
        """Crossover probability of the channel."""
        return self._p

    @p.setter
    def p(self, new_p: float) -> None:
        self._validate_p(new_p)
        self._p = float(new_p)

    @property
    def is_ideal(self) -> bool:
        """True if channel is in ideal error-free mode (p == 0.0)."""
        return self._p == 0.0

    @property
    def noiseless_llr(self) -> float:
        """LLR magnitude used for noiseless transmissions."""
        return self._noiseless_llr

    @property
    def rng(self) -> np.random.Generator:
        """Underlying random number generator."""
        return self._rng

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the internal generator seed for deterministic simulation reruns."""
        self._rng = get_rng(seed)

    def compute_llr(self, received_bits: Union[np.ndarray, Sequence[int]]) -> np.ndarray:
        """
        Compute theoretical soft Log-Likelihood Ratios for received binary bits.

        LLR Convention:
            LLR(y) = ln( P(y | x = 0) / P(y | x = 1) )
            LLR > 0 indicates bit 0 is more probable.
            LLR < 0 indicates bit 1 is more probable.

        Parameters
        ----------
        received_bits : np.ndarray or Sequence[int]
            1D array of binary bits {0, 1}.

        Returns
        -------
        np.ndarray
            1D array of float64 LLR values matching input length.
        """
        arr = np.asarray(received_bits, dtype=np.uint8)
        if arr.ndim != 1:
            raise InvalidChannelInputError(f"Received bits must be 1D, got ndim={arr.ndim}")

        if arr.size == 0:
            return np.empty(0, dtype=np.float64)

        if not np.all((arr == 0) | (arr == 1)):
            raise InvalidChannelInputError("Received bits must contain only binary values 0 and 1")

        # Boundary Case 1: p = 0.0 (Noiseless)
        if self._p == 0.0:
            return np.where(arr == 0, self._noiseless_llr, -self._noiseless_llr).astype(np.float64)

        # Boundary Case 2: p = 0.5 (Maximum uncertainty / zero information)
        if self._p == 0.5:
            return np.zeros(arr.size, dtype=np.float64)

        # General Case: 0 < p < 0.5
        # LLR = (1 - 2*y) * ln((1 - p) / p)
        llr_weight = math.log((1.0 - self._p) / self._p)
        return np.where(arr == 0, llr_weight, -llr_weight).astype(np.float64)

    def transmit(
        self, bits: Union[np.ndarray, Sequence[int]]
    ) -> ChannelResult:
        """
        Transmit a binary bitstream through the channel, introducing independent bit flips.

        Guarantees:
        - len(received_bits) == len(transmitted_bits)
        - Zero packet loss, dropping, or reordering
        - Independent Bernoulli(p) bit flips

        Parameters
        ----------
        bits : np.ndarray or Sequence[int]
            1D array of binary bits {0, 1} to transmit.

        Returns
        -------
        ChannelResult
            Detailed telemetry object containing transmitted, received bits,
            flip mask, error counts, empirical BER, and soft LLRs.
        """
        tx = np.asarray(bits, dtype=np.uint8)
        if tx.ndim != 1:
            raise InvalidChannelInputError(f"Transmitted bits must be 1D, got ndim={tx.ndim}")

        if tx.size == 0:
            return ChannelResult(
                transmitted_bits=np.empty(0, dtype=np.uint8),
                received_bits=np.empty(0, dtype=np.uint8),
                flip_mask=np.empty(0, dtype=bool),
                error_count=0,
                empirical_ber=0.0,
                llrs=np.empty(0, dtype=np.float64),
                p=self._p,
            )

        if not np.all((tx == 0) | (tx == 1)):
            raise InvalidChannelInputError("Transmitted bits must contain only binary values 0 and 1")

        # Error-free mode optimization
        if self._p == 0.0:
            rx = tx.copy()
            flip_mask = np.zeros(tx.size, dtype=bool)
            error_count = 0
            ber = 0.0
        else:
            # Generate independent Bernoulli(p) flips
            random_draws = self._rng.random(tx.size)
            flip_mask = random_draws < self._p
            # Flip bits where flip_mask is True: rx = tx ^ flip_mask
            rx = (tx ^ flip_mask.astype(np.uint8))
            error_count = int(np.count_nonzero(flip_mask))
            ber = float(error_count) / float(tx.size)

        llrs = self.compute_llr(rx)

        return ChannelResult(
            transmitted_bits=tx.copy(),
            received_bits=rx,
            flip_mask=flip_mask,
            error_count=error_count,
            empirical_ber=ber,
            llrs=llrs,
            p=self._p,
        )


# Semantic alias
WiredChannel = BinarySymmetricChannel
