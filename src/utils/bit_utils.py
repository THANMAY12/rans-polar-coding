"""
Bit packing and unpacking utilities for source-to-channel bitstream manipulation.
"""

from typing import Sequence, Tuple, Union
import numpy as np


def bytes_to_bits(data: Union[bytes, bytearray], msb_first: bool = True) -> np.ndarray:
    """
    Convert a byte sequence into a 1D numpy array of bits (0 or 1, dtype uint8).

    Parameters
    ----------
    data : bytes or bytearray
        Input byte stream (e.g., UTF-8 encoded text bytes or compressed bytes).
    msb_first : bool, default=True
        If True, bit 7 (MSB) comes first in each byte (standard network order).
        If False, bit 0 (LSB) comes first.

    Returns
    -------
    np.ndarray
        1D array of uint8 values (0 or 1), length = len(data) * 8.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(data).__name__}")

    if len(data) == 0:
        return np.empty(0, dtype=np.uint8)

    byte_array = np.frombuffer(data, dtype=np.uint8)
    bitorder = "big" if msb_first else "little"
    bits = np.unpackbits(byte_array, bitorder=bitorder)
    return bits.astype(np.uint8)


def bits_to_bytes(bits: Union[np.ndarray, Sequence[int]], msb_first: bool = True) -> bytes:
    """
    Convert a sequence of bits (0s and 1s) into a byte string.

    Parameters
    ----------
    bits : np.ndarray or Sequence[int]
        Input 1D sequence of bits. Length must be a multiple of 8.
    msb_first : bool, default=True
        If True, bits are packed MSB-first. If False, LSB-first.

    Returns
    -------
    bytes
        Packed bytes representation.

    Raises
    ------
    ValueError
        If bit sequence length is not a multiple of 8 or contains values other than 0 and 1.
    """
    arr = np.asarray(bits, dtype=np.uint8)

    if arr.ndim != 1:
        raise ValueError(f"Bit array must be 1D, got ndim={arr.ndim}")

    if arr.size == 0:
        return b""

    if arr.size % 8 != 0:
        raise ValueError(
            f"Bit array length ({arr.size}) must be an exact multiple of 8 to form bytes. "
            f"Use unpad_bits or pad_bits before packing."
        )

    # Validate binary values
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("Bit array must contain only 0s and 1s.")

    bitorder = "big" if msb_first else "little"
    packed = np.packbits(arr, bitorder=bitorder)
    return packed.tobytes()


def pad_bits(
    bits: Union[np.ndarray, Sequence[int]], block_size: int, pad_value: int = 0
) -> Tuple[np.ndarray, int]:
    """
    Pad a 1D bit array to a multiple of `block_size`.

    Parameters
    ----------
    bits : np.ndarray or Sequence[int]
        Input 1D bit array.
    block_size : int
        Target block size multiple (e.g., Polar code information block length K).
    pad_value : int, default=0
        Value for pad bits (0 or 1).

    Returns
    -------
    Tuple[np.ndarray, int]
        (padded_bits, pad_count) where pad_count is the number of added bits.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if pad_value not in (0, 1):
        raise ValueError(f"pad_value must be 0 or 1, got {pad_value}")

    arr = np.asarray(bits, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"Bit array must be 1D, got ndim={arr.ndim}")

    remainder = arr.size % block_size
    if remainder == 0:
        return arr.copy(), 0

    pad_count = block_size - remainder
    padding = np.full(pad_count, pad_value, dtype=np.uint8)
    padded = np.concatenate([arr, padding])
    return padded, pad_count


def unpad_bits(bits: Union[np.ndarray, Sequence[int]], pad_count: int) -> np.ndarray:
    """
    Remove padding bits from the end of a bit array.

    Parameters
    ----------
    bits : np.ndarray or Sequence[int]
        Input padded 1D bit array.
    pad_count : int
        Number of pad bits to remove from the tail.

    Returns
    -------
    np.ndarray
        Unpadded 1D bit array.
    """
    if pad_count < 0:
        raise ValueError(f"pad_count cannot be negative, got {pad_count}")

    arr = np.asarray(bits, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"Bit array must be 1D, got ndim={arr.ndim}")

    if pad_count == 0:
        return arr.copy()

    if pad_count > arr.size:
        raise ValueError(
            f"Cannot remove {pad_count} pad bits from an array of size {arr.size}"
        )

    return arr[:-pad_count].copy()
