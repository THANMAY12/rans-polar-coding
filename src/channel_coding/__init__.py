"""
Channel coding package: 5G Arikan Polar Codes and Successive Cancellation Decoding.

Strict exclusion: Low-Density Parity-Check (LDPC) codes.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .polar_codec import (
        PolarCodec,
        PolarConfig,
        PolarDecodingError,
        PolarEncodingError,
        PolarError,
        PolarConfigurationError,
        bits_to_llr,
        llr_to_hard_bits,
    )

__all__ = [
    "PolarCodec",
    "PolarConfig",
    "PolarError",
    "PolarConfigurationError",
    "PolarEncodingError",
    "PolarDecodingError",
    "bits_to_llr",
    "llr_to_hard_bits",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import polar_codec
        return getattr(polar_codec, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
