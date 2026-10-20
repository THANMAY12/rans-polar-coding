"""
Source coding package: byte-oriented range Asymmetric Numeral Systems (rANS).

Strict exclusion: Shannon-Fano, Shannon-Fano-Elias, Huffman, Minimum-variance Huffman, LZW.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .rans_codec import (
        CompressionError,
        CorruptedDataError,
        DecompressionError,
        DEFAULT_SCALE_BITS,
        FrequencyTable,
        RANS_L,
        RANS_MAGIC,
        RANS_VERSION,
        RansContainer,
        RansDecoder,
        RansEncoder,
        RansError,
        rans_compress,
        rans_decompress,
        run_rans_demo,
    )

__all__ = [
    "RansEncoder",
    "RansDecoder",
    "FrequencyTable",
    "RansContainer",
    "rans_compress",
    "rans_decompress",
    "run_rans_demo",
    "RansError",
    "CompressionError",
    "DecompressionError",
    "CorruptedDataError",
    "DEFAULT_SCALE_BITS",
    "RANS_MAGIC",
    "RANS_VERSION",
    "RANS_L",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import rans_codec
        return getattr(rans_codec, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
