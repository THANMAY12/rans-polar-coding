"""
Channel simulation models package.

Models:
- BinarySymmetricChannel (WiredChannel): No-packet-loss wired channel with
  ideal error-free mode (p = 0) and controlled bit-flip simulation (0 < p <= 0.5).
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .binary_symmetric_channel import (
        BinarySymmetricChannel,
        ChannelError,
        ChannelResult,
        InvalidChannelInputError,
        InvalidChannelParameterError,
        WiredChannel,
    )

__all__ = [
    "BinarySymmetricChannel",
    "WiredChannel",
    "ChannelResult",
    "ChannelError",
    "InvalidChannelParameterError",
    "InvalidChannelInputError",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import binary_symmetric_channel
        return getattr(binary_symmetric_channel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
