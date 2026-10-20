"""
Test module imports for internal packages and verified dependencies.
"""

import pytest


def test_internal_module_imports():
    """Verify that all core project modules import without error."""
    import src
    import src.channel
    import src.channel_coding
    import src.source_coding
    import src.utils
    import src.utils.bit_utils
    import src.utils.seed

    assert src.__version__ == "0.1.0"
    assert hasattr(src.source_coding, "RansEncoder")
    assert hasattr(src.source_coding, "RansDecoder")
    assert hasattr(src.source_coding, "rans_compress")
    assert hasattr(src.source_coding, "rans_decompress")
    assert hasattr(src.channel_coding, "PolarCodec")
    assert hasattr(src.channel_coding, "PolarConfig")
    assert hasattr(src.channel_coding, "bits_to_llr")
    assert hasattr(src.channel_coding, "llr_to_hard_bits")


def test_external_dependency_imports():
    """Verify that required external dependencies are installed and importable."""
    import matplotlib
    import numpy as np
    import pandas as pd
    import polarcodes

    assert hasattr(polarcodes, "PolarCode")
    assert hasattr(polarcodes, "Construct")
    assert hasattr(polarcodes, "Encode")
    assert hasattr(polarcodes, "Decode")
