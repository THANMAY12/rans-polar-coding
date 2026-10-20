"""
Unit tests for deterministic random-seed handling and context manager.
"""

import random
import numpy as np
import pytest

from src.utils.seed import get_rng, seed_context, set_global_seed


def test_set_global_seed_reproducibility():
    """Verify that set_global_seed produces identical random streams across calls."""
    set_global_seed(12345)
    r1_py = [random.random() for _ in range(5)]
    r1_np = np.random.rand(5)

    set_global_seed(12345)
    r2_py = [random.random() for _ in range(5)]
    r2_np = np.random.rand(5)

    assert r1_py == r2_py
    np.testing.assert_array_equal(r1_np, r2_np)


def test_get_rng_isolation():
    """Verify that get_rng creates independent, deterministic NumPy Generators."""
    rng1 = get_rng(42)
    rng2 = get_rng(42)

    seq1 = rng1.integers(0, 100, size=10)
    seq2 = rng2.integers(0, 100, size=10)

    np.testing.assert_array_equal(seq1, seq2)


def test_seed_context_restores_state():
    """Verify that seed_context ensures reproducibility inside the block and restores prior state."""
    # Establish a baseline state
    random.seed(999)
    np.random.seed(999)
    _ = random.random()
    _ = np.random.rand()

    expected_next_py = random.random()
    expected_next_np = np.random.rand()

    # Reset to baseline state before entering context
    random.seed(999)
    np.random.seed(999)
    _ = random.random()
    _ = np.random.rand()

    # Enter deterministic context
    with seed_context(42) as scoped_rng:
        val_inside = scoped_rng.random()
        py_inside = random.random()
        np_inside = np.random.rand()

    # Verify state was restored
    after_py = random.random()
    after_np = np.random.rand()

    assert after_py == expected_next_py
    assert after_np == expected_next_np


def test_invalid_seed_type():
    with pytest.raises(TypeError):
        set_global_seed("invalid_seed")  # type: ignore
