"""
Deterministic random-seed configuration and management utilities for experiments.
"""

from contextlib import contextmanager
import os
import random
from typing import Generator, Optional
import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set seeds for Python standard library random and NumPy global RNG.

    Parameters
    ----------
    seed : int
        The integer seed value.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    seed_val = int(seed)
    random.seed(seed_val)
    np.random.seed(seed_val)
    os.environ["PYTHONHASHSEED"] = str(seed_val)


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Construct an isolated NumPy Generator for deterministic local operations.

    Using Generator instances instead of mutating global RNG state ensures
    thread-safe, reproducible, and isolated experiment execution.

    Parameters
    ----------
    seed : int, optional
        Seed for the Generator. If None, default entropy is used.

    Returns
    -------
    np.random.Generator
        Isolated NumPy Generator instance.
    """
    return np.random.default_rng(seed)


@contextmanager
def seed_context(seed: int) -> Generator[np.random.Generator, None, None]:
    """
    Context manager to execute a code block with deterministic RNG states,
    restoring previous RNG states on exit.

    Parameters
    ----------
    seed : int
        Deterministic seed for the duration of the context.

    Yields
    ------
    np.random.Generator
        A seeded NumPy generator scoped to the context block.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()

    set_global_seed(seed)
    rng = get_rng(seed)

    try:
        yield rng
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
