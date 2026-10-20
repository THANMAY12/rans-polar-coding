"""
Command-line demonstration script for rANS source compression and decompression.

Usage:
    python experiments/demonstrate_rans.py <path_to_text_file>
    python experiments/demonstrate_rans.py --sample
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path for direct invocation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.source_coding.rans_codec import run_rans_demo


SAMPLE_TEXT = """Information Theory and Coding Research Project:
End-to-End Evaluation of rANS Source Coding and Polar Channel Coding for Reliable Text Transmission over a No-Packet-Loss Wired Channel Model.

Range Asymmetric Numeral Systems (rANS) represents a state-of-the-art breakthrough in entropy coding.
Unlike Huffman coding, which is constrained to assign an integer number of bits to each symbol, rANS operates on continuous frequency intervals, achieving compression rates virtually indistinguishable from the theoretical Shannon entropy bound H(X).

At the same time, rANS avoids the bit-level serialization bottleneck of traditional arithmetic coding by performing state transitions and renormalization directly on native machine words (bytes), achieving orders of magnitude faster throughput.

This pipeline operates on raw UTF-8 text bytes without lossy preprocessing, preserving full Unicode fidelity:
Mathematical notation: sum_{i=1}^n P(x_i) log_2 (1 / P(x_i)) <= L(C) < H(X) + 1
Multilingual test: "L'entropie d'une source d'information caractérise son contenu informationnel."
Non-ASCII characters: Übergrößenträger, déjà-vu, こんにちは, 🚀, 📡, 🔬.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate rANS compression, decompression, SHA-256 verification, and performance profiling."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to an input text or binary file to compress and verify.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use a built-in multi-domain UTF-8 technical benchmark sample.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to save the compressed container.",
    )

    args = parser.parse_args()

    if args.sample or args.input_file is None:
        sample_path = Path("data/raw/sample_itc_corpus.txt")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(SAMPLE_TEXT * 10, encoding="utf-8")
        target_file = sample_path
        print(f"Using sample benchmark corpus: {sample_path} ({sample_path.stat().st_size:,} bytes)")
    else:
        target_file = Path(args.input_file)

    if not target_file.exists():
        print(f"Error: file not found: {target_file}", file=sys.stderr)
        sys.exit(1)

    result = run_rans_demo(target_file, output_path=args.output, verbose=True)

    if not result["exact_reconstruction"]:
        print("ERROR: Reconstruction failed!", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
