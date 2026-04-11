#!/usr/bin/env python3
"""
Generate 10 randomized input sequences from 22 controller inputs.

Requirements implemented:
- 22 unique inputs total
- 100 copies of each input overall
- 2200 inputs total
- Split into 10 lists of 220 inputs each

Inputs:
  A, B, X, Y, LB, RB,
  Left-[Left, Right, Up, Down, LeftUp, RightUp, LeftDown, RightDown],
  Right-[Left, Right, Up, Down, LeftUp, RightUp, LeftDown, RightDown]

Usage:
  python3 generate_input_sequences.py
  python3 generate_input_sequences.py --seed 42
  python3 generate_input_sequences.py --output sequences.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List


def get_inputs() -> List[str]:
    buttons = ["A", "B", "X", "Y", "LB", "RB"]
    directions = [
        "Left",
        "Right",
        "Up",
        "Down",
        "LeftUp",
        "RightUp",
        "LeftDown",
        "RightDown",
    ]
    left_stick = [f"Left-{direction}" for direction in directions]
    right_stick = [f"Right-{direction}" for direction in directions]
    return buttons + left_stick + right_stick


def generate_sequences(
    num_sequences: int = 10,
    sequence_length: int = 220,
    copies_per_input: int = 100,
    seed: int | None = None,
) -> List[List[str]]:
    inputs = get_inputs()
    total_needed = num_sequences * sequence_length
    total_available = len(inputs) * copies_per_input

    if total_available != total_needed:
        raise ValueError(
            f"Configuration mismatch: {len(inputs)} inputs x {copies_per_input} copies "
            f"= {total_available}, but {num_sequences} sequences x {sequence_length} "
            f"= {total_needed}."
        )

    rng = random.Random(seed)
    pool = []
    for item in inputs:
        pool.extend([item] * copies_per_input)

    rng.shuffle(pool)
    sequences = [
        pool[i * sequence_length : (i + 1) * sequence_length]
        for i in range(num_sequences)
    ]

    validate_sequences(
        sequences=sequences,
        expected_inputs=inputs,
        expected_total_per_input=copies_per_input,
        expected_num_sequences=num_sequences,
        expected_sequence_length=sequence_length,
    )

    return sequences


def validate_sequences(
    sequences: List[List[str]],
    expected_inputs: List[str],
    expected_total_per_input: int,
    expected_num_sequences: int,
    expected_sequence_length: int,
) -> None:
    if len(sequences) != expected_num_sequences:
        raise ValueError(
            f"Expected {expected_num_sequences} sequences, got {len(sequences)}."
        )

    for i, sequence in enumerate(sequences, start=1):
        if len(sequence) != expected_sequence_length:
            raise ValueError(
                f"Sequence {i} has length {len(sequence)}; "
                f"expected {expected_sequence_length}."
            )

    counts = Counter(item for sequence in sequences for item in sequence)

    for item in expected_inputs:
        if counts[item] != expected_total_per_input:
            raise ValueError(
                f"Input {item!r} appears {counts[item]} times; "
                f"expected {expected_total_per_input}."
            )


def save_sequences(sequences: List[List[str]], output_path: Path) -> None:
    payload = {
        "metadata": {
            "num_unique_inputs": len(get_inputs()),
            "copies_per_input": 100,
            "num_sequences": len(sequences),
            "sequence_length": len(sequences[0]) if sequences else 0,
            "total_inputs": sum(len(seq) for seq in sequences),
        },
        "inputs": get_inputs(),
        "sequences": [
            {"sequence_id": i + 1, "inputs": sequence}
            for i, sequence in enumerate(sequences)
        ],
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(sequences: List[List[str]]) -> None:
    counts = Counter(item for sequence in sequences for item in sequence)
    print(f"Generated {len(sequences)} sequences.")
    print(f"Each sequence length: {len(sequences[0]) if sequences else 0}")
    print(f"Total inputs: {sum(len(seq) for seq in sequences)}")
    print("\nOverall counts:")
    for item in get_inputs():
        print(f"  {item:15s} {counts[item]}")

    print("\nFirst 10 items of each sequence:")
    for i, sequence in enumerate(sequences, start=1):
        preview = ", ".join(sequence[:10])
        print(f"  Sequence {i:2d}: {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 10 randomized sequences of controller inputs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sequences.json"),
        help="Where to save the generated sequences JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = generate_sequences(seed=args.seed)
    save_sequences(sequences, args.output)
    print_summary(sequences)
    print(f"\nSaved sequences to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
