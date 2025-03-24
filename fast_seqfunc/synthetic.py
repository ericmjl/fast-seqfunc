"""Synthetic sequence-function data for testing and benchmarking.

This module provides functions to generate synthetic sequence-function data
with controllable properties and varying levels of complexity for testing
models and algorithms.
"""

import random
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


def generate_random_sequences(
    length: int = 20,
    count: int = 500,
    alphabet: str = "ACGT",
    fixed_length: bool = True,
    length_range: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Generate random sequences with the given properties.

    :param length: Length of each sequence (if fixed_length=True)
    :param count: Number of sequences to generate
    :param alphabet: Characters to use in the sequences
    :param fixed_length: Whether all sequences should have the same length
    :param length_range: Range of lengths (min, max) if fixed_length=False
    :return: List of randomly generated sequences
    """
    sequences = []

    if not fixed_length and length_range is not None:
        min_length, max_length = length_range
    else:
        min_length = max_length = length

    for _ in range(count):
        if fixed_length:
            seq_length = length
        else:
            seq_length = random.randint(min_length, max_length)

        sequence = "".join(random.choice(alphabet) for _ in range(seq_length))
        sequences.append(sequence)

    return sequences


def count_matches(sequence: str, pattern: str) -> int:
    """Count non-overlapping occurrences of a pattern in a sequence.

    :param sequence: Input sequence
    :param pattern: Pattern to search for
    :return: Count of pattern occurrences
    """
    count = 0
    pos = 0

    while True:
        pos = sequence.find(pattern, pos)
        if pos == -1:
            break
        count += 1
        pos += len(pattern)

    return count


def create_gc_content_task(
    count: int = 500,
    length: int = 30,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target is the GC content of DNA sequences.

    This is a simple linear task.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their GC content
    """
    sequences = generate_random_sequences(
        length=length, count=count, alphabet="ACGT", fixed_length=True
    )

    # Calculate GC content
    targets = [
        (sequence.count("G") + sequence.count("C")) / len(sequence)
        for sequence in sequences
    ]

    # Add noise if specified
    if noise_level > 0:
        targets = [t + np.random.normal(0, noise_level) for t in targets]

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_g_count_task(
    count: int = 500,
    length: int = 30,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target is the count of G in DNA sequences.

    This is a simple linear task.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their G count
    """
    sequences = generate_random_sequences(
        length=length, count=count, alphabet="ACGT", fixed_length=True
    )

    # Count G's
    targets = [sequence.count("G") for sequence in sequences]

    # Add noise if specified
    if noise_level > 0:
        targets = [t + np.random.normal(0, noise_level) for t in targets]

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_motif_position_task(
    count: int = 500,
    length: int = 50,
    motif: str = "GATA",
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends on the position of a motif.

    This is a nonlinear task where the position of a motif determines the function.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param motif: Motif to insert
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet="ACGT", fixed_length=True
    )

    # Insert motif at random positions in some sequences
    targets = []
    for i in range(count):
        if random.random() < 0.7:  # 70% chance to have the motif
            pos = random.randint(0, length - len(motif))
            seq_list = list(sequences[i])
            seq_list[pos : pos + len(motif)] = motif
            sequences[i] = "".join(seq_list)

            # Function depends on position (nonlinear transformation)
            norm_pos = pos / (length - len(motif))  # Normalize position to 0-1
            target = np.sin(norm_pos * np.pi) * 5  # Sinusoidal function of position
        else:
            target = 0.0

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_motif_count_task(
    count: int = 500,
    length: int = 50,
    motifs: List[str] = None,
    weights: List[float] = None,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends on the count of multiple motifs.

    This is a linear task with multiple features.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param motifs: List of motifs to count
    :param weights: Weight for each motif's contribution
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    if motifs is None:
        motifs = ["AT", "GC", "TG", "CA"]

    if weights is None:
        weights = [1.0, -0.5, 2.0, -1.5]

    if len(motifs) != len(weights):
        raise ValueError("Length of motifs and weights must match")

    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet="ACGT", fixed_length=True
    )

    # Calculate target based on motif counts
    targets = []
    for sequence in sequences:
        target = 0.0
        for motif, weight in zip(motifs, weights):
            count = sequence.count(motif)
            target += count * weight

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_length_dependent_task(
    count: int = 500,
    min_length: int = 20,
    max_length: int = 50,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends on sequence length.

    This tests the model's ability to handle variable-length sequences.

    :param count: Number of sequences to generate
    :param min_length: Minimum sequence length
    :param max_length: Maximum sequence length
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    # Generate random sequences of varying length
    sequences = generate_random_sequences(
        count=count,
        alphabet="ACGT",
        fixed_length=False,
        length_range=(min_length, max_length),
    )

    # Calculate target based on sequence length (nonlinear)
    targets = []
    for sequence in sequences:
        length = len(sequence)
        norm_length = (length - min_length) / (
            max_length - min_length
        )  # Normalize to 0-1
        target = np.log(1 + norm_length * 10)  # Logarithmic function of length

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_nonlinear_composition_task(
    count: int = 500,
    length: int = 30,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends nonlinearly on base composition.

    This task requires nonlinear models to solve effectively.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet="ACGT", fixed_length=True
    )

    # Calculate target based on nonlinear combination of base counts
    targets = []
    for sequence in sequences:
        a_count = sequence.count("A") / length
        c_count = sequence.count("C") / length
        g_count = sequence.count("G") / length
        t_count = sequence.count("T") / length

        # Nonlinear function of base composition
        target = (a_count * g_count) / (0.1 + c_count * t_count) * 10

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_interaction_task(
    count: int = 500,
    length: int = 40,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends on interactions between positions.

    This task tests the model's ability to capture position dependencies.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet="ACGT", fixed_length=True
    )

    # Calculate target based on interactions between positions
    targets = []
    for sequence in sequences:
        target = 0.0

        # Look for specific pairs with a gap between them
        for i in range(length - 6):
            if sequence[i] == "A" and sequence[i + 5] == "T":
                target += 1.0
            if sequence[i] == "G" and sequence[i + 5] == "C":
                target += 1.5

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_classification_task(
    count: int = 500,
    length: int = 30,
    noise_level: float = 0.1,
) -> pd.DataFrame:
    """Create a binary classification dataset based on sequence patterns.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param noise_level: Probability of label flipping for noise
    :return: DataFrame with sequences and their class labels
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet="ACGT", fixed_length=True
    )

    # Define patterns for positive class
    positive_patterns = ["GATA", "TATA", "CAAT"]

    # Assign classes based on pattern presence
    labels = []
    for sequence in sequences:
        has_pattern = any(pattern in sequence for pattern in positive_patterns)
        label = 1 if has_pattern else 0

        # Add noise by flipping some labels
        if random.random() < noise_level:
            label = 1 - label  # Flip the label

        labels.append(label)

    return pd.DataFrame({"sequence": sequences, "function": labels})


def create_multiclass_task(
    count: int = 500,
    length: int = 30,
    noise_level: float = 0.1,
) -> pd.DataFrame:
    """Create a multi-class classification dataset based on sequence patterns.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param noise_level: Probability of label incorrect assignment for noise
    :return: DataFrame with sequences and their class labels
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet="ACGT", fixed_length=True
    )

    # Define patterns for different classes
    class_patterns = {
        0: ["AAAA", "TTTT"],  # Class 0 patterns
        1: ["GGGG", "CCCC"],  # Class 1 patterns
        2: ["GATA", "TATA"],  # Class 2 patterns
        3: ["CAAT", "ATTG"],  # Class 3 patterns
    }

    # Assign classes based on pattern presence
    labels = []
    for sequence in sequences:
        # Determine class based on patterns
        class_label = 0  # Default class
        for cls, patterns in class_patterns.items():
            if any(pattern in sequence for pattern in patterns):
                class_label = cls
                break

        # Add noise by randomly reassigning some classes
        if random.random() < noise_level:
            # Assign to a random class different from the current one
            other_classes = [c for c in class_patterns.keys() if c != class_label]
            class_label = random.choice(other_classes)

        labels.append(class_label)

    return pd.DataFrame({"sequence": sequences, "function": labels})


def generate_dataset_by_task(
    task: Literal[
        "g_count",
        "gc_content",
        "motif_position",
        "motif_count",
        "length_dependent",
        "nonlinear_composition",
        "interaction",
        "classification",
        "multiclass",
    ],
    count: int = 500,
    noise_level: float = 0.1,
    **kwargs,
) -> pd.DataFrame:
    """Generate a dataset for a specific sequence-function task.

    :param task: Name of the task to generate
    :param count: Number of sequences to generate
    :param noise_level: Level of noise to add
    :param kwargs: Additional parameters for specific tasks
    :return: DataFrame with sequences and their function values
    """
    task_functions = {
        "g_count": create_g_count_task,
        "gc_content": create_gc_content_task,
        "motif_position": create_motif_position_task,
        "motif_count": create_motif_count_task,
        "length_dependent": create_length_dependent_task,
        "nonlinear_composition": create_nonlinear_composition_task,
        "interaction": create_interaction_task,
        "classification": create_classification_task,
        "multiclass": create_multiclass_task,
    }

    if task not in task_functions:
        raise ValueError(
            f"Unknown task: {task}. Available tasks: {list(task_functions.keys())}"
        )

    return task_functions[task](count=count, noise_level=noise_level, **kwargs)
