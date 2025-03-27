"""Synthetic sequence-function data for testing and benchmarking.

This module provides functions to generate synthetic sequence-function data
with controllable properties and varying levels of complexity for testing
models and algorithms.
"""

import random
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fast_seqfunc.alphabets import Alphabet


def generate_random_sequences(
    length: int = 20,
    count: int = 500,
    alphabet: Union[str, Alphabet] = "ACGT",
    fixed_length: bool = True,
    length_range: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Generate random sequences with the given properties.

    :param length: Length of each sequence (if fixed_length=True)
    :param count: Number of sequences to generate
    :param alphabet: Characters to use in the sequences or an Alphabet instance
    :param fixed_length: Whether all sequences should have the same length
    :param length_range: Range of lengths (min, max) if fixed_length=False
    :return: List of randomly generated sequences
    """
    sequences = []

    if not fixed_length and length_range is not None:
        min_length, max_length = length_range
    else:
        min_length = max_length = length

    # Handle different alphabet types
    if isinstance(alphabet, Alphabet):
        tokens = alphabet.tokens
        delimiter = alphabet.delimiter
        # Filter out the gap character
        tokens = [t for t in tokens if t != alphabet.gap_character]
    else:
        tokens = list(alphabet)
        delimiter = None

    for _ in range(count):
        if fixed_length:
            seq_length = length
        else:
            seq_length = random.randint(min_length, max_length)

        # Generate a random sequence of tokens
        seq_tokens = [random.choice(tokens) for _ in range(seq_length)]

        # Convert to a string based on delimiter
        if delimiter is not None:
            sequence = delimiter.join(seq_tokens)
        else:
            sequence = "".join(seq_tokens)

        sequences.append(sequence)

    return sequences


def generate_integer_sequences(
    length: int = 5,
    count: int = 500,
    max_value: int = 9,
    fixed_length: bool = True,
    length_range: Optional[Tuple[int, int]] = None,
    delimiter: str = ",",
) -> List[str]:
    """Generate random sequences of comma-delimited integers.

    :param length: Length of each sequence (number of integers)
    :param count: Number of sequences to generate
    :param max_value: Maximum integer value (inclusive)
    :param fixed_length: Whether all sequences should have the same length
    :param length_range: Range of lengths (min, max) if fixed_length=False
    :param delimiter: Delimiter between integers (default: comma)
    :return: List of randomly generated integer sequences
    """
    # Create an integer alphabet
    alphabet = Alphabet.integer(max_value=max_value)

    # Override the delimiter if needed
    if delimiter != ",":
        alphabet.delimiter = delimiter

    # Generate sequences using the alphabet
    return generate_random_sequences(
        length=length,
        count=count,
        alphabet=alphabet,
        fixed_length=fixed_length,
        length_range=length_range,
    )


def generate_sequence_function_data(
    count: int = 500,
    sequence_length: int = 20,
    alphabet: Union[str, Alphabet] = "ACGT",
    function_type: Literal["linear", "nonlinear"] = "linear",
    noise_level: float = 0.1,
    classification: bool = False,
    num_classes: int = 2,
    fixed_length: bool = True,
    length_range: Optional[Tuple[int, int]] = None,
    position_weights: Optional[List[float]] = None,
    motif_effects: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Generate synthetic sequence-function data with controllable properties.

    :param count: Number of sequences to generate
    :param sequence_length: Length of each sequence
    :param alphabet: Characters to use in the sequences or an Alphabet instance
    :param function_type: Type of sequence-function relationship
    :param noise_level: Standard deviation of Gaussian noise to add
    :param classification: Whether to generate classification data
    :param num_classes: Number of classes for classification
    :param fixed_length: Whether all sequences should have the same length
    :param length_range: Range of lengths (min, max) if fixed_length=False
    :param position_weights: Optional weights for each position
    :param motif_effects: Optional dictionary mapping motifs to effect sizes
    :return: DataFrame with sequences and function values
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=sequence_length,
        count=count,
        alphabet=alphabet,
        fixed_length=fixed_length,
        length_range=length_range,
    )

    # Get alphabet tokens
    if isinstance(alphabet, Alphabet):
        tokens = alphabet.tokens
        # Filter out the gap character
        tokens = [t for t in tokens if t != alphabet.gap_character]
        delimiter = alphabet.delimiter
    else:
        tokens = list(alphabet)
        delimiter = None

    # Create mapping of tokens to numeric values (for linear model)
    token_values = {token: i / len(tokens) for i, token in enumerate(tokens)}

    # Generate function values based on sequences
    function_values = []
    for sequence in sequences:
        # Tokenize sequence
        if delimiter is not None:
            sequence_tokens = sequence.split(delimiter)
        else:
            sequence_tokens = list(sequence)

        # Apply position weights if provided
        if position_weights is not None:
            # Ensure weights match sequence length
            if len(position_weights) < len(sequence_tokens):
                # Extend weights with zeros
                weights = position_weights + [0] * (
                    len(sequence_tokens) - len(position_weights)
                )
            elif len(position_weights) > len(sequence_tokens):
                # Truncate weights
                weights = position_weights[: len(sequence_tokens)]
            else:
                weights = position_weights
        else:
            # Equal weights for all positions
            weights = [1 / len(sequence_tokens)] * len(sequence_tokens)

        # Calculate base function value
        if function_type == "linear":
            # Simple linear model: weighted sum of token values
            value = sum(
                token_values.get(token, 0) * weight
                for token, weight in zip(sequence_tokens, weights)
            )
        else:  # nonlinear
            # Nonlinear model: introduce interactions between positions
            value = 0
            for i in range(len(sequence_tokens) - 1):
                token1 = sequence_tokens[i]
                token2 = sequence_tokens[i + 1]
                # Interaction effect depends on both tokens
                interaction = token_values.get(token1, 0) * token_values.get(token2, 0)
                value += interaction * weights[i]

        # Add effects of specific motifs if provided
        if motif_effects is not None:
            joined_sequence = "".join(sequence_tokens)
            for motif, effect in motif_effects.items():
                if motif in joined_sequence:
                    value += effect

        # Add random noise
        value += np.random.normal(0, noise_level)

        # Store function value
        function_values.append(value)

    # Convert to classification if requested
    if classification:
        # Discretize function values into classes
        bins = np.linspace(min(function_values), max(function_values), num_classes + 1)
        class_values = np.digitize(function_values, bins[1:])
        df = pd.DataFrame({"sequence": sequences, "function": class_values})
    else:
        df = pd.DataFrame({"sequence": sequences, "function": function_values})

    return df


def generate_integer_function_data(
    count: int = 500,
    sequence_length: int = 5,
    max_value: int = 9,
    function_type: Literal["linear", "nonlinear"] = "linear",
    noise_level: float = 0.1,
    classification: bool = False,
    num_classes: int = 2,
    fixed_length: bool = True,
    length_range: Optional[Tuple[int, int]] = None,
    position_weights: Optional[List[float]] = None,
    delimiter: str = ",",
) -> pd.DataFrame:
    """Generate synthetic sequence-function data with comma-delimited integers.

    :param count: Number of sequences to generate
    :param sequence_length: Length of each sequence (number of integers)
    :param max_value: Maximum integer value (inclusive)
    :param function_type: Type of sequence-function relationship
    :param noise_level: Standard deviation of Gaussian noise to add
    :param classification: Whether to generate classification data
    :param num_classes: Number of classes for classification
    :param fixed_length: Whether all sequences should have the same length
    :param length_range: Range of lengths (min, max) if fixed_length=False
    :param position_weights: Optional weights for each position
    :param delimiter: Delimiter between integers (default: comma)
    :return: DataFrame with sequences and function values
    """
    # Create an integer alphabet
    alphabet = Alphabet.integer(max_value=max_value)

    # Override the delimiter if needed
    if delimiter != ",":
        alphabet.delimiter = delimiter

    # Generate sequence-function data using the alphabet
    return generate_sequence_function_data(
        count=count,
        sequence_length=sequence_length,
        alphabet=alphabet,
        function_type=function_type,
        noise_level=noise_level,
        classification=classification,
        num_classes=num_classes,
        fixed_length=fixed_length,
        length_range=length_range,
        position_weights=position_weights,
    )


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


#
# Generic task generators for any alphabet
#


def create_token_count_task(
    count: int = 500,
    length: int = 30,
    alphabet: Union[str, Alphabet] = "ACGT",
    token: str = "G",
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target is the count of a specific token in sequences.

    This is a generalization of the g_count task that works with any alphabet.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param alphabet: Characters to use in the sequences or an Alphabet instance
    :param token: The token to count
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their token count
    """
    sequences = generate_random_sequences(
        length=length, count=count, alphabet=alphabet, fixed_length=True
    )

    # If we have a delimited alphabet, we need to count tokens differently
    delimiter = alphabet.delimiter if isinstance(alphabet, Alphabet) else None

    # Calculate token counts
    targets = []
    for sequence in sequences:
        if delimiter:
            # Split by delimiter for multi-character tokens
            tokens = sequence.split(delimiter)
            target = tokens.count(token)
        else:
            # For character-level alphabets
            target = sequence.count(token)

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_content_ratio_task(
    count: int = 500,
    length: int = 30,
    alphabet: Union[str, Alphabet] = "ACGT",
    numerator_tokens: Optional[List[str]] = None,
    denominator_tokens: Optional[List[str]] = None,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target is the ratio of specific tokens in sequences.

    This is a generalization of the gc_content task that works with any alphabet.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param alphabet: Characters to use in the sequences or an Alphabet instance
    :param numerator_tokens: List of tokens for the numerator
        (default: ["G", "C"] for GC content)
    :param denominator_tokens: List of tokens for the denominator
        (default: all tokens, meaning total length)
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their token ratio
    """
    if numerator_tokens is None:
        numerator_tokens = ["G", "C"]  # Default for GC content

    sequences = generate_random_sequences(
        length=length, count=count, alphabet=alphabet, fixed_length=True
    )

    # If we have a delimited alphabet, we need to count tokens differently
    delimiter = alphabet.delimiter if isinstance(alphabet, Alphabet) else None

    # Calculate token ratios
    targets = []
    for sequence in sequences:
        if delimiter:
            # Split by delimiter for multi-character tokens
            tokens = sequence.split(delimiter)
            numerator_count = sum(1 for t in tokens if t in numerator_tokens)

            if denominator_tokens is None:
                # Use total length as denominator
                denominator_count = len(tokens)
            else:
                denominator_count = sum(1 for t in tokens if t in denominator_tokens)
        else:
            # For character-level alphabets
            numerator_count = sum(sequence.count(t) for t in numerator_tokens)

            if denominator_tokens is None:
                # Use total length as denominator
                denominator_count = len(sequence)
            else:
                denominator_count = sum(sequence.count(t) for t in denominator_tokens)

        # Calculate ratio (prevent division by zero)
        target = numerator_count / max(1, denominator_count)

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_pattern_position_task(
    count: int = 500,
    length: int = 50,
    alphabet: Union[str, Alphabet] = "ACGT",
    pattern: str = "GATA",
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends on the position of a pattern.

    This is a generalization of the motif_position task that works with any alphabet.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param alphabet: Characters to use in the sequences or an Alphabet instance
    :param pattern: Pattern to insert
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet=alphabet, fixed_length=True
    )

    # If we have a delimited alphabet, handle differently
    delimiter = alphabet.delimiter if isinstance(alphabet, Alphabet) else None

    # For delimited sequences, make sure pattern is a list of tokens
    if delimiter is not None and delimiter not in pattern:
        # Convert the pattern to a delimited string
        pattern_tokens = pattern.split(",") if "," in pattern else [pattern]
        pattern = delimiter.join(pattern_tokens)

    # Calculate the effective length of the pattern
    if delimiter is not None:
        pattern_length = len(pattern.split(delimiter))
    else:
        pattern_length = len(pattern)

    # Insert pattern at random positions in some sequences
    targets = []
    for i in range(count):
        if random.random() < 0.7:  # 70% chance to have the pattern
            if delimiter is not None:
                # For delimited sequences
                seq_tokens = sequences[i].split(delimiter)
                pos = random.randint(0, len(seq_tokens) - pattern_length)
                pattern_tokens = pattern.split(delimiter)

                # Replace tokens at the selected position
                seq_tokens[pos : pos + pattern_length] = pattern_tokens
                sequences[i] = delimiter.join(seq_tokens)
            else:
                # For character sequences
                pos = random.randint(0, length - len(pattern))
                seq_list = list(sequences[i])
                seq_list[pos : pos + len(pattern)] = pattern
                sequences[i] = "".join(seq_list)

            # Function depends on position (nonlinear transformation)
            if delimiter is not None:
                # For delimited sequences, normalize using token count
                seq_tokens = sequences[i].split(delimiter)
                norm_pos = pos / (len(seq_tokens) - pattern_length)
            else:
                # For character sequences
                norm_pos = pos / (length - len(pattern))

            target = np.sin(norm_pos * np.pi) * 5  # Sinusoidal function of position
        else:
            target = 0.0

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_pattern_count_task(
    count: int = 500,
    length: int = 50,
    alphabet: Union[str, Alphabet] = "ACGT",
    patterns: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends on the count of multiple patterns.

    This is a generalization of the motif_count task that works with any alphabet.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param alphabet: Characters to use in the sequences or an Alphabet instance
    :param patterns: List of patterns to count
    :param weights: Weight for each pattern's contribution
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    # Set default patterns based on alphabet type
    if patterns is None:
        if isinstance(alphabet, Alphabet) and alphabet.delimiter is not None:
            # For integer/delimited alphabets, use numeric patterns
            tokens = alphabet.tokens[:4]  # Use first 4 tokens
            patterns = [f"{tokens[0]}{tokens[1]}", f"{tokens[2]}{tokens[3]}"]
        else:
            # For character alphabets, use default DNA motifs
            patterns = ["AT", "GC", "TG", "CA"]

    if weights is None:
        weights = [1.0, -0.5, 2.0, -1.5][
            : len(patterns)
        ]  # Use default weights, limited to pattern count

    if len(patterns) != len(weights):
        raise ValueError("Length of patterns and weights must match")

    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet=alphabet, fixed_length=True
    )

    # If we have a delimited alphabet, handle pattern counting differently
    delimiter = alphabet.delimiter if isinstance(alphabet, Alphabet) else None

    # Calculate target based on pattern counts
    targets = []
    for sequence in sequences:
        target = 0.0

        if delimiter is not None:
            # For delimited sequences, join with delimiter for pattern matching
            for pattern, weight in zip(patterns, weights):
                # Ensure pattern has proper delimiters
                if delimiter not in pattern:
                    pattern_tokens = pattern.split(",") if "," in pattern else [pattern]
                    formatted_pattern = delimiter.join(pattern_tokens)
                else:
                    formatted_pattern = pattern

                # Count occurrences of the pattern
                count = sequence.count(formatted_pattern)
                target += count * weight
        else:
            # For character-level alphabets
            for pattern, weight in zip(patterns, weights):
                count = sequence.count(pattern)
                target += count * weight

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_nonlinear_composition_generic_task(
    count: int = 500,
    length: int = 30,
    alphabet: Union[str, Alphabet] = "ACGT",
    target_tokens: List[str] = None,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends nonlinearly on token composition.

    This is a generalization of the nonlinear_composition task
    that works with any alphabet.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param alphabet: Characters to use in the sequences or an Alphabet instance
    :param target_tokens: List of tokens to use for the nonlinear function
        (default: first 4 tokens)
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet=alphabet, fixed_length=True
    )

    # Default token selection
    if target_tokens is None:
        if isinstance(alphabet, Alphabet):
            # Use the first 4 tokens (exclude gap character)
            tokens = [t for t in alphabet.tokens if t != alphabet.gap_character]
            target_tokens = tokens[: min(4, len(tokens))]
        else:
            # For character alphabets, use default DNA tokens
            target_tokens = ["A", "C", "G", "T"]

    # If we have a delimited alphabet, handle token counting differently
    delimiter = alphabet.delimiter if isinstance(alphabet, Alphabet) else None

    # Calculate nonlinear function of token composition
    targets = []
    for sequence in sequences:
        if delimiter is not None:
            # For delimited sequences
            seq_tokens = sequence.split(delimiter)
            seq_length = len(seq_tokens)

            # Count token frequencies
            token_counts = {
                token: seq_tokens.count(token) / seq_length for token in target_tokens
            }
        else:
            # For character-level alphabets
            seq_length = len(sequence)
            token_counts = {
                token: sequence.count(token) / seq_length for token in target_tokens
            }

        # Ensure we have all required tokens (with 0 counts if not present)
        for token in target_tokens:
            if token not in token_counts:
                token_counts[token] = 0.0

        # Apply a nonlinear function (similar to nonlinear_composition_task)
        # We need at least 4 tokens for this formula,
        # so use available tokens with modulo
        t = target_tokens
        t_count = len(t)

        # Nonlinear function depends on token counts
        # Using a general version that works with any number of tokens
        if t_count >= 4:
            # Similar to the original nonlinear_composition_task for DNA
            numerator = token_counts[t[0]] * token_counts[t[2]]
            denominator = 0.1 + token_counts[t[1]] * token_counts[t[3]]
            target = numerator / denominator * 10
        elif t_count >= 2:
            # Simpler version for fewer tokens
            numerator = token_counts[t[0]]
            denominator = 0.1 + token_counts[t[1]]
            target = numerator / denominator * 10
        else:
            # Fallback for single token
            target = token_counts[t[0]] * 10

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


def create_interaction_generic_task(
    count: int = 500,
    length: int = 40,
    alphabet: Union[str, Alphabet] = "ACGT",
    interaction_pairs: List[Tuple[str, str, float]] = None,
    gap: int = 5,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target depends on interactions between positions.

    This is a generalization of the interaction task that works with any alphabet.

    :param count: Number of sequences to generate
    :param length: Length of each sequence
    :param alphabet: Characters to use in the sequences or an Alphabet instance
    :param interaction_pairs: List of (token1, token2, weight) tuples for interactions
    :param gap: Gap between interacting positions
    :param noise_level: Standard deviation of Gaussian noise to add
    :return: DataFrame with sequences and their function values
    """
    # Generate random sequences
    sequences = generate_random_sequences(
        length=length, count=count, alphabet=alphabet, fixed_length=True
    )

    # Set default interaction pairs based on alphabet
    if interaction_pairs is None:
        if isinstance(alphabet, Alphabet):
            # Use the first few tokens (exclude gap character)
            tokens = [t for t in alphabet.tokens if t != alphabet.gap_character]
            if len(tokens) >= 4:
                interaction_pairs = [
                    (tokens[0], tokens[3], 1.0),
                    (tokens[1], tokens[2], 1.5),
                ]
            else:
                # Fallback for alphabets with fewer tokens
                interaction_pairs = [(tokens[0], tokens[0], 1.0)]
        else:
            # For character alphabets, use default DNA interactions
            interaction_pairs = [("A", "T", 1.0), ("G", "C", 1.5)]

    # If we have a delimited alphabet, handle token access differently
    delimiter = alphabet.delimiter if isinstance(alphabet, Alphabet) else None

    # Calculate target based on interactions between positions
    targets = []
    for sequence in sequences:
        target = 0.0

        if delimiter is not None:
            # For delimited sequences
            seq_tokens = sequence.split(delimiter)

            # Look for specific pairs with a gap between them
            for i in range(len(seq_tokens) - gap - 1):
                token1 = seq_tokens[i]
                token2 = seq_tokens[i + gap]

                # Check all interaction pairs
                for t1, t2, weight in interaction_pairs:
                    if token1 == t1 and token2 == t2:
                        target += weight
        else:
            # For character-level alphabets
            # Look for specific pairs with a gap between them
            for i in range(len(sequence) - gap - 1):
                token1 = sequence[i]
                token2 = sequence[i + gap]

                # Check all interaction pairs
                for t1, t2, weight in interaction_pairs:
                    if token1 == t1 and token2 == t2:
                        target += weight

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


# Specialized tasks for biological sequences using partial functions
# These maintain backward compatibility with the original task names

# DNA G count task
create_g_count_task = partial(
    create_token_count_task,
    alphabet="ACGT",
    token="G",
    length=30,
)

# DNA GC content task
create_gc_content_task = partial(
    create_content_ratio_task,
    alphabet="ACGT",
    numerator_tokens=["G", "C"],
    length=30,
)

# DNA motif position task
create_motif_position_task = partial(
    create_pattern_position_task,
    alphabet="ACGT",
    pattern="GATA",  # Default pattern
    length=50,
)

# DNA motif count task
create_motif_count_task = partial(
    create_pattern_count_task,
    alphabet="ACGT",
    patterns=["AT", "GC", "TG", "CA"],  # Default patterns
    weights=[1.0, -0.5, 2.0, -1.5],  # Default weights
    length=50,
)

# DNA nonlinear composition task
create_nonlinear_composition_task = partial(
    create_nonlinear_composition_generic_task,
    alphabet="ACGT",
    target_tokens=["A", "C", "G", "T"],
    length=30,
)

# DNA interaction task
create_interaction_task = partial(
    create_interaction_generic_task,
    alphabet="ACGT",
    interaction_pairs=[("A", "T", 1.0), ("G", "C", 1.5)],
    gap=5,
    length=40,
)


# DNA classification task
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


# DNA multiclass task
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
            if other_classes:  # Only reassign if there are other classes available
                class_label = random.choice(other_classes)

        labels.append(class_label)

    return pd.DataFrame({"sequence": sequences, "function": labels})


# Specialized tasks for integer sequences using partial functions
# These provide the equivalents of the nucleotide-based tasks for integer sequences

# Integer version of G-count (sum of digits)
create_integer_sum_task = partial(
    create_token_count_task,
    alphabet=Alphabet.integer(max_value=9),
    token="5",  # Count 5's instead of G's
    length=5,
)


# Integer version of max value
def create_integer_max_task(
    count: int = 500,
    length: int = 5,
    max_value: int = 9,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where the target is the maximum integer in the sequence."""
    sequences = generate_integer_sequences(
        length=length, count=count, max_value=max_value, fixed_length=True
    )

    # Find max value in each sequence
    targets = []
    for sequence in sequences:
        # Parse integers from the comma-delimited sequence
        integers = [int(x) for x in sequence.split(",")]
        target = max(integers)

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


# Integer version of motif position (pattern position)
create_integer_pattern_task = partial(
    create_pattern_position_task,
    alphabet=Alphabet.integer(max_value=9),
    pattern="1,2,3",
    length=5,
)


# Integer version of nonlinear task (sum of squares)
def create_integer_nonlinear_task(
    count: int = 500,
    length: int = 5,
    max_value: int = 9,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset with nonlinear relationship - sum of squares."""
    sequences = generate_integer_sequences(
        length=length, count=count, max_value=max_value, fixed_length=True
    )

    # Calculate nonlinear function (sum of squares)
    targets = []
    for sequence in sequences:
        # Parse integers
        integers = [int(x) for x in sequence.split(",")]
        # Sum of squares divided by length
        target = sum(x**2 for x in integers) / length

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


# Integer version of interaction task
def create_integer_interaction_task(
    count: int = 500,
    length: int = 5,
    max_value: int = 9,
    noise_level: float = 0.0,
) -> pd.DataFrame:
    """Create a dataset where target depends on interactions between adjacent ints."""
    sequences = generate_integer_sequences(
        length=length, count=count, max_value=max_value, fixed_length=True
    )

    # Calculate interactions between adjacent positions
    targets = []
    for sequence in sequences:
        # Parse integers
        integers = [int(x) for x in sequence.split(",")]
        # Product of adjacent pairs
        interactions = [integers[i] * integers[i + 1] for i in range(len(integers) - 1)]
        target = sum(interactions) / (length - 1) if interactions else 0

        # Add noise if specified
        if noise_level > 0:
            target += np.random.normal(0, noise_level)

        targets.append(target)

    return pd.DataFrame({"sequence": sequences, "function": targets})


# Integer classification task
def create_integer_classification_task(
    count: int = 500,
    length: int = 5,
    max_value: int = 9,
    noise_level: float = 0.1,
) -> pd.DataFrame:
    """Create binary classification based on median value threshold."""
    sequences = generate_integer_sequences(
        length=length, count=count, max_value=max_value, fixed_length=True
    )

    # Determine class based on median value
    classes = []
    threshold = max_value / 2

    for sequence in sequences:
        # Parse integers
        integers = [int(x) for x in sequence.split(",")]
        # Class 1 if median > threshold, otherwise 0
        median_value = sorted(integers)[len(integers) // 2]
        class_label = 1 if median_value > threshold else 0

        # Add noise by randomly flipping the class with probability noise_level
        if random.random() < noise_level:
            class_label = 1 - class_label

        classes.append(class_label)

    return pd.DataFrame({"sequence": sequences, "function": classes})


# Integer multiclass task
def create_integer_multiclass_task(
    count: int = 500,
    length: int = 5,
    max_value: int = 9,
    num_classes: int = 3,
    noise_level: float = 0.1,
) -> pd.DataFrame:
    """Create a multi-class classification based on average value."""
    sequences = generate_integer_sequences(
        length=length, count=count, max_value=max_value, fixed_length=True
    )

    # Determine class based on average value
    classes = []

    for sequence in sequences:
        # Parse integers
        integers = [int(x) for x in sequence.split(",")]
        # Determine class based on average value
        avg_value = sum(integers) / len(integers)
        # Map to class range
        class_label = min(
            num_classes - 1, int(avg_value / (max_value + 1) * num_classes)
        )

        # Add noise by randomly reassigning with probability noise_level
        if random.random() < noise_level:
            class_label = random.randint(0, num_classes - 1)

        classes.append(class_label)

    return pd.DataFrame({"sequence": sequences, "function": classes})


# Integer version of GC-content (ratio of high values to total)
create_integer_ratio_task = partial(
    create_content_ratio_task,
    alphabet=Alphabet.integer(max_value=9),
    numerator_tokens=["5", "6", "7", "8", "9"],  # High values
    length=5,
)

# Integer version of motif position (pattern position)
create_integer_pattern_position_task = partial(
    create_pattern_position_task,
    alphabet=Alphabet.integer(max_value=9),
    pattern="1,2,3",
    length=5,
)

# Integer version of motif count (pattern count)
create_integer_pattern_count_task = partial(
    create_pattern_count_task,
    alphabet=Alphabet.integer(max_value=9),
    patterns=["1,2", "3,4", "5,6", "7,8"],
    weights=[1.0, -0.5, 2.0, -1.5],
    length=5,
)

# Integer version of nonlinear composition
create_integer_nonlinear_composition_task = partial(
    create_nonlinear_composition_generic_task,
    alphabet=Alphabet.integer(max_value=9),
    target_tokens=["1", "3", "5", "7"],
    length=5,
)

# Integer version of interaction task
create_integer_position_interaction_task = partial(
    create_interaction_generic_task,
    alphabet=Alphabet.integer(max_value=9),
    interaction_pairs=[("1", "9", 1.0), ("2", "8", 1.5), ("3", "7", 2.0)],
    gap=2,
    length=5,
)


def generate_dataset_by_task(
    task: Literal[
        # Biological sequence tasks
        "g_count",
        "gc_content",
        "motif_position",
        "motif_count",
        "length_dependent",
        "nonlinear_composition",
        "interaction",
        "classification",
        "multiclass",
        # Integer sequence tasks
        "integer_sum",
        "integer_max",
        "integer_pattern",
        "integer_nonlinear",
        "integer_interaction",
        "integer_classification",
        "integer_multiclass",
        "integer_token_count",
        "integer_ratio",
        "integer_pattern_position",
        "integer_pattern_count",
        "integer_nonlinear_composition",
        "integer_position_interaction",
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
    # Map tasks to their generator functions
    task_functions = {
        # Biological sequence tasks
        "g_count": create_g_count_task,
        "gc_content": create_gc_content_task,
        "motif_position": create_motif_position_task,
        "motif_count": create_motif_count_task,
        "length_dependent": create_length_dependent_task,
        "nonlinear_composition": create_nonlinear_composition_task,
        "interaction": create_interaction_task,
        "classification": create_classification_task,
        "multiclass": create_multiclass_task,
        # Integer tasks
        "integer_sum": create_integer_sum_task,
        "integer_token_count": create_integer_sum_task,  # Alias for integer_sum
        "integer_max": create_integer_max_task,
        "integer_pattern": create_integer_pattern_task,
        "integer_pattern_position": create_integer_pattern_position_task,
        "integer_nonlinear": create_integer_nonlinear_task,
        "integer_nonlinear_composition": create_integer_nonlinear_composition_task,
        "integer_interaction": create_integer_interaction_task,
        "integer_position_interaction": create_integer_position_interaction_task,
        "integer_classification": create_integer_classification_task,
        "integer_multiclass": create_integer_multiclass_task,
        "integer_ratio": create_integer_ratio_task,
        "integer_pattern_count": create_integer_pattern_count_task,
    }

    if task not in task_functions:
        raise ValueError(
            f"Unknown task: {task}. Available tasks: {list(task_functions.keys())}"
        )

    return task_functions[task](count=count, noise_level=noise_level, **kwargs)
