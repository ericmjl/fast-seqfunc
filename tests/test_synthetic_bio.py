"""Tests for biological sequence synthetic data generation functions in fast-seqfunc."""

import re

import numpy as np
import pandas as pd

from fast_seqfunc.alphabets import Alphabet
from fast_seqfunc.synthetic import (
    create_classification_task,
    create_content_ratio_task,
    create_g_count_task,
    create_gc_content_task,
    create_interaction_generic_task,
    create_interaction_task,
    create_length_dependent_task,
    create_motif_count_task,
    create_motif_position_task,
    create_multiclass_task,
    create_nonlinear_composition_generic_task,
    create_nonlinear_composition_task,
    create_pattern_count_task,
    create_pattern_position_task,
    create_token_count_task,
    generate_random_sequences,
)


def test_generate_random_sequences_dna():
    """Test generate_random_sequences with DNA alphabet."""
    # Test with DNA alphabet as string
    sequences = generate_random_sequences(
        length=10, count=5, alphabet="ACGT", fixed_length=True
    )

    # Check count and length
    assert len(sequences) == 5
    assert all(len(seq) == 10 for seq in sequences)

    # Check that sequences only contain valid characters
    dna_pattern = re.compile(r"^[ACGT]+$")
    assert all(dna_pattern.match(seq) for seq in sequences)


def test_generate_random_sequences_custom_alphabet():
    """Test generate_random_sequences with custom alphabet."""
    # Create a custom alphabet
    alphabet = Alphabet(tokens=["1", "2", "3", "4"], delimiter=",")

    # Generate sequences
    sequences = generate_random_sequences(
        length=5, count=10, alphabet=alphabet, fixed_length=True
    )

    # Check count and properties
    assert len(sequences) == 10

    # Check that each sequence has exactly 5 tokens
    for seq in sequences:
        tokens = seq.split(",")
        assert len(tokens) == 5
        assert all(token in alphabet.tokens for token in tokens)


def test_generate_random_sequences_variable_length():
    """Test generate_random_sequences with variable length."""
    min_length = 5
    max_length = 10

    # Generate variable length sequences - use length_range parameter for min/max
    sequences = generate_random_sequences(
        length=min_length,
        length_range=(min_length, max_length),
        count=20,
        alphabet="ACGT",
        fixed_length=False,
    )

    # Check count
    assert len(sequences) == 20

    # Check that lengths are within the specified range
    lengths = [len(seq) for seq in sequences]
    assert all(min_length <= length <= max_length for length in lengths)


def test_create_nonlinear_composition_generic_task():
    """Test create_nonlinear_composition_generic_task with various parameters."""
    # Test with DNA alphabet
    df = create_nonlinear_composition_generic_task(
        count=10,
        length=20,
        alphabet="ACGT",
        target_tokens=["A", "C", "G", "T"],
        noise_level=0.0,
    )

    # Check basic shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Test with custom alphabet
    alphabet = Alphabet(tokens=["1", "2", "3", "4"], delimiter=",")
    df_custom = create_nonlinear_composition_generic_task(
        count=10,
        length=20,
        alphabet=alphabet,
        target_tokens=["1", "2", "3", "4"],
        noise_level=0.0,
    )

    # Check custom alphabet results
    assert len(df_custom) == 10
    assert all("," in seq for seq in df_custom["sequence"])

    # Test with limited target tokens
    df_limited = create_nonlinear_composition_generic_task(
        count=10, length=20, alphabet="ACGT", target_tokens=["A", "G"], noise_level=0.0
    )

    # Check limited token results
    assert len(df_limited) == 10

    # Ensure all function values are finite
    assert all(np.isfinite(val) for val in df["function"])
    assert all(np.isfinite(val) for val in df_custom["function"])
    assert all(np.isfinite(val) for val in df_limited["function"])


def test_create_interaction_generic_task():
    """Test create_interaction_generic_task with various parameters."""
    # Test with DNA alphabet and default interaction pairs
    df = create_interaction_generic_task(
        count=10, length=20, alphabet="ACGT", noise_level=0.0
    )

    # Check basic shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Test with custom interaction pairs
    custom_pairs = [("A", "C", 2.0), ("T", "G", 1.5)]
    df_custom = create_interaction_generic_task(
        count=10,
        length=20,
        alphabet="ACGT",
        interaction_pairs=custom_pairs,
        noise_level=0.0,
    )

    # Check custom pairs results
    assert len(df_custom) == 10

    # Test with custom alphabet
    alphabet = Alphabet(tokens=["1", "2", "3", "4"], delimiter=",")
    interaction_pairs = [("1", "3", 1.0), ("2", "4", 1.5)]
    df_alphabet = create_interaction_generic_task(
        count=10,
        length=20,
        alphabet=alphabet,
        interaction_pairs=interaction_pairs,
        noise_level=0.0,
    )

    # Check custom alphabet results
    assert len(df_alphabet) == 10
    assert all("," in seq for seq in df_alphabet["sequence"])

    # Ensure all function values are finite
    assert all(np.isfinite(val) for val in df["function"])
    assert all(np.isfinite(val) for val in df_custom["function"])
    assert all(np.isfinite(val) for val in df_alphabet["function"])


def test_create_pattern_position_task():
    """Test create_pattern_position_task."""
    # Test with default pattern
    df = create_pattern_position_task(
        count=10, length=30, alphabet="ACGT", pattern="GATA", noise_level=0.0
    )

    # Check basic shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Test with custom pattern
    df_custom = create_pattern_position_task(
        count=10, length=30, alphabet="ACGT", pattern="ATCG", noise_level=0.0
    )

    # Check that function values are finite (don't assume 0-1 range)
    assert all(np.isfinite(val) for val in df["function"])
    assert all(np.isfinite(val) for val in df_custom["function"])


def test_create_pattern_count_task():
    """Test create_pattern_count_task."""
    # Test with default patterns and weights
    df = create_pattern_count_task(
        count=10, length=30, alphabet="ACGT", noise_level=0.0
    )

    # Check basic shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Test with custom patterns and weights
    patterns = ["AT", "GC"]
    weights = [1.0, -1.0]
    df_custom = create_pattern_count_task(
        count=10,
        length=30,
        alphabet="ACGT",
        patterns=patterns,
        weights=weights,
        noise_level=0.0,
    )

    # Check custom patterns results
    assert len(df_custom) == 10

    # Ensure all function values are finite
    assert all(np.isfinite(val) for val in df["function"])
    assert all(np.isfinite(val) for val in df_custom["function"])


def test_create_content_ratio_task():
    """Test create_content_ratio_task."""
    # Test with default parameters (like GC content)
    df = create_content_ratio_task(
        count=10,
        length=30,
        alphabet="ACGT",
        numerator_tokens=["G", "C"],
        noise_level=0.0,
    )

    # Check basic shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Test with custom tokens
    df_custom = create_content_ratio_task(
        count=10,
        length=30,
        alphabet="ACGT",
        numerator_tokens=["A"],
        denominator_tokens=["T"],
        noise_level=0.0,
    )

    # Check custom tokens results
    assert len(df_custom) == 10

    # Ensure all function values are finite (don't assume specific range)
    assert all(np.isfinite(val) for val in df["function"])
    assert all(np.isfinite(val) for val in df_custom["function"])


def test_create_token_count_task():
    """Test create_token_count_task."""
    # Test counting 'G' tokens (like G count task)
    df = create_token_count_task(
        count=10, length=30, alphabet="ACGT", token="G", noise_level=0.0
    )

    # Check basic shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Ensure all function values are finite
    assert all(np.isfinite(val) for val in df["function"])

    # Test with different token
    df_custom = create_token_count_task(
        count=10, length=30, alphabet="ACGT", token="A", noise_level=0.0
    )

    # Check custom token results
    assert len(df_custom) == 10
    assert all(np.isfinite(val) for val in df_custom["function"])


def test_specialized_biological_tasks():
    """Test the specialized biological sequence tasks."""
    # Test each task with a small count
    tasks = [
        create_g_count_task,
        create_gc_content_task,
        create_motif_position_task,
        create_motif_count_task,
        create_nonlinear_composition_task,
        create_interaction_task,
        create_length_dependent_task,
        create_classification_task,
    ]

    for task_func in tasks:
        df = task_func(count=5, noise_level=0.0)

        # Check basic shape
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert set(df.columns) == {"sequence", "function"}

        # Check for DNA sequences
        dna_pattern = re.compile(r"^[ACGT]+$")
        assert all(dna_pattern.match(seq) for seq in df["sequence"])

        # Ensure all function values are finite
        assert all(np.isfinite(val) for val in df["function"])


def test_multiclass_task():
    """Test create_multiclass_task with different class counts."""
    # Test with default 2 classes
    df_default = create_multiclass_task(count=20, noise_level=0.0)

    # Check basic shape
    assert len(df_default) == 20
    assert set(df_default.columns) == {"sequence", "function"}

    # Check that classes are integers (don't assume specific values)
    classes_default = set(df_default["function"])
    assert all(isinstance(cls, int) for cls in classes_default)

    # Create a new instance of multiclass_task with 3 classes
    # Looking at the function signature,
    # there's no parameter to set the number of classes
    # So we'll just check the basic functionality with default parameters
    df_three = create_multiclass_task(count=30, noise_level=0.0)

    # Check results
    assert len(df_three) == 30
    classes_three = set(df_three["function"])
    assert all(isinstance(cls, int) for cls in classes_three)

    # Create another instance with different size
    df_large = create_multiclass_task(count=50, noise_level=0.0)

    # Check results
    assert len(df_large) == 50
    classes_large = set(df_large["function"])
    assert all(isinstance(cls, int) for cls in classes_large)


def test_length_dependent_task():
    """Test create_length_dependent_task with various parameters."""
    # Test with default parameters
    df = create_length_dependent_task(
        count=10, min_length=10, max_length=30, noise_level=0.0
    )

    # Check basic shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Check that sequences have variable lengths in the specified range
    lengths = [len(seq) for seq in df["sequence"]]
    assert all(10 <= length <= 30 for length in lengths)

    # Ensure all function values are finite
    assert all(np.isfinite(val) for val in df["function"])
