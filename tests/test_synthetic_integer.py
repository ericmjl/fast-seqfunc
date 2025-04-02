"""Tests for integer sequence synthetic data generation functions in fast-seqfunc."""

import numpy as np
import pandas as pd

from fast_seqfunc.synthetic import (
    create_integer_classification_task,
    create_integer_interaction_task,
    create_integer_max_task,
    create_integer_multiclass_task,
    create_integer_nonlinear_composition_task,
    create_integer_nonlinear_task,
    create_integer_pattern_count_task,
    create_integer_pattern_position_task,
    create_integer_pattern_task,
    create_integer_position_interaction_task,
    create_integer_ratio_task,
    create_integer_sum_task,
    generate_dataset_by_task,
)


def test_integer_sum_task():
    """Test the integer sum task generator."""
    df = create_integer_sum_task(count=10, length=5, noise_level=0.0)

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Check that sequences are properly formatted
    for _, row in df.iterrows():
        sequence = row["sequence"]
        # Verify it contains comma-separated values (not necessarily all digits,
        # may be negative)
        parts = sequence.split(",")
        assert len(parts) > 0
        # Verify that all parts can be converted to integers
        try:
            [int(part) for part in parts]
        except ValueError:
            assert False, f"Sequence {sequence} contains non-integer values"


def test_integer_max_task():
    """Test the integer max task generator."""
    df = create_integer_max_task(count=10, length=5, noise_level=0.0)

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Check that sequences are properly formatted
    for _, row in df.iterrows():
        sequence = row["sequence"]
        # Verify it contains comma-separated values
        parts = sequence.split(",")
        assert len(parts) > 0
        # Verify that all parts can be converted to integers
        try:
            [int(part) for part in parts]
        except ValueError:
            assert False, f"Sequence {sequence} contains non-integer values"


def test_integer_pattern_task():
    """Test the integer pattern task generator."""
    # Convert pattern to strings and join with delimiter to match function expectations
    pattern = "1,2,3"  # Use comma-delimited string directly
    df = create_integer_pattern_task(
        count=10, length=10, pattern=pattern, noise_level=0.0
    )

    # Check shape and columns
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Check that function values are finite
    assert all(np.isfinite(val) for val in df["function"])


def test_integer_nonlinear_task():
    """Test the integer nonlinear task generator."""
    df = create_integer_nonlinear_task(count=10, length=5, noise_level=0.0)

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Test that the function produces values within an expected range
    # (we can't test exact values due to the nonlinear nature)
    for _, row in df.iterrows():
        function = row["function"]
        # Ensure values are not NaN or infinite
        assert np.isfinite(function)


def test_integer_interaction_task():
    """Test the integer interaction task generator."""
    # Note: In the implementation, interaction_pairs parameter might be different
    # Adapt test to match actual function signature
    df = create_integer_interaction_task(count=10, length=10, noise_level=0.0)

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Ensure function values are finite
    assert all(np.isfinite(val) for val in df["function"])


def test_integer_classification_task():
    """Test the integer classification task generator."""
    df = create_integer_classification_task(count=20, noise_level=0.0)

    # Check shape
    assert len(df) == 20
    assert set(df.columns) == {"sequence", "function"}

    # Verify that function values are binary (0 or 1)
    for _, row in df.iterrows():
        function = row["function"]
        assert function in [0, 1]


def test_integer_multiclass_task():
    """Test the integer multiclass task generator."""
    df = create_integer_multiclass_task(count=30, num_classes=3, noise_level=0.0)

    # Check shape
    assert len(df) == 30
    assert set(df.columns) == {"sequence", "function"}

    # Verify that function values are integers in the range [0, num_classes)
    classes = set(df["function"].tolist())
    assert all(cls in {0, 1, 2} for cls in classes)


def test_integer_ratio_task():
    """Test the integer ratio task generator."""
    # Convert tokens to strings
    numerator_tokens = ["1", "3", "5"]
    denominator_tokens = ["2", "4", "6"]

    df = create_integer_ratio_task(
        count=10,
        length=10,
        numerator_tokens=numerator_tokens,
        denominator_tokens=denominator_tokens,
        noise_level=0.0,
    )

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Ensure all function values are finite
    assert all(np.isfinite(val) for val in df["function"])


def test_integer_pattern_position_task():
    """Test the integer pattern position task generator."""
    # Convert pattern to proper string format that the function expects
    pattern = "1,2,3"  # Use comma-delimited string
    df = create_integer_pattern_position_task(
        count=10, length=20, pattern=pattern, noise_level=0.0
    )

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # For each sequence, check that function values are finite
    for _, row in df.iterrows():
        function = row["function"]
        assert np.isfinite(function)


def test_integer_pattern_count_task():
    """Test the integer pattern count task generator."""
    # Convert patterns to proper format that the function expects
    patterns = ["1,2", "3,4"]  # Use comma-delimited strings
    weights = [1.0, -0.5]
    df = create_integer_pattern_count_task(
        count=10, length=15, patterns=patterns, weights=weights, noise_level=0.0
    )

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Ensure function values are finite
    assert all(np.isfinite(val) for val in df["function"])


def test_integer_nonlinear_composition_task():
    """Test the integer nonlinear composition task generator."""
    # Convert target tokens to strings
    target_tokens = ["1", "2", "3", "4"]
    df = create_integer_nonlinear_composition_task(
        count=10, length=10, target_tokens=target_tokens, noise_level=0.0
    )

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Ensure function values are finite
    assert all(np.isfinite(val) for val in df["function"])


def test_integer_position_interaction_task():
    """Test the integer position interaction task generator."""
    df = create_integer_position_interaction_task(count=10, length=10, noise_level=0.0)

    # Check shape
    assert len(df) == 10
    assert set(df.columns) == {"sequence", "function"}

    # Ensure function values are finite
    assert all(np.isfinite(val) for val in df["function"])


def test_generate_dataset_by_task_integer_tasks():
    """Test generate_dataset_by_task with different integer tasks."""
    # Test each integer task type
    integer_tasks = [
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
    ]

    for task in integer_tasks:
        # Generate a small dataset for each task
        df = generate_dataset_by_task(task=task, count=5, noise_level=0.1)

        # Check basic properties
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert set(df.columns) == {"sequence", "function"}

        # Check that sequences are comma-delimited integers
        for seq in df["sequence"]:
            # Verify it's a string
            assert isinstance(seq, str)

            # Verify it contains comma-separated integers
            parts = seq.split(",")
            try:
                [int(part) for part in parts]
                valid = True
            except ValueError:
                valid = False
            assert valid, (
                f"Sequence {seq} is not a valid comma-delimited integer sequence"
            )
