"""Tests for the synthetic data generation module."""

import pandas as pd
import pytest

from fast_seqfunc.synthetic import (
    count_matches,
    create_classification_task,
    create_g_count_task,
    create_gc_content_task,
    create_length_dependent_task,
    create_motif_count_task,
    create_motif_position_task,
    create_multiclass_task,
    generate_dataset_by_task,
    generate_random_sequences,
)


def test_generate_random_sequences():
    """Test generating random sequences."""
    # Test fixed length
    sequences = generate_random_sequences(length=10, count=5, alphabet="ACGT")
    assert len(sequences) == 5
    assert all(len(seq) == 10 for seq in sequences)
    assert all(all(c in "ACGT" for c in seq) for seq in sequences)

    # Test variable length
    sequences = generate_random_sequences(
        count=5, alphabet="ACGT", fixed_length=False, length_range=(5, 15)
    )
    assert len(sequences) == 5
    assert all(5 <= len(seq) <= 15 for seq in sequences)


def test_count_matches():
    """Test counting pattern matches."""
    assert count_matches("AAAA", "A") == 4
    assert count_matches("ACGTACGT", "ACGT") == 2
    assert count_matches("ACGT", "X") == 0
    assert count_matches("AAAAA", "AA") == 2  # Non-overlapping


def test_g_count_task():
    """Test G-count task generation."""
    df = create_g_count_task(count=10, length=20, noise_level=0.0)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "sequence" in df.columns
    assert "function" in df.columns
    assert all(len(seq) == 20 for seq in df["sequence"])

    # Without noise, function should exactly match G count
    for i, row in df.iterrows():
        assert row["function"] == row["sequence"].count("G")


def test_gc_content_task():
    """Test GC-content task generation."""
    df = create_gc_content_task(count=10, length=20, noise_level=0.0)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10

    # Without noise, function should exactly match GC content
    for i, row in df.iterrows():
        expected_gc = (row["sequence"].count("G") + row["sequence"].count("C")) / len(
            row["sequence"]
        )
        assert row["function"] == expected_gc


def test_motif_position_task():
    """Test motif position task generation."""
    motif = "GATA"
    df = create_motif_position_task(count=20, length=30, motif=motif, noise_level=0.0)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20

    # Check that some sequences contain the motif
    assert any(motif in seq for seq in df["sequence"])


def test_motif_count_task():
    """Test motif count task generation."""
    motifs = ["AT", "GC"]
    weights = [1.0, 2.0]
    df = create_motif_count_task(
        count=10, length=30, motifs=motifs, weights=weights, noise_level=0.0
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10

    # Without noise, function should match weighted count
    for i, row in df.iterrows():
        expected = (
            row["sequence"].count(motifs[0]) * weights[0]
            + row["sequence"].count(motifs[1]) * weights[1]
        )
        assert row["function"] == expected


def test_length_dependent_task():
    """Test length-dependent task generation."""
    df = create_length_dependent_task(
        count=10, min_length=10, max_length=20, noise_level=0.0
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert all(10 <= len(seq) <= 20 for seq in df["sequence"])


def test_classification_task():
    """Test classification task generation."""
    df = create_classification_task(count=50, length=20, noise_level=0.0)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert set(df["function"].unique()).issubset({0, 1})


def test_multiclass_task():
    """Test multi-class task generation."""
    df = create_multiclass_task(count=100, length=20, noise_level=0.0)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert 1 < len(df["function"].unique()) <= 4  # Should have 2-4 classes


def test_generate_dataset_by_task():
    """Test the task selection function."""
    for task in ["g_count", "gc_content", "motif_position", "classification"]:
        df = generate_dataset_by_task(task=task, count=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "sequence" in df.columns
        assert "function" in df.columns

    # Test invalid task
    with pytest.raises(ValueError):
        generate_dataset_by_task(task="invalid_task")
