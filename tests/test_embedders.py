"""Tests for the embedders module."""

import numpy as np
import pytest

from fast_seqfunc.alphabets import Alphabet
from fast_seqfunc.embedders import (
    OneHotEmbedder,
    get_embedder,
)


def test_one_hot_embedder_init():
    """Test initialization with different parameters."""
    # Default initialization
    embedder = OneHotEmbedder()
    assert embedder.sequence_type == "auto"
    assert embedder.alphabet is None
    assert embedder.alphabet_size is None
    assert embedder.pad_sequences is True
    assert embedder.gap_character == "-"

    # Custom parameters
    embedder = OneHotEmbedder(
        sequence_type="protein", max_length=10, pad_sequences=False, gap_character="X"
    )
    assert embedder.sequence_type == "protein"
    assert embedder.max_length == 10
    assert embedder.pad_sequences is False
    assert embedder.gap_character == "X"


def test_one_hot_embedder_fit():
    """Test fitting to sequences."""
    embedder = OneHotEmbedder()

    # Protein sequences
    protein_seqs = ["ACDEFG", "GHIKLMN", "PQRSTVWY"]
    embedder.fit(protein_seqs)
    assert embedder.sequence_type == "protein"
    # Check the alphabet is a protein alphabet
    assert isinstance(embedder.alphabet, Alphabet)
    assert embedder.alphabet.name == "protein"
    assert embedder.alphabet_size == 21  # 20 amino acids + gap
    assert embedder.max_length == 8  # Length of longest sequence

    # DNA sequences with custom gap
    dna_seqs = ["ACGT", "TGCA", "AATT"]
    embedder = OneHotEmbedder(gap_character="X")
    embedder.fit(dna_seqs)
    assert embedder.sequence_type == "dna"
    # Check the alphabet is a DNA alphabet with custom gap
    assert isinstance(embedder.alphabet, Alphabet)
    assert embedder.alphabet.gap_character == "X"
    assert embedder.alphabet_size == 5  # 4 nucleotides + gap
    assert embedder.max_length == 4  # All sequences are same length

    # Explicit sequence type
    embedder = OneHotEmbedder(sequence_type="rna")
    embedder.fit(["ACGU", "UGCA"])
    assert embedder.sequence_type == "rna"
    # Check the alphabet is an RNA alphabet
    assert isinstance(embedder.alphabet, Alphabet)
    assert embedder.alphabet.name == "rna"
    assert embedder.alphabet_size == 5  # 4 nucleotides + gap


def test_one_hot_encode():
    """Test one-hot encoding a single sequence."""
    # DNA sequence
    embedder = OneHotEmbedder(sequence_type="dna")
    embedder.fit(["ACGT"])

    # "ACGT" with 5 letters in alphabet (including gap) = 4x5 matrix
    # (flattened to 20 values)
    embedding = embedder._one_hot_encode("ACGT")
    assert embedding.shape == (20,)  # 4 positions * 5 letters (including gap)

    # One-hot encoding should have exactly one 1 per position
    embedding_2d = embedding.reshape(4, 5)
    assert np.sum(embedding_2d) == 4  # One 1 per position
    assert np.array_equal(np.sum(embedding_2d, axis=1), np.ones(4))

    # Test gap character handling
    embedding = embedder._one_hot_encode("AC-T")
    embedding_2d = embedding.reshape(4, 5)
    assert np.sum(embedding_2d) == 4  # One 1 per position
    # Gap should be encoded in the position that corresponds to the gap character
    gap_idx = embedder.alphabet.token_to_idx["-"]
    assert embedding_2d[2, gap_idx] == 1


def test_preprocess_sequences():
    """Test sequence preprocessing with padding and truncation."""
    embedder = OneHotEmbedder(sequence_type="dna", max_length=5)
    embedder.fit(["ACGT"])  # Set up alphabet

    # Test padding
    sequences = ["AC", "ACGT", "ACGTGC"]
    processed = embedder._preprocess_sequences(sequences)

    assert len(processed) == 3
    assert processed[0] == "AC---"  # Padded with 3 gap characters
    assert processed[1] == "ACGT-"  # Padded with 1 gap character
    assert processed[2] == "ACGTG"  # Truncated to 5 characters

    # Test with custom gap character
    embedder = OneHotEmbedder(sequence_type="dna", max_length=4, gap_character="X")
    embedder.fit(["ACGT"])

    processed = embedder._preprocess_sequences(["A", "ACG"])
    assert processed[0] == "AXXX"  # Padded with custom gap
    assert processed[1] == "ACGX"

    # Test with padding disabled
    embedder = OneHotEmbedder(sequence_type="dna", pad_sequences=False)
    embedder.fit(["ACGT"])

    # Should not modify sequences
    processed = embedder._preprocess_sequences(["A", "ACGT", "ACGTGC"])
    assert processed == ["A", "ACGT", "ACGTGC"]


def test_transform_with_padding():
    """Test transforming sequences of different lengths with padding."""
    # Sequences of different lengths
    sequences = ["A", "ACG", "ACGT", "ACGTGC"]

    # With padding enabled (default)
    embedder = OneHotEmbedder(sequence_type="dna")
    embeddings = embedder.fit_transform(sequences)

    # Should pad to longest sequence length (6)
    # Each sequence with alphabet size 5 (ACGT-)
    assert embeddings.shape == (4, 30)  # 4 sequences, 6 positions * 5 alphabet size

    # With padding disabled
    embedder = OneHotEmbedder(sequence_type="dna", pad_sequences=False)
    embeddings = embedder.fit_transform(sequences)

    # Each sequence should have its own length * alphabet size (5)
    # But these are flattened to different lengths
    assert len(embeddings) == 4
    # First sequence: length 1 * alphabet size 5
    assert embeddings[0].shape == (5,)
    # Last sequence: length 6 * alphabet size 5
    assert embeddings[3].shape == (30,)

    # With explicit max_length
    embedder = OneHotEmbedder(sequence_type="dna", max_length=4)
    embeddings = embedder.fit_transform(sequences)

    # Should truncate/pad to max_length
    assert embeddings.shape == (4, 20)  # 4 sequences, 4 positions * 5 alphabet size


def test_variable_length_input():
    """Test with variable length input sequences."""
    # Protein sequences of different lengths
    sequences = ["ACK", "ACDEFGHI", "P"]

    # Default behavior: pad to max length
    embedder = OneHotEmbedder(sequence_type="protein")
    embedder.fit(sequences)

    # Max length is 8, alphabet size is 21 (20 aa + gap)
    embeddings = embedder.transform(sequences)
    assert embeddings.shape == (3, 168)  # 3 sequences, 8 positions * 21 alphabet size

    # Transform a new sequence
    new_embedding = embedder.transform(["ACDKL"])
    assert new_embedding.shape == (1, 168)  # Padded to same shape


def test_transform():
    """Test transforming multiple sequences."""
    embedder = OneHotEmbedder(sequence_type="protein")
    embedder.fit(["ACDEF", "GHIKL"])

    # Transform multiple sequences
    embeddings = embedder.transform(["ACDEF", "GHIKL"])

    # With alphabet of 21 characters (20 amino acids + gap) and length 5
    assert embeddings.shape == (2, 105)  # 2 sequences, 5 positions * 21 amino acids


def test_fit_transform():
    """Test fit_transform method."""
    embedder = OneHotEmbedder()
    sequences = ["ACGT", "TGCA"]

    # fit_transform should do both operations
    embeddings = embedder.fit_transform(sequences)

    # Should have fitted
    assert embedder.sequence_type == "dna"
    assert isinstance(embedder.alphabet, Alphabet)
    assert embedder.alphabet.name == "dna"
    assert embedder.alphabet_size == 5

    # Should have transformed
    assert embeddings.shape == (2, 20)  # 2 sequences, 4 positions * 5 alphabet chars


def test_get_embedder():
    """Test the embedder factory function."""
    # Get one-hot embedder
    embedder = get_embedder("one-hot")
    assert isinstance(embedder, OneHotEmbedder)

    # With custom parameters
    embedder = get_embedder(
        "one-hot", max_length=10, pad_sequences=False, gap_character="X"
    )
    assert embedder.max_length == 10
    assert embedder.pad_sequences is False
    assert embedder.gap_character == "X"

    # Test invalid method
    with pytest.raises(ValueError):
        get_embedder("invalid-method")
