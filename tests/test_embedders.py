"""Tests for the embedders module."""

import numpy as np
import pytest

from fast_seqfunc.embedders import (
    OneHotEmbedder,
    get_embedder,
)


class TestOneHotEmbedder:
    """Test suite for OneHotEmbedder."""

    def test_init(self):
        """Test initialization with different parameters."""
        # Default initialization
        embedder = OneHotEmbedder()
        assert embedder.sequence_type == "auto"
        assert embedder.alphabet is None
        assert embedder.alphabet_size is None

        # Custom parameters
        embedder = OneHotEmbedder(sequence_type="protein")
        assert embedder.sequence_type == "protein"

    def test_fit(self):
        """Test fitting to sequences."""
        embedder = OneHotEmbedder()

        # Protein sequences
        protein_seqs = ["ACDEFG", "GHIKLMN", "PQRSTVWY"]
        embedder.fit(protein_seqs)
        assert embedder.sequence_type == "protein"
        assert embedder.alphabet == "ACDEFGHIKLMNPQRSTVWY"
        assert embedder.alphabet_size == 20

        # DNA sequences
        dna_seqs = ["ACGT", "TGCA", "AATT"]
        embedder = OneHotEmbedder()
        embedder.fit(dna_seqs)
        assert embedder.sequence_type == "dna"
        assert embedder.alphabet == "ACGT"
        assert embedder.alphabet_size == 4

        # Explicit sequence type
        embedder = OneHotEmbedder(sequence_type="rna")
        embedder.fit(["ACGU", "UGCA"])
        assert embedder.sequence_type == "rna"
        assert embedder.alphabet == "ACGU"
        assert embedder.alphabet_size == 4

    def test_one_hot_encode(self):
        """Test one-hot encoding a single sequence."""
        # DNA sequence
        embedder = OneHotEmbedder(sequence_type="dna")
        embedder.fit(["ACGT"])

        # "ACGT" with 4 letters in alphabet = 4x4 matrix (flattened to 16 values)
        embedding = embedder._one_hot_encode("ACGT")
        assert embedding.shape == (16,)  # 4 positions * 4 letters

        # One-hot encoding should have exactly one 1 per position
        embedding_2d = embedding.reshape(4, 4)
        assert np.sum(embedding_2d) == 4  # One 1 per position
        assert np.array_equal(np.sum(embedding_2d, axis=1), np.ones(4))

        # Check correct positions have 1s
        # A should be encoded as [1,0,0,0]
        # C should be encoded as [0,1,0,0]
        # G should be encoded as [0,0,1,0]
        # T should be encoded as [0,0,0,1]
        expected = np.eye(4).flatten()
        assert np.array_equal(embedding, expected)

    def test_transform(self):
        """Test transforming multiple sequences."""
        embedder = OneHotEmbedder(sequence_type="protein")
        embedder.fit(["ACDEF", "GHIKL"])

        # Transform multiple sequences
        embeddings = embedder.transform(["ACDEF", "GHIKL"])

        # With alphabet of 20 amino acids and length 5, each embedding should be 100
        assert embeddings.shape == (2, 100)  # 2 sequences, 5 positions * 20 amino acids

    def test_fit_transform(self):
        """Test fit_transform method."""
        embedder = OneHotEmbedder()
        sequences = ["ACGT", "TGCA"]

        # fit_transform should do both operations
        embeddings = embedder.fit_transform(sequences)

        # Should have fitted
        assert embedder.sequence_type == "dna"
        assert embedder.alphabet == "ACGT"

        # Should have transformed
        assert embeddings.shape == (2, 16)  # 2 sequences, 4 positions * 4 nucleotides


def test_get_embedder():
    """Test the embedder factory function."""
    # Get one-hot embedder
    embedder = get_embedder("one-hot")
    assert isinstance(embedder, OneHotEmbedder)

    # Test invalid method
    with pytest.raises(ValueError):
        get_embedder("invalid-method")
