"""Sequence embedding methods for fast-seqfunc.

This module provides one-hot encoding for protein or nucleotide sequences.
"""

from typing import List, Literal, Union

import numpy as np
import pandas as pd


class OneHotEmbedder:
    """One-hot encoding for protein or nucleotide sequences.

    :param sequence_type: Type of sequences to encode ("protein", "dna", "rna",
        or "auto")
    :param max_length: Maximum sequence length (will pad/truncate to this length)
    """

    def __init__(
        self,
        sequence_type: Literal["protein", "dna", "rna", "auto"] = "auto",
    ):
        self.sequence_type = sequence_type
        self.alphabet = None
        self.alphabet_size = None

    def fit(self, sequences: Union[List[str], pd.Series]) -> "OneHotEmbedder":
        """Determine alphabet and set up the embedder.

        :param sequences: Sequences to fit to
        :return: Self for chaining
        """
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()

        # Determine sequence type if auto
        if self.sequence_type == "auto":
            self.sequence_type = self._detect_sequence_type(sequences)

        # Set alphabet based on sequence type
        if self.sequence_type == "protein":
            self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        elif self.sequence_type == "dna":
            self.alphabet = "ACGT"
        elif self.sequence_type == "rna":
            self.alphabet = "ACGU"
        else:
            raise ValueError(f"Unknown sequence type: {self.sequence_type}")

        self.alphabet_size = len(self.alphabet)
        return self

    def transform(self, sequences: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform sequences to one-hot encodings.

        :param sequences: List or Series of sequences to embed
        :return: Array of one-hot encodings
        """
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()

        if self.alphabet is None:
            raise ValueError("Embedder has not been fit yet. Call fit() first.")

        # Encode each sequence
        embeddings = []
        for sequence in sequences:
            embedding = self._one_hot_encode(sequence)
            embeddings.append(embedding)

        return np.vstack(embeddings)

    def fit_transform(self, sequences: Union[List[str], pd.Series]) -> np.ndarray:
        """Fit and transform in one step.

        :param sequences: Sequences to encode
        :return: Array of one-hot encodings
        """
        return self.fit(sequences).transform(sequences)

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode a single sequence.

        :param sequence: Sequence to encode
        :return: Flattened one-hot encoding
        """
        sequence = sequence.upper()

        # Create matrix of zeros
        encoding = np.zeros((len(sequence), self.alphabet_size))

        # Fill in one-hot values
        for i, char in enumerate(sequence):
            if char in self.alphabet:
                j = self.alphabet.index(char)
                encoding[i, j] = 1

        # Flatten to a vector
        return encoding.flatten()

    def _detect_sequence_type(self, sequences: List[str]) -> str:
        """Auto-detect sequence type from content.

        :param sequences: Sequences to analyze
        :return: Detected sequence type
        """
        # Use a sample of sequences for efficiency
        sample = sequences[:100] if len(sequences) > 100 else sequences
        sample_text = "".join(sample).upper()

        # Count characteristic letters
        u_count = sample_text.count("U")
        t_count = sample_text.count("T")
        protein_chars = "EDFHIKLMPQRSVWY"
        protein_count = sum(sample_text.count(c) for c in protein_chars)

        # Make decision based on counts
        if u_count > 0 and t_count == 0:
            return "rna"
        elif protein_count > 0:
            return "protein"
        else:
            return "dna"  # Default to DNA


def get_embedder(method: str) -> OneHotEmbedder:
    """Get an embedder instance based on method name.

    Currently only supports one-hot encoding.

    :param method: Embedding method (only "one-hot" supported)
    :return: Configured embedder
    """
    if method != "one-hot":
        raise ValueError(
            f"Unsupported embedding method: {method}. Only 'one-hot' is supported."
        )

    return OneHotEmbedder()
