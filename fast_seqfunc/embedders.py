"""Sequence embedding methods for fast-seqfunc.

This module provides one-hot encoding for protein or nucleotide sequences.
"""

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd


class OneHotEmbedder:
    """One-hot encoding for protein or nucleotide sequences.

    :param sequence_type: Type of sequences to encode ("protein", "dna", "rna",
        or "auto")
    :param max_length: Maximum sequence length (will pad/truncate to this length)
    :param pad_sequences: Whether to pad sequences of different lengths
        to the maximum length
    :param gap_character: Character to use for padding (default: "-")
    """

    def __init__(
        self,
        sequence_type: Literal["protein", "dna", "rna", "auto"] = "auto",
        max_length: Optional[int] = None,
        pad_sequences: bool = True,
        gap_character: str = "-",
    ):
        self.sequence_type = sequence_type
        self.alphabet = None
        self.alphabet_size = None
        self.max_length = max_length
        self.pad_sequences = pad_sequences
        self.gap_character = gap_character

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
            self.alphabet = "ACDEFGHIKLMNPQRSTVWY" + self.gap_character
        elif self.sequence_type == "dna":
            self.alphabet = "ACGT" + self.gap_character
        elif self.sequence_type == "rna":
            self.alphabet = "ACGU" + self.gap_character
        else:
            raise ValueError(f"Unknown sequence type: {self.sequence_type}")

        self.alphabet_size = len(self.alphabet)

        # If max_length not specified, determine from data
        if self.max_length is None and self.pad_sequences:
            self.max_length = max(len(seq) for seq in sequences)

        return self

    def transform(
        self, sequences: Union[List[str], pd.Series]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Transform sequences to one-hot encodings.

        If sequences are of different lengths and pad_sequences=True, they
        will be padded to the max_length with the gap character.

        If pad_sequences=False, this returns a list of arrays of different sizes.

        :param sequences: List or Series of sequences to embed
        :return: Array of one-hot encodings if pad_sequences=True,
            otherwise list of arrays
        """
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()

        if self.alphabet is None:
            raise ValueError("Embedder has not been fit yet. Call fit() first.")

        # Preprocess sequences if padding is enabled
        if self.pad_sequences:
            sequences = self._preprocess_sequences(sequences)

        # Encode each sequence
        embeddings = []
        for sequence in sequences:
            embedding = self._one_hot_encode(sequence)
            embeddings.append(embedding)

        # If padding is enabled, stack the embeddings
        # Otherwise, return the list of embeddings
        if self.pad_sequences:
            return np.vstack(embeddings)
        else:
            return embeddings

    def fit_transform(
        self, sequences: Union[List[str], pd.Series]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Fit and transform in one step.

        :param sequences: Sequences to encode
        :return: Array of one-hot encodings if pad_sequences=True,
            otherwise list of arrays
        """
        return self.fit(sequences).transform(sequences)

    def _preprocess_sequences(self, sequences: List[str]) -> List[str]:
        """Preprocess sequences by padding or truncating.

        :param sequences: Sequences to preprocess
        :return: Preprocessed sequences
        """
        if not self.pad_sequences or self.max_length is None:
            return sequences

        processed = []
        for seq in sequences:
            if len(seq) > self.max_length:
                # Truncate
                processed.append(seq[: self.max_length])
            elif len(seq) < self.max_length:
                # Pad with gap character
                padding = self.gap_character * (self.max_length - len(seq))
                processed.append(seq + padding)
            else:
                processed.append(seq)

        return processed

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
            elif char == self.gap_character:
                # Special handling for gap character if not explicitly in alphabet
                j = self.alphabet.index(self.gap_character)
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


def get_embedder(method: str, **kwargs) -> OneHotEmbedder:
    """Get an embedder instance based on method name.

    Currently only supports one-hot encoding.

    :param method: Embedding method (only "one-hot" supported)
    :param kwargs: Additional arguments to pass to the embedder
    :return: Configured embedder
    """
    if method != "one-hot":
        raise ValueError(
            f"Unsupported embedding method: {method}. Only 'one-hot' is supported."
        )

    return OneHotEmbedder(**kwargs)
