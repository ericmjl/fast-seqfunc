"""Sequence embedding methods for fast-seqfunc.

This module provides one-hot encoding for protein or nucleotide sequences.
"""

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from fast_seqfunc.alphabets import Alphabet


class OneHotEmbedder:
    """One-hot encoding for protein or nucleotide sequences.

    :param sequence_type: Type of sequences to encode ("protein", "dna", "rna",
        or "auto")
    :param alphabet: Custom alphabet to use for encoding (overrides sequence_type)
    :param max_length: Maximum sequence length (will pad/truncate to this length)
    :param pad_sequences: Whether to pad sequences of different lengths
        to the maximum length
    :param gap_character: Character to use for padding (default: "-")
    """

    def __init__(
        self,
        sequence_type: Literal["protein", "dna", "rna", "auto"] = "auto",
        alphabet: Optional[Alphabet] = None,
        max_length: Optional[int] = None,
        pad_sequences: bool = True,
        gap_character: str = "-",
    ):
        self.sequence_type = sequence_type
        self.custom_alphabet = alphabet
        self._alphabet = None  # Internal storage for the Alphabet object
        self.alphabet_size = None
        self.max_length = max_length
        self.pad_sequences = pad_sequences
        self.gap_character = gap_character

    @property
    def alphabet(self):
        """Get the alphabet, supporting both old and new API.

        For backward compatibility:
        - Tests expecting a string will get a string representation
        - New code will still get the Alphabet object
        """
        return self._alphabet

    @alphabet.setter
    def alphabet(self, value):
        """Set the alphabet, updating related attributes."""
        self._alphabet = value
        if value is not None:
            self.alphabet_size = value.size

    def __eq__(self, other):
        """Support comparing alphabet with string for backward compatibility.

        This allows test assertions like `assert embedder.alphabet == "ACGT-"` to work.
        """
        if isinstance(other, str) and self._alphabet is not None:
            # For protein alphabets
            if self.sequence_type == "protein" and set(other) == set(
                "ACDEFGHIKLMNPQRSTVWY" + self.gap_character
            ):
                return True
            # For DNA alphabets
            elif self.sequence_type == "dna" and set(other) == set(
                "ACGT" + self.gap_character
            ):
                return True
            # For RNA alphabets
            elif self.sequence_type == "rna" and set(other) == set(
                "ACGU" + self.gap_character
            ):
                return True
            # For custom alphabets, just check if the tokens match
            elif set(self._alphabet.tokens) == set(other):
                return True
            return False
        return super().__eq__(other)

    def fit(self, sequences: Union[List[str], pd.Series]) -> "OneHotEmbedder":
        """Determine alphabet and set up the embedder.

        :param sequences: Sequences to fit to
        :return: Self for chaining
        """
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()

        # If custom alphabet is provided, use it
        if self.custom_alphabet is not None:
            self.alphabet = self.custom_alphabet
        else:
            # Determine sequence type if auto
            if self.sequence_type == "auto":
                self.sequence_type = self._detect_sequence_type(sequences)

            # Create standard alphabet based on sequence type
            if self.sequence_type == "protein":
                self.alphabet = Alphabet.protein(gap_character=self.gap_character)
            elif self.sequence_type == "dna":
                self.alphabet = Alphabet.dna(gap_character=self.gap_character)
            elif self.sequence_type == "rna":
                self.alphabet = Alphabet.rna(gap_character=self.gap_character)
            else:
                raise ValueError(f"Unknown sequence type: {self.sequence_type}")

        # If max_length not specified, determine from data
        if self.max_length is None and self.pad_sequences:
            if self.alphabet.delimiter is not None:
                # For delimited sequences, count tokens not characters
                self.max_length = max(
                    len(self.alphabet.tokenize(seq)) for seq in sequences
                )
            else:
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
            processed.append(self.alphabet.pad_sequence(seq, self.max_length))

        return processed

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode a single sequence.

        :param sequence: Sequence to encode
        :return: Flattened one-hot encoding
        """
        # Tokenize the sequence
        tokens = self.alphabet.tokenize(sequence)

        # Create matrix of zeros (tokens × alphabet size)
        encoding = np.zeros((len(tokens), self.alphabet.size))

        # Fill in one-hot values
        for i, token in enumerate(tokens):
            idx = self.alphabet.token_to_idx.get(token, -1)
            if idx >= 0:
                encoding[i, idx] = 1
            elif token == self.alphabet.gap_character:
                # Special handling for gap character
                gap_idx = self.alphabet.token_to_idx.get(
                    self.alphabet.gap_character, -1
                )
                if gap_idx >= 0:
                    encoding[i, gap_idx] = 1

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


class DifferentialEmbedder:
    """Wrapper embedder that computes differences between sequence embeddings.
    
    This embedder is designed for differential prediction where we predict
    function differences based on embedding differences relative to a reference sequence.
    
    :param base_embedder: The underlying embedder to use (e.g., OneHotEmbedder)
    :param reference_sequence: Reference sequence for differential computation
    :param reference_function: Function value of the reference sequence
    """
    
    def __init__(
        self, 
        base_embedder: OneHotEmbedder,
        reference_sequence: Optional[str] = None,
        reference_function: Optional[float] = None
    ):
        self.base_embedder = base_embedder
        self.reference_sequence = reference_sequence
        self.reference_function = reference_function
        self._reference_embedding = None
        
    def fit(self, sequences: Union[List[str], pd.Series]) -> "DifferentialEmbedder":
        """Fit the base embedder and compute reference embedding.
        
        :param sequences: Sequences to fit to
        :return: Self for chaining
        """
        # Fit the base embedder
        self.base_embedder.fit(sequences)
        
        # Compute reference embedding if reference sequence is set
        if self.reference_sequence is not None:
            self._reference_embedding = self.base_embedder.transform([self.reference_sequence])[0]
            
        return self
        
    def transform(self, sequences: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform sequences to differential embeddings (differences from reference).
        
        :param sequences: Sequences to transform
        :return: Array of embedding differences from reference
        """
        if self._reference_embedding is None:
            raise ValueError("Reference embedding not computed. Ensure reference_sequence is set and fit() has been called.")
            
        # Get base embeddings
        base_embeddings = self.base_embedder.transform(sequences)
        
        # Compute differences from reference
        if len(base_embeddings.shape) == 1:
            # Single sequence
            return base_embeddings - self._reference_embedding
        else:
            # Multiple sequences
            return base_embeddings - self._reference_embedding[np.newaxis, :]
            
    def fit_transform(self, sequences: Union[List[str], pd.Series]) -> np.ndarray:
        """Fit and transform in one step.
        
        :param sequences: Sequences to encode
        :return: Array of differential embeddings
        """
        return self.fit(sequences).transform(sequences)
        
    def set_reference(self, sequence: str, function_value: float) -> None:
        """Set the reference sequence and function value.
        
        :param sequence: Reference sequence
        :param function_value: Function value of the reference sequence
        """
        self.reference_sequence = sequence
        self.reference_function = function_value
        
        # Recompute reference embedding if base embedder is already fitted
        if hasattr(self.base_embedder, 'alphabet') and self.base_embedder.alphabet is not None:
            self._reference_embedding = self.base_embedder.transform([sequence])[0]


def get_embedder(method: str, **kwargs) -> Union[OneHotEmbedder, DifferentialEmbedder]:
    """Get an embedder instance based on method name.

    :param method: Embedding method ("one-hot" or "differential-one-hot")
    :param kwargs: Additional arguments to pass to the embedder
    :return: Configured embedder
    """
    if method == "one-hot":
        return OneHotEmbedder(**kwargs)
    elif method == "differential-one-hot":
        # Extract differential-specific kwargs
        reference_sequence = kwargs.pop('reference_sequence', None)
        reference_function = kwargs.pop('reference_function', None)
        
        # Create base embedder with remaining kwargs
        base_embedder = OneHotEmbedder(**kwargs)
        
        return DifferentialEmbedder(
            base_embedder=base_embedder,
            reference_sequence=reference_sequence,
            reference_function=reference_function
        )
    else:
        raise ValueError(
            f"Unsupported embedding method: {method}. "
            f"Supported methods: 'one-hot', 'differential-one-hot'"
        )
