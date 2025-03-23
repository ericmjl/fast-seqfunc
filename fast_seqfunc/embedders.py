"""Sequence embedding methods for fast-seqfunc.

This module implements various ways to convert protein or nucleotide sequences
into numerical representations (embeddings) that can be used as input for ML models.
"""

import hashlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from lazy_loader import lazy
from loguru import logger

np = lazy.load("numpy")
pd = lazy.load("pandas")


class SequenceEmbedder(ABC):
    """Abstract base class for sequence embedding methods.

    :param cache_dir: Directory to cache embeddings
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    @abstractmethod
    def _embed_sequence(self, sequence: str) -> np.ndarray:
        """Embed a single sequence.

        :param sequence: Protein or nucleotide sequence
        :return: Embedding vector
        """
        pass

    def _get_cache_path(self, sequence: str) -> Optional[Path]:
        """Get the cache file path for a sequence.

        :param sequence: Sequence to generate cache path for
        :return: Path to cache file or None if caching is disabled
        """
        if self.cache_dir is None:
            return None

        # Generate a hash of the sequence for the filename
        h = hashlib.md5(sequence.encode()).hexdigest()
        return self.cache_dir / f"{self.__class__.__name__}_{h}.pkl"

    def _load_from_cache(self, sequence: str) -> Optional[np.ndarray]:
        """Try to load embedding from cache.

        :param sequence: Sequence to load embedding for
        :return: Cached embedding or None if not cached
        """
        if self.cache_dir is None:
            return None

        cache_path = self._get_cache_path(sequence)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")

        return None

    def _save_to_cache(self, sequence: str, embedding: np.ndarray) -> None:
        """Save embedding to cache.

        :param sequence: Sequence the embedding was generated for
        :param embedding: Embedding to cache
        """
        if self.cache_dir is None:
            return

        cache_path = self._get_cache_path(sequence)
        if cache_path:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

    def transform(self, sequences: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform sequences to embeddings.

        :param sequences: List or Series of sequences to embed
        :return: Array of embeddings
        """
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()

        embeddings = []
        for sequence in sequences:
            # Try to load from cache first
            embedding = self._load_from_cache(sequence)

            # If not in cache, compute and cache
            if embedding is None:
                embedding = self._embed_sequence(sequence)
                self._save_to_cache(sequence, embedding)

            embeddings.append(embedding)

        return np.vstack(embeddings)

    def fit(self, sequences: Union[List[str], pd.Series]) -> "SequenceEmbedder":
        """Fit the embedder to the sequences (no-op for most embedders).

        :param sequences: Sequences to fit to
        :return: Self for chaining
        """
        return self

    def fit_transform(self, sequences: Union[List[str], pd.Series]) -> np.ndarray:
        """Fit the embedder and transform sequences in one step.

        :param sequences: Sequences to fit and transform
        :return: Array of embeddings
        """
        return self.fit(sequences).transform(sequences)


class OneHotEmbedder(SequenceEmbedder):
    """One-hot encoding for protein or nucleotide sequences.

    :param sequence_type: Type of sequences to encode
    :param max_length: Maximum sequence length (will pad/truncate to this length)
    :param padding: Whether to pad at the beginning or end of sequences
    :param truncating: Whether to truncate at the beginning or end of sequences
    :param cache_dir: Directory to cache embeddings
    """

    def __init__(
        self,
        sequence_type: Literal["protein", "dna", "rna", "auto"] = "auto",
        max_length: Optional[int] = None,
        padding: Literal["pre", "post"] = "post",
        truncating: Literal["pre", "post"] = "post",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.sequence_type = sequence_type
        self.max_length = max_length
        self.padding = padding
        self.truncating = truncating
        self.alphabet = None
        self.alphabet_size = None

    def fit(self, sequences: Union[List[str], pd.Series]) -> "OneHotEmbedder":
        """Determine alphabet and max length from sequences.

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

        # Determine max length if not specified
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences)

        return self

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

    def _embed_sequence(self, sequence: str) -> np.ndarray:
        """Convert a sequence to one-hot encoding.

        :param sequence: Sequence to encode
        :return: One-hot encoded matrix
        """
        sequence = sequence.upper()

        # Handle sequences longer than max_length
        if self.max_length is not None and len(sequence) > self.max_length:
            if self.truncating == "pre":
                sequence = sequence[-self.max_length :]
            else:
                sequence = sequence[: self.max_length]

        # Create empty matrix
        length = (
            min(len(sequence), self.max_length) if self.max_length else len(sequence)
        )
        one_hot = np.zeros((length, self.alphabet_size))

        # Fill in one-hot matrix
        for i, char in enumerate(sequence[:length]):
            if char in self.alphabet:
                idx = self.alphabet.index(char)
                one_hot[i, idx] = 1

        # Handle padding if needed
        if self.max_length is not None and len(sequence) < self.max_length:
            padding_length = self.max_length - len(sequence)
            padding_matrix = np.zeros((padding_length, self.alphabet_size))

            if self.padding == "pre":
                one_hot = np.vstack((padding_matrix, one_hot))
            else:
                one_hot = np.vstack((one_hot, padding_matrix))

        # Flatten for simpler ML model input
        return one_hot.flatten()


class CARPEmbedder(SequenceEmbedder):
    """CARP embeddings for protein sequences.

    :param model_name: Name of CARP model to use
    :param cache_dir: Directory to cache embeddings
    """

    def __init__(
        self,
        model_name: str = "carp_600k",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.model_name = model_name
        self.model = None

    def fit(self, sequences: Union[List[str], pd.Series]) -> "CARPEmbedder":
        """Load the CARP model if not already loaded.

        :param sequences: Sequences (not used for fitting)
        :return: Self for chaining
        """
        if self.model is None:
            try:
                # Defer import to avoid dependency if not used
                # This will be implemented when adding the actual CARP dependency
                raise ImportError("CARP is not yet implemented")
            except ImportError:
                logger.warning(
                    "CARP embeddings not available. Please install the CARP package:"
                    "\npip install git+https://github.com/microsoft/protein-sequence-models.git"
                )
                raise
        return self

    def _embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate CARP embedding for a sequence.

        :param sequence: Protein sequence
        :return: CARP embedding vector
        """
        # This is a placeholder that will be implemented when adding CARP
        raise NotImplementedError("CARP embeddings not yet implemented")


class ESM2Embedder(SequenceEmbedder):
    """ESM2 embeddings for protein sequences.

    :param model_name: Name of ESM2 model to use
    :param layer: Which layer's embeddings to use (-1 for last layer)
    :param cache_dir: Directory to cache embeddings
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        layer: int = -1,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.model_name = model_name
        self.layer = layer
        self.model = None
        self.tokenizer = None

    def fit(self, sequences: Union[List[str], pd.Series]) -> "ESM2Embedder":
        """Load the ESM2 model if not already loaded.

        :param sequences: Sequences (not used for fitting)
        :return: Self for chaining
        """
        if self.model is None:
            try:
                # Defer import to avoid dependency if not used
                # This will be implemented when adding the actual ESM2 dependency
                raise ImportError("ESM2 is not yet implemented")
            except ImportError:
                logger.warning(
                    "ESM2 embeddings not available. Please install the ESM package:"
                    "\npip install fair-esm"
                )
                raise
        return self

    def _embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate ESM2 embedding for a sequence.

        :param sequence: Protein sequence
        :return: ESM2 embedding vector
        """
        # This is a placeholder that will be implemented when adding ESM2
        raise NotImplementedError("ESM2 embeddings not yet implemented")


def get_embedder(
    method: str,
    **kwargs: Any,
) -> SequenceEmbedder:
    """Factory function to get embedder by name.

    :param method: Name of embedding method
    :param kwargs: Additional arguments to pass to embedder
    :return: SequenceEmbedder instance
    """
    if method == "one-hot":
        return OneHotEmbedder(**kwargs)
    elif method == "carp":
        return CARPEmbedder(**kwargs)
    elif method == "esm2":
        return ESM2Embedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding method: {method}")
