# Custom Alphabets Design Document

## Overview

This document outlines the design for enhancing fast-seqfunc with support for custom alphabets, particularly focusing on handling mixed-length characters and various sequence storage formats. This feature will enable the library to work with non-standard sequence types, such as chemically modified amino acids, custom nucleotides, or integer-based sequence representations.

## Current Implementation

The current implementation in fast-seqfunc handles alphabets in a straightforward but limited way:

1. Alphabets are represented as strings where each character is a valid "token" in the sequence.
2. Sequences are encoded as strings with one character per position.
3. The embedder assumes each position in the sequence maps to a single character in the alphabet.
4. Pre-defined alphabets are hardcoded for common sequence types (protein, DNA, RNA).
5. No support for custom alphabets beyond the standard ones.

This approach works well for standard biological sequences but has limitations for:

- Chemically modified amino acids
- Non-standard nucleotides
- Multi-character tokens
- Integer-based representations
- Delimited sequences

## Proposed Design

### 1. Alphabet Class

Create a dedicated `Alphabet` class to represent custom token sets:

```python
from typing import Dict, Iterable, List, Optional, Sequence, Union
from pathlib import Path
import json
import re


class Alphabet:
    """Represent a custom alphabet for sequence encoding.

    This class handles tokenization and mapping between tokens and indices,
    supporting both single character and multi-character tokens.

    :param tokens: Collection of tokens that define the alphabet
    :param delimiter: Optional delimiter used when tokenizing sequences
    :param name: Optional name for this alphabet
    :param description: Optional description
    """

    def __init__(
        self,
        tokens: Iterable[str],
        delimiter: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        # Store unique tokens in a deterministic order
        self.tokens = sorted(set(tokens))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.name = name or "custom"
        self.description = description
        self.delimiter = delimiter

        # Derive regex pattern for tokenization if no delimiter is specified
        if not delimiter and any(len(token) > 1 for token in self.tokens):
            # Sort tokens by length (longest first) to handle overlapping tokens
            sorted_tokens = sorted(self.tokens, key=len, reverse=True)
            # Escape tokens to avoid regex characters
            escaped_tokens = [re.escape(token) for token in sorted_tokens]
            self.pattern = re.compile('|'.join(escaped_tokens))
        else:
            self.pattern = None

    @property
    def size(self) -> int:
        """Get the number of unique tokens in the alphabet."""
        return len(self.tokens)

    def tokenize(self, sequence: str) -> List[str]:
        """Convert a sequence string to tokens.

        :param sequence: The input sequence
        :return: List of tokens
        """
        if self.delimiter is not None:
            # Split by delimiter and filter out empty tokens
            return [t for t in sequence.split(self.delimiter) if t]

        elif self.pattern is not None:
            # Use regex to match tokens
            return self.pattern.findall(sequence)

        else:
            # Default: treat each character as a token
            return list(sequence)

    def indices_to_sequence(self, indices: Sequence[int], delimiter: Optional[str] = None) -> str:
        """Convert a list of token indices back to a sequence string.

        :param indices: List of token indices
        :param delimiter: Optional delimiter to use (overrides the alphabet's default)
        :return: Sequence string
        """
        tokens = [self.idx_to_token.get(idx, "") for idx in indices]
        delimiter_to_use = delimiter if delimiter is not None else self.delimiter

        if delimiter_to_use is not None:
            return delimiter_to_use.join(tokens)
        else:
            return "".join(tokens)

    def encode_to_indices(self, sequence: str) -> List[int]:
        """Convert a sequence string to token indices.

        :param sequence: The input sequence
        :return: List of token indices
        """
        tokens = self.tokenize(sequence)
        return [self.token_to_idx.get(token, -1) for token in tokens]

    def decode_from_indices(self, indices: Sequence[int], delimiter: Optional[str] = None) -> str:
        """Decode token indices back to a sequence string.

        This is an alias for indices_to_sequence.

        :param indices: List of token indices
        :param delimiter: Optional delimiter to use
        :return: Sequence string
        """
        return self.indices_to_sequence(indices, delimiter)

    def validate_sequence(self, sequence: str) -> bool:
        """Check if a sequence can be fully tokenized with this alphabet.

        :param sequence: The sequence to validate
        :return: True if sequence is valid, False otherwise
        """
        tokens = self.tokenize(sequence)
        return all(token in self.token_to_idx for token in tokens)

    @classmethod
    def from_config(cls, config: Dict) -> "Alphabet":
        """Create an Alphabet instance from a configuration dictionary.

        :param config: Dictionary with alphabet configuration
        :return: Alphabet instance
        """
        return cls(
            tokens=config["tokens"],
            delimiter=config.get("delimiter"),
            name=config.get("name"),
            description=config.get("description"),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Alphabet":
        """Load an alphabet from a JSON file.

        :param path: Path to the JSON configuration file
        :return: Alphabet instance
        """
        path = Path(path)
        with open(path, "r") as f:
            config = json.load(f)
        return cls.from_config(config)

    def to_dict(self) -> Dict:
        """Convert the alphabet to a dictionary for serialization.

        :return: Dictionary representation
        """
        return {
            "tokens": self.tokens,
            "delimiter": self.delimiter,
            "name": self.name,
            "description": self.description,
        }

    def to_json(self, path: Union[str, Path]) -> None:
        """Save the alphabet to a JSON file.

        :param path: Path to save the configuration
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def protein(cls) -> "Alphabet":
        """Create a standard protein alphabet.

        :return: Alphabet for standard amino acids
        """
        return cls(
            tokens="ACDEFGHIKLMNPQRSTVWY",
            name="protein",
            description="Standard 20 amino acids",
        )

    @classmethod
    def dna(cls) -> "Alphabet":
        """Create a standard DNA alphabet.

        :return: Alphabet for DNA
        """
        return cls(
            tokens="ACGT",
            name="dna",
            description="Standard DNA nucleotides",
        )

    @classmethod
    def rna(cls) -> "Alphabet":
        """Create a standard RNA alphabet.

        :return: Alphabet for RNA
        """
        return cls(
            tokens="ACGU",
            name="rna",
            description="Standard RNA nucleotides",
        )

    @classmethod
    def integer(cls, max_value: int) -> "Alphabet":
        """Create an integer-based alphabet (0 to max_value).

        :param max_value: Maximum integer value (inclusive)
        :return: Alphabet with integer tokens
        """
        return cls(
            tokens=[str(i) for i in range(max_value + 1)],
            name=f"integer-0-{max_value}",
            description=f"Integer values from 0 to {max_value}",
            delimiter=",",
        )

    @classmethod
    def auto_detect(cls, sequences: List[str]) -> "Alphabet":
        """Automatically detect alphabet from sequences.

        :param sequences: List of example sequences
        :return: Inferred alphabet
        """
        # Sample for efficiency
        sample = sequences[:100] if len(sequences) > 100 else sequences
        sample_text = "".join(sample).upper()

        # Count characteristic letters
        u_count = sample_text.count("U")
        t_count = sample_text.count("T")
        protein_chars = "EDFHIKLMPQRSVWY"
        protein_count = sum(sample_text.count(c) for c in protein_chars)

        # Make decision based on counts
        if u_count > 0 and t_count == 0:
            return cls.rna()
        elif protein_count > 0:
            return cls.protein()
        else:
            return cls.dna()
```

### 2. Updated OneHotEmbedder

Modify the `OneHotEmbedder` class to work with the new `Alphabet` class:

```python
class OneHotEmbedder:
    """One-hot encoding for sequences with custom alphabets.

    :param alphabet: Alphabet to use for encoding (or predefined type)
    :param max_length: Maximum sequence length (will pad/truncate to this length)
    """

    def __init__(
        self,
        alphabet: Union[Alphabet, Literal["protein", "dna", "rna", "auto"]] = "auto",
        max_length: Optional[int] = None,
    ):
        if isinstance(alphabet, Alphabet):
            self.alphabet = alphabet
        elif alphabet == "protein":
            self.alphabet = Alphabet.protein()
        elif alphabet == "dna":
            self.alphabet = Alphabet.dna()
        elif alphabet == "rna":
            self.alphabet = Alphabet.rna()
        elif alphabet == "auto":
            self.alphabet = None  # Will be set during fit
        else:
            raise ValueError(f"Unknown alphabet: {alphabet}")

        self.max_length = max_length

    def fit(self, sequences: Union[List[str], pd.Series]) -> "OneHotEmbedder":
        """Determine alphabet and set up the embedder.

        :param sequences: Sequences to fit to
        :return: Self for chaining
        """
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()

        # Auto-detect alphabet if needed
        if self.alphabet is None:
            self.alphabet = Alphabet.auto_detect(sequences)

        # Determine max_length if not specified
        if self.max_length is None:
            self.max_length = max(len(self.alphabet.tokenize(seq)) for seq in sequences)

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

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode a single sequence.

        :param sequence: Sequence to encode
        :return: Flattened one-hot encoding
        """
        # Tokenize the sequence
        tokens = self.alphabet.tokenize(sequence)

        # Limit to max_length if needed
        if self.max_length is not None:
            tokens = tokens[:self.max_length]

        # Create matrix of zeros (tokens × alphabet size)
        encoding = np.zeros((len(tokens), self.alphabet.size))

        # Fill in one-hot values
        for i, token in enumerate(tokens):
            idx = self.alphabet.token_to_idx.get(token, -1)
            if idx >= 0:
                encoding[i, idx] = 1

        # Pad if needed
        if self.max_length is not None and len(tokens) < self.max_length:
            padding = np.zeros((self.max_length - len(tokens), self.alphabet.size))
            encoding = np.vstack([encoding, padding])

        # Flatten to a vector
        return encoding.flatten()
```

### 3. Configuration File Format

Define a standard JSON format for alphabet configuration files:

```json
{
  "name": "modified_amino_acids",
  "description": "Amino acids with chemical modifications",
  "tokens": ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "pS", "pT", "pY", "me3K"],
  "delimiter": null
}
```

For integer-based representations:

```json
{
  "name": "amino_acid_indices",
  "description": "Numbered amino acids (0-25) with comma delimiter",
  "tokens": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"],
  "delimiter": ","
}
```

### 4. Inferred Alphabets

Implement functionality to automatically infer alphabets from sequences:

```python
def infer_alphabet(sequences: List[str], delimiter: Optional[str] = None) -> Alphabet:
    """Infer an alphabet from a list of sequences.

    :param sequences: List of sequences to analyze
    :param delimiter: Optional delimiter used in sequences
    :return: Inferred Alphabet
    """
    all_tokens = set()

    # Create a temporary alphabet just for tokenization
    temp_alphabet = Alphabet(
        tokens=set("".join(sequences)) if delimiter is None else set(),
        delimiter=delimiter
    )

    # Extract all tokens from sequences
    for seq in sequences:
        all_tokens.update(temp_alphabet.tokenize(seq))

    # Create final alphabet with the discovered tokens
    return Alphabet(
        tokens=all_tokens,
        delimiter=delimiter,
        name="inferred",
        description=f"Alphabet inferred from {len(sequences)} sequences"
    )
```

### 5. Integration with Existing Code

1. Update the `get_embedder` function to support custom alphabets:

```python
def get_embedder(
    method: str = "one-hot",
    alphabet: Union[str, Path, Alphabet, List[str], Dict] = "auto",
    **kwargs
) -> OneHotEmbedder:
    """Get an embedder instance based on method name.

    :param method: Embedding method (currently only "one-hot" supported)
    :param alphabet: Alphabet specification, can be:
                     - Standard type string: "protein", "dna", "rna", "auto"
                     - Path to a JSON alphabet configuration
                     - Alphabet instance
                     - List of tokens to create a new alphabet
                     - Dictionary with alphabet configuration
    :return: Configured embedder
    """
    if method != "one-hot":
        raise ValueError(
            f"Unsupported embedding method: {method}. Only 'one-hot' is supported."
        )

    # Resolve the alphabet
    if isinstance(alphabet, (str, Path)) and alphabet not in ["protein", "dna", "rna", "auto"]:
        # Load from file
        alphabet = Alphabet.from_json(alphabet)
    elif isinstance(alphabet, list):
        # Create from token list
        alphabet = Alphabet(tokens=alphabet)
    elif isinstance(alphabet, dict):
        # Create from config dictionary
        alphabet = Alphabet.from_config(alphabet)

    # Pass to embedder
    return OneHotEmbedder(alphabet=alphabet, **kwargs)
```

2. Update the training workflow to handle custom alphabets:

```python
def train_model(
    train_data,
    val_data=None,
    test_data=None,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    alphabet="auto",
    model_type="regression",
    optimization_metric=None,
    **kwargs
):
    # Create or load the alphabet
    if alphabet != "auto" and not isinstance(alphabet, Alphabet):
        alphabet = get_alphabet(alphabet)  # Utility function to resolve alphabets

    # Get the appropriate embedder
    embedder = get_embedder(method=embedding_method, alphabet=alphabet)

    # Rest of the training logic...
```

## Examples of Supported Use Cases

### 1. Standard Sequences

```python
# Standard protein sequences
protein_alphabet = Alphabet.protein()
sequences = ["ACDE", "KLMNP", "QRSTV"]
embedder = OneHotEmbedder(alphabet=protein_alphabet)
embeddings = embedder.fit_transform(sequences)
```

### 2. Chemically Modified Amino Acids

```python
# Amino acids with modifications (phosphorylation, methylation)
aa_tokens = list("ACDEFGHIKLMNPQRSTVWY") + ["pS", "pT", "pY", "me3K"]
mod_aa_alphabet = Alphabet(tokens=aa_tokens, name="modified_aa")

# Example sequences with modified AAs
sequences = ["ACDEpS", "KLMme3KNP", "QRSTpYV"]
embedder = OneHotEmbedder(alphabet=mod_aa_alphabet)
embeddings = embedder.fit_transform(sequences)
```

### 3. Integer-Based Representation

```python
# Integer representation with comma delimiter
int_alphabet = Alphabet(
    tokens=[str(i) for i in range(30)],
    delimiter=",",
    name="integer_aa"
)

# Example sequences as comma-separated integers
sequences = ["0,1,2,3,20", "10,11,12,25,14", "15,16,17,18,19,21"]
embedder = OneHotEmbedder(alphabet=int_alphabet)
embeddings = embedder.fit_transform(sequences)
```

### 4. Custom Alphabet from Configuration

```python
# Load a custom alphabet from a JSON file
alphabet = Alphabet.from_json("path/to/custom_alphabet.json")
embedder = OneHotEmbedder(alphabet=alphabet)
```

### 5. Automatically Inferred Alphabet

```python
# Infer alphabet from sequences
sequences = ["ADHpK", "VWme3K", "EFGHpY"]
alphabet = infer_alphabet(sequences)
print(f"Inferred alphabet with {alphabet.size} tokens: {alphabet.tokens}")

# Use the inferred alphabet for encoding
embedder = OneHotEmbedder(alphabet=alphabet)
embeddings = embedder.fit_transform(sequences)
```

## Implementation Considerations

1. **Backwards Compatibility**: The design maintains compatibility with existing code by:
   - Keeping the same function signatures
   - Providing default alphabets that match current behavior
   - Allowing "auto" detection as currently implemented

2. **Performance**: For optimal performance:
   - Pre-compiled regex patterns for tokenization
   - Caching of tokenized sequences
   - Efficient lookups using dictionaries

3. **Extensibility**: The design allows for future extensions:
   - Support for embeddings beyond one-hot
   - Integration with custom tokenizers
   - Support for sequence generation/decoding

4. **Validation**: The design includes validation capabilities:
   - Checking if sequences can be tokenized with an alphabet
   - Reporting invalid or unknown tokens
   - Validating alphabet configurations

## Testing Strategy

1. Unit tests for the `Alphabet` class:
   - Testing all constructors and factory methods
   - Testing tokenization with various delimiters
   - Testing serialization/deserialization

2. Unit tests for the updated `OneHotEmbedder`:
   - Ensuring it works with all alphabet types
   - Testing padding and truncation
   - Testing encoding/decoding roundtrip

3. Integration tests:
   - End-to-end workflow with custom alphabets
   - Performance benchmarks for large alphabets
   - Compatibility with existing model code

## Conclusion

This design provides a flexible, maintainable solution for handling custom alphabets in fast-seqfunc, supporting a wide range of sequence representations while maintaining the simplicity of the original code. The `Alphabet` class encapsulates all the complexity of tokenization and mapping, while the embedding system remains clean and focused on its primary task of feature generation.
