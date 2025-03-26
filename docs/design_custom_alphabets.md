# Custom Alphabets Design Document

## Overview

This document outlines the design for enhancing fast-seqfunc with support for custom alphabets, particularly focusing on handling mixed-length characters and various sequence storage formats. This feature will enable the library to work with non-standard sequence types, such as chemically modified amino acids, custom nucleotides, or (more generically) integer-based sequence representations.

## Current Implementation

The current implementation in fast-seqfunc handles alphabets in a straightforward but limited way:

1. Alphabets are represented as strings where each character is a valid "token" in the sequence.
2. Sequences are encoded as strings with one character per position.
3. The embedder assumes each position in the sequence maps to a single character in the alphabet.
4. Pre-defined alphabets are hardcoded for common sequence types (protein, DNA, RNA).
5. No support for custom alphabets beyond the standard ones.
6. Sequences of different lengths are padded to the maximum length with a configurable gap character.

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
    :param gap_character: Character to use for padding sequences (default: "-")
    """

    def __init__(
        self,
        tokens: Iterable[str],
        delimiter: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        gap_character: str = "-",
    ):
        # Ensure gap character is included in tokens
        all_tokens = set(tokens)
        all_tokens.add(gap_character)

        # Store unique tokens in a deterministic order
        self.tokens = sorted(all_tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.name = name or "custom"
        self.description = description
        self.delimiter = delimiter
        self.gap_character = gap_character

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

    def pad_sequence(self, sequence: str, length: int) -> str:
        """Pad a sequence to the specified length.

        :param sequence: The sequence to pad
        :param length: Target length
        :return: Padded sequence
        """
        tokens = self.tokenize(sequence)
        if len(tokens) >= length:
            # Truncate if needed
            return self.tokens_to_sequence(tokens[:length])
        else:
            # Pad with gap character
            padding_needed = length - len(tokens)
            padded_tokens = tokens + [self.gap_character] * padding_needed
            return self.tokens_to_sequence(padded_tokens)

    def tokens_to_sequence(self, tokens: List[str]) -> str:
        """Convert tokens back to a sequence string.

        :param tokens: List of tokens
        :return: Sequence string
        """
        if self.delimiter is not None:
            return self.delimiter.join(tokens)
        else:
            return "".join(tokens)

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
            gap_character=config.get("gap_character", "-"),
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
            "gap_character": self.gap_character,
        }

    def to_json(self, path: Union[str, Path]) -> None:
        """Save the alphabet to a JSON file.

        :param path: Path to save the configuration
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def protein(cls, gap_character: str = "-") -> "Alphabet":
        """Create a standard protein alphabet.

        :param gap_character: Character to use for padding (default: "-")
        :return: Alphabet for standard amino acids
        """
        return cls(
            tokens="ACDEFGHIKLMNPQRSTVWY" + gap_character,
            name="protein",
            description="Standard 20 amino acids with gap character",
            gap_character=gap_character,
        )

    @classmethod
    def dna(cls, gap_character: str = "-") -> "Alphabet":
        """Create a standard DNA alphabet.

        :param gap_character: Character to use for padding (default: "-")
        :return: Alphabet for DNA
        """
        return cls(
            tokens="ACGT" + gap_character,
            name="dna",
            description="Standard DNA nucleotides with gap character",
            gap_character=gap_character,
        )

    @classmethod
    def rna(cls, gap_character: str = "-") -> "Alphabet":
        """Create a standard RNA alphabet.

        :param gap_character: Character to use for padding (default: "-")
        :return: Alphabet for RNA
        """
        return cls(
            tokens="ACGU" + gap_character,
            name="rna",
            description="Standard RNA nucleotides with gap character",
            gap_character=gap_character,
        )

    @classmethod
    def integer(cls, max_value: int, gap_value: str = "-1", gap_character: str = "-") -> "Alphabet":
        """Create an integer-based alphabet (0 to max_value).

        :param max_value: Maximum integer value (inclusive)
        :param gap_value: String representation of the gap value (default: "-1")
        :param gap_character: Character to use for padding in string representation (default: "-")
        :return: Alphabet with integer tokens
        """
        return cls(
            tokens=[str(i) for i in range(max_value + 1)] + [gap_value],
            name=f"integer-0-{max_value}",
            description=f"Integer values from 0 to {max_value} with gap value {gap_value}",
            delimiter=",",
            gap_character=gap_character,
        )

    @classmethod
    def auto_detect(cls, sequences: List[str], gap_character: str = "-") -> "Alphabet":
        """Automatically detect alphabet from sequences.

        :param sequences: List of example sequences
        :param gap_character: Character to use for padding (default: "-")
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
            return cls.rna(gap_character=gap_character)
        elif protein_count > 0:
            return cls.protein(gap_character=gap_character)
        else:
            return cls.dna(gap_character=gap_character)
```

### 2. Updated OneHotEmbedder

Modify the `OneHotEmbedder` class to work with the new `Alphabet` class and handle padding for sequences of different lengths:

```python
class OneHotEmbedder:
    """One-hot encoding for sequences with custom alphabets.

    :param alphabet: Alphabet to use for encoding (or predefined type)
    :param max_length: Maximum sequence length (will pad/truncate to this length)
    :param pad_sequences: Whether to pad sequences of different lengths
    :param gap_character: Character to use for padding (default: "-")
    """

    def __init__(
        self,
        alphabet: Union[Alphabet, Literal["protein", "dna", "rna", "auto"]] = "auto",
        max_length: Optional[int] = None,
        pad_sequences: bool = True,
        gap_character: str = "-",
    ):
        self.pad_sequences = pad_sequences
        self.gap_character = gap_character

        if isinstance(alphabet, Alphabet):
            self.alphabet = alphabet
        elif alphabet == "protein":
            self.alphabet = Alphabet.protein(gap_character=gap_character)
        elif alphabet == "dna":
            self.alphabet = Alphabet.dna(gap_character=gap_character)
        elif alphabet == "rna":
            self.alphabet = Alphabet.rna(gap_character=gap_character)
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
            self.alphabet = Alphabet.auto_detect(
                sequences, gap_character=self.gap_character
            )

        # Determine max_length if not specified but padding is enabled
        if self.max_length is None and self.pad_sequences:
            self.max_length = max(len(self.alphabet.tokenize(seq)) for seq in sequences)

        return self

    def transform(self, sequences: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform sequences to one-hot encodings.

        If sequences are of different lengths and pad_sequences=True, they
        will be padded to max_length with the gap character.

        :param sequences: List or Series of sequences to embed
        :return: Array of one-hot encodings
        """
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()

        if self.alphabet is None:
            raise ValueError("Embedder has not been fit yet. Call fit() first.")

        # Preprocess sequences if padding is enabled
        if self.pad_sequences and self.max_length is not None:
            preprocessed = []
            for seq in sequences:
                tokens = self.alphabet.tokenize(seq)
                if len(tokens) > self.max_length:
                    # Truncate
                    tokens = tokens[:self.max_length]
                elif len(tokens) < self.max_length:
                    # Pad with gap character
                    tokens += [self.alphabet.gap_character] * (self.max_length - len(tokens))

                preprocessed.append(self.alphabet.tokens_to_sequence(tokens))
            sequences = preprocessed

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

        # Create matrix of zeros (tokens × alphabet size)
        encoding = np.zeros((len(tokens), self.alphabet.size))

        # Fill in one-hot values
        for i, token in enumerate(tokens):
            idx = self.alphabet.token_to_idx.get(token, -1)
            if idx >= 0:
                encoding[i, idx] = 1
            elif token == self.alphabet.gap_character:
                # Special handling for gap character
                gap_idx = self.alphabet.token_to_idx.get(self.alphabet.gap_character, -1)
                if gap_idx >= 0:
                    encoding[i, gap_idx] = 1

        # Flatten to a vector
        return encoding.flatten()
```

### 3. Configuration File Format

Extend the standard JSON format for alphabet configuration files to include gap character:

```json
{
  "name": "modified_amino_acids",
  "description": "Amino acids with chemical modifications",
  "tokens": ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "pS", "pT", "pY", "me3K", "-"],
  "delimiter": null,
  "gap_character": "-"
}
```

For integer-based representations:

```json
{
  "name": "amino_acid_indices",
  "description": "Numbered amino acids (0-25) with comma delimiter",
  "tokens": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "-1"],
  "delimiter": ",",
  "gap_character": "-",
  "gap_value": "-1"
}
```

### 4. Inferred Alphabets

Update the alphabet inference to include gap characters:

```python
def infer_alphabet(
    sequences: List[str],
    delimiter: Optional[str] = None,
    gap_character: str = "-"
) -> Alphabet:
    """Infer an alphabet from a list of sequences.

    :param sequences: List of sequences to analyze
    :param delimiter: Optional delimiter used in sequences
    :param gap_character: Character to use for padding
    :return: Inferred Alphabet
    """
    all_tokens = set()

    # Create a temporary alphabet just for tokenization
    temp_tokens = set("".join(sequences)) if delimiter is None else set()
    temp_tokens.add(gap_character)

    temp_alphabet = Alphabet(
        tokens=temp_tokens,
        delimiter=delimiter,
        gap_character=gap_character
    )

    # Extract all tokens from sequences
    for seq in sequences:
        all_tokens.update(temp_alphabet.tokenize(seq))

    # Ensure gap character is included
    all_tokens.add(gap_character)

    # Create final alphabet with the discovered tokens
    return Alphabet(
        tokens=all_tokens,
        delimiter=delimiter,
        name="inferred",
        description=f"Alphabet inferred from {len(sequences)} sequences",
        gap_character=gap_character
    )
```

### 5. Integration with Existing Code

1. Update the `get_embedder` function to support custom alphabets and padding:

```python
def get_embedder(
    method: str = "one-hot",
    alphabet: Union[str, Path, Alphabet, List[str], Dict] = "auto",
    max_length: Optional[int] = None,
    pad_sequences: bool = True,
    gap_character: str = "-",
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
    :param max_length: Maximum sequence length (for padding/truncation)
    :param pad_sequences: Whether to pad sequences to the same length
    :param gap_character: Character to use for padding
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
        alphabet = Alphabet(tokens=alphabet, gap_character=gap_character)
    elif isinstance(alphabet, dict):
        # Create from config dictionary
        if "gap_character" not in alphabet:
            alphabet["gap_character"] = gap_character
        alphabet = Alphabet.from_config(alphabet)

    # Pass to embedder
    return OneHotEmbedder(
        alphabet=alphabet,
        max_length=max_length,
        pad_sequences=pad_sequences,
        gap_character=gap_character,
        **kwargs
    )
```

2. Update the training workflow to handle custom alphabets and padding:

```python
def train_model(
    train_data,
    val_data=None,
    test_data=None,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    alphabet="auto",
    max_length=None,
    pad_sequences=True,
    gap_character="-",
    model_type="regression",
    optimization_metric=None,
    **kwargs
):
    # Create or load the alphabet
    if alphabet != "auto" and not isinstance(alphabet, Alphabet):
        alphabet = get_alphabet(alphabet, gap_character=gap_character)

    # Get the appropriate embedder
    embedder = get_embedder(
        method=embedding_method,
        alphabet=alphabet,
        max_length=max_length,
        pad_sequences=pad_sequences,
        gap_character=gap_character
    )

    # Rest of the training logic...
```

## Sequence Padding Implementation

A key enhancement in this design is the automatic handling of sequences with different lengths. The implementation:

1. Automatically detects the maximum sequence length during fitting (unless explicitly specified)
2. Pads shorter sequences to the maximum length using a configurable gap character (default: "-")
3. Truncates longer sequences to the maximum length if necessary
4. Ensures the gap character is included in all alphabets for consistent encoding
5. Allows disabling padding via the `pad_sequences` parameter

This approach provides several advantages:

1. **Simplified Model Training**: All sequences are encoded to the same dimensionality, which is required by most machine learning models
2. **Configurable Gap Character**: Different domains may use different symbols for gaps/padding
3. **Padding Awareness**: The embedder is aware of padding, ensuring proper handling during encoding and decoding
4. **Integration with Custom Alphabets**: The padding system works seamlessly with all alphabet types

## Examples of Supported Use Cases

### 1. Sequences with Different Lengths

```python
# Sequences of different lengths
sequences = ["ACDE", "KLMNPQR", "ST"]
embedder = OneHotEmbedder(alphabet="protein", pad_sequences=True)
embeddings = embedder.fit_transform(sequences)
# Sequences are padded to length 7: "ACDE---", "KLMNPQR", "ST-----"
```

### 2. Custom Gap Character

```python
# Using a custom gap character "X"
sequences = ["ACDE", "KLMNP", "QR"]
embedder = OneHotEmbedder(alphabet="protein", pad_sequences=True, gap_character="X")
embeddings = embedder.fit_transform(sequences)
# Sequences are padded to length 5: "ACDEXX", "KLMNP", "QRXXX"
```

### 3. Chemically Modified Amino Acids with Padding

```python
# Amino acids with modifications and variable lengths
aa_tokens = list("ACDEFGHIKLMNPQRSTVWY") + ["pS", "pT", "pY", "me3K", "X"]
mod_aa_alphabet = Alphabet(
    tokens=aa_tokens,
    name="modified_aa",
    gap_character="X"
)

# Example sequences with modified AAs of different lengths
sequences = ["ACDEpS", "KLMme3KNP", "QR"]
embedder = OneHotEmbedder(alphabet=mod_aa_alphabet, pad_sequences=True)
embeddings = embedder.fit_transform(sequences)
# Sequences are padded: "ACDEpSXX", "KLMme3KNP", "QRXXXXXX"
```

### 4. Integer-Based Representation with Custom Gap Value

```python
# Integer representation with comma delimiter and -1 as gap
int_alphabet = Alphabet(
    tokens=[str(i) for i in range(30)] + ["-1"],
    delimiter=",",
    name="integer_aa",
    gap_character="-",  # Character for string representation
    gap_value="-1"      # Value used in encoded form
)

# Example sequences as comma-separated integers of different lengths
sequences = ["0,1,2", "10,11,12,25,14", "15,16"]
embedder = OneHotEmbedder(alphabet=int_alphabet, pad_sequences=True)
embeddings = embedder.fit_transform(sequences)
# Padded with gap values: "0,1,2,-1,-1", "10,11,12,25,14", "15,16,-1,-1,-1"
```

## Implementation Considerations

1. **Backwards Compatibility**: The design maintains compatibility with existing code by:
   - Making padding behavior configurable but enabled by default
   - Providing the same function signatures with additional optional parameters
   - Using a standard gap character ("-") that's common in bioinformatics

2. **Performance**: For optimal performance with padding:
   - Precompute max length during fit to avoid recomputing for each transform
   - Use vectorized operations for padding where possible
   - Cache tokenized and padded sequences when appropriate

3. **Extensibility**: The padding system is designed for future extensions:
   - Support for different padding strategies (pre-padding vs. post-padding)
   - Integration with alignment-aware embeddings
   - Support for variable-length sequence models

## Testing Strategy

Additional tests should be added to validate the padding functionality:

1. Tests for the `Alphabet` class:
   - Test padding sequences to a specified length
   - Test inclusion of gap characters in the token set
   - Test tokenization with gap characters

2. Tests for the updated `OneHotEmbedder`:
   - Test handling of sequences with different lengths
   - Test padding with different gap characters
   - Test disable/enable padding functionality

## Conclusion

This design provides a flexible, maintainable solution for handling custom alphabets and sequences of different lengths in `fast-seqfunc`. The inclusion of automatic padding with configurable gap characters makes the library more robust and user-friendly, particularly for cases where sequences have naturally variable lengths. The `Alphabet` class encapsulates all the complexity of tokenization, mapping, and padding, while the embedding system remains clean and focused on its primary task of feature generation.
