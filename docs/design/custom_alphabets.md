# Custom Alphabets Design Document

## Overview

This document outlines the design for enhancing fast-seqfunc with support for custom alphabets, particularly focusing on handling mixed-length characters and various sequence storage formats. This feature enables the library to work with non-standard sequence types, such as chemically modified amino acids, custom nucleotides, or integer-based sequence representations.

## Current Implementation

The current implementation in fast-seqfunc handles alphabets in a straightforward manner:

1. Alphabets are represented as instances of the `Alphabet` class that encapsulate tokens and tokenization rules.
2. Sequences can be encoded using various tokenization strategies (character-based, delimited, or regex-based).
3. The `OneHotEmbedder` uses alphabets to transform sequences into one-hot encodings for model training.
4. Pre-defined alphabets are available for common sequence types (protein, DNA, RNA).
5. Custom alphabets are supported through the `Alphabet` class.
6. Sequences of different lengths can be padded to the maximum length with a configurable gap character.

## Alphabet Class

The `Alphabet` class is at the core of the custom alphabets implementation:

```python
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
    )

    @property
    def size(self) -> int:
        """Get the number of unique tokens in the alphabet."""

    def tokenize(self, sequence: str) -> List[str]:
        """Convert a sequence string to tokens.

        :param sequence: The input sequence
        :return: List of tokens
        """

    def pad_sequence(self, sequence: str, length: int) -> str:
        """Pad a sequence to the specified length.

        :param sequence: The sequence to pad
        :param length: Target length
        :return: Padded sequence
        """

    def tokens_to_sequence(self, tokens: List[str]) -> str:
        """Convert tokens back to a sequence string.

        :param tokens: List of tokens
        :return: Sequence string
        """

    def indices_to_sequence(
        self, indices: Sequence[int], delimiter: Optional[str] = None
    ) -> str:
        """Convert a list of token indices back to a sequence string.

        :param indices: List of token indices
        :param delimiter: Optional delimiter to use (overrides the alphabet's default)
        :return: Sequence string
        """

    def encode_to_indices(self, sequence: str) -> List[int]:
        """Convert a sequence string to token indices.

        :param sequence: The input sequence
        :return: List of token indices
        """

    def decode_from_indices(
        self, indices: Sequence[int], delimiter: Optional[str] = None
    ) -> str:
        """Decode token indices back to a sequence string.

        This is an alias for indices_to_sequence.

        :param indices: List of token indices
        :param delimiter: Optional delimiter to use
        :return: Sequence string
        """

    def validate_sequence(self, sequence: str) -> bool:
        """Check if a sequence can be fully tokenized with this alphabet.

        :param sequence: The sequence to validate
        :return: True if sequence is valid, False otherwise
        """

    @classmethod
    def from_config(cls, config: Dict) -> "Alphabet":
        """Create an Alphabet instance from a configuration dictionary.

        :param config: Dictionary with alphabet configuration
        :return: Alphabet instance
        """

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Alphabet":
        """Load an alphabet from a JSON file.

        :param path: Path to the JSON configuration file
        :return: Alphabet instance
        """

    def to_dict(self) -> Dict:
        """Convert the alphabet to a dictionary for serialization.

        :return: Dictionary representation
        """

    def to_json(self, path: Union[str, Path]) -> None:
        """Save the alphabet to a JSON file.

        :param path: Path to save the configuration
        """

    @classmethod
    def protein(cls, gap_character: str = "-") -> "Alphabet":
        """Create a standard protein alphabet.

        :param gap_character: Character to use for padding (default: "-")
        :return: Alphabet for standard amino acids
        """

    @classmethod
    def dna(cls, gap_character: str = "-") -> "Alphabet":
        """Create a standard DNA alphabet.

        :param gap_character: Character to use for padding (default: "-")
        :return: Alphabet for DNA
        """

    @classmethod
    def rna(cls, gap_character: str = "-") -> "Alphabet":
        """Create a standard RNA alphabet.

        :param gap_character: Character to use for padding (default: "-")
        :return: Alphabet for RNA
        """

    @classmethod
    def integer(
        cls, max_value: int, gap_value: str = "-1", gap_character: str = "-"
    ) -> "Alphabet":
        """Create an integer-based alphabet (0 to max_value).

        :param max_value: Maximum integer value (inclusive)
        :param gap_value: String representation of the gap value (default: "-1")
        :param gap_character: Character to use for padding in string representation
            (default: "-")
        :return: Alphabet with integer tokens
        """
```

## OneHotEmbedder Implementation

The `OneHotEmbedder` class works with the `Alphabet` class to create one-hot encodings:

```python
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
    )

    @property
    def alphabet(self):
        """Get the alphabet, supporting both old and new API."""

    def fit(self, sequences: Union[List[str], pd.Series]) -> "OneHotEmbedder":
        """Determine alphabet and set up the embedder.

        :param sequences: Sequences to fit to
        :return: Self for chaining
        """

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

    def fit_transform(
        self, sequences: Union[List[str], pd.Series]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Fit and transform in one step.

        :param sequences: Sequences to encode
        :return: Array of one-hot encodings if pad_sequences=True,
            otherwise list of arrays
        """
```

## Helper Functions

### get_embedder

```python
def get_embedder(method: str, **kwargs) -> OneHotEmbedder:
    """Get an embedder instance based on method name.

    Currently only supports one-hot encoding.

    :param method: Embedding method (only "one-hot" supported)
    :param kwargs: Additional arguments to pass to the embedder
    :return: Configured embedder
    """
```

### infer_alphabet

```python
def infer_alphabet(
    sequences: List[str], delimiter: Optional[str] = None, gap_character: str = "-"
) -> Alphabet:
    """Infer an alphabet from a list of sequences.

    :param sequences: List of sequences to analyze
    :param delimiter: Optional delimiter used in sequences
    :param gap_character: Character to use for padding
    :return: Inferred Alphabet
    """
```

## Usage Examples

### Creating Custom Alphabets

```python
# Standard alphabets
protein_alphabet = Alphabet.protein()
dna_alphabet = Alphabet.dna()
rna_alphabet = Alphabet.rna()

# Custom alphabet with standard and modified amino acids
aa_tokens = list("ACDEFGHIKLMNPQRSTVWY") + ["pS", "pT", "pY", "me3K"]
mod_aa_alphabet = Alphabet(
    tokens=aa_tokens,
    name="modified_aa",
    gap_character="X"
)

# Integer alphabet (0-29 with -1 as gap value)
int_alphabet = Alphabet.integer(max_value=29, gap_value="-1")

# Custom alphabet from configuration
alphabet = Alphabet.from_json("path/to/alphabet_config.json")
```

### Using the OneHotEmbedder

```python
# Auto-detect sequence type
embedder = get_embedder("one-hot")
embeddings = embedder.fit_transform(sequences)

# Specify sequence type
embedder = get_embedder("one-hot", sequence_type="protein", pad_sequences=True)
embeddings = embedder.fit_transform(sequences)

# Use custom alphabet
embedder = get_embedder("one-hot", alphabet=mod_aa_alphabet)
embeddings = embedder.fit_transform(sequences)

# Control padding behavior
embedder = get_embedder("one-hot", max_length=10, pad_sequences=True, gap_character="X")
embeddings = embedder.fit_transform(sequences)
```

### Working with Sequences of Different Lengths

```python
# Sequences of different lengths
sequences = ["ACDE", "KLMNPQR", "ST"]
embedder = OneHotEmbedder(sequence_type="protein", pad_sequences=True)
embeddings = embedder.fit_transform(sequences)
# Sequences are padded to length 7: "ACDE---", "KLMNPQR", "ST-----"

# Disable padding (returns a list of arrays of different sizes)
embedder = OneHotEmbedder(sequence_type="protein", pad_sequences=False)
embedding_list = embedder.fit_transform(sequences)
```

### Handling Special Sequence Types

#### Chemically Modified Amino Acids

```python
# Amino acids with modifications
aa_tokens = list("ACDEFGHIKLMNPQRSTVWY") + ["pS", "pT", "pY", "me3K", "X"]
mod_aa_alphabet = Alphabet(
    tokens=aa_tokens,
    name="modified_aa",
    gap_character="X"
)

# Example sequences with modified AAs
sequences = ["ACDEpS", "KLMme3KNP", "QR"]
embedder = OneHotEmbedder(alphabet=mod_aa_alphabet, pad_sequences=True)
embeddings = embedder.fit_transform(sequences)
```

#### Integer-Based Sequences

```python
# Integer representation with comma delimiter
int_alphabet = Alphabet.integer(max_value=29, gap_value="-1")

# Example sequences as comma-separated integers
sequences = ["0,1,2", "10,11,12,25,14", "15,16"]
embedder = OneHotEmbedder(alphabet=int_alphabet, pad_sequences=True)
embeddings = embedder.fit_transform(sequences)
```

### Integration with Model Training

```python
# Create a custom alphabet
alphabet = Alphabet.integer(max_value=10)

# Get the embedder with the custom alphabet
embedder = get_embedder("one-hot", alphabet=alphabet)

# Embed sequences
X_train_embedded = embedder.fit_transform(train_df[sequence_col])

# Create column names for the embedded features
embed_cols = [f"embed_{i}" for i in range(X_train_embedded.shape[1])]

# Create DataFrame for model training
train_processed = pd.DataFrame(X_train_embedded, columns=embed_cols)
train_processed["target"] = train_df[target_col].values

# Now train your model with train_processed...
```

## Alphabet Configuration Format

You can save and load alphabet configurations using JSON files:

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
  "gap_character": "-"
}
```

## Key Features

- **Flexible Tokenization**: Support for single-character, multi-character, and delimited tokens
- **Custom Alphabets**: Define your own token sets for any sequence type
- **Gap Handling**: Configurable padding for sequences of different lengths
- **Standard Bioinformatics Alphabets**: Built-in support for protein, DNA, and RNA
- **Integer Sequences**: Special support for integer-based sequence representations
- **Serialization**: Save and load alphabet configurations as JSON
- **Automatic Type Detection**: Automatically infer sequence type from content

## Conclusion

The custom alphabets implementation in `fast-seqfunc` provides a flexible, robust solution for handling various sequence types and tokenization schemes. This design enables working with non-standard sequence types, mixed-length characters, and integer-based sequences in a clean, consistent way.
