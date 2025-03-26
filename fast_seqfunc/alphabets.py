"""Custom alphabets for sequence encoding.

This module provides tools to work with custom alphabets, including
character-based alphabets, multi-character tokens, and delimited sequences.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


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
            self.pattern = re.compile("|".join(escaped_tokens))
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

    def indices_to_sequence(
        self, indices: Sequence[int], delimiter: Optional[str] = None
    ) -> str:
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

    def decode_from_indices(
        self, indices: Sequence[int], delimiter: Optional[str] = None
    ) -> str:
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
        return cls(
            tokens=[str(i) for i in range(max_value + 1)] + [gap_value],
            name=f"integer-0-{max_value}",
            description=(
                f"Integer values from 0 to {max_value} with gap value {gap_value}"
            ),
            delimiter=",",
            gap_character=gap_character,
        )


def infer_alphabet(
    sequences: List[str], delimiter: Optional[str] = None, gap_character: str = "-"
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
        tokens=temp_tokens, delimiter=delimiter, gap_character=gap_character
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
        gap_character=gap_character,
    )
