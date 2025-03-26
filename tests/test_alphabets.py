"""Tests for the Alphabet class."""

import tempfile
from pathlib import Path

import pytest

from fast_seqfunc.alphabets import Alphabet


def test_init_with_comma_delimited_integers():
    """Test initialization with comma-delimited integers."""
    # Create an integer alphabet
    alphabet = Alphabet(
        tokens=[str(i) for i in range(10)],
        delimiter=",",
        name="integer",
        description="Integer alphabet",
        gap_character="-1",
    )

    # Check basic properties
    assert alphabet.size == 11  # 10 integers + gap
    assert alphabet.name == "integer"
    assert alphabet.description == "Integer alphabet"
    assert alphabet.delimiter == ","
    assert alphabet.gap_character == "-1"
    assert set(alphabet.tokens) == set([str(i) for i in range(10)] + ["-1"])

    # Test the integer factory method
    int_alphabet = Alphabet.integer(max_value=9)
    assert int_alphabet.size == 11  # 0-9 + gap value
    assert int_alphabet.delimiter == ","
    assert int_alphabet.gap_character == "-"
    assert "-1" in int_alphabet.tokens


@pytest.mark.parametrize(
    "sequence,expected_tokens",
    [
        ("1,2,3", ["1", "2", "3"]),
        ("10,20,30", ["10", "20", "30"]),
        ("0,1,2,3,4,5", ["0", "1", "2", "3", "4", "5"]),
        ("-1,5,10", ["-1", "5", "10"]),
        ("", []),
    ],
)
def test_tokenize_comma_delimited_integers(sequence, expected_tokens):
    """Test tokenization of comma-delimited integer sequences."""
    alphabet = Alphabet.integer(max_value=30)
    tokens = alphabet.tokenize(sequence)
    assert tokens == expected_tokens


def test_tokens_to_sequence_with_integers():
    """Test converting tokens back to a sequence with comma delimiter."""
    alphabet = Alphabet.integer(max_value=20)
    tokens = ["1", "5", "10", "15"]
    sequence = alphabet.tokens_to_sequence(tokens)
    assert sequence == "1,5,10,15"


def test_tokenize_invalid_format():
    """Test tokenizing a sequence in an invalid format."""
    alphabet = Alphabet.integer(max_value=10)
    # No commas in the sequence - should tokenize as individual characters
    tokens = alphabet.tokenize("12345")
    # Should be treated as individual characters, not as comma-delimited
    assert tokens == ["1", "2", "3", "4", "5"]


@pytest.mark.parametrize(
    "alphabet_factory,sequence,expected_indices",
    [
        (lambda: Alphabet.integer(max_value=10), "1,2,3", [1, 2, 3]),
        (lambda: Alphabet.integer(max_value=20), "10,15,20", [10, 15, 20]),
        (lambda: Alphabet.protein(), "ACGT", [0, 1, 3, 16]),
        (lambda: Alphabet.dna(), "ACGT", [0, 1, 2, 3]),
    ],
)
def test_encode_to_indices(alphabet_factory, sequence, expected_indices):
    """Test encoding a sequence to token indices."""
    alphabet = alphabet_factory()
    indices = alphabet.encode_to_indices(sequence)
    assert indices == expected_indices


@pytest.mark.parametrize(
    "alphabet_factory,sequence,expected_indices",
    [
        (lambda: Alphabet.integer(max_value=10), "1,2,3", [1, 2, 3]),
        (lambda: Alphabet.integer(max_value=20), "10,15,20", [10, 15, 20]),
        (lambda: Alphabet.protein(), "ACGT", [0, 1, 3, 16]),
        (lambda: Alphabet.dna(), "ACGT", [0, 1, 2, 3]),
    ],
)
def test_indices_to_sequence(alphabet_factory, sequence, expected_indices):
    """Test converting indices back to a sequence."""
    alphabet = alphabet_factory()
    decoded = alphabet.indices_to_sequence(expected_indices)
    # Check if the decoded sequence matches the original after tokenization
    assert alphabet.tokenize(decoded) == alphabet.tokenize(sequence)


@pytest.mark.parametrize(
    "alphabet_factory,sequence,expected_indices",
    [
        (lambda: Alphabet.integer(max_value=10), "1,2,3", [1, 2, 3]),
        (lambda: Alphabet.integer(max_value=20), "10,15,20", [10, 15, 20]),
        (lambda: Alphabet.protein(), "ACGT", [0, 1, 3, 16]),
        (lambda: Alphabet.dna(), "ACGT", [0, 1, 2, 3]),
    ],
)
def test_roundtrip_encoding(alphabet_factory, sequence, expected_indices):
    """Test round-trip encoding and decoding."""
    alphabet = alphabet_factory()
    indices = alphabet.encode_to_indices(sequence)
    decoded = alphabet.decode_from_indices(indices)
    assert alphabet.tokenize(decoded) == alphabet.tokenize(sequence)


@pytest.mark.parametrize(
    "alphabet_factory,valid_sequence,invalid_sequence",
    [
        (lambda: Alphabet.integer(max_value=10), "1,2,3,10", "1,2,3,11"),
        (lambda: Alphabet.protein(), "ACDEFG", "ACDEFGB"),
        (lambda: Alphabet.dna(), "ACGT", "ACGTU"),
    ],
)
def test_validate_valid_sequence(alphabet_factory, valid_sequence, invalid_sequence):
    """Test validation of a valid sequence."""
    alphabet = alphabet_factory()
    assert alphabet.validate_sequence(valid_sequence) is True


@pytest.mark.parametrize(
    "alphabet_factory,valid_sequence,invalid_sequence",
    [
        (lambda: Alphabet.integer(max_value=10), "1,2,3,10", "1,2,3,11"),
        (lambda: Alphabet.protein(), "ACDEFG", "ACDEFGB"),
        (lambda: Alphabet.dna(), "ACGT", "ACGTU"),
    ],
)
def test_validate_invalid_sequence(alphabet_factory, valid_sequence, invalid_sequence):
    """Test validation of an invalid sequence."""
    alphabet = alphabet_factory()
    assert alphabet.validate_sequence(invalid_sequence) is False


@pytest.mark.parametrize(
    "alphabet_factory,sequence,target_length,expected_padded",
    [
        (lambda: Alphabet.integer(max_value=10), "1,2,3", 5, "1,2,3,-1,-1"),
        (lambda: Alphabet.protein(), "ACDEF", 8, "ACDEF---"),
        (lambda: Alphabet.dna(), "ACGT", 6, "ACGT--"),
        (lambda: Alphabet.integer(max_value=20), "10,15,20", 2, "10,15"),
    ],
)
def test_pad_sequence(alphabet_factory, sequence, target_length, expected_padded):
    """Test padding a sequence to the target length."""
    alphabet = alphabet_factory()
    padded = alphabet.pad_sequence(sequence, target_length)
    assert padded == expected_padded


@pytest.mark.parametrize(
    "alphabet_factory,sequence,target_length,expected_padded",
    [
        (lambda: Alphabet.integer(max_value=10), "1,2,3", 5, "1,2,3,-1,-1"),
        (lambda: Alphabet.protein(), "ACDEF", 8, "ACDEF---"),
        (lambda: Alphabet.dna(), "ACGT", 6, "ACGT--"),
        (lambda: Alphabet.integer(max_value=20), "10,15,20", 2, "10,15"),
    ],
)
def test_truncate_sequence(alphabet_factory, sequence, target_length, expected_padded):
    """Test truncating a sequence to the target length."""
    alphabet = alphabet_factory()
    if len(alphabet.tokenize(sequence)) <= target_length:
        pytest.skip("Sequence is not long enough to test truncation")

    truncated = alphabet.pad_sequence(sequence, 1)
    assert len(alphabet.tokenize(truncated)) == 1
    assert alphabet.tokenize(truncated)[0] == alphabet.tokenize(sequence)[0]


def test_to_dict_from_config():
    """Test converting an alphabet to a dictionary and back."""
    alphabet = Alphabet.integer(max_value=15)
    config = alphabet.to_dict()

    # Check essential properties
    assert config["delimiter"] == ","
    assert config["gap_character"] == "-"
    assert config["name"] == "integer-0-15"

    # Recreate from config
    recreated = Alphabet.from_config(config)
    assert recreated.size == alphabet.size
    assert recreated.delimiter == alphabet.delimiter
    assert recreated.gap_character == alphabet.gap_character


def test_to_json_from_json():
    """Test serializing and deserializing an alphabet to/from JSON."""
    alphabet = Alphabet.integer(max_value=20)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Save to JSON
        alphabet.to_json(tmp_path)

        # Load from JSON
        loaded = Alphabet.from_json(tmp_path)

        # Check if the loaded alphabet matches the original
        assert loaded.size == alphabet.size
        assert loaded.delimiter == alphabet.delimiter
        assert loaded.gap_character == alphabet.gap_character
        assert set(loaded.tokens) == set(alphabet.tokens)
    finally:
        # Cleanup
        tmp_path.unlink(missing_ok=True)
