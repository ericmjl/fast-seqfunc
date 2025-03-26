"""Tests for the OneHotEmbedder with custom alphabets."""

import numpy as np

from fast_seqfunc.alphabets import Alphabet
from fast_seqfunc.embedders import OneHotEmbedder


def test_init_with_integer_alphabet():
    """Test initialization with an integer alphabet."""
    # Create a custom integer alphabet
    alphabet = Alphabet.integer(max_value=10)

    # Initialize embedder with custom alphabet
    embedder = OneHotEmbedder(alphabet=alphabet)

    # Check initial state
    assert embedder.custom_alphabet is alphabet
    assert embedder.pad_sequences is True
    assert embedder.max_length is None


def test_fit_with_integer_sequences():
    """Test fit method with integer sequences."""
    # Create sequences and alphabet
    sequences = ["0,1,2", "3,4,5,6", "7,8,9,10"]
    alphabet = Alphabet.integer(max_value=10)

    # Fit embedder
    embedder = OneHotEmbedder(alphabet=alphabet)
    embedder.fit(sequences)

    # Check that the alphabet was set correctly
    assert embedder.alphabet is alphabet
    assert embedder.alphabet_size == alphabet.size
    assert embedder.max_length == 4  # Longest sequence has 4 tokens


def test_transform_with_integer_sequences():
    """Test transform method with integer sequences."""
    # Create sequences and alphabet
    sequences = ["0,1,2", "3,4,5,6", "7,8,9,10"]
    alphabet = Alphabet.integer(max_value=10)

    # Fit and transform
    embedder = OneHotEmbedder(alphabet=alphabet)
    embeddings = embedder.fit_transform(sequences)

    # Check embeddings shape
    assert embeddings.shape == (
        3,
        4 * 12,
    )  # 3 sequences, 4 tokens per sequence, 12 token types

    # Check that each embedding has the right number of 1s (one per position)
    for i in range(3):
        assert np.sum(embeddings[i]) == 4  # 4 tokens per sequence


def test_padding_of_integer_sequences():
    """Test padding of integer sequences of different lengths."""
    # Create sequences and alphabet
    sequences = ["0,1", "3,4,5", "7,8,9,10"]
    alphabet = Alphabet.integer(max_value=10)

    # Set fixed max_length
    max_length = 5
    embedder = OneHotEmbedder(alphabet=alphabet, max_length=max_length)
    embeddings = embedder.fit_transform(sequences)

    # Check embeddings shape
    assert embeddings.shape == (
        3,
        max_length * 12,
    )  # 3 sequences, 5 tokens max, 12 token types

    # Check first sequence (padded with 3 gap tokens)
    seq1_tokens = embedder.alphabet.tokenize(sequences[0]) + ["-1"] * 3
    expected_indices = [embedder.alphabet.token_to_idx[t] for t in seq1_tokens]

    # Reconstruct one-hot encoding for first sequence
    one_hot = np.zeros((max_length, embedder.alphabet_size))
    for i, idx in enumerate(expected_indices):
        one_hot[i, idx] = 1
    expected_embedding = one_hot.flatten()

    # Compare first embedding with expected
    assert np.array_equal(embeddings[0], expected_embedding)


def test_truncation_of_integer_sequences():
    """Test truncation of integer sequences longer than max_length."""
    # Create a long sequence and alphabet
    sequences = ["0,1,2,3,4,5,6,7,8,9,10"]
    alphabet = Alphabet.integer(max_value=10)

    # Set fixed max_length shorter than sequence
    max_length = 3
    embedder = OneHotEmbedder(alphabet=alphabet, max_length=max_length)
    embeddings = embedder.fit_transform(sequences)

    # Check embeddings shape
    assert embeddings.shape == (
        1,
        max_length * 12,
    )  # 1 sequence, 3 tokens max, 12 token types

    # Check truncated sequence (only first 3 tokens)
    truncated_tokens = embedder.alphabet.tokenize(sequences[0])[:max_length]
    expected_indices = [embedder.alphabet.token_to_idx[t] for t in truncated_tokens]

    # Reconstruct one-hot encoding for truncated sequence
    one_hot = np.zeros((max_length, embedder.alphabet_size))
    for i, idx in enumerate(expected_indices):
        one_hot[i, idx] = 1
    expected_embedding = one_hot.flatten()

    # Compare embedding with expected
    assert np.array_equal(embeddings[0], expected_embedding)


def test_handling_of_gap_values():
    """Test handling of gap values in integer sequences."""
    # Create sequences with gap values and alphabet
    sequences = ["0,1,-1,3", "-1,5,6", "7,8,-1"]
    alphabet = Alphabet.integer(max_value=10)

    # Fit and transform
    embedder = OneHotEmbedder(alphabet=alphabet)
    embeddings = embedder.fit_transform(sequences)

    # Check embeddings shape
    assert embeddings.shape == (3, 4 * 12)  # 3 sequences, 4 tokens max, 12 token types

    # Get the gap token index
    gap_idx = embedder.alphabet.token_to_idx["-1"]

    # Check that gap tokens are properly one-hot encoded
    # For the first sequence, position 2 should be a gap
    seq1_embedding = embeddings[0].reshape(4, 12)
    assert seq1_embedding[2, gap_idx] == 1


def test_empty_sequences():
    """Test embedding empty sequences."""
    # Create sequences with an empty sequence and alphabet
    sequences = ["0,1,2", "", "3,4,5"]
    alphabet = Alphabet.integer(max_value=5)

    # Fit and transform
    embedder = OneHotEmbedder(alphabet=alphabet)
    embeddings = embedder.fit_transform(sequences)

    # Check embeddings shape (empty sequence should be padded)
    assert embeddings.shape == (3, 3 * 7)  # 3 sequences, 3 tokens max, 7 token types

    # The empty sequence should be all gap characters
    empty_seq_embedding = embeddings[1].reshape(3, 7)
    gap_idx = embedder.alphabet.token_to_idx["-1"]
    assert np.sum(empty_seq_embedding[:, gap_idx]) == 3  # All positions should be gaps


def test_invalid_tokens():
    """Test sequences with tokens not in the alphabet."""
    # Create sequences with invalid tokens and alphabet
    sequences = ["0,1,2", "3,99,5", "6,7,8"]  # 99 is not in alphabet
    alphabet = Alphabet.integer(max_value=10)

    # Fit and transform - should not raise an error but invalid tokens
    # won't be one-hot encoded
    embedder = OneHotEmbedder(alphabet=alphabet)
    embeddings = embedder.fit_transform(sequences)

    # Check second sequence with invalid token
    seq2_embedding = embeddings[1].reshape(3, 12)

    # Position 1 should have no one-hot encoding (all zeros)
    # since 99 is not in the alphabet
    assert np.sum(seq2_embedding[1]) == 0


def test_mixed_alphabets():
    """Test with sequences using mixed alphabet types."""
    # Create sequences with mixed alphabet types
    sequences = ["0,1,2", "A,C,G,T", "3,4,5"]  # Second sequence is DNA, not integers
    alphabet = Alphabet.integer(max_value=5)

    # Fit and transform - invalid tokens in second sequence won't be encoded
    embedder = OneHotEmbedder(alphabet=alphabet)
    embeddings = embedder.fit_transform(sequences)

    # Check second sequence with non-integer tokens
    seq2_embedding = embeddings[1].reshape(
        4, 7
    )  # 7 token types in alphabet (0-5 + gap)

    # All positions should have no one-hot encoding
    # since A,C,G,T are not in the integer alphabet
    assert np.sum(seq2_embedding) == 0


def test_prepare_data_for_model():
    """Test preparing data for model training."""
    # Create a synthetic dataset with integer sequences
    sequences = [
        "0,1,2,3",
        "1,2,3,4",
        "2,3,4,5",
        "3,4,5,0",
        "4,5,0,1",
    ]
    labels = [0, 1, 2, 1, 0]  # Classification labels

    # Create alphabet and embedder
    alphabet = Alphabet.integer(max_value=5)
    embedder = OneHotEmbedder(alphabet=alphabet)

    # Embed sequences
    X = embedder.fit_transform(sequences)
    y = np.array(labels)

    # Check shapes
    assert X.shape == (5, 4 * 7)  # 5 sequences, 4 tokens, 7 token types
    assert y.shape == (5,)


def test_model_inference():
    """Test model inference with embedded sequences."""

    # Create a simple "model" for testing (just returns sum of embedding)
    class DummyModel:
        """A simple dummy model for testing embeddings.

        This class simulates a machine learning model by providing a predict method
        that returns the sum of the input features along axis 1.
        """

        def predict(self, X):
            """Make predictions by summing the input features.

            :param X: Input feature matrix
            :return: Array of predictions (sum of each row in X)
            """
            return np.sum(X, axis=1)

    # Create sequences and labels
    sequences = ["0,1,2", "3,4,5", "0,0,0"]

    # Create alphabet and embedder
    alphabet = Alphabet.integer(max_value=5)
    embedder = OneHotEmbedder(alphabet=alphabet)

    # Embed sequences
    X = embedder.fit_transform(sequences)

    # Create dummy model
    model = DummyModel()

    # Make predictions
    predictions = model.predict(X)

    # Check predictions shape
    assert predictions.shape == (3,)

    # The dummy model just sums the embeddings, so sequences with more 1s
    # in their one-hot encoding should have higher predictions
    assert predictions[0] == 3  # 3 tokens, each with one 1
    assert predictions[1] == 3  # 3 tokens, each with one 1
    assert predictions[2] == 3  # 3 tokens, each with one 1
