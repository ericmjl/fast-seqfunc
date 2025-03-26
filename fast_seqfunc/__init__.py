"""Fast-seqfunc: A library for training sequence-function models.

This library provides tools for embedding biological sequences and training
machine learning models to predict functions from sequence data.
"""

from fast_seqfunc.alphabets import Alphabet, infer_alphabet
from fast_seqfunc.core import predict, train_model
from fast_seqfunc.embedders import OneHotEmbedder, get_embedder
from fast_seqfunc.synthetic import (
    generate_integer_function_data,
    generate_integer_sequences,
    generate_random_sequences,
    generate_sequence_function_data,
)

__all__ = [
    "train_model",
    "predict",
    "get_embedder",
    "OneHotEmbedder",
    "Alphabet",
    "infer_alphabet",
    "generate_random_sequences",
    "generate_integer_sequences",
    "generate_sequence_function_data",
    "generate_integer_function_data",
]
