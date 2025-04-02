"""Fast-seqfunc: A library for training sequence-function models.

This library provides tools for embedding biological sequences and training
machine learning models to predict functions from sequence data.
"""

from fast_seqfunc.alphabets import Alphabet, infer_alphabet
from fast_seqfunc.core import predict, save_model, train_model
from fast_seqfunc.embedders import OneHotEmbedder, get_embedder
from fast_seqfunc.synthetic import (
    create_content_ratio_task,
    # Biological sequence tasks
    create_g_count_task,
    create_gc_content_task,
    create_integer_classification_task,
    create_integer_interaction_task,
    create_integer_max_task,
    create_integer_multiclass_task,
    create_integer_nonlinear_composition_task,
    create_integer_nonlinear_task,
    create_integer_pattern_count_task,
    create_integer_pattern_position_task,
    create_integer_pattern_task,
    create_integer_position_interaction_task,
    create_integer_ratio_task,
    # Specialized integer tasks
    create_integer_sum_task,
    create_interaction_generic_task,
    create_interaction_task,
    create_motif_count_task,
    create_motif_position_task,
    create_nonlinear_composition_generic_task,
    create_nonlinear_composition_task,
    create_pattern_count_task,
    create_pattern_position_task,
    # Generic task generators
    create_token_count_task,
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
    # Generic task generators
    "create_token_count_task",
    "create_content_ratio_task",
    "create_pattern_position_task",
    "create_pattern_count_task",
    "create_nonlinear_composition_generic_task",
    "create_interaction_generic_task",
    # Biological sequence tasks
    "create_g_count_task",
    "create_gc_content_task",
    "create_motif_position_task",
    "create_motif_count_task",
    "create_nonlinear_composition_task",
    "create_interaction_task",
    # Specialized integer tasks
    "create_integer_sum_task",
    "create_integer_ratio_task",
    "create_integer_pattern_position_task",
    "create_integer_pattern_count_task",
    "create_integer_nonlinear_composition_task",
    "create_integer_position_interaction_task",
    "create_integer_max_task",
    "create_integer_pattern_task",
    "create_integer_nonlinear_task",
    "create_integer_interaction_task",
    "create_integer_classification_task",
    "create_integer_multiclass_task",
    "save_model",
]
