"""Top-level API for fast-seqfunc.

This is the file from which you can do:

    from fast_seqfunc import train_model, predict, save_model, load_model

Provides a simple interface for sequence-function modeling of proteins and nucleotides.
"""

from fast_seqfunc.core import (
    evaluate_model,
    load_model,
    predict,
    save_model,
    train_model,
)

# Import synthetic data generation functions
from fast_seqfunc.synthetic import (
    create_classification_task,
    create_g_count_task,
    create_gc_content_task,
    create_interaction_task,
    create_length_dependent_task,
    create_motif_count_task,
    create_motif_position_task,
    create_multiclass_task,
    create_nonlinear_composition_task,
    generate_dataset_by_task,
)

__all__ = [
    # Core functionality
    "train_model",
    "predict",
    "save_model",
    "load_model",
    "evaluate_model",
    # Synthetic data
    "create_g_count_task",
    "create_gc_content_task",
    "create_motif_position_task",
    "create_motif_count_task",
    "create_length_dependent_task",
    "create_nonlinear_composition_task",
    "create_interaction_task",
    "create_classification_task",
    "create_multiclass_task",
    "generate_dataset_by_task",
]
