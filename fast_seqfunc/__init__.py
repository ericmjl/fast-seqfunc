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

__all__ = ["train_model", "predict", "save_model", "load_model", "evaluate_model"]
