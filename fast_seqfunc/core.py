"""Core functionality for fast-seqfunc.

This module implements the main API functions for training sequence-function models,
making predictions, and managing trained models.
"""

import pickle
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from lazy_loader import lazy
from loguru import logger

from fast_seqfunc.embedders import get_embedder
from fast_seqfunc.models import SequenceFunctionModel

pd = lazy.load("pandas")
np = lazy.load("numpy")


def train_model(
    train_data: Union[pd.DataFrame, Path, str],
    val_data: Optional[Union[pd.DataFrame, Path, str]] = None,
    test_data: Optional[Union[pd.DataFrame, Path, str]] = None,
    sequence_col: str = "sequence",
    target_col: str = "function",
    embedding_method: Union[
        Literal["one-hot", "carp", "esm2", "auto"], List[str]
    ] = "auto",
    model_type: Literal["regression", "classification", "multi-class"] = "regression",
    optimization_metric: Optional[str] = None,
    custom_models: Optional[List[Any]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    background: bool = False,
    **kwargs: Any,
) -> SequenceFunctionModel:
    """Train a sequence-function model with automated ML.

    This function takes sequence data with corresponding function values, embeds the
    sequences using specified method(s), and trains models using PyCaret's automated
    machine learning pipeline. The best model is returned.

    :param train_data: DataFrame or path to CSV/FASTA file with training data
    :param val_data: Optional validation data for early stopping and model selection
    :param test_data: Optional test data for final evaluation
    :param sequence_col: Column name containing sequences
    :param target_col: Column name containing target values
    :param embedding_method: Method(s) to use for embedding sequences.
                             Options: "one-hot", "carp", "esm2", or "auto".
                             Can also be a list of methods to try multiple embeddings.
    :param model_type: Type of modeling problem
    :param optimization_metric: Metric to optimize during model selection
    :param custom_models: Optional list of custom models to include in the search
    :param cache_dir: Directory to cache embeddings
    :param background: Whether to run training in background
    :param kwargs: Additional arguments to pass to PyCaret's setup function
    :return: Trained SequenceFunctionModel
    """
    if background:
        # Logic to run in background will be implemented in a future phase
        # For now, just log that this feature is coming soon
        logger.info("Background processing requested. This feature is coming soon!")

    # Load data if paths are provided
    train_df = _load_data(train_data, sequence_col, target_col)
    val_df = _load_data(val_data, sequence_col, target_col) if val_data else None
    test_df = _load_data(test_data, sequence_col, target_col) if test_data else None

    # Determine which embedding method(s) to use
    if embedding_method == "auto":
        # For now, default to one-hot. In the future, this could be more intelligent
        embedding_methods = ["one-hot"]
    elif isinstance(embedding_method, list):
        embedding_methods = embedding_method
    else:
        embedding_methods = [embedding_method]

    # Get sequence embeddings
    embeddings = {}
    for method in embedding_methods:
        logger.info(f"Generating {method} embeddings...")
        embedder = get_embedder(method, cache_dir=cache_dir)

        # Fit embedder on training data
        train_embeddings = embedder.fit_transform(train_df[sequence_col])
        embeddings[method] = {
            "train": train_embeddings,
            "val": (
                embedder.transform(val_df[sequence_col]) if val_df is not None else None
            ),
            "test": (
                embedder.transform(test_df[sequence_col])
                if test_df is not None
                else None
            ),
        }

    # Train models using PyCaret
    # This will be expanded in the implementation
    logger.info("Training models using PyCaret...")
    model = SequenceFunctionModel(
        embeddings=embeddings,
        model_type=model_type,
        optimization_metric=optimization_metric,
        embedding_method=(
            embedding_methods[0] if len(embedding_methods) == 1 else embedding_methods
        ),
    )

    # Fit the model
    model.fit(
        X_train=train_df[sequence_col],
        y_train=train_df[target_col],
        X_val=val_df[sequence_col] if val_df is not None else None,
        y_val=val_df[target_col] if val_df is not None else None,
    )

    # Evaluate on test data if provided
    if test_df is not None:
        test_results = model.evaluate(test_df[sequence_col], test_df[target_col])
        logger.info(f"Test evaluation: {test_results}")

    return model


def predict(
    model: SequenceFunctionModel,
    sequences: Union[List[str], pd.DataFrame, pd.Series],
    sequence_col: Optional[str] = "sequence",
    return_confidence: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Generate predictions for new sequences using a trained model.

    :param model: Trained SequenceFunctionModel
    :param sequences: List of sequences or DataFrame/Series containing sequences
    :param sequence_col: Column name containing sequences (if DataFrame provided)
    :param return_confidence: Whether to return confidence estimates if available
    :return: Array of predictions or tuple of (predictions, confidence)
    """
    # Extract sequences if a DataFrame is provided
    if isinstance(sequences, pd.DataFrame):
        if sequence_col not in sequences.columns:
            raise ValueError(f"Column '{sequence_col}' not found in provided DataFrame")
        sequences = sequences[sequence_col]

    # Generate predictions
    if return_confidence:
        return model.predict_with_confidence(sequences)
    else:
        return model.predict(sequences)


def load_model(model_path: Union[str, Path]) -> SequenceFunctionModel:
    """Load a trained sequence-function model from disk.

    :param model_path: Path to saved model file
    :return: Loaded SequenceFunctionModel
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if not isinstance(model, SequenceFunctionModel):
        raise TypeError("Loaded object is not a SequenceFunctionModel")

    return model


def _load_data(
    data: Optional[Union[pd.DataFrame, Path, str]],
    sequence_col: str,
    target_col: str,
) -> Optional[pd.DataFrame]:
    """Helper function to load data from various sources.

    :param data: DataFrame or path to data file
    :param sequence_col: Column name for sequences
    :param target_col: Column name for target values
    :return: DataFrame with sequence and target columns
    """
    if data is None:
        return None

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() in [".csv", ".tsv"]:
            df = pd.read_csv(path)
        elif path.suffix.lower() in [".fasta", ".fa"]:
            # This will be implemented in fast_seqfunc.utils
            # For now, we'll raise an error
            raise NotImplementedError("FASTA parsing not yet implemented")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Validate required columns
    if sequence_col not in df.columns:
        raise ValueError(f"Sequence column '{sequence_col}' not found in data")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    return df
