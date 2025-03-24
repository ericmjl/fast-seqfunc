"""Core functionality for fast-seqfunc.

This module implements the main API functions for training sequence-function models,
and making predictions with a simpler design using PyCaret directly.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from fast_seqfunc.embedders import get_embedder

# Global session counter for PyCaret
_session_id = 42


def train_model(
    train_data: Union[pd.DataFrame, Path, str],
    val_data: Optional[Union[pd.DataFrame, Path, str]] = None,
    test_data: Optional[Union[pd.DataFrame, Path, str]] = None,
    sequence_col: str = "sequence",
    target_col: str = "function",
    embedding_method: Literal["one-hot", "carp", "esm2"] = "one-hot",
    model_type: Literal["regression", "classification"] = "regression",
    optimization_metric: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a sequence-function model using PyCaret.

    This function takes sequence data with corresponding function values, embeds the
    sequences, and trains multiple models using PyCaret's automated ML pipeline.
    The best model is selected and returned.

    :param train_data: DataFrame or path to CSV file with training data
    :param val_data: Optional validation data (not directly used, reserved for future)
    :param test_data: Optional test data for final evaluation
    :param sequence_col: Column name containing sequences
    :param target_col: Column name containing target values
    :param embedding_method: Method to use for embedding sequences
    :param model_type: Type of modeling problem (regression or classification)
    :param optimization_metric: Metric to optimize during model selection
    :param kwargs: Additional arguments for PyCaret setup
    :return: Dictionary containing the trained model and related metadata
    """
    global _session_id

    # Load data
    train_df = _load_data(train_data, sequence_col, target_col)
    test_df = (
        _load_data(test_data, sequence_col, target_col)
        if test_data is not None
        else None
    )

    # Get embedder
    logger.info(f"Generating {embedding_method} embeddings...")
    embedder = get_embedder(embedding_method)

    # Create column names for embeddings
    X_train_embedded = embedder.fit_transform(train_df[sequence_col])
    embed_cols = [f"embed_{i}" for i in range(X_train_embedded.shape[1])]

    # Create DataFrame with embeddings
    train_processed = pd.DataFrame(X_train_embedded, columns=embed_cols)
    train_processed["target"] = train_df[target_col].values

    # Setup PyCaret environment
    logger.info(f"Setting up PyCaret for {model_type} modeling...")

    try:
        if model_type == "regression":
            from pycaret.regression import compare_models, finalize_model, setup

            # Setup regression environment
            setup_args = {
                "data": train_processed,
                "target": "target",
                "session_id": _session_id,
                "verbose": False,
            }

            # Setup PyCaret environment - optimization metric is passed to
            # compare_models, not setup
            if optimization_metric:
                logger.info(f"Will optimize for metric: {optimization_metric}")

            # Setup PyCaret environment
            setup(**setup_args)

            # Train multiple models and select the best one
            logger.info("Training and comparing multiple models...")
            compare_args = {"n_select": 1}

            # Add sort parameter if optimization metric is specified
            if optimization_metric:
                compare_args["sort"] = optimization_metric

            models = compare_models(**compare_args)

            # Finalize model (train on all data)
            logger.info("Finalizing best model...")
            final_model = finalize_model(models)

        elif model_type == "classification":
            from pycaret.classification import compare_models, finalize_model, setup

            # Setup classification environment
            setup_args = {
                "data": train_processed,
                "target": "target",
                "session_id": _session_id,
                "verbose": False,
            }

            # Setup PyCaret environment - optimization metric is passed to
            # compare_models, not setup
            if optimization_metric:
                logger.info(f"Will optimize for metric: {optimization_metric}")

            # Setup PyCaret environment
            setup(**setup_args)

            # Train multiple models and select the best one
            logger.info("Training and comparing multiple models...")
            compare_args = {"n_select": 1}

            # Add sort parameter if optimization metric is specified
            if optimization_metric:
                compare_args["sort"] = optimization_metric

            models = compare_models(**compare_args)

            # Finalize model (train on all data)
            logger.info("Finalizing best model...")
            final_model = finalize_model(models)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Increment session ID for next run
        _session_id += 1

        # Evaluate on test data if provided
        if test_df is not None:
            logger.info("Evaluating on test data...")
            test_results = evaluate_model(
                final_model,
                test_df[sequence_col],
                test_df[target_col],
                embedder=embedder,
                model_type=model_type,
                embed_cols=embed_cols,
            )
            logger.info(f"Test evaluation: {test_results}")
        else:
            test_results = None

        # Return model information
        return {
            "model": final_model,
            "model_type": model_type,
            "embedder": embedder,
            "embed_cols": embed_cols,
            "test_results": test_results,
        }

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise RuntimeError(f"Failed to train model: {str(e)}") from e


def predict(
    model_info: Dict[str, Any],
    sequences: Union[List[str], pd.DataFrame, pd.Series],
    sequence_col: str = "sequence",
    return_confidence: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Generate predictions for new sequences using a trained model.

    :param model_info: Dictionary containing model and related information
    :param sequences: Sequences to predict (list, Series, or DataFrame)
    :param sequence_col: Column name in DataFrame containing sequences
    :param return_confidence: Whether to return confidence estimates
    :return: Array of predictions or tuple of (predictions, confidence)
        if return_confidence=True
    """
    # Extract sequences if a DataFrame is provided
    if isinstance(sequences, pd.DataFrame):
        if sequence_col not in sequences.columns:
            raise ValueError(f"Column '{sequence_col}' not found in provided DataFrame")
        sequences = sequences[sequence_col]

    # Extract model components
    model = model_info["model"]
    model_type = model_info["model_type"]
    embedder = model_info["embedder"]
    embed_cols = model_info["embed_cols"]

    # Embed sequences
    try:
        X_embedded = embedder.transform(sequences)
        X_df = pd.DataFrame(X_embedded, columns=embed_cols)

        # Use the right PyCaret function based on model type
        if model_type == "regression":
            from pycaret.regression import predict_model
        else:
            from pycaret.classification import predict_model

        # Make predictions
        predictions = predict_model(model, data=X_df)

        # Extract prediction column (name varies by PyCaret version)
        pred_cols = [
            col
            for col in predictions.columns
            if any(
                kw in col.lower() for kw in ["prediction", "predict", "label", "class"]
            )
        ]

        if not pred_cols:
            logger.error(
                f"Cannot identify prediction column. Columns: {predictions.columns}"
            )
            raise ValueError("Unable to identify prediction column in output")

        # If confidence is requested, generate a dummy confidence score
        # This is a placeholder - in a real implementation, you'd want to
        # derive this from the model's uncertainty estimates
        if return_confidence:
            # Generate dummy confidence values (0.8-1.0 range)
            confidence = np.random.uniform(0.8, 1.0, len(sequences))
            return predictions[pred_cols[0]].values, confidence

        return predictions[pred_cols[0]].values

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise RuntimeError(f"Failed to generate predictions: {str(e)}") from e


def evaluate_model(
    model: Any,
    X_test: Union[List[str], pd.Series],
    y_test: Union[List[float], pd.Series],
    embedder: Any,
    model_type: str,
    embed_cols: List[str],
) -> Dict[str, float]:
    """Evaluate model performance on test data.

    :param model: Trained model
    :param X_test: Test sequences
    :param y_test: True target values
    :param embedder: Embedder to transform sequences
    :param model_type: Type of model (regression or classification)
    :param embed_cols: Column names for embedded features
    :return: Dictionary of performance metrics
    """
    # Embed test sequences
    X_test_embedded = embedder.transform(X_test)
    X_test_df = pd.DataFrame(X_test_embedded, columns=embed_cols)

    # Make predictions
    if model_type == "regression":
        from pycaret.regression import predict_model
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Get predictions
        preds = predict_model(model, data=X_test_df)
        pred_col = [col for col in preds.columns if "prediction" in col.lower()][0]
        y_pred = preds[pred_col].values

        # Calculate metrics
        metrics = {
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
        }

    else:  # classification
        from pycaret.classification import predict_model
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        # Get predictions
        preds = predict_model(model, data=X_test_df)
        pred_col = [
            col
            for col in preds.columns
            if any(x in col.lower() for x in ["prediction", "class", "label"])
        ][0]
        y_pred = preds[pred_col].values

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }

    return metrics


def save_model(model_info: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save the model to disk.

    :param model_info: Dictionary containing model and related information
    :param path: Path to save the model
    """
    path = Path(path)

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model
    with open(path, "wb") as f:
        pickle.dump(model_info, f)

    logger.info(f"Model saved to {path}")


def load_model(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a trained model from disk.

    :param path: Path to saved model file
    :return: Dictionary containing the model and related information
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        model_info = pickle.load(f)

    return model_info


def _load_data(
    data: Union[pd.DataFrame, Path, str],
    sequence_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Helper function to load data from various sources.

    :param data: DataFrame or path to data file
    :param sequence_col: Column name for sequences
    :param target_col: Column name for target values
    :return: DataFrame with sequence and target columns
    """
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() in [".csv", ".tsv"]:
            df = pd.read_csv(path)
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
