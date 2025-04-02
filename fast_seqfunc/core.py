"""Core functionality for fast-seqfunc.

This module implements the main API functions for training sequence-function models,
and making predictions with a simpler design using PyCaret directly.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fast_seqfunc.embedders import get_embedder

# Global session counter for PyCaret
_session_id = 42


def train_model(
    train_data: Union[pd.DataFrame, Path, str],
    val_data: Optional[Union[pd.DataFrame, Path, str]] = None,
    test_data: Optional[Union[pd.DataFrame, Path, str]] = None,
    sequence_col: str = "sequence",
    target_col: str = "function",
    additional_predictor_cols: Optional[List[str]] = None,
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
    :param additional_predictor_cols: Optional list of column names
        for additional predictors
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

    # Validate additional predictor columns
    if additional_predictor_cols:
        _validate_additional_predictors(train_df, additional_predictor_cols)
        if test_df is not None:
            _validate_additional_predictors(test_df, additional_predictor_cols)
        logger.info(f"Using additional predictor columns: {additional_predictor_cols}")

    # Get embedder for sequences
    logger.info(f"Generating {embedding_method} embeddings...")
    embedder = get_embedder(embedding_method)

    # Create column names for embeddings
    X_train_embedded = embedder.fit_transform(train_df[sequence_col])
    embed_cols = [f"embed_{i}" for i in range(X_train_embedded.shape[1])]

    # Create DataFrame with embeddings
    train_processed = pd.DataFrame(X_train_embedded, columns=embed_cols)

    # Prepare additional predictors preprocessing pipeline
    additional_predictor_preprocessing = None
    if additional_predictor_cols:
        # Process additional predictors
        logger.info("Processing additional predictor columns...")
        additional_predictor_preprocessing = _create_predictor_preprocessing_pipeline(
            train_df[additional_predictor_cols]
        )

        # Transform additional predictors
        additional_predictors_transformed = (
            additional_predictor_preprocessing.fit_transform(
                train_df[additional_predictor_cols]
            )
        )

        # Convert to DataFrame if it's a sparse matrix
        if hasattr(additional_predictors_transformed, "toarray"):
            additional_predictors_transformed = (
                additional_predictors_transformed.toarray()
            )

        # Add additional predictors to the training data
        additional_cols = [
            f"additional_{i}" for i in range(additional_predictors_transformed.shape[1])
        ]
        additional_predictors_df = pd.DataFrame(
            additional_predictors_transformed, columns=additional_cols
        )

        # Combine with sequence embeddings
        train_processed = pd.concat([train_processed, additional_predictors_df], axis=1)

        logger.info(
            f"Combined {len(embed_cols)} sequence embedding features with "
            f"{len(additional_cols)} additional predictor features"
        )

    # Add target column to processed data
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

            # Check if the dataset is extremely small (3 samples or fewer)
            # In this case, bypass compare_models entirely and use create_model
            if len(train_processed) <= 3:
                logger.warning(
                    f"Extremely small training dataset detected ({len(train_processed)} samples). "  # noqa: E501
                    "Bypassing model comparison and directly creating a simple model."
                )
                from pycaret.regression import create_model

                # Use a simple ridge regression which can work with tiny datasets
                models = create_model("ridge", cross_validation=False)
            # Check if the dataset is very small (less than 10 samples)
            # PyCaret struggles with cross-validation on tiny datasets
            elif len(train_processed) < 10:
                logger.warning(
                    f"Very small training dataset detected ({len(train_processed)} samples). "  # noqa: E501
                    "Disabling cross-validation and using simpler models."
                )
                compare_args["cross_validation"] = False
                compare_args["include"] = ["lr", "ridge", "lasso", "dt", "knn"]
                # Try to train models
                models = compare_models(**compare_args)
            else:
                # For normal-sized datasets, use the standard approach
                models = compare_models(**compare_args)

            # Finalize model (train on all data)
            logger.info("Finalizing best model...")

            # Check if models is empty and provide a fallback model
            if not models:
                logger.warning(
                    "No models returned by compare_models, using fallback model"
                )
                models = RandomForestRegressor(random_state=_session_id)

                # Fit the fallback model directly
                models.fit(
                    train_processed.drop("target", axis=1), train_processed["target"]
                )

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

            # Check if the dataset is extremely small (3 samples or fewer)
            # In this case, bypass compare_models entirely and use create_model
            if len(train_processed) <= 3:
                logger.warning(
                    f"Extremely small training dataset detected ({len(train_processed)} samples). "  # noqa: E501
                    "Bypassing model comparison and directly creating a simple model."
                )
                from pycaret.classification import create_model

                # Use a simple logistic regression which can work with tiny datasets
                models = create_model("lr", cross_validation=False)
            # Check if the dataset is very small (less than 10 samples)
            # PyCaret struggles with cross-validation on tiny datasets
            elif len(train_processed) < 10:
                logger.warning(
                    f"Very small training dataset detected ({len(train_processed)} samples). "  # noqa: E501
                    "Disabling cross-validation and using simpler models."
                )
                compare_args["cross_validation"] = False
                compare_args["include"] = ["lr", "lda", "qda", "dt", "knn"]
                # Try to train models
                models = compare_models(**compare_args)
            else:
                # For normal-sized datasets, use the standard approach
                models = compare_models(**compare_args)

            # Finalize model (train on all data)
            logger.info("Finalizing best model...")

            # Check if models is empty and provide a fallback model
            if not models:
                logger.warning(
                    "No models returned by compare_models, using fallback model"
                )
                models = RandomForestClassifier(random_state=_session_id)

                # Fit the fallback model directly
                models.fit(
                    train_processed.drop("target", axis=1), train_processed["target"]
                )

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
                additional_predictor_cols=additional_predictor_cols,
                additional_predictor_preprocessing=additional_predictor_preprocessing,
                data=test_df,
            )
            logger.info(f"Test evaluation: {test_results}")
        else:
            test_results = None

        # Return model information
        model_info = {
            "model": final_model,
            "model_type": model_type,
            "embedder": embedder,
            "embed_cols": embed_cols,
            "test_results": test_results,
        }

        # Add additional predictor information if used
        if additional_predictor_cols:
            model_info["additional_predictor_cols"] = additional_predictor_cols
            model_info["additional_predictor_preprocessing"] = (
                additional_predictor_preprocessing
            )

        return model_info

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise RuntimeError(f"Failed to train model: {str(e)}") from e


def predict(
    model_info: Dict[str, Any],
    sequences: Union[List[str], pd.DataFrame, pd.Series],
    sequence_col: str = "sequence",
) -> np.ndarray:
    """Generate predictions for new sequences using a trained model.

    :param model_info: Dictionary containing model and related information
    :param sequences: Sequences to predict (list, Series, or DataFrame)
    :param sequence_col: Column name in DataFrame containing sequences
    :return: Array of predictions
    """
    # Check if additional predictors were used in training
    has_additional_predictors = (
        "additional_predictor_cols" in model_info
        and model_info["additional_predictor_cols"] is not None
    )

    # Extract sequences and additional predictors if a DataFrame is provided
    if isinstance(sequences, pd.DataFrame):
        # Validate that the sequence column exists
        if sequence_col not in sequences.columns:
            raise ValueError(f"Column '{sequence_col}' not found in provided DataFrame")

        # Check if additional predictors are needed
        if has_additional_predictors:
            # Validate that all required additional predictor columns exist
            _validate_additional_predictors(
                sequences, model_info["additional_predictor_cols"]
            )
            # Keep the DataFrame for additional predictors access
            sequences_df = sequences
            sequences = sequences_df[sequence_col]
        else:
            # If no additional predictors, just extract the sequence column
            sequences = sequences[sequence_col]
    elif has_additional_predictors:
        # If we have a list or Series
        # but the model requires additional predictors, raise an error
        raise ValueError(
            "When using additional predictors, "
            "sequences must be provided as a DataFrame "
            "containing both sequence and additional predictor columns."
        )

    # Extract model components
    model = model_info["model"]
    model_type = model_info["model_type"]
    embedder = model_info["embedder"]
    embed_cols = model_info["embed_cols"]

    # Embed sequences
    X_embedded = embedder.transform(sequences)
    X_df = pd.DataFrame(X_embedded, columns=embed_cols)

    # Add additional predictors if needed
    if has_additional_predictors:
        additional_predictor_cols = model_info["additional_predictor_cols"]
        additional_predictor_preprocessing = model_info[
            "additional_predictor_preprocessing"
        ]

        # Transform additional predictors
        additional_predictors_transformed = (
            additional_predictor_preprocessing.transform(
                sequences_df[additional_predictor_cols]
            )
        )

        # Convert to DataFrame if it's a sparse matrix
        if hasattr(additional_predictors_transformed, "toarray"):
            additional_predictors_transformed = (
                additional_predictors_transformed.toarray()
            )

        # Add additional predictors to the features
        additional_cols = [
            f"additional_{i}" for i in range(additional_predictors_transformed.shape[1])
        ]
        additional_predictors_df = pd.DataFrame(
            additional_predictors_transformed, columns=additional_cols
        )

        # Combine with sequence embeddings
        X_df = pd.concat([X_df, additional_predictors_df], axis=1)

    # Make predictions
    if model_type == "regression":
        predictions = model.predict(X_df)
    elif model_type == "classification":
        predictions = model.predict_proba(X_df)[:, 1]  # Probability of positive class
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return predictions


def evaluate_model(
    model: Any,
    X_test: Union[List[str], pd.Series],
    y_test: Union[List[float], pd.Series],
    embedder: Any,
    model_type: str,
    embed_cols: List[str],
    additional_predictor_cols: Optional[List[str]] = None,
    additional_predictor_preprocessing: Optional[Any] = None,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Evaluate model performance on test data.

    :param model: Trained model
    :param X_test: Test sequences
    :param y_test: True target values
    :param embedder: Embedder used for sequences
    :param model_type: Type of model (regression or classification)
    :param embed_cols: Column names for embedded features
    :param additional_predictor_cols: Optional list of column names
        for additional predictors
    :param additional_predictor_preprocessing: Optional preprocessing pipeline
        for additional predictors
    :param data: Optional DataFrame containing additional predictor columns
    :return: Dictionary containing metrics and prediction data with structure:
             {
                 "metric1": value1,
                 "metric2": value2,
                 ...
             }
    """
    # Embed sequences
    X_test_embedded = embedder.transform(X_test)
    X_test_df = pd.DataFrame(X_test_embedded, columns=embed_cols)

    # Process additional predictors if provided
    if (
        additional_predictor_cols
        and additional_predictor_preprocessing
        and data is not None
    ):
        # Transform additional predictors
        additional_predictors_transformed = (
            additional_predictor_preprocessing.transform(
                data[additional_predictor_cols]
            )
        )

        # Convert to DataFrame if it's a sparse matrix
        if hasattr(additional_predictors_transformed, "toarray"):
            additional_predictors_transformed = (
                additional_predictors_transformed.toarray()
            )

        # Add additional predictors to the features
        additional_cols = [
            f"additional_{i}" for i in range(additional_predictors_transformed.shape[1])
        ]
        additional_predictors_df = pd.DataFrame(
            additional_predictors_transformed, columns=additional_cols
        )

        # Combine with sequence embeddings
        X_test_df = pd.concat([X_test_df, additional_predictors_df], axis=1)

    # Make predictions
    if model_type == "regression":
        y_pred = model.predict(X_test_df)
        metrics = _evaluate_regression(y_test, y_pred)
    elif model_type == "classification":
        y_pred_proba = model.predict_proba(X_test_df)[:, 1]
        y_pred_class = model.predict(X_test_df)
        metrics = _evaluate_classification(y_test, y_pred_proba, y_pred_class)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return metrics


def _validate_additional_predictors(
    data: pd.DataFrame, predictor_cols: List[str]
) -> None:
    """Validate that all required additional predictor columns exist in the data.

    :param data: DataFrame to validate
    :param predictor_cols: List of required predictor column names
    :raises ValueError: If a required predictor column is missing
    """
    missing_cols = [col for col in predictor_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required predictor column(s): {', '.join(missing_cols)}. "
            f"Available columns: {', '.join(data.columns)}"
        )


def _create_predictor_preprocessing_pipeline(data: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline for additional predictor columns.

    This function creates a preprocessing pipeline that handles:
    - Numerical features: StandardScaler
    - Categorical features: OneHotEncoder

    :param data: DataFrame containing predictor columns
    :return: Preprocessing pipeline for additional predictors
    """
    # Identify numeric and categorical columns
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(sparse_output=True, handle_unknown="ignore"),
                categorical_cols,
            ),
        ],
        remainder="passthrough",
    )

    logger.info(
        f"Created preprocessing pipeline with {len(numeric_cols)} numeric columns and "
        f"{len(categorical_cols)} categorical columns"
    )

    return preprocessor


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


def save_detailed_metrics(
    metrics_data: Dict[str, Any],
    output_dir: Path,
    model_type: str,
    embedding_method: str = "unknown",
) -> None:
    """Save detailed model metrics to files in the specified directory.

    :param metrics_data: Dictionary containing metrics and prediction data
    :param output_dir: Directory to save metrics files
    :param model_type: Type of model (regression or classification)
    :param embedding_method: Embedding method used for this model
    """
    import json

    from matplotlib import pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    metrics = metrics_data.get("metrics", {})
    predictions_data = metrics_data.get("predictions_data", {})
    y_true = predictions_data.get("y_true", [])
    y_pred = predictions_data.get("y_pred", [])

    # Save metrics to JSON file
    metrics_file = output_dir / f"{embedding_method}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save raw predictions to CSV
    predictions_df = pd.DataFrame(
        {
            "true_value": y_true,
            "prediction": y_pred,
        }
    )
    predictions_file = output_dir / f"{embedding_method}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)

    # Create visualizations based on model type
    if model_type == "regression":
        # Scatter plot of predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(f"True vs Predicted Values - {embedding_method}")
        plt.savefig(output_dir / f"{embedding_method}_scatter_plot.png")
        plt.close()

        # Residual plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot - {embedding_method}")
        plt.savefig(output_dir / f"{embedding_method}_residual_plot.png")
        plt.close()

    else:  # classification
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix - {embedding_method}")
            plt.savefig(output_dir / f"{embedding_method}_confusion_matrix.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create confusion matrix: {e}")

    logger.info(f"Detailed metrics saved to {output_dir}")


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics.

    :param y_true: True target values
    :param y_pred: Predicted values
    :return: Dictionary of regression metrics
    """
    from sklearn.metrics import (
        explained_variance_score,
        max_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    # Calculate metrics
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
    }

    # Try to compute MAPE, but handle potential division by zero
    try:
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        metrics["mape"] = np.nan

    return metrics


def _evaluate_classification(
    y_true: np.ndarray, y_pred_proba: np.ndarray, y_pred_class: np.ndarray
) -> Dict[str, float]:
    """Calculate classification metrics.

    :param y_true: True target values
    :param y_pred_proba: Predicted probabilities
    :param y_pred_class: Predicted class labels
    :return: Dictionary of classification metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_class),
        "f1": f1_score(y_true, y_pred_class, average="weighted"),
        "precision": precision_score(y_true, y_pred_class, average="weighted"),
        "recall": recall_score(y_true, y_pred_class, average="weighted"),
    }

    # Try to calculate ROC AUC if binary classification
    try:
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        metrics["roc_auc"] = np.nan

    return metrics
