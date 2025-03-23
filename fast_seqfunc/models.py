"""Custom model code for fast-seqfunc."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from lazy_loader import lazy
from loguru import logger

np = lazy.load("numpy")
pd = lazy.load("pandas")

try:
    from pycaret.classification import compare_models as compare_models_classification
    from pycaret.classification import finalize_model as finalize_model_classification
    from pycaret.classification import setup as setup_classification
    from pycaret.regression import compare_models as compare_models_regression
    from pycaret.regression import finalize_model as finalize_model_regression
    from pycaret.regression import setup as setup_regression

    PYCARET_AVAILABLE = True
except ImportError:
    logger.warning("PyCaret not available. Please install it with: pip install pycaret")
    PYCARET_AVAILABLE = False


class SequenceFunctionModel:
    """Model for sequence-function prediction using PyCaret and various embeddings.

    :param embeddings: Dictionary of embeddings by method and split
                      {method: {"train": array, "val": array, "test": array}}
    :param model_type: Type of modeling problem
    :param optimization_metric: Metric to optimize during model selection
    :param embedding_method: Method(s) used for embedding
    """

    def __init__(
        self,
        embeddings: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        model_type: Literal[
            "regression", "classification", "multi-class"
        ] = "regression",
        optimization_metric: Optional[str] = None,
        embedding_method: Union[str, List[str]] = "one-hot",
    ):
        if not PYCARET_AVAILABLE:
            raise ImportError("PyCaret is required for SequenceFunctionModel")

        self.embeddings = embeddings or {}
        self.model_type = model_type
        self.optimization_metric = optimization_metric
        self.embedding_method = embedding_method

        # Properties to be set during fit
        self.best_model = None
        self.embedding_columns = None
        self.training_results = None
        self.is_fitted = False

    def fit(
        self,
        X_train: Union[List[str], pd.Series],
        y_train: Union[List[float], pd.Series],
        X_val: Optional[Union[List[str], pd.Series]] = None,
        y_val: Optional[Union[List[float], pd.Series]] = None,
        **kwargs: Any,
    ) -> "SequenceFunctionModel":
        """Train the model on training data.

        :param X_train: Training sequences
        :param y_train: Training target values
        :param X_val: Validation sequences
        :param y_val: Validation target values
        :param kwargs: Additional arguments for PyCaret setup
        :return: Self for chaining
        """
        if not self.embeddings:
            raise ValueError(
                "No embeddings provided. Did you forget to run embedding first?"
            )

        # Use the first embedding method in the dict as default
        primary_method = (
            self.embedding_method[0]
            if isinstance(self.embedding_method, list)
            else self.embedding_method
        )

        # Create a DataFrame with the embeddings and target
        train_embeddings = self.embeddings[primary_method]["train"]

        # Create column names for the embedding features
        self.embedding_columns = [
            f"embed_{i}" for i in range(train_embeddings.shape[1])
        ]

        # Create DataFrame for PyCaret
        train_df = pd.DataFrame(train_embeddings, columns=self.embedding_columns)
        train_df["target"] = y_train

        # Setup PyCaret environment
        if self.model_type == "regression":
            setup_func = setup_regression
            compare_func = compare_models_regression
            finalize_func = finalize_model_regression
        elif self.model_type in ["classification", "multi-class"]:
            setup_func = setup_classification
            compare_func = compare_models_classification
            finalize_func = finalize_model_classification
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Configure validation approach
        fold_strategy = None
        fold = 5  # default

        if X_val is not None and y_val is not None:
            # If validation data is provided, use it for validation
            val_embeddings = self.embeddings[primary_method]["val"]
            val_df = pd.DataFrame(val_embeddings, columns=self.embedding_columns)
            val_df["target"] = y_val

            # Custom data split with provided validation set
            from sklearn.model_selection import PredefinedSplit

            # Create one combined DataFrame
            combined_df = pd.concat([train_df, val_df], ignore_index=True)

            # Define test_fold where -1 indicates train and 0 indicates test
            test_fold = [-1] * len(train_df) + [0] * len(val_df)
            fold_strategy = PredefinedSplit(test_fold)
            fold = 1  # With predefined split, only need one fold

            # Use combined data
            train_df = combined_df

        # Setup the PyCaret environment
        setup_args = {
            "data": train_df,
            "target": "target",
            "fold": fold,
            "fold_strategy": fold_strategy,
            "silent": True,
            "verbose": False,
            **kwargs,
        }

        if self.optimization_metric:
            setup_args["optimize"] = self.optimization_metric

        logger.info(f"Setting up PyCaret for {self.model_type} modeling...")
        setup_func(**setup_args)

        # Compare models to find the best one
        logger.info("Comparing models to find best performer...")
        self.best_model = compare_func(n_select=1)

        # Finalize the model using all data
        logger.info("Finalizing model...")
        self.best_model = finalize_func(self.best_model)

        self.is_fitted = True
        return self

    def predict(
        self,
        sequences: Union[List[str], pd.Series],
    ) -> np.ndarray:
        """Generate predictions for new sequences.

        :param sequences: Sequences to predict
        :return: Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Please call fit() first.")

        # Get embeddings for the sequences
        # This would normally be done by the embedder
        # but since we don't have access here,
        # we'll just assume the sequences are already embedded in the correct format

        # For now, return a placeholder
        return np.zeros(len(sequences))

    def predict_with_confidence(
        self,
        sequences: Union[List[str], pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with confidence estimates.

        :param sequences: Sequences to predict
        :return: Tuple of (predictions, confidence)
        """
        # For now, return placeholders
        predictions = self.predict(sequences)
        confidence = np.ones_like(predictions) * 0.95  # Placeholder confidence

        return predictions, confidence

    def evaluate(
        self,
        X_test: Union[List[str], pd.Series],
        y_test: Union[List[float], pd.Series],
    ) -> Dict[str, float]:
        """Evaluate model performance on test data.

        :param X_test: Test sequences
        :param y_test: True target values
        :return: Dictionary of performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Please call fit() first.")

        # Get predictions
        y_pred = self.predict(X_test)

        # Calculate metrics based on model type
        if self.model_type == "regression":
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            metrics = {
                "r2": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
            }
        else:  # classification
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
            }

        return metrics

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.

        :param path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        path = Path(path)

        # Create directory if it doesn't exist
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SequenceFunctionModel":
        """Load a model from disk.

        :param path: Path to saved model
        :return: Loaded model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")

        return model
