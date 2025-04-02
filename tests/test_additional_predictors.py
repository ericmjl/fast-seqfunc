"""Tests for additional predictor functionality in fast-seqfunc."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fast_seqfunc.core import (
    evaluate_model,
    load_model,
    predict,
    save_model,
    train_model,
)
from fast_seqfunc.synthetic import generate_dataset_with_predictors

# Set random seed for reproducibility
np.random.seed(42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    # Clean up after tests
    shutil.rmtree(tmp_dir)


@pytest.fixture
def sequence_data_with_predictors():
    """Create test data with sequences and additional predictors."""
    # Create sample data with sequences, additional predictors, and function values
    data = pd.DataFrame(
        {
            "sequence": ["ACDEFG", "GHIKLM", "NOPQRS", "TUVWXY"],
            "predictor1": [0.1, 0.2, 0.3, 0.4],
            "predictor2": [10, 20, 30, 40],
            "categorical_pred": ["A", "B", "A", "B"],
            "function": [0.5, 0.6, 0.7, 0.8],
        }
    )
    return data


@pytest.fixture
def larger_sequence_data_with_predictors():
    """Create larger test dataset with synthetic data generator."""
    # Use the synthetic data generator to create a dataset with 100 samples
    data = generate_dataset_with_predictors(
        task="nonlinear_composition",  # Nonlinear regression task
        count=100,  # 100 samples
        noise_level=0.2,  # Add some noise
        num_numeric_predictors=2,  # Add 2 numeric predictors
        num_categorical_predictors=1,  # Add 1 categorical predictor
        correlation_strength=0.7,  # Strong correlation with target
    )
    return data


def test_train_with_additional_predictors(sequence_data_with_predictors):
    """Test training a model with additional predictors."""
    # Split data into train and test
    train_data = sequence_data_with_predictors.iloc[:3]
    test_data = sequence_data_with_predictors.iloc[3:]

    # Train model with additional predictors
    model_info = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        additional_predictor_cols=["predictor1", "predictor2"],
        embedding_method="one-hot",
        model_type="regression",
    )

    # Check that model_info contains additional predictor information
    assert "additional_predictor_cols" in model_info
    assert model_info["additional_predictor_cols"] == ["predictor1", "predictor2"]
    assert "additional_predictor_preprocessing" in model_info

    # Test prediction
    predictions = predict(model_info, test_data)
    assert len(predictions) == len(test_data)
    assert isinstance(predictions, np.ndarray)


def test_train_with_categorical_predictors(sequence_data_with_predictors):
    """Test training a model with categorical predictors that require encoding."""
    # Split data into train and test
    train_data = sequence_data_with_predictors.iloc[:3]
    test_data = sequence_data_with_predictors.iloc[3:]

    # Train model with categorical predictor
    model_info = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        additional_predictor_cols=["categorical_pred"],
        embedding_method="one-hot",
        model_type="regression",
    )

    # Check that model_info contains additional predictor information
    assert "additional_predictor_cols" in model_info
    assert model_info["additional_predictor_cols"] == ["categorical_pred"]

    # Test prediction
    predictions = predict(model_info, test_data)
    assert len(predictions) == len(test_data)
    assert isinstance(predictions, np.ndarray)


def test_train_with_multiple_predictor_types(sequence_data_with_predictors):
    """Test training a model with multiple types of predictors."""
    # Split data into train and test
    train_data = sequence_data_with_predictors.iloc[:3]
    test_data = sequence_data_with_predictors.iloc[3:]

    # Train model with multiple predictor types
    model_info = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        additional_predictor_cols=["predictor1", "predictor2", "categorical_pred"],
        embedding_method="one-hot",
        model_type="regression",
    )

    # Check that model_info contains additional predictor information
    assert "additional_predictor_cols" in model_info
    assert set(model_info["additional_predictor_cols"]) == {
        "predictor1",
        "predictor2",
        "categorical_pred",
    }

    # Test prediction
    predictions = predict(model_info, test_data)
    assert len(predictions) == len(test_data)
    assert isinstance(predictions, np.ndarray)


def test_serialization_with_additional_predictors(
    sequence_data_with_predictors, temp_dir
):
    """Test serialization and deserialization of model with additional predictors."""
    # Split data into train and test
    train_data = sequence_data_with_predictors.iloc[:3]
    test_data = sequence_data_with_predictors.iloc[3:]

    # Train model with additional predictors
    model_info = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        additional_predictor_cols=["predictor1", "predictor2", "categorical_pred"],
        embedding_method="one-hot",
        model_type="regression",
    )

    # Save model
    model_path = Path(temp_dir) / "model_with_predictors.pkl"
    save_model(model_info, model_path)

    # Load model
    loaded_model = load_model(model_path)

    # Check that loaded model contains additional predictor information
    assert "additional_predictor_cols" in loaded_model
    assert set(loaded_model["additional_predictor_cols"]) == {
        "predictor1",
        "predictor2",
        "categorical_pred",
    }
    assert "additional_predictor_preprocessing" in loaded_model

    # Test prediction with loaded model
    predictions = predict(loaded_model, test_data)
    assert len(predictions) == len(test_data)
    assert isinstance(predictions, np.ndarray)


def test_backwards_compatibility(sequence_data_with_predictors):
    """Test that the implementation maintains backward compatibility."""
    # Train model without additional predictors (old behavior)
    train_data = sequence_data_with_predictors.iloc[:3]
    test_data = sequence_data_with_predictors.iloc[3:]

    # Train model without additional predictors
    model_info = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        embedding_method="one-hot",
        model_type="regression",
    )

    # Check that the old model still works
    assert (
        "additional_predictor_cols" not in model_info
        or model_info["additional_predictor_cols"] is None
    )

    # Test prediction with sequences only
    sequences_only = pd.Series(test_data["sequence"].values)
    predictions = predict(model_info, sequences_only)
    assert len(predictions) == len(test_data)
    assert isinstance(predictions, np.ndarray)


def test_missing_predictors_error():
    """Test that an appropriate error is raised when a predictor is missing."""
    # Create train data with predictors
    train_data = pd.DataFrame(
        {
            "sequence": ["ACDEFG", "GHIKLM", "NOPQRS"],
            "predictor1": [0.1, 0.2, 0.3],
            "function": [0.5, 0.6, 0.7],
        }
    )

    # Create test data missing a predictor
    test_data = pd.DataFrame({"sequence": ["TUVWXY"], "function": [0.8]})

    # Train model with predictor
    model_info = train_model(
        train_data=train_data,
        test_data=None,
        sequence_col="sequence",
        target_col="function",
        additional_predictor_cols=["predictor1"],
        embedding_method="one-hot",
        model_type="regression",
    )

    # Prediction should raise a ValueError for missing predictor
    with pytest.raises(ValueError, match="Missing required predictor column"):
        predict(model_info, test_data)


def test_evaluate_model_with_predictors(sequence_data_with_predictors):
    """Test evaluate_model function with additional predictors."""
    # Split data into train and test
    train_data = sequence_data_with_predictors.iloc[:3]
    test_data = sequence_data_with_predictors.iloc[3:]

    # Train model with additional predictors
    model_info = train_model(
        train_data=train_data,
        test_data=None,
        sequence_col="sequence",
        target_col="function",
        additional_predictor_cols=["predictor1", "predictor2"],
        embedding_method="one-hot",
        model_type="regression",
    )

    # Evaluate model
    results = evaluate_model(
        model_info["model"],
        test_data["sequence"],
        test_data["function"],
        embedder=model_info["embedder"],
        model_type="regression",
        embed_cols=model_info["embed_cols"],
        additional_predictor_cols=model_info["additional_predictor_cols"],
        additional_predictor_preprocessing=model_info[
            "additional_predictor_preprocessing"
        ],
        data=test_data,
    )

    # Check that evaluation results exist
    assert "r2" in results
    assert "rmse" in results


def test_prediction_with_sequence_list_and_predictors(sequence_data_with_predictors):
    """Test prediction with a list of sequences and a DataFrame of predictors."""
    # Split data into train and test
    train_data = sequence_data_with_predictors.iloc[:3]
    test_sequences = sequence_data_with_predictors.iloc[3:]["sequence"].tolist()
    test_data = sequence_data_with_predictors.iloc[3:]

    # Train model with additional predictors
    model_info = train_model(
        train_data=train_data,
        test_data=None,
        sequence_col="sequence",
        target_col="function",
        additional_predictor_cols=["predictor1", "predictor2"],
        embedding_method="one-hot",
        model_type="regression",
    )

    # Test predicting with a list of sequences and separate predictors
    with pytest.raises(ValueError, match="When using additional predictors"):
        predict(model_info, test_sequences)

    # Should work when passing a DataFrame with all required columns
    predictions = predict(model_info, test_data)
    assert len(predictions) == len(test_data)


def test_train_with_larger_dataset(larger_sequence_data_with_predictors):
    """Test training a model with a larger dataset containing additional predictors."""
    # Use 80% of data for training, 20% for testing
    train_size = int(len(larger_sequence_data_with_predictors) * 0.8)
    train_data = larger_sequence_data_with_predictors.iloc[:train_size]
    test_data = larger_sequence_data_with_predictors.iloc[train_size:]

    # Get the column names from the dataset
    numeric_cols = [col for col in train_data.columns if col.startswith("predictor")]
    categorical_cols = [
        col for col in train_data.columns if col.startswith("categorical_pred")
    ]
    predictor_cols = numeric_cols + categorical_cols

    # Train model with additional predictors
    model_info = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        additional_predictor_cols=predictor_cols,
        embedding_method="one-hot",
        model_type="regression",
    )

    # Check that model_info contains additional predictor information
    assert "additional_predictor_cols" in model_info
    assert set(model_info["additional_predictor_cols"]) == set(predictor_cols)
    assert "additional_predictor_preprocessing" in model_info

    # Test prediction
    predictions = predict(model_info, test_data)
    assert len(predictions) == len(test_data)
    assert isinstance(predictions, np.ndarray)

    # Evaluate the model
    eval_results = evaluate_model(
        model=model_info["model"],
        X_test=test_data["sequence"],
        y_test=test_data["function"],
        embedder=model_info["embedder"],
        model_type="regression",
        embed_cols=model_info["embed_cols"],
        additional_predictor_cols=model_info["additional_predictor_cols"],
        additional_predictor_preprocessing=model_info[
            "additional_predictor_preprocessing"
        ],
        data=test_data,
    )

    # Check that evaluation results are reasonable
    assert "r2" in eval_results
    assert "rmse" in eval_results
    assert eval_results["r2"] > 0.0  # Should have at least some predictive power
