"""Fast unit tests for additional predictor helper functions in fast-seqfunc."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from fast_seqfunc.core import (
    _create_predictor_preprocessing_pipeline,
    _load_data,
    _validate_additional_predictors,
    load_model,
    predict,
    save_model,
)
from fast_seqfunc.embedders import OneHotEmbedder


def test_validate_additional_predictors_success():
    """Test that _validate_additional_predictors passes with valid inputs."""
    data = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [0.1, 0.2, 0.3]}
    )
    # Should not raise an exception
    _validate_additional_predictors(data, ["col1", "col2"])


def test_validate_additional_predictors_failure():
    """Test that _validate_additional_predictors raises error with missing columns."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    # Should raise a ValueError
    with pytest.raises(ValueError, match="Missing required predictor column"):
        _validate_additional_predictors(data, ["col1", "col3"])


def test_validate_additional_predictors_empty():
    """Test that _validate_additional_predictors handles empty predictor list."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    # Should not raise an exception
    _validate_additional_predictors(data, [])


def test_create_predictor_preprocessing_numeric_only():
    """Test preprocessing pipeline with only numeric columns."""
    data = pd.DataFrame({"num1": [1, 2, 3], "num2": [0.1, 0.2, 0.3]})
    pipeline = _create_predictor_preprocessing_pipeline(data)
    # Check that the pipeline has the correct components
    transformers = pipeline.transformers
    assert any(name == "num" for name, _, _ in transformers)
    # Verify the numeric columns are assigned to the correct transformer
    for name, _, cols in transformers:
        if name == "num":
            assert set(cols) == {"num1", "num2"}

    # Verify it transforms correctly
    result = pipeline.fit_transform(data)
    assert result.shape == (3, 2)  # Should maintain same dimensions for numeric only


def test_create_predictor_preprocessing_categorical_only():
    """Test preprocessing pipeline with only categorical columns."""
    data = pd.DataFrame({"cat1": ["a", "b", "c"], "cat2": ["x", "y", "z"]})
    pipeline = _create_predictor_preprocessing_pipeline(data)
    # Check pipeline structure
    transformers = pipeline.transformers
    assert any(name == "cat" for name, _, _ in transformers)
    # Verify the categorical columns are assigned to the correct transformer
    for name, _, cols in transformers:
        if name == "cat":
            assert set(cols) == {"cat1", "cat2"}

    # Verify it transforms correctly
    result = pipeline.fit_transform(data)
    assert result.shape[0] == 3  # Should still have 3 rows
    assert (
        result.shape[1] > 2
    )  # After one-hot encoding, should have more columns than original


def test_create_predictor_preprocessing_mixed():
    """Test preprocessing pipeline with mixed column types."""
    data = pd.DataFrame({"num1": [1, 2, 3], "cat1": ["a", "b", "c"]})
    pipeline = _create_predictor_preprocessing_pipeline(data)
    # Verify basic pipeline structure
    transformers = pipeline.transformers
    assert any(name == "num" for name, _, _ in transformers)
    assert any(name == "cat" for name, _, _ in transformers)

    # Verify columns are assigned correctly
    for name, _, cols in transformers:
        if name == "num":
            assert set(cols) == {"num1"}
        elif name == "cat":
            assert set(cols) == {"cat1"}

    # Transform and verify result shape
    result = pipeline.fit_transform(data)
    assert result.shape[0] == 3  # Should maintain original row count
    # Should have at least 3 columns after one-hot encoding
    # (1 numeric + at least 2 one-hot)
    assert result.shape[1] >= 3


def test_preprocessor_transform_direct():
    """Directly test the preprocessing pipeline transformation."""
    # Create data with mixed types
    data = pd.DataFrame({"num1": [1.0, 2.0], "cat1": ["a", "b"]})

    # Create preprocessing pipeline directly
    preprocessor = _create_predictor_preprocessing_pipeline(data)

    # Fit and transform
    preprocessor.fit(data)
    result = preprocessor.transform(data)

    # Assertions
    assert result is not None
    assert hasattr(result, "shape")

    # Test the pipeline with new data
    new_data = pd.DataFrame({"num1": [3.0], "cat1": ["c"]})
    new_result = preprocessor.transform(new_data)
    assert new_result is not None
    assert new_result.shape[0] == 1  # One row
    assert (
        new_result.shape[1] == result.shape[1]
    )  # Same number of columns after transformation


def test_preprocessor_handles_new_categorical_values():
    """Test that the preprocessor correctly handles new categorical values."""
    # Create data with categorical column
    data = pd.DataFrame({"num1": [1.0, 2.0], "cat1": ["a", "b"]})

    # Create and fit preprocessing pipeline
    preprocessor = _create_predictor_preprocessing_pipeline(data)
    preprocessor.fit(data)

    # Create new data with unseen categorical value
    new_data = pd.DataFrame(
        {
            "num1": [3.0],
            "cat1": ["unseen_category"],  # New category not in training data
        }
    )

    # Should not raise an error due to handle_unknown='ignore' in OneHotEncoder
    result = preprocessor.transform(new_data)
    assert result is not None
    assert result.shape[0] == 1  # Should still return one row


def test_load_data_from_dataframe():
    """Test loading data from a DataFrame."""
    # Create a test DataFrame
    df = pd.DataFrame(
        {
            "sequence": ["AAA", "CCC", "GGG"],
            "function": [0.1, 0.2, 0.3],
            "other_col": [1, 2, 3],
        }
    )

    # Load data using _load_data
    result = _load_data(df, "sequence", "function")

    # Check that the result is the same DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "sequence" in result.columns
    assert "function" in result.columns
    assert "other_col" in result.columns


def test_load_data_missing_columns():
    """Test _load_data when required columns are missing."""
    # Create a test DataFrame missing the target column
    df = pd.DataFrame({"sequence": ["AAA", "CCC", "GGG"], "other_col": [1, 2, 3]})

    # Should raise ValueError due to missing 'function' column
    with pytest.raises(ValueError, match="Target column 'function' not found in data"):
        _load_data(df, "sequence", "function")

    # Create a test DataFrame missing the sequence column
    df = pd.DataFrame({"other_col": [1, 2, 3], "function": [0.1, 0.2, 0.3]})

    # Should raise ValueError due to missing 'sequence' column
    with pytest.raises(
        ValueError, match="Sequence column 'sequence' not found in data"
    ):
        _load_data(df, "sequence", "function")


def test_save_load_model_with_additional_predictors():
    """Test saving and loading a model with additional predictor information."""
    # Create a minimal model info dictionary with additional predictors
    embedder = OneHotEmbedder()
    embedder.fit(["A", "C", "G"])

    # Create a simple preprocessing pipeline
    data = pd.DataFrame({"num1": [1, 2, 3], "cat1": ["a", "b", "c"]})
    preprocessor = _create_predictor_preprocessing_pipeline(data)

    # Create a simple model info dictionary
    model_info = {
        "model": LinearRegression(),
        "model_type": "regression",
        "embedder": embedder,
        "embed_cols": ["embed_0", "embed_1", "embed_2"],
        "additional_predictor_cols": ["num1", "cat1"],
        "additional_predictor_preprocessing": preprocessor,
        "test_results": {"r2": 0.9, "rmse": 0.1},
    }

    # Create a temporary file for the model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        model_path = tmp.name

    try:
        # Save the model
        save_model(model_info, model_path)

        # Verify the file exists
        assert Path(model_path).exists()

        # Load the model
        loaded_model_info = load_model(model_path)

        # Check that all keys are preserved
        assert set(loaded_model_info.keys()) == set(model_info.keys())
        assert loaded_model_info["model_type"] == "regression"
        assert loaded_model_info["additional_predictor_cols"] == ["num1", "cat1"]
        assert isinstance(
            loaded_model_info["additional_predictor_preprocessing"], ColumnTransformer
        )

    finally:
        # Clean up the temporary file
        Path(model_path).unlink(missing_ok=True)


def test_predict_additional_predictors_validation():
    """Test that predict properly validates additional predictors."""
    # Create a minimal model with additional predictors
    embedder = OneHotEmbedder()
    embedder.fit(["A", "C", "G"])

    # Create a preprocessing pipeline
    data = pd.DataFrame({"num_pred": [1, 2, 3], "cat_pred": ["a", "b", "c"]})
    preprocessor = _create_predictor_preprocessing_pipeline(data)

    # Create model info with additional predictors
    model_info = {
        "model": LinearRegression(),
        "model_type": "regression",
        "embedder": embedder,
        "embed_cols": ["embed_0", "embed_1", "embed_2"],
        "additional_predictor_cols": ["num_pred", "cat_pred"],
        "additional_predictor_preprocessing": preprocessor,
    }

    # Test with DataFrame missing required columns
    test_data = pd.DataFrame(
        {
            "sequence": ["AAA", "CCC"],
            "num_pred": [4, 5],  # Missing cat_pred
        }
    )

    # Should raise ValueError due to missing 'cat_pred'
    with pytest.raises(ValueError, match="Missing required predictor column"):
        predict(model_info, test_data)

    # Test with non-DataFrame input when additional predictors are needed
    with pytest.raises(ValueError, match="When using additional predictors"):
        predict(model_info, ["AAA", "CCC"])


def test_predict_sequence_column_validation():
    """Test that predict properly validates the sequence column."""
    # Create a minimal model without additional predictors
    embedder = OneHotEmbedder()
    embedder.fit(["A", "C", "G"])

    model_info = {
        "model": LinearRegression(),
        "model_type": "regression",
        "embedder": embedder,
        "embed_cols": ["embed_0", "embed_1", "embed_2"],
    }

    # Test with DataFrame missing sequence column
    test_data = pd.DataFrame(
        {
            "wrong_col": ["AAA", "CCC"],
        }
    )

    # Should raise ValueError due to missing sequence column
    with pytest.raises(
        ValueError, match="Column 'sequence' not found in provided DataFrame"
    ):
        predict(model_info, test_data)
