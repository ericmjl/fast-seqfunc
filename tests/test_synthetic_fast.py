"""Fast unit tests for synthetic data generation functions in fast-seqfunc."""

import numpy as np
import pandas as pd
import pytest

from fast_seqfunc.synthetic import (
    add_additional_predictors,
    generate_dataset_with_predictors,
)


# Mock the generate_dataset_by_task function to avoid testing the entire pipeline
@pytest.fixture
def mock_dataset():
    """Create a simple dataset for testing additional predictors."""
    return pd.DataFrame(
        {"sequence": ["ACGT", "CGTA", "GTAC", "TACG"], "function": [0.1, 0.2, 0.3, 0.4]}
    )


def test_add_additional_predictors_default_parameters(mock_dataset):
    """Test add_additional_predictors function with default parameters."""
    result = add_additional_predictors(mock_dataset)

    # Check that original columns are preserved
    assert "sequence" in result.columns
    assert "function" in result.columns

    # Check that correct number of predictor columns were added
    assert sum(col.startswith("predictor") for col in result.columns) == 2
    assert sum(col.startswith("categorical_pred") for col in result.columns) == 1

    # Check that row count is preserved
    assert len(result) == len(mock_dataset)


def test_add_additional_predictors_numeric_only(mock_dataset):
    """Test add_additional_predictors with only numeric predictors."""
    result = add_additional_predictors(
        mock_dataset, num_numeric_predictors=3, num_categorical_predictors=0
    )

    # Check that numeric predictors were added
    assert sum(col.startswith("predictor") for col in result.columns) == 3
    assert all(f"predictor{i + 1}" in result.columns for i in range(3))

    # Check that no categorical predictors were added
    assert sum(col.startswith("categorical_pred") for col in result.columns) == 0


def test_add_additional_predictors_categorical_only(mock_dataset):
    """Test add_additional_predictors with only categorical predictors."""
    result = add_additional_predictors(
        mock_dataset, num_numeric_predictors=0, num_categorical_predictors=2
    )

    # Check that no numeric predictors were added
    assert sum(col.startswith("predictor") for col in result.columns) == 0

    # Check that categorical predictors were added
    assert sum(col.startswith("categorical_pred") for col in result.columns) == 2
    assert all(f"categorical_pred{i + 1}" in result.columns for i in range(2))

    # Check that categorical values are correct
    for i in range(2):
        unique_values = result[f"categorical_pred{i + 1}"].unique()
        assert all(val in ["A", "B", "C"] for val in unique_values)


def test_add_additional_predictors_custom_categories(mock_dataset):
    """Test add_additional_predictors with custom categorical values."""
    custom_categories = ["X", "Y", "Z"]
    result = add_additional_predictors(
        mock_dataset,
        num_numeric_predictors=0,
        num_categorical_predictors=1,
        categorical_values=custom_categories,
    )

    # Check that categorical values match custom categories
    unique_values = result["categorical_pred1"].unique()
    assert all(val in custom_categories for val in unique_values)


def test_add_additional_predictors_correlation_strength(mock_dataset):
    """Test add_additional_predictors with different correlation strengths."""
    # With low correlation, predictor should be mostly random
    low_corr_result = add_additional_predictors(
        mock_dataset,
        num_numeric_predictors=1,
        num_categorical_predictors=0,
        correlation_strength=0.0,  # No correlation
    )

    # With high correlation, predictor should follow target closely
    high_corr_result = add_additional_predictors(
        mock_dataset,
        num_numeric_predictors=1,
        num_categorical_predictors=0,
        correlation_strength=1.0,  # Perfect correlation
    )

    # The high correlation dataset should have a stronger correlation between
    # predictor1 and function than the low correlation dataset
    low_corr = np.corrcoef(low_corr_result["function"], low_corr_result["predictor1"])[
        0, 1
    ]
    high_corr = np.corrcoef(
        high_corr_result["function"], high_corr_result["predictor1"]
    )[0, 1]

    # Check that high_corr is greater than low_corr
    # We can't check exact values due to randomness in the generation process
    assert abs(high_corr) > abs(low_corr)


def test_add_additional_predictors_target_column(mock_dataset):
    """Test add_additional_predictors with custom target column."""
    # Add a custom target column
    df = mock_dataset.copy()
    df["custom_target"] = [0.5, 0.6, 0.7, 0.8]

    result = add_additional_predictors(
        df,
        target_col="custom_target",
        num_numeric_predictors=1,
        num_categorical_predictors=0,
    )

    # Check correlation between predictor and custom target
    corr = np.corrcoef(result["custom_target"], result["predictor1"])[0, 1]
    assert not np.isnan(corr)  # There should be some correlation


# Integration test with mocked generate_dataset_by_task
def test_generate_dataset_with_predictors(monkeypatch):
    """Test generate_dataset_with_predictors function."""

    # Mock the generate_dataset_by_task function
    def mock_generate_dataset_by_task(task, count, noise_level, **kwargs):
        return pd.DataFrame(
            {"sequence": ["ACGT"] * count, "function": np.random.rand(count)}
        )

    monkeypatch.setattr(
        "fast_seqfunc.synthetic.generate_dataset_by_task", mock_generate_dataset_by_task
    )

    # Generate dataset with predictors
    result = generate_dataset_with_predictors(
        task="mocked_task",
        count=10,
        noise_level=0.1,
        num_numeric_predictors=1,
        num_categorical_predictors=1,
        correlation_strength=0.5,
    )

    # Check that the result has the correct structure
    assert len(result) == 10
    assert "sequence" in result.columns
    assert "function" in result.columns
    assert "predictor1" in result.columns
    assert "categorical_pred1" in result.columns
