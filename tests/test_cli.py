"""Tests for fast_seqfunc.cli."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fast_seqfunc.cli import app
from fast_seqfunc.synthetic import (
    create_classification_task,
    create_g_count_task,
    create_multiclass_task,
    generate_dataset_by_task,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    # Clean up after tests
    shutil.rmtree(tmp_dir)


@pytest.fixture
def g_count_data(temp_dir):
    """Generate a G-count dataset and save to CSV."""
    # Generate a simple dataset where the function is the count of G's
    df = create_g_count_task(count=500, length=20, noise_level=0.1)

    # Save to CSV in the temp directory
    data_path = Path(temp_dir) / "g_count_data.csv"
    df.to_csv(data_path, index=False)

    return data_path


@pytest.fixture
def binary_classification_data(temp_dir):
    """Generate a classification dataset and save to CSV."""
    df = create_classification_task(count=500, length=20, noise_level=0.1)

    # Save to CSV in the temp directory
    data_path = Path(temp_dir) / "classification_data.csv"
    df.to_csv(data_path, index=False)

    return data_path


@pytest.fixture
def multiclass_data(temp_dir):
    """Generate a multi-class dataset and save to CSV."""
    df = create_multiclass_task(count=500, length=20, noise_level=0.1)

    # Save to CSV in the temp directory
    data_path = Path(temp_dir) / "multiclass_data.csv"
    df.to_csv(data_path, index=False)

    return data_path


@pytest.fixture
def test_tasks():
    """Define a list of test tasks."""
    return [
        "g_count",
        "gc_content",
        "motif_position",
        "motif_count",
        "nonlinear_composition",
        "interaction",
    ]


def test_cli_hello():
    """Test the hello command."""
    runner = CliRunner()
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "fast-seqfunc" in result.stdout


def test_cli_describe():
    """Test the describe command."""
    runner = CliRunner()
    result = runner.invoke(app, ["describe"])
    assert result.exit_code == 0
    assert "sequence-function" in result.stdout


def test_cli_g_count_regression(g_count_data, temp_dir):
    """Test CLI with G-count regression task."""
    runner = CliRunner()
    model_path = Path(temp_dir) / "model.pkl"

    # Train model
    result = runner.invoke(
        app,
        [
            "train",
            str(g_count_data),
            "--sequence-col",
            "sequence",
            "--target-col",
            "function",
            "--embedding-method",
            "one-hot",
            "--model-type",
            "regression",
            "--output-path",
            str(model_path),
        ],
    )

    assert result.exit_code == 0
    assert model_path.exists()

    # Make predictions
    predictions_path = Path(temp_dir) / "predictions.csv"
    result = runner.invoke(
        app,
        [
            "predict-cmd",
            str(model_path),
            str(g_count_data),
            "--sequence-col",
            "sequence",
            "--output-path",
            str(predictions_path),
        ],
    )

    assert result.exit_code == 0
    assert predictions_path.exists()

    # Verify predictions file has expected columns
    predictions_df = pd.read_csv(predictions_path)
    assert "sequence" in predictions_df.columns
    assert "prediction" in predictions_df.columns


def test_cli_classification(binary_classification_data, temp_dir):
    """Test CLI with binary classification task."""
    runner = CliRunner()
    model_path = Path(temp_dir) / "model_classification.pkl"

    # Train model
    result = runner.invoke(
        app,
        [
            "train",
            str(binary_classification_data),
            "--sequence-col",
            "sequence",
            "--target-col",
            "function",
            "--embedding-method",
            "one-hot",
            "--model-type",
            "classification",
            "--output-path",
            str(model_path),
        ],
    )

    assert result.exit_code == 0
    assert model_path.exists()

    # Make predictions
    predictions_path = Path(temp_dir) / "predictions_classification.csv"
    result = runner.invoke(
        app,
        [
            "predict-cmd",
            str(model_path),
            str(binary_classification_data),
            "--sequence-col",
            "sequence",
            "--output-path",
            str(predictions_path),
        ],
    )

    assert result.exit_code == 0
    assert predictions_path.exists()

    # Verify predictions file has expected columns
    predictions_df = pd.read_csv(predictions_path)
    assert "sequence" in predictions_df.columns
    assert "prediction" in predictions_df.columns


def test_cli_multiclass(multiclass_data, temp_dir):
    """Test CLI with multi-class classification task."""
    runner = CliRunner()
    model_path = Path(temp_dir) / "model_multiclass.pkl"

    # Train model
    result = runner.invoke(
        app,
        [
            "train",
            str(multiclass_data),
            "--sequence-col",
            "sequence",
            "--target-col",
            "function",
            "--embedding-method",
            "one-hot",
            "--model-type",
            "multi-class",
            "--output-path",
            str(model_path),
        ],
    )

    assert result.exit_code == 0
    assert model_path.exists()

    # Make predictions
    predictions_path = Path(temp_dir) / "predictions_multiclass.csv"
    result = runner.invoke(
        app,
        [
            "predict-cmd",
            str(model_path),
            str(multiclass_data),
            "--sequence-col",
            "sequence",
            "--output-path",
            str(predictions_path),
        ],
    )

    assert result.exit_code == 0
    assert predictions_path.exists()

    # Verify predictions file has expected columns
    predictions_df = pd.read_csv(predictions_path)
    assert "sequence" in predictions_df.columns
    assert "prediction" in predictions_df.columns


def test_cli_with_confidence(g_count_data, temp_dir):
    """Test CLI with confidence estimation."""
    runner = CliRunner()
    model_path = Path(temp_dir) / "model_confidence.pkl"

    # Train model
    result = runner.invoke(
        app, ["train", str(g_count_data), "--output-path", str(model_path)]
    )

    assert result.exit_code == 0
    assert model_path.exists()

    # Make predictions with confidence
    predictions_path = Path(temp_dir) / "predictions_confidence.csv"
    result = runner.invoke(
        app,
        [
            "predict-cmd",
            str(model_path),
            str(g_count_data),
            "--with-confidence",
            "--output-path",
            str(predictions_path),
        ],
    )

    assert result.exit_code == 0
    assert predictions_path.exists()

    # Verify predictions file has expected columns
    predictions_df = pd.read_csv(predictions_path)
    assert "sequence" in predictions_df.columns
    assert "prediction" in predictions_df.columns
    assert "confidence" in predictions_df.columns


def test_cli_compare_embeddings(g_count_data, temp_dir):
    """Test CLI for comparing embedding methods."""
    runner = CliRunner()
    comparison_path = Path(temp_dir) / "embedding_comparison.csv"

    # Run comparison
    result = runner.invoke(
        app,
        [
            "compare-embeddings",
            str(g_count_data),
            "--output-path",
            str(comparison_path),
        ],
    )

    # NOTE: This test might take longer as it compares multiple embedding methods
    # We just check that the command runs without error
    assert result.exit_code == 0

    # The comparison might not complete if some embedding methods aren't available,
    # but the file should at least be created
    assert comparison_path.exists()


@pytest.mark.parametrize(
    "task",
    [
        "g_count",
        "gc_content",
        "motif_position",
    ],
)
def test_cli_with_different_tasks(task, temp_dir):
    """Test CLI with different sequence-function tasks."""
    runner = CliRunner()

    # Generate dataset
    df = generate_dataset_by_task(task=task, count=500, noise_level=0.1)
    data_path = Path(temp_dir) / f"{task}_data.csv"
    df.to_csv(data_path, index=False)

    # Train model
    model_path = Path(temp_dir) / f"{task}_model.pkl"
    result = runner.invoke(
        app, ["train", str(data_path), "--output-path", str(model_path)]
    )

    assert result.exit_code == 0
    assert model_path.exists()

    # Make predictions
    predictions_path = Path(temp_dir) / f"{task}_predictions.csv"
    result = runner.invoke(
        app,
        [
            "predict-cmd",
            str(model_path),
            str(data_path),
            "--output-path",
            str(predictions_path),
        ],
    )

    assert result.exit_code == 0
    assert predictions_path.exists()
