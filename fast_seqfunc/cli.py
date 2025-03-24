"""Custom CLI for fast-seqfunc.

This module provides a command-line interface for training sequence-function models
and making predictions on new sequences.

Typer's docs can be found at:
    https://typer.tiangolo.com
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger

from fast_seqfunc.core import load_model, predict, save_model, train_model

app = typer.Typer()


@app.command()
def train(
    train_data: Path = typer.Argument(..., help="Path to CSV file with training data"),
    sequence_col: str = typer.Option("sequence", help="Column name for sequences"),
    target_col: str = typer.Option("function", help="Column name for target values"),
    val_data: Optional[Path] = typer.Option(
        None, help="Optional path to validation data"
    ),
    test_data: Optional[Path] = typer.Option(None, help="Optional path to test data"),
    embedding_method: str = typer.Option(
        "one-hot", help="Embedding method: one-hot, carp, esm2, or auto"
    ),
    model_type: str = typer.Option(
        "regression", help="Model type: regression, classification, or multi-class"
    ),
    output_path: Path = typer.Option(
        Path("model.pkl"), help="Path to save trained model"
    ),
    cache_dir: Optional[Path] = typer.Option(
        None, help="Directory to cache embeddings"
    ),
):
    """Train a sequence-function model on protein or nucleotide sequences."""
    logger.info(f"Training model using {embedding_method} embeddings...")

    # Parse embedding methods if multiple are provided
    if "," in embedding_method:
        embedding_method = [m.strip() for m in embedding_method.split(",")]

    # Train the model
    model = train_model(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        sequence_col=sequence_col,
        target_col=target_col,
        embedding_method=embedding_method,
        model_type=model_type,
        cache_dir=cache_dir,
    )

    # Save the trained model
    save_model(model, output_path)
    logger.info(f"Model saved to {output_path}")


@app.command()
def predict_cmd(
    model_path: Path = typer.Argument(..., help="Path to saved model"),
    input_data: Path = typer.Argument(
        ..., help="Path to CSV file with sequences to predict"
    ),
    sequence_col: str = typer.Option("sequence", help="Column name for sequences"),
    output_path: Path = typer.Option(
        Path("predictions.csv"), help="Path to save predictions"
    ),
    with_confidence: bool = typer.Option(
        False, help="Include confidence estimates if available"
    ),
):
    """Generate predictions for new sequences using a trained model."""
    logger.info(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Load input data
    logger.info(f"Loading sequences from {input_data}...")
    data = pd.read_csv(input_data)

    # Check if sequence column exists
    if sequence_col not in data.columns:
        logger.error(f"Column '{sequence_col}' not found in input data")
        raise typer.Exit(1)

    # Generate predictions
    logger.info("Generating predictions...")
    if with_confidence:
        predictions, confidence = predict(
            model=model,
            sequences=data[sequence_col],
            return_confidence=True,
        )

        # Save predictions with confidence
        result_df = pd.DataFrame(
            {
                sequence_col: data[sequence_col],
                "prediction": predictions,
                "confidence": confidence,
            }
        )
    else:
        predictions = predict(
            model=model,
            sequences=data[sequence_col],
        )

        # Save predictions
        result_df = pd.DataFrame(
            {
                sequence_col: data[sequence_col],
                "prediction": predictions,
            }
        )

    # Save to CSV
    result_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


@app.command()
def compare_embeddings(
    train_data: Path = typer.Argument(..., help="Path to CSV file with training data"),
    sequence_col: str = typer.Option("sequence", help="Column name for sequences"),
    target_col: str = typer.Option("function", help="Column name for target values"),
    val_data: Optional[Path] = typer.Option(
        None, help="Optional path to validation data"
    ),
    test_data: Optional[Path] = typer.Option(
        None, help="Optional path to test data for final evaluation"
    ),
    model_type: str = typer.Option(
        "regression", help="Model type: regression, classification, or multi-class"
    ),
    output_path: Path = typer.Option(
        Path("embedding_comparison.csv"), help="Path to save comparison results"
    ),
    cache_dir: Optional[Path] = typer.Option(
        None, help="Directory to cache embeddings"
    ),
):
    """Compare different embedding methods on the same dataset."""
    logger.info("Comparing embedding methods...")

    # List of embedding methods to compare
    embedding_methods = ["one-hot", "carp", "esm2"]
    results = []

    # Train models with each embedding method
    for method in embedding_methods:
        try:
            logger.info(f"Training with {method} embeddings...")

            # Train model with this embedding method
            model = train_model(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                sequence_col=sequence_col,
                target_col=target_col,
                embedding_method=method,
                model_type=model_type,
                cache_dir=cache_dir,
            )

            # Evaluate on test data if provided
            if test_data:
                test_df = pd.read_csv(test_data)
                metrics = model.evaluate(test_df[sequence_col], test_df[target_col])

                # Add method and metrics to results
                result = {"embedding_method": method, **metrics}
                results.append(result)
        except Exception as e:
            logger.error(f"Error training with {method}: {e}")

    # Create DataFrame with results
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    logger.info(f"Comparison results saved to {output_path}")


@app.command()
def hello():
    """Echo the project's name."""
    typer.echo("This project's name is fast-seqfunc")


@app.command()
def describe():
    """Describe the project."""
    typer.echo("Painless sequence-function models for proteins and nucleotides.")


if __name__ == "__main__":
    app()
