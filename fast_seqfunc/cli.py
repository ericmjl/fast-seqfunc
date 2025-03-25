"""Custom CLI for fast-seqfunc.

This module provides a command-line interface for training sequence-function models
and making predictions on new sequences.

Typer's docs can be found at:
    https://typer.tiangolo.com
"""

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger

from fast_seqfunc import synthetic
from fast_seqfunc.core import (
    evaluate_model,
    load_model,
    predict,
    save_model,
    train_model,
)

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
        "regression", help="Model type: regression or classification"
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
):
    """Generate predictions for new sequences using a trained model."""
    logger.info(f"Loading model from {model_path}...")
    model_info = load_model(model_path)

    # Load input data
    logger.info(f"Loading sequences from {input_data}...")
    data = pd.read_csv(input_data)

    # Check if sequence column exists
    if sequence_col not in data.columns:
        logger.error(f"Column '{sequence_col}' not found in input data")
        raise typer.Exit(1)

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = predict(
        model_info=model_info,
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
        "regression", help="Model type: regression or classification"
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
            model_info = train_model(
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

                # Extract model components
                model = model_info["model"]
                embedder = model_info["embedder"]
                embed_cols = model_info["embed_cols"]

                metrics = evaluate_model(
                    model=model,
                    X_test=test_df[sequence_col],
                    y_test=test_df[target_col],
                    embedder=embedder,
                    model_type=model_type,
                    embed_cols=embed_cols,
                )

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


@app.command()
def generate_synthetic(
    task: str = typer.Argument(
        ...,
        help="Type of synthetic data task to generate. Options: g_count, gc_content, "
        "motif_position, motif_count, length_dependent, nonlinear_composition, "
        "interaction, classification, multiclass",
    ),
    output_dir: Path = typer.Option(
        Path("synthetic_data"), help="Directory to save generated datasets"
    ),
    total_count: int = typer.Option(1000, help="Total number of sequences to generate"),
    train_ratio: float = typer.Option(
        0.7, help="Proportion of data to use for training set"
    ),
    val_ratio: float = typer.Option(
        0.15, help="Proportion of data to use for validation set"
    ),
    test_ratio: float = typer.Option(
        0.15, help="Proportion of data to use for test set"
    ),
    split_data: bool = typer.Option(
        True, help="Whether to split data into train/val/test sets"
    ),
    sequence_length: int = typer.Option(
        30, help="Length of each sequence (for fixed-length tasks)"
    ),
    min_length: int = typer.Option(
        20, help="Minimum sequence length (for variable-length tasks)"
    ),
    max_length: int = typer.Option(
        50, help="Maximum sequence length (for variable-length tasks)"
    ),
    noise_level: float = typer.Option(0.1, help="Level of noise to add to the data"),
    sequence_type: str = typer.Option(
        "dna", help="Type of sequences to generate: dna, rna, or protein"
    ),
    alphabet: Optional[str] = typer.Option(
        None, help="Custom alphabet for sequences. Overrides sequence_type if provided."
    ),
    motif: Optional[str] = typer.Option(
        None, help="Custom motif for motif-based tasks"
    ),
    motifs: Optional[str] = typer.Option(
        None, help="Comma-separated list of motifs for motif_count task"
    ),
    weights: Optional[str] = typer.Option(
        None, help="Comma-separated list of weights for motif_count task"
    ),
    prefix: str = typer.Option("", help="Prefix for output filenames"),
    random_seed: Optional[int] = typer.Option(
        None, help="Random seed for reproducibility"
    ),
):
    """Generate synthetic sequence-function data for testing and benchmarking.

    This command creates synthetic datasets with controllable properties and
    complexity to test sequence-function models. Data can be split into
    train/validation/test sets.

    Each task produces a different type of sequence-function relationship:

    - g_count: Linear relationship based on count of G nucleotides
    - gc_content: Linear relationship based on GC content
    - motif_position: Function depends on the position of a motif (nonlinear)
    - motif_count: Function depends on counts of multiple motifs (linear)
    - length_dependent: Function depends on sequence length (nonlinear)
    - nonlinear_composition: Nonlinear function of base composition
    - interaction: Function depends on interactions between positions
    - classification: Binary classification based on presence of motifs
    - multiclass: Multi-class classification based on different patterns

    Example usage:

    $ fast-seqfunc generate-synthetic gc_content --output-dir data/gc_task

    $ fast-seqfunc generate-synthetic motif_position --motif ATCG --noise-level 0.2

    $ fast-seqfunc generate-synthetic classification \
        --sequence-type protein \
        --no-split-data
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    logger.info(f"Generating synthetic data for task: {task}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set alphabet based on sequence type
    if alphabet is None:
        sequence_type = sequence_type.lower()
        if sequence_type == "dna":
            alphabet = "ACGT"
        elif sequence_type == "rna":
            alphabet = "ACGU"
        elif sequence_type == "protein":
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
        else:
            logger.warning(
                f"Unknown sequence type: {sequence_type}. Using DNA alphabet."
            )
            alphabet = "ACGT"

    logger.info(f"Using alphabet: {alphabet}")

    # Task-specific parameters
    task_params: Dict[str, Any] = {}

    # Add common parameters that apply to most tasks
    if task != "length_dependent":
        task_params["length"] = sequence_length

    # We need to patch the generate_random_sequences function to use our alphabet
    # This approach uses monkey patching to avoid having to modify all task functions
    original_generate_random_sequences = synthetic.generate_random_sequences

    def patched_generate_random_sequences(*args, **kwargs):
        """
        Patched version of `generate_random_sequences` that uses a custom alphabet.

        This function overrides the alphabet parameter with our custom alphabet while
        preserving all other parameters passed to the original function.

        :param args: Positional arguments to pass to the original function
        :param kwargs: Keyword arguments to pass to the original function
        :return: Result from the original generate_random_sequences function
        """
        # Override the alphabet parameter with our custom alphabet,
        # but keep other parameters
        kwargs["alphabet"] = alphabet
        return original_generate_random_sequences(*args, **kwargs)

    # Replace the function temporarily
    synthetic.generate_random_sequences = patched_generate_random_sequences

    # Add task-specific parameters based on the task type
    if task == "motif_position":
        # Use custom motif if provided
        if motif:
            task_params["motif"] = motif
        else:
            # Default motif depends on alphabet
            if len(alphabet) == 4:  # DNA/RNA
                task_params["motif"] = "".join(random.sample(alphabet, 4))
            else:  # Protein
                task_params["motif"] = "".join(
                    random.sample(alphabet, min(4, len(alphabet)))
                )
            logger.info(f"Using default motif: {task_params['motif']}")

    elif task == "motif_count":
        # Parse custom motifs if provided
        if motifs:
            task_params["motifs"] = [m.strip() for m in motifs.split(",")]
        else:
            # Generate default motifs based on alphabet
            if len(alphabet) <= 8:  # DNA/RNA
                task_params["motifs"] = [
                    "".join(random.sample(alphabet, 2)) for _ in range(4)
                ]
            else:  # Protein
                task_params["motifs"] = [
                    "".join(random.sample(alphabet, 3)) for _ in range(4)
                ]
            logger.info(f"Using default motifs: {task_params['motifs']}")

        # Parse custom weights if provided
        if weights:
            try:
                weight_values = [float(w.strip()) for w in weights.split(",")]
                if len(weight_values) != len(task_params["motifs"]):
                    logger.warning(
                        "Number of weights doesn't match number of motifs. "
                        "Using default weights."
                    )
                    task_params["weights"] = [1.0, -0.5, 2.0, -1.5]
                else:
                    task_params["weights"] = weight_values
            except ValueError:
                logger.warning("Invalid weight values. Using default weights.")
                task_params["weights"] = [1.0, -0.5, 2.0, -1.5]
        else:
            task_params["weights"] = [1.0, -0.5, 2.0, -1.5]

    elif task == "length_dependent":
        task_params["min_length"] = min_length
        task_params["max_length"] = max_length

    # Validate the task
    valid_tasks = [
        "g_count",
        "gc_content",
        "motif_position",
        "motif_count",
        "length_dependent",
        "nonlinear_composition",
        "interaction",
        "classification",
        "multiclass",
    ]

    if task not in valid_tasks:
        logger.error(
            f"Invalid task: {task}. Valid options are: {', '.join(valid_tasks)}"
        )
        raise typer.Exit(1)

    # The task functions don't directly accept an alphabet parameter
    # so we need to remove it from task_params
    if "alphabet" in task_params:
        del task_params["alphabet"]

    # Generate the dataset
    try:
        df = synthetic.generate_dataset_by_task(
            task=task, count=total_count, noise_level=noise_level, **task_params
        )

        logger.info(f"Generated {len(df)} sequences for task: {task}")

        # Create filename prefix if provided
        file_prefix = f"{prefix}_" if prefix else ""

        # Save the full dataset if not splitting
        if not split_data:
            output_path = output_dir / f"{file_prefix}{task}_data.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved full dataset to {output_path}")
            # Restore original function
            synthetic.generate_random_sequences = original_generate_random_sequences
            return

        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            logger.warning("Split ratios don't sum to 1.0. Normalizing.")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

        # Shuffle the data
        df = df.sample(frac=1.0, random_state=random_seed)

        # Calculate split indices
        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)

        # Split the data
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]

        # Save the splits
        train_path = output_dir / f"{file_prefix}train.csv"
        val_path = output_dir / f"{file_prefix}val.csv"
        test_path = output_dir / f"{file_prefix}test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved train set ({len(train_df)} samples) to {train_path}")
        logger.info(f"Saved validation set ({len(val_df)} samples) to {val_path}")
        logger.info(f"Saved test set ({len(test_df)} samples) to {test_path}")

        # Save task metadata
        metadata = {
            "task": task,
            "sequence_type": sequence_type,
            "alphabet": alphabet,
            "total_count": total_count,
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "noise_level": noise_level,
            **task_params,
        }

        metadata_path = output_dir / f"{file_prefix}metadata.csv"
        pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")

    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise typer.Exit(1)
    finally:
        # Make sure to restore the original function even if an error occurs
        synthetic.generate_random_sequences = original_generate_random_sequences


@app.command()
def list_synthetic_tasks():
    """List all available synthetic sequence-function data tasks with descriptions."""
    tasks = {
        "g_count": "A simple linear task where the function value is the count of G "
        "nucleotides in the sequence.",
        "gc_content": "A simple linear task where the function value is the GC content "
        "(proportion of G and C) of the sequence.",
        "motif_position": "A nonlinear task where the function value depends on the "
        "position of a specific motif in the sequence.",
        "motif_count": "A linear task where the function value is a weighted sum of "
        "counts of multiple motifs in the sequence.",
        "length_dependent": "A task with variable-length sequences where the function "
        "value depends nonlinearly on the sequence length.",
        "nonlinear_composition": "A complex nonlinear task where the function depends "
        "on ratios between different nucleotide frequencies.",
        "interaction": "A task testing positional interactions, "
        "where specific nucleotide pairs at certain positions "
        "contribute to the function.",
        "classification": "A binary classification task where the class depends on the "
        "presence of specific patterns in the sequence.",
        "multiclass": "A multi-class classification task "
        "with multiple sequence patterns "
        "corresponding to different classes.",
    }

    typer.echo("Available synthetic sequence-function data tasks:")
    typer.echo("")

    for task, description in tasks.items():
        typer.echo(f"{task}:")
        typer.echo(f"  {description}")
        typer.echo("")

    typer.echo("Usage:")
    typer.echo("  fast-seqfunc generate-synthetic TASK [OPTIONS]")
    typer.echo("")
    typer.echo("For detailed options:")
    typer.echo("  fast-seqfunc generate-synthetic --help")


if __name__ == "__main__":
    app()
