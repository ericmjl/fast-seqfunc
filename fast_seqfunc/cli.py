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
    save_detailed_metrics,
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
    output_dir: Path = typer.Option(
        Path("outputs"), help="Directory for all outputs (model, metrics, cache)"
    ),
    model_filename: str = typer.Option(
        "model.pkl", help="Filename for the saved model within the output directory"
    ),
):
    """Train a sequence-function model on protein or nucleotide sequences."""
    logger.info(f"Training model using {embedding_method} embeddings...")

    # Create the output directory and its subdirectories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for metrics and cache
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Model path will be in the main output directory
    model_path = output_dir / model_filename

    # Parse embedding methods if multiple are provided
    if "," in embedding_method:
        embedding_method = [m.strip() for m in embedding_method.split(",")]

    # Train the model
    model_info = train_model(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        sequence_col=sequence_col,
        target_col=target_col,
        embedding_method=embedding_method,
        model_type=model_type,
        cache_dir=cache_dir,
    )

    # Get the PyCaret model
    model = model_info["model"]

    # If test data was provided, we'll have test results to save
    test_results = model_info.get("test_results")

    # Save detailed metrics if test results are available
    if test_results:
        # If embedding_method is a list, convert it to a string for filenames
        embedding_str = embedding_method
        if isinstance(embedding_method, list):
            embedding_str = "_".join(embedding_method)

        # Process test results and generate metrics
        _save_test_metrics(test_results, metrics_dir, model_type, embedding_str, model)

    # Save the trained model
    save_model(model_info, model_path)
    logger.info(f"Model saved to {model_path}")

    # Create and save summary information
    _save_model_summary(
        output_dir, model_path, metrics_dir, cache_dir, embedding_method, model_type
    )

    # Clean up any leftover PNG files
    _cleanup_png_files()

    logger.info(f"All outputs saved to {output_dir}")


def _save_model_summary(
    output_dir: Path,
    model_path: Path,
    metrics_dir: Path,
    cache_dir: Path,
    embedding_method: str,
    model_type: str,
) -> None:
    """Create and save a summary of model information.

    :param output_dir: Directory for all outputs
    :param model_path: Path to the saved model
    :param metrics_dir: Directory for metrics
    :param cache_dir: Directory for cache files
    :param embedding_method: Method used for embedding
    :param model_type: Type of model (regression or classification)
    """
    # Save a summary of the output locations
    summary = {
        "model_path": str(model_path),
        "metrics_dir": str(metrics_dir),
        "cache_dir": str(cache_dir),
        "embedding_method": embedding_method,
        "model_type": model_type,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save summary as JSON
    with open(output_dir / "summary.json", "w") as f:
        import json

        json.dump(summary, f, indent=2)


def _save_test_metrics(
    test_results: Dict[str, Any],
    metrics_dir: Path,
    model_type: str,
    embedding_str: str,
    model: Any,
) -> None:
    """Save detailed metrics and generate plots for test results.

    :param test_results: Test results from model evaluation
    :param metrics_dir: Directory to save metrics
    :param model_type: Type of model (regression or classification)
    :param embedding_str: String representation of embedding method
    :param model: Trained model
    """
    logger.info(f"Saving detailed performance metrics to {metrics_dir}")

    # Save detailed metrics using our custom function
    save_detailed_metrics(
        metrics_data=test_results,
        output_dir=metrics_dir,
        model_type=model_type,
        embedding_method=embedding_str,
    )

    # Generate plots using PyCaret
    _generate_model_plots(model, model_type, metrics_dir, embedding_str)

    # Try to save HTML performance report
    _save_performance_html(model_type, metrics_dir, embedding_str)


def _generate_model_plots(
    model: Any,
    model_type: str,
    metrics_dir: Path,
    embedding_str: str,
) -> None:
    """Generate and save performance plots for the model.

    :param model: The trained model
    :param model_type: Type of model (regression or classification)
    :param metrics_dir: Directory to save metrics and plots
    :param embedding_str: String representation of embedding method
    """
    try:
        plot_types = []
        if model_type == "regression":
            from pycaret.regression import plot_model

            plot_types = [
                "residuals",
                "error",
                "feature",
                "cooks",
                "learning",
                "vc",
                "manifold",
            ]
        else:  # classification
            from pycaret.classification import plot_model

            plot_types = [
                "auc",
                "confusion_matrix",
                "boundary",
                "pr",
                "class_report",
                "feature",
                "learning",
                "manifold",
            ]

        for plot_type in plot_types:
            try:
                logger.info(f"Generating {plot_type} plot...")
                # In PyCaret 3.0, plot_model with save=True returns a string path
                # to the saved file or a figure object depending on the PyCaret version
                result = plot_model(
                    model,
                    plot=plot_type,
                    save=True,
                    verbose=False,
                )
                _handle_plot_result(result, plot_type, metrics_dir, embedding_str)
            except Exception as e:
                logger.warning(f"Failed to generate {plot_type} plot: {e}")
    except Exception as e:
        logger.warning(f"Error generating PyCaret plots: {e}")


def _save_performance_html(
    model_type: str,
    metrics_dir: Path,
    embedding_str: str,
) -> None:
    """Save a detailed HTML performance report.

    :param model_type: Type of model (regression or classification)
    :param metrics_dir: Directory to save metrics
    :param embedding_str: String representation of embedding method
    """
    try:
        if model_type == "regression":
            from pycaret.regression import pull
        else:  # classification
            from pycaret.classification import pull

        performance_html = metrics_dir / f"{embedding_str}_performance.html"
        with open(performance_html, "w") as f:
            f.write(pull().to_html())
        logger.info(f"Saved performance HTML report to {performance_html}")
    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")


@app.command()
def predict_cmd(
    model_path: Path = typer.Argument(..., help="Path to saved model"),
    input_data: Path = typer.Argument(
        ..., help="Path to CSV file with sequences to predict"
    ),
    sequence_col: str = typer.Option("sequence", help="Column name for sequences"),
    output_dir: Path = typer.Option(
        Path("prediction_outputs"), help="Directory to save prediction results"
    ),
    predictions_filename: str = typer.Option(
        "predictions.csv",
        help="Filename for predictions CSV within the output directory",
    ),
):
    """Generate predictions for new sequences using a trained model."""
    logger.info(f"Loading model from {model_path}...")
    model_info = load_model(model_path)

    # Create the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path for predictions file
    predictions_path = output_dir / predictions_filename

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
    result_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")

    # Try to create a histogram of predictions if numerical
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Only create histogram for numerical predictions
        if np.issubdtype(predictions.dtype, np.number):
            plt.figure(figsize=(10, 6))
            plt.hist(predictions, bins=30, alpha=0.7)
            plt.xlabel("Predicted Values")
            plt.ylabel("Frequency")
            plt.title("Distribution of Predictions")
            plt.savefig(output_dir / "predictions_histogram.png")
            plt.close()
            logger.info(
                f"Saved predictions histogram to "
                f"{output_dir / 'predictions_histogram.png'}"
            )
    except Exception as e:
        logger.warning(f"Could not create prediction histogram: {e}")

    # Save a summary of the prediction
    summary = {
        "model_path": str(model_path),
        "input_data": str(input_data),
        "predictions_file": str(predictions_path),
        "sequence_count": len(data),
        "model_type": model_info.get("model_type", "unknown"),
        "embedding_method": str(model_info.get("embedding_method", "unknown")),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save summary as JSON
    with open(output_dir / "prediction_summary.json", "w") as f:
        import json

        json.dump(summary, f, indent=2)

    logger.info(f"All prediction outputs saved to {output_dir}")


def _handle_plot_result(
    result,
    plot_type: str,
    output_dir: Path,
    method_name: str,
) -> None:
    """Handle the result of calling plot_model, which could be a string path or figure
    object.

    :param result: Result from plot_model call (could be str, figure object, or None)
    :param plot_type: The type of plot being generated
    :param output_dir: Directory to save the plot
    :param method_name: Name of the embedding method for the filename
    """
    import os
    import shutil

    if result is None:
        logger.warning(f"No result returned for {plot_type} plot")
        return

    if isinstance(result, str):
        # If a string is returned, it's the path to the saved file
        source_file = result

        # If the path doesn't exist but the file might be in the current directory
        if not os.path.exists(source_file):
            # PyCaret 3.0 may generate files with names like "-N Residuals.png"
            possible_files = [
                f
                for f in os.listdir()
                if f.endswith(".png")
                and (
                    plot_type.lower() in f.lower()
                    or plot_type.lower().replace("_", " ") in f.lower()
                )
            ]
            if possible_files:
                source_file = possible_files[0]

        if os.path.exists(source_file):
            destination_file = output_dir / f"{method_name}_{plot_type}.png"
            shutil.move(source_file, destination_file)
            logger.info(
                f"Moved {plot_type} plot from {source_file} to {destination_file}"
            )
        else:
            logger.warning(f"Could not find saved {plot_type} plot file")

    elif hasattr(result, "figure_"):
        # If a figure object is returned, save it directly
        save_path = output_dir / f"{method_name}_{plot_type}.png"
        try:
            result.figure_.savefig(save_path)
            logger.info(f"Saved {plot_type} plot to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save {plot_type} plot: {e}")


def _cleanup_png_files() -> None:
    """Clean up any leftover PNG files in the root directory."""
    try:
        import glob
        import os

        # Find all PNG files in the current directory
        png_files = glob.glob("*.png")
        for png_file in png_files:
            # Only remove files with PyCaret-style naming
            # like "-N Feature Importance.png"
            if png_file.startswith("-") or "plot" in png_file.lower():
                logger.info(f"Cleaning up leftover file: {png_file}")
                os.remove(png_file)
    except Exception as e:
        logger.warning(f"Error cleaning up PNG files: {e}")


def _create_comparison_plot(
    results_df: pd.DataFrame, model_type: str, output_dir: Path
) -> None:
    """Create a comparison plot for multiple embedding methods.

    :param results_df: DataFrame with metrics for different embedding methods
    :param model_type: Type of model (regression or classification)
    :param output_dir: Directory to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set up the figure
        plt.figure(figsize=(12, 8))

        # For each metric, create a grouped bar chart
        if model_type == "regression":
            metrics_to_plot = ["r2", "rmse", "mae"]
        else:  # classification
            metrics_to_plot = ["accuracy", "f1", "precision", "recall"]

        # Filter to just include metrics we have
        available_metrics = [m for m in metrics_to_plot if m in results_df.columns]

        if available_metrics:
            # Melt the dataframe to get it in the right format for seaborn
            melted_df = pd.melt(
                results_df,
                id_vars=["embedding_method"],
                value_vars=available_metrics,
                var_name="Metric",
                value_name="Value",
            )

            # Create the plot
            sns.barplot(x="Metric", y="Value", hue="embedding_method", data=melted_df)
            plt.title(f"Comparison of Embedding Methods ({model_type})")
            plt.ylabel("Score")
            plt.tight_layout()

            # Save the figure
            comparison_plot_path = output_dir / "embedding_comparison_plot.png"
            plt.savefig(comparison_plot_path)
            plt.close()
            logger.info(f"Saved embedding comparison plot to {comparison_plot_path}")
    except Exception as e:
        logger.warning(f"Failed to create comparison plot: {e}")


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
    output_dir: Path = typer.Option(
        Path("comparison_outputs"),
        help="Directory for all outputs (results, metrics, models, cache)",
    ),
):
    """Compare different embedding methods on the same dataset."""
    logger.info("Comparing embedding methods...")

    # Create the output directory and its subdirectories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Results file path
    results_file = output_dir / "embedding_comparison.csv"

    # List of embedding methods to compare
    embedding_methods = ["one-hot", "carp", "esm2"]
    results = []

    # Dictionary to store model info for each method
    models_info = {}

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

            # Save the model
            model_path = models_dir / f"{method}_model.pkl"
            save_model(model_info, model_path)
            logger.info(f"Saved {method} model to {model_path}")

            # Store model info for later reference
            models_info[method] = model_info

            # Evaluate on test data if provided
            if not test_data:
                continue

            test_df = pd.read_csv(test_data)

            # Extract model components
            model = model_info["model"]
            embedder = model_info["embedder"]
            embed_cols = model_info["embed_cols"]

            # Get full test results
            test_results = evaluate_model(
                model=model,
                X_test=test_df[sequence_col],
                y_test=test_df[target_col],
                embedder=embedder,
                model_type=model_type,
                embed_cols=embed_cols,
            )

            # Save detailed metrics
            logger.info(f"Saving detailed metrics for {method} embedding")

            # Save detailed metrics using our custom function
            save_detailed_metrics(
                metrics_data=test_results,
                output_dir=metrics_dir,
                model_type=model_type,
                embedding_method=method,
            )

            # Generate plots for this model
            _generate_plots(model, model_type, metrics_dir, method)

            # Add method and metrics to results for the comparison table
            result = {"embedding_method": method, **test_results.get("metrics", {})}
            results.append(result)

        except Exception as e:
            logger.error(f"Error training with {method}: {e}")

    # Create DataFrame with results
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(results_file, index=False)
    logger.info(f"Comparison results saved to {results_file}")

    # Create comparison plots if we have metrics for multiple methods
    if len(results) > 1:
        _create_comparison_plot(results_df, model_type, output_dir)

    # Save a summary of the output locations
    summary = {
        "results_file": str(results_file),
        "metrics_dir": str(metrics_dir),
        "models_dir": str(models_dir),
        "cache_dir": str(cache_dir),
        "embedding_methods": embedding_methods,
        "model_type": model_type,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save summary as JSON
    with open(output_dir / "summary.json", "w") as f:
        import json

        json.dump(summary, f, indent=2)

    # Clean up any leftover PNG files
    _cleanup_png_files()

    logger.info(f"All outputs saved to {output_dir}")


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


def _generate_plots(model, model_type: str, metrics_dir: Path, method: str) -> None:
    """Generate and save performance plots for a model.

    :param model: The trained model
    :param model_type: Type of model (regression or classification)
    :param metrics_dir: Directory to save metrics and plots
    :param method: Name of the embedding method
    """
    try:
        plot_types = []
        if model_type == "regression":
            from pycaret.regression import plot_model

            plot_types = ["residuals", "error", "feature", "cooks", "learning"]
        else:  # classification
            from pycaret.classification import plot_model

            plot_types = ["auc", "confusion_matrix", "boundary", "pr", "class_report"]

        for plot_type in plot_types:
            try:
                logger.info(f"Generating {plot_type} plot for {method}...")
                # In PyCaret 3.0, plot_model with save=True returns a string path
                # to the saved file or a figure object depending on the PyCaret version
                result = plot_model(
                    model,
                    plot=plot_type,
                    save=True,
                    verbose=False,
                )
                _handle_plot_result(result, plot_type, metrics_dir, method)
            except Exception as e:
                logger.warning(f"Failed to generate {plot_type} plot for {method}: {e}")
    except Exception as e:
        logger.warning(f"Error generating PyCaret plots for {method}: {e}")


if __name__ == "__main__":
    app()
