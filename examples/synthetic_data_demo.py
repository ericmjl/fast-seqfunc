# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fast-seqfunc",
#   "matplotlib",
#   "seaborn",
#   "pandas",
#   "numpy",
# ]
# ///

"""Demo script for generating and visualizing synthetic sequence-function data.

This script demonstrates how to generate various synthetic datasets using
the fast-seqfunc.synthetic module and train models on them.
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from fast_seqfunc.core import predict, train_model
from fast_seqfunc.synthetic import generate_dataset_by_task


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and visualize synthetic sequence-function data"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="g_count",
        choices=[
            "g_count",
            "gc_content",
            "motif_position",
            "motif_count",
            "length_dependent",
            "nonlinear_composition",
            "interaction",
            "classification",
            "multiclass",
        ],
        help="Sequence-function task to generate",
    )
    parser.add_argument(
        "--count", type=int, default=500, help="Number of sequences to generate"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1, help="Noise level to add to the data"
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_data.csv", help="Output file path"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate plots of the data"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train a model on the generated data"
    )

    return parser.parse_args()


def visualize_data(df, task_name):
    """Create visualizations for the generated data.

    :param df: DataFrame with sequences and functions
    :param task_name: Name of the task for plot title
    """
    plt.figure(figsize=(14, 6))

    # For classification tasks, show class distribution
    if task_name in ["classification", "multiclass"]:
        plt.subplot(1, 2, 1)
        df["function"].value_counts().plot(kind="bar")
        plt.title(f"Class Distribution for {task_name}")
        plt.xlabel("Class")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        # Show sequence length distribution
        df["seq_length"] = df["sequence"].apply(len)
        sns.histplot(df["seq_length"], kde=True)
        plt.title("Sequence Length Distribution")
        plt.xlabel("Sequence Length")
    else:
        # For regression tasks, show function distribution
        plt.subplot(1, 2, 1)
        sns.histplot(df["function"], kde=True)
        plt.title(f"Function Distribution for {task_name}")
        plt.xlabel("Function Value")

        plt.subplot(1, 2, 2)
        # For tasks with variable length, plot function vs length
        if task_name == "length_dependent":
            df["seq_length"] = df["sequence"].apply(len)
            sns.scatterplot(x="seq_length", y="function", data=df)
            plt.title("Function vs Sequence Length")
            plt.xlabel("Sequence Length")
            plt.ylabel("Function Value")
        # For GC content, show relationship with function
        elif task_name in ["g_count", "gc_content"]:
            df["gc_content"] = df["sequence"].apply(
                lambda s: (s.count("G") + s.count("C")) / len(s)
            )
            sns.scatterplot(x="gc_content", y="function", data=df)
            plt.title("Function vs GC Content")
            plt.xlabel("GC Content")
            plt.ylabel("Function Value")
        # For other tasks, show example sequences
        else:
            # Sample 10 random sequences to display
            examples = df.sample(min(10, len(df)))
            plt.clf()
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(examples)), examples["function"])
            plt.xticks(range(len(examples)), examples["sequence"], rotation=45)
            plt.title(f"Example Sequences for {task_name}")
            plt.xlabel("Sequence")
            plt.ylabel("Function Value")

    plt.tight_layout()
    plt.savefig(f"{task_name}_visualization.png")
    logger.info(f"Visualization saved to {task_name}_visualization.png")
    plt.close()


def train_and_evaluate(df, task_name):
    """Train a model on the generated data and evaluate it.

    :param df: DataFrame with sequences and functions
    :param task_name: Name of the task
    """
    # Split data into train/test
    np.random.seed(42)
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk].reset_index(drop=True)
    test_df = df[~msk].reset_index(drop=True)

    # Save train/test data to temp files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        train_path = tmp_dir / "train_data.csv"
        test_path = tmp_dir / "test_data.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Determine model type based on task
        if task_name == "classification":
            model_type = "classification"
        elif task_name == "multiclass":
            model_type = "multi-class"
        else:
            model_type = "regression"

        logger.info(f"Training {model_type} model for {task_name} task")

        # Train model
        model = train_model(
            train_data=train_path,
            test_data=test_path,
            sequence_col="sequence",
            target_col="function",
            embedding_method="one-hot",
            model_type=model_type,
        )

        # Make predictions on test data
        predictions = predict(model, test_df["sequence"])

        # Calculate and print metrics
        if model_type == "regression":
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            mae = mean_absolute_error(test_df["function"], predictions)
            rmse = np.sqrt(mean_squared_error(test_df["function"], predictions))
            r2 = r2_score(test_df["function"], predictions)

            logger.info(f"Test MAE: {mae:.4f}")
            logger.info(f"Test RMSE: {rmse:.4f}")
            logger.info(f"Test R²: {r2:.4f}")

            # Scatter plot of actual vs predicted values
            plt.figure(figsize=(8, 8))
            plt.scatter(test_df["function"], predictions, alpha=0.5)
            plt.plot(
                [test_df["function"].min(), test_df["function"].max()],
                [test_df["function"].min(), test_df["function"].max()],
                "k--",
                lw=2,
            )
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Actual vs Predicted for {task_name}")
            plt.savefig(f"{task_name}_predictions.png")
            plt.close()

        else:  # Classification
            from sklearn.metrics import accuracy_score, classification_report

            accuracy = accuracy_score(test_df["function"], predictions.round())
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            report = classification_report(test_df["function"], predictions.round())
            logger.info(report)

            # Confusion matrix
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(test_df["function"], predictions.round())
            plt.figure(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {task_name}")
            plt.savefig(f"{task_name}_confusion_matrix.png")
            plt.close()


def main():
    """Run the demo."""
    args = parse_args()

    logger.info(f"Generating {args.count} sequences for {args.task} task")
    df = generate_dataset_by_task(
        task=args.task,
        count=args.count,
        noise_level=args.noise,
    )

    # Save data to CSV
    df.to_csv(args.output, index=False)
    logger.info(f"Data saved to {args.output}")

    # Generate plots if requested
    if args.plot:
        logger.info("Generating visualizations")
        visualize_data(df, args.task)

    # Train model if requested
    if args.train:
        logger.info("Training model on generated data")
        train_and_evaluate(df, args.task)


if __name__ == "__main__":
    main()
