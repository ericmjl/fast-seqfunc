#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fast-seqfunc",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "pycaret[full]>=3.0.0",
#   "scikit-learn>=1.0.0",
#   "fast-seqfunc @ git+https://github.com/ericmjl/fast-seqfunc.git@first-implementation",
# ]
# ///

"""
Basic usage example for fast-seqfunc.

This script demonstrates how to:
1. Generate synthetic DNA sequence-function data
2. Train a sequence-function model using one-hot encoding
3. Evaluate the model
4. Make predictions on new sequences
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fast_seqfunc import load_model, predict, save_model, train_model

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_random_nucleotide(length=100):
    """Generate a random DNA sequence of specified length."""
    nucleotides = "ACGT"
    return "".join(random.choice(nucleotides) for _ in range(length))


def generate_synthetic_data(n_samples=1000, seq_length=100):
    """Generate synthetic sequence-function data.

    Creates sequences with a simple pattern:
    - Higher function value if more 'A' and 'G' nucleotides
    - Lower function value if more 'C' and 'T' nucleotides
    """
    sequences = []
    functions = []

    for _ in range(n_samples):
        # Generate random DNA sequence
        seq = generate_random_nucleotide(seq_length)
        sequences.append(seq)

        # Calculate function value based on simple rules
        # More A and G -> higher function
        a_count = seq.count("A")
        g_count = seq.count("G")
        c_count = seq.count("C")
        t_count = seq.count("T")

        # Simple function with some noise
        func_value = (
            0.5 * (a_count + g_count) / seq_length
            - 0.3 * (c_count + t_count) / seq_length
        )
        # func_value += np.random.normal(0, 0.1)  # Add noise
        functions.append(func_value)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "sequence": sequences,
            "function": functions,
        }
    )

    return df


def main():
    """Run the example pipeline."""
    print("Fast-SeqFunc Basic Example")
    print("=========================\n")

    # Create directory for outputs
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 5000
    all_data = generate_synthetic_data(n_samples=n_samples)

    # Split into train and test sets (validation handled internally)
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size

    train_data = all_data[:train_size].copy()
    test_data = all_data[train_size:].copy()

    print(f"Data split: {train_size} train, {test_size} test samples")

    # Save data files
    train_data.to_csv(output_dir / "train_data.csv", index=False)
    test_data.to_csv(output_dir / "test_data.csv", index=False)

    # Train and compare multiple models automatically
    print("\nTraining and comparing sequence-function models...")
    model_info = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        embedding_method="one-hot",
        model_type="regression",
        optimization_metric="r2",  # Optimize for R-squared
    )

    # Display test results if available
    if model_info.get("test_results"):
        print("\nTest metrics from training:")
        for metric, value in model_info["test_results"].items():
            print(f"  {metric}: {value:.4f}")

    # Save the model
    model_path = output_dir / "model.pkl"
    save_model(model_info, model_path)
    print(f"Model saved to {model_path}")

    # Make predictions on test data
    print("\nMaking predictions on test data...")
    test_predictions = predict(model_info, test_data["sequence"])

    # Create a results DataFrame
    results_df = test_data.copy()
    results_df["prediction"] = test_predictions
    results_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Calculate metrics manually
    true_values = test_data["function"]
    mse = ((test_predictions - true_values) ** 2).mean()
    r2 = (
        1
        - ((test_predictions - true_values) ** 2).sum()
        / ((true_values - true_values.mean()) ** 2).sum()
    )

    print("Manual test metrics calculation:")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  R²: {r2:.4f}")

    # Create a scatter plot of true vs predicted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=true_values, y=test_predictions, alpha=0.6)
    plt.plot(
        [min(true_values), max(true_values)],
        [min(true_values), max(true_values)],
        "r--",
    )
    plt.xlabel("True Function Value")
    plt.ylabel("Predicted Function Value")
    plt.title("True vs Predicted Function Values")
    plt.tight_layout()
    plt.savefig(output_dir / "true_vs_predicted.png", dpi=300)
    print(f"Plot saved to {output_dir / 'true_vs_predicted.png'}")

    # Create plots showing function score vs nucleotide counts
    print("\nCreating nucleotide count vs function plots...")

    # Calculate nucleotide counts for all sequences
    all_data_with_counts = all_data.copy()
    all_data_with_counts["A_count"] = all_data["sequence"].apply(lambda x: x.count("A"))
    all_data_with_counts["G_count"] = all_data["sequence"].apply(lambda x: x.count("G"))
    all_data_with_counts["C_count"] = all_data["sequence"].apply(lambda x: x.count("C"))
    all_data_with_counts["T_count"] = all_data["sequence"].apply(lambda x: x.count("T"))

    # Create a 2x2 grid of scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot function vs A count
    sns.scatterplot(
        x="A_count", y="function", data=all_data_with_counts, alpha=0.6, ax=axes[0, 0]
    )
    axes[0, 0].set_title("Function vs A Count")
    axes[0, 0].set_xlabel("Number of A's")
    axes[0, 0].set_ylabel("Function Value")

    # Plot function vs G count
    sns.scatterplot(
        x="G_count", y="function", data=all_data_with_counts, alpha=0.6, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Function vs G Count")
    axes[0, 1].set_xlabel("Number of G's")
    axes[0, 1].set_ylabel("Function Value")

    # Plot function vs C count
    sns.scatterplot(
        x="C_count", y="function", data=all_data_with_counts, alpha=0.6, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Function vs C Count")
    axes[1, 0].set_xlabel("Number of C's")
    axes[1, 0].set_ylabel("Function Value")

    # Plot function vs T count
    sns.scatterplot(
        x="T_count", y="function", data=all_data_with_counts, alpha=0.6, ax=axes[1, 1]
    )
    axes[1, 1].set_title("Function vs T Count")
    axes[1, 1].set_xlabel("Number of T's")
    axes[1, 1].set_ylabel("Function Value")

    plt.tight_layout()
    plt.savefig(output_dir / "nucleotide_counts_vs_function.png", dpi=300)
    print(
        f"Nucleotide count plots saved to "
        f"{output_dir / 'nucleotide_counts_vs_function.png'}"
    )

    # Test loading the model
    print("\nTesting model loading...")
    load_model(model_path)
    print("Model loaded successfully")


if __name__ == "__main__":
    main()
