#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fast-seqfunc",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn>=1.0.0",
#   "fast-seqfunc @ git+https://github.com/ericmjl/fast-seqfunc.git",
# ]
# ///

"""
Variable Length Sequences Example for fast-seqfunc.

This script demonstrates how to:
1. Generate synthetic DNA sequences of variable lengths
2. Use padding options to train a sequence-function model
3. Compare different padding strategies
4. Make predictions on new sequences of different lengths
"""

import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from fast_seqfunc import load_model, predict, save_model, train_model
from fast_seqfunc.embedders import OneHotEmbedder

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_variable_length_sequence(
    min_length: int = 50, max_length: int = 150
) -> str:
    """Generate a random DNA sequence with variable length.

    :param min_length: Minimum sequence length
    :param max_length: Maximum sequence length
    :return: Random DNA sequence
    """
    length = random.randint(min_length, max_length)
    nucleotides = "ACGT"
    return "".join(random.choice(nucleotides) for _ in range(length))


def generate_variable_length_data(
    n_samples: int = 1000, min_length: int = 50, max_length: int = 150
) -> pd.DataFrame:
    """Generate synthetic variable-length sequence-function data.

    The function value depends on:
    1. The GC content (proportion of G and C nucleotides)
    2. The length of the sequence

    :param n_samples: Number of samples to generate
    :param min_length: Minimum sequence length
    :param max_length: Maximum sequence length
    :return: DataFrame with sequences and function values
    """
    sequences = []
    lengths = []

    for _ in range(n_samples):
        seq = generate_variable_length_sequence(min_length, max_length)
        sequences.append(seq)
        lengths.append(len(seq))

    # Calculate function values based on GC content and length
    gc_contents = [(seq.count("G") + seq.count("C")) / len(seq) for seq in sequences]

    # Function value = normalized GC content + normalized length + noise
    normalized_gc = [(gc - 0.5) * 2 for gc in gc_contents]  # -1 to 1
    normalized_length = [
        (length - min_length) / (max_length - min_length) for length in lengths
    ]  # 0 to 1

    functions = [
        0.6 * gc + 0.4 * length + np.random.normal(0, 0.05)
        for gc, length in zip(normalized_gc, normalized_length)
    ]

    # Create DataFrame
    df = pd.DataFrame(
        {
            "sequence": sequences,
            "function": functions,
            "length": lengths,
            "gc_content": gc_contents,
        }
    )

    return df


def compare_padding_strategies(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[dict, dict, dict]:
    """Compare different padding strategies for variable-length sequences.

    :param train_data: Training data
    :param test_data: Test data
    :return: Tuple of model info for each strategy
        (no padding, default padding, custom padding)
    """
    logger.info("Training model with padding disabled...")
    model_no_padding = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        embedding_method="one-hot",
        model_type="regression",
        optimization_metric="r2",
        embedder_kwargs={"pad_sequences": False},
    )

    logger.info("Training model with default padding (gap character '-')...")
    model_default_padding = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        embedding_method="one-hot",
        model_type="regression",
        optimization_metric="r2",
        embedder_kwargs={"pad_sequences": True, "gap_character": "-"},
    )

    logger.info("Training model with custom padding (gap character 'X')...")
    model_custom_padding = train_model(
        train_data=train_data,
        test_data=test_data,
        sequence_col="sequence",
        target_col="function",
        embedding_method="one-hot",
        model_type="regression",
        optimization_metric="r2",
        embedder_kwargs={"pad_sequences": True, "gap_character": "X"},
    )

    return model_no_padding, model_default_padding, model_custom_padding


def demonstrate_embedder_usage() -> None:
    """Demonstrate direct usage of the OneHotEmbedder with padding options."""
    logger.info("Demonstrating direct usage of OneHotEmbedder...")

    # Create some example sequences of different lengths
    sequences = ["ACGT", "AATT", "GCGCGCGC", "A"]
    logger.info(f"Example sequences: {sequences}")

    # Default embedder (pads with '-')
    embedder = OneHotEmbedder(sequence_type="dna")
    embeddings = embedder.fit_transform(sequences)
    logger.info("Default embedder (padding enabled):")
    logger.info(f"  - Embeddings shape: {embeddings.shape}")
    logger.info(f"  - Max length detected: {embedder.max_length}")
    logger.info(f"  - Alphabet: {embedder.alphabet}")

    # Embedder with explicit max_length
    embedder_max = OneHotEmbedder(sequence_type="dna", max_length=10)
    embeddings_max = embedder_max.fit_transform(sequences)
    logger.info("Embedder with explicit max_length=10:")
    logger.info(f"  - Embeddings shape: {embeddings_max.shape}")

    # Embedder with custom gap character
    embedder_custom = OneHotEmbedder(sequence_type="dna", gap_character="X")
    _ = embedder_custom.fit_transform(sequences)
    logger.info("Embedder with custom gap character 'X':")
    logger.info(f"  - Alphabet: {embedder_custom.alphabet}")

    # Embedder with padding disabled
    embedder_no_pad = OneHotEmbedder(sequence_type="dna", pad_sequences=False)
    embeddings_no_pad = embedder_no_pad.fit_transform(sequences)
    logger.info("Embedder with padding disabled:")
    logger.info(f"  - Number of embeddings: {len(embeddings_no_pad)}")
    logger.info("  - Shapes of individual embeddings:")
    for i, emb in enumerate(embeddings_no_pad):
        logger.info(
            f"    - Sequence {i} ({len(sequences[i])} nucleotides): {emb.shape}"
        )


def plot_results(
    test_data: pd.DataFrame,
    models: List[dict],
    model_names: List[str],
    output_dir: Path,
) -> None:
    """Plot comparison of different padding strategies.

    :param test_data: Test data
    :param models: List of trained models
    :param model_names: Names of the models
    :param output_dir: Output directory for plots
    """
    # Plot test predictions for each model
    plt.figure(figsize=(10, 8))

    true_values = test_data["function"]

    for model, name in zip(models, model_names):
        predictions = predict(model, test_data["sequence"])

        # Calculate R²
        r2 = (
            1
            - ((predictions - true_values) ** 2).sum()
            / ((true_values - true_values.mean()) ** 2).sum()
        )

        # Plot
        plt.scatter(
            true_values, predictions, alpha=0.5, label=f"{name} (R² = {r2:.4f})"
        )

    # Plot identity line
    plt.plot(
        [min(true_values), max(true_values)],
        [min(true_values), max(true_values)],
        "r--",
    )

    plt.xlabel("True Function Value")
    plt.ylabel("Predicted Function Value")
    plt.title("Comparison of Padding Strategies for Variable-Length Sequences")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "padding_comparison.png", dpi=300)
    logger.info(f"Plot saved to {output_dir / 'padding_comparison.png'}")

    # Plot function vs length
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="length", y="function", data=test_data, alpha=0.6)
    plt.xlabel("Sequence Length")
    plt.ylabel("Function Value")
    plt.title("Function Value vs Sequence Length")
    plt.tight_layout()
    plt.savefig(output_dir / "function_vs_length.png", dpi=300)
    logger.info(f"Plot saved to {output_dir / 'function_vs_length.png'}")

    # Plot function vs GC content
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="gc_content", y="function", data=test_data, alpha=0.6)
    plt.xlabel("GC Content")
    plt.ylabel("Function Value")
    plt.title("Function Value vs GC Content")
    plt.tight_layout()
    plt.savefig(output_dir / "function_vs_gc_content.png", dpi=300)
    logger.info(f"Plot saved to {output_dir / 'function_vs_gc_content.png'}")


def main() -> None:
    """Run the example pipeline."""
    logger.info("Fast-SeqFunc Variable Length Sequences Example")
    logger.info("============================================")

    # Create directory for outputs
    output_dir = Path("examples/output/variable_length")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    logger.info("Generating synthetic data with variable-length sequences...")
    n_samples = 2000
    min_length = 50
    max_length = 150
    all_data = generate_variable_length_data(
        n_samples=n_samples, min_length=min_length, max_length=max_length
    )

    # Display statistics
    logger.info(
        f"Generated {n_samples} sequences "
        f"with lengths from {min_length} to {max_length}"
    )
    logger.info("Sequence length statistics:")
    logger.info(f"  - Mean: {all_data['length'].mean():.1f}")
    logger.info(f"  - Min: {all_data['length'].min()}")
    logger.info(f"  - Max: {all_data['length'].max()}")

    # Split into train and test sets
    train_size = int(0.8 * n_samples)
    train_data = all_data[:train_size].copy()
    test_data = all_data[train_size:].copy()

    logger.info(
        f"Data split: {train_size} train, {n_samples - train_size} test samples"
    )

    # Save data files
    train_data.to_csv(output_dir / "train_data.csv", index=False)
    test_data.to_csv(output_dir / "test_data.csv", index=False)

    # Demonstrate direct usage of the OneHotEmbedder
    demonstrate_embedder_usage()

    # Compare different padding strategies
    logger.info("\nComparing different padding strategies...")
    model_no_padding, model_default_padding, model_custom_padding = (
        compare_padding_strategies(train_data, test_data)
    )

    # Display test results for each model
    for name, model in [
        ("No Padding", model_no_padding),
        ("Default Padding", model_default_padding),
        ("Custom Padding", model_custom_padding),
    ]:
        if model.get("test_results"):
            logger.info(f"\nTest metrics for {name}:")
            for metric, value in model["test_results"].items():
                logger.info(f"  {metric}: {value:.4f}")

    # Save models
    save_model(model_default_padding, output_dir / "model_default_padding.pkl")
    logger.info(
        f"Default padding model saved to {output_dir / 'model_default_padding.pkl'}"
    )

    # Plot results
    logger.info("\nCreating comparison plots...")
    plot_results(
        test_data,
        [model_no_padding, model_default_padding, model_custom_padding],
        ["No Padding", "Default Padding (-)", "Custom Padding (X)"],
        output_dir,
    )

    # Generate new test sequences with different lengths
    logger.info("\nTesting prediction on new sequences of different lengths...")
    new_sequences = [generate_variable_length_sequence(30, 200) for _ in range(5)]

    # Show predictions using the default padding model
    loaded_model = load_model(output_dir / "model_default_padding.pkl")
    predictions = predict(loaded_model, new_sequences)

    # Display results
    for seq, pred in zip(new_sequences, predictions):
        gc_content = (seq.count("G") + seq.count("C")) / len(seq)
        logger.info(
            f"Sequence length: {len(seq)}, GC content: {gc_content:.2f}, "
            f"Predicted function: {pred:.4f}"
        )


if __name__ == "__main__":
    main()
