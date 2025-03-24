# /// script
# requires-python = "<=3.11"
# dependencies = [
#     "anthropic==0.49.0",
#     "marimo",
#     "numpy==1.26.4",
#     "pandas==2.1.4",
#     "pycaret[full]==3.3.2",
#     "scikit-learn==1.4.2",
# ]
# ///

import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    from itertools import product

    import numpy as np

    # Protein sequence data
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    protein_length = 10
    n_protein_samples = 1000

    # Generate random protein sequences
    protein_sequences = [
        "".join(np.random.choice(list(amino_acids), protein_length))
        for _ in range(n_protein_samples)
    ]

    # Complex function for proteins based on:
    # - hydrophobicity patterns
    # - charge distribution
    # - sequence motif presence
    hydrophobic = "AILMFWV"
    charged = "DEKR"
    motif = "KR"

    def protein_function(seq):
        # Hydrophobicity score
        hydro_score = sum(
            1 for i, aa in enumerate(seq) if aa in hydrophobic and i > len(seq) / 2
        )

        # Charge distribution
        charge_pairs = sum(
            1
            for i in range(len(seq) - 1)
            if seq[i] in charged and seq[i + 1] in charged
        )

        # Motif presence with position weight
        motif_score = sum(i / len(seq) for i, aa in enumerate(seq) if aa in motif)

        # Combine non-linearly
        return (
            np.exp(hydro_score * 0.5)
            + (charge_pairs**2)
            + (motif_score * 3)
            + np.sin(hydro_score * charge_pairs * 0.3)
        )

    protein_values = np.array([protein_function(seq) for seq in protein_sequences])

    # DNA sequence data
    nucleotides = "ACGTU-"
    dna_length = 20
    n_dna_samples = 1000

    # Generate random DNA sequences
    dna_sequences = [
        "".join(np.random.choice(list(nucleotides), dna_length))
        for _ in range(n_dna_samples)
    ]

    # Complex function for DNA based on:
    # - GC content variation
    # - palindrome presence
    # - specific motif positioning
    def dna_function(seq):
        # GC content with position weights
        gc_score = sum(2 / (i + 1) for i, nt in enumerate(seq) if nt in "GC")

        # Palindrome contribution
        palindrome_score = sum(
            1 for i in range(len(seq) - 3) if seq[i : i + 4] == seq[i : i + 4][::-1]
        )

        # TATA-like motif presence with spacing effects
        tata_score = 0
        for i in range(len(seq) - 3):
            if seq[i : i + 2] == "TA" and seq[i + 2 : i + 4] == "TA":
                tata_score += np.log(i + 1)

        # Combine non-linearly
        return (
            np.exp(gc_score * 0.3)
            + (palindrome_score**1.5)
            + np.cos(tata_score) * np.sqrt(gc_score + palindrome_score + 1)
        )

    dna_values = np.array([dna_function(seq) for seq in dna_sequences])

    # Normalize both value sets to similar ranges
    protein_values = (protein_values - protein_values.mean()) / protein_values.std()
    dna_values = (dna_values - dna_values.mean()) / dna_values.std()
    return (
        amino_acids,
        charged,
        dna_function,
        dna_length,
        dna_sequences,
        dna_values,
        hydrophobic,
        motif,
        n_dna_samples,
        n_protein_samples,
        np,
        nucleotides,
        product,
        protein_function,
        protein_length,
        protein_sequences,
        protein_values,
    )


@app.cell
def _(dna_sequences, dna_values, protein_sequences, protein_values):
    import pandas as pd

    protein_df = pd.DataFrame(
        {"sequence": protein_sequences, "function": protein_values}
    )

    dna_df = pd.DataFrame({"sequence": dna_sequences, "function": dna_values})
    return dna_df, pd, protein_df


@app.cell
def _(protein_df):
    protein_df
    return


@app.cell
def _(dna_df):
    dna_df
    return


@app.cell
def _(np):
    def one_hot_encode(sequence, alphabet, flatten=False):
        seq_length = len(sequence)
        alphabet_length = len(alphabet)

        # Create mapping from characters to indices
        char_to_idx = {char: idx for idx, char in enumerate(alphabet)}

        # Initialize one-hot matrix
        one_hot = np.zeros((alphabet_length, seq_length))

        # Fill the matrix
        for pos, char in enumerate(sequence):
            one_hot[char_to_idx[char], pos] = 1

        if flatten:
            return one_hot.flatten()
        return one_hot

    return (one_hot_encode,)


@app.cell
def _(
    amino_acids,
    dna_sequences,
    dna_values,
    np,
    nucleotides,
    one_hot_encode,
    pd,
    protein_sequences,
    protein_values,
):
    from sklearn.model_selection import train_test_split

    # One-hot encode sequences
    protein_encoded = np.array(
        [one_hot_encode(seq, amino_acids, flatten=True) for seq in protein_sequences]
    )
    dna_encoded = np.array(
        [one_hot_encode(seq, nucleotides, flatten=True) for seq in dna_sequences]
    )

    # Create new dataframes with encoded sequences
    protein_encoded_df = pd.DataFrame(protein_encoded, index=protein_sequences)
    protein_encoded_df["function"] = protein_values

    dna_encoded_df = pd.DataFrame(dna_encoded, index=dna_sequences)
    dna_encoded_df["function"] = dna_values

    # Split data into train (60%), test (20%), and heldout (20%) sets
    train_size = 0.6
    test_size = 0.2
    random_state = 42

    # Protein data splits
    protein_train, protein_temp = train_test_split(
        protein_encoded_df, train_size=train_size, random_state=random_state
    )
    protein_test, protein_heldout = train_test_split(
        protein_temp, test_size=0.5, random_state=random_state
    )

    # DNA data splits
    dna_train, dna_temp = train_test_split(
        dna_encoded_df, train_size=train_size, random_state=random_state
    )
    dna_test, dna_heldout = train_test_split(
        dna_temp, test_size=0.5, random_state=random_state
    )
    return (
        dna_encoded,
        dna_encoded_df,
        dna_heldout,
        dna_temp,
        dna_test,
        dna_train,
        protein_encoded,
        protein_encoded_df,
        protein_heldout,
        protein_temp,
        protein_test,
        protein_train,
        random_state,
        test_size,
        train_size,
        train_test_split,
    )


@app.cell
def _(protein_train):
    from pycaret.regression import setup

    s = setup(protein_train, target="function", session_id=123)
    return s, setup


@app.cell
def _():
    from pycaret.regression import compare_models

    best = compare_models()
    return best, compare_models


@app.cell
def _(best):
    from pycaret.regression import evaluate_model, plot_model

    plot_model(best, plot="residuals")
    return evaluate_model, plot_model


@app.cell
def _(best, plot_model):
    plot_model(best, plot="error")
    return


@app.cell
def _(best):
    from pycaret.regression import predict_model

    predict_model(best)
    return (predict_model,)


@app.cell
def _(best, predict_model, protein_heldout):
    predictions = predict_model(best, data=protein_heldout)
    predictions["function"].sort_values(ascending=False)

    return (predictions,)


if __name__ == "__main__":
    app.run()
