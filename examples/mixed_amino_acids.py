"""Example demonstrating sequence-function modeling with mixed amino acids.

This script shows how to use fast-seqfunc to model sequences that represent
mixtures of natural and synthetic amino acids, encoded as integers.

In this example, we:
1. Generate synthetic data with integer-encoded amino acids
2. Define a custom alphabet for the mixed amino acid set
3. Train a model on this data
4. Make predictions on new sequences
5. Save and load the trained model
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fast-seqfunc>=0.1.0",
#   "matplotlib>=3.7.0",
#   "scikit-learn>=1.0.0",
#   "pandas>=2.0.0",
#   "numpy>=1.24.0",
#   "loguru>=0.6.0",
# ]
# ///

import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from fast_seqfunc.alphabets import Alphabet
from fast_seqfunc.core import predict, train_model
from fast_seqfunc.embedders import OneHotEmbedder
from fast_seqfunc.synthetic import generate_integer_function_data


def main():
    """Run the mixed amino acid example."""
    logger.info("Starting mixed amino acid example")

    # Create output directory if it doesn't exist
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mixed_amino_acid_model.pkl"

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # ------------------------------------------------------------------------
    # 1. Generate synthetic data with integer-encoded sequences
    # ------------------------------------------------------------------------
    logger.info("Generating synthetic data")

    # In this scenario, integers 0-19 represent the 20 standard amino acids
    # and integers 20-25 represent 6 synthetic amino acids

    # Generate synthetic data for regression
    data = generate_integer_function_data(
        count=500,  # Generate 500 sequences
        sequence_length=10,  # Each sequence has 10 amino acids
        max_value=25,  # Integers 0-25 (20 natural + 6 synthetic amino acids)
        function_type="nonlinear",  # Use a nonlinear function for the relationship
        noise_level=0.2,  # Add some noise to make it realistic
        classification=False,  # Regression problem
        position_weights=[
            1.5,
            1.2,
            1.0,
            0.8,
            0.6,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
        ],  # Position-specific weights
    )

    logger.info(f"Generated {len(data)} sequences")
    logger.info(f"First few sequences:\n{data.head()}")

    # Plot the distribution of function values
    plt.figure(figsize=(10, 6))
    plt.hist(data["function"], bins=30)
    plt.title("Distribution of Function Values")
    plt.xlabel("Function Value")
    plt.ylabel("Count")
    plt.savefig(output_dir / "function_distribution.png")

    # ------------------------------------------------------------------------
    # 2. Demonstrate creating a custom alphabet for mixed amino acids
    # ------------------------------------------------------------------------

    # This maps integers to a representation of amino acids
    # Standard amino acids are "A" through "Y"
    # Synthetic amino acids are "Z1" through "Z6"
    aa_names = {
        "0": "A",
        "1": "C",
        "2": "D",
        "3": "E",
        "4": "F",
        "5": "G",
        "6": "H",
        "7": "I",
        "8": "K",
        "9": "L",
        "10": "M",
        "11": "N",
        "12": "P",
        "13": "Q",
        "14": "R",
        "15": "S",
        "16": "T",
        "17": "V",
        "18": "W",
        "19": "Y",
        "20": "Z1",
        "21": "Z2",
        "22": "Z3",
        "23": "Z4",
        "24": "Z5",
        "25": "Z6",
        "-1": "-",  # Gap character
    }

    # Create a custom alphabet with meaningful names
    # First, let's create a function to convert the integer sequences to named sequences
    def convert_to_names(int_sequence: str) -> str:
        """Convert comma-delimited integer sequence to amino acid names."""
        int_tokens = int_sequence.split(",")
        return ",".join(aa_names.get(token, "?") for token in int_tokens)

    # Create a new column with the amino acid names
    data["aa_sequence"] = data["sequence"].apply(convert_to_names)

    logger.info("Added amino acid name representations:")
    logger.info(f"Integer sequence: {data.iloc[0]['sequence']}")
    logger.info(f"AA sequence: {data.iloc[0]['aa_sequence']}")

    # ------------------------------------------------------------------------
    # 3. Train a model using PyCaret's automated ML
    # ------------------------------------------------------------------------

    # Split data into train and test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    logger.info(f"Training data: {len(train_data)} sequences")
    logger.info(f"Test data: {len(test_data)} sequences")

    # Option 1: Use fast-seqfunc's train_model function with PyCaret
    # This automatically handles embedding and model selection
    try:
        logger.info("Training model with PyCaret")
        model_info = train_model(
            train_data=train_data,
            test_data=test_data,
            sequence_col="sequence",
            target_col="function",
            embedding_method="one-hot",
            model_type="regression",
        )

        # Make predictions on test data
        test_sequences = test_data["sequence"].tolist()
        predictions = predict(model_info, test_sequences)

        # Evaluate the model
        mse = mean_squared_error(test_data["function"], predictions)
        r2 = r2_score(test_data["function"], predictions)

        logger.info(f"PyCaret model performance - MSE: {mse:.4f}, R²: {r2:.4f}")

        # Save the trained model
        with open(model_path, "wb") as f:
            pickle.dump(model_info, f)
        logger.info(f"Saved model to {model_path}")

    except Exception as e:
        logger.warning(f"PyCaret training failed: {str(e)}")
        logger.warning("Falling back to manual model training with scikit-learn")
        model_info = None

    # ------------------------------------------------------------------------
    # 4. Alternative: Manual model training with scikit-learn
    # ------------------------------------------------------------------------
    # This approach gives more control over the embedding and model training

    # Create a custom alphabet for the integer-encoded sequences
    alphabet = Alphabet.integer(max_value=25)

    # Initialize and fit the embedder
    embedder = OneHotEmbedder(alphabet=alphabet)

    # Transform the sequences to one-hot encodings
    X_train = embedder.fit_transform(train_data["sequence"])
    y_train = train_data["function"].values

    X_test = embedder.transform(test_data["sequence"])
    y_test = test_data["function"].values

    # Train a Ridge regression model
    logger.info("Training Ridge regression model")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Ridge model performance - MSE: {mse:.4f}, R²: {r2:.4f}")

    # Create a scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
    plt.xlabel("Actual Function Values")
    plt.ylabel("Predicted Function Values")
    plt.title("Actual vs Predicted Function Values")
    plt.savefig(output_dir / "prediction_scatter.png")

    # ------------------------------------------------------------------------
    # 5. Demonstrate using the trained model for new sequences
    # ------------------------------------------------------------------------

    # Create some new sequences with a mix of natural and synthetic amino acids
    new_sequences = [
        "0,1,2,3,20,21,22,17,18,19",  # Mix of natural (0-19) and synthetic (20+)
        "20,21,22,23,24,25,20,21,22,23",  # All synthetic
        "0,1,2,3,4,5,6,7,8,9",  # All natural
        "19,18,17,16,25,24,23,2,1,0",  # Mix in reverse order
    ]

    # Convert to amino acid names for display
    new_aa_sequences = [convert_to_names(seq) for seq in new_sequences]

    # Make predictions with the Ridge model
    X_new = embedder.transform(new_sequences)
    new_predictions = model.predict(X_new)

    # Display results
    logger.info("Predictions for new sequences:")
    for i, (seq, aa_seq, pred) in enumerate(
        zip(new_sequences, new_aa_sequences, new_predictions)
    ):
        logger.info(f"Sequence {i + 1}: {seq}")
        logger.info(f"AA Names: {aa_seq}")
        logger.info(f"Predicted value: {pred:.4f}")
        logger.info("-" * 40)

    # ------------------------------------------------------------------------
    # 6. If we have a PyCaret model, load it and make predictions
    # ------------------------------------------------------------------------

    if model_info is not None:
        # Load the saved model
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        # Make predictions with the loaded model
        loaded_predictions = predict(loaded_model, new_sequences)

        logger.info("Predictions from loaded PyCaret model:")
        for i, (seq, pred) in enumerate(zip(new_aa_sequences, loaded_predictions)):
            logger.info(f"Sequence: {seq}")
            logger.info(f"Predicted value: {pred:.4f}")
            logger.info("-" * 40)

    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
