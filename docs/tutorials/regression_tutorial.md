# Sequence Regression with Fast-SeqFunc

This tutorial demonstrates how to use `fast-seqfunc` for regression problems, where you want to predict continuous values from biological sequences.

## Overview

In sequence regression, we want to learn to predict a continuous value (e.g., binding affinity, enzyme efficiency, or protein stability) from a biological sequence (DNA, RNA, or protein). This tutorial will walk you through:

1. Setting up your environment
2. Preparing sequence-function data
3. Training a regression model
4. Evaluating model performance
5. Making predictions on new sequences
6. Visualizing results

## Prerequisites

- Python 3.11 or higher
- The following packages:
  ```bash
  pip install fast-seqfunc pandas numpy matplotlib seaborn scikit-learn loguru
  ```

## Setup

First, let's import all necessary packages:

```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from fast_seqfunc import train_model, predict, save_model, load_model
from loguru import logger
```

## Working with Sequence-Function Data

Sequence-function data typically consists of biological sequences paired with measurements of a functional property. For this tutorial, we'll create synthetic data:

```python
from fast_seqfunc import generate_dataset_by_task

# Generate a GC content dataset as an example
# (where the function is simply the GC content of DNA sequences)
data = generate_dataset_by_task(
    task="gc_content",
    count=1000,  # Number of sequences to generate
    length=50,   # Sequence length
    noise_level=0.1,  # Add some noise to make the task more realistic
)

# Examine the data
print(data.head())
print(f"Data shape: {data.shape}")
print(f"Target distribution: min={data['function'].min():.3f}, "
      f"max={data['function'].max():.3f}, "
      f"mean={data['function'].mean():.3f}")
```

### Preparing Your Own Data

If you have your own data, it should be structured in a DataFrame with at least two columns:
- A column containing the sequences (e.g., "sequence")
- A column containing the target values (e.g., "function")

For example:
```python
# Load your own data
# data = pd.read_csv("your_sequence_function_data.csv")
```

## Splitting Data for Training and Testing

It's important to evaluate your model on data it hasn't seen during training:

```python
# Split into train and test sets (80/20 split)
train_size = int(0.8 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test samples")

# Create output directory for results
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
```

## Training a Regression Model

Now we can train a regression model using `fast-seqfunc`:

```python
# Train a regression model
logger.info("Training regression model...")
model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",  # Column containing sequences
    target_col="function",    # Column containing target values
    embedding_method="one-hot",  # Method to convert sequences to numerical features
    model_type="regression",     # Specify regression task
    optimization_metric="r2",    # Metric to optimize (r2, rmse, mae)
)

# Display test results
if model_info.get("test_results"):
    logger.info("Test metrics from training:")
    for metric, value in model_info["test_results"].items():
        logger.info(f"  {metric}: {value:.4f}")

# Save the model for later use
model_path = output_dir / "regression_model.pkl"
save_model(model_info, model_path)
logger.info(f"Model saved to {model_path}")
```

### Understanding Embedding Methods

The `embedding_method` parameter determines how sequences are converted to numerical features:

- `"one-hot"`: Each position in the sequence is encoded as a one-hot vector indicating which amino acid or nucleotide is present.

Future versions of `fast-seqfunc` will include more advanced embedding methods such as ESM2 for proteins and CARP for nucleic acids.

## Making Predictions

After training, you can use your model to make predictions on new sequences:

```python
# Generate some new data for prediction
new_data = generate_dataset_by_task(
    task="gc_content",
    count=200,
    length=50,
)

# Make predictions
predictions = predict(model_info, new_data["sequence"])

# Create results DataFrame
results_df = new_data.copy()
results_df["predicted"] = predictions
results_df.to_csv(output_dir / "regression_predictions.csv", index=False)

print(results_df.head())
```

## Evaluating Regression Performance

Let's evaluate our model more thoroughly:

```python
# Calculate regression metrics
true_values = test_data["function"]
predicted_values = predict(model_info, test_data["sequence"])

# Calculate metrics
mse = mean_squared_error(true_values, predicted_values)
rmse = np.sqrt(mse)
r2 = r2_score(true_values, predicted_values)
mae = mean_absolute_error(true_values, predicted_values)

# Print metrics
print(f"Test Set Performance:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"  MAE:  {mae:.4f}")
```

## Visualizing Results

Visualizations help in understanding model performance:

```python
# Create a prediction vs. actual scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(true_values, predicted_values, alpha=0.6)
plt.plot([true_values.min(), true_values.max()],
         [true_values.min(), true_values.max()],
         'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "regression_scatter_plot.png", dpi=300)

# Plot residuals
residuals = true_values - predicted_values
plt.figure(figsize=(10, 8))
plt.scatter(predicted_values, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "regression_residuals.png", dpi=300)

# Plot distribution of residuals
plt.figure(figsize=(10, 8))
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "regression_residuals_distribution.png", dpi=300)

logger.info("Visualizations saved to output directory")
```

## Working with Different Sequence Types

`fast-seqfunc` automatically detects and handles different types of biological sequences:

- DNA (containing A, C, G, T)
- RNA (containing A, C, G, U)
- Proteins (containing amino acid letters)

```python
# Example with protein sequences
from fast_seqfunc import generate_random_sequences

# Generate random protein sequences
protein_sequences = generate_random_sequences(
    length=30,
    count=100,
    alphabet="ACDEFGHIKLMNPQRSTVWY",  # Protein alphabet
    fixed_length=True,
)

# Create a dummy function (e.g., number of hydrophobic residues)
hydrophobic = "AVILMFYW"
function_values = [
    sum(seq.count(aa) for aa in hydrophobic) / len(seq)
    for seq in protein_sequences
]

# Create dataset
protein_data = pd.DataFrame({
    "sequence": protein_sequences,
    "function": function_values
})

# Now you could train a model on this protein data using the same workflow
```

## Advanced Model Training Options

`fast-seqfunc` uses PyCaret behind the scenes, which allows for customization:

```python
# Example with more options
advanced_model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    model_type="regression",
    optimization_metric="r2",
    # Additional PyCaret setup options:
    n_jobs=-1,  # Use all available CPU cores
    fold=5,     # 5-fold cross-validation
    normalize=True,  # Normalize features
    polynomial_features=True,  # Generate polynomial features
    feature_selection=True,  # Perform feature selection
)
```

## Loading and Reusing Models

You can load saved models for reuse:

```python
# Load a previously saved model
loaded_model_info = load_model(model_path)

# Use the loaded model for predictions
new_predictions = predict(loaded_model_info, new_data["sequence"])

# Verify the predictions match those from the original model
np.allclose(predictions, new_predictions)
```

## Conclusion

You've now learned how to:
1. Prepare sequence-function data
2. Train a regression model using `fast-seqfunc`
3. Make predictions on new sequences
4. Evaluate model performance
5. Visualize results

For more advanced features and applications, check out the [API reference](../api_reference.md) and [additional tutorials](./classification_tutorial.md).

## Next Steps

- Try different regression tasks (e.g., "motif_position", "interaction")
- Experiment with different model parameters
- Apply these techniques to your own sequence-function data
