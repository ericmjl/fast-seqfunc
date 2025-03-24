# Fast-SeqFunc Quickstart

This guide demonstrates how to use `fast-seqfunc` for training sequence-function models and making predictions with your own sequence data.

## Prerequisites

- Python 3.11 or higher
- The following packages:
  - `fast-seqfunc`
  - `pandas`
  - `numpy`
  - `matplotlib` and `seaborn` (for visualization)
  - `pycaret[full]>=3.0.0`
  - `scikit-learn>=1.0.0`

## Setup

Start by importing the necessary modules:

```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fast_seqfunc import train_model, predict, save_model, load_model
```

## Data Preparation

For this tutorial, we assume you already have a sequence-function dataset with the following format:

```
sequence,function
ACGTACGT...,0.75
TACGTACG...,0.63
...
```

Let's load and split our data:

```python
# Load your sequence-function data
data = pd.read_csv("your_data.csv")

# Split into train and test sets (80/20 split)
train_size = int(0.8 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

print(f"Data split: {len(train_data)} train, {len(test_data)} test samples")

# Create directory for outputs
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
```

## Training a Model

With `fast-seqfunc`, you can train a model with just a few lines of code:

```python
# Train and compare multiple models automatically
model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",  # Column containing sequences
    target_col="function",    # Column containing function values
    embedding_method="one-hot",  # Method to encode sequences
    model_type="regression",     # For predicting continuous values
    optimization_metric="r2",    # Optimize for R-squared
)

# Display test results if available
if model_info.get("test_results"):
    print("\nTest metrics from training:")
    for metric, value in model_info["test_results"].items():
        print(f"  {metric}: {value:.4f}")
```

## Saving and Loading Models

You can easily save your trained model for later use:

```python
# Save the model
model_path = output_dir / "model.pkl"
save_model(model_info, model_path)
print(f"Model saved to {model_path}")

# Later, you can load the model
loaded_model = load_model(model_path)
```

## Making Predictions

Making predictions on new sequences is straightforward:

```python
# Make predictions on test data
predictions = predict(model_info, test_data["sequence"])

# Create a results DataFrame
results_df = test_data.copy()
results_df["prediction"] = predictions
results_df.to_csv(output_dir / "predictions.csv", index=False)
```

## Evaluating Model Performance

You can evaluate how well your model performs:

```python
# Calculate metrics manually
true_values = test_data["function"]
mse = ((predictions - true_values) ** 2).mean()
r2 = 1 - ((predictions - true_values) ** 2).sum() / ((true_values - true_values.mean()) ** 2).sum()

print("Model performance:")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  R²: {r2:.4f}")
```

## Visualizing Results

Visualizing the model's performance can provide insights:

```python
# Create a scatter plot of true vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=true_values, y=predictions, alpha=0.6)
plt.plot(
    [min(true_values), max(true_values)],
    [min(true_values), max(true_values)],
    "r--"  # Add a diagonal line
)
plt.xlabel("True Function Value")
plt.ylabel("Predicted Function Value")
plt.title("True vs Predicted Function Values")
plt.tight_layout()
plt.savefig(output_dir / "true_vs_predicted.png", dpi=300)
```

## Next Steps

After mastering the basics, you can:

1. Try different embedding methods (currently only `one-hot` is supported, with more coming soon)
2. Experiment with classification problems by setting `model_type="classification"`
3. Optimize for different metrics by changing the `optimization_metric` parameter
4. Explore the internal model structure and customize it for your specific needs

For more details, check out the [API documentation](api_reference.md).
