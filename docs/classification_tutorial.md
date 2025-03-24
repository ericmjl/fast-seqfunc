# Sequence Classification with Fast-SeqFunc

This tutorial demonstrates how to use `fast-seqfunc` for classification problems, where you want to predict discrete categories from biological sequences.

## Prerequisites

- Python 3.11 or higher
- The following packages:
  - `fast-seqfunc`
  - `pandas`
  - `numpy`
  - `matplotlib` and `seaborn` (for visualization)
  - `scikit-learn`
  - `pycaret[full]>=3.0.0`

## Setup

Import the necessary modules:

```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from fast_seqfunc import train_model, predict, save_model, load_model
from loguru import logger
```

## Working with Classification Data

For this tutorial, we assume you have a dataset with sequences and corresponding class labels:

```
sequence,class
ACGTACGT...,0
TACGTACG...,1
...
```

Where the class column contains categorical values (0, 1, 2, etc. or text labels).

```python
# Load your sequence-classification data
data = pd.read_csv("your_classification_data.csv")

# If classes are text labels, you might want to convert them to integers
# data['class'] = data['class'].astype('category').cat.codes

# Split into train and test sets (80/20 split)
train_size = int(0.8 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test samples")
logger.info(f"Class distribution in training data:\n{train_data['class'].value_counts()}")

# Create output directory
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
```

## Training a Classification Model

For classification tasks, we need to specify `model_type="classification"`:

```python
# Train a classification model
logger.info("Training classification model...")
model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",
    target_col="class",
    embedding_method="one-hot",
    model_type="classification",
    optimization_metric="accuracy",  # Could also use 'f1', 'auc', etc.
)

# Display test results
if model_info.get("test_results"):
    logger.info("Test metrics from training:")
    for metric, value in model_info["test_results"].items():
        logger.info(f"  {metric}: {value:.4f}")

# Save the model
model_path = output_dir / "classification_model.pkl"
save_model(model_info, model_path)
logger.info(f"Model saved to {model_path}")
```

## Making Predictions

Making predictions works the same way as with regression:

```python
# Predict on test data
predictions = predict(model_info, test_data["sequence"])

# Create results DataFrame
results_df = test_data.copy()
results_df["predicted_class"] = predictions
results_df.to_csv(output_dir / "classification_predictions.csv", index=False)
```

## Evaluating Classification Performance

For classification tasks, we can use different evaluation metrics:

```python
# Calculate classification metrics
true_values = test_data["class"]
predicted_values = predictions

# Print classification report
print("\nClassification Report:")
print(classification_report(true_values, predicted_values))

# Create confusion matrix
cm = confusion_matrix(true_values, predicted_values)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(data["class"].unique()),
            yticklabels=sorted(data["class"].unique()))
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
```

## Visualizing Class Distributions

For sequence classification, it can be useful to visualize sequence properties by class:

```python
# Example: calculate sequence length by class
data["seq_length"] = data["sequence"].str.len()

plt.figure(figsize=(10, 6))
sns.boxplot(x="class", y="seq_length", data=data)
plt.title("Sequence Length Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Sequence Length")
plt.tight_layout()
plt.savefig(output_dir / "seq_length_by_class.png", dpi=300)

# Example: nucleotide composition by class (for DNA/RNA)
if any(nuc in data["sequence"].iloc[0].upper() for nuc in "ACGT"):
    data["A_percent"] = data["sequence"].apply(lambda x: x.upper().count("A") / len(x) * 100)
    data["C_percent"] = data["sequence"].apply(lambda x: x.upper().count("C") / len(x) * 100)
    data["G_percent"] = data["sequence"].apply(lambda x: x.upper().count("G") / len(x) * 100)
    data["T_percent"] = data["sequence"].apply(lambda x: x.upper().count("T") / len(x) * 100)

    # Melt the data for easier plotting
    plot_data = pd.melt(
        data,
        id_vars=["class"],
        value_vars=["A_percent", "C_percent", "G_percent", "T_percent"],
        var_name="Nucleotide",
        value_name="Percentage"
    )

    # Plot nucleotide composition by class
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="class", y="Percentage", hue="Nucleotide", data=plot_data)
    plt.title("Nucleotide Composition by Class")
    plt.xlabel("Class")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "nucleotide_composition_by_class.png", dpi=300)
```

## Working with Multi-Class Problems

If you have more than two classes, the process is the same, but you might want to adjust some metrics:

```python
# For multi-class problems, you might want to:
# 1. Use a different optimization metric
multi_class_model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",
    target_col="class",
    embedding_method="one-hot",
    model_type="multi-class",  # Specify multi-class explicitly
    optimization_metric="f1",  # F1 with 'weighted' average is good for imbalanced classes
)

# 2. Visualize per-class performance
# Create a heatmap of the confusion matrix with normalization
def plot_normalized_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Normalized confusion matrix saved to {output_path}")

# Use the function
class_labels = sorted(data["class"].unique())
plot_normalized_confusion_matrix(
    true_values,
    predictions,
    class_labels,
    output_dir / "normalized_confusion_matrix.png"
)
```

## Next Steps

After mastering sequence classification, you can:

1. Experiment with different model types in PyCaret
2. Try different embedding methods as they become available
3. Work with protein sequences for function classification
4. Apply the model to predict classes for new, unlabeled sequences

For more details on the API, check out the [API reference](api_reference.md).
