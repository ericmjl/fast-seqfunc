# Sequence Classification with Fast-SeqFunc

This tutorial demonstrates how to use `fast-seqfunc` for classification problems, where you want to predict discrete categories from biological sequences.

## Overview

In sequence classification, we want to learn to predict discrete categories (e.g., protein function, gene families, or binding/non-binding sequences) from biological sequences. This tutorial will walk you through:

1. Setting up your environment
2. Preparing sequence classification data
3. Training binary and multi-class classification models
4. Evaluating model performance
5. Making predictions on new sequences
6. Visualizing classification results

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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from fast_seqfunc import train_model, predict, save_model, load_model
from loguru import logger
```

## Working with Classification Data

For classification tasks, each sequence is associated with a discrete class label. Let's create synthetic data for this tutorial:

```python
from fast_seqfunc import generate_dataset_by_task

# Generate a binary classification dataset
# (sequences with or without specific patterns)
binary_data = generate_dataset_by_task(
    task="classification",
    count=1000,  # Number of sequences to generate
    length=30,   # Sequence length
    noise_level=0.1,  # Add some noise to make the task more realistic
)

# Generate a multi-class classification dataset
multi_data = generate_dataset_by_task(
    task="multiclass",
    count=1000,  # Number of sequences to generate
    length=30,   # Sequence length
    noise_level=0.1,  # Add some noise
)

# Examine the data
print("Binary Classification Dataset:")
print(binary_data.head())
print(f"Binary class distribution:\n{binary_data['function'].value_counts()}")

print("\nMulti-class Classification Dataset:")
print(multi_data.head())
print(f"Multi-class distribution:\n{multi_data['function'].value_counts()}")
```

### Preparing Your Own Data

If you have your own data, it should be structured in a DataFrame with at least two columns:
- A column containing the sequences (e.g., "sequence")
- A column containing the class labels (e.g., "class" or "function")

For example:
```python
# Load your own data
# data = pd.read_csv("your_classification_data.csv")

# If your classes are text labels, you might want to convert them to integers
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# data['class_encoded'] = label_encoder.fit_transform(data['class'])
```

## Binary Classification Example

Let's start with a binary classification problem:

```python
# For this tutorial, we'll use our binary dataset
data = binary_data

# Split into train and test sets (80/20 split)
train_size = int(0.8 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test samples")
logger.info(f"Class distribution in training data:\n{train_data['function'].value_counts()}")

# Create output directory for results
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
```

### Training a Binary Classification Model

Now we can train a classification model:

```python
# Train a classification model
logger.info("Training binary classification model...")
model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",    # Column containing sequences
    target_col="function",      # Column containing class labels
    embedding_method="one-hot", # Method to convert sequences to numerical features
    model_type="classification", # Specify classification task
    optimization_metric="auc",  # Metric to optimize (auc, accuracy, f1, etc.)
)

# Display test results
if model_info.get("test_results"):
    logger.info("Test metrics from training:")
    for metric, value in model_info["test_results"].items():
        logger.info(f"  {metric}: {value:.4f}")

# Save the model for later use
model_path = output_dir / "binary_classification_model.pkl"
save_model(model_info, model_path)
logger.info(f"Model saved to {model_path}")
```

### Making Predictions

With our trained model, we can now make predictions:

```python
# Generate some new data for prediction
new_data = generate_dataset_by_task(
    task="classification",
    count=200,
    length=30,
)

# Make predictions
predictions = predict(model_info, new_data["sequence"])

# Create results DataFrame
results_df = new_data.copy()
results_df["predicted_class"] = predictions
results_df.to_csv(output_dir / "binary_classification_predictions.csv", index=False)

print(results_df.head())
```

### Evaluating Binary Classification Performance

Let's evaluate our model more thoroughly:

```python
# Calculate classification metrics
true_values = test_data["function"]
predicted_values = predict(model_info, test_data["sequence"])

# Print classification report
print("\nClassification Report:")
print(classification_report(true_values, predicted_values))

# Create confusion matrix
cm = confusion_matrix(true_values, predicted_values)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "binary_confusion_matrix.png", dpi=300)

# Try to get probability estimates (if model supports it)
try:
    # For this tutorial, we'll use a simplified approach since confidence scores are not available
    # In a real implementation, you'd need access to the raw model's predict_proba method

    logger.warning("ROC and PR curves require probability estimates which are not supported in this version")
    logger.warning("We're skipping these visualizations in this tutorial")

    """
    # Below is example code that would work if your model provides probability estimates:

    # Get class probabilities (if available)
    # y_prob = model.predict_proba(X_test)[:, 1]  # probability of positive class

    # Plot ROC curve
    # fpr, tpr, _ = roc_curve(true_values, y_prob)
    # roc_auc = auc(fpr, tpr)

    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")
    # plt.savefig(output_dir / "binary_roc_curve.png", dpi=300)
    """

except Exception as e:
    logger.warning(f"Could not generate ROC/PR curves: {e}")
    logger.warning("This is normal if the model doesn't support probability output")
```

## Multi-Class Classification Example

Now let's work with a multi-class problem:

```python
# Switch to multi-class data
data = multi_data

# Split into train and test sets (80/20 split)
train_size = int(0.8 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

logger.info(f"Multi-class data split: {len(train_data)} train, {len(test_data)} test samples")
logger.info(f"Class distribution in training data:\n{train_data['function'].value_counts()}")
```

### Training a Multi-Class Model

Training a multi-class model is very similar to binary classification:

```python
# Train a multi-class classification model
logger.info("Training multi-class classification model...")
multi_model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    model_type="classification",  # Can also use "multi-class" explicitly
    optimization_metric="f1",     # F1 with 'weighted' average is good for imbalanced classes
)

# Display test results
if multi_model_info.get("test_results"):
    logger.info("Multi-class test metrics from training:")
    for metric, value in multi_model_info["test_results"].items():
        logger.info(f"  {metric}: {value:.4f}")

# Save the model
multi_model_path = output_dir / "multiclass_model.pkl"
save_model(multi_model_info, multi_model_path)
logger.info(f"Multi-class model saved to {multi_model_path}")
```

### Evaluating Multi-Class Performance

Evaluation for multi-class problems:

```python
# Calculate multi-class metrics
multi_true_values = test_data["function"]
multi_predicted_values = predict(multi_model_info, test_data["sequence"])

# Print classification report
print("\nMulti-class Classification Report:")
print(classification_report(multi_true_values, multi_predicted_values))

# Create confusion matrix
class_labels = sorted(data["function"].unique())
multi_cm = confusion_matrix(multi_true_values, multi_predicted_values)

plt.figure(figsize=(10, 8))
sns.heatmap(multi_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Multi-class Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "multiclass_confusion_matrix.png", dpi=300)

# Create a normalized confusion matrix for better visualization
# with unbalanced classes
multi_cm_normalized = multi_cm.astype('float') / multi_cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(multi_cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Normalized Multi-class Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "multiclass_normalized_confusion_matrix.png", dpi=300)
```

## Visualizing Sequence Features by Class

For classification tasks, it can be useful to visualize sequence properties by class:

```python
# Calculate sequence length by class
data["seq_length"] = data["sequence"].str.len()

plt.figure(figsize=(10, 6))
sns.boxplot(x="function", y="seq_length", data=data)
plt.title("Sequence Length Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Sequence Length")
plt.tight_layout()
plt.savefig(output_dir / "seq_length_by_class.png", dpi=300)

# For DNA/RNA sequences, calculate nucleotide composition by class
if any(nuc in data["sequence"].iloc[0].upper() for nuc in "ACGT"):
    data["A_percent"] = data["sequence"].apply(lambda x: x.upper().count("A") / len(x) * 100)
    data["C_percent"] = data["sequence"].apply(lambda x: x.upper().count("C") / len(x) * 100)
    data["G_percent"] = data["sequence"].apply(lambda x: x.upper().count("G") / len(x) * 100)
    data["T_percent"] = data["sequence"].apply(lambda x: x.upper().count("T") / len(x) * 100)

    # Melt the data for easier plotting
    plot_data = pd.melt(
        data,
        id_vars=["function"],
        value_vars=["A_percent", "C_percent", "G_percent", "T_percent"],
        var_name="Nucleotide",
        value_name="Percentage"
    )

    # Plot nucleotide composition by class
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="function", y="Percentage", hue="Nucleotide", data=plot_data)
    plt.title("Nucleotide Composition by Class")
    plt.xlabel("Class")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "nucleotide_composition_by_class.png", dpi=300)
```

## Working with Imbalanced Classes

When dealing with imbalanced class distributions (where one class is much more frequent than others), you can use special techniques:

```python
# Example: Create an imbalanced dataset
from sklearn.utils import resample

# Assume class 0 is much more frequent than class 1
class_0 = binary_data[binary_data['function'] == 0]
class_1 = binary_data[binary_data['function'] == 1]

# Downsample class 0 to match class 1
class_0_downsampled = resample(
    class_0,
    replace=False,  # Don't sample with replacement
    n_samples=len(class_1),  # Match minority class
    random_state=42  # For reproducibility
)

# Combine the downsampled majority class with the minority class
balanced_data = pd.concat([class_0_downsampled, class_1])

# Now you can train on this balanced dataset
print(f"Original class distribution: {binary_data['function'].value_counts()}")
print(f"Balanced class distribution: {balanced_data['function'].value_counts()}")
```

Alternatively, you can use class weights in PyCaret:

```python
# Train with class weights (handled automatically by PyCaret)
weighted_model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    model_type="classification",
    optimization_metric="f1",  # F1 is good for imbalanced classes
    # Additional PyCaret settings for imbalanced data:
    fix_imbalance=True,  # Automatically fix class imbalance
)
```

## Loading and Using a Classification Model

You can load a saved model and use it for predictions:

```python
# Load a previously saved classification model
loaded_model_info = load_model(model_path)

# Use the model to classify new sequences
sequences_to_classify = [
    "ACGTACGTACGTACGTACGTACGTACGTAC",
    "GATAGATAGATAGATAGATAGATAGATA",
    "CTACCTACCTACCTACCTACCTACCTAC"
]

# Make predictions
predictions = predict(loaded_model_info, sequences_to_classify)

# Print results
for seq, pred in zip(sequences_to_classify, predictions):
    print(f"Sequence (first 10 chars): {seq[:10]}... | Predicted class: {pred}")
```

## Advanced Model Training Options

`fast-seqfunc` uses PyCaret behind the scenes, allowing customization:

```python
# Example with more options
advanced_model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    model_type="classification",
    optimization_metric="f1",
    # Additional PyCaret setup options:
    n_jobs=-1,  # Use all available CPU cores
    fold=5,     # 5-fold cross-validation
    normalize=True,  # Normalize features
    feature_selection=True,  # Perform feature selection
    # Classification-specific options:
    fix_imbalance=True,  # For imbalanced datasets
    remove_outliers=True,  # Remove outliers
)
```

## Conclusion

You've now learned how to:
1. Prepare sequence classification data
2. Train binary and multi-class classification models
3. Evaluate classification performance
4. Make predictions on new sequences
5. Handle special cases like imbalanced classes

For more advanced features and applications, check out the [API reference](../api_reference.md) and [additional tutorials](./regression_tutorial.md).

## Next Steps

- Try different classification tasks (e.g., protein function prediction)
- Experiment with different model types and parameters
- Apply these techniques to your own sequence classification data
