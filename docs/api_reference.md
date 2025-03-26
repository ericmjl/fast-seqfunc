# API Reference

This document provides details on the main functions and classes available in the `fast-seqfunc` package.

## Core Functions

### `train_model`

```python
from fast_seqfunc import train_model

model_info = train_model(
    train_data,
    val_data=None,
    test_data=None,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    model_type="regression",
    optimization_metric=None,
    **kwargs
)
```

Trains a sequence-function model using PyCaret.

**Parameters**:

- `train_data`: DataFrame or path to CSV file with training data.
- `val_data`: Optional validation data (not directly used, reserved for future).
- `test_data`: Optional test data for final evaluation.
- `sequence_col`: Column name containing sequences.
- `target_col`: Column name containing target values.
- `embedding_method`: Method to use for embedding sequences. Currently only "one-hot" is supported.
- `model_type`: Type of modeling problem ("regression" or "classification").
- `optimization_metric`: Metric to optimize during model selection (e.g., "r2", "accuracy", "f1").
- `**kwargs`: Additional arguments passed to PyCaret setup.

**Returns**:

- Dictionary containing the trained model and related metadata.

### `predict`

```python
from fast_seqfunc import predict

predictions = predict(
    model_info,
    sequences,
    sequence_col="sequence"
)
```

Generates predictions for new sequences using a trained model.

**Parameters**:

- `model_info`: Dictionary from `train_model` containing model and related information.
- `sequences`: Sequences to predict (list, Series, or DataFrame).
- `sequence_col`: Column name in DataFrame containing sequences.

**Returns**:

- Array of predictions.

### `save_model`

```python
from fast_seqfunc import save_model

save_model(model_info, path)
```

Saves the model to disk.

**Parameters**:

- `model_info`: Dictionary containing model and related information.
- `path`: Path to save the model.

**Returns**:

- None

### `load_model`

```python
from fast_seqfunc import load_model

model_info = load_model(path)
```

Loads a trained model from disk.

**Parameters**:

- `path`: Path to saved model file.

**Returns**:

- Dictionary containing the model and related information.

## Embedder Classes

### `OneHotEmbedder`

```python
from fast_seqfunc.embedders import OneHotEmbedder

embedder = OneHotEmbedder(sequence_type="auto")
embeddings = embedder.fit_transform(sequences)
```

One-hot encoding for protein or nucleotide sequences.

**Parameters**:

- `sequence_type`: Type of sequences to encode ("protein", "dna", "rna", or "auto").

**Methods**:

- `fit(sequences)`: Determine alphabet and set up the embedder.
- `transform(sequences)`: Transform sequences to one-hot encodings.
- `fit_transform(sequences)`: Fit and transform in one step.

## Helper Functions

### `get_embedder`

```python
from fast_seqfunc.embedders import get_embedder

embedder = get_embedder(method="one-hot")
```

Get an embedder instance based on method name.

**Parameters**:

- `method`: Embedding method (currently only "one-hot" is supported).

**Returns**:

- Configured embedder instance.

### `evaluate_model`

```python
from fast_seqfunc.core import evaluate_model

results = evaluate_model(
    model,
    X_test,
    y_test,
    embedder,
    model_type,
    embed_cols
)
```

Evaluate model performance on test data.

**Parameters**:

- `model`: Trained model.
- `X_test`: Test sequences.
- `y_test`: True target values.
- `embedder`: Embedder to transform sequences.
- `model_type`: Type of model (regression or classification).
- `embed_cols`: Column names for embedded features.

**Returns**:

- Dictionary containing metrics and prediction data with structure:
  ```
  {
     "metrics": {metric_name: value, ...},
     "predictions_data": {
         "y_true": [...],
         "y_pred": [...]
     }
  }
  ```

### `save_detailed_metrics`

```python
from fast_seqfunc.core import save_detailed_metrics

save_detailed_metrics(
    metrics_data,
    output_dir,
    model_type,
    embedding_method="unknown"
)
```

Save detailed model metrics to files in the specified directory.

**Parameters**:

- `metrics_data`: Dictionary containing metrics and prediction data from `evaluate_model`.
- `output_dir`: Directory to save metrics files.
- `model_type`: Type of model (regression or classification).
- `embedding_method`: Embedding method used for this model.

**Returns**:

- None

**Output Files**:

- JSON file with detailed metrics
- CSV file with raw predictions and true values
- Visualization plots based on model type:
  - For regression: scatter plot, residual plot
  - For classification: confusion matrix
