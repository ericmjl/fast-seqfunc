# fast-seqfunc

Painless sequence-function models for proteins and nucleotides.

Made with ❤️ by Eric Ma (@ericmjl).

## Overview

Fast-SeqFunc is a Python package designed for efficient sequence-function modeling for proteins and nucleotide machine learning problems. It provides a simple, high-level API that handles various sequence embedding methods and automates model selection and training.

### Key Features

- **Multiple Embedding Methods**:
  - One-hot encoding
  - CARP (Microsoft's protein-sequence-models)
  - ESM2 (Facebook's ESM)

- **Automated Machine Learning**:
  - Uses PyCaret for model selection and hyperparameter tuning
  - Supports regression and classification tasks
  - Evaluates performance with appropriate metrics

- **Sequence Handling**:
  - Flexible handling of variable-length sequences
  - Configurable padding options for consistent embeddings
  - Custom alphabets support

- **Simple API**:
  - Single function call to train models
  - Handles data loading and preprocessing

- **Command-line Interface**:
  - Train models directly from the command line
  - Make predictions on new sequences
  - Compare different embedding methods

## Installation

### Using pip

```bash
pip install fast-seqfunc
```

### From Source

```bash
git clone git@github.com:ericmjl/fast-seqfunc
cd fast-seqfunc
pixi install
```

## Quick Start

### Python API

```python
from fast_seqfunc import train_model, predict
import pandas as pd

# Load your sequence-function data
train_data = pd.read_csv("train_data.csv")
val_data = pd.read_csv("val_data.csv")

# Train a model
model = train_model(
    train_data=train_data,
    val_data=val_data,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",  # or "carp", "esm2", "auto"
    model_type="regression",     # or "classification"
)

# Make predictions on new sequences
new_data = pd.read_csv("new_sequences.csv")
predictions = predict(model, new_data["sequence"])

# Save the model for later use
model.save("my_model.pkl")
```

### Command-line Interface

Train a model:

```bash
fast-seqfunc train train_data.csv --sequence-col sequence --target-col function --embedding-method one-hot --output-path model.pkl
```

Make predictions:

```bash
fast-seqfunc predict-cmd model.pkl new_sequences.csv --output-path predictions.csv
```

Compare embedding methods:

```bash
fast-seqfunc compare-embeddings train_data.csv --test-data test_data.csv --output-path comparison.csv
```

## Advanced Usage

### Using Multiple Embedding Methods

You can try multiple embedding methods in one run:

```python
model = train_model(
    train_data=train_data,
    embedding_method=["one-hot", "carp", "esm2"],
)
```

### Custom Metrics for Optimization

Specify metrics to optimize during model selection:

```python
model = train_model(
    train_data=train_data,
    model_type="regression",
    optimization_metric="r2"  # or "rmse", "mae", etc.
)
```

### Getting Confidence Estimates

```python
predictions, confidence = predict(
    model,
    sequences,
    return_confidence=True
)
```

### Handling Variable Length Sequences

Fast-SeqFunc handles variable length sequences with configurable padding:

```python
# Default behavior pads all sequences to the max length with "-"
model = train_model(
    train_data=train_data,
    embedding_method="one-hot",
    embedder_kwargs={"pad_sequences": True, "gap_character": "-"}
)

# Disable padding for sequences of different lengths
model = train_model(
    train_data=train_data,
    embedding_method="one-hot",
    embedder_kwargs={"pad_sequences": False}
)

# Set a fixed maximum length and custom gap character
model = train_model(
    train_data=train_data,
    embedding_method="one-hot",
    embedder_kwargs={"max_length": 100, "gap_character": "X"}
)
```

For a complete example, see `examples/variable_length_sequences.py`.

## Documentation

For full documentation, visit [https://ericmjl.github.io/fast-seqfunc/](https://ericmjl.github.io/fast-seqfunc/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
