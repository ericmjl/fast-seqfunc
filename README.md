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
# All outputs (model, metrics, cache) will be saved to the 'outputs' directory
fast-seqfunc train train_data.csv --sequence-col sequence --target-col function --embedding-method one-hot --output-dir outputs
```

Make predictions:

```bash
# All prediction outputs will be saved to the 'prediction_outputs' directory
fast-seqfunc predict-cmd outputs/model.pkl new_sequences.csv --output-dir prediction_outputs
```

Compare embedding methods:

```bash
# All outputs (comparison results, metrics, models, cache) will be saved to the 'comparison_outputs' directory
fast-seqfunc compare-embeddings train_data.csv --test-data test_data.csv --output-dir comparison_outputs
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

### Detailed Performance Metrics and Visualizations

The output directories from CLI commands contain comprehensive model performance metrics and visualizations:

```
outputs/                          # Main output directory
├── model.pkl                     # Saved model
├── summary.json                  # Summary of output locations and parameters
├── metrics/                      # Performance metrics and visualizations
│   ├── one-hot_metrics.json      # Detailed metrics in JSON format
│   ├── one-hot_predictions.csv   # Raw predictions and true values
│   ├── one-hot_scatter_plot.png  # Visualization plots
│   ├── one-hot_residual_plot.png
│   └── ...
└── cache/                        # Cached embeddings
```

For predictions:

```
prediction_outputs/               # Prediction output directory
├── predictions.csv               # Saved predictions
├── predictions_histogram.png     # Histogram of prediction values (for regression)
└── prediction_summary.json       # Summary of prediction parameters
```

When comparing embedding methods, a similar structure is created:

```
comparison_outputs/
├── embedding_comparison.csv      # Table comparing all methods
├── embedding_comparison_plot.png # Bar chart comparing metrics across methods
├── summary.json                  # Summary of output locations and parameters
├── models/                       # Saved models for each method
│   ├── one-hot_model.pkl
│   ├── carp_model.pkl
│   └── esm2_model.pkl
├── metrics/                      # Performance metrics for each method
│   ├── one-hot_metrics.json
│   ├── carp_metrics.json
│   └── ...
└── cache/                        # Cached embeddings
```

You can also generate these outputs programmatically:

```python
from pathlib import Path
from fast_seqfunc import train_model, save_model, save_detailed_metrics

# Create output directories
output_dir = Path("my_model_outputs")
output_dir.mkdir(exist_ok=True)
metrics_dir = output_dir / "metrics"
metrics_dir.mkdir(exist_ok=True)
cache_dir = output_dir / "cache"
cache_dir.mkdir(exist_ok=True)

# Train model
model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    embedding_method="one-hot",
    cache_dir=cache_dir,
)

# Save model
save_model(model_info, output_dir / "model.pkl")

# Save detailed metrics if test data was provided
if model_info.get("test_results"):
    save_detailed_metrics(
        metrics_data=model_info["test_results"],
        output_dir=metrics_dir,
        model_type=model_info["model_type"],
        embedding_method="one-hot"
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
