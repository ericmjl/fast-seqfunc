# Fast-SeqFunc Documentation

`fast-seqfunc` is a Python library for building sequence-function models quickly and easily, leveraging PyCaret and machine learning techniques to predict functional properties from biological sequences.

## Getting Started

- [Quickstart Tutorial](quickstart.md) - Learn the basics of training and using sequence-function models
- [Regression Tutorial](tutorials/regression_tutorial.md) - Learn how to predict continuous values from sequences
- [Classification Tutorial](tutorials/classification_tutorial.md) - Learn how to classify sequences into discrete categories

## Installation

Install `fast-seqfunc` using pip:

```bash
pip install fast-seqfunc
```

Or directly from GitHub for the latest version:

```bash
pip install git+https://github.com/ericmjl/fast-seqfunc.git
```

## Key Features

- **Easy-to-use API**: Train models and make predictions with just a few lines of code
- **Automatic Model Selection**: Uses PyCaret to automatically compare and select the best model
- **Sequence Embedding**: Currently supports one-hot encoding with more methods coming soon
- **Regression and Classification**: Support for both continuous values and categorical outputs
- **Comprehensive Evaluation**: Built-in metrics and visualization utilities

## Why Fast-SeqFunc?

The primary motivation behind Fast-SeqFunc is to quickly answer a crucial question in sequence-function modeling: **Is there detectable signal in my data?**

In biological sequence-function problems, determining whether a predictive relationship exists is a critical first step before investing significant resources in complex modeling approaches. Fast-SeqFunc allows you to:

1. **Rapidly detect signal**: Quickly build baseline models to determine if your sequence data contains predictive information
2. **Make early decisions**: Identify promising directions early in your research process
3. **Rank candidates efficiently**: Use simple but effective models to score and prioritize candidate sequences for experimental testing
4. **Validate before scaling**: Confirm signal exists before investing time in developing more complex neural network models
5. **Iterate strategically**: When signal is detected, use that knowledge to guide the development of more sophisticated models

By providing a fast path to baseline model development, Fast-SeqFunc helps you make informed decisions about where to focus your modeling efforts.

## Basic Usage

### Command-Line Interface

Fast-SeqFunc provides a convenient command-line interface for common tasks:

```bash
# Train a model
fast-seqfunc train train_data.csv --sequence-col sequence --target-col function --embedding-method one-hot

# Make predictions with a trained model
fast-seqfunc predict-cmd model.pkl new_sequences.csv --sequence-col sequence --output-dir predictions --predictions-filename predictions.csv

# Compare different embedding methods
fast-seqfunc compare-embeddings train_data.csv --test-data test_data.csv
```

### Python API

You can also use Fast-SeqFunc programmatically in your Python code:

```python
from fast_seqfunc import train_model, predict, save_model

# Train a model
model_info = train_model(
    train_data=train_df,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    model_type="regression"
)

# Make predictions
predictions = predict(model_info, new_sequences)

# Save the model
save_model(model_info, "model.pkl")
```

## Roadmap

Future development plans include:

1. Additional embedding methods (ESM, CARP, etc.)
2. Integration with more advanced deep learning models
3. Enhanced visualization and interpretation tools
4. Expanded support for various sequence types
5. Benchmarking against established methods

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to discuss improvements or feature requests.
