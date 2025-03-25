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

## Basic Usage

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
