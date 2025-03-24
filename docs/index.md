# Fast-SeqFunc

Welcome to Fast-SeqFunc, a Python package designed for efficient sequence-function modeling of proteins and nucleotides.

## Overview

Fast-SeqFunc provides a simple, high-level API that handles various sequence embedding methods and automates model selection and training through the PyCaret framework.

* [Design Document](design.md): Learn about the architecture and design principles
* [API Documentation](api.md): Explore the package API

## Quickstart

### Install from PyPI

```bash
pip install fast-seqfunc
```

### Install from source

```bash
git clone git@github.com:ericmjl/fast-seqfunc.git
cd fast-seqfunc
pip install -e .
```

### Basic Usage

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
fast-seqfunc train train_data.csv --sequence-col sequence --target-col function
```

Make predictions:

```bash
fast-seqfunc predict-cmd model.pkl new_sequences.csv --output-path predictions.csv
```

## Documentation

For full documentation, see the [design document](design.md) and [API reference](api.md).

## Why this project exists

Fast-SeqFunc was created to simplify the process of sequence-function modeling for proteins and nucleotide sequences. It eliminates the need for users to implement their own embedding methods or model selection processes, allowing them to focus on their research questions.

By integrating state-of-the-art embedding methods like CARP and ESM2 with automated machine learning from PyCaret, Fast-SeqFunc makes advanced ML techniques accessible to researchers without requiring deep ML expertise.
