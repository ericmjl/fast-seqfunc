# Fast-SeqFunc: Design Document

## Overview

Fast-SeqFunc is a Python package designed for efficient sequence-function modeling of proteins and nucleotide sequences. It provides a simple, high-level API that handles various sequence embedding methods and automates model selection and training through the PyCaret framework.

## Design Goals

1. **Simplicity**: Provide a clean, intuitive API for training sequence-function models
2. **Flexibility**: Support multiple embedding methods for different sequence types
3. **Automation**: Leverage PyCaret to automate model selection and hyperparameter tuning
4. **Performance**: Enable efficient processing through caching and lazy loading

## Architecture

### Core Components

The package is structured around these key components:

1. **Core API** (`core.py`)
   - High-level functions for training, prediction, and model management
   - Handles data loading and orchestration between embedders and models

2. **Embedders** (`embedders.py`)
   - Abstract base class `SequenceEmbedder` defining common interface
   - Concrete implementations for different embedding methods:
     - `OneHotEmbedder`: Simple one-hot encoding for any sequence type
     - `CARPEmbedder`: Protein embeddings using Microsoft's CARP models
     - `ESM2Embedder`: Protein embeddings using Facebook's ESM2 models

3. **Models** (`models.py`)
   - `SequenceFunctionModel`: Main model class integrating with PyCaret
   - Handles training, prediction, evaluation, and persistence

4. **CLI** (`cli.py`)
   - Command-line interface built with Typer
   - Commands for training, prediction, and embedding comparison

### Data Flow

1. User provides sequence-function data (sequences + target values)
2. Data is validated and preprocessed
3. Sequences are embedded using selected method(s)
4. PyCaret explores various ML models on the embeddings
5. Best model is selected, fine-tuned, and returned
6. Results and model artifacts are saved

## API Design

### High-Level API

```python
from fast_seqfunc import train_model, predict, load_model

# Train a model
model = train_model(
    train_data,
    val_data=None,
    test_data=None,
    sequence_col="sequence",
    target_col="function",
    embedding_method="auto",  # or "one-hot", "carp", "esm2"
    model_type="regression",  # or "classification", "multi-class"
    optimization_metric="r2",  # or other metrics
    background=False,  # run in background
)

# Make predictions
predictions = predict(model, new_sequences)

# Save/load models
model.save("model_path")
loaded_model = load_model("model_path")
```

### Command-Line Interface

The CLI provides commands for training, prediction, and embedding comparison:

```bash
# Train a model
fast-seqfunc train train_data.csv --sequence-col sequence --target-col function --embedding-method one-hot

# Make predictions
fast-seqfunc predict-cmd model.pkl new_sequences.csv --output-path predictions.csv

# Compare embedding methods
fast-seqfunc compare-embeddings train_data.csv --test-data test_data.csv
```

## Key Design Decisions

### 1. Embedding Strategy

- **Abstract Base Class**: Created an abstract `SequenceEmbedder` class to ensure all embedding methods share a common interface
- **Caching Mechanism**: Built-in caching for embeddings to avoid redundant computation
- **Auto-Detection**: Auto-detection of sequence type (protein, DNA, RNA)
- **Lazy Loading**: Used lazy loader for heavy dependencies to minimize import overhead

### 2. Model Integration

- **PyCaret Integration**: Leveraged PyCaret for automated model selection
- **Model Type Flexibility**: Support for regression and classification tasks
- **Validation Strategy**: Support for custom validation sets
- **Performance Evaluation**: Built-in metrics calculation based on model type

### 3. Performance Optimizations

- **Lazy Loading**: Used for numpy, pandas, and other large dependencies
- **Disk Caching**: Cache embeddings to disk for reuse
- **Memory Efficiency**: Process data in batches when possible

## Implementation Details

### Embedders

1. **OneHotEmbedder**:
   - Supports protein, DNA, and RNA sequences
   - Auto-detects sequence type
   - Handles padding and truncating
   - Returns flattened one-hot encoding

2. **CARPEmbedder** (placeholder implementation):
   - Will integrate with Microsoft's protein-sequence-models
   - Supports different CARP model sizes

3. **ESM2Embedder** (placeholder implementation):
   - Will integrate with Facebook's ESM models
   - Supports different ESM2 model sizes and layer selection

### SequenceFunctionModel

- Integrates with PyCaret for model training
- Handles different model types (regression, classification)
- Manages embeddings dictionary
- Provides model evaluation methods
- Supports serialization for saving/loading

### Testing Strategy

- Unit tests for each component
- Integration tests for the full pipeline
- Test fixtures for synthetic data

## Dependencies

- Core dependencies:
  - pandas: Data handling
  - numpy: Numerical operations
  - pycaret: Automated ML
  - scikit-learn: Model evaluation metrics
  - loguru: Logging
  - typer: CLI
  - lazy-loader: Lazy imports

- Optional dependencies (for advanced embedders):
  - protein-sequence-models (for CARP)
  - fair-esm (for ESM2)

## Future Enhancements

1. **Complete Advanced Embedders**:
   - Implement full CARP integration
   - Implement full ESM2 integration

2. **Add Background Processing**:
   - Implement multiprocessing for background training and prediction

3. **Enhance PyCaret Integration**:
   - Add more customization options for model selection
   - Support for custom models

4. **Expand Data Loading**:
   - Support for FASTA file formats
   - Support for more complex dataset structures

5. **Add Visualization**:
   - Built-in visualizations for model performance
   - Sequence importance analysis

6. **Optimization**:
   - GPU acceleration for embedding generation
   - Distributed computing support for large datasets

## Conclusion

Fast-SeqFunc provides a streamlined approach to sequence-function modeling with a focus on simplicity and automation. The architecture balances flexibility with ease of use, allowing users to train models with minimal code while providing options for advanced customization.

The design leverages modern machine learning automation through PyCaret while providing domain-specific functionality for biological sequence data. The modular architecture allows for future extensions and optimizations.
