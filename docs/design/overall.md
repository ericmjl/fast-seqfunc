# Fast-SeqFunc: Design Document

## Overview

Fast-SeqFunc is a Python package designed for efficient sequence-function modeling of proteins and nucleotide sequences. It provides a simple, high-level API that handles sequence embedding methods and automates model selection and training through the PyCaret framework.

The primary purpose of Fast-SeqFunc is to quickly detect whether there is meaningful "signal" in sequence-function data. By enabling rapid model development, researchers can determine early if predictive relationships exist, opportunistically use these models for ranking candidate sequences, and make informed decisions about investing in more complex modeling approaches when signal is detected.

## Design Goals

1. **Simplicity**: Provide a clean, intuitive API for training sequence-function models
2. **Flexibility**: Support multiple sequence types with custom alphabet capabilities
3. **Automation**: Leverage PyCaret to automate model selection and hyperparameter tuning
4. **Performance**: Enable efficient processing through lazy loading and clean architecture
5. **Signal Detection**: Rapidly determine if predictive relationships exist in the data
6. **Decision Support**: Help users make informed choices about modeling approaches based on signal strength
7. **Candidate Ranking**: Enable efficient prioritization of sequences for experimental testing

## Architecture

### Core Components

The package is structured around these key components:

1. **Core API** (`core.py`)
   - High-level functions for training, prediction, and model management
   - Handles data loading and orchestration between embedders and models

2. **Embedders** (`embedders.py`)
   - `OneHotEmbedder`: One-hot encoding for protein, DNA, RNA, and custom alphabets
   - Factory function `get_embedder` to create embedder instances

3. **Alphabets** (`alphabets.py`)
   - `Alphabet` class for representing character sets and tokenization rules
   - Support for standard alphabets (protein, DNA, RNA) and custom alphabets
   - Handles mixed-length tokens and various sequence formats

4. **Models** (`models.py`)
   - `SequenceFunctionModel`: Main model class integrating with PyCaret
   - Handles training, prediction, evaluation, and persistence

5. **CLI** (`cli.py`)
   - Command-line interface built with Typer
   - Commands for training, prediction, and embedding comparison

6. **Synthetic Data** (`synthetic.py`)
   - Functions for generating synthetic sequence-function data
   - Various task generators for different use cases

### Data Flow

1. User provides sequence-function data (sequences + target values)
2. Data is validated and preprocessed
3. Sequences are embedded using the selected embedding method
4. PyCaret explores various ML models on the embeddings
5. Best model is selected, fine-tuned, and returned
6. Results and model artifacts are saved

## API Design

### High-Level API

```python
from fast_seqfunc import train_model, predict, load_model

# Train a model
model_info = train_model(
    train_data="train_data.csv",
    test_data="test_data.csv",
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    model_type="regression",
    optimization_metric="r2",
)

# Make predictions
predictions = predict(model_info, new_sequences)

# Save and load models
with open("model.pkl", "wb") as f:
    pickle.dump(model_info, f)

with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
```

### Command-Line Interface

The CLI provides commands for training, prediction, and embedding comparison:

```bash
# Train a model
fast-seqfunc train train_data.csv --sequence-col sequence --target-col function --embedding-method one-hot

# Make predictions
fast-seqfunc predict model.pkl new_sequences.csv --output-path predictions.csv

# Compare embedding methods
fast-seqfunc compare-embeddings train_data.csv --test-data test_data.csv
```

## Key Design Decisions

### 1. Embedding Strategy

- **One-Hot Encoding**: Primary embedding method for all sequence types
- **Custom Alphabets**: Support for user-defined alphabets through the `Alphabet` class
- **Auto-Detection**: Auto-detection of sequence type (protein, DNA, RNA)
- **Gap Handling**: Configurable padding for sequences of different lengths

### 2. Alphabet Design

- **Flexible Tokenization**: Support for character-based, delimited, and regex-based tokenization
- **Standard Alphabets**: Built-in support for protein, DNA, and RNA
- **Token Mapping**: Bidirectional mapping between tokens and indices
- **Sequence Padding**: Automatic handling of variable-length sequences

### 3. Model Integration

- **PyCaret Integration**: Leverage PyCaret for automated model selection
- **Model Type Flexibility**: Support for regression and classification tasks
- **Performance Evaluation**: Built-in metrics calculation based on model type
- **Serialization**: Simple model saving and loading

### 4. Synthetic Data Generation

- **Task Generators**: Functions for creating various sequence-function relationships
- **Customization**: Configurable difficulty, noise, and relationship types
- **Data Types**: Support for protein, DNA, RNA, and integer sequences

## Implementation Details

### OneHotEmbedder

- Supports protein, DNA, RNA, and custom sequences
- Auto-detects sequence type when configured to 'auto'
- Handles padding and truncating with configurable gap character
- Provides both flattened and 2D one-hot encodings

### Alphabet Class

- Represents sets of tokens with various tokenization strategies
- Provides factory methods for standard biological alphabets
- Supports custom token sets with arbitrary delimiters
- Handles sequence padding and token-to-index mappings

### SequenceFunctionModel

- Integrates with PyCaret for model training
- Handles different model types (regression, classification)
- Provides model evaluation methods
- Supports serialization for saving/loading

### Synthetic Data Generation

- Generate random sequences with controlled properties
- Create sequence-function datasets with known relationships
- Support for various task types (count, position, pattern, etc.)
- Configurable noise and complexity levels

## Dependencies

- Core dependencies:
  - pandas: Data handling
  - numpy: Numerical operations
  - pycaret: Automated ML
  - scikit-learn: Model evaluation metrics
  - loguru: Logging
  - typer: CLI
  - lazy-loader: Lazy imports

## Future Enhancements

1. **Advanced Embedders**:
   - Implement CARP integration for protein embeddings
   - Implement ESM2 integration for protein embeddings

2. **Caching Mechanism**:
   - Add disk caching for embeddings to improve performance on repeated runs

3. **Enhance PyCaret Integration**:
   - Add more customization options for model selection
   - Support for custom models

4. **Expand Data Loading**:
   - Support for FASTA file formats
   - Support for more complex dataset structures

5. **Add Visualization**:
   - Built-in visualizations for model performance
   - Sequence importance analysis

## Conclusion

Fast-SeqFunc provides a streamlined approach to sequence-function modeling with a focus on simplicity and automation. The architecture balances flexibility with ease of use, allowing users to train models with minimal code while providing options for custom alphabets and sequence types.

The current implementation focuses on one-hot encoding with strong support for custom alphabets, while laying the groundwork for more advanced embedding methods in the future.
