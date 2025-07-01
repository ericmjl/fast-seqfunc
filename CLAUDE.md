# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fast-SeqFunc is a Python package for efficient sequence-function modeling for proteins and nucleotides. It provides automated machine learning for biological sequence data using PyCaret and embedding methods like one-hot encoding. The package supports both standard prediction and differential prediction modes for enhanced performance.

## Development Environment

This project uses **Pixi** for dependency management instead of pip/conda. All development commands should use Pixi environments.

## Essential Commands

### Setup
```bash
pixi install                    # Install all dependencies
pixi run setup                  # Setup pre-commit hooks and update them
```

### Testing
```bash
pixi run test                   # Run all tests with pytest
pytest -m "not slow"            # Skip slow tests
pytest tests/test_specific.py   # Run specific test file
```

### Linting and Code Quality
```bash
pixi run lint                   # Run pre-commit hooks on all files
pre-commit run --all-files      # Alternative lint command
```

### Documentation
```bash
pixi run build-docs             # Build MkDocs documentation
pixi run serve-docs             # Serve documentation locally
```

### CLI Usage
```bash
pixi run fast-seqfunc           # Run the CLI tool

# Standard training:
fast-seqfunc train data.csv --sequence-col sequence --target-col function

# Differential prediction training:
fast-seqfunc train data.csv --differential-prediction --reference-strategy median

# Making predictions:
fast-seqfunc predict-cmd model.pkl new_data.csv --output-dir predictions
```

## Code Architecture

### Core Components

- **`fast_seqfunc/core.py`**: Main API functions (`train_model`, `predict`, `save_model`)
- **`fast_seqfunc/embedders.py`**: Sequence embedding implementations (OneHotEmbedder, DifferentialEmbedder, get_embedder)
- **`fast_seqfunc/models.py`**: Model training and evaluation logic using PyCaret
- **`fast_seqfunc/cli.py`**: Command-line interface using Typer
- **`fast_seqfunc/synthetic.py`**: Synthetic data generation for testing and examples
- **`fast_seqfunc/alphabets.py`**: Custom alphabet handling for sequences
- **`fast_seqfunc/preprocessing.py`**: Data preprocessing utilities
- **`fast_seqfunc/schemas.py`**: Data validation schemas

### Key Design Patterns

1. **Embedding System**: Uses `get_embedder()` factory function to create embedding instances (OneHotEmbedder, DifferentialEmbedder)
2. **PyCaret Integration**: Core ML functionality built on PyCaret's automated ML pipeline
3. **CLI Design**: Typer-based CLI with separate commands for train/predict/compare
4. **Output Structure**: Organized output directories with models/, metrics/, and cache/ subdirectories
5. **Differential Prediction**: Trains on sequence embedding differences and function differences, predicts absolute values

### Data Flow

#### Standard Prediction:
1. Load sequence data (CSV/DataFrame)
2. Embed sequences using specified method (one-hot encoding)
3. Train models using PyCaret's automated ML
4. Evaluate and save best model with metrics
5. Generate predictions on new data

#### Differential Prediction:
1. Load sequence data and select reference sequence (median/mean/random strategy)
2. Generate n² pairwise training data: for each pair (seq_i, seq_j), target = func_i - func_j
3. Compute pairwise embedding differences: embedding(seq_i) - embedding(seq_j) for all pairs
4. Train models on pairwise embedding differences → pairwise function differences
5. For prediction: compute embedding(new_seq) - embedding(reference), predict difference, add reference function

## Testing Strategy

- **Unit tests**: Individual component testing in `tests/`
- **Integration tests**: End-to-end CLI and API testing  
- **Slow tests**: Marked with `@pytest.mark.slow` for computationally expensive tests
- **Synthetic data**: Uses `synthetic.py` module for generating test datasets
- **Differential prediction tests**: `tests/test_differential_prediction.py` validates the differential prediction workflow

## Development Notes

### Pixi Environments
- `default`: Core development environment with tests, devtools, notebook, setup
- `docs`: Documentation building
- `tests`: Testing only
- `cuda`: CUDA-enabled environment for GPU acceleration

### Pre-commit Hooks
- Ruff for linting and formatting
- Interrogate for docstring coverage (100% required)
- nbstripout for notebook cleaning
- Standard pre-commit hooks (trailing whitespace, YAML validation)

### Configuration Files
- `pyproject.toml`: Main configuration for build, dependencies, and tools
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `mkdocs.yaml`: Documentation configuration

## Differential Prediction Feature

### Usage

**Python API:**
```python
from fast_seqfunc import train_model, predict

# Train differential model
model_info = train_model(
    train_data=data,
    differential_prediction=True,
    reference_strategy="median",  # or "mean", "random"
    reference_sequence="ACDEFG",  # optional: specify reference
    reference_function=5.2        # optional: specify reference function
)

# Predict (returns absolute function values)
predictions = predict(model_info, new_sequences)
```

**CLI:**
```bash
fast-seqfunc train data.csv \
    --differential-prediction \
    --reference-strategy median \
    --reference-sequence "ACDEFG" \
    --reference-function 5.2
```

### Key Implementation Details

- **Reference Selection**: Automatically selects reference using median/mean function value or random choice
- **Training Data**: Generates n² pairwise comparisons between all sequences with function differences as targets  
- **Embeddings**: DifferentialEmbedder computes embedding differences from reference sequence
- **Model Storage**: Saves reference sequence and function value with trained model
- **Prediction**: Converts predicted function differences back to absolute values using stored reference

### When to Use Differential Prediction

- When you have many sequences with small functional differences
- When you want to capture relative changes between sequences
- When standard prediction struggles with the sequence-function relationship
- For applications where the difference between sequences is more meaningful than absolute values

## Memories

- No need to worry about linting. When I git commit, I will have pre-commit hooks running :)