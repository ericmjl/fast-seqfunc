# Additional Predictors in Fast-SeqFunc: Design Document

## Overview

Fast-SeqFunc is designed to model the relationship between biological sequences and a target function/property. Currently, it supports using only the sequence as the predictor. This design document outlines the necessary changes to expand Fast-SeqFunc to incorporate additional predictor columns alongside the sequence data.

## Motivation

In many real-world applications, sequence-function relationships may depend on additional contextual variables or experimental conditions. For example:

1. Protein function might depend not only on the amino acid sequence but also on pH, temperature, or salt concentration
2. Gene expression levels may depend on the DNA sequence as well as cell type, developmental stage, or treatment conditions
3. Binding affinity might depend on sequence and additional information about binding partners

By supporting additional predictor columns, Fast-SeqFunc will become more versatile and applicable to a wider range of biological problems where context matters.

## Design Goals

1. **Maintain API Simplicity**: Preserve the current simple API while extending it to handle additional predictors
2. **Backward Compatibility**: Ensure existing code continues to work without changes
3. **Flexible Integration**: Allow for simple integration of sequence embeddings with additional predictors
4. **CLI Support**: Extend the command-line interface to handle additional predictor columns
5. **Consistent Implementation**: Apply changes consistently across both the Python API and CLI

## Architecture Changes

### Core Components Affected

1. **Core API** (`core.py`):
   - Extend `train_model` function to accept additional predictor columns
   - Modify data processing to incorporate additional predictors with embeddings

2. **Models** (`models.py`):
   - Update `SequenceFunctionModel` to handle additional predictors alongside sequence embeddings

3. **CLI** (`cli.py`):
   - Add options to specify additional predictor columns
   - Update data processing for CLI commands

4. **Documentation**:
   - Update API reference
   - Add examples showing how to use additional predictors

### Data Flow Modifications

Current data flow:

1. User provides sequences + target values
2. Sequences are embedded
3. ML models are trained on embeddings

New data flow:

1. User provides sequences + additional predictors + target values
2. Sequences are embedded
3. Embeddings are combined with additional predictors
4. ML models are trained on the combined features

## API Design

### Python API Changes

#### Current API

```python
model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",
    target_col="function",
    embedding_method="one-hot",
    model_type="regression",
    optimization_metric="r2",
)
```

#### Enhanced API

```python
model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col="sequence",
    target_col="function",
    additional_predictor_cols=["pH", "temperature"],  # New parameter
    embedding_method="one-hot",
    model_type="regression",
    optimization_metric="r2",
)
```

#### Predict Function Changes

Current:

```python
predictions = predict(model_info, new_sequences)
```

Enhanced:

```python
predictions = predict(
    model_info,
    new_data,  # Now can be DataFrame with sequence and additional predictor columns
    sequence_col="sequence"
)
```

### CLI Changes

Current CLI:

```bash
fast-seqfunc train train_data.csv --sequence-col sequence --target-col function
```

Enhanced CLI:

```bash
fast-seqfunc train train_data.csv --sequence-col sequence --target-col function --additional-predictors pH,temperature
```

## Implementation Strategy

### Feature Combination Method

The implementation will use a simple concatenation approach to combine sequence embeddings with additional predictors. This means that additional predictor columns will be appended to the sequence embedding features to create the final feature matrix for model training.

### Data Processing Flow

1. Load CSV or DataFrame data as currently implemented
2. Validate presence of additional predictor columns if specified
3. Embed sequence data using the existing embedding pipeline
4. Process additional predictors (scaling, handling missing values, etc.)
5. Combine sequence embeddings with additional predictors
6. Train models on the combined feature set

### Data Validation and Preprocessing

Additional predictors may require preprocessing:

- Handling missing values
- Scaling numerical features
- Encoding categorical features
- Type validation

We'll implement a preprocessing pipeline for additional predictors that handles these tasks automatically.

## Model Information Enhancement

The model_info dictionary will be enhanced to include:

```python
{
    "model": trained_model,
    "model_type": model_type,
    "embedder": embedder,
    "embed_cols": embed_cols,
    "additional_predictor_cols": additional_predictor_cols,  # New
    "additional_predictor_preprocessing": preprocessing_pipeline,  # New
    "test_results": test_results,
}
```

This ensures all information needed for making predictions with additional predictors is preserved.

## Serialization Changes

The model serialization format will need to include information about additional predictors. We'll maintain backward compatibility by checking for the presence of additional predictor information when loading existing models.

## Code Changes Required

### In `core.py`

1. Update `train_model` function signature to accept additional predictors
2. Modify data processing to handle additional predictors
3. Update model information dictionary to include additional predictor metadata
4. Modify `predict` function to handle additional predictors in prediction data

### In `models.py`

1. Update `SequenceFunctionModel` class to handle additional predictors
2. Modify the `fit` and `predict` methods to incorporate additional predictors
3. Update serialization methods to handle additional predictor information

### In `cli.py`

1. Add new CLI options for specifying additional predictors
2. Update data loading and processing in CLI commands
3. Update help text and documentation

## Backwards Compatibility

To maintain backward compatibility:

- All new parameters will be optional with sensible defaults
- Existing code paths will work without modification
- Model serialization will be backward compatible

## Testing Strategy

Tests will be expanded to cover:

1. Training with various combinations of additional predictors
2. Making predictions with additional predictors
3. Serialization and deserialization of models with additional predictor information
4. CLI functionality for additional predictors
5. Edge cases (missing values, type mismatches, etc.)

## Future Enhancements

1. Add automated feature selection for additional predictors
2. Support for feature importance analysis that includes additional predictors
3. Add specialized visualizations for understanding the impact of additional predictors

## Conclusion

Adding support for additional predictor columns will significantly enhance the versatility of Fast-SeqFunc, making it applicable to a wider range of biological problems where context matters alongside sequence. The implementation will maintain the simplicity and user-friendliness of the current API while providing powerful new capabilities.
