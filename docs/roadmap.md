# Roadmap

This document outlines the planned development path for fast-seqfunc.

## Current Roadmap Items

### Expanded Embedding Methods

Support for more sequence embedding methods beyond one-hot encoding. While the CLI and core functions reference "carp" and "esm2" embedders, these are not currently implemented. Integrating with ESM2, CARP, or other pre-trained models will enhance the library's capabilities.

### Signal Detection and Analysis Tools

Enhance Fast-SeqFunc's primary purpose of detecting signal in sequence-function data:

1. **Automated Signal Detection Reports**: Generate comprehensive reports that clearly indicate whether signal exists in the data with statistical confidence measures
2. **Benchmark Comparisons**: Automatically compare model performance against random expectation and theoretical upper bounds
3. **Advanced Visualization Tools**: Create specialized visualizations that highlight the presence or absence of signal in different ways
4. **Candidate Generation Strategies**: Develop smarter approaches for generating candidate sequences when signal is detected
5. **Signal-Guided Model Selection**: Automatically recommend more advanced modeling approaches when sufficient signal is detected
6. **Effect Size Analysis**: Tools to estimate the magnitude of sequence effects on function

### Batch Processing for Large Datasets

Implement efficient batch processing for datasets that are too large to fit in memory, especially when using more complex embedding methods that require significant computational resources.

### Cluster-Based Cross-Validation Framework

Enhance the validation strategy with cluster-based cross-validation, where sequences are clustered at a specified identity level (e.g., using CD-HIT) and entire clusters are left out during training. This approach provides a more realistic assessment of model generalizability to truly novel sequences.

### ONNX Model Integration

Add support for exporting models to ONNX format and rehydrating models from ONNX rather than pickle files, improving portability, performance, and security.

### Caching Mechanism

Add disk caching for embeddings to improve performance on repeated runs with the same sequences.

### Enhanced Visualization Options

Develop built-in visualizations for model performance and sequence importance analysis to help users better understand their models.

### FASTA File Support

Add direct support for loading sequence data from FASTA files, a common format in bioinformatics.

## Completed Items

The following items from the previous roadmap have been implemented and are now available:

- **Custom Alphabets via Configuration File**: The `Alphabet` class supports loading/saving custom alphabets through JSON configuration files with the `from_json` and `to_json` methods.

- **Auto-Inferred Alphabets**: The `infer_alphabet` function can automatically infer alphabets from input sequences, and is exposed in the public API.

- **Automatic Cluster Splits**: Basic data splitting capabilities are available through the synthetic data generation functionality, though not based on sequence clustering.

## Future Considerations

*Additional roadmap items will be added here after review.*
