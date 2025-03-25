# Roadmap

This document outlines the planned development path for fast-seqfunc.

## Current Roadmap Items

### Custom Alphabets via Configuration File
Implement support for user-defined alphabets through a configuration file format. This will make the library more flexible and allow it to work with a wider range of sequence types beyond the standard DNA/RNA/protein alphabets.

### Auto-Inferred Alphabets
Add functionality to automatically infer alphabets from input sequences. The inferred alphabets will be saved to a configuration file for future reference, improving usability while maintaining reproducibility.

### Automatic Cluster Splits
Develop an automatic method for splitting clusters of sequences based on internal metrics. This will enhance the quality of sequence classification and make the process more user-friendly.

### Expanded Embedding Methods
Support for more sequence embedding methods beyond one-hot encoding, such as integrating with ESM2, CARP, or other pre-trained models that are mentioned in the CLI but not fully implemented in the current embedders module.

### Batch Processing for Large Datasets
Implement efficient batch processing for datasets that are too large to fit in memory, especially when using more complex embedding methods that require significant computational resources.

### Cluster-Based Cross-Validation Framework
Enhance the validation strategy with cluster-based cross-validation, where sequences are clustered at a specified identity level (e.g., using CD-HIT) and entire clusters are left out during training. This approach provides a more realistic assessment of model generalizability to truly novel sequences.

### ONNX Model Integration
Add support for exporting models to ONNX format and rehydrating models from ONNX rather than pickle files, improving portability, performance, and security.

## Future Considerations

*Additional roadmap items will be added here after review.*
