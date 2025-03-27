# Fast-SeqFunc Quickstart

This guide demonstrates how to use `fast-seqfunc` for training sequence-function models and making predictions with your own sequence data.

## Prerequisites

- Python 3.11 or higher
- The `fast-seqfunc` package installed

## Setup

The `fast-seqfunc` package comes with a command-line interface (CLI) that makes it easy to train models and make predictions without writing any code.

To see all available commands, run:

```bash
fast-seqfunc --help
```

For help with a specific command, use:

```bash
fast-seqfunc [command] --help
```

## Data Preparation

For this tutorial, we assume you already have a sequence-function dataset with the following format:

```
sequence,function
ACGTACGT...,0.75
TACGTACG...,0.63
...
```

You'll need to split your data into training and test sets. You can use any CSV file manipulation tool for this, or you can use the built-in synthetic data generator to create sample data:

```bash
# Generate synthetic regression data (DNA sequences with G count function)
fast-seqfunc generate-synthetic g_count --output-dir data --total-count 1000 --split-data
```

This will create `train.csv`, `val.csv`, and `test.csv` files in the `data` directory.

## Training a Model

With `fast-seqfunc`, you can train a model with a single command:

```bash
# Train and compare multiple models automatically
fast-seqfunc train data/train.csv \
  --val-data data/val.csv \
  --test-data data/test.csv \
  --sequence-col sequence \
  --target-col function \
  --embedding-method one-hot \
  --model-type regression \
  --output-dir outputs

# The model will be saved to outputs/model.pkl by default
```

The command above will:

1. Load your training, validation, and test data
2. Embed the sequences using one-hot encoding
3. Train multiple regression models using PyCaret
4. Select the best model based on performance
5. Evaluate the model on the test data
6. Save the model and performance metrics

## Making Predictions

Making predictions on new sequences is straightforward:

```bash
# Make predictions on test data
fast-seqfunc predict-cmd outputs/model.pkl data/test.csv \
  --sequence-col sequence \
  --output-dir prediction_outputs \
  --predictions-filename predictions.csv

# Results will be saved to prediction_outputs/predictions.csv
# A histogram of predictions will be generated (if applicable)
```

This command will:

1. Load your trained model
2. Load the sequences from your test data
3. Generate predictions for each sequence
4. Save the results to a CSV file with both the original sequences and the predictions

## Comparing Embedding Methods

You can also compare different embedding methods to see which works best for your data:

```bash
# Compare different embedding methods on the same dataset
fast-seqfunc compare-embeddings data/train.csv \
  --test-data data/test.csv \
  --sequence-col sequence \
  --target-col function \
  --model-type regression \
  --output-dir comparison_outputs

# Results will be saved to comparison_outputs/embedding_comparison.csv
# Individual models will be saved in comparison_outputs/models/
```

This command will:

1. Train models using different embedding methods (one-hot, and others if available)
2. Evaluate each model on the test data
3. Compare the performance metrics
4. Save the results and models

## Generating Synthetic Data

Fast-SeqFunc includes a powerful synthetic data generator for different sequence-function relationships:

```bash
# See available synthetic data tasks
fast-seqfunc list-synthetic-tasks

# Generate data for a specific task
fast-seqfunc generate-synthetic motif_position \
  --sequence-type dna \
  --motif ATCG \
  --noise-level 0.2 \
  --output-dir data/motif_task

# Generate classification data
fast-seqfunc generate-synthetic classification \
  --sequence-type protein \
  --output-dir data/classification_task
```

The synthetic data generator can create datasets with various sequence-function relationships, including:

- Linear relationships (G count, GC content)
- Position-dependent functions (motif position)
- Nonlinear relationships (length-dependent functions)
- Classification problems (presence/absence of patterns)
- And many more!

## Interpreting Results for Signal Detection

One of the primary purposes of Fast-SeqFunc is to quickly determine if there is meaningful "signal" in your sequence-function data. Here's how to interpret your results:

### Evaluating Signal Presence

1. **Check performance metrics**:
   - For regression: R², RMSE, and MAE values
   - For classification: Accuracy, F1 score, AUC-ROC

2. **Use visualizations**:
   - Scatter plots of predicted vs. actual values
   - Residual plots showing systematic patterns or random noise
   - ROC curves for classification tasks

3. **Benchmarks for determining signal**:
   - Models significantly outperforming random guessing indicate signal
   - R² values above 0.3-0.4 suggest detectable relationships
   - AUC-ROC values above 0.6-0.7 indicate useful classification signal

### Leveraging Early Signal

When you detect signal:

1. **Prioritize candidates**: Use model predictions to rank and select promising sequences for experimental testing
2. **Iterate experimentally**: Test top-ranked sequences and use results to refine your model
3. **Decide on complexity**: Strong signal warrants investment in more sophisticated models like neural networks
4. **Compare embedding methods**: If signal is present, explore if more complex embeddings (ESM, CARP) improve performance

Remember that even modest performance can be valuable for prioritizing experimental candidates and guiding exploration of sequence space.

## Next Steps

After mastering the basics, you can:

1. Try different embedding methods (currently only `one-hot` is supported, with more coming soon)
2. Experiment with classification problems by setting `--model-type classification`
3. Generate different types of synthetic data to benchmark your approach
4. Explore the Python API for more advanced customization

For more details, check out the [API documentation](api_reference.md).
