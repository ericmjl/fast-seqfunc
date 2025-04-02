# Additional Predictors: Usage Examples

This document provides examples of how to use additional predictor columns with Fast-SeqFunc.

## Example 1: Basic Usage with Python API

The following example demonstrates how to train a model with protein sequences and additional predictors (pH and temperature) that may affect protein function:

```python
import pandas as pd
from fast_seqfunc import train_model, predict

# Sample data with sequences, additional predictors, and function values
data = pd.DataFrame({
    'sequence': ['MKALIVLGL', 'MKHPIVLLL', 'MKLIVPMGL', 'MKAIVLELL'],
    'pH': [6.5, 7.0, 7.5, 8.0],
    'temperature': [25, 30, 35, 40],
    'activity': [0.45, 0.62, 0.78, 0.34]
})

# Split into train and test sets
train_data = data.iloc[:3]
test_data = data.iloc[3:]

# Train model using sequence and additional predictors
model_info = train_model(
    train_data=train_data,
    test_data=test_data,
    sequence_col='sequence',
    target_col='activity',
    additional_predictor_cols=['pH', 'temperature'],
    embedding_method='one-hot',
    model_type='regression',
    optimization_metric='r2'
)

# Make predictions on new data
new_data = pd.DataFrame({
    'sequence': ['MKAIVLELL', 'MKLIVLELL'],
    'pH': [7.0, 7.5],
    'temperature': [37, 38]
})

predictions = predict(model_info, new_data)
print(f"Predicted activities: {predictions}")
```

## Example 2: Using the CLI

You can also use additional predictors via the command-line interface:

```bash
# Train a model with additional predictors
fast-seqfunc train protein_data.csv \
    --sequence-col sequence \
    --target-col activity \
    --additional-predictors pH,temperature \
    --embedding-method one-hot \
    --model-type regression

# Make predictions
fast-seqfunc predict model.pkl new_sequences.csv \
    --output-path predictions.csv
```

The input CSV files should contain the sequence column, the target column, and any additional predictor columns specified.

## Example 3: Handling Categorical Predictors

Additional predictors can be numeric or categorical. Fast-SeqFunc will automatically handle the encoding of categorical predictors:

```python
import pandas as pd
from fast_seqfunc import train_model

# Data with both numeric and categorical predictors
data = pd.DataFrame({
    'sequence': ['MKALIVLGL', 'MKHPIVLLL', 'MKLIVPMGL', 'MKAIVLELL'],
    'pH': [6.5, 7.0, 7.5, 8.0],
    'buffer_type': ['phosphate', 'tris', 'phosphate', 'tris'],
    'cell_line': ['HEK293', 'CHO', 'HEK293', 'CHO'],
    'activity': [0.45, 0.62, 0.78, 0.34]
})

# Train model with both numeric and categorical predictors
model_info = train_model(
    train_data=data,
    sequence_col='sequence',
    target_col='activity',
    additional_predictor_cols=['pH', 'buffer_type', 'cell_line'],
    embedding_method='one-hot',
    model_type='regression'
)
```

## Example 4: Analyzing Feature Importance

With additional predictors, it becomes important to understand their relative importance:

```python
import matplotlib.pyplot as plt
from fast_seqfunc import train_model, feature_importance

# Train model with additional predictors
model_info = train_model(
    train_data=data,
    sequence_col='sequence',
    target_col='activity',
    additional_predictor_cols=['pH', 'temperature', 'buffer_type']
)

# Get feature importance
importance = feature_importance(model_info)

# Plot feature importance
plt.figure(figsize=(10, 6))
importance.plot(kind='bar')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

## Example 5: Saving and Loading Models

Models with additional predictors can be saved and loaded just like regular models:

```python
from fast_seqfunc import save_model, load_model

# Save model
save_model(model_info, 'protein_activity_model.pkl')

# Load model
loaded_model = load_model('protein_activity_model.pkl')

# Make predictions using the loaded model
predictions = predict(
    loaded_model,
    new_data,
    sequence_col='sequence'
)
```
