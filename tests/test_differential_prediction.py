#!/usr/bin/env python3
"""Test script for differential prediction functionality."""

import pandas as pd
import numpy as np
from fast_seqfunc import train_model, predict, generate_sequence_function_data, Alphabet

def test_differential_prediction():
    """Test differential prediction functionality using realistic protein data."""
    
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Generate realistic protein sequence-function data
    train_data = generate_sequence_function_data(
        count=50,  # Smaller dataset for faster testing
        sequence_length=12,  # Short peptides 
        alphabet=Alphabet.protein(),
        function_type="nonlinear",
        noise_level=0.2,
        position_weights=[2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1],
        motif_effects={"RGD": 3.0, "KKK": -2.0, "LLL": 1.5}  # Common peptide motifs
    )
    
    print("Training data:")
    print(train_data)
    print()
    
    # Test 1: Train with differential prediction enabled
    print("Testing differential prediction...")
    
    model_info = train_model(
        train_data=train_data,
        differential_prediction=True,
        reference_strategy="median",
        model_type="regression"
    )
    
    print(f"Model trained with differential prediction: {model_info['differential_prediction']}")
    print(f"Reference sequence: {model_info['reference_sequence']}")
    print(f"Reference function: {model_info['reference_function']}")
    print()
    
    # Test 2: Make predictions on new sequences (realistic protein sequences)
    new_sequences = ["ACDEFGHIKLMN", "PQRSTVWYACDE"]  # Valid protein sequences
    predictions = predict(model_info, new_sequences)
    
    print("Predictions on new sequences:")
    for seq, pred in zip(new_sequences, predictions):
        print(f"  {seq}: {pred:.3f}")
    print()
    
    # Test 3: Compare with regular prediction
    print("Comparing with regular prediction...")
    
    regular_model_info = train_model(
        train_data=train_data,
        differential_prediction=False,
        model_type="regression"
    )
    
    regular_predictions = predict(regular_model_info, new_sequences)
    
    print("Regular predictions:")
    for seq, pred in zip(new_sequences, regular_predictions):
        print(f"  {seq}: {pred:.3f}")
    print()
    
    print("Differential vs Regular differences:")
    for seq, diff_pred, reg_pred in zip(new_sequences, predictions, regular_predictions):
        diff = abs(diff_pred - reg_pred)
        print(f"  {seq}: |{diff_pred:.3f} - {reg_pred:.3f}| = {diff:.3f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_differential_prediction()