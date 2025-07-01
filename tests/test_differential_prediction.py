#!/usr/bin/env python3
"""Test script for differential prediction functionality."""

import pandas as pd
import numpy as np
from fast_seqfunc import train_model, predict

def test_differential_prediction():
    """Test basic differential prediction functionality."""
    
    # Create synthetic sequence-function data
    sequences = [
        "ACDEFG",
        "ACDEFH", 
        "ACDEFI",
        "ACDEJG",
        "ACDEJH",
        "ACDEKI",
        "BDEFGH",
        "BDEFGI",
        "BDEJGH",
        "BDEJGI"
    ]
    
    # Create function values with some pattern
    # Let's say function depends on sequence length and certain amino acids
    functions = []
    for seq in sequences:
        func = len(seq) * 2.0  # Base function value
        func += seq.count('A') * 1.5  # A amino acid bonus
        func += seq.count('B') * 2.0  # B amino acid bonus
        func += np.random.normal(0, 0.1)  # Small noise
        functions.append(func)
    
    # Create training DataFrame
    train_data = pd.DataFrame({
        'sequence': sequences,
        'function': functions
    })
    
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
    
    # Test 2: Make predictions on new sequences
    new_sequences = ["ACDEFX", "BDEJXY"]
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