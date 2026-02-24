"""
Data Enhancement Script

Enhances existing prediction files with dates and OHLC data.
This is necessary because current predictions don't include timestamps
or historical price data needed for realistic backtesting.

Usage:
    python data_enhancement.py
    
Output:
    Creates *_enhanced.pkl files in predictions/ directory with:
    - y_pred: Predictions (unchanged)
    - y_true: Actual values (unchanged)
    - dates: Array of dates for each prediction
    - ohlc_data: DataFrame with OHLC data
    - metadata: Enhanced metadata
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Import root config first (before modifying sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import config as root_config

# Now add backtest_engine to path to import utils
sys.path.insert(0, os.path.join(current_dir, 'backtest_engine'))
import utils

# Use utils functions
reconstruct_test_dates = utils.reconstruct_test_dates
load_ohlc_data = utils.load_ohlc_data


def enhance_prediction_file(model_name: str):
    """
    Enhance a single prediction file with dates and OHLC data.
    
    Args:
        model_name: Name of model (e.g., 'DEFAULT_ANN', 'LSTM')
    """
    print(f"\nEnhancing predictions for {model_name}...")
    
    # Load original prediction file
    original_path = os.path.join(root_config.PREDICT_DIR, f"{model_name}_evaluate_data.pkl")
    
    if not os.path.exists(original_path):
        print(f"  ✗ File not found: {original_path}")
        return False
    
    with open(original_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract prediction arrays
    y_pred = data['y_pred']
    y_true = data['y_true']
    test_size = len(y_pred)
    
    print(f"  Loaded predictions: {test_size} samples")
    
    # Reconstruct dates for test set
    try:
        dates = reconstruct_test_dates(root_config.TICKER, test_size)
        print(f"  ✓ Reconstructed dates: {dates[0]} to {dates[-1]}")
    except Exception as e:
        print(f"  ✗ Failed to reconstruct dates: {e}")
        return False
    
    # Load OHLC data for the date range
    try:
        start_date = dates[0]
        end_date = dates[-1]
        ohlc_data = load_ohlc_data(root_config.TICKER, start_date, end_date)
        print(f"  ✓ Loaded OHLC data: {len(ohlc_data)} records")
    except Exception as e:
        print(f"  ✗ Failed to load OHLC data: {e}")
        return False
    
    # Create enhanced metadata
    enhanced_metadata = {
        'model': data.get('metadata', {}).get('model', {}),
        'ticker': root_config.TICKER,
        'test_start_date': str(dates[0]),
        'test_end_date': str(dates[-1]),
        'test_size': test_size,
        'enhanced_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_file': original_path,
    }
    
    # Create enhanced data structure
    enhanced_data = {
        'y_pred': y_pred,
        'y_true': y_true,
        'dates': dates,
        'ohlc_data': ohlc_data,
        'model_name': model_name,
        'metadata': enhanced_metadata
    }
    
    # Save enhanced file
    enhanced_path = os.path.join(root_config.PREDICT_DIR, f"{model_name}_evaluate_data_enhanced.pkl")
    with open(enhanced_path, 'wb') as f:
        pickle.dump(enhanced_data, f)
    
    print(f"  ✓ Saved enhanced file: {enhanced_path}")
    
    # Verify the enhanced file
    with open(enhanced_path, 'rb') as f:
        verify_data = pickle.load(f)
    
    required_keys = ['y_pred', 'y_true', 'dates', 'ohlc_data', 'metadata']
    missing_keys = [k for k in required_keys if k not in verify_data]
    
    if missing_keys:
        print(f"  ✗ Verification failed: Missing keys {missing_keys}")
        return False
    
    print(f"  ✓ Verification passed: All required keys present")
    return True


def main():
    """
    Main function to enhance all prediction files.
    """
    print("="*70)
    print("Data Enhancement Script for Backtesting")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Ticker: {root_config.TICKER}")
    print(f"  Sequence Length: {root_config.SEQUENCE_LENGTH}")
    print(f"  Prediction Steps: {root_config.PREDICTION_STEPS}")
    print(f"  Prediction Directory: {root_config.PREDICT_DIR}")
    
    # Find all prediction files
    if not os.path.exists(root_config.PREDICT_DIR):
        print(f"\n✗ Error: Prediction directory not found: {root_config.PREDICT_DIR}")
        print("Please run ann_models.py first to generate predictions.")
        return
    
    # Get list of model names from prediction files
    model_files = [f for f in os.listdir(root_config.PREDICT_DIR) 
                   if f.endswith('_evaluate_data.pkl') and not 'enhanced' in f]
    
    if not model_files:
        print(f"\n✗ Error: No prediction files found in {root_config.PREDICT_DIR}")
        print("Please run ann_models.py first to generate predictions.")
        return
    
    # Extract model names
    model_names = [f.replace('_evaluate_data.pkl', '') for f in model_files]
    
    print(f"\nFound {len(model_names)} model(s) to enhance:")
    for name in model_names:
        print(f"  - {name}")
    
    # Enhance each model's predictions
    success_count = 0
    for model_name in model_names:
        if enhance_prediction_file(model_name):
            success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("Enhancement Summary")
    print("="*70)
    print(f"Total models: {len(model_names)}")
    print(f"Successfully enhanced: {success_count}")
    print(f"Failed: {len(model_names) - success_count}")
    
    if success_count == len(model_names):
        print("\n✓ All predictions enhanced successfully!")
        print("\nYou can now run:")
        print("  - python run_backtest.py (to run backtests)")
        print("  - python -m backtest_engine.engine (to test engine)")
    else:
        print("\n✗ Some enhancements failed. Please check error messages above.")
    
    print("="*70)


if __name__ == '__main__':
    main()
