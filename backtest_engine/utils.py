"""
Utility functions for backtesting engine

Provides helper functions for data loading, date reconstruction,
and OHLC data retrieval from original dataset.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as root_config


def reconstruct_test_dates(ticker: str, 
                          test_size: int,
                          data_path: str = None) -> np.ndarray:
    """
    Reconstruct dates for test set by mapping indices back to original data.
    
    The challenge: predictions don't include dates. We need to reverse-engineer
    which dates correspond to the test set based on:
    - Total data length
    - Train/val/test split ratios
    - Sequence length and prediction steps
    
    Args:
        ticker: Stock symbol (e.g., 'ACB')
        test_size: Number of test samples (e.g., 563)
        data_path: Path to original CSV (default from config)
    
    Returns:
        Array of datetime objects corresponding to test samples
        
    Example:
        >>> reconstruct_test_dates('ACB', 563)
        >>> print(dates[0])  # 2024-01-15
        >>> print(dates[-1]) # 2026-02-20
    """
    data_path = data_path or root_config.DATA_PATH
    
    # Load and filter data
    df = pd.read_csv(data_path)
    df_ticker = df[df['Ticker'] == ticker].copy()
    df_ticker['time'] = pd.to_datetime(df_ticker['time'])
    df_ticker = df_ticker.sort_values('time').reset_index(drop=True)
    
    # Remove rows with NaN (same as preprocessing)
    df_ticker = df_ticker.dropna().reset_index(drop=True)
    
    # Calculate sequence parameters
    seq_len = root_config.SEQUENCE_LENGTH
    pred_steps = root_config.PREDICTION_STEPS
    
    # Total sequences = len(data) - seq_len - pred_steps + 1
    total_sequences = len(df_ticker) - seq_len - pred_steps + 1
    
    # Calculate split indices
    train_test_ratio = root_config.TRAIN_TEST_SPLIT
    train_val_size = int(total_sequences * train_test_ratio)
    test_start_idx = train_val_size
    
    # Extract test dates (date when prediction is made = date at end of sequence window)
    test_dates = []
    for i in range(test_start_idx, total_sequences):
        # Date at end of sequence window (when prediction is made)
        date_idx = i + seq_len - 1
        test_dates.append(df_ticker.loc[date_idx, 'time'])
    
    return np.array(test_dates)


def load_ohlc_data(ticker: str, 
                   start_date: datetime = None,
                   end_date: datetime = None,
                   data_path: str = None) -> pd.DataFrame:
    """
    Load OHLC data from original dataset for specified date range.
    
    Args:
        ticker: Stock symbol
        start_date: Start date for data retrieval (None = all)
        end_date: End date for data retrieval (None = all)
        data_path: Path to CSV file
    
    Returns:
        DataFrame with columns: [time, open, high, low, close, volume]
        
    Example:
        >>> df = load_ohlc_data('ACB', 
        ...                     datetime(2024, 1, 1), 
        ...                     datetime(2026, 2, 24))
        >>> print(df.head())
    """
    data_path = data_path or root_config.DATA_PATH
    
    df = pd.read_csv(data_path)
    df_ticker = df[df['Ticker'] == ticker].copy()
    df_ticker['time'] = pd.to_datetime(df_ticker['time'])
    df_ticker = df_ticker.sort_values('time').reset_index(drop=True)
    
    # Filter by date range if provided
    if start_date is not None or end_date is not None:
        mask = pd.Series([True] * len(df_ticker))
        if start_date is not None:
            mask &= (df_ticker['time'] >= start_date)
        if end_date is not None:
            mask &= (df_ticker['time'] <= end_date)
        df_filtered = df_ticker.loc[mask, ['time', 'open', 'high', 'low', 'close', 'volume']]
    else:
        df_filtered = df_ticker[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    return df_filtered.reset_index(drop=True)


def align_predictions_with_dates(y_pred: np.ndarray,
                                 y_true: np.ndarray,
                                 dates: np.ndarray,
                                 ticker: str) -> pd.DataFrame:
    """
    Align predictions with dates and create a comprehensive DataFrame.
    
    Args:
        y_pred: Predictions array (N, 3) for t+1, t+2, t+3
        y_true: Actual values array (N, 3)
        dates: Date array (N,)
        ticker: Stock symbol
    
    Returns:
        DataFrame with columns: [date, ticker, 
                                pred_t1, pred_t2, pred_t3,
                                actual_t1, actual_t2, actual_t3]
    
    Example:
        >>> df = align_predictions_with_dates(y_pred, y_true, dates, 'ACB')
        >>> print(df.head())
    """
    df = pd.DataFrame({
        'date': dates,
        'ticker': ticker,
        'pred_t1': y_pred[:, 0],
        'pred_t2': y_pred[:, 1],
        'pred_t3': y_pred[:, 2],
        'actual_t1': y_true[:, 0],
        'actual_t2': y_true[:, 1],
        'actual_t3': y_true[:, 2],
    })
    
    return df


def load_enhanced_predictions(model_name: str, 
                              predict_dir: str = None) -> Dict[str, Any]:
    """
    Load enhanced prediction data including dates and OHLC.
    
    Args:
        model_name: Name of model (e.g., 'DEFAULT_ANN', 'LSTM')
        predict_dir: Directory containing predictions
    
    Returns:
        Dictionary with keys: y_pred, y_true, dates, ohlc_data, metadata
        
    Example:
        >>> data = load_enhanced_predictions('DEFAULT_ANN')
        >>> print(data['dates'][0])
        >>> print(data['ohlc_data'].head())
    """
    predict_dir = predict_dir or root_config.PREDICT_DIR
    filepath = os.path.join(predict_dir, f"{model_name}_evaluate_data_enhanced.pkl")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Enhanced predictions not found: {filepath}\n"
            f"Please run data_enhancement.py first to create enhanced predictions."
        )
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Validate required keys
    required_keys = ['y_pred', 'y_true', 'dates', 'ohlc_data', 'metadata']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in enhanced predictions: {key}")
    
    return data


def calculate_trading_days(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of trading days between two dates.
    
    Assumes 252 trading days per year (Vietnam stock market).
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        Number of trading days
    """
    total_days = (end_date - start_date).days
    trading_days = int(total_days * (252 / 365))
    
    return max(trading_days, 1)  # At least 1 trading day


if __name__ == '__main__':
    # Test utility functions
    print("Testing utility functions...")
    
    # Test date reconstruction
    try:
        dates = reconstruct_test_dates('ACB', 563)
        print(f"✓ Reconstructed {len(dates)} test dates")
        print(f"  Date range: {dates[0]} to {dates[-1]}")
    except Exception as e:
        print(f"✗ Date reconstruction failed: {e}")
    
    # Test OHLC loading
    try:
        df_ohlc = load_ohlc_data('ACB', 
                                 datetime(2024, 1, 1), 
                                 datetime(2026, 2, 24))
        print(f"✓ Loaded {len(df_ohlc)} OHLC records")
        print(f"  Columns: {df_ohlc.columns.tolist()}")
    except Exception as e:
        print(f"✗ OHLC loading failed: {e}")
    
    print("\n✓ Utils test complete!")
