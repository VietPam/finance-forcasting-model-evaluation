"""
Data preprocessing module for stock forecasting
Handles data loading, cleaning, feature engineering, and preparation for implicit and explicit sequence models
"""
import numpy as np
import pandas as pd
import os, pickle, ta
from datetime import datetime
import config

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, scaler_type=None, data_path='data/VN30_Dataset_2015_2026.csv'):
        """
        Initialize the DataPreprocessor with optional data path.
        
        :param data_path: Path to the dataset CSV file. If None, uses default from config.
        """
        self.data_path = data_path or config.DATA_PATH

        if scaler_type == 'standard':
            self.scaler_features = StandardScaler()
            self.scaler_targets = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler_features = RobustScaler()
            self.scaler_targets = RobustScaler()
        else:
            self.scaler_features = MinMaxScaler()
            self.scaler_targets = MinMaxScaler()

        self.data = None
        self.features = None
        #self.targets = None

    def load_data(self, ticker=None):
        """
        Load data from CSV file.
        :param ticker: Optional ticker symbol to filter the data.
        """
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data.sort_values(by='time')
        
        if ticker:
            self.data = self.data[self.data['Ticker'] == ticker].copy()
            print(f"Filtered data for ticker: {ticker}")
        
        print(f"Loaded {len(self.data)} rows of data")
        return self.data
    
    def clean_data(self, df):
        """
        Clean the data by handling missing values.
        :param df: DataFrame to clean.
        :return: Cleaned DataFrame.
        """

        print("Cleaning data...")
        initial_len = len(df)

        # Remove rows with missing values in critical columns
        critial_cols = ['open', 'close', 'high', 'low', 'volume']
        df = df.dropna(subset=critial_cols)

        # Remove rows with invalid OHLC values
        df = df[df['high'] >= df['low']]

        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]
        df = df[(df['open'] >= df['low']) & (df['open'] <= df['high'])]

        # Reset index
        final_len = len(df)
        print(f"Removed {initial_len - final_len} rows with missing values")

        return df
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame.
        
        :param df: DataFrame to which technical indicators will be added.
        """
        print("Adding technical indicators...")
        df = df.copy()
        
        # Simple Moving Averages
        for window in config.TECHNICAL_INDICATORS['SMA']:
            df[f'SMA_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
        
        # Exponential Moving Averages
        for window in config.TECHNICAL_INDICATORS['EMA']:
            df[f'EMA_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=config.TECHNICAL_INDICATORS['RSI'])
        
        # MACD
        if config.TECHNICAL_INDICATORS['MACD']:
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=config.TECHNICAL_INDICATORS['BB'])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_width'] = bb.bollinger_wband()
        
        # On-Balance Volume
        if config.TECHNICAL_INDICATORS['OBV']:
            df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=10).std()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        print(f"Added {df.shape[1] - 7} technical indicators")
        return df
    
    def create_sequence_and_prepare_features(self, df, explicit_sequence=False):
        """
        Create sequences of data for time series modeling.
        
        :param df: Timeseries DataFrame with features
        :param explicit_sequence: If True, flatten sequences to 2D for non-sequential models.
                                  If False, keep 3D shape (samples, timesteps, features) for LSTM/RNN.
        :return: Tuple of (X_seq, y_seq) as numpy arrays
                 - X_seq: Features array, shape depends on explicit_sequence parameter
                 - y_seq: Targets array with shape (samples, predict_steps, num_predict_columns)
        """
        sequence_length = config.SEQUENCE_LENGTH
        predict_steps = config.PREDICTION_STEPS
        predict_columns = config.PREDICT_COLUMNS

        # Prepare feature columns
        exclude_cols = ['time', 'Ticker']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Convert to numpy arrays for efficient slicing
        features_array = df[feature_columns].values
        targets_array = df[predict_columns].values
        
        # Pre-allocate arrays for better performance
        num_sequences = len(df) - sequence_length - predict_steps + 1
        num_features = len(feature_columns)
        num_targets = len(predict_columns)
        
        X_seq = np.zeros((num_sequences, sequence_length, num_features))
        y_seq = np.zeros((num_sequences, predict_steps, num_targets))
        
        for i in range(num_sequences):
            # Use past sequence_length days as features
            X_seq[i] = features_array[i:i + sequence_length]
            
            # Predict the next predict_steps days
            y_seq[i] = targets_array[i + sequence_length:i + sequence_length + predict_steps]

        # Flatten sequences for non-sequential models (e.g., ANN, SVM, XGBoost)
        if not explicit_sequence:
            X_seq = X_seq.reshape((X_seq.shape[0], -1))
        
        return X_seq, y_seq

    def slit_data(self, X, y, test_size=None, validation_split=None):
        """
        Docstring for slit_data
        
        :param self: Description
        :param X: Description
        :param y: Description
        :param test_size: Description
        :param validation_split: Description
        """
        if test_size is None:
            test_size = 1 - config.TRAIN_TEST_SPLIT
        if validation_split is None:
            validation_split = config.VALIDATION_SPLIT
        
        # Split into train+val and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Split train+val into train and val sets
        X_train, X_val,y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_split, shuffle=False
        )

        print(f"Data split into train ({len(X_train)}), validation ({len(X_val)}), test ({len(X_test)})")
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def scale_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Scale features and targets using the specified scalers.
        
        :param X_train: Training features
        :param y_train: Training targets
        :param X_val: Validation features
        :param y_val: Validation targets
        :param X_test: Test features
        :param y_test: Test targets
        :return: Scaled datasets
        """
        
        # Scale Features
        X_train_shape = X_train.shape
        X_val_shape = X_val.shape
        X_test_shape = X_test.shape

        if len(X_train_shape) == 3:
            X_train_2d = X_train.reshape(-1, X_train_shape[2])
            X_val_2d = X_val.reshape(-1, X_val_shape[2])
            X_test_2d = X_test.reshape(-1, X_test_shape[2])
            # X_train_2d = X_train.reshape(X_train_shape[0], -1)
            # X_val_2d = X_val.reshape(X_val_shape[0], -1)
            # X_test_2d = X_test.reshape(X_test_shape[0], -1)
        else:
            X_train_2d = X_train
            X_val_2d = X_val
            X_test_2d = X_test
        
        self.scaler_features.fit(X_train_2d)
        X_train_scaled = self.scaler_features.transform(X_train_2d)
        X_val_scaled = self.scaler_features.transform(X_val_2d)
        X_test_scaled = self.scaler_features.transform(X_test_2d)

        if len(X_train_shape) == 3: 
            X_train_scaled = X_train_scaled.reshape(X_train_shape)
            X_val_scaled = X_val_scaled.reshape(X_val_shape)
            X_test_scaled = X_test_scaled.reshape(X_test_shape)
        
        # Scale Targets
        y_train_shape = y_train.shape
        y_val_shape = y_val.shape
        y_test_shape = y_test.shape

        if len(y_train_shape) == 1:
            y_train_2d = y_train.reshape(-1, 1)
            y_val_2d = y_val.reshape(-1, 1)
            y_test_2d = y_test.reshape(-1, 1)
        elif len(y_train_shape) == 3:
            # Reshape (samples, timesteps, features) -> (samples, timesteps * features)
            y_train_2d = y_train.reshape(y_train_shape[0], -1)
            y_val_2d = y_val.reshape(y_val_shape[0], -1)
            y_test_2d = y_test.reshape(y_test_shape[0], -1)
        else:
            y_train_2d = y_train
            y_val_2d = y_val
            y_test_2d = y_test
        
        self.scaler_targets.fit(y_train_2d)
        y_train_scaled = self.scaler_targets.transform(y_train_2d)
        y_val_scaled = self.scaler_targets.transform(y_val_2d)
        y_test_scaled = self.scaler_targets.transform(y_test_2d)

        if len(y_train_shape) == 1:
            y_train_scaled = y_train_scaled.reshape(-1)
            y_val_scaled = y_val_scaled.reshape(-1)
            y_test_scaled = y_test_scaled.reshape(-1)
        elif len(y_train_shape) == 3:
            y_train_scaled = y_train_scaled.reshape(y_train_shape)
            y_val_scaled = y_val_scaled.reshape(y_val_shape)
            y_test_scaled = y_test_scaled.reshape(y_test_shape)

        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled
    
    def inverse_transform_targets(self, y):
        """
        Inverse transform scaled targets back to original scale
        
        :param y: Scaled target values (1D, 2D, or 3D array)
        :return: Inverse transformed targets in original scale
        """
        y_original_shape = y.shape

        if len(y_original_shape) == 1:
            y_2d = y.reshape(-1, 1)
            y_inverted = self.scaler_targets.inverse_transform(y_2d)
            return y_inverted.reshape(-1)
        
        elif len(y_original_shape) == 2:
            return self.scaler_targets.inverse_transform(y)
        
        elif len(y_original_shape) == 3:
            # Reshape (samples, timesteps, features) -> (samples, timesteps * features)
            y_2d = y.reshape(y_original_shape[0], -1)
            y_inverted = self.scaler_targets.inverse_transform(y_2d)
            return y_inverted.reshape(y_original_shape)
        
        else:
            raise ValueError(f"Unsupported shape: {y_original_shape}. Expected 1D, 2D, or 3D array")
        
    
    def preprocess_data(self, ticker=None, explicit_sequence=False):
        """
        Full preprocessing pipeline: load, clean, add indicators, create sequences, split, and scale.
        
        :param ticker: Optional ticker symbol to filter the data.
        :return: Scaled and split datasets.
        """
        print(ticker)
        df = self.load_data(ticker)
        df = self.clean_data(df)
        df = self.add_technical_indicators(df)
        X, y = self.create_sequence_and_prepare_features(df, explicit_sequence=explicit_sequence)
        X_train, y_train, X_val, y_val, X_test, y_test = self.slit_data(X, y)

        print(X_train)
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = self.scale_data(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        processed_data = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
            'y_test': y_test_scaled,
            'scaler_features': self.scaler_features,
            'scaler_targets': self.scaler_targets,
            'metadata': {
                'n_features': X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1],
                'n_targets': y_train.shape[1],
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        print(f"X_train shape: {X_train_scaled.shape}, y_train shape: {y_train_scaled.shape}")
        print(f"X_val shape: {X_val_scaled.shape}, y_val shape: {y_val_scaled.shape}")
        print(f"X_test shape: {X_test_scaled.shape}, y_test shape: {y_test_scaled.shape}")
        output_file = os.path.join(config.PROCESSED_DATA_DIR, f"{ticker}_explicit_sequence.pkl" if explicit_sequence else f"{ticker}_implicit_sequence.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
                    
        with open(os.path.join(config.PROCESSED_DATA_DIR, f"{ticker}_explicit_sequence.txt" if explicit_sequence else f"{ticker}_implicit_sequence.txt"), 'w') as f:
            f.write("Data Preprocessing Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Processed on: {processed_data['metadata']['processed_date']}\n")
            f.write(f"\nDataset splits:\n")
            f.write(f"  Train: {processed_data['metadata']['train_samples']} samples\n")
            f.write(f"  Validation: {processed_data['metadata']['val_samples']} samples\n")
            f.write(f"  Test: {processed_data['metadata']['test_samples']} samples\n")
            f.write(f"\nData shapes:\n")
            f.write(f"  X_train: {X_train.shape}\n")
            f.write(f"  y_train: {y_train.shape}\n")

if __name__ == '__main__':
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    np.set_printoptions(suppress=True)
        
    # Test with first ticker
    target_ticker = config.TICKER
    print(f"\nTesting with ticker: {target_ticker}")
    
    print("\n=== Prepare Data for Implicit Sequence Input Models (ANN / MLP, SVM, Linear / Logistic Regression, XGBoost, Random Forest) ===")
    implicit_sequence_data = preprocessor.preprocess_data(ticker=target_ticker, explicit_sequence=False)

    print("\n=== Prepare Data for Explicit Sequence Input Models (LSTM / GRU, TCN, Transformer, CNN-1D) ===")
    explicit_sequence_data = preprocessor.preprocess_data(ticker=target_ticker, explicit_sequence=True)