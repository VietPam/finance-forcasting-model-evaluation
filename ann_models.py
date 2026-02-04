"""
Artificial Neural Network (ANN) models for stock forecasting
Includes LSTM architectures
"""
from datetime import datetime
import json
import config, os, pickle
import numpy as np

from data_preprocess import DataPreprocessor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ANNModel:
    """Base class for ANN models"""
    def __init__(self, input_shape, model_name='default_ann'):
        """
        Initialize the ANN model
        :param input_shape: Shape of the input data
        :param model_type: Type of ANN model ('dense' or 'lstm')
        """

        self.input_shape = input_shape
        self.model_name = model_name

        self.model = None
        self.history = None
        self.config = config.MODELS[model_name.upper() + '_CONFIG']

    def build_model(self):
        """
        Build the Neural Network model based on the configuration
        """

        print(f"Building {self.model_name} model with input shape {self.input_shape}")

        model = keras.Sequential()

        model.add(keras.Input(shape=self.input_shape))

        for layer_config in self.config['architecture']:
            layer_type = layer_config['type']

            if layer_type == 'dense':
                model.add(layers.Dense(
                    layer_config['units'],
                    activation=layer_config['activation']
                ))
            
            elif layer_type == 'lstm':
                model.add(layers.LSTM(
                    layer_config['units'],
                    return_sequences=layer_config.get('return_sequences', False)
                ))

            elif layer_type == 'dropout':
                model.add(layers.Dropout(rate=layer_config['rate']))
        
        # Compile the model
        model.compile(
            optimizer = self.config['optimizer'],
            loss = self.config['loss'],
            metrics = self.config['metrics']
        )

        self.model = model
        print(f"Model ({self.model_name}) built successfully.")
    
    def train(self, X_train, y_train, X_val, y_val, verbose=1):
        """
        Train the model
        
        :param X_train: Training features
        :param y_train: Training targets
        :param X_val: Validation features
        :param y_val: Validation targets
        :param verbose: Verbosity level
        :return: Training history
        """

        print(f"Training {self.model_name} model...")

        # Callbacks
        callbacks = []

        # Early Stopping

        if 'early_stopping' in self.config:
            es_config = self.config['early_stopping']
            early_stopping = EarlyStopping(
                monitor=es_config['monitor'],
                patience=es_config['patience'],
                restore_best_weights=es_config['restore_best_weights'],
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Model Checkpoint
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        checkpoint_path = os.path.join(config.MODEL_DIR, f"{self.model_name}_best_model.keras")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=verbose
        )

        print(f"Training of {self.model_name} model completed.")
        return self.history
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        :param X: Input features
        :return: Predicted values
        """
        predictions = self.model.predict(X)
        return predictions

    def load_model(self, file_path=config.MODEL_DIR):
        """
        Load a pre-trained model from file
        
        :param file_path: Path to the saved model file
        """
        self.model = keras.models.load_model(file_path)
        print(f"Model loaded: {self.model.summary()}")

    def inverse_transform_targets(self, y, scaler):
        """
        Inverse transform scaled targets back to original scale
        
        :param y: Scaled target values (1D, 2D, or 3D array)
        :param scaler: Fitted scaler object
        :return: Inverse transformed targets in original scale
        """
        y_original_shape = y.shape

        if len(y_original_shape) == 1:
            y_2d = y.reshape(-1, 1)
            y_inverted = scaler.inverse_transform(y_2d)
            return y_inverted.reshape(-1)
        
        elif len(y_original_shape) == 2:
            return scaler.inverse_transform(y)
        
        elif len(y_original_shape) == 3:
            # Reshape (samples, timesteps, features) -> (samples, timesteps * features)
            y_2d = y.reshape(y_original_shape[0], -1)
            y_inverted = scaler.inverse_transform(y_2d)
            return y_inverted.reshape(y_original_shape)
        
        else:
            raise ValueError(f"Unsupported shape: {y_original_shape}. Expected 1D, 2D, or 3D array")
        
if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(1)
    np.set_printoptions(suppress=True)
    
    # Load preprocessed data
    with open(f"{config.PROCESSED_DATA_DIR}/{config.TICKER}_implicit_sequence.pkl", "rb") as f:
        implicit_sequence_data = pickle.load(f)
        X_train_imp = implicit_sequence_data['X_train']
        X_val_imp = implicit_sequence_data['X_val']
        X_test_imp = implicit_sequence_data['X_test']
        y_train_imp = implicit_sequence_data['y_train'].reshape(implicit_sequence_data['y_train'].shape[0], -1)
        y_val_imp = implicit_sequence_data['y_val'].reshape(implicit_sequence_data['y_val'].shape[0], -1)
        y_test_imp = implicit_sequence_data['y_test'].reshape(implicit_sequence_data['y_test'].shape[0], -1)
        scaler_features_imp = implicit_sequence_data['scaler_features']
        scaler_targets_imp = implicit_sequence_data['scaler_targets']
        print(f"Implicit sequence data loaded. X_train shape: {X_train_imp.shape}, y_train shape: {y_train_imp.shape}")


    with open(f"{config.PROCESSED_DATA_DIR}/{config.TICKER}_explicit_sequence.pkl", "rb") as f:
        explicit_sequence_data = pickle.load(f)
        X_train_exp = explicit_sequence_data['X_train']
        X_val_exp = explicit_sequence_data['X_val']
        X_test_exp = explicit_sequence_data['X_test']
        y_train_exp = explicit_sequence_data['y_train'].reshape(explicit_sequence_data['y_train'].shape[0], -1)
        y_val_exp = explicit_sequence_data['y_val'].reshape(explicit_sequence_data['y_val'].shape[0], -1)
        y_test_exp = explicit_sequence_data['y_test'].reshape(explicit_sequence_data['y_test'].shape[0], -1)
        scaler_features_exp = explicit_sequence_data['scaler_features']
        scaler_targets_exp = explicit_sequence_data['scaler_targets']
        print(f"Explicit sequence data loaded. X_train shape: {X_train_exp.shape}, y_train shape: {y_train_exp.shape}")

    for model_name in [name.replace('_CONFIG', '') for name in config.MODELS.keys()]:
        if model_name in ['DEFAULT_ANN']:
            input_shape = X_train_imp.shape[1:]
            model = ANNModel(input_shape=input_shape, model_name=model_name)
            model.build_model()
            model.model.summary()
            model.train(
                X_train=X_train_imp,
                y_train=y_train_imp,
                X_val=X_val_imp,
                y_val=y_val_imp,
                verbose=1
            )
            y_pred_scaled = model.predict(X_test_imp)
            y_pred = model.inverse_transform_targets(y_pred_scaled, scaler_targets_imp)
            y_true = model.inverse_transform_targets(y_test_imp, scaler_targets_imp)
            evaluate_data = {
                'y_pred': y_pred,
                'y_true': y_true,
                'model_name': model_name,
                'model': config.MODELS[model_name+'_CONFIG'],
                'metadata': {
                    'model': config.MODELS[model_name+'_CONFIG'],
                    'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
           
        elif model_name in ['LSTM']:
            input_shape = X_train_exp.shape[1:]
            model = ANNModel(input_shape=input_shape, model_name=model_name)
            model.build_model()
            print(model.model.summary())
            model.train(
                X_train=X_train_exp,
                y_train=y_train_exp,
                X_val=X_val_exp,
                y_val=y_val_exp,
                verbose=1
            )
            y_pred_scaled = model.predict(X_test_exp)
            y_pred = model.inverse_transform_targets(y_pred_scaled, scaler_targets_exp)
            y_true = model.inverse_transform_targets(y_test_exp, scaler_targets_exp)
            evaluate_data = {
                'y_pred': y_pred,
                'y_true': y_true,
                'model_name': model_name,
                'model': config.MODELS[model_name+'_CONFIG'],
                'metadata': {
                    'model': config.MODELS[model_name+'_CONFIG'],
                    'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        
        os.makedirs(config.PREDICT_DIR, exist_ok=True)
        with open(f"{config.PREDICT_DIR}/{model_name}_evaluate_data.pkl", "wb") as f:
            pickle.dump(evaluate_data, f)
        with open(f"{config.PREDICT_DIR}/{model_name}_evaluate_data_summary.txt", "w") as f:
            f.write("Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Processed on: {evaluate_data['metadata']['processed_date']}\n")
            f.write(f"Model config: {json.dumps(evaluate_data['metadata']['model'], indent=4)}\n")
