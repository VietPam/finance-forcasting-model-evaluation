# Financial Forecasting Model Evaluation Framework

A comprehensive deep learning framework for multi-step stock price forecasting and model comparison, focusing on Vietnamese stock market (VN30) data. This project implements and evaluates different neural network architectures for predicting stock closing prices up to 3 days ahead.

---

## 🎯 The Big Picture: Financial Forecasting Research

### Overview of Financial Forecasting

Financial forecasting, particularly stock price prediction, remains one of the most challenging problems in quantitative finance and machine learning. The efficient market hypothesis suggests that stock prices follow a random walk, making accurate prediction extremely difficult. However, recent advances in deep learning have shown promising results in capturing non-linear patterns and temporal dependencies in financial time series data.

### Popular Approaches in Financial Forecasting

**1. Traditional Statistical Methods**
- ARIMA (AutoRegressive Integrated Moving Average)
- GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)
- Vector Autoregression (VAR)

**2. Machine Learning Approaches**
- Support Vector Machines (SVM)
- Random Forests and Gradient Boosting (XGBoost, LightGBM)
- Traditional Neural Networks (ANNs)

**3. Deep Learning Methods**
- **LSTM (Long Short-Term Memory)**: Captures long-term dependencies in sequential data
- **GRU (Gated Recurrent Units)**: Simplified variant of LSTM
- **CNN**: Convolutional networks for pattern recognition in time series
- **Transformer Models**: Attention-based architectures for sequence modeling
- **Hybrid Architectures**: Combining multiple approaches

### Where This Project Fits

This project addresses a critical gap in financial forecasting research: **systematic comparison of deep learning architectures under identical data conditions**. Many studies compare models using different datasets, features, or preprocessing steps, making it difficult to draw meaningful conclusions about model performance.

**Key Contributions:**
- **Controlled Experimental Design**: All models use identical input features, preprocessing pipeline, and data splits
- **Multi-Step Forecasting**: Predicts closing prices for t+1, t+2, and t+3 days ahead (not just next-day prediction)
- **Comprehensive Technical Analysis**: Incorporates 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, OBV)
- **Dual Sequence Formats**: Supports both explicit sequences (3D for LSTM) and implicit sequences (2D for feedforward networks)
- **Standardized Evaluation**: Uses multiple metrics (MAE, MSE, RMSE, MAPE, R², Directional Accuracy)

**Research Focus:**
The project specifically compares:
- **Feedforward Neural Networks (Default ANN)**: Tests whether simple dense architectures can perform competitively with minimal complexity
- **LSTM Networks**: Evaluates the benefit of explicit temporal modeling for financial time series

This controlled comparison helps answer: *"Does the added complexity of recurrent architectures justify their use for stock price prediction?"*

---

## 📊 Current Stage of the Project

### Implemented Features

✅ **Data Processing Pipeline**
- Automated data loading and cleaning for VN30 dataset (2015-2026)
- Feature engineering with 20+ technical indicators
- Support for multiple scaling strategies (MinMax, Standard, Robust)
- Sequence creation for both explicit (3D) and implicit (2D) formats
- Train/validation/test splitting with temporal ordering preserved

✅ **Model Architectures**
- **Default ANN**: 3-layer feedforward network (128→64→32 neurons) with dropout regularization
- **LSTM**: 2-layer LSTM network (400→200 units) with dense output layer
- Configurable architecture through `config.py`
- Early stopping and model checkpointing

✅ **Training Infrastructure**
- TensorFlow/Keras implementation
- Adam optimizer with MSE loss
- Validation-based early stopping (patience=10)
- Automatic saving of best models

✅ **Evaluation Framework**
- Multi-output evaluation (separate metrics for t+1, t+2, t+3)
- Comprehensive metrics: MAE, MSE, RMSE, MAPE, R², Directional Accuracy
- Automated comparison table generation
- Results serialization (CSV + text summaries)

### Current Results

Based on ACB stock (Asia Commercial Bank) predictions:

| Model       | MAE   | RMSE  | R²    | Directional Accuracy |
|-------------|-------|-------|-------|---------------------|
| Default ANN | 1.007 | 1.269 | 0.784 | 44.84%              |
| LSTM        | 1.061 | 1.309 | 0.788 | 43.42%              |

**Key Observations:**
- Both models show comparable performance
- Default ANN is slightly better in MAE and directional accuracy
- LSTM shows marginally better R² score
- Both models perform worse on longer horizons (t+3 predictions)

### Limitations & Future Work

⚠️ **Current Limitations:**
- Single stock evaluation (ACB only)
- Limited hyperparameter tuning
- No ensemble methods implemented
- Missing SVM baseline (configured but not trained)
- No cross-validation across different stocks
- Limited feature selection analysis

🔮 **Planned Enhancements:**
- Expand to full VN30 basket (30 stocks)
- Implement SVM and XGBoost baselines
- Add GRU and Transformer architectures
- Feature importance analysis
- Hyperparameter optimization (Optuna/Keras Tuner)
- Ensemble strategies (stacking, voting)
- Walk-forward validation
- Risk-adjusted metrics (Sharpe ratio, maximum drawdown)

---

## 🚀 How to Use This Project

### Prerequisites

**Required Python Packages:**
```
numpy
pandas
scikit-learn
tensorflow (>= 2.0)
ta (technical analysis library)
matplotlib
```

**Installation:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install numpy pandas scikit-learn tensorflow ta matplotlib
```

### Project Structure

```
finance-forcasting-model-evaluation/
│
├── config.py                    # Central configuration file
├── data_preprocess.py          # Data loading and preprocessing
├── ann_models.py               # Neural network model definitions
├── evaluation.py               # Model evaluation and comparison
│
├── data/
│   ├── VN30_Dataset_2015_2026.csv
│   └── processed_data/         # Cached preprocessed sequences
│       ├── ACB_explicit_sequence.pkl    # 3D sequences for LSTM
│       └── ACB_implicit_sequence.pkl    # 2D sequences for ANN
│
├── models/                     # Trained model weights
│   ├── DEFAULT_ANN_best_model.keras
│   └── LSTM_best_model.keras
│
├── predictions/                # Prediction outputs
│   ├── DEFAULT_ANN_evaluate_data.pkl
│   └── LSTM_evaluate_data.pkl
│
└── results/                    # Evaluation results
    └── model_comparison.csv
```

### Configuration (`config.py`)

The `config.py` file is the central control panel for the entire project:

**Data Configuration:**
- `TICKER`: Stock symbol to analyze (default: 'ACB')
- `DATA_PATH`: Path to CSV dataset
- `TRAIN_TEST_SPLIT`: Training data ratio (default: 0.8)
- `VALIDATION_SPLIT`: Validation split from training (default: 0.2)
- `SEQUENCE_LENGTH`: Historical window size (default: 30 days)
- `PREDICTION_STEPS`: Forecast horizon (default: 3 days)

**Technical Indicators:**
- `TECHNICAL_INDICATORS`: Dictionary defining which indicators to compute
  - SMA windows: [5, 10, 20]
  - EMA windows: [5, 10, 20]
  - RSI period: 14
  - MACD, Bollinger Bands (20-day), OBV

**Model Architectures:**
- `MODELS['DEFAULT_ANN_CONFIG']`: Feedforward network specification
- `MODELS['LSTM_CONFIG']`: LSTM network specification
- Each includes: architecture layers, optimizer, loss, metrics, training parameters

**Example: Modifying Prediction Horizon**
```python
# In config.py
PREDICTION_STEPS = 5  # Change from 3 to 5 days ahead
```

### Workflow

#### Step 1: Data Preprocessing

**Purpose:** Load raw data, engineer features, create sequences, and save for training.

**Command:**
```bash
python data_preprocess.py
```

**What it does:**
1. Loads VN30 dataset and filters by ticker
2. Cleans data (removes invalid OHLC values, missing data)
3. Adds 20+ technical indicators
4. Creates sliding window sequences (30-day windows → predict next 3 days)
5. Splits into train/validation/test sets (temporal ordering preserved)
6. Scales features and targets
7. Saves two formats:
   - **Explicit sequences** (3D: samples × timesteps × features) for LSTM
   - **Implicit sequences** (2D: samples × flattened features) for ANN

**Output files:**
```
data/processed_data/
├── ACB_explicit_sequence.pkl
├── ACB_explicit_sequence.txt   # Human-readable summary
├── ACB_implicit_sequence.pkl
└── ACB_implicit_sequence.txt
```

**Key Class: `DataPreprocessor`**
```python
from data_preprocess import DataPreprocessor

# Initialize with scaler type
preprocessor = DataPreprocessor(scaler_type='minmax')

# Load and process data
df = preprocessor.load_data(ticker='ACB')
df_clean = preprocessor.clean_data(df)
df_features = preprocessor.add_technical_indicators(df_clean)

# Create sequences
X, y = preprocessor.create_sequence_and_prepare_features(
    df_features, 
    explicit_sequence=True  # False for ANN, True for LSTM
)

# Split and scale
X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.slit_data(X, y)
X_train_scaled, y_train_scaled, ... = preprocessor.scale_data(...)
```

#### Step 2: Model Training

**Purpose:** Train neural network models and save the best weights.

**Command:**
```bash
python ann_models.py
```

**What it does:**
1. Loads preprocessed data from pickle files
2. For each model in `config.MODELS`:
   - Builds architecture based on configuration
   - Trains with early stopping (monitors validation loss)
   - Saves best model weights to `models/`
   - Makes predictions on test set
   - Saves predictions and true values to `predictions/`

**Output files:**
```
models/
├── DEFAULT_ANN_best_model.keras
└── LSTM_best_model.keras

predictions/
├── DEFAULT_ANN_evaluate_data.pkl
├── DEFAULT_ANN_evaluate_data_summary.txt
├── LSTM_evaluate_data.pkl
└── LSTM_evaluate_data_summary.txt
```

**Key Class: `ANNModel`**
```python
from ann_models import ANNModel

# Initialize model
model = ANNModel(input_shape=(30, 35), model_name='LSTM')

# Build from config
model.build_model()

# Train
history = model.train(X_train, y_train, X_val, y_val)

# Predict
predictions = model.predict(X_test)

# Inverse transform to original scale
predictions_original = model.inverse_transform_targets(predictions, scaler)
```

#### Step 3: Model Evaluation and Comparison

**Purpose:** Compute comprehensive metrics and compare all models.

**Command:**
```bash
python evaluation.py
```

**What it does:**
1. Loads prediction data from all models
2. Calculates metrics for each model:
   - Overall metrics (MAE, MSE, RMSE, MAPE, R², DA)
   - Per-output metrics (separate for t+1, t+2, t+3)
3. Creates comparison CSV table
4. Generates visualization plots (optional)

**Output files:**
```
results/
├── model_comparison.csv         # Comprehensive comparison table
└── [visualization plots]
```

**Key Class: `ModelEvaluator`**
```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Load all model predictions
evaluator.load_data(dir='predictions/')

# Evaluate each model
for model_name in evaluator.predictions.keys():
    metrics = evaluator.evaluate_model(
        model_name,
        evaluator.true_values[model_name],
        evaluator.predictions[model_name]
    )

# Compare all models
evaluator.compare_models()
```

### Running the Full Pipeline

```bash
# 1. Preprocess data (only needed once or when changing configuration)
python data_preprocess.py

# 2. Train all models
python ann_models.py

# 3. Evaluate and compare
python evaluation.py

# View results
cat results/model_comparison.csv
```

### Customization Examples

**Example 1: Train on Different Stock**
```python
# In config.py
TICKER = 'VNM'  # Change from 'ACB' to 'VNM' (Vinamilk)

# Then run pipeline
python data_preprocess.py
python ann_models.py
python evaluation.py
```

**Example 2: Add New Model Architecture**
```python
# In config.py
MODELS['GRU_CONFIG'] = {
    'architecture': [
        {'type': 'gru', 'units': 300, 'return_sequences': True},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'gru', 'units': 150, 'return_sequences': False},
        {'type': 'dense', 'units': 3, 'activation': 'linear'}
    ],
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    'epochs': 100,
    'batch_size': 32,
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'restore_best_weights': True
    }
}

# Update ann_models.py to support 'gru' layer type in build_model()
```

**Example 3: Change Sequence Length**
```python
# In config.py
SEQUENCE_LENGTH = 60  # Use 60 days instead of 30

# Rerun preprocessing and training
python data_preprocess.py
python ann_models.py
```

### Evaluation Metrics Explained

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values. Lower is better. Unit: same as stock price.
  
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences. Penalizes large errors more than MAE. Lower is better.

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error. Scale-independent. Lower is better.

- **R² (R-squared)**: Proportion of variance explained by the model. Range: (-∞, 1]. Higher is better (1 = perfect fit).

- **Directional Accuracy (DA)**: Percentage of correct direction predictions (up/down). Range: [0, 100]. Higher is better. Around 50% = random guessing.

### Troubleshooting

**Issue: Out of Memory during training**
```python
# In config.py, reduce batch size
'batch_size': 16  # Instead of 32
```

**Issue: Missing data file**
```bash
# Ensure VN30 dataset exists
ls data/VN30_Dataset_2015_2026.csv

# If missing, place your CSV file in data/ folder
```

**Issue: NaN values in predictions**
```python
# Check if test set has sufficient data
# Ensure SEQUENCE_LENGTH + PREDICTION_STEPS doesn't exceed dataset length
```

---

## 📝 Citation

If you use this code for academic research, please cite:

```
Financial Forecasting Model Evaluation Framework
VN30 Stock Market Analysis (2015-2026)
GitHub: finance-forcasting-model-evaluation
```

---

## 📧 Contact

For questions or collaborations, please open an issue in the repository.

---

## 📄 License

This project is for educational and research purposes.
