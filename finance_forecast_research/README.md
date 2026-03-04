# Finance Forecast Research Package

This package contains all model training, evaluation, and research code for stock price forecasting using Artificial Neural Networks (ANN) and LSTM models.

## 📦 Package Structure

```
finance_forecast_research/
├── __init__.py              # Package initialization and exports
├── config.py                # Central configuration for all parameters
├── data_preprocess.py       # Data loading and feature engineering
├── ann_models.py            # ANN/LSTM model architectures and training
├── evaluation.py            # Model evaluation and comparison
├── predictions/             # Model prediction outputs
│   ├── DEFAULT_ANN_evaluate_data.pkl
│   ├── DEFAULT_ANN_evaluate_data_enhanced.pkl
│   ├── DEFAULT_ANN_evaluate_data_summary.txt
│   ├── LSTM_evaluate_data.pkl
│   ├── LSTM_evaluate_data_enhanced.pkl
│   └── LSTM_evaluate_data_summary.txt
└── results/                 # Evaluation results and visualizations
    ├── model_comparison.csv
    ├── predictions_comparison.png
    ├── error_distribution.png
    ├── metrics_comparison.png
    └── scatter_comparison.png
```

## 🎯 Purpose

This package is a self-contained research module that:
- **Preprocesses** stock market data with 24 technical indicators
- **Trains** DEFAULT_ANN and LSTM models for multi-step price prediction
- **Evaluates** model performance using multiple metrics
- **Stores** predictions and results for analysis

The package is designed to be imported by other components (e.g., backtesting engine, web applications) that need access to preprocessing logic or trained models.

## 🚀 Usage

### Importing the Package

From the project root, you can import components:

```python
# Import configuration
from finance_forecast_research import config

# Import main classes
from finance_forecast_research.data_preprocess import DataPreprocessor
from finance_forecast_research.ann_models import ANNModel
from finance_forecast_research.evaluation import ModelEvaluator
```

### Example: Preprocessing Data

```python
from finance_forecast_research.data_preprocess import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    scaler_type='minmax',
    data_path='../data/VN30_Dataset_2015_2026.csv'
)

# Load and preprocess ACB stock data
X_train, X_test, y_train, y_test = preprocessor.prepare_implicit_sequence_data(
    ticker='ACB',
    save=True
)
```

### Example: Training a Model

```python
from finance_forecast_research.ann_models import ANNModel
from finance_forecast_research import config

# Initialize model
model = ANNModel(model_name='LSTM')

# Train the model
history = model.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=config.MODELS['LSTM_CONFIG']['epochs'],
    batch_size=config.MODELS['LSTM_CONFIG']['batch_size']
)

# Save the best model
model.save_model()
```

### Example: Evaluating Models

```python
from finance_forecast_research.evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Load prediction data
data_lstm = evaluator.load_data(file_name='LSTM_evaluate_data.pkl')
data_ann = evaluator.load_data(file_name='DEFAULT_ANN_evaluate_data.pkl')

# Compare models
evaluator.compare_models([data_lstm, data_ann])
```

### Example: Using from Backtest Engine

```python
# In finance_forecast_backtest_engine/data_loader.py
from finance_forecast_research.data_preprocess import DataPreprocessor
from finance_forecast_research import config

def prepare_features_for_date(df, target_date):
    """Prepare features using the same logic as training"""
    preprocessor = DataPreprocessor()
    # Use preprocessor methods to calculate technical indicators
    # and apply the same scaler transformation
    ...
```

## 📊 Data Flow

```
Raw Data (data/VN30_Dataset_2015_2026.csv)
    ↓
DataPreprocessor (data_preprocess.py)
    ├─→ Calculate 24 technical indicators
    ├─→ Create 30-day sequences
    └─→ Scale features with MinMaxScaler
    ↓
Processed Data (data/processed_data/)
    ↓
ANNModel (ann_models.py)
    ├─→ Train DEFAULT_ANN
    └─→ Train LSTM
    ↓
Trained Models (models/)
    ↓
ModelEvaluator (evaluation.py)
    ├─→ Generate predictions
    ├─→ Calculate metrics
    └─→ Create visualizations
    ↓
Results (predictions/ & results/)
```

## 🔧 Configuration

All parameters are centralized in `config.py`:

- **Data**: Ticker, train/test split, sequence length
- **Features**: 24 technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **Models**: Architecture definitions for DEFAULT_ANN and LSTM
- **Training**: Batch size, epochs, learning rate, early stopping
- **Paths**: Data directory, model directory, output directories

### Key Configuration Variables

```python
config.TICKER = 'ACB'                    # Stock ticker to analyze
config.DATA_PATH                         # Path to CSV dataset
config.SEQUENCE_LENGTH = 30              # Historical window size
config.PREDICTION_STEPS = 3              # Predict t+1, t+2, t+3
config.MODEL_DIR                         # Where trained models are saved
config.PREDICT_DIR                       # Where predictions are saved
config.EVALUATE_DIR                      # Where results are saved
```

## 📈 Features Calculated

The `DataPreprocessor` calculates 24 technical indicators for each trading day:

1. **Moving Averages**: SMA (5, 10, 20), EMA (5, 10, 20)
2. **Momentum**: RSI (14), Stochastic Oscillator, ROC
3. **Trend**: MACD, ADX
4. **Volatility**: ATR, Bollinger Bands
5. **Volume**: OBV, Volume MA
6. **Price**: Open, High, Low, Close

All features are scaled using MinMaxScaler before training.

## 🎓 Model Architectures

### DEFAULT_ANN
- Flattened input: (batch, 720) where 720 = 30 days × 24 features
- Dense layers: [128, 64, 32] with ReLU activation and dropout
- Output: 3 values (t+1, t+2, t+3 closing prices)

### LSTM
- 3D input: (batch, 30, 24) for sequence learning
- LSTM layers: [64, 32] with dropout
- Dense output layer: 3 values (t+1, t+2, t+3 closing prices)

Both models use:
- **Loss**: Huber loss (robust to outliers)
- **Optimizer**: Adam with learning rate 0.001
- **Early Stopping**: Patience of 10 epochs

## 📊 Evaluation Metrics

The `ModelEvaluator` calculates:

- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Error as percentage
- **R² Score**: Explained variance (0-1, higher is better)
- **Directional Accuracy**: % of correct UP/DOWN predictions

Results are saved in `results/model_comparison.csv` and visualized in PNG files.

## 🗂️ Output Files

### Predictions Directory
- `{MODEL}_evaluate_data.pkl`: Raw prediction arrays and metrics
- `{MODEL}_evaluate_data_enhanced.pkl`: Enhanced data with dates and analysis
- `{MODEL}_evaluate_data_summary.txt`: Human-readable summary

### Results Directory
- `model_comparison.csv`: Side-by-side model metrics
- `predictions_comparison.png`: Actual vs predicted prices over time
- `error_distribution.png`: Histogram of prediction errors
- `metrics_comparison.png`: Bar chart comparing model performance
- `scatter_comparison.png`: Scatter plots of predictions vs actuals

## 🔗 Integration with Other Components

### Backtest Engine
The backtest engine imports preprocessing logic to ensure consistency:
```python
from finance_forecast_research.data_preprocess import DataPreprocessor
from finance_forecast_research import config
```

### Web Application
The web app can load trained models and use the evaluator:
```python
from finance_forecast_research.ann_models import ANNModel
from finance_forecast_research.evaluation import ModelEvaluator
```

## 📝 Notes

- All file paths in `config.py` are relative to the `finance_forecast_research/` directory
- Data and models directories remain at project root for sharing across components
- Predictions and results are stored within this package for data locality
- The package uses absolute imports: `from finance_forecast_research import ...`

## 🛠️ Development

To add new models or features:
1. Update `config.py` with new parameters
2. Extend `DataPreprocessor` for new indicators
3. Add model architecture to `ANNModel`
4. Update `ModelEvaluator` for new metrics
5. Update this README with changes

## 📚 References

- Stock data: VN30 index components (2015-2026)
- Technical indicators: Using `ta` (Technical Analysis Library)
- Deep learning: TensorFlow/Keras
- Evaluation: scikit-learn metrics
