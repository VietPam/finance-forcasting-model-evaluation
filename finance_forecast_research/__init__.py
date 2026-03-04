"""
Finance Forecast Research Package

This package contains all model training, evaluation, and research code
for the stock price forecasting project.

Main Components:
- DataPreprocessor: Load and preprocess stock data with technical indicators
- ANNModel: Train and evaluate ANN/LSTM models
- ModelEvaluator: Compare and analyze model performance
- config: Central configuration for all research parameters

Usage:
    from finance_forecast_research import config
    from finance_forecast_research.data_preprocess import DataPreprocessor
    from finance_forecast_research.ann_models import ANNModel
    from finance_forecast_research.evaluation import ModelEvaluator
"""

__version__ = "1.0.0"

# Import config module to make it directly accessible
from finance_forecast_research import config

# Import main classes for convenience
from finance_forecast_research.data_preprocess import DataPreprocessor
from finance_forecast_research.ann_models import ANNModel
from finance_forecast_research.evaluation import ModelEvaluator

__all__ = [
    'config',
    'DataPreprocessor',
    'ANNModel',
    'ModelEvaluator',
]
