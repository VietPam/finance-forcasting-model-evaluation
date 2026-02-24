"""
Backtesting Engine for Financial Forecasting Models

This package provides tools for backtesting trading strategies using
machine learning model predictions. It includes portfolio simulation,
strategy implementations, performance metrics, and visualization tools.

Example:
    >>> from backtest_engine import BacktestEngine, DirectionalStrategy
    >>> engine = BacktestEngine(initial_capital=100_000_000)
    >>> results = engine.run_backtest('DEFAULT_ANN', DirectionalStrategy())
    >>> print(results.sharpe_ratio)
"""

from .engine import BacktestEngine
from .strategies import (
    BaseStrategy,
    DirectionalStrategy,
    ThresholdStrategy,
    MultiStepStrategy,
    ConservativeStrategy
)
from .metrics import PerformanceReport, calculate_sharpe_ratio, calculate_max_drawdown
# Lazy import for visualizer to avoid matplotlib dependency issues
# from .visualizer import BacktestVisualizer
from .utils import load_enhanced_predictions, reconstruct_test_dates

__version__ = '0.1.0'

def get_visualizer():
    """Lazy import of visualizer to avoid matplotlib dependency issues."""
    from .visualizer import BacktestVisualizer
    return BacktestVisualizer

# Make BacktestVisualizer available but lazy-loaded
BacktestVisualizer = property(lambda self: get_visualizer())

__all__ = [
    'BacktestEngine',
    'BaseStrategy',
    'DirectionalStrategy',
    'ThresholdStrategy',
    'MultiStepStrategy',
    'ConservativeStrategy',
    'PerformanceReport',
    'get_visualizer',  # Use this to get visualizer
    'load_enhanced_predictions',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'reconstruct_test_dates',
]
