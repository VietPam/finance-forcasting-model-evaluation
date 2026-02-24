"""
Configuration for Backtesting Engine

Contains parameters for backtesting simulation including
transaction costs, initial capital, and strategy parameters.
"""

# Portfolio Configuration
BACKTEST_CONFIG = {
    # Initial capital in VND (100 million)
    'initial_capital': 100_000_000,
    
    # Transaction costs (Vietnam stock market)
    'transaction_cost': 0.0015,  # 0.15% brokerage fee
    'slippage': 0.001,            # 0.1% slippage assumption
    
    # Risk-free rate for Sharpe ratio (Vietnam government bonds)
    'risk_free_rate': 0.05,  # 5% annual
    
    # Trading parameters
    'min_trade_size': 100,        # Minimum shares per trade
    'max_position_size': 1.0,     # Maximum position as fraction of capital (1.0 = 100%)
}

# Strategy-specific parameters
STRATEGY_PARAMS = {
    'directional': {
        'description': 'Buy if predicted > current, sell if predicted < current',
    },
    
    'threshold': {
        'description': 'Trade only if predicted return exceeds threshold',
        'min_return_threshold': 0.01,  # 1% minimum predicted return
        'stop_loss': -0.05,             # 5% stop loss
    },
    
    'multi_step': {
        'description': 'Use all 3 predictions (t+1, t+2, t+3) for consensus',
        'min_agreement': 2,  # At least 2 out of 3 predictions must agree
        'weights': [0.5, 0.3, 0.2],  # Weight t+1 more than t+2, t+3
    },
    
    'conservative': {
        'description': 'Risk-averse strategy with strict entry criteria',
        'min_return_threshold': 0.02,  # 2% minimum return
        'require_all_agree': True,      # All 3 predictions must agree
        'max_drawdown_limit': 0.15,     # Stop trading if drawdown > 15%
    },
}

# Visualization settings
PLOT_CONFIG = {
    'figsize': (14, 8),
    'style': 'seaborn-v0_8-darkgrid',
    'colors': {
        'portfolio': '#2E86AB',
        'benchmark': '#A23B72',
        'buy_marker': '^',
        'sell_marker': 'v',
        'buy_color': 'green',
        'sell_color': 'red',
    }
}

# Performance metrics to calculate
METRICS = [
    'total_return',
    'annualized_return',
    'volatility',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'win_rate',
    'profit_factor',
    'total_trades',
    'avg_trade_duration',
]
