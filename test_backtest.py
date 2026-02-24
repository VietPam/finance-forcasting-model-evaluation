"""
Quick test of the backtesting engine

Tests if the backtesting infrastructure works correctly.
"""

import sys
import os

# Avoid matplotlib import issues
os.environ['MPLBACKEND'] = 'Agg'

from backtest_engine.engine import BacktestEngine
from backtest_engine.strategies import DirectionalStrategy

print("Testing Backtesting Engine...")
print("="*70)

# Initialize engine
engine = BacktestEngine(initial_capital=100_000_000)
print("✓ BacktestEngine initialized")

# Create a simple strategy
strategy = DirectionalStrategy(threshold=0.01)
print(f"✓ Strategy created: {strategy.name}")

# Run backtest for DEFAULT_ANN
try:
    print("\nRunning backtest for DEFAULT_ANN...")
    report = engine.run_backtest('DEFAULT_ANN', strategy, verbose=False)
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Model: {report.model_name}")
    print(f"Strategy: {report.strategy_name}")
    print(f"Ticker: {report.ticker}")
    print(f"\nTotal Return:        {report.total_return*100:>8.2f}%")
    print(f"Annualized Return:   {report.annualized_return*100:>8.2f}%")
    print(f"Sharpe Ratio:        {report.sharpe_ratio:>8.3f}")
    print(f"Max Drawdown:        {report.max_drawdown*100:>8.2f}%")
    print(f"Win Rate:            {report.win_rate*100:>8.2f}%")
    print(f"Profit Factor:       {report.profit_factor:>8.2f}")
    print(f"Total Trades:        {report.total_trades:>8.0f}")
    print(f"\nInitial Capital:     {report.initial_capital:>12,.0f} VND")
    print(f"Final Value:         {report.final_value:>12,.0f} VND")
    print("="*70)
    
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
