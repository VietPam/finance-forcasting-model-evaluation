"""
Performance Metrics for Backtesting

Calculates comprehensive performance metrics including returns,
risk-adjusted metrics, drawdown analysis, and trade statistics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine.config import BACKTEST_CONFIG


def calculate_returns(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Calculate period-to-period returns.
    
    Args:
        portfolio_values: Array of portfolio values over time
        
    Returns:
        Array of returns (length = len(portfolio_values) - 1)
    """
    if len(portfolio_values) < 2:
        return np.array([])
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return returns


def calculate_sharpe_ratio(returns: np.ndarray, 
                          risk_free_rate: float = None,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default from config)
        periods_per_year: Trading periods per year (252 for daily)
        
    Returns:
        Sharpe ratio
        
    Formula:
        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        Annualized = Daily Sharpe * sqrt(252)
    """
    if len(returns) == 0:
        return 0.0
    
    risk_free_rate = risk_free_rate or BACKTEST_CONFIG['risk_free_rate']
    
    # Calculate annualized return and volatility
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # Sample std dev
    
    if std_return == 0:
        return 0.0
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_volatility = std_return * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    
    return sharpe


def calculate_sortino_ratio(returns: np.ndarray,
                           risk_free_rate: float = None,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (penalizes only downside volatility).
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Sortino ratio
        
    Note:
        Similar to Sharpe but uses downside deviation instead of total volatility
    """
    if len(returns) == 0:
        return 0.0
    
    risk_free_rate = risk_free_rate or BACKTEST_CONFIG['risk_free_rate']
    
    # Calculate downside returns only
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(returns) > 0 else 0.0
    
    mean_return = np.mean(returns)
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_downside_vol = downside_std * np.sqrt(periods_per_year)
    
    sortino = (annual_return - risk_free_rate) / annual_downside_vol
    
    return sortino


def calculate_max_drawdown(portfolio_values: np.ndarray) -> tuple:
    """
    Calculate maximum drawdown and recovery information.
    
    Args:
        portfolio_values: Array of portfolio values
        
    Returns:
        Tuple of (max_drawdown, drawdown_duration, recovery_duration)
        
    Formula:
        Drawdown = (Trough - Peak) / Peak
        Max Drawdown = Minimum of all drawdowns
    """
    if len(portfolio_values) < 2:
        return 0.0, 0, 0
    
    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Calculate drawdown at each point
    drawdown = (portfolio_values - running_max) / running_max
    
    # Maximum drawdown
    max_dd = np.min(drawdown)
    
    # Find drawdown period
    max_dd_idx = np.argmin(drawdown)
    peak_idx = np.argmax(running_max[:max_dd_idx+1]) if max_dd_idx > 0 else 0
    drawdown_duration = max_dd_idx - peak_idx
    
    # Find recovery period (if recovered)
    recovery_duration = 0
    if max_dd_idx < len(portfolio_values) - 1:
        peak_value = running_max[max_dd_idx]
        recovery_mask = portfolio_values[max_dd_idx:] >= peak_value
        if np.any(recovery_mask):
            recovery_idx = max_dd_idx + np.argmax(recovery_mask)
            recovery_duration = recovery_idx - max_dd_idx
    
    return max_dd, drawdown_duration, recovery_duration


def calculate_win_rate(trade_log: List[Dict]) -> float:
    """
    Calculate win rate (percentage of profitable trades).
    
    Args:
        trade_log: List of trade dictionaries
        
    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    if len(trade_log) == 0:
        return 0.0
    
    # Filter sell trades (which have profit info)
    sell_trades = [t for t in trade_log if t['action'] == 'SELL']
    
    if len(sell_trades) == 0:
        return 0.0
    
    winning_trades = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
    win_rate = winning_trades / len(sell_trades)
    
    return win_rate


def calculate_profit_factor(trade_log: List[Dict]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trade_log: List of trade dictionaries
        
    Returns:
        Profit factor (>1 means profitable overall)
    """
    sell_trades = [t for t in trade_log if t['action'] == 'SELL']
    
    if len(sell_trades) == 0:
        return 0.0
    
    gross_profit = sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) > 0)
    gross_loss = abs(sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


class PerformanceReport:
    """
    Comprehensive performance report for backtest results.
    
    Aggregates all metrics and provides formatted output.
    
    Attributes:
        total_return: Total return over backtest period
        annualized_return: Annualized return
        sharpe_ratio: Sharpe ratio (risk-adjusted return)
        sortino_ratio: Sortino ratio (downside risk-adjusted)
        max_drawdown: Maximum drawdown
        win_rate: Percentage of winning trades
        profit_factor: Ratio of gross profit to gross loss
        total_trades: Number of round-trip trades
        
    Example:
        >>> report = PerformanceReport(portfolio_values, trade_log, dates, ...)
        >>> print(report)
        >>> print(f"Sharpe: {report.sharpe_ratio:.2f}")
    """
    
    def __init__(self,
                 portfolio_values: np.ndarray,
                 trade_log: List[Dict],
                 dates: np.ndarray,
                 initial_capital: float,
                 model_name: str = "",
                 strategy_name: str = "",
                 ticker: str = ""):
        """
        Initialize performance report.
        
        Args:
            portfolio_values: Array of portfolio values over time
            trade_log: List of executed trades
            dates: Array of dates corresponding to portfolio values
            initial_capital: Starting capital
            model_name: Name of model used
            strategy_name: Name of strategy used
            ticker: Stock ticker
        """
        self.portfolio_values = portfolio_values
        self.trade_log = trade_log
        self.dates = dates
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.strategy_name = strategy_name
        self.ticker = ticker
        
        # Calculate all metrics
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """Calculate all performance metrics."""
        
        # Basic returns
        returns = calculate_returns(self.portfolio_values)
        self.returns = returns
        
        final_value = self.portfolio_values[-1] if len(self.portfolio_values) > 0 else self.initial_capital
        self.total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate time period
        if len(self.dates) > 1:
            total_days = (self.dates[-1] - self.dates[0]).days
            years = total_days / 365.25
            self.annualized_return = ((1 + self.total_return) ** (1 / years)) - 1 if years > 0 else 0
        else:
            self.annualized_return = 0.0
        
        # Volatility
        self.volatility = np.std(returns, ddof=1) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        # Risk-adjusted metrics
        self.sharpe_ratio = calculate_sharpe_ratio(returns)
        self.sortino_ratio = calculate_sortino_ratio(returns)
        
        # Drawdown
        self.max_drawdown, self.drawdown_duration, self.recovery_duration = calculate_max_drawdown(self.portfolio_values)
        
        # Trade statistics
        self.win_rate = calculate_win_rate(self.trade_log)
        self.profit_factor = calculate_profit_factor(self.trade_log)
        
        # Count trades
        buy_trades = sum(1 for t in self.trade_log if t['action'] == 'BUY')
        sell_trades = sum(1 for t in self.trade_log if t['action'] == 'SELL')
        self.total_trades = min(buy_trades, sell_trades)  # Round trips
        
        # Calculate average trade duration
        if len(self.trade_log) >= 2:
            durations = []
            last_buy_date = None
            for trade in self.trade_log:
                if trade['action'] == 'BUY':
                    last_buy_date = trade['date']
                elif trade['action'] == 'SELL' and last_buy_date:
                    duration = (trade['date'] - last_buy_date).days
                    durations.append(duration)
                    last_buy_date = None
            self.avg_trade_duration = np.mean(durations) if durations else 0
        else:
            self.avg_trade_duration = 0
        
        # Final metrics
        self.final_value = final_value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'Model': self.model_name,
            'Strategy': self.strategy_name,
            'Ticker': self.ticker,
            'Total Return (%)': self.total_return * 100,
            'Annualized Return (%)': self.annualized_return * 100,
            'Volatility (%)': self.volatility * 100,
            'Sharpe Ratio': self.sharpe_ratio,
            'Sortino Ratio': self.sortino_ratio,
            'Max Drawdown (%)': self.max_drawdown * 100,
            'Win Rate (%)': self.win_rate * 100,
            'Profit Factor': self.profit_factor,
            'Total Trades': self.total_trades,
            'Avg Trade Duration (days)': self.avg_trade_duration,
            'Initial Capital': self.initial_capital,
            'Final Value': self.final_value,
        }
    
    def __str__(self) -> str:
        """Format report as string."""
        lines = [
            f"\n{'='*60}",
            f"Performance Report: {self.model_name} - {self.strategy_name}",
            f"Ticker: {self.ticker}",
            f"{'='*60}",
            f"\nReturns:",
            f"  Total Return:        {self.total_return*100:>8.2f}%",
            f"  Annualized Return:   {self.annualized_return*100:>8.2f}%",
            f"  Volatility:          {self.volatility*100:>8.2f}%",
            f"\nRisk-Adjusted Metrics:",
            f"  Sharpe Ratio:        {self.sharpe_ratio:>8.3f}",
            f"  Sortino Ratio:       {self.sortino_ratio:>8.3f}",
            f"\nDrawdown:",
            f"  Max Drawdown:        {self.max_drawdown*100:>8.2f}%",
            f"  Drawdown Duration:   {self.drawdown_duration:>8.0f} days",
            f"\nTrade Statistics:",
            f"  Win Rate:            {self.win_rate*100:>8.2f}%",
            f"  Profit Factor:       {self.profit_factor:>8.2f}",
            f"  Total Trades:        {self.total_trades:>8.0f}",
            f"  Avg Trade Duration:  {self.avg_trade_duration:>8.1f} days",
            f"\nPortfolio:",
            f"  Initial Capital:     {self.initial_capital:>12,.0f} VND",
            f"  Final Value:         {self.final_value:>12,.0f} VND",
            f"{'='*60}\n",
        ]
        return '\n'.join(lines)


if __name__ == '__main__':
    # Test metrics calculation
    print("Testing Performance Metrics...\n")
    
    # Simulate portfolio values
    initial = 100_000_000
    values = initial * np.cumprod(1 + np.random.randn(100) * 0.02)  # Random walk
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Test individual metrics
    returns = calculate_returns(values)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd, dd_dur, rec_dur = calculate_max_drawdown(values)
    
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Sortino Ratio: {sortino:.3f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Drawdown Duration: {dd_dur} days")
    
    print("\n✓ Metrics test complete!")
