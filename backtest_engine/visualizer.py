"""
Visualization Tools for Backtesting Results

Provides plotting functions for analyzing backtest performance including
equity curves, drawdown charts, trade markers, and model comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict, Any
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine.config import PLOT_CONFIG
from backtest_engine.metrics import PerformanceReport


class BacktestVisualizer:
    """
    Visualization toolkit for backtest results.
    
    Creates publication-quality plots for analyzing trading performance.
    
    Example:
        >>> viz = BacktestVisualizer()
        >>> viz.plot_portfolio_value(report, save_path='results/equity_curve.png')
        >>> viz.plot_drawdown(report, save_path='results/drawdown.png')
    """
    
    def __init__(self, style: str = None, figsize: tuple = None):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style (default from config)
            figsize: Figure size (default from config)
        """
        self.style = style or PLOT_CONFIG.get('style', 'seaborn-v0_8-darkgrid')
        self.figsize = figsize or PLOT_CONFIG.get('figsize', (14, 8))
        self.colors = PLOT_CONFIG.get('colors', {})
        
        # Set style
        try:
            plt.style.use(self.style)
        except:
            print(f"Warning: Style '{self.style}' not available, using default")
    
    def plot_portfolio_value(self, 
                            report: PerformanceReport,
                            benchmark_values: np.ndarray = None,
                            save_path: str = None,
                            show: bool = True):
        """
        Plot portfolio value over time (equity curve).
        
        Args:
            report: PerformanceReport instance
            benchmark_values: Optional benchmark portfolio values (e.g., buy-and-hold)
            save_path: Path to save plot
            show: Display plot
            
        Example:
            >>> viz.plot_portfolio_value(report, save_path='equity_curve.png')
        """
        # TODO: Implement equity curve plotting
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot portfolio value
        dates = report.dates
        values = report.portfolio_values
        
        ax.plot(dates, values, 
                label=f'{report.model_name} - {report.strategy_name}',
                color=self.colors.get('portfolio', '#2E86AB'),
                linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_values is not None:
            ax.plot(dates, benchmark_values,
                   label='Buy & Hold Benchmark',
                   color=self.colors.get('benchmark', '#A23B72'),
                   linewidth=2,
                   linestyle='--',
                   alpha=0.7)
        
        # Add initial capital line
        ax.axhline(y=report.initial_capital, 
                  color='gray', 
                  linestyle=':', 
                  alpha=0.5,
                  label='Initial Capital')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value (VND)', fontsize=12)
        ax.set_title(f'Portfolio Equity Curve: {report.ticker}', fontsize=14, fontweight='bold')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved equity curve to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_drawdown(self,
                     report: PerformanceReport,
                     save_path: str = None,
                     show: bool = True):
        """
        Plot drawdown over time (underwater plot).
        
        Shows how far portfolio has fallen from its peak at each point.
        
        Args:
            report: PerformanceReport instance
            save_path: Path to save plot
            show: Display plot
        """
        # TODO: Implement drawdown plotting
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate drawdown
        values = report.portfolio_values
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        
        # Plot
        ax.fill_between(report.dates, drawdown * 100, 0, 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(report.dates, drawdown * 100, color='darkred', linewidth=1.5)
        
        # Highlight maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd_date = report.dates[max_dd_idx]
        max_dd_value = drawdown[max_dd_idx] * 100
        
        ax.scatter([max_dd_date], [max_dd_value], 
                  color='red', s=100, zorder=5,
                  label=f'Max DD: {max_dd_value:.2f}%')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(f'Drawdown Analysis: {report.model_name} - {report.strategy_name}', 
                    fontsize=14, fontweight='bold')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved drawdown plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_trades(self,
                   report: PerformanceReport,
                   price_data: pd.DataFrame = None,
                   save_path: str = None,
                   show: bool = True):
        """
        Plot price chart with buy/sell trade markers.
        
        Args:
            report: PerformanceReport instance
            price_data: DataFrame with price data (optional, uses portfolio value if None)
            save_path: Path to save plot
            show: Display plot
        """
        # TODO: Implement trade markers plot
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price (or portfolio value if price not provided)
        if price_data is not None and 'close' in price_data.columns:
            ax1.plot(price_data.index, price_data['close'], 
                    color='black', linewidth=1.5, label='Price')
            price_label = 'Stock Price'
        else:
            ax1.plot(report.dates, report.portfolio_values,
                    color=self.colors.get('portfolio', '#2E86AB'),
                    linewidth=1.5, label='Portfolio Value')
            price_label = 'Portfolio Value (VND)'
        
        # Mark trades
        buy_trades = [t for t in report.trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in report.trade_log if t['action'] == 'SELL']
        
        if buy_trades:
            buy_dates = [t['date'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax1.scatter(buy_dates, buy_prices, 
                       marker=self.colors.get('buy_marker', '^'),
                       color=self.colors.get('buy_color', 'green'),
                       s=100, zorder=5, label='Buy', alpha=0.7)
        
        if sell_trades:
            sell_dates = [t['date'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax1.scatter(sell_dates, sell_prices,
                       marker=self.colors.get('sell_marker', 'v'),
                       color=self.colors.get('sell_color', 'red'),
                       s=100, zorder=5, label='Sell', alpha=0.7)
        
        ax1.set_ylabel(price_label, fontsize=12)
        ax1.set_title(f'Trading Activity: {report.model_name} - {report.strategy_name}',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Plot trade returns in bottom panel
        if sell_trades:
            trade_returns = [t.get('return_pct', 0) for t in sell_trades]
            trade_dates = [t['date'] for t in sell_trades]
            
            colors = ['green' if r > 0 else 'red' for r in trade_returns]
            ax2.bar(trade_dates, trade_returns, color=colors, alpha=0.6)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('Return (%)', fontsize=10)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved trade plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_model_comparison(self,
                             reports: List[PerformanceReport],
                             metrics: List[str] = None,
                             save_path: str = None,
                             show: bool = True):
        """
        Compare performance metrics across multiple models/strategies.
        
        Args:
            reports: List of PerformanceReport instances
            metrics: List of metric names to compare
            save_path: Path to save plot
            show: Display plot
        """
        # TODO: Implement model comparison plot
        
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Prepare data
        labels = [f"{r.model_name}\n{r.strategy_name}" for r in reports]
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Extract metric values
            if metric == 'total_return':
                values = [r.total_return * 100 for r in reports]
                ylabel = 'Total Return (%)'
            elif metric == 'sharpe_ratio':
                values = [r.sharpe_ratio for r in reports]
                ylabel = 'Sharpe Ratio'
            elif metric == 'max_drawdown':
                values = [abs(r.max_drawdown) * 100 for r in reports]
                ylabel = 'Max Drawdown (%)'
            elif metric == 'win_rate':
                values = [r.win_rate * 100 for r in reports]
                ylabel = 'Win Rate (%)'
            else:
                values = [getattr(r, metric, 0) for r in reports]
                ylabel = metric.replace('_', ' ').title()
            
            # Color bars based on value
            colors = ['green' if v > 0 else 'red' for v in values]
            if metric == 'max_drawdown':
                colors = ['red' if v > 15 else 'orange' if v > 10 else 'green' for v in values]
            
            # Plot
            bars = ax.bar(range(len(values)), values, color=colors, alpha=0.6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model & Strategy Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_returns_distribution(self,
                                 report: PerformanceReport,
                                 save_path: str = None,
                                 show: bool = True):
        """
        Plot distribution of returns (histogram).
        
        Args:
            report: PerformanceReport instance
            save_path: Path to save plot
            show: Display plot
        """
        # TODO: Implement returns distribution plot
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        returns = report.returns * 100  # Convert to percentage
        
        # Plot histogram
        n, bins, patches = ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Color bars by return sign
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('red')
                patch.set_alpha(0.6)
            else:
                patch.set_facecolor('green')
                patch.set_alpha(0.6)
        
        # Add vertical line at mean
        mean_return = np.mean(returns)
        ax.axvline(mean_return, color='darkblue', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_return:.3f}%')
        
        # Add vertical line at median
        median_return = np.median(returns)
        ax.axvline(median_return, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_return:.3f}%')
        
        # Formatting
        ax.set_xlabel('Daily Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Returns Distribution: {report.model_name} - {report.strategy_name}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text box
        stats_text = f'Std Dev: {np.std(returns):.3f}%\n'
        stats_text += f'Skewness: {pd.Series(returns).skew():.3f}\n'
        stats_text += f'Kurtosis: {pd.Series(returns).kurtosis():.3f}'
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved returns distribution to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':
    # Test visualizer with dummy data
    print("Testing BacktestVisualizer...\n")
    
    # Create dummy report
    from backtest_engine.metrics import PerformanceReport
    
    initial = 100_000_000
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    values = initial * np.cumprod(1 + np.random.randn(100) * 0.02)
    
    # Create dummy trade log
    trade_log = [
        {'date': dates[10], 'action': 'BUY', 'shares': 1000, 'price': 50.0},
        {'date': dates[30], 'action': 'SELL', 'shares': 1000, 'price': 55.0, 'profit': 5000, 'return_pct': 10},
        {'date': dates[50], 'action': 'BUY', 'shares': 1000, 'price': 52.0},
        {'date': dates[70], 'action': 'SELL', 'shares': 1000, 'price': 48.0, 'profit': -4000, 'return_pct': -7.7},
    ]
    
    report = PerformanceReport(
        portfolio_values=values,
        trade_log=trade_log,
        dates=dates,
        initial_capital=initial,
        model_name='TEST_MODEL',
        strategy_name='Test Strategy',
        ticker='TEST'
    )
    
    # Test visualizer
    viz = BacktestVisualizer()
    
    print("Creating test plots...")
    viz.plot_portfolio_value(report, show=False, save_path='test_equity.png')
    viz.plot_drawdown(report, show=False, save_path='test_drawdown.png')
    viz.plot_trades(report, show=False, save_path='test_trades.png')
    viz.plot_returns_distribution(report, show=False, save_path='test_returns_dist.png')
    
    print("\n✓ Visualizer test complete!")
    print("Test plots saved (test_*.png)")
