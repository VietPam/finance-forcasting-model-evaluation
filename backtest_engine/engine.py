"""
Core Backtesting Engine

Simulates portfolio performance using model predictions and trading strategies.
Handles trade execution, position tracking, and portfolio valuation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine.config import BACKTEST_CONFIG
from backtest_engine.strategies import BaseStrategy
from backtest_engine.metrics import PerformanceReport
from backtest_engine.utils import load_enhanced_predictions


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    
    Simulates realistic trading with transaction costs, slippage,
    and position tracking. Supports multiple trading strategies
    and generates comprehensive performance metrics.
    
    Attributes:
        initial_capital: Starting capital in VND
        cash: Current available cash
        shares: Current share holdings
        positions: List of open positions
        portfolio_values: Historical portfolio values
        trade_log: Record of all executed trades
        
    Example:
        >>> from backtest_engine import BacktestEngine, DirectionalStrategy
        >>> engine = BacktestEngine(initial_capital=100_000_000)
        >>> strategy = DirectionalStrategy(threshold=0.01)
        >>> results = engine.run_backtest('DEFAULT_ANN', strategy)
        >>> print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    """
    
    def __init__(self, 
                 initial_capital: float = None,
                 transaction_cost: float = None,
                 slippage: float = None):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital (default from config)
            transaction_cost: Transaction fee per trade (default from config)
            slippage: Price slippage assumption (default from config)
        """
        self.initial_capital = initial_capital or BACKTEST_CONFIG['initial_capital']
        self.transaction_cost = transaction_cost or BACKTEST_CONFIG['transaction_cost']
        self.slippage = slippage or BACKTEST_CONFIG['slippage']
        
        # Portfolio state
        self.cash = self.initial_capital
        self.shares = 0
        self.entry_price = 0.0
        
        # Tracking
        self.portfolio_values = []
        self.trade_log = []
        self.dates = []
        
        # Data storage
        self.predictions = None
        self.actuals = None
        self.ohlc_data = None
        
    def load_data(self, model_name: str):
        """
        Load enhanced prediction data for backtesting.
        
        Args:
            model_name: Name of model (e.g., 'DEFAULT_ANN', 'LSTM')
            
        Raises:
            FileNotFoundError: If enhanced predictions not found
        """
        data = load_enhanced_predictions(model_name)
        
        self.predictions = data['y_pred']
        self.actuals = data['y_true']
        self.dates = data['dates']
        self.ohlc_data = data['ohlc_data']
        self.model_name = model_name
        self.ticker = data['metadata'].get('ticker', 'UNKNOWN')
        
        print(f"Loaded data for {model_name} on {self.ticker}")
        print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
        print(f"Total samples: {len(self.predictions)}")
        
    def execute_trade(self, 
                     signal: int, 
                     current_price: float, 
                     timestamp: datetime,
                     reason: str = ""):
        """
        Execute a trade based on signal.
        
        Args:
            signal: Trading signal (1=buy, -1=sell, 0=hold)
            current_price: Current stock price
            timestamp: Trade timestamp
            reason: Optional description of trade reason
            
        Returns:
            Trade execution details (dict) or None if no trade
            
        Note:
            - Applies transaction costs and slippage
            - Updates cash and share positions
            - Records trade in trade_log
        """
        
        if signal == 1 and self.cash > 0 and self.shares == 0:
            # BUY signal - buy with all available cash
            
            # Calculate effective price (with slippage)
            effective_price = current_price * (1 + self.slippage)
            
            # Calculate shares to buy (accounting for transaction cost)
            total_cost_multiplier = 1 + self.transaction_cost
            shares_to_buy = int(self.cash / (effective_price * total_cost_multiplier))
            
            if shares_to_buy >= BACKTEST_CONFIG['min_trade_size']:
                # Execute buy
                total_cost = shares_to_buy * effective_price * total_cost_multiplier
                
                self.shares = shares_to_buy
                self.cash -= total_cost
                self.entry_price = effective_price
                
                trade_record = {
                    'date': timestamp,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': effective_price,
                    'cost': total_cost,
                    'transaction_fee': shares_to_buy * effective_price * self.transaction_cost,
                    'reason': reason
                }
                self.trade_log.append(trade_record)
                
                return trade_record
                
        elif signal == -1 and self.shares > 0:
            # SELL signal - sell all shares
            
            # Calculate effective price (with slippage)
            effective_price = current_price * (1 - self.slippage)
            
            # Calculate revenue (minus transaction cost)
            gross_revenue = self.shares * effective_price
            transaction_fee = gross_revenue * self.transaction_cost
            net_revenue = gross_revenue - transaction_fee
            
            # Calculate profit/loss
            profit = net_revenue - (self.shares * self.entry_price * (1 + self.transaction_cost))
            
            # Execute sell
            self.cash += net_revenue
            shares_sold = self.shares
            self.shares = 0
            self.entry_price = 0.0
            
            trade_record = {
                'date': timestamp,
                'action': 'SELL',
                'shares': shares_sold,
                'price': effective_price,
                'revenue': net_revenue,
                'transaction_fee': transaction_fee,
                'profit': profit,
                'return_pct': (profit / (shares_sold * self.entry_price)) * 100 if self.entry_price > 0 else 0,
                'reason': reason
            }
            self.trade_log.append(trade_record)
            
            return trade_record
        
        return None
    
    def get_portfolio_value(self, current_price: float) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_price: Current stock price
            
        Returns:
            Total portfolio value (cash + stock holdings)
        """
        stock_value = self.shares * current_price
        return self.cash + stock_value
    
    def run_backtest(self, 
                    model_name: str,
                    strategy: BaseStrategy,
                    verbose: bool = True) -> PerformanceReport:
        """
        Run backtest simulation using specified model and strategy.
        
        Args:
            model_name: Model name (e.g., 'DEFAULT_ANN', 'LSTM')
            strategy: Trading strategy instance
            verbose: Print progress information
            
        Returns:
            PerformanceReport with comprehensive metrics
            
        Example:
            >>> from backtest_engine import DirectionalStrategy
            >>> strategy = DirectionalStrategy(threshold=0.01)
            >>> report = engine.run_backtest('DEFAULT_ANN', strategy)
        """
        
        # Reset state
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_values = []
        self.trade_log = []
        
        # Load data
        self.load_data(model_name)
        
        if verbose:
            print(f"\nRunning backtest: {model_name} with {strategy.name}")
            print(f"Initial capital: {self.initial_capital:,.0f} VND")
            print(f"Transaction cost: {self.transaction_cost*100:.2f}%")
            print("-" * 60)
        
        # Main simulation loop
        for i in range(len(self.predictions)):
            # Get current data
            current_date = self.dates[i]
            pred_t1 = self.predictions[i, 0]  # t+1 prediction
            pred_t2 = self.predictions[i, 1]  # t+2 prediction
            pred_t3 = self.predictions[i, 2]  # t+3 prediction
            actual_price = self.actuals[i, 0]  # actual t+1 price
            
            # Get current price (price at time of prediction)
            # This is tricky: we need the price BEFORE the prediction
            # For simplicity, use the actual price at t-1 or from OHLC data
            if i > 0:
                current_price = self.actuals[i-1, 0]
            else:
                current_price = actual_price  # First sample assumption
            
            # Generate trading signal using strategy
            signal = strategy.generate_signal(
                current_price=current_price,
                pred_t1=pred_t1,
                pred_t2=pred_t2,
                pred_t3=pred_t3,
                position=self.shares,
                portfolio_value=self.get_portfolio_value(current_price)
            )
            
            # Execute trade if signal != 0
            if signal != 0:
                trade = self.execute_trade(signal, actual_price, current_date, 
                                          reason=strategy.get_signal_reason())
                if trade and verbose:
                    action = trade['action']
                    shares = trade['shares']
                    price = trade['price']
                    print(f"{current_date.strftime('%Y-%m-%d')}: {action} {shares} shares @ {price:.2f}")
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value(actual_price)
            self.portfolio_values.append(portfolio_value)
        
        # Final liquidation (close any open positions)
        if self.shares > 0:
            final_price = self.actuals[-1, 0]
            final_date = self.dates[-1]
            self.execute_trade(-1, final_price, final_date, reason="Final liquidation")
            final_value = self.get_portfolio_value(final_price)
            self.portfolio_values[-1] = final_value
        
        if verbose:
            print("-" * 60)
            print(f"Final portfolio value: {self.portfolio_values[-1]:,.0f} VND")
            print(f"Total trades: {len(self.trade_log)}")
            print()
        
        # Generate performance report
        report = PerformanceReport(
            portfolio_values=np.array(self.portfolio_values),
            trade_log=self.trade_log,
            dates=self.dates,
            initial_capital=self.initial_capital,
            model_name=model_name,
            strategy_name=strategy.name,
            ticker=self.ticker
        )
        
        return report
    
    def run_multiple_strategies(self, 
                               model_name: str,
                               strategies: List[BaseStrategy],
                               verbose: bool = False) -> Dict[str, PerformanceReport]:
        """
        Run backtest with multiple strategies for comparison.
        
        Args:
            model_name: Model name
            strategies: List of strategy instances
            verbose: Print progress
            
        Returns:
            Dictionary mapping strategy names to PerformanceReports
        """
        
        results = {}
        
        for strategy in strategies:
            print(f"\nTesting strategy: {strategy.name}")
            report = self.run_backtest(model_name, strategy, verbose=verbose)
            results[strategy.name] = report
            
            # Print summary
            print(f"  Total Return: {report.total_return*100:.2f}%")
            print(f"  Sharpe Ratio: {report.sharpe_ratio:.3f}")
            print(f"  Max Drawdown: {report.max_drawdown*100:.2f}%")
        
        return results


if __name__ == '__main__':
    # Test backtest engine
    from backtest_engine.strategies import DirectionalStrategy
    
    print("Testing BacktestEngine...")
    
    # Initialize engine
    engine = BacktestEngine()
    
    # Create strategy
    strategy = DirectionalStrategy(threshold=0.01)
    
    # Run backtest
    try:
        report = engine.run_backtest('DEFAULT_ANN', strategy, verbose=True)
        print("\n=== Performance Summary ===")
        print(report)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run data_enhancement.py first to create enhanced predictions.")
