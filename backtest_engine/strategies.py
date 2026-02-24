"""
Trading Strategy Implementations

Contains various trading strategies that generate buy/sell signals
based on model predictions. All strategies inherit from BaseStrategy.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement generate_signal() method.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name for identification
        """
        self.name = name
        self.last_signal_reason = ""
        
    @abstractmethod
    def generate_signal(self,
                       current_price: float,
                       pred_t1: float,
                       pred_t2: float,
                       pred_t3: float,
                       position: int = 0,
                       portfolio_value: float = 0) -> int:
        """
        Generate trading signal based on predictions.
        
        Args:
            current_price: Current stock price
            pred_t1: Predicted price for t+1
            pred_t2: Predicted price for t+2
            pred_t3: Predicted price for t+3
            position: Current position (shares held)
            portfolio_value: Current portfolio value
            
        Returns:
            Signal: 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    def get_signal_reason(self) -> str:
        """Get explanation for last signal generated."""
        return self.last_signal_reason


class DirectionalStrategy(BaseStrategy):
    """
    Simple directional trading strategy.
    
    - BUY if predicted price > current price
    - SELL if predicted price < current price
    - Use t+1 prediction only
    
    Parameters:
        threshold: Minimum predicted return to trigger trade (default: 0.0)
        
    Example:
        >>> strategy = DirectionalStrategy(threshold=0.01)  # 1% threshold
        >>> signal = strategy.generate_signal(100, 102, 103, 104)
        >>> print(signal)  # 1 (buy, because 102 > 100)
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Initialize directional strategy.
        
        Args:
            threshold: Minimum return threshold (e.g., 0.01 = 1%)
        """
        super().__init__(name=f"Directional(threshold={threshold})")
        self.threshold = threshold
        
    def generate_signal(self,
                       current_price: float,
                       pred_t1: float,
                       pred_t2: float,
                       pred_t3: float,
                       position: int = 0,
                       portfolio_value: float = 0) -> int:
        """Generate signal based on t+1 prediction direction."""
        
        # Calculate predicted return
        pred_return = (pred_t1 - current_price) / current_price
        
        # Generate signal
        if position == 0:  # No position, consider buying
            if pred_return > self.threshold:
                self.last_signal_reason = f"Buy: predicted return {pred_return*100:.2f}% > threshold {self.threshold*100:.2f}%"
                return 1  # BUY
        else:  # Have position, consider selling
            if pred_return < -self.threshold:
                self.last_signal_reason = f"Sell: predicted return {pred_return*100:.2f}% < -{self.threshold*100:.2f}%"
                return -1  # SELL
        
        self.last_signal_reason = "Hold: no signal triggered"
        return 0  # HOLD


class ThresholdStrategy(BaseStrategy):
    """
    Threshold-based trading strategy with stop-loss.
    
    - Only BUY if predicted return > min_return_threshold
    - SELL if predicted return < -min_return_threshold OR hit stop_loss
    - More conservative than DirectionalStrategy
    
    Parameters:
        min_return_threshold: Minimum return to enter trade (e.g., 0.02 = 2%)
        stop_loss: Stop loss threshold (e.g., -0.05 = -5%)
        
    Example:
        >>> strategy = ThresholdStrategy(min_return_threshold=0.02, stop_loss=-0.05)
    """
    
    def __init__(self, min_return_threshold: float = 0.02, stop_loss: float = -0.05):
        """
        Initialize threshold strategy.
        
        Args:
            min_return_threshold: Minimum return to trigger buy
            stop_loss: Stop loss level (negative value)
        """
        super().__init__(name=f"Threshold(return>{min_return_threshold}, SL={stop_loss})")
        self.min_return_threshold = min_return_threshold
        self.stop_loss = stop_loss
        self.entry_price = None
        
    def generate_signal(self,
                       current_price: float,
                       pred_t1: float,
                       pred_t2: float,
                       pred_t3: float,
                       position: int = 0,
                       portfolio_value: float = 0) -> int:
        """Generate signal with strict entry/exit criteria."""
        
        pred_return = (pred_t1 - current_price) / current_price
        
        if position == 0:  # No position
            # Only buy if predicted return exceeds threshold
            if pred_return > self.min_return_threshold:
                self.entry_price = current_price
                self.last_signal_reason = f"Buy: predicted {pred_return*100:.2f}% > {self.min_return_threshold*100:.2f}%"
                return 1  # BUY
        else:  # Have position
            # Check stop loss
            if self.entry_price is not None:
                actual_return = (current_price - self.entry_price) / self.entry_price
                if actual_return < self.stop_loss:
                    self.last_signal_reason = f"Sell: stop loss triggered ({actual_return*100:.2f}%)"
                    self.entry_price = None
                    return -1  # SELL
            
            # Check predicted reversal
            if pred_return < -self.min_return_threshold:
                self.last_signal_reason = f"Sell: predicted decline {pred_return*100:.2f}%"
                self.entry_price = None
                return -1  # SELL
        
        self.last_signal_reason = "Hold: threshold not met"
        return 0  # HOLD


class MultiStepStrategy(BaseStrategy):
    """
    Multi-step consensus strategy using all 3 predictions.
    
    - Uses predictions for t+1, t+2, and t+3
    - Requires consensus across predictions (min_agreement)
    - Can weight predictions differently (nearer predictions weighted more)
    
    Parameters:
        min_agreement: Minimum number of predictions that must agree (2 or 3)
        weights: Weights for [t+1, t+2, t+3] predictions
        threshold: Minimum weighted return to trigger trade
        
    Example:
        >>> strategy = MultiStepStrategy(min_agreement=2, weights=[0.5, 0.3, 0.2])
    """
    
    def __init__(self, 
                 min_agreement: int = 2,
                 weights: list = None,
                 threshold: float = 0.01):
        """
        Initialize multi-step strategy.
        
        Args:
            min_agreement: How many predictions must agree (2 or 3)
            weights: Prediction weights [t+1, t+2, t+3]
            threshold: Minimum return threshold
        """
        super().__init__(name=f"MultiStep(agreement={min_agreement})")
        self.min_agreement = min_agreement
        self.weights = weights or [0.5, 0.3, 0.2]  # Default: weight t+1 most
        self.threshold = threshold
        
    def generate_signal(self,
                       current_price: float,
                       pred_t1: float,
                       pred_t2: float,
                       pred_t3: float,
                       position: int = 0,
                       portfolio_value: float = 0) -> int:
        """Generate signal based on multi-step consensus."""
        
        # Calculate returns for each prediction
        return_t1 = (pred_t1 - current_price) / current_price
        return_t2 = (pred_t2 - current_price) / current_price
        return_t3 = (pred_t3 - current_price) / current_price
        
        # Count agreement (how many predictions suggest upward/downward)
        predictions = [return_t1, return_t2, return_t3]
        bullish_count = sum(1 for r in predictions if r > self.threshold)
        bearish_count = sum(1 for r in predictions if r < -self.threshold)
        
        # Calculate weighted return
        weighted_return = (return_t1 * self.weights[0] + 
                          return_t2 * self.weights[1] + 
                          return_t3 * self.weights[2])
        
        if position == 0:  # No position
            # Buy if enough predictions are bullish AND weighted return positive
            if bullish_count >= self.min_agreement and weighted_return > self.threshold:
                self.last_signal_reason = f"Buy: {bullish_count}/3 bullish, weighted return {weighted_return*100:.2f}%"
                return 1  # BUY
        else:  # Have position
            # Sell if enough predictions are bearish OR weighted return negative
            if bearish_count >= self.min_agreement or weighted_return < -self.threshold:
                self.last_signal_reason = f"Sell: {bearish_count}/3 bearish, weighted return {weighted_return*100:.2f}%"
                return -1  # SELL
        
        self.last_signal_reason = f"Hold: insufficient consensus ({bullish_count} bullish, {bearish_count} bearish)"
        return 0  # HOLD


class ConservativeStrategy(BaseStrategy):
    """
    Conservative risk-averse strategy.
    
    - Requires ALL 3 predictions to agree (unanimous consensus)
    - Higher return threshold (2% instead of 1%)
    - Stops trading if drawdown exceeds limit
    - Most conservative strategy for capital preservation
    
    Parameters:
        min_return_threshold: Minimum return to trade (default: 0.02 = 2%)
        max_drawdown_limit: Stop trading if drawdown exceeds this (default: 0.15 = 15%)
        
    Example:
        >>> strategy = ConservativeStrategy(min_return_threshold=0.02)
    """
    
    def __init__(self, 
                 min_return_threshold: float = 0.02,
                 max_drawdown_limit: float = 0.15):
        """
        Initialize conservative strategy.
        
        Args:
            min_return_threshold: Minimum return for entry
            max_drawdown_limit: Maximum acceptable drawdown before stopping
        """
        super().__init__(name=f"Conservative(return>{min_return_threshold})")
        self.min_return_threshold = min_return_threshold
        self.max_drawdown_limit = max_drawdown_limit
        self.peak_value = 0
        self.trading_enabled = True
        
    def generate_signal(self,
                       current_price: float,
                       pred_t1: float,
                       pred_t2: float,
                       pred_t3: float,
                       position: int = 0,
                       portfolio_value: float = 0) -> int:
        """Generate signal with strict consensus requirements."""
        
        # Update peak value for drawdown calculation
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Check if drawdown limit exceeded
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if current_drawdown > self.max_drawdown_limit:
                self.trading_enabled = False
                self.last_signal_reason = f"Trading disabled: drawdown {current_drawdown*100:.2f}% > limit {self.max_drawdown_limit*100:.2f}%"
                if position > 0:
                    return -1  # SELL to stop losses
                return 0  # HOLD (no new positions)
        
        # Calculate returns
        return_t1 = (pred_t1 - current_price) / current_price
        return_t2 = (pred_t2 - current_price) / current_price
        return_t3 = (pred_t3 - current_price) / current_price
        
        if position == 0:  # No position
            # Require ALL predictions to be bullish
            all_bullish = (return_t1 > self.min_return_threshold and 
                          return_t2 > self.min_return_threshold and 
                          return_t3 > self.min_return_threshold)
            
            if all_bullish and self.trading_enabled:
                avg_return = (return_t1 + return_t2 + return_t3) / 3
                self.last_signal_reason = f"Buy: unanimous bullish signal, avg return {avg_return*100:.2f}%"
                return 1  # BUY
        else:  # Have position
            # Sell if ANY prediction suggests decline
            any_bearish = (return_t1 < -self.min_return_threshold or 
                          return_t2 < -self.min_return_threshold or 
                          return_t3 < -self.min_return_threshold)
            
            if any_bearish:
                self.last_signal_reason = "Sell: one or more bearish predictions"
                return -1  # SELL
        
        self.last_signal_reason = "Hold: no unanimous consensus"
        return 0  # HOLD


if __name__ == '__main__':
    # Test strategies
    print("Testing Trading Strategies...\n")
    
    # Test case: current price = 100, predictions slightly higher
    current = 100.0
    pred_t1 = 101.5  # +1.5%
    pred_t2 = 102.0  # +2.0%
    pred_t3 = 101.0  # +1.0%
    
    strategies = [
        DirectionalStrategy(threshold=0.01),
        ThresholdStrategy(min_return_threshold=0.02),
        MultiStepStrategy(min_agreement=2),
        ConservativeStrategy(min_return_threshold=0.02)
    ]
    
    for strategy in strategies:
        signal = strategy.generate_signal(current, pred_t1, pred_t2, pred_t3, position=0)
        signal_name = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}[signal]
        print(f"{strategy.name}:")
        print(f"  Signal: {signal_name}")
        print(f"  Reason: {strategy.get_signal_reason()}")
        print()
    
    print("✓ Strategies test complete!")
