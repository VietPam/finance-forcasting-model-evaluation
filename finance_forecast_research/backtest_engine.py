import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BacktestEngine:
    def __init__(self, initial_capital=100000000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0
        self.equity_curve = []
        self.trade_log = []

    def run_backtest(self, actual_prices, predicted_prices):
        self.reset()
        n_days = len(actual_prices)
        
        for i in range(n_days):
            current_price = actual_prices[i]
            prediction_t1 = predicted_prices[i][0]
            
            if self.position == 0:
                if prediction_t1 > current_price * 1.01:
                    shares_to_buy = self.cash // (current_price * (1 + self.commission))
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        fee = cost * self.commission
                        self.cash -= (cost + fee)
                        self.position = shares_to_buy
                        self.trade_log.append({'type': 'BUY', 'price': current_price, 'day': i})
            
            elif self.position > 0:
                if prediction_t1 < current_price:
                    revenue = self.position * current_price
                    fee = revenue * self.commission
                    self.cash += (revenue - fee)
                    self.trade_log.append({'type': 'SELL', 'price': current_price, 'day': i})
                    self.position = 0

            current_equity = self.cash + (self.position * current_price)
            self.equity_curve.append(current_equity)

        return self.calculate_metrics()

    def calculate_metrics(self):
        final_value = self.equity_curve[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        return {
            'Total Return (%)': round(total_return, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Final Value (VND)': round(final_value, 0),
            'Trade Count': len(self.trade_log)
        }

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve - Performance')
        plt.ylabel('VND')
        plt.xlabel('Days')
        plt.grid(True)
        plt.show()