# Backtesting Results Summary

## Overview

This document summarizes the backtesting results for the financial forecasting models (DEFAULT_ANN and LSTM) on ACB stock using various trading strategies.

**Test Period**: September 25, 2023 to December 31, 2025 (563 trading days)  
**Initial Capital**: 100,000,000 VND  
**Transaction Cost**: 0.15% per trade  
**Slippage**: 0.1% per trade  

---

## Key Findings

### 🏆 Best Overall Performance

**DEFAULT_ANN with Conservative Strategy**
- Total Return: **33.14%**
- Annualized Return: **13.46%**
- Sharpe Ratio: **0.571**
- Max Drawdown: **-12.80%**
- Win Rate: **77.78%**
- Total Trades: 9
- Final Portfolio Value: **133,140,166 VND**

### Top 3 Strategies

1. **DEFAULT_ANN - Conservative(return>0.02)**
   - Return: 33.14% | Sharpe: 0.571 | Drawdown: -12.80%

2. **DEFAULT_ANN - Directional(threshold=0.01)**
   - Return: 30.29% | Sharpe: 0.510 | Drawdown: -16.37%

3. **DEFAULT_ANN - MultiStep(agreement=2)**
   - Return: 29.18% | Sharpe: 0.454 | Drawdown: -18.23%

---

## Model Comparison

### DEFAULT_ANN Performance
- **Most Profitable**: Conservative strategy (33.14% return)
- **Best Sharpe Ratio**: 0.571 (Conservative strategy)
- **Highest Win Rate**: 77.78% (Conservative and MultiStep)
- **Average Return**: 23.73% across profitable strategies

### LSTM Performance
- **Most Profitable**: Directional(threshold=0.01) with 13.68% return
- **Best Sharpe Ratio**: 0.142
- **Highest Win Rate**: 100% (but only 2 trades)
- **Conservative Strategy**: Made 0 trades (too strict criteria)

**Winner**: DEFAULT_ANN significantly outperforms LSTM in backtesting

---

## Strategy Analysis

### 1. Conservative Strategy (High Threshold + Unanimous Consensus)
- **Best for**: Capital preservation with steady returns
- **Characteristics**: Requires all 3 predictions to agree, 2% minimum return threshold
- **DEFAULT_ANN**: 33.14% return, 77.78% win rate, 9 trades
- **LSTM**: 0% return (no trades - too restrictive for LSTM's predictions)

### 2. Directional with 1% Threshold
- **Best for**: Balanced risk/reward
- **Characteristics**: Simple buy/sell based on t+1 prediction with 1% threshold
- **DEFAULT_ANN**: 30.29% return, 62.50% win rate, 16 trades
- **LSTM**: 13.68% return, 100% win rate, 2 trades

### 3. MultiStep Consensus
- **Best for**: Medium-term trend following
- **Characteristics**: Uses all 3 predictions (t+1, t+2, t+3), requires 2/3 agreement
- **DEFAULT_ANN**: 29.18% return, 77.78% win rate, 9 trades
- **LSTM**: 12.28% return, 66.67% win rate, 3 trades

### 4. Threshold with Stop-Loss
- **Best for**: Risk-averse trading
- **Characteristics**: 2% entry threshold, -5% stop loss
- **DEFAULT_ANN**: 19.96% return, 54.55% win rate, 11 trades
- **LSTM**: 6.48% return, 66.67% win rate, 3 trades

### 5. Pure Directional (No Threshold)
- **Best for**: Active trading (NOT RECOMMENDED)
- **Characteristics**: Trade on every prediction regardless of magnitude
- **DEFAULT_ANN**: 6.02% return, -0.054 Sharpe, 47 trades ❌
- **LSTM**: 7.83% return, -0.154 Sharpe, 6 trades ❌

**Lesson**: Strategies without thresholds perform poorly due to excessive trading and transaction costs.

---

## Risk-Adjusted Performance

### Sharpe Ratio Rankings
1. DEFAULT_ANN - Conservative: **0.571** ⭐
2. DEFAULT_ANN - Directional(0.01): **0.510**
3. DEFAULT_ANN - MultiStep: **0.454**
4. DEFAULT_ANN - Threshold: **0.288**
5. LSTM - Directional(0.01): **0.142**

**Threshold for Good Performance**: Sharpe > 0.5 indicates strong risk-adjusted returns

### Maximum Drawdown Analysis
- **Best (Lowest)**: LSTM Conservative - 0% (but made no trades)
- **Best (Active)**: LSTM Directional(0.01) - **-3.17%** ⭐
- **Worst**: DEFAULT_ANN Directional(0) - **-19.23%**

**Acceptable Range**: Drawdowns under -15% are manageable

---

## Trading Activity

### Trade Frequency
- **Most Active**: DEFAULT_ANN Directional(0) - 47 trades (too many)
- **Most Selective**: LSTM Directional(0.01) - 2 trades
- **Optimal Range**: 9-16 trades (best balance)

### Win Rate Analysis
- **Perfect**: LSTM Directional(0.01) - 100% (but only 2 trades)
- **Excellent**: DEFAULT_ANN Conservative/MultiStep - 77.78%
- **Good**: DEFAULT_ANN Directional(0.01) - 62.50%

**Target Win Rate**: >60% for sustainable profitability

### Profit Factor
- **Best**: LSTM Directional(0) - 15.04
- **Excellent**: LSTM MultiStep - 7.58
- **Very Good**: DEFAULT_ANN Conservative - 3.39

**Interpretation**: Profit Factor > 2.0 indicates strong profitability

---

## Conclusions

### 1. Model Selection
**DEFAULT_ANN is Superior for Trading**
- While LSTM may have slightly better prediction metrics (MAE, RMSE), DEFAULT_ANN generates significantly higher trading profits
- Reason: DEFAULT_ANN's predictions are more actionable and consistent for directional trading

### 2. Strategy Selection
**Conservative Strategy Wins**
- Highest total return (33.14%)
- Best risk-adjusted return (Sharpe 0.571)
- Excellent win rate (77.78%)
- Manageable drawdown (-12.80%)

### 3. Importance of Thresholds
**Thresholds are Critical**
- Pure directional trading (no threshold) performs poorly
- 1-2% minimum return thresholds dramatically improve performance
- Prevents overtrading and reduces transaction costs

### 4. Multi-Step Predictions
**Using All 3 Predictions Helps**
- MultiStep strategy achieves 77.78% win rate
- Consensus across t+1, t+2, t+3 provides higher confidence
- Slightly lower return than Conservative but more diversified signals

### 5. Realistic Expectations
**Achievable Returns**
- 10-15% annualized return is achievable with proper strategy
- 30%+ total return over 2+ years is realistic
- Win rates of 60-80% are attainable
- Drawdowns of 10-15% should be expected

---

## Recommendations

### For Maximum Profit
**Use**: DEFAULT_ANN with Conservative Strategy
- Expected Annualized Return: ~13.5%
- Trade Frequency: Low (9 trades over 2+ years)
- Risk Level: Moderate

### For Balanced Approach
**Use**: DEFAULT_ANN with Directional(threshold=0.01)
- Expected Annualized Return: ~12.4%
- Trade Frequency: Medium (16 trades)
- Risk Level: Moderate-High

### For Risk-Averse Investors
**Use**: DEFAULT_ANN with Threshold Strategy (stop-loss)
- Expected Annualized Return: ~8.4%
- Trade Frequency: Medium (11 trades)
- Risk Level: Low-Moderate
- Built-in protection: -5% stop loss

### NOT Recommended
- ❌ Pure directional strategies (no threshold)
- ❌ LSTM Conservative (too restrictive, makes no trades)
- ❌ Any strategy with Sharpe Ratio < 0

---

## Next Steps

### To Reproduce Results
```bash
# 1. Enhance predictions with dates/OHLC
python3 data_enhancement.py

# 2. Run comprehensive backtest
python3 run_backtest.py --no-plots

# 3. View results
cat results/backtest_comparison.csv
```

### To Implement in Production
1. Use DEFAULT_ANN model (models/DEFAULT_ANN_best_model.keras)
2. Implement Conservative Strategy with 2% threshold
3. Monitor drawdown and exit if exceeds 15%
4. Rebalance monthly or when signals trigger
5. Track actual vs. backtested performance

### Future Enhancements
- [ ] Test on other VN30 stocks
- [ ] Implement ensemble strategies (combine multiple models)
- [ ] Add confidence-weighted position sizing
- [ ] Incorporate sentiment analysis
- [ ] Develop real-time trading dashboard

---

## Files Generated

- `results/backtest_comparison.csv` - Complete results table
- `predictions/*_enhanced.pkl` - Enhanced prediction files with dates
- `backtest_engine/` - Complete backtesting framework

## Framework Features

✅ Realistic transaction costs (0.15%)  
✅ Slippage simulation (0.1%)  
✅ Multiple trading strategies  
✅ Comprehensive performance metrics  
✅ Risk-adjusted analysis (Sharpe, Sortino)  
maxDrawdown tracking  
✅ Trade-by-trade logging  
✅ Position management  

---

**Generated**: February 24, 2026  
**Framework Version**: 0.1.0  
**Test Period**: Sept 2023 - Dec 2025  
**Stock**: ACB (Asia Commercial Bank)
