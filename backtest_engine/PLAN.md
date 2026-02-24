# Backtesting Engine Implementation Plan

## TL;DR
Build a backtesting engine to evaluate trading profitability of ML predictions. The engine will simulate realistic trading with transaction costs, support multiple strategies, and calculate comprehensive risk-adjusted metrics. **Key challenge:** enhance predictions with date/OHLC data since current predictions lack timestamps. The implementation follows a modular design with separate components for strategy, execution, metrics, and visualization. **Expected outcome:** answer "Would trading on these predictions be profitable?" with quantitative evidence (Sharpe ratio, max drawdown, total return).

**Critical Decision:** Enhance existing predictions **before** building backtest engine to add dates and OHLC data rather than rebuilding prediction pipeline.

---

## Architecture

```
backtest_engine/
├── __init__.py          # Package exports
├── config.py            # Configuration parameters
├── engine.py            # Core BacktestEngine class
├── strategies.py        # Trading strategy implementations
├── metrics.py           # Performance metric calculations
├── visualizer.py        # Visualization tools
└── utils.py             # Helper functions

data_enhancement.py      # Script to add dates/OHLC to predictions
run_backtest.py          # Main execution script
```

---

## Implementation Phases

### Phase 1: Data Enhancement ✓ (To be completed first)
**Goal:** Add dates and OHLC data to existing predictions

**Files to create:**
- [backtest_engine/utils.py](backtest_engine/utils.py) - Helper functions
- [data_enhancement.py](data_enhancement.py) - Enhancement script

**Key functions:**
- `reconstruct_test_dates()` - Map test set indices to calendar dates
- `load_ohlc_data()` - Retrieve historical OHLC from CSV
- `enhance_predictions()` - Add dates/OHLC to prediction pickles

**Verification:**
```python
# After running data_enhancement.py
import pickle
with open('predictions/DEFAULT_ANN_evaluate_data_enhanced.pkl', 'rb') as f:
    data = pickle.load(f)
print('Keys:', data.keys())  # Should include 'dates', 'ohlc_data'
```

---

### Phase 2: Core Infrastructure
**Goal:** Build backtesting engine with transaction costs

**Files to create:**
- [backtest_engine/config.py](backtest_engine/config.py) - Configuration
- [backtest_engine/metrics.py](backtest_engine/metrics.py) - Performance metrics
- [backtest_engine/engine.py](backtest_engine/engine.py) - Main engine

**Key classes:**
- `BacktestEngine` - Portfolio simulation with trade execution
- `PerformanceReport` - Comprehensive metrics aggregation

**Features:**
- Transaction cost modeling (0.15% Vietnam market fee)
- Slippage simulation (0.1%)
- Position tracking (cash + shares)
- Trade logging for analysis

---

### Phase 3: Trading Strategies
**Goal:** Implement multiple trading strategies

**File to create:**
- [backtest_engine/strategies.py](backtest_engine/strategies.py) - All strategies

**Strategies:**
1. **DirectionalStrategy** - Simple trend following (buy if pred > current)
2. **ThresholdStrategy** - Minimum return threshold with stop-loss
3. **MultiStepStrategy** - Multi-horizon consensus (uses t+1, t+2, t+3)
4. **ConservativeStrategy** - Risk-averse with unanimous agreement

**Testing:**
```python
from backtest_engine.strategies import DirectionalStrategy
strategy = DirectionalStrategy(threshold=0.01)
signal = strategy.generate_signal(100, 102, 103, 104, position=0)
# Returns: 1 (buy) because 102 > 100
```

---

### Phase 4: Visualization
**Goal:** Create plots for analysis

**File to create:**
- [backtest_engine/visualizer.py](backtest_engine/visualizer.py) - Plotting tools

**Plots:**
- Portfolio equity curve over time
- Drawdown chart (underwater plot)
- Trade markers on price chart (buy/sell arrows)
- Model comparison (side-by-side performance)

---

### Phase 5: Integration
**Goal:** Integrate with existing evaluation pipeline

**Files to modify:**
- [evaluation.py](evaluation.py) - Add backtest to evaluation
- [README.md](README.md) - Document backtesting

**Files to create:**
- [run_backtest.py](run_backtest.py) - Standalone backtest script

**Output:**
- `results/backtest_comparison.csv` - Results table
- `results/backtest_plots/` - Visualization folder

---

## Success Criteria

- [ ] All predictions have dates and OHLC data
- [ ] Backtest engine executes trades with realistic costs
- [ ] 4+ strategies implemented and tested
- [ ] Comprehensive metrics calculated (Sharpe, drawdown, etc.)
- [ ] Visualizations generated for analysis
- [ ] Integration with evaluation.py complete
- [ ] Documentation updated

---

## Technical Specifications

### Market Parameters (Vietnam Stock Market)
```python
BACKTEST_CONFIG = {
    'initial_capital': 100_000_000,  # 100M VND
    'transaction_cost': 0.0015,       # 0.15% brokerage fee
    'slippage': 0.001,                # 0.1% slippage assumption
    'risk_free_rate': 0.05,           # 5% annual (gov bonds)
}
```

### Performance Metrics
- **Total Return (%)** - Overall gain/loss
- **Annualized Return (%)** - Year-over-year equivalent
- **Sharpe Ratio** - Risk-adjusted return (uses risk-free rate = 5%)
- **Sortino Ratio** - Downside risk-adjusted return
- **Maximum Drawdown (%)** - Largest peak-to-trough decline
- **Win Rate (%)** - Percentage of profitable trades
- **Profit Factor** - Gross profit / gross loss ratio
- **Total Trades** - Number of round-trip trades

### Strategy Parameters
- **Position Sizing:** All-in/all-out (binary: 100% invested or 0%)
- **Trade Execution:** End-of-day (close price)
- **Data:** Uses t+1 predictions for next-day trading

---

## Implementation Steps

### Step 1: Data Enhancement
```bash
# Run data enhancement to add dates/OHLC
python data_enhancement.py

# Expected output:
# Enhancing predictions for DEFAULT_ANN...
# Enhancing predictions for LSTM...
# Enhanced predictions saved to predictions/*_enhanced.pkl
```

### Step 2: Test Core Engine
```bash
# Test backtest engine with DirectionalStrategy
python -m backtest_engine.engine

# Expected: Run basic backtest and display results
```

### Step 3: Test All Strategies
```bash
# Test all strategies
python -m backtest_engine.strategies

# Expected: Show signal generation for test cases
```

### Step 4: Run Full Backtest
```bash
# Run comprehensive backtest
python run_backtest.py

# Expected: 
# - Backtest all models with all strategies
# - Save results/backtest_comparison.csv
# - Generate plots in results/backtest_plots/
```

### Step 5: Integration
```bash
# Run full evaluation including backtesting
python evaluation.py --include-backtest

# Expected: Complete evaluation report with backtest metrics
```

---

## Key Decisions

### Decision 1: Enhance existing predictions vs. rebuild pipeline
- **Chose:** Enhance existing predictions with dates/OHLC post-hoc
- **Reasoning:** Faster implementation, doesn't require retraining models, backward compatible
- **Alternative:** Modify `data_preprocess.py` and `ann_models.py` to save dates during training
- **Trade-off:** Less elegant architecturally, but pragmatic for POC

### Decision 2: Use closing prices only vs. full OHLC for execution
- **Chose:** Use closing prices for trade execution (matched to predictions)
- **Reasoning:** Models predict only close prices; intraday execution unrealistic for EOD predictions
- **Alternative:** Use next day's open price for execution (more realistic)
- **Trade-off:** Slightly optimistic (perfect execution at close), but consistent with prediction task

### Decision 3: Single position vs. position sizing
- **Chose:** All-in/all-out (binary position: 100% invested or 0%)
- **Reasoning:** Simpler logic, clear comparison between strategies
- **Alternative:** Implement position sizing based on prediction confidence
- **Future Enhancement:** Can extend with `confidence_weighted_position_size()` method

### Decision 4: Strategy architecture - inheritance vs. composition
- **Chose:** Inheritance from `BaseStrategy` abstract class
- **Reasoning:** Clear interface, enforces `generate_signal()` contract, extensible
- **Alternative:** Functional approach with strategy functions
- **Trade-off:** More boilerplate, but better for complex strategies

### Decision 5: Metrics calculation - during backtest vs. post-processing
- **Chose:** Post-processing after backtest completes
- **Reasoning:** Separates concerns, easier to add new metrics without modifying engine
- **Alternative:** Calculate metrics incrementally during backtest
- **Trade-off:** Slightly less efficient (two passes), but more flexible

---

## Verification Checklist

### After Phase 1 (Data Enhancement):
```bash
# Verify enhanced predictions
python -c "
import pickle
with open('predictions/DEFAULT_ANN_evaluate_data_enhanced.pkl', 'rb') as f:
    data = pickle.load(f)
print('Keys:', data.keys())
print('Dates shape:', data['dates'].shape)
print('First date:', data['dates'][0])
print('Last date:', data['dates'][-1])
"
```

### After Phase 2-3 (Core Engine):
```bash
# Run basic backtest
python run_backtest.py --model DEFAULT_ANN --strategy directional

# Expected output:
# Total Return: X%
# Sharpe Ratio: Y
# Max Drawdown: Z%
```

### After Phase 4-5 (Complete System):
```bash
# Run full evaluation with backtesting
python evaluation.py --include-backtest

# Check outputs:
ls results/backtest_comparison.csv
ls results/backtest_plots/*.png
```

### Manual Verification:
- [ ] Verify total trades × 2 × transaction_cost matches expected costs
- [ ] Confirm final portfolio value = cash + shares × final_price
- [ ] Check buy/sell signals make logical sense given predictions
- [ ] Validate Sharpe ratio = (annual_return - risk_free_rate) / volatility
- [ ] Ensure drawdown never exceeds -100% (sanity check)
- [ ] Compare results across models (should be different)

---

## Next Steps

1. ✓ Create PLAN.md (this file)
2. → Create backtest_engine package files
3. → Create data_enhancement.py script
4. → Run data enhancement to prepare predictions
5. → Test strategies independently
6. → Run full backtest
7. → Generate visualizations
8. → Integrate with evaluation pipeline
9. → Update documentation

---

## Notes

- All monetary values in VND (Vietnamese Dong)
- Assumes 252 trading days per year for annualization
- Risk-free rate set to 5% (Vietnam government bond yield)
- Models predict closing prices only (no intraday data)
- Test set contains 563 samples for ACB ticker
- Enhanced predictions stored separately (backward compatible)
