"""
Main Backtesting Execution Script

Runs comprehensive backtesting for all models and strategies.
Generates comparison tables and visualization plots.

Usage:
    # Run all models with all strategies
    python run_backtest.py
    
    # Run specific model
    python run_backtest.py --model DEFAULT_ANN
    
    # Run specific strategy
    python run_backtest.py --strategy directional
    
    # Disable plots
    python run_backtest.py --no-plots

Output:
    - results/backtest_comparison.csv - Comparison table
    - results/backtest_plots/ - Visualization plots
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import config

from backtest_engine import (
    BacktestEngine,
    DirectionalStrategy,
    ThresholdStrategy,
    MultiStepStrategy,
    ConservativeStrategy,
    get_visualizer
)


def setup_output_directories():
    """Create output directories if they don't exist."""
    os.makedirs(config.EVALUATE_DIR, exist_ok=True)
    plots_dir = os.path.join(config.EVALUATE_DIR, 'backtest_plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def get_available_models():
    """Get list of models with enhanced predictions."""
    if not os.path.exists(config.PREDICT_DIR):
        return []
    
    model_files = [f for f in os.listdir(config.PREDICT_DIR) 
                   if f.endswith('_evaluate_data_enhanced.pkl')]
    
    model_names = [f.replace('_evaluate_data_enhanced.pkl', '') for f in model_files]
    return model_names


def create_strategies():
    """Create all strategy instances for testing."""
    strategies = [
        DirectionalStrategy(threshold=0.0),
        DirectionalStrategy(threshold=0.01),
        ThresholdStrategy(min_return_threshold=0.02, stop_loss=-0.05),
        MultiStepStrategy(min_agreement=2, weights=[0.5, 0.3, 0.2]),
        ConservativeStrategy(min_return_threshold=0.02, max_drawdown_limit=0.15)
    ]
    return strategies


def run_comprehensive_backtest(model_names, strategies, generate_plots=True, verbose=False):
    """
    Run backtest for all models and strategies.
    
    Args:
        model_names: List of model names to test
        strategies: List of strategy instances
        generate_plots: Whether to generate visualization plots
        verbose: Print detailed output
        
    Returns:
        DataFrame with comparison results
    """
    print("="*70)
    print("Comprehensive Backtesting")
    print("="*70)
    print(f"\nModels to test: {len(model_names)}")
    for name in model_names:
        print(f"  - {name}")
    
    print(f"\nStrategies to test: {len(strategies)}")
    for strategy in strategies:
        print(f"  - {strategy.name}")
    
    print("\n" + "="*70)
    
    # Run backtests
    all_results = []
    all_reports = []
    
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*70}")
        
        engine = BacktestEngine()
        
        for strategy in strategies:
            try:
                print(f"\nStrategy: {strategy.name}")
                print("-" * 60)
                
                report = engine.run_backtest(model_name, strategy, verbose=verbose)
                
                # Print summary
                print(f"\nResults:")
                print(f"  Total Return:     {report.total_return*100:>8.2f}%")
                print(f"  Sharpe Ratio:     {report.sharpe_ratio:>8.3f}")
                print(f"  Max Drawdown:     {report.max_drawdown*100:>8.2f}%")
                print(f"  Win Rate:         {report.win_rate*100:>8.2f}%")
                print(f"  Total Trades:     {report.total_trades:>8.0f}")
                
                # Store results
                result_dict = report.to_dict()
                all_results.append(result_dict)
                all_reports.append(report)
                
            except Exception as e:
                print(f"\n✗ Error running backtest: {e}")
                continue
    
    # Create comparison DataFrame
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # Sort by Sharpe Ratio (descending)
        df_results = df_results.sort_values('Sharpe Ratio', ascending=False)
        
        # Save to CSV
        output_path = os.path.join(config.EVALUATE_DIR, 'backtest_comparison.csv')
        df_results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
        
        # Generate plots if requested
        if generate_plots and all_reports:
            generate_visualization_plots(all_reports)
        
        return df_results, all_reports
    else:
        print("\n✗ No results generated")
        return None, []


def generate_visualization_plots(reports):
    """
    Generate visualization plots for all reports.
    
    Args:
        reports: List of PerformanceReport instances
    """
    print("\n" + "="*70)
    print("Generating Visualization Plots")
    print("="*70)
    
    plots_dir = os.path.join(config.EVALUATE_DIR, 'backtest_plots')
    
    try:
        BacktestVisualizer = get_visualizer()
        viz = BacktestVisualizer()
    except Exception as e:
        print(f"\n✗ Warning: Could not load visualizer: {e}")
        print("  Skipping plot generation (backtesting results still saved)")
        return
    
    # Generate individual plots for each report
    for report in reports:
        safe_name = f"{report.model_name}_{report.strategy_name}".replace('(', '_').replace(')', '_').replace(' ', '_')
        
        # Equity curve
        equity_path = os.path.join(plots_dir, f"{safe_name}_equity.png")
        viz.plot_portfolio_value(report, save_path=equity_path, show=False)
        
        # Drawdown
        drawdown_path = os.path.join(plots_dir, f"{safe_name}_drawdown.png")
        viz.plot_drawdown(report, save_path=drawdown_path, show=False)
        
        # Trades
        trades_path = os.path.join(plots_dir, f"{safe_name}_trades.png")
        viz.plot_trades(report, save_path=trades_path, show=False)
    
    # Generate comparison plots
    # Group by model
    models = {}
    for report in reports:
        if report.model_name not in models:
            models[report.model_name] = []
        models[report.model_name].append(report)
    
    for model_name, model_reports in models.items():
        comparison_path = os.path.join(plots_dir, f"{model_name}_strategy_comparison.png")
        viz.plot_model_comparison(model_reports, save_path=comparison_path, show=False)
    
    # Overall comparison across all models (best strategy for each)
    best_reports = []
    for model_name, model_reports in models.items():
        best_report = max(model_reports, key=lambda r: r.sharpe_ratio)
        best_reports.append(best_report)
    
    if len(best_reports) > 1:
        overall_path = os.path.join(plots_dir, "overall_model_comparison.png")
        viz.plot_model_comparison(best_reports, save_path=overall_path, show=False)
    
    print(f"\n✓ Plots saved to {plots_dir}/")


def print_summary_table(df_results):
    """Print a formatted summary table."""
    print("\n" + "="*70)
    print("BACKTEST RESULTS SUMMARY")
    print("="*70)
    
    # Select key columns for display
    display_cols = ['Model', 'Strategy', 'Total Return (%)', 'Sharpe Ratio', 
                   'Max Drawdown (%)', 'Win Rate (%)', 'Total Trades']
    
    if all(col in df_results.columns for col in display_cols):
        df_display = df_results[display_cols].copy()
        
        # Format numbers
        df_display['Total Return (%)'] = df_display['Total Return (%)'].apply(lambda x: f'{x:.2f}')
        df_display['Sharpe Ratio'] = df_display['Sharpe Ratio'].apply(lambda x: f'{x:.3f}')
        df_display['Max Drawdown (%)'] = df_display['Max Drawdown (%)'].apply(lambda x: f'{x:.2f}')
        df_display['Win Rate (%)'] = df_display['Win Rate (%)'].apply(lambda x: f'{x:.2f}')
        
        print(df_display.to_string(index=False))
    else:
        print(df_results.to_string(index=False))
    
    print("="*70)
    
    # Print best performers
    print("\nBEST PERFORMERS:")
    print("-" * 70)
    
    if 'Sharpe Ratio' in df_results.columns:
        best_sharpe = df_results.loc[df_results['Sharpe Ratio'].idxmax()]
        print(f"Highest Sharpe Ratio: {best_sharpe['Model']} - {best_sharpe['Strategy']}")
        print(f"  Sharpe: {best_sharpe['Sharpe Ratio']:.3f}")
    
    if 'Total Return (%)' in df_results.columns:
        best_return = df_results.loc[df_results['Total Return (%)'].idxmax()]
        print(f"\nHighest Total Return: {best_return['Model']} - {best_return['Strategy']}")
        print(f"  Return: {best_return['Total Return (%)']:.2f}%")
    
    if 'Max Drawdown (%)' in df_results.columns:
        best_drawdown = df_results.loc[df_results['Max Drawdown (%)'].idxmax()]
        print(f"\nLowest Drawdown: {best_drawdown['Model']} - {best_drawdown['Strategy']}")
        print(f"  Drawdown: {best_drawdown['Max Drawdown (%)']:.2f}%")
    
    print("="*70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run backtesting for stock forecasting models')
    parser.add_argument('--model', type=str, help='Specific model to test (e.g., DEFAULT_ANN)')
    parser.add_argument('--strategy', type=str, help='Specific strategy to test (e.g., directional)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    
    args = parser.parse_args()
    
    # Setup
    plots_dir = setup_output_directories()
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        print("\n✗ Error: No enhanced predictions found!")
        print("\nPlease run data_enhancement.py first:")
        print("  python data_enhancement.py")
        return
    
    # Filter models if specified
    if args.model:
        if args.model in available_models:
            model_names = [args.model]
        else:
            print(f"\n✗ Error: Model '{args.model}' not found!")
            print(f"Available models: {', '.join(available_models)}")
            return
    else:
        model_names = available_models
    
    # Create strategies
    all_strategies = create_strategies()
    
    # Filter strategies if specified
    if args.strategy:
        strategies = [s for s in all_strategies if args.strategy.lower() in s.name.lower()]
        if not strategies:
            print(f"\n✗ Error: No strategies matching '{args.strategy}'!")
            print(f"Available strategies: {[s.name for s in all_strategies]}")
            return
    else:
        strategies = all_strategies
    
    # Run backtests
    df_results, reports = run_comprehensive_backtest(
        model_names, 
        strategies, 
        generate_plots=not args.no_plots,
        verbose=args.verbose
    )
    
    if df_results is not None:
        # Print summary
        print_summary_table(df_results)
        
        print(f"\nResults saved to:")
        print(f"  - {os.path.join(config.EVALUATE_DIR, 'backtest_comparison.csv')}")
        if not args.no_plots:
            print(f"  - {plots_dir}/*.png")
    
    print("\n✓ Backtesting complete!")


if __name__ == '__main__':
    main()
