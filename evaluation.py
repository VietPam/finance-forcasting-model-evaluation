import config, pickle, os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self):
        """Initialize the ModelEvaluator"""
        os.makedirs(config.EVALUATE_DIR, exist_ok=True)
        self.results = {}
        self.predictions = {}
        self.true_values = {}

    def load_data(self, dir = config.PREDICT_DIR):
        """Load evaluation data from a pickle file"""
        for file_name in os.listdir(dir):
            if file_name.endswith('_evaluate_data.pkl'):
                print(f"Loading evaluation results from {file_name}...")
                with open(os.path.join(dir, file_name), 'rb') as f:
                    model_name = file_name.replace('_evaluate_data.pkl', '')
                    data = pickle.load(f)
                    self.predictions[model_name] = data['y_pred']
                    self.true_values[model_name] = data['y_true'] 
        # Check if all models have the same prediction length
        lengths = [len(v) for v in self.predictions.values()]
        print("Prediction lengths from models:", lengths)
        if len(set(lengths)) != 1:
            raise ValueError("Mismatch in prediction lengths among models.")
        
        return self.predictions, self.true_values

    @staticmethod
    def calculate_mae(y_true, y_pred):
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_mse(y_true, y_pred):
        """Calculate Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """Calculate Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_r2(y_true, y_pred):
        """Calculate R-squared"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def calculate_directional_accuracy(y_true, y_pred):
        """
        Calculate Directional Accuracy (DA)
        Percentage of correct direction predictions
        """
        # Calculate changes in direction
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate accuracy
        correct_directions = np.sum(true_direction == pred_direction)
        total_predictions = len(true_direction)
        
        return (correct_directions / total_predictions) * 100

    def evaluate_model(self, model_name, y_true, y_pred):
        """
        Evaluate a model and calculate all metrics
        Supports both single-output and multi-output predictions
        
        Args:
            model_name: Name of the model
            y_true: True values (can be 1D or 2D for multi-output)
            y_pred: Predicted values (can be 1D or 2D for multi-output)
            
        Returns:
            Dictionary with all metrics
        """
        # Check if multi-output
        is_multi_output = len(y_true.shape) > 1 and y_true.shape[1] > 1
        
        if is_multi_output:
            # Multi-step multi-output evaluation
            # Use output names from config (t+1_open, t+1_close, etc.)
            output_names = config.OUTPUT_NAMES if hasattr(config, 'OUTPUT_NAMES') else \
                          [f'output_{i}' for i in range(y_true.shape[1])]
            
            metrics = {'Model': model_name}
            
            # Calculate metrics for each output separately
            per_output_metrics = {}
            for i, output_name in enumerate(output_names):
                y_true_i = y_true[:, i]
                y_pred_i = y_pred[:, i]
                
                per_output_metrics[output_name] = {
                    'MAE': self.calculate_mae(y_true_i, y_pred_i),
                    'MSE': self.calculate_mse(y_true_i, y_pred_i),
                    'RMSE': self.calculate_rmse(y_true_i, y_pred_i),
                    'MAPE': self.calculate_mape(y_true_i, y_pred_i),
                    'R2': self.calculate_r2(y_true_i, y_pred_i)
                }
                
                # Store per-output metrics in main dict
                metrics[f'{output_name}_MAE'] = per_output_metrics[output_name]['MAE']
                metrics[f'{output_name}_RMSE'] = per_output_metrics[output_name]['RMSE']
                metrics[f'{output_name}_R2'] = per_output_metrics[output_name]['R2']
            
            # Calculate overall metrics (average across all outputs)
            metrics['MAE'] = np.mean([m['MAE'] for m in per_output_metrics.values()])
            metrics['MSE'] = np.mean([m['MSE'] for m in per_output_metrics.values()])
            metrics['RMSE'] = np.mean([m['RMSE'] for m in per_output_metrics.values()])
            metrics['MAPE'] = np.mean([m['MAPE'] for m in per_output_metrics.values()])
            metrics['R2'] = np.mean([m['R2'] for m in per_output_metrics.values()])
            
            # Calculate directional accuracy for t+1 close price
            # Find index of t+1_close
            close_idx = next((i for i, name in enumerate(output_names) if 't+1_close' in name), 1)
            metrics['DA'] = self.calculate_directional_accuracy(y_true[:, close_idx], y_pred[:, close_idx])
            
            # Store per-output metrics for detailed analysis
            metrics['per_output'] = per_output_metrics
            
        else:
            # Single-output evaluation
            metrics = {
                'Model': model_name,
                'MAE': self.calculate_mae(y_true, y_pred),
                'MSE': self.calculate_mse(y_true, y_pred),
                'RMSE': self.calculate_rmse(y_true, y_pred),
                'MAPE': self.calculate_mape(y_true, y_pred),
                'R2': self.calculate_r2(y_true, y_pred),
                'DA': self.calculate_directional_accuracy(y_true, y_pred)
            }
        
        self.results[model_name] = metrics
        return metrics
    
    def print_evaluation(self, model_name):
        """Print evaluation results for a model"""
        if model_name not in self.results:
            print(f"No results for model: {model_name}")
            return
        
        metrics = self.results[model_name]
        print(f"\n{'='*70}")
        print(f"Evaluation Results for {model_name}")
        print(f"{'='*70}")
        print(f"\nOverall Metrics (averaged across all outputs):")
        print(f"  MAE (Mean Absolute Error):           {metrics['MAE']:.6f}")
        print(f"  MSE (Mean Squared Error):            {metrics['MSE']:.6f}")
        print(f"  RMSE (Root Mean Squared Error):      {metrics['RMSE']:.6f}")
        print(f"  MAPE (Mean Absolute % Error):        {metrics['MAPE']:.2f}%")
        print(f"  R² (R-squared Score):                {metrics['R2']:.6f}")
        print(f"  DA (Directional Accuracy):           {metrics['DA']:.2f}%")
        
        # Print per-output metrics if available
        if 'per_output' in metrics:
            print(f"\nPer-Output Metrics (MAE / RMSE / R²):")
            print(f"  {'Output':<15} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
            print(f"  {'-'*47}")
            for output_name, output_metrics in metrics['per_output'].items():
                print(f"  {output_name:<15} {output_metrics['MAE']:>10.6f} "
                      f"{output_metrics['RMSE']:>10.6f} {output_metrics['R2']:>10.6f}")
        
        print(f"{'='*70}\n")

    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            print("No models to compare!")
            return None
        
        # Create comparison dataframe
        df = pd.DataFrame(self.results).T
        
        print(f"\n{'='*80}")
        print("Model Comparison Summary")
        print(f"{'='*80}")
        
        # Find best model for each metric
        print("Best Models by Metric:")
        print("-" * 80)
        
        # For MAE, MSE, RMSE, MAPE: lower is better
        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            best_model = df[metric].idxmin()
            best_value = df[metric].min()
            print(f"{metric:30s}: {best_model:20s} ({best_value:.6f})")
        
        # For R2, DA: higher is better
        for metric in ['R2', 'DA']:
            best_model = df[metric].idxmax()
            best_value = df[metric].max()
            print(f"{metric:30s}: {best_model:20s} ({best_value:.6f})")
        
        print(f"{'='*80}\n")
        
        # Save comparison to CSV
        output_path = os.path.join(config.EVALUATE_DIR, 'model_comparison.csv')
        df.to_csv(output_path)
        print(f"Comparison saved to {output_path}")
        
        return df

    def plot_predictions(self, y_true, predictions_dict, title='Model Predictions Comparison', 
                        save_path=None, n_samples=200):
        """
        Plot actual vs predicted values for multiple models
        Supports both single-output and multi-output predictions
        
        Args:
            y_true: True values (can be 1D or 2D for multi-output)
            predictions_dict: Dictionary of {model_name: predictions}
            title: Plot title
            save_path: Path to save the plot
            n_samples: Number of samples to plot (for readability)
        """
        # Check if multi-output
        #is_multi_output = len(y_true.shape) > 1 and y_true.shape[1] > 1
        is_multi_output = len(y_true.shape) > 1

        if is_multi_output:
            # Multi-output plotting
            #output_names = config.PREDICT_COLUMNS if hasattr(config, 'PREDICT_COLUMNS') else [f'output_{i}' for i in range(y_true.shape[1])]
            output_names = [f'output_{i}' for i in range(y_true.shape[1])]
            n_outputs = len(output_names)
            
            fig, axes = plt.subplots(n_outputs, 1, figsize=(15, 4*n_outputs))
            if n_outputs == 1:
                axes = [axes]
            
            # Plot only last n_samples for clarity
            # plot_range = slice(-n_samples, None)
            plot_range = slice(None)
            x = np.arange(len(y_true[plot_range]))
            
            for i, output_name in enumerate(output_names):
                # Plot true values
                axes[i].plot(x, y_true[plot_range, i], 'k-', label='Actual', linewidth=2, alpha=0.7)
                
                # Plot predictions for each model
                colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
                for (model_name, y_pred), color in zip(predictions_dict.items(), colors):
                    axes[i].plot(x, y_pred[plot_range, i], '--', label=model_name, 
                               linewidth=1.5, alpha=0.7, color=color)
                
                axes[i].set_xlabel('Time Steps', fontsize=12)
                axes[i].set_ylabel(output_name.upper(), fontsize=12)
                axes[i].set_title(f'{title} - {output_name.upper()}', fontsize=14, fontweight='bold')
                axes[i].legend(loc='best', fontsize=10)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
        else:
            # Single-output plotting (original)
            plt.figure(figsize=(15, 6))
            
            plot_range = slice(-n_samples, None)
            x = np.arange(len(y_true[plot_range]))
            
            plt.plot(x, y_true[plot_range], 'k-', label='Actual', linewidth=2, alpha=0.7)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
            for (model_name, y_pred), color in zip(predictions_dict.items(), colors):
                plt.plot(x, y_pred[plot_range], '--', label=model_name, 
                        linewidth=1.5, alpha=0.7, color=color)
            
            plt.xlabel('Time Steps', fontsize=12)
            plt.ylabel('Stock Price', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.EVALUATE_DIR, 'predictions_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
        plt.close()

    def plot_error_distribution(self, y_true, predictions_dict, save_path=None):
        """
        Plot error distribution for multiple models
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of {model_name: predictions}
            save_path: Path to save the plot
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
            errors = y_true - y_pred
            
            ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Prediction Error', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{model_name}\nMean Error: {np.mean(errors):.4f}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.EVALUATE_DIR, 'error_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, save_path=None):
        """
        Plot bar charts comparing metrics across models
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            print("No results to plot!")
            return
        
        df = pd.DataFrame(self.results).T
        
        # Select metrics to plot
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'R2', 'DA']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            values = df[metric].values
            models = df.index.tolist()
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Highlight best model
            if metric in ['MAE', 'RMSE', 'MAPE']:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.EVALUATE_DIR, 'metrics_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")
        plt.close()
    
    def plot_scatter(self, y_true, predictions_dict, save_path=None):
        """
        Plot scatter plots of actual vs predicted for each model
        Supports both single-output and multi-output predictions
        
        Args:
            y_true: True values (can be 1D or 2D for multi-output)
            predictions_dict: Dictionary of {model_name: predictions}
            save_path: Path to save the plot
        """
        is_multi_output = len(y_true.shape) > 1 and y_true.shape[1] > 1
        
        if is_multi_output:
            # For multi-output, plot scatter for close price only (index 2)
            output_names = config.PREDICT_COLUMNS if hasattr(config, 'PREDICT_COLUMNS') else [f'output_{i}' for i in range(y_true.shape[1])]
            close_idx = output_names.index('close') if 'close' in output_names else 2
            
            n_models = len(predictions_dict)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
            
            if n_models == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
                y_true_close = y_true[:, close_idx]
                y_pred_close = y_pred[:, close_idx]
                
                ax.scatter(y_true_close, y_pred_close, alpha=0.5, s=10)
                
                # Plot perfect prediction line
                min_val = min(y_true_close.min(), y_pred_close.min())
                max_val = max(y_true_close.max(), y_pred_close.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                
                ax.set_xlabel('Actual Close Price', fontsize=10)
                ax.set_ylabel('Predicted Close Price', fontsize=10)
                ax.set_title(f'{model_name}\nR² = {r2_score(y_true_close, y_pred_close):.4f}', 
                            fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Remove extra subplots
            for idx in range(n_models, len(axes)):
                fig.delaxes(axes[idx])
        else:
            # Single-output scatter plot (original)
            n_models = len(predictions_dict)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
            
            if n_models == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
                ax.scatter(y_true, y_pred, alpha=0.5, s=10)
                
                # Plot perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                
                ax.set_xlabel('Actual Values', fontsize=10)
                ax.set_ylabel('Predicted Values', fontsize=10)
                ax.set_title(f'{model_name}\nR² = {r2_score(y_true, y_pred):.4f}', 
                            fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Remove extra subplots
            for idx in range(n_models, len(axes)):
                fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.EVALUATE_DIR, 'scatter_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
        plt.close()

    def generate_report(self, ticker=None):
        """
        Generate complete evaluation report with all plots and metrics
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of {model_name: predictions}
            ticker: Stock ticker (optional)
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*80 + "\n")
        self.load_data()

        # Calculate metrics for all models
        for model_name, y_pred in self.predictions.items():
            y_true = self.true_values[model_name]
            self.evaluate_model(model_name, y_true, y_pred)
            self.print_evaluation(model_name)
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        title_suffix = f" - {ticker}" if ticker else ""
        
        self.plot_predictions(y_true, self.predictions, 
                            title=f'Stock Price Predictions{title_suffix}')
        self.plot_error_distribution(y_true, self.predictions)
        self.plot_metrics_comparison()
        self.plot_scatter(y_true, self.predictions)
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80 + "\n")
        
        return comparison_df

if __name__=='__main__':
    evaluator = ModelEvaluator()
    evaluator.generate_report()