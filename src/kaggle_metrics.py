import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class KaggleMetrics:
    """
    Comprehensive Kaggle-style metrics for time series prediction evaluation.
    Includes metrics commonly used in Kaggle competitions and financial time series.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(self, y_true, y_pred, sample_weights=None):
        """
        Calculate all Kaggle-style metrics for time series prediction.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weights: Optional sample weights for weighted metrics
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        self.metrics = {}
        
        # Basic regression metrics
        self._calculate_basic_metrics(y_true, y_pred)
        
        # Time series specific metrics
        self._calculate_timeseries_metrics(y_true, y_pred)
        
        # Financial metrics
        self._calculate_financial_metrics(y_true, y_pred)
        
        # Directional accuracy metrics
        self._calculate_directional_metrics(y_true, y_pred)
        
        # Correlation metrics
        self._calculate_correlation_metrics(y_true, y_pred)
        
        # Custom Kaggle-style metrics
        self._calculate_custom_metrics(y_true, y_pred, sample_weights)
        
        return self.metrics
    
    def _calculate_basic_metrics(self, y_true, y_pred):
        """Calculate basic regression metrics."""
        self.metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        self.metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        self.metrics['MSE'] = mean_squared_error(y_true, y_pred)
        self.metrics['R2'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        self.metrics['MAPE'] = mape
        
        # Symmetric Mean Absolute Percentage Error
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        self.metrics['SMAPE'] = smape
    
    def _calculate_timeseries_metrics(self, y_true, y_pred):
        """Calculate time series specific metrics."""
        # Mean Absolute Scaled Error (MASE)
        # Using naive forecast (previous value) as baseline
        naive_forecast = np.roll(y_true, 1)
        naive_forecast[0] = y_true[0]  # First value same as actual
        
        mae_naive = mean_absolute_error(y_true[1:], naive_forecast[1:])
        mae_model = mean_absolute_error(y_true, y_pred)
        
        if mae_naive > 0:
            mase = mae_model / mae_naive
        else:
            mase = np.inf
        
        self.metrics['MASE'] = mase
        
        # Root Mean Square Percentage Error
        rmse_percentage = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100
        self.metrics['RMSPE'] = rmse_percentage
        
        # Mean Absolute Scaled Error (alternative calculation)
        # Using rolling mean as baseline
        rolling_mean = pd.Series(y_true).rolling(window=7, min_periods=1).mean().values
        mae_rolling = mean_absolute_error(y_true, rolling_mean)
        
        if mae_rolling > 0:
            mase_rolling = mae_model / mae_rolling
        else:
            mase_rolling = np.inf
        
        self.metrics['MASE_Rolling'] = mase_rolling
    
    def _calculate_financial_metrics(self, y_true, y_pred):
        """Calculate financial time series specific metrics."""
        # Returns-based metrics
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        if len(true_returns) > 1:
            sharpe_true = np.mean(true_returns) / np.std(true_returns) if np.std(true_returns) > 0 else 0
            sharpe_pred = np.mean(pred_returns) / np.std(pred_returns) if np.std(pred_returns) > 0 else 0
            self.metrics['Sharpe_True'] = sharpe_true
            self.metrics['Sharpe_Pred'] = sharpe_pred
            self.metrics['Sharpe_Diff'] = abs(sharpe_true - sharpe_pred)
        
        # Maximum Drawdown
        cumulative_true = np.cumprod(1 + true_returns)
        cumulative_pred = np.cumprod(1 + pred_returns)
        
        max_dd_true = self._calculate_max_drawdown(cumulative_true)
        max_dd_pred = self._calculate_max_drawdown(cumulative_pred)
        
        self.metrics['MaxDD_True'] = max_dd_true
        self.metrics['MaxDD_Pred'] = max_dd_pred
        self.metrics['MaxDD_Diff'] = abs(max_dd_true - max_dd_pred)
        
        # Volatility
        vol_true = np.std(true_returns) * np.sqrt(252)  # Annualized
        vol_pred = np.std(pred_returns) * np.sqrt(252)
        self.metrics['Volatility_True'] = vol_true
        self.metrics['Volatility_Pred'] = vol_pred
        self.metrics['Volatility_Diff'] = abs(vol_true - vol_pred)
    
    def _calculate_directional_metrics(self, y_true, y_pred):
        """Calculate directional accuracy metrics."""
        # Directional accuracy
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        directional_accuracy = np.mean(true_direction == pred_direction)
        self.metrics['Directional_Accuracy'] = directional_accuracy
        
        # Confusion matrix for direction
        tp = np.sum((true_direction == True) & (pred_direction == True))
        tn = np.sum((true_direction == False) & (pred_direction == False))
        fp = np.sum((true_direction == False) & (pred_direction == True))
        fn = np.sum((true_direction == True) & (pred_direction == False))
        
        # Precision, Recall, F1 for direction prediction
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.metrics['Direction_Precision'] = precision
        self.metrics['Direction_Recall'] = recall
        self.metrics['Direction_F1'] = f1
    
    def _calculate_correlation_metrics(self, y_true, y_pred):
        """Calculate correlation-based metrics."""
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        self.metrics['Pearson_Correlation'] = pearson_corr
        self.metrics['Pearson_P_Value'] = pearson_p
        
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        self.metrics['Spearman_Correlation'] = spearman_corr
        self.metrics['Spearman_P_Value'] = spearman_p
        
        # Returns correlation
        if len(y_true) > 1:
            true_returns = np.diff(y_true) / y_true[:-1]
            pred_returns = np.diff(y_pred) / y_pred[:-1]
            
            returns_pearson, _ = pearsonr(true_returns, pred_returns)
            returns_spearman, _ = spearmanr(true_returns, pred_returns)
            
            self.metrics['Returns_Pearson'] = returns_pearson
            self.metrics['Returns_Spearman'] = returns_spearman
    
    def _calculate_custom_metrics(self, y_true, y_pred, sample_weights=None):
        """Calculate custom Kaggle-style metrics."""
        # Weighted metrics if sample weights provided
        if sample_weights is not None:
            weighted_mae = np.average(np.abs(y_true - y_pred), weights=sample_weights)
            weighted_rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weights))
            self.metrics['Weighted_MAE'] = weighted_mae
            self.metrics['Weighted_RMSE'] = weighted_rmse
        
        # Custom loss function (similar to some Kaggle competitions)
        # Penalize more for larger errors
        custom_loss = np.mean(np.abs(y_true - y_pred) * (1 + np.abs(y_true - y_pred) / y_true))
        self.metrics['Custom_Loss'] = custom_loss
        
        # Relative error metrics
        relative_errors = np.abs(y_true - y_pred) / y_true
        self.metrics['Mean_Relative_Error'] = np.mean(relative_errors)
        self.metrics['Median_Relative_Error'] = np.median(relative_errors)
        self.metrics['Std_Relative_Error'] = np.std(relative_errors)
        
        # Quantile-based metrics
        q25 = np.percentile(relative_errors, 25)
        q75 = np.percentile(relative_errors, 75)
        self.metrics['Q25_Relative_Error'] = q25
        self.metrics['Q75_Relative_Error'] = q75
        self.metrics['IQR_Relative_Error'] = q75 - q25
        
        # Outlier detection (errors > 2 standard deviations)
        error_std = np.std(y_true - y_pred)
        outliers = np.sum(np.abs(y_true - y_pred) > 2 * error_std)
        outlier_rate = outliers / len(y_true)
        self.metrics['Outlier_Rate'] = outlier_rate
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown from cumulative returns."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        return max_drawdown
    
    def get_metric_summary(self):
        """Get a summary of all calculated metrics."""
        summary = {}
        
        # Group metrics by category
        basic_metrics = ['MAE', 'RMSE', 'MSE', 'R2', 'MAPE', 'SMAPE']
        timeseries_metrics = ['MASE', 'RMSPE', 'MASE_Rolling']
        financial_metrics = ['Sharpe_True', 'Sharpe_Pred', 'MaxDD_True', 'MaxDD_Pred', 'Volatility_True', 'Volatility_Pred']
        directional_metrics = ['Directional_Accuracy', 'Direction_Precision', 'Direction_Recall', 'Direction_F1']
        correlation_metrics = ['Pearson_Correlation', 'Spearman_Correlation', 'Returns_Pearson', 'Returns_Spearman']
        
        for metric in basic_metrics + timeseries_metrics + financial_metrics + directional_metrics + correlation_metrics:
            if metric in self.metrics:
                summary[metric] = self.metrics[metric]
        
        return summary
    
    def print_metrics_report(self):
        """Print a formatted report of all metrics."""
        print("=" * 60)
        print("KAGGLE-STYLE METRICS REPORT")
        print("=" * 60)
        
        categories = {
            'Basic Metrics': ['MAE', 'RMSE', 'MSE', 'R2', 'MAPE', 'SMAPE'],
            'Time Series Metrics': ['MASE', 'RMSPE', 'MASE_Rolling'],
            'Financial Metrics': ['Sharpe_True', 'Sharpe_Pred', 'MaxDD_True', 'MaxDD_Pred', 'Volatility_True', 'Volatility_Pred'],
            'Directional Metrics': ['Directional_Accuracy', 'Direction_Precision', 'Direction_Recall', 'Direction_F1'],
            'Correlation Metrics': ['Pearson_Correlation', 'Spearman_Correlation', 'Returns_Pearson', 'Returns_Spearman']
        }
        
        for category, metrics in categories.items():
            print(f"\n{category}:")
            print("-" * 30)
            for metric in metrics:
                if metric in self.metrics:
                    value = self.metrics[metric]
                    if isinstance(value, float):
                        print(f"{metric:25s}: {value:.6f}")
                    else:
                        print(f"{metric:25s}: {value}")
        
        # Print key performance indicators
        print("\n" + "=" * 60)
        print("KEY PERFORMANCE INDICATORS")
        print("=" * 60)
        
        key_metrics = ['R2', 'Directional_Accuracy', 'Pearson_Correlation', 'MAPE', 'MASE']
        for metric in key_metrics:
            if metric in self.metrics:
                value = self.metrics[metric]
                if isinstance(value, float):
                    print(f"{metric:25s}: {value:.6f}")
                else:
                    print(f"{metric:25s}: {value}")
        
        print("=" * 60) 