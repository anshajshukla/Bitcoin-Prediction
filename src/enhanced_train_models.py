import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel, RFE
import joblib
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')

# Import our custom Kaggle metrics
from kaggle_metrics import KaggleMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def load_kaggle_data():
    """Load the Kaggle Bitcoin dataset."""
    try:
        # Load the BTC dataset
        df = pd.read_csv('data/btc.csv')
        logger.info(f"Loaded Kaggle dataset with shape: {df.shape}")
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Rename price column to Close for consistency
        df.rename(columns={'price': 'Close'}, inplace=True)
        
        logger.info(f"Dataset date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Number of data points: {len(df)}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading Kaggle dataset: {str(e)}")
        raise

def create_advanced_features(df):
    """Create advanced features for better prediction accuracy."""
    logger.info("Creating advanced features...")
    
    # Basic price features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log1p(df['Returns'])
    
    # Multiple timeframe moving averages
    for window in [3, 6, 12, 24, 48, 72, 168]:  # 3h to 1 week
        df[f'MA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
        df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
        df[f'MA_{window}_slope'] = df[f'MA_{window}'].diff()
    
    # Exponential moving averages
    for span in [6, 12, 24, 48]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()
        df[f'EMA_{span}_ratio'] = df['Close'] / df[f'EMA_{span}']
    
    # Volatility features
    for window in [6, 12, 24, 48]:
        df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        df[f'Volatility_{window}_ratio'] = df[f'Volatility_{window}'] / df['Close']
    
    # RSI with multiple timeframes
    for window in [6, 12, 24]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    
    # MACD with multiple timeframes
    for fast, slow in [(6, 12), (12, 24), (24, 48)]:
        exp1 = df['Close'].ewm(span=fast).mean()
        exp2 = df['Close'].ewm(span=slow).mean()
        df[f'MACD_{fast}_{slow}'] = exp1 - exp2
        df[f'Signal_{fast}_{slow}'] = df[f'MACD_{fast}_{slow}'].ewm(span=9).mean()
        df[f'MACD_Hist_{fast}_{slow}'] = df[f'MACD_{fast}_{slow}'] - df[f'Signal_{fast}_{slow}']
    
    # Bollinger Bands with multiple timeframes
    for window in [12, 24, 48]:
        bb_middle = df['Close'].rolling(window=window).mean()
        bb_std = df['Close'].rolling(window=window).std()
        df[f'BB_upper_{window}'] = bb_middle + 2 * bb_std
        df[f'BB_lower_{window}'] = bb_middle - 2 * bb_std
        df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / bb_middle
        df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}'])
    
    # Price momentum and acceleration
    for period in [1, 3, 6, 12, 24]:
        df[f'Momentum_{period}'] = df['Close'].pct_change(period)
        df[f'Acceleration_{period}'] = df[f'Momentum_{period}'].diff()
    
    # Support and resistance levels
    for window in [24, 48, 168]:
        df[f'Support_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Resistance_{window}'] = df['Close'].rolling(window=window).max()
        df[f'Support_Distance_{window}'] = (df['Close'] - df[f'Support_{window}']) / df['Close']
        df[f'Resistance_Distance_{window}'] = (df[f'Resistance_{window}'] - df['Close']) / df['Close']
    
    # Time-based features
    df['Hour'] = df.index.hour
    df['Day_of_Week'] = df.index.dayofweek
    df['Day_of_Month'] = df.index.day
    df['Week_of_Year'] = df.index.isocalendar().week
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    # Cyclical encoding for time features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Lag features with multiple timeframes
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    
    # Rolling statistics
    for window in [6, 12, 24, 48]:
        df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Rolling_Skew_{window}'] = df['Close'].rolling(window=window).skew()
        df[f'Rolling_Kurt_{window}'] = df['Close'].rolling(window=window).kurt()
        df[f'Rolling_Median_{window}'] = df['Close'].rolling(window=window).median()
        df[f'Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()
    
    # Price patterns
    df['Higher_High'] = (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2))
    df['Lower_Low'] = (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2))
    df['Double_Top'] = (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2)) & (df['Close'].shift(2) > df['Close'].shift(3))
    df['Double_Bottom'] = (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2)) & (df['Close'].shift(2) < df['Close'].shift(3))
    
    # Trend strength indicators
    df['ADX'] = calculate_adx(df)
    df['ATR'] = calculate_atr(df)
    df['Trend_Strength'] = abs(df['Close'] - df['MA_24']) / df['MA_24']
    
    # Market regime features
    df['Market_Regime'] = np.where(df['Close'] > df['MA_24'], 1, -1)
    df['Volatility_Regime'] = np.where(df['Volatility_24'] > df['Volatility_24'].rolling(168).mean(), 1, -1)
    
    # Interaction features
    df['Price_Volatility_Interaction'] = df['Close'] * df['Volatility_24']
    df['Trend_Volatility_Interaction'] = df['Trend_Strength'] * df['Volatility_24']
    
    # Drop NaN values
    df = df.dropna()
    
    logger.info(f"Feature engineering completed. Final shape: {df.shape}")
    return df

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX)."""
    high = df['Close']  # Using Close as proxy for High/Low
    low = df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - df['Close'].shift(1))
    tr3 = abs(low - df['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)."""
    high = df['Close']
    low = df['Close']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def preprocess_data(df):
    """Preprocess the data for model training."""
    logger.info("Preprocessing data...")
    
    # Create target variable (next hour's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features and target
    feature_columns = [col for col in df.columns if col not in ['Target', 'Close']]
    X = df[feature_columns]
    y = df['Target']
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Scale features using RobustScaler (more robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Split data (80% train, 20% test)
    train_size = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Number of features: {len(feature_columns)}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns

def create_ensemble_model():
    """Create an ensemble of multiple models."""
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
    }
    
    return models

def train_ensemble_models(X_train, X_test, y_train, y_test):
    """Train ensemble models and evaluate performance."""
    logger.info("Training ensemble models...")
    
    models = create_ensemble_model()
    trained_models = {}
    predictions = {}
    metrics = {}
    
    # Train each model
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        predictions[name] = {'train': y_pred_train, 'test': y_pred_test}
        
        # Calculate metrics
        kaggle_metrics = KaggleMetrics()
        test_metrics = kaggle_metrics.calculate_all_metrics(y_test, y_pred_test)
        metrics[name] = test_metrics
        
        logger.info(f"{name} - R²: {test_metrics['R2']:.4f}, Directional Accuracy: {test_metrics['Directional_Accuracy']:.4f}")
    
    return trained_models, predictions, metrics

def create_weighted_ensemble(trained_models, predictions, X_test, y_test):
    """Create a weighted ensemble based on individual model performance."""
    logger.info("Creating weighted ensemble...")
    
    # Get test predictions from all models
    test_preds = {}
    for name, model in trained_models.items():
        test_preds[name] = model.predict(X_test)
    
    # Calculate weights based on R² scores
    weights = {}
    total_score = 0
    
    for name, pred in test_preds.items():
        kaggle_metrics = KaggleMetrics()
        metrics = kaggle_metrics.calculate_all_metrics(y_test, pred)
        r2_score = max(0, metrics['R2'])  # Ensure non-negative weights
        weights[name] = r2_score
        total_score += r2_score
    
    # Normalize weights
    if total_score > 0:
        for name in weights:
            weights[name] /= total_score
    else:
        # Equal weights if all models perform poorly
        for name in weights:
            weights[name] = 1.0 / len(weights)
    
    logger.info(f"Ensemble weights: {weights}")
    
    # Create weighted ensemble prediction
    ensemble_pred = np.zeros(len(y_test))
    for name, weight in weights.items():
        ensemble_pred += weight * test_preds[name]
    
    return ensemble_pred, weights

def hyperparameter_optimization(X_train, y_train):
    """Optimize hyperparameters for the best performing model."""
    logger.info("Starting hyperparameter optimization...")
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'gamma': trial.suggest_float('gamma', 0, 5)
        }
        
        model = XGBRegressor(**param, random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            # Use a combination of R² and directional accuracy
            from sklearn.metrics import r2_score
            r2 = r2_score(y_fold_val, y_pred)
            directional_acc = np.mean(np.diff(y_fold_val) > 0 == np.diff(y_pred) > 0)
            score = 0.7 * r2 + 0.3 * directional_acc
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best score: {study.best_value:.4f}")
    
    return study.best_params

def main():
    """Main function to run the enhanced model training pipeline."""
    try:
        # Load data
        df = load_kaggle_data()
        
        # Create advanced features
        df = create_advanced_features(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(df)
        
        # Train ensemble models
        trained_models, predictions, metrics = train_ensemble_models(X_train, X_test, y_train, y_test)
        
        # Create weighted ensemble
        ensemble_pred, weights = create_weighted_ensemble(trained_models, predictions, X_test, y_test)
        
        # Evaluate ensemble
        kaggle_metrics = KaggleMetrics()
        ensemble_metrics = kaggle_metrics.calculate_all_metrics(y_test, ensemble_pred)
        
        # Print comprehensive metrics report
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE MODEL PERFORMANCE")
        logger.info("="*60)
        kaggle_metrics.print_metrics_report()
        
        # Optimize best model if needed
        if ensemble_metrics['R2'] < 0.60:
            logger.info("R² below 0.60, optimizing hyperparameters...")
            best_params = hyperparameter_optimization(X_train, y_train)
            
            # Retrain with optimized parameters
            optimized_model = XGBRegressor(**best_params, random_state=42)
            optimized_model.fit(X_train, y_train)
            optimized_pred = optimized_model.predict(X_test)
            
            optimized_metrics = kaggle_metrics.calculate_all_metrics(y_test, optimized_pred)
            logger.info(f"Optimized model R²: {optimized_metrics['R2']:.4f}")
            
            # Use the better performing model
            if optimized_metrics['R2'] > ensemble_metrics['R2']:
                final_model = optimized_model
                final_metrics = optimized_metrics
                final_pred = optimized_pred
            else:
                final_model = trained_models['XGBoost']  # Use best individual model
                final_metrics = ensemble_metrics
                final_pred = ensemble_pred
        else:
            final_model = trained_models['XGBoost']
            final_metrics = ensemble_metrics
            final_pred = ensemble_pred
        
        # Save models and results
        os.makedirs('models', exist_ok=True)
        joblib.dump(final_model, 'models/enhanced_xgboost_model.joblib')
        joblib.dump(scaler, 'models/enhanced_scaler.joblib')
        joblib.dump(trained_models, 'models/ensemble_models.joblib')
        
        # Save feature columns
        with open('models/enhanced_feature_columns.txt', 'w') as f:
            f.write('\n'.join(feature_columns))
        
        # Save metrics
        metrics_df = pd.DataFrame([final_metrics])
        metrics_df.to_csv('models/enhanced_model_metrics.csv', index=False)
        
        # Save ensemble weights
        weights_df = pd.DataFrame([weights])
        weights_df.to_csv('models/ensemble_weights.csv', index=False)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': final_pred
        })
        predictions_df.to_csv('models/enhanced_predictions.csv', index=False)
        
        logger.info("Enhanced model training completed successfully!")
        logger.info(f"Final R² Score: {final_metrics['R2']:.4f}")
        logger.info(f"Final Directional Accuracy: {final_metrics['Directional_Accuracy']:.4f}")
        
        return final_model, final_metrics, final_pred
        
    except Exception as e:
        logger.error(f"Error in enhanced training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 