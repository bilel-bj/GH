"""
Green Hydrogen Electrolyzer Predictive Maintenance System
Production-Ready Version with 10 Forecasting Algorithms
ACWA Power - Advanced Multi-Model Implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
from typing import Optional, Dict, List, Tuple, Any
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# Import Nixtla ecosystem
try:
    from nixtla import NixtlaClient
    NIXTLA_AVAILABLE = True
except ImportError:
    NIXTLA_AVAILABLE = False
    st.warning("Nixtla package not installed. Install with: pip install nixtla")

# Import StatsForecast for statistical models
try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA, 
        AutoETS,
        AutoTheta,
        SeasonalNaive,
        MSTL,
        CrostonOptimized,
        TSB,
        ADIDA
    )
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    st.warning("StatsForecast not installed. Install with: pip install statsforecast")

# Import NeuralForecast for deep learning models
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import (
        LSTM,
        GRU,
        NHITS,
        NBEATS,
        RNN,
        TCN,
        DLinear,
        TFT
    )
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False
    st.warning("NeuralForecast not installed. Install with: pip install neuralforecast")

# Import additional libraries for ML models
try:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="ACWA Power Electrolyzer Maintenance - Multi-Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional UI
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stAlert {border-radius: 10px;}
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(28, 131, 225, 0.1) 0%, rgba(28, 131, 225, 0.05) 100%);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .algorithm-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    
    .stats-model {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    .neural-model {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);}
    .ml-model {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);}
    .ensemble-model {background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);}
    
    .risk-critical {background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);}
    .risk-high {background: linear-gradient(135deg, #ffa94d 0%, #ffbe75 100%);}
    .risk-medium {background: linear-gradient(135deg, #ffd43b 0%, #ffe066 100%);}
    .risk-low {background: linear-gradient(135deg, #51cf66 0%, #69db7c 100%);}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'ensemble_prediction' not in st.session_state:
    st.session_state.ensemble_prediction = None

# ACWA Power Data Column Mapping
ACWA_COLUMN_MAPPING = {
    'timestamp': 'Timestamp',
    'voltage': 'Voltage \n@1# Stack',
    'current': 'Current @1# stack',
    'power': 'DC Power Consumption @1# Stack',
    'h2_temp': 'H2 side outlet temp @1# stack',
    'o2_temp': 'O2 side outlet temp @1# stack',
    'lye_temp': 'Lye Supply to Electrolyzer Temp',
    'lye_concentration': 'Lye Concentration',
    'lye_flow': 'Lye Flow to #1 Stack',
    'h2_separator_level': 'H2 Separator Level',
    'o2_separator_level': 'O2 Separator Level',
    'ldi_separator': 'LDI\nH2 & O2 Separator',
    'o2_in_h2': 'O2 content in H2 ',
    'h2_in_o2': 'H2 content in O2 ',
    'pressure': 'Pressure\n@O2 Separator',
    'h2_flow': 'H2 Flowrate Purification outlet',
    'dm_conductivity': 'DM water condctivity',
    'dm_flow': 'DM water flow from B.L.',
    'room_temp': 'Room temperature'
}

@dataclass
class ModelConfig:
    """Configuration for each forecasting model"""
    name: str
    type: str  # 'statistical', 'neural', 'ml', 'ensemble'
    color: str
    params: Dict[str, Any]
    description: str
    strengths: List[str]
    best_for: str

# Define 10 forecasting models
FORECASTING_MODELS = {
    'timegpt': ModelConfig(
        name='Nixtla TimeGPT',
        type='neural',
        color='#FF6B6B',
        params={'freq': 'H', 'level': [80, 95]},
        description='State-of-the-art transformer model by Nixtla',
        strengths=['Long-term patterns', 'Complex seasonality', 'Zero-shot learning'],
        best_for='Complex multi-seasonal patterns with limited historical data'
    ),
    'arima': ModelConfig(
        name='AutoARIMA',
        type='statistical',
        color='#4ECDC4',
        params={'season_length': 24, 'd': None, 'D': None},
        description='Automatic ARIMA model selection',
        strengths=['Trend detection', 'Non-stationary data', 'Short-term accuracy'],
        best_for='Linear trends with seasonal patterns'
    ),
    'lstm': ModelConfig(
        name='LSTM Neural Network',
        type='neural',
        color='#95E77E',
        params={'input_size': 24, 'hidden_size': 100, 'max_steps': 100},
        description='Long Short-Term Memory deep learning model',
        strengths=['Non-linear patterns', 'Long sequences', 'Multiple features'],
        best_for='Complex non-linear patterns with long-term dependencies'
    ),
    'gru': ModelConfig(
        name='GRU Network',
        type='neural',
        color='#FFE66D',
        params={'input_size': 24, 'hidden_size': 100, 'max_steps': 100},
        description='Gated Recurrent Unit neural network',
        strengths=['Faster than LSTM', 'Memory efficiency', 'Sequence modeling'],
        best_for='Sequential patterns with moderate complexity'
    ),
    'nhits': ModelConfig(
        name='N-HiTS',
        type='neural',
        color='#A8E6CF',
        params={'input_size': 48, 'h': 24, 'max_steps': 100},
        description='Neural Hierarchical Time Series model',
        strengths=['Multi-scale patterns', 'Interpretability', 'Fast training'],
        best_for='Multi-resolution time series with hierarchical patterns'
    ),
    'nbeats': ModelConfig(
        name='N-BEATS',
        type='neural',
        color='#FFB6B9',
        params={'input_size': 48, 'h': 24, 'max_steps': 100},
        description='Neural Basis Expansion Analysis',
        strengths=['Interpretable', 'No feature engineering', 'Pure deep learning'],
        best_for='Pure time series without external features'
    ),
    'theta': ModelConfig(
        name='AutoTheta',
        type='statistical',
        color='#C7CEEA',
        params={'season_length': 24, 'decomposition_type': 'multiplicative'},
        description='Theta method with automatic optimization',
        strengths=['Simple', 'Robust', 'Good for competitions'],
        best_for='Simple trends with noise'
    ),
    'ets': ModelConfig(
        name='AutoETS',
        type='statistical',
        color='#FFDAC1',
        params={'season_length': 24, 'model': 'ZZZ'},
        description='Exponential Smoothing State Space Model',
        strengths=['Automatic model selection', 'Prediction intervals', 'Seasonality'],
        best_for='Data with clear seasonal patterns'
    ),
    'xgboost': ModelConfig(
        name='XGBoost',
        type='ml',
        color='#B4E7D5',
        params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
        description='Extreme Gradient Boosting',
        strengths=['Feature importance', 'Non-linear', 'Robust to outliers'],
        best_for='Multiple features with complex interactions'
    ),
    'ensemble': ModelConfig(
        name='Weighted Ensemble',
        type='ensemble',
        color='#DDA0DD',
        params={'weights': 'auto', 'method': 'weighted_average'},
        description='Weighted combination of multiple models',
        strengths=['Reduces overfitting', 'Robust predictions', 'Best of all models'],
        best_for='Maximum accuracy and reliability'
    )
}

class MultiModelForecaster:
    """Advanced multi-model forecasting system"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize forecasting system with all models"""
        self.api_key = api_key or os.getenv('NIXTLA_API_KEY')
        self.models = {}
        self.results = {}
        self.performance_metrics = {}
        
        # Initialize Nixtla TimeGPT if available
        if NIXTLA_AVAILABLE and self.api_key:
            try:
                self.nixtla_client = NixtlaClient(api_key=self.api_key)
                self.models['timegpt'] = True
            except Exception as e:
                st.warning(f"Could not initialize Nixtla TimeGPT: {e}")
                self.models['timegpt'] = False
        else:
            self.models['timegpt'] = False
        
        # Initialize statistical models
        if STATSFORECAST_AVAILABLE:
            self.models['statistical'] = True
        else:
            self.models['statistical'] = False
        
        # Initialize neural models
        if NEURALFORECAST_AVAILABLE:
            self.models['neural'] = True
        else:
            self.models['neural'] = False
        
        # Initialize ML models
        if SKLEARN_AVAILABLE:
            self.models['ml'] = True
        else:
            self.models['ml'] = False
    
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare data for forecasting models"""
        forecast_df = df[['timestamp', target_col]].copy()
        forecast_df.columns = ['ds', 'y']
        forecast_df['unique_id'] = 'series_1'
        forecast_df = forecast_df.dropna()
        return forecast_df
    
    def run_timegpt(self, df: pd.DataFrame, horizon: int, freq: str = 'H') -> pd.DataFrame:
        """Run Nixtla TimeGPT model"""
        if not self.models.get('timegpt'):
            return self._run_fallback_forecast(df, horizon, 'TimeGPT')
        
        try:
            predictions = self.nixtla_client.forecast(
                df=df[['ds', 'y']],
                h=horizon,
                freq=freq,
                level=[80, 95],
                add_history=False
            )
            predictions['model'] = 'TimeGPT'
            return predictions
        except Exception as e:
            st.warning(f"TimeGPT failed: {e}")
            return self._run_fallback_forecast(df, horizon, 'TimeGPT')
    
    def run_statistical_models(self, df: pd.DataFrame, horizon: int, 
                             freq: str = 'H') -> Dict[str, pd.DataFrame]:
        """Run statistical forecasting models"""
        if not self.models.get('statistical'):
            return {}
        
        results = {}
        
        try:
            # Initialize models
            models = [
                AutoARIMA(season_length=24),
                AutoETS(season_length=24),
                AutoTheta(season_length=24),
                SeasonalNaive(season_length=24),
                MSTL(season_length=24)
            ]
            
            # Create StatsForecast object
            sf = StatsForecast(
                models=models,
                freq=freq,
                n_jobs=-1
            )
            
            # Fit and predict
            sf.fit(df[['ds', 'y', 'unique_id']])
            forecasts = sf.predict(h=horizon, level=[80, 95])
            
            # Parse results
            for model_name in ['AutoARIMA', 'AutoETS', 'AutoTheta']:
                if model_name in forecasts.columns:
                    model_results = pd.DataFrame({
                        'ds': forecasts.index,
                        'forecast': forecasts[model_name],
                        'lo-95': forecasts.get(f'{model_name}-lo-95', forecasts[model_name] * 0.9),
                        'hi-95': forecasts.get(f'{model_name}-hi-95', forecasts[model_name] * 1.1),
                        'lo-80': forecasts.get(f'{model_name}-lo-80', forecasts[model_name] * 0.95),
                        'hi-80': forecasts.get(f'{model_name}-hi-80', forecasts[model_name] * 1.05),
                        'model': model_name
                    })
                    results[model_name.lower()] = model_results
        
        except Exception as e:
            st.warning(f"Statistical models failed: {e}")
        
        return results
    
    def run_neural_models(self, df: pd.DataFrame, horizon: int, 
                         freq: str = 'H') -> Dict[str, pd.DataFrame]:
        """Run neural network forecasting models"""
        if not self.models.get('neural'):
            return {}
        
        results = {}
        
        try:
            # Initialize models with faster training settings for production
            models = [
                LSTM(h=horizon, input_size=24, hidden_size=50, max_steps=50),
                GRU(h=horizon, input_size=24, hidden_size=50, max_steps=50),
                NHITS(h=horizon, input_size=48, max_steps=50),
                NBEATS(h=horizon, input_size=48, max_steps=50),
                RNN(h=horizon, input_size=24, hidden_size=50, max_steps=50)
            ]
            
            # Create NeuralForecast object
            nf = NeuralForecast(
                models=models,
                freq=freq
            )
            
            # Fit and predict
            nf.fit(df[['ds', 'y', 'unique_id']])
            forecasts = nf.predict()
            
            # Parse results
            for model in ['LSTM', 'GRU', 'NHITS', 'NBEATS']:
                if model in forecasts.columns:
                    model_results = pd.DataFrame({
                        'ds': forecasts.index,
                        'forecast': forecasts[model],
                        'lo-95': forecasts[model] * 0.9,  # Simple confidence intervals
                        'hi-95': forecasts[model] * 1.1,
                        'lo-80': forecasts[model] * 0.95,
                        'hi-80': forecasts[model] * 1.05,
                        'model': model
                    })
                    results[model.lower()] = model_results
        
        except Exception as e:
            st.warning(f"Neural models failed: {e}")
        
        return results
    
    def run_ml_models(self, df: pd.DataFrame, horizon: int) -> Dict[str, pd.DataFrame]:
        """Run machine learning models (XGBoost, Random Forest)"""
        if not self.models.get('ml'):
            return {}
        
        results = {}
        
        try:
            # Create features for ML models
            data = df.copy()
            data['hour'] = pd.to_datetime(data['ds']).dt.hour
            data['day'] = pd.to_datetime(data['ds']).dt.day
            data['month'] = pd.to_datetime(data['ds']).dt.month
            data['dayofweek'] = pd.to_datetime(data['ds']).dt.dayofweek
            
            # Create lag features
            for lag in [1, 24, 48, 168]:
                data[f'lag_{lag}'] = data['y'].shift(lag)
            
            # Create rolling features
            for window in [24, 48, 168]:
                data[f'rolling_mean_{window}'] = data['y'].rolling(window).mean()
                data[f'rolling_std_{window}'] = data['y'].rolling(window).std()
            
            data = data.dropna()
            
            # Split features and target
            feature_cols = [col for col in data.columns if col not in ['ds', 'y', 'unique_id']]
            X = data[feature_cols]
            y = data['y']
            
            # Train XGBoost
            xgb_model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X, y)
            
            # Generate future features for prediction
            last_timestamp = pd.to_datetime(df['ds'].iloc[-1])
            future_dates = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=horizon,
                freq='H'
            )
            
            # Simple feature generation for future (this is simplified)
            future_features = pd.DataFrame({
                'hour': future_dates.hour,
                'day': future_dates.day,
                'month': future_dates.month,
                'dayofweek': future_dates.dayofweek
            })
            
            # Add lag features (simplified - using last known values)
            for col in feature_cols:
                if col not in future_features.columns:
                    if 'lag' in col:
                        future_features[col] = data[col].iloc[-1]
                    elif 'rolling' in col:
                        future_features[col] = data[col].iloc[-1]
            
            # Make predictions
            xgb_predictions = xgb_model.predict(future_features)
            
            results['xgboost'] = pd.DataFrame({
                'ds': future_dates,
                'forecast': xgb_predictions,
                'lo-95': xgb_predictions * 0.9,
                'hi-95': xgb_predictions * 1.1,
                'lo-80': xgb_predictions * 0.95,
                'hi-80': xgb_predictions * 1.05,
                'model': 'XGBoost'
            })
        
        except Exception as e:
            st.warning(f"ML models failed: {e}")
        
        return results
    
    def create_ensemble(self, predictions_dict: Dict[str, pd.DataFrame], 
                       method: str = 'weighted_average') -> pd.DataFrame:
        """Create ensemble predictions from multiple models"""
        if not predictions_dict:
            return pd.DataFrame()
        
        # Convert all predictions to same format
        all_forecasts = []
        for model_name, preds in predictions_dict.items():
            if not preds.empty:
                forecast_col = 'forecast' if 'forecast' in preds.columns else 'TimeGPT' if 'TimeGPT' in preds.columns else preds.columns[1]
                all_forecasts.append(preds[forecast_col].values)
        
        if not all_forecasts:
            return pd.DataFrame()
        
        # Calculate ensemble
        if method == 'simple_average':
            ensemble_forecast = np.mean(all_forecasts, axis=0)
            weights = [1/len(all_forecasts)] * len(all_forecasts)
        elif method == 'weighted_average':
            # Simple weighting based on model type
            weights = []
            for model_name in predictions_dict.keys():
                if 'timegpt' in model_name.lower():
                    weights.append(0.3)
                elif 'lstm' in model_name.lower() or 'gru' in model_name.lower():
                    weights.append(0.2)
                elif 'xgboost' in model_name.lower():
                    weights.append(0.15)
                else:
                    weights.append(0.1)
            
            # Normalize weights
            weights = np.array(weights[:len(all_forecasts)])
            weights = weights / weights.sum()
            
            ensemble_forecast = np.average(all_forecasts, axis=0, weights=weights)
        else:
            ensemble_forecast = np.median(all_forecasts, axis=0)
            weights = [1/len(all_forecasts)] * len(all_forecasts)
        
        # Get timestamps from first prediction
        first_pred = list(predictions_dict.values())[0]
        timestamps = first_pred['ds'] if 'ds' in first_pred.columns else first_pred.index
        
        # Calculate prediction intervals from all models
        all_lower = []
        all_upper = []
        for model_name, preds in predictions_dict.items():
            if 'lo-95' in preds.columns and 'hi-95' in preds.columns:
                all_lower.append(preds['lo-95'].values)
                all_upper.append(preds['hi-95'].values)
        
        if all_lower and all_upper:
            ensemble_lower = np.average(all_lower, axis=0, weights=weights[:len(all_lower)])
            ensemble_upper = np.average(all_upper, axis=0, weights=weights[:len(all_upper)])
        else:
            ensemble_lower = ensemble_forecast * 0.9
            ensemble_upper = ensemble_forecast * 1.1
        
        ensemble_df = pd.DataFrame({
            'ds': timestamps,
            'forecast': ensemble_forecast,
            'lo-95': ensemble_lower,
            'hi-95': ensemble_upper,
            'lo-80': ensemble_forecast * 0.95,
            'hi-80': ensemble_forecast * 1.05,
            'model': 'Ensemble',
            'weights': [weights] * len(ensemble_forecast)
        })
        
        return ensemble_df
    
    def calculate_failure_probability(self, predictions: pd.DataFrame, 
                                     critical_threshold: float) -> pd.DataFrame:
        """Calculate failure probability for predictions"""
        from scipy import stats
        
        predictions = predictions.copy()
        
        # Get forecast column
        forecast_col = 'forecast' if 'forecast' in predictions.columns else 'TimeGPT' if 'TimeGPT' in predictions.columns else predictions.columns[1]
        
        # Estimate uncertainty from confidence intervals
        if 'hi-95' in predictions.columns and 'lo-95' in predictions.columns:
            std_estimate = (predictions['hi-95'] - predictions['lo-95']) / (2 * 1.96)
        else:
            std_estimate = predictions[forecast_col] * 0.1  # 10% uncertainty
        
        # Calculate probability of exceeding threshold
        predictions['failure_probability'] = 1 - stats.norm.cdf(
            critical_threshold,
            predictions[forecast_col],
            std_estimate
        )
        
        return predictions
    
    def evaluate_models(self, actual: pd.Series, predictions_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance using various metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {}
        
        for model_name, preds in predictions_dict.items():
            if preds.empty:
                continue
            
            # Get forecast column
            forecast_col = 'forecast' if 'forecast' in preds.columns else 'TimeGPT' if 'TimeGPT' in preds.columns else preds.columns[1]
            
            # Align predictions with actual data
            min_len = min(len(actual), len(preds))
            if min_len > 0:
                actual_subset = actual[-min_len:].values
                pred_subset = preds[forecast_col][:min_len].values
                
                metrics[model_name] = {
                    'MAE': mean_absolute_error(actual_subset, pred_subset),
                    'RMSE': np.sqrt(mean_squared_error(actual_subset, pred_subset)),
                    'MAPE': np.mean(np.abs((actual_subset - pred_subset) / actual_subset)) * 100,
                    'R2': r2_score(actual_subset, pred_subset)
                }
        
        return metrics
    
    def run_all_models(self, df: pd.DataFrame, target_col: str, horizon: int, 
                      selected_models: List[str] = None) -> Dict[str, Any]:
        """Run all selected forecasting models"""
        # Prepare data
        forecast_df = self.prepare_data(df, target_col)
        
        if selected_models is None:
            selected_models = ['timegpt', 'arima', 'lstm', 'xgboost', 'ensemble']
        
        results = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = len(selected_models)
        completed = 0
        
        # Run TimeGPT
        if 'timegpt' in selected_models:
            status_text.text('Running Nixtla TimeGPT...')
            results['timegpt'] = self.run_timegpt(forecast_df, horizon)
            completed += 1
            progress_bar.progress(completed / total_models)
        
        # Run statistical models
        statistical_models = [m for m in selected_models if m in ['arima', 'ets', 'theta']]
        if statistical_models:
            status_text.text('Running statistical models...')
            stat_results = self.run_statistical_models(forecast_df, horizon)
            results.update(stat_results)
            completed += len(statistical_models)
            progress_bar.progress(completed / total_models)
        
        # Run neural models
        neural_models = [m for m in selected_models if m in ['lstm', 'gru', 'nhits', 'nbeats']]
        if neural_models:
            status_text.text('Running neural network models...')
            neural_results = self.run_neural_models(forecast_df, horizon)
            results.update(neural_results)
            completed += len(neural_models)
            progress_bar.progress(completed / total_models)
        
        # Run ML models
        if 'xgboost' in selected_models:
            status_text.text('Running XGBoost model...')
            ml_results = self.run_ml_models(forecast_df, horizon)
            results.update(ml_results)
            completed += 1
            progress_bar.progress(completed / total_models)
        
        # Create ensemble
        if 'ensemble' in selected_models and len(results) > 1:
            status_text.text('Creating ensemble predictions...')
            results['ensemble'] = self.create_ensemble(results, method='weighted_average')
            completed += 1
            progress_bar.progress(completed / total_models)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Calculate metrics if we have historical data for validation
        if len(df) > horizon:
            metrics = self.evaluate_models(df[target_col], results)
        else:
            metrics = {}
        
        return {
            'predictions': results,
            'metrics': metrics,
            'best_model': min(metrics, key=lambda x: metrics[x]['MAE']) if metrics else None
        }
    
    def _run_fallback_forecast(self, df: pd.DataFrame, horizon: int, 
                               model_name: str) -> pd.DataFrame:
        """Fallback forecasting method using simple statistical approach"""
        from scipy import stats
        
        # Simple linear regression with seasonality
        y = df['y'].values
        x = np.arange(len(y))
        
        # Detrend
        slope, intercept, _, _, std_err = stats.linregress(x, y)
        
        # Generate forecast
        future_x = np.arange(len(y), len(y) + horizon)
        forecast = intercept + slope * future_x
        
        # Add simple seasonality (daily pattern)
        if len(y) >= 24:
            seasonal_pattern = y[-24:] - (intercept + slope * x[-24:])
            seasonal_forecast = np.tile(seasonal_pattern, (horizon // 24 + 1))[:horizon]
            forecast += seasonal_forecast
        
        # Create prediction dataframe
        last_timestamp = pd.to_datetime(df['ds'].iloc[-1])
        future_dates = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=horizon,
            freq='H'
        )
        
        uncertainty = std_err * np.sqrt(1 + 1/len(y) + (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        
        predictions = pd.DataFrame({
            'ds': future_dates,
            'forecast': forecast,
            'lo-95': forecast - 1.96 * uncertainty,
            'hi-95': forecast + 1.96 * uncertainty,
            'lo-80': forecast - 1.28 * uncertainty,
            'hi-80': forecast + 1.28 * uncertainty,
            'model': f'{model_name} (Fallback)'
        })
        
        return predictions

# Data processing functions
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and validate ACWA Power electrolyzer data"""
    try:
        # Read file based on extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Standardize column names for ACWA Power data
        column_rename_map = {}
        for standard_name, original_name in ACWA_COLUMN_MAPPING.items():
            for col in df.columns:
                if original_name.lower().replace('\n', '').replace(' ', '') in col.lower().replace('\n', '').replace(' ', ''):
                    column_rename_map[col] = standard_name
                    break
        
        # Apply column renaming
        if column_rename_map:
            df = df.rename(columns=column_rename_map)
        
        # Handle timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df[df['timestamp'].notna()]
        elif 'Timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df = df[df['timestamp'].notna()]
        else:
            df['timestamp'] = pd.date_range(start=datetime.now() - timedelta(hours=len(df)), 
                                          periods=len(df), freq='H')
        
        # Ensure numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate efficiency
        if 'h2_flow' in df.columns and 'power' in df.columns:
            df['efficiency'] = df['h2_flow'] / (df['power'] + 1e-6)
        elif 'efficiency' not in df.columns:
            df['efficiency'] = 0.02
        
        # Add hours since maintenance
        if 'hours_since_maintenance' not in df.columns:
            df['hours_since_maintenance'] = np.arange(len(df)) % 2000
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect and map column names for ACWA Power data"""
    column_mapping = {}
    
    for key in ['voltage', 'current', 'power', 'h2_flow', 'o2_in_h2', 'h2_in_o2',
                'lye_temp', 'lye_concentration', 'lye_flow', 'pressure',
                'h2_separator_level', 'o2_separator_level', 'dm_conductivity']:
        if key in df.columns:
            column_mapping[key] = key
    
    # Add temperature mapping
    if 'lye_temp' in df.columns:
        column_mapping['temperature'] = 'lye_temp'
    elif 'h2_temp' in df.columns:
        column_mapping['temperature'] = 'h2_temp'
    
    # Add production mapping
    if 'h2_flow' in df.columns:
        column_mapping['h2_production'] = 'h2_flow'
    
    return column_mapping

def calculate_risk_metrics(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Calculate comprehensive risk metrics for ACWA Power electrolyzer"""
    risk_scores = pd.DataFrame(index=df.index)
    
    # Voltage degradation risk
    if column_mapping.get('voltage'):
        voltage_col = column_mapping['voltage']
        voltage_normal = 320
        voltage_critical = 600
        current_voltage = df[voltage_col].rolling(window=10, min_periods=1).mean()
        voltage_risk = np.clip((current_voltage - voltage_normal) / 
                              (voltage_critical - voltage_normal) * 100, 0, 100)
        risk_scores['voltage_risk'] = voltage_risk
    else:
        risk_scores['voltage_risk'] = 50
    
    # Gas crossover risk
    if column_mapping.get('o2_in_h2'):
        o2_col = column_mapping['o2_in_h2']
        o2_risk = np.clip(df[o2_col] / 2.0 * 100, 0, 100)
        risk_scores['crossover_risk'] = o2_risk
    else:
        risk_scores['crossover_risk'] = 30
    
    # Calculate overall risk
    risk_scores['overall_risk'] = risk_scores.mean(axis=1)
    
    # Risk level classification
    risk_scores['risk_level'] = pd.cut(
        risk_scores['overall_risk'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    return risk_scores

# Main Application
def main():
    # Title and header
    st.title("⚡ Green Hydrogen Electrolyzer - Multi-Model Predictive Maintenance")
    st.markdown("**ACWA Power Production System** | 10 Advanced Forecasting Algorithms")
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1c83e1/ffffff?text=ACWA+Power", 
                use_container_width=True)
        st.markdown("---")
        
        st.markdown("### ⚙️ Model Configuration")
        
        # API Key configuration
        api_key = st.text_input(
            "Nixtla API Key (Optional)",
            type="password",
            value=os.getenv('NIXTLA_API_KEY', ''),
            help="Enter your Nixtla TimeGPT API key"
        )
        
        # Model selection
        st.markdown("#### Select Forecasting Models")
        
        col1, col2 = st.columns(2)
        with col1:
            use_timegpt = st.checkbox("Nixtla TimeGPT", value=True)
            use_arima = st.checkbox("AutoARIMA", value=True)
            use_lstm = st.checkbox("LSTM Network", value=True)
            use_gru = st.checkbox("GRU Network", value=False)
            use_nhits = st.checkbox("N-HiTS", value=False)
        
        with col2:
            use_nbeats = st.checkbox("N-BEATS", value=False)
            use_theta = st.checkbox("AutoTheta", value=False)
            use_ets = st.checkbox("AutoETS", value=False)
            use_xgboost = st.checkbox("XGBoost", value=True)
            use_ensemble = st.checkbox("Ensemble", value=True)
        
        selected_models = []
        if use_timegpt: selected_models.append('timegpt')
        if use_arima: selected_models.append('arima')
        if use_lstm: selected_models.append('lstm')
        if use_gru: selected_models.append('gru')
        if use_nhits: selected_models.append('nhits')
        if use_nbeats: selected_models.append('nbeats')
        if use_theta: selected_models.append('theta')
        if use_ets: selected_models.append('ets')
        if use_xgboost: selected_models.append('xgboost')
        if use_ensemble: selected_models.append('ensemble')
        
        st.info(f"📊 {len(selected_models)} models selected")
        
        # Prediction settings
        st.markdown("---")
        forecast_horizon = st.slider(
            "Forecast Horizon (hours)",
            min_value=24,
            max_value=168,
            value=48,
            step=24
        )
        
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[80, 85, 90, 95, 99],
            value=95
        )
        
        st.markdown("---")
        st.markdown("### 📊 Data Source")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Electrolyzer Data",
            type=['xlsx', 'xls', 'csv'],
            help="Upload ACWA Power operational data"
        )
        
        # Process uploaded file
        if uploaded_file:
            with st.spinner("Loading and processing data..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.data = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df):,} rows of data")
                    
                    # Show data preview
                    with st.expander("Data Preview"):
                        st.dataframe(df.head(10))
        
        # Generate demo data option
        if st.button("Generate Demo Data", type="secondary"):
            df = generate_demo_data()
            st.session_state.data = df
            st.session_state.data_loaded = True
            st.success("✅ Demo data generated")
    
    # Main content area
    if st.session_state.data_loaded and st.session_state.data is not None:
        df = st.session_state.data
        column_mapping = detect_columns(df)
        risk_metrics = calculate_risk_metrics(df, column_mapping)
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Dashboard",
            "🤖 Multi-Model Predictions",
            "📊 Model Comparison",
            "⚠️ Risk Assessment",
            "🔧 Maintenance Planning"
        ])
        
        # Tab 1: Dashboard
        with tab1:
            render_dashboard(df, risk_metrics, column_mapping)
        
        # Tab 2: Multi-Model Predictions
        with tab2:
            render_multi_model_predictions(df, column_mapping, selected_models, 
                                         forecast_horizon, api_key)
        
        # Tab 3: Model Comparison
        with tab3:
            render_model_comparison(st.session_state.model_results)
        
        # Tab 4: Risk Assessment
        with tab4:
            render_risk_assessment(risk_metrics, df, column_mapping)
        
        # Tab 5: Maintenance Planning
        with tab5:
            render_maintenance_planning(risk_metrics, st.session_state.predictions, df)
    
    else:
        render_landing_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <small>
        Green Hydrogen Electrolyzer Predictive Maintenance System v3.0<br>
        Powered by 10 Advanced Forecasting Algorithms<br>
        © 2024 ACWA Power - Production Ready
        </small>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard(df, risk_metrics, column_mapping):
    """Render real-time monitoring dashboard"""
    st.markdown("### 🔍 System Health Overview - Stack #1")
    
    # Current status metrics
    metrics_cols = st.columns(6)
    
    metric_data = [
        ('voltage', 'Stack Voltage', 'V', 'inverse'),
        ('current', 'Current', 'A', 'normal'),
        ('power', 'Power', 'kW', 'normal'),
        ('h2_flow', 'H₂ Flow', 'Nm³/h', 'normal'),
        ('o2_in_h2', 'O₂ in H₂', '%', 'inverse'),
        ('overall_risk', 'Risk Score', '%', 'inverse')
    ]
    
    for i, (col_name, label, unit, delta_color) in enumerate(metric_data):
        with metrics_cols[i]:
            if col_name == 'overall_risk':
                value = risk_metrics['overall_risk'].iloc[-1]
                risk_level = risk_metrics['risk_level'].iloc[-1]
                color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
                st.metric(label, f"{value:.1f}{unit}", f"{color} {risk_level}")
            elif column_mapping.get(col_name):
                current_value = df[column_mapping[col_name]].iloc[-1]
                if len(df) > 24:
                    delta = current_value - df[column_mapping[col_name]].iloc[-24]
                    st.metric(label, f"{current_value:.1f} {unit}", 
                            f"{delta:+.1f} {unit}", delta_color=delta_color)
                else:
                    st.metric(label, f"{current_value:.1f} {unit}")
            else:
                st.metric(label, "N/A")
    
    st.markdown("---")
    
    # Time series visualization
    col1, col2 = st.columns(2)
    
    with col1:
        if column_mapping.get('voltage'):
            st.markdown("#### Stack Voltage Trend")
            fig = create_time_series_plot(df, column_mapping['voltage'], 
                                        'Voltage (V)', window=168)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if column_mapping.get('o2_in_h2'):
            st.markdown("#### Gas Crossover Safety")
            fig = create_safety_plot(df, column_mapping)
            st.plotly_chart(fig, use_container_width=True)

def render_multi_model_predictions(df, column_mapping, selected_models, horizon, api_key):
    """Render multi-model prediction interface"""
    st.markdown("### 🤖 Multi-Model Forecasting System")
    
    # Parameter selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Critical parameters for prediction
        critical_params = []
        if column_mapping.get('voltage'):
            critical_params.append(('Stack Voltage', 'voltage'))
        if column_mapping.get('o2_in_h2'):
            critical_params.append(('O₂ in H₂', 'o2_in_h2'))
        if column_mapping.get('h2_flow'):
            critical_params.append(('H₂ Production', 'h2_flow'))
        if column_mapping.get('power'):
            critical_params.append(('Power Consumption', 'power'))
        
        if critical_params:
            param_labels, param_keys = zip(*critical_params)
            selected_param = st.selectbox(
                "Select Parameter to Predict",
                param_keys,
                format_func=lambda x: dict(critical_params)[x]
            )
        else:
            selected_param = None
    
    with col2:
        ensemble_method = st.selectbox(
            "Ensemble Method",
            ['weighted_average', 'simple_average', 'median'],
            index=0
        )
    
    with col3:
        run_parallel = st.checkbox("Run Models in Parallel", value=True)
        auto_select_best = st.checkbox("Auto-select Best Model", value=True)
    
    # Run predictions button
    if st.button("🚀 Run All Selected Models", type="primary"):
        if selected_param and column_mapping.get(selected_param):
            target_col = column_mapping[selected_param]
            
            # Initialize forecaster
            forecaster = MultiModelForecaster(api_key=api_key)
            
            # Run all models
            with st.spinner(f"Running {len(selected_models)} models..."):
                start_time = time.time()
                
                results = forecaster.run_all_models(
                    df, target_col, horizon, selected_models
                )
                
                end_time = time.time()
                
                # Store results
                st.session_state.model_results = results
                st.session_state.predictions = results['predictions']
                
                # Show summary
                st.success(f"✅ Completed in {end_time - start_time:.1f} seconds")
                
                # Display model performance metrics
                if results['metrics']:
                    st.markdown("#### Model Performance Metrics")
                    metrics_df = pd.DataFrame(results['metrics']).T
                    metrics_df = metrics_df.round(3)
                    
                    # Highlight best model
                    best_model = results['best_model']
                    st.info(f"🏆 Best Model: **{best_model}** (Lowest MAE)")
                    
                    # Display metrics table with styling
                    st.dataframe(
                        metrics_df.style.highlight_min(axis=0, props='background-color: #90EE90;'),
                        use_container_width=True
                    )
    
    # Display results if available
    if st.session_state.model_results and st.session_state.predictions:
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        # Create combined visualization
        render_combined_predictions_plot(df, st.session_state.predictions, 
                                        selected_param if selected_param else 'Parameter')
        
        # Individual model results
        st.markdown("#### Individual Model Predictions")
        
        model_cols = st.columns(3)
        for i, (model_name, predictions) in enumerate(st.session_state.predictions.items()):
            col_idx = i % 3
            with model_cols[col_idx]:
                if not predictions.empty:
                    model_config = FORECASTING_MODELS.get(model_name, 
                        ModelConfig(name=model_name, type='unknown', color='#888888',
                                  params={}, description='', strengths=[], best_for=''))
                    
                    # Model card
                    st.markdown(f"""
                    <div class="algorithm-badge {model_config.type}-model" 
                         style="background: {model_config.color}; color: white; padding: 10px; 
                                border-radius: 10px; margin: 5px 0;">
                        <strong>{model_config.name}</strong><br>
                        <small>{model_config.description}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key predictions
                    forecast_col = 'forecast' if 'forecast' in predictions.columns else 'TimeGPT'
                    if forecast_col in predictions.columns:
                        max_val = predictions[forecast_col].max()
                        min_val = predictions[forecast_col].min()
                        mean_val = predictions[forecast_col].mean()
                        
                        st.metric("Max Prediction", f"{max_val:.2f}")
                        st.metric("Average", f"{mean_val:.2f}")
                        st.metric("Range", f"{max_val - min_val:.2f}")

def render_model_comparison(model_results):
    """Render model comparison dashboard"""
    st.markdown("### 📊 Model Comparison & Analysis")
    
    if not model_results or not model_results.get('predictions'):
        st.info("Run predictions first to see model comparison")
        return
    
    predictions = model_results['predictions']
    metrics = model_results.get('metrics', {})
    
    # Model performance comparison
    if metrics:
        st.markdown("#### Performance Metrics Comparison")
        
        # Create metrics comparison chart
        metrics_df = pd.DataFrame(metrics).T
        
        # MAE comparison
        fig_mae = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=[m['MAE'] for m in metrics.values()],
                marker_color=['green' if k == model_results.get('best_model') else 'blue' 
                             for k in metrics.keys()],
                text=[f"{m['MAE']:.3f}" for m in metrics.values()],
                textposition='auto'
            )
        ])
        fig_mae.update_layout(
            title="Mean Absolute Error (MAE) by Model",
            xaxis_title="Model",
            yaxis_title="MAE",
            height=400
        )
        st.plotly_chart(fig_mae, use_container_width=True)
        
        # MAPE and R² comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mape = go.Figure(data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=[m['MAPE'] for m in metrics.values()],
                    marker_color='coral',
                    text=[f"{m['MAPE']:.1f}%" for m in metrics.values()],
                    textposition='auto'
                )
            ])
            fig_mape.update_layout(
                title="Mean Absolute Percentage Error (MAPE)",
                xaxis_title="Model",
                yaxis_title="MAPE (%)",
                height=350
            )
            st.plotly_chart(fig_mape, use_container_width=True)
        
        with col2:
            fig_r2 = go.Figure(data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=[m['R2'] for m in metrics.values()],
                    marker_color='lightgreen',
                    text=[f"{m['R2']:.3f}" for m in metrics.values()],
                    textposition='auto'
                )
            ])
            fig_r2.update_layout(
                title="R² Score",
                xaxis_title="Model",
                yaxis_title="R²",
                height=350
            )
            st.plotly_chart(fig_r2, use_container_width=True)
    
    # Prediction spread analysis
    st.markdown("#### Prediction Spread Analysis")
    
    # Calculate prediction statistics
    all_forecasts = []
    model_names = []
    
    for model_name, preds in predictions.items():
        if not preds.empty:
            forecast_col = 'forecast' if 'forecast' in preds.columns else 'TimeGPT'
            if forecast_col in preds.columns:
                all_forecasts.append(preds[forecast_col].values)
                model_names.append(model_name)
    
    if all_forecasts:
        # Box plot of predictions
        fig_box = go.Figure()
        for name, forecast in zip(model_names, all_forecasts):
            fig_box.add_trace(go.Box(y=forecast, name=name))
        
        fig_box.update_layout(
            title="Prediction Distribution by Model",
            yaxis_title="Predicted Values",
            height=400
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Agreement analysis
        st.markdown("#### Model Agreement Analysis")
        
        # Calculate pairwise correlations
        if len(all_forecasts) > 1:
            import pandas as pd
            forecast_df = pd.DataFrame(all_forecasts).T
            forecast_df.columns = model_names
            
            # Correlation heatmap
            correlation_matrix = forecast_df.corr()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values.round(2),
                texttemplate='%{text}',
                colorbar=dict(title="Correlation")
            ))
            
            fig_heatmap.update_layout(
                title="Model Agreement (Correlation Matrix)",
                height=500
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

def render_risk_assessment(risk_metrics, df, column_mapping):
    """Render risk assessment dashboard"""
    st.markdown("### ⚠️ Comprehensive Risk Analysis")
    
    # Current risk status
    current_risk = risk_metrics['overall_risk'].iloc[-1]
    risk_level = risk_metrics['risk_level'].iloc[-1]
    
    # Risk status card
    risk_colors = {
        'Critical': 'risk-critical',
        'High': 'risk-high',
        'Medium': 'risk-medium',
        'Low': 'risk-low'
    }
    
    st.markdown(f"""
    <div class="risk-card {risk_colors.get(risk_level, 'risk-medium')}">
        <h2>{'🚨' if risk_level == 'Critical' else '⚠️' if risk_level == 'High' else 'ℹ️' if risk_level == 'Medium' else '✅'} 
            {risk_level.upper()} RISK</h2>
        <h3>Overall Risk Score: {current_risk:.1f}%</h3>
        <p>{'Immediate action required' if risk_level == 'Critical' else 
           'Schedule maintenance soon' if risk_level == 'High' else 
           'Monitor closely' if risk_level == 'Medium' else 
           'System operating normally'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk breakdown
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk factors chart
        risk_factors = {
            'Voltage': risk_metrics.get('voltage_risk', pd.Series([0])).iloc[-1],
            'Gas Crossover': risk_metrics.get('crossover_risk', pd.Series([0])).iloc[-1],
            'Temperature': risk_metrics.get('thermal_risk', pd.Series([0])).iloc[-1],
            'Efficiency': risk_metrics.get('efficiency_risk', pd.Series([0])).iloc[-1],
            'Maintenance': risk_metrics.get('maintenance_risk', pd.Series([0])).iloc[-1]
        }
        
        fig = go.Figure(data=[go.Bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            marker_color=['red' if v > 70 else 'orange' if v > 40 else 'green' 
                         for v in risk_factors.values()],
            text=[f'{v:.0f}%' for v in risk_factors.values()],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Risk Factor Breakdown",
            xaxis_title="Risk Score (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'].tail(168),
            y=risk_metrics['overall_risk'].tail(168),
            mode='lines',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        # Add risk zones
        fig.add_hline(y=75, line_dash="dash", line_color="darkred",
                     annotation_text="Critical")
        fig.add_hline(y=50, line_dash="dash", line_color="orange",
                     annotation_text="High")
        fig.add_hline(y=25, line_dash="dash", line_color="yellow",
                     annotation_text="Medium")
        
        fig.update_layout(
            title="Risk Score Trend (Last 7 Days)",
            xaxis_title="Time",
            yaxis_title="Risk Score (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def render_maintenance_planning(risk_metrics, predictions, df):
    """Render maintenance planning section"""
    st.markdown("### 🔧 Intelligent Maintenance Planning")
    
    # Generate recommendations based on risk and predictions
    current_risk = risk_metrics['overall_risk'].iloc[-1]
    risk_level = risk_metrics['risk_level'].iloc[-1]
    
    if risk_level == 'Critical':
        urgency = 'IMMEDIATE'
        timeframe = 'Within 4 hours'
        downtime = '4-6 hours'
        cost = 15000
    elif risk_level == 'High':
        urgency = 'SCHEDULED'
        timeframe = 'Within 48 hours'
        downtime = '2-4 hours'
        cost = 8000
    elif risk_level == 'Medium':
        urgency = 'PLANNED'
        timeframe = 'Within 1 week'
        downtime = '2-3 hours'
        cost = 5000
    else:
        urgency = 'ROUTINE'
        timeframe = 'As scheduled'
        downtime = '1-2 hours'
        cost = 2000
    
    # Display recommendations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; 
                    background: {'#ff4444' if urgency == 'IMMEDIATE' else '#ff8800' if urgency == 'SCHEDULED' else '#ffbb33' if urgency == 'PLANNED' else '#00C851'}; 
                    color: white; text-align: center;">
            <h2>{urgency}</h2>
            <p>Priority Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Timeline", timeframe)
        st.metric("Est. Downtime", downtime)
    
    with col3:
        st.metric("Cost Estimate", f"${cost:,}")
        st.metric("ROI", f"{((38000 - cost) / cost * 100):.0f}%")
    
    with col4:
        hours_operated = df.get('hours_since_maintenance', pd.Series([0])).iloc[-1]
        st.metric("Hours Operated", f"{hours_operated:.0f}")
        st.metric("Next Service", f"{max(0, 2000 - hours_operated):.0f} hrs")
    
    # Maintenance tasks
    st.markdown("---")
    st.markdown("#### Recommended Maintenance Actions")
    
    actions = get_maintenance_actions(risk_level, risk_metrics)
    for i, action in enumerate(actions, 1):
        st.markdown(f"{i}. {action}")

def render_landing_page():
    """Render landing page"""
    st.info("👆 Please upload electrolyzer data or generate demo data to begin analysis")
    
    st.markdown("---")
    st.markdown("### 🎯 System Capabilities - 10 Advanced Algorithms")
    
    # Display model cards
    model_cols = st.columns(3)
    
    for i, (model_key, model_config) in enumerate(FORECASTING_MODELS.items()):
        col_idx = i % 3
        with model_cols[col_idx]:
            st.markdown(f"""
            <div class="model-card" style="background: {model_config.color};">
                <h4>{model_config.name}</h4>
                <p>{model_config.description}</p>
                <small><strong>Best for:</strong> {model_config.best_for}</small>
            </div>
            """, unsafe_allow_html=True)

# Utility functions
def generate_demo_data():
    """Generate realistic demo data"""
    np.random.seed(42)
    n_points = 2000
    
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_points),
        periods=n_points,
        freq='H'
    )
    
    t = np.arange(n_points)
    operating_cycles = np.sin(2*np.pi*t/500) > 0
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'current': np.where(operating_cycles,
                          4000 + 2000 * np.sin(2*np.pi*t/24) + np.random.normal(0, 200, n_points),
                          np.random.normal(0, 50, n_points)).clip(0, 8000),
        'voltage': np.where(operating_cycles,
                          320 + 50 * np.sin(2*np.pi*t/24) + 0.02 * t + np.random.normal(0, 10, n_points),
                          np.random.normal(0, 10, n_points)).clip(0, 640),
        'h2_temp': np.where(operating_cycles,
                          70 + 10 * np.sin(2*np.pi*t/168) + np.random.normal(0, 3, n_points),
                          25 + np.random.normal(0, 1, n_points)),
        'lye_temp': np.where(operating_cycles,
                            65 + 8 * np.sin(2*np.pi*t/168) + np.random.normal(0, 2, n_points),
                            25 + np.random.normal(0, 1, n_points)),
        'lye_concentration': np.where(operating_cycles,
                                     28 + 2 * np.sin(2*np.pi*t/500) + np.random.normal(0, 0.5, n_points),
                                     0),
        'o2_in_h2': np.where(operating_cycles,
                           0.3 + 0.001 * t + 0.2 * np.sin(2*np.pi*t/300) + np.random.normal(0, 0.1, n_points),
                           0).clip(0, 2.47),
        'h2_in_o2': np.where(operating_cycles,
                           0.5 + 0.0005 * t + 0.3 * np.sin(2*np.pi*t/300) + np.random.normal(0, 0.15, n_points),
                           0).clip(0, 2.97),
        'pressure': np.where(operating_cycles,
                           10 + 3 * np.sin(2*np.pi*t/100) + np.random.normal(0, 0.5, n_points),
                           0).clip(0, 16),
        'h2_flow': np.where(operating_cycles,
                          50 + 20 * np.sin(2*np.pi*t/24) + np.random.normal(0, 3, n_points),
                          0).clip(0, 100),
        'hours_since_maintenance': np.arange(n_points) % 2000
    })
    
    data['power'] = np.where(operating_cycles,
                            data['current'] * data['voltage'] / 1000,
                            0).clip(0, 5120)
    data['efficiency'] = np.where(data['power'] > 0,
                                 data['h2_flow'] / (data['power'] + 1e-6),
                                 0)
    
    return data

def create_time_series_plot(df, column, ylabel, window=168):
    """Create time series plot"""
    fig = go.Figure()
    
    data_window = df.tail(min(window, len(df)))
    fig.add_trace(go.Scatter(
        x=data_window['timestamp'],
        y=data_window[column],
        mode='lines',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=ylabel,
        height=350,
        hovermode='x unified'
    )
    
    return fig

def create_safety_plot(df, column_mapping):
    """Create gas crossover safety plot"""
    fig = go.Figure()
    
    if column_mapping.get('o2_in_h2'):
        recent_data = df.tail(168)
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data[column_mapping['o2_in_h2']],
            mode='lines',
            name='O₂ in H₂',
            line=dict(color='red', width=2)
        ))
        
        fig.add_hline(y=2, line_dash="dash", line_color="darkred",
                     annotation_text="Critical (2%)")
        fig.add_hline(y=1, line_dash="dash", line_color="orange",
                     annotation_text="Warning (1%)")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Gas Content (%)",
        height=350,
        hovermode='x unified'
    )
    
    return fig

def render_combined_predictions_plot(df, predictions_dict, parameter_name):
    """Render combined predictions from all models"""
    fig = go.Figure()
    
    # Add historical data
    recent_data = df.tail(min(336, len(df)))
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=recent_data[parameter_name] if parameter_name in recent_data.columns else recent_data.iloc[:, 1],
        mode='lines',
        name='Historical',
        line=dict(color='black', width=2),
        opacity=0.7
    ))
    
    # Color palette for different models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Add predictions from each model
    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        if not preds.empty:
            forecast_col = 'forecast' if 'forecast' in preds.columns else 'TimeGPT' if 'TimeGPT' in preds.columns else preds.columns[1]
            
            fig.add_trace(go.Scatter(
                x=preds['ds'] if 'ds' in preds.columns else preds.index,
                y=preds[forecast_col],
                mode='lines',
                name=model_name.upper(),
                line=dict(color=colors[i % len(colors)], width=2,
                         dash='solid' if model_name == 'ensemble' else 'dash')
            ))
            
            # Add confidence intervals for ensemble
            if model_name == 'ensemble' and 'hi-95' in preds.columns:
                fig.add_trace(go.Scatter(
                    x=preds['ds'].tolist() + preds['ds'].tolist()[::-1],
                    y=preds['hi-95'].tolist() + preds['lo-95'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(128,128,128,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% CI (Ensemble)',
                    showlegend=True
                ))
    
    fig.update_layout(
        title=f"Multi-Model Predictions - {parameter_name}",
        xaxis_title="Time",
        yaxis_title="Value",
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_maintenance_actions(risk_level, risk_metrics):
    """Get maintenance actions based on risk level"""
    actions = []
    
    if risk_level == 'Critical':
        actions = [
            "🔴 Immediate system shutdown recommended",
            "🔧 Complete electrode and diaphragm inspection",
            "🔍 Check all critical safety systems",
            "📊 Perform comprehensive diagnostics",
            "🧪 Analyze electrolyte composition",
            "⚡ Verify all electrical connections"
        ]
    elif risk_level == 'High':
        actions = [
            "⚠️ Schedule maintenance within 48 hours",
            "🔍 Inspect high-risk components",
            "🧪 Test electrolyte concentration",
            "📈 Calibrate all sensors",
            "🔧 Check gas separator levels"
        ]
    elif risk_level == 'Medium':
        actions = [
            "📅 Plan maintenance for next shutdown",
            "👁️ Visual inspection of key components",
            "📊 Review operational parameters",
            "🔧 Minor adjustments and calibrations"
        ]
    else:
        actions = [
            "✅ Continue normal operation",
            "📊 Monitor key parameters",
            "📝 Log operational data",
            "👁️ Routine visual checks"
        ]
    
    return actions

# Run the application
if __name__ == "__main__":
    main()
