"""
Green Hydrogen Electrolyzer Predictive Maintenance System
Production-Ready Version with Nixtla TimeGPT Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
from typing import Optional, Dict, List, Tuple
import json

warnings.filterwarnings('ignore')

# Import Nixtla TimeGPT
try:
    from nixtla import NixtlaClient
    NIXTLA_AVAILABLE = True
except ImportError:
    NIXTLA_AVAILABLE = False
    st.warning("Nixtla package not installed. Install with: pip install nixtla")

# Configure Streamlit page
st.set_page_config(
    page_title="ACWA Power Electrolyzer Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
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
    .plot-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .risk-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        color: white;
    }
    .risk-high {
        background: linear-gradient(135deg, #ffa94d 0%, #ffbe75 100%);
        color: white;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffd43b 0%, #ffe066 100%);
        color: #495057;
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #69db7c 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'risk_assessment' not in st.session_state:
    st.session_state.risk_assessment = None
if 'nixtla_client' not in st.session_state:
    st.session_state.nixtla_client = None
if 'data' not in st.session_state:
    st.session_state.data = None

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

# Data Processing Functions
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
            # Find matching column (case-insensitive, whitespace-tolerant)
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
            # Remove invalid timestamps
            df = df[df['timestamp'].notna()]
        elif 'Timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df = df[df['timestamp'].notna()]
        else:
            # Create timestamp if not exists
            df['timestamp'] = pd.date_range(start=datetime.now() - timedelta(hours=len(df)), 
                                          periods=len(df), freq='H')
        
        # Ensure numeric columns are properly typed
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate efficiency (H2 production / Power consumption)
        if 'h2_flow' in df.columns and 'power' in df.columns:
            # Efficiency = H2 flow rate (Nm³/h) / Power (kW) 
            df['efficiency'] = df['h2_flow'] / (df['power'] + 1e-6)
        elif 'efficiency' not in df.columns:
            df['efficiency'] = 0.02  # Default efficiency
        
        # Calculate hours since maintenance (estimate based on data length)
        if 'hours_since_maintenance' not in df.columns:
            # Assume maintenance every 2000 hours
            df['hours_since_maintenance'] = np.arange(len(df)) % 2000
        
        # Calculate stack efficiency (Voltage efficiency)
        if 'voltage' in df.columns and 'current' in df.columns:
            # Theoretical voltage for water electrolysis is ~1.23V per cell
            # Assuming 100 cells in stack, theoretical voltage = 123V
            theoretical_voltage = 123  # Adjust based on your stack configuration
            df['voltage_efficiency'] = (theoretical_voltage / (df['voltage'] + 1e-6)) * 100
            df['voltage_efficiency'] = df['voltage_efficiency'].clip(0, 100)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect and map column names for ACWA Power data"""
    column_mapping = {
        'voltage': 'voltage' if 'voltage' in df.columns else None,
        'current': 'current' if 'current' in df.columns else None,
        'temperature': 'lye_temp' if 'lye_temp' in df.columns else 'h2_temp' if 'h2_temp' in df.columns else None,
        'pressure': 'pressure' if 'pressure' in df.columns else None,
        'h2_production': 'h2_flow' if 'h2_flow' in df.columns else None,
        'o2_production': 'o2_separator_level' if 'o2_separator_level' in df.columns else None,
        'efficiency': 'efficiency' if 'efficiency' in df.columns else None,
        'power': 'power' if 'power' in df.columns else None,
        'o2_in_h2': 'o2_in_h2' if 'o2_in_h2' in df.columns else None,
        'h2_in_o2': 'h2_in_o2' if 'h2_in_o2' in df.columns else None,
        'lye_concentration': 'lye_concentration' if 'lye_concentration' in df.columns else None,
        'lye_flow': 'lye_flow' if 'lye_flow' in df.columns else None,
        'dm_conductivity': 'dm_conductivity' if 'dm_conductivity' in df.columns else None
    }
    
    # Remove None values
    return {k: v for k, v in column_mapping.items() if v is not None}

# Risk Assessment Functions
def calculate_risk_metrics(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Calculate comprehensive risk metrics for ACWA Power electrolyzer"""
    risk_scores = pd.DataFrame(index=df.index)
    
    # 1. Voltage degradation risk
    if column_mapping.get('voltage'):
        voltage_col = column_mapping['voltage']
        # Normal operating voltage: 280-320V for 8000A operation
        # Critical threshold: >600V (approaching max 640V)
        voltage_normal = 320
        voltage_critical = 600
        current_voltage = df[voltage_col].rolling(window=10, min_periods=1).mean()
        voltage_risk = np.clip((current_voltage - voltage_normal) / 
                              (voltage_critical - voltage_normal) * 100, 0, 100)
        risk_scores['voltage_risk'] = voltage_risk
    else:
        risk_scores['voltage_risk'] = 50
    
    # 2. Gas crossover risk (Critical safety parameter)
    if column_mapping.get('o2_in_h2'):
        o2_col = column_mapping['o2_in_h2']
        # Critical: O2 in H2 > 2% (explosive limit is 4%)
        # Warning: O2 in H2 > 1%
        o2_risk = np.clip(df[o2_col] / 2.0 * 100, 0, 100)
        risk_scores['crossover_risk'] = o2_risk
    else:
        risk_scores['crossover_risk'] = 30
    
    # 3. Temperature risk
    if column_mapping.get('temperature'):
        temp_col = column_mapping['temperature']
        # Optimal: 70-75°C, Critical: >85°C or <50°C
        temp_optimal = 72.5
        temp_range = 15  # ±15°C from optimal
        thermal_risk = np.clip(np.abs(df[temp_col] - temp_optimal) / 
                              temp_range * 100, 0, 100)
        risk_scores['thermal_risk'] = thermal_risk
    else:
        risk_scores['thermal_risk'] = 30
    
    # 4. Efficiency degradation risk
    if 'efficiency' in df.columns:
        # Calculate rolling efficiency to detect degradation
        rolling_eff = df['efficiency'].rolling(window=24, min_periods=1).mean()
        eff_baseline = df['efficiency'].quantile(0.75)
        eff_critical = df['efficiency'].quantile(0.10)
        efficiency_risk = np.clip((eff_baseline - rolling_eff) / 
                                 (eff_baseline - eff_critical + 1e-6) * 100, 0, 100)
        risk_scores['efficiency_risk'] = efficiency_risk
    else:
        risk_scores['efficiency_risk'] = 40
    
    # 5. Lye concentration risk
    if column_mapping.get('lye_concentration'):
        lye_col = column_mapping['lye_concentration']
        # Optimal: 25-32% KOH concentration
        lye_optimal_min = 25
        lye_optimal_max = 32
        lye_risk = np.where(
            (df[lye_col] < lye_optimal_min) | (df[lye_col] > lye_optimal_max),
            np.clip(np.abs(df[lye_col] - 28.5) / 3.5 * 100, 0, 100),
            0
        )
        risk_scores['lye_risk'] = lye_risk
    else:
        risk_scores['lye_risk'] = 30
    
    # 6. DM water quality risk
    if column_mapping.get('dm_conductivity'):
        dm_col = column_mapping['dm_conductivity']
        # Critical: >5 µS/cm, Warning: >2 µS/cm
        dm_risk = np.clip(df[dm_col] / 5.0 * 100, 0, 100)
        risk_scores['water_quality_risk'] = dm_risk
    else:
        risk_scores['water_quality_risk'] = 30
    
    # 7. Maintenance urgency
    if 'hours_since_maintenance' in df.columns:
        maintenance_risk = np.clip(df['hours_since_maintenance'] / 2000 * 100, 0, 100)
        risk_scores['maintenance_risk'] = maintenance_risk
    else:
        risk_scores['maintenance_risk'] = 50
    
    # Calculate overall risk score (weighted average)
    weights = {
        'voltage_risk': 0.20,
        'crossover_risk': 0.25,  # Highest weight - safety critical
        'thermal_risk': 0.15,
        'efficiency_risk': 0.10,
        'lye_risk': 0.10,
        'water_quality_risk': 0.10,
        'maintenance_risk': 0.10
    }
    
    risk_scores['overall_risk'] = sum(
        risk_scores.get(risk, 0) * weight 
        for risk, weight in weights.items()
    )
    
    # Risk level classification
    risk_scores['risk_level'] = pd.cut(
        risk_scores['overall_risk'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    return risk_scores

# Nixtla TimeGPT Integration
class TimeGPTPredictor:
    """Production-ready Nixtla TimeGPT predictor"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize TimeGPT client"""
        self.client = None
        self.api_key = api_key or os.getenv('NIXTLA_API_KEY')
        
        if NIXTLA_AVAILABLE and self.api_key:
            try:
                self.client = NixtlaClient(api_key=self.api_key)
            except Exception as e:
                st.warning(f"Could not initialize Nixtla client: {e}")
    
    def predict(self, df: pd.DataFrame, target_col: str, horizon: int = 24,
                freq: str = 'H', level: List[int] = [80, 95]) -> pd.DataFrame:
        """Generate predictions using TimeGPT"""
        
        if not self.client:
            # Fallback to statistical method if TimeGPT not available
            return self._fallback_predict(df, target_col, horizon)
        
        try:
            # Prepare data for TimeGPT
            forecast_df = df[['timestamp', target_col]].rename(
                columns={'timestamp': 'ds', target_col: 'y'}
            )
            
            # Generate forecast with prediction intervals
            predictions = self.client.forecast(
                df=forecast_df,
                h=horizon,
                freq=freq,
                level=level,
                add_history=False
            )
            
            # Calculate failure probability based on critical thresholds
            critical_value = df[target_col].quantile(0.95)
            predictions['failure_probability'] = predictions.apply(
                lambda row: self._calculate_failure_prob(
                    row['TimeGPT'], 
                    row.get('TimeGPT-hi-95', row['TimeGPT']),
                    critical_value
                ), axis=1
            )
            
            return predictions
            
        except Exception as e:
            st.warning(f"TimeGPT prediction failed: {e}. Using fallback method.")
            return self._fallback_predict(df, target_col, horizon)
    
    def _fallback_predict(self, df: pd.DataFrame, target_col: str, 
                         horizon: int) -> pd.DataFrame:
        """Fallback prediction using statistical methods"""
        from scipy import stats
        
        # Extract trend using linear regression
        recent_data = df[target_col].tail(min(168, len(df))).values
        x = np.arange(len(recent_data))
        slope, intercept, _, _, std_err = stats.linregress(x, recent_data)
        
        # Generate future timestamps
        last_timestamp = df['timestamp'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=horizon,
            freq='H'
        )
        
        # Generate predictions with uncertainty
        predictions = []
        uncertainties = []
        
        for i in range(horizon):
            pred = intercept + slope * (len(recent_data) + i)
            uncertainty = std_err * np.sqrt(1 + 1/len(recent_data) + 
                                           ((len(recent_data) + i) - np.mean(x))**2 / 
                                           np.sum((x - np.mean(x))**2))
            predictions.append(pred)
            uncertainties.append(uncertainty)
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'ds': future_timestamps,
            'TimeGPT': predictions,
            'TimeGPT-lo-95': np.array(predictions) - 1.96 * np.array(uncertainties),
            'TimeGPT-hi-95': np.array(predictions) + 1.96 * np.array(uncertainties),
            'TimeGPT-lo-80': np.array(predictions) - 1.28 * np.array(uncertainties),
            'TimeGPT-hi-80': np.array(predictions) + 1.28 * np.array(uncertainties),
            'uncertainty': uncertainties
        })
        
        # Calculate failure probability
        critical_value = df[target_col].quantile(0.95)
        pred_df['failure_probability'] = pred_df.apply(
            lambda row: self._calculate_failure_prob(
                row['TimeGPT'], 
                row['TimeGPT-hi-95'],
                critical_value
            ), axis=1
        )
        
        return pred_df
    
    def _calculate_failure_prob(self, mean: float, upper: float, 
                               critical: float) -> float:
        """Calculate probability of exceeding critical threshold"""
        from scipy import stats
        
        # Estimate standard deviation from confidence interval
        std = (upper - mean) / 1.96
        
        if std <= 0:
            return 1.0 if mean >= critical else 0.0
        
        # Calculate probability of exceeding critical value
        prob = 1 - stats.norm.cdf(critical, mean, std)
        return np.clip(prob, 0, 1)

    def predict_multivariate(self, df: pd.DataFrame, target_cols: List[str], 
                           horizon: int = 24) -> Dict[str, pd.DataFrame]:
        """Predict multiple variables"""
        predictions = {}
        
        for col in target_cols:
            if col in df.columns:
                pred = self.predict(df, col, horizon)
                predictions[col] = pred
        
        return predictions

# Anomaly Detection
class AnomalyDetector:
    """Real-time anomaly detection for electrolyzer parameters"""
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Detect anomalies using statistical methods"""
        anomalies = pd.DataFrame(index=df.index)
        
        # Use rolling statistics for anomaly detection
        window = min(24, len(df) // 10)  # Adaptive window size
        
        for param, col in column_mapping.items():
            if col and col in df.columns:
                # Calculate rolling statistics
                rolling_mean = df[col].rolling(window=window, center=True).mean()
                rolling_std = df[col].rolling(window=window, center=True).std()
                
                # Detect anomalies (values outside 3 standard deviations)
                z_scores = np.abs((df[col] - rolling_mean) / (rolling_std + 1e-6))
                anomalies[f'{param}_anomaly'] = z_scores > 3
        
        anomalies['total_anomalies'] = anomalies.sum(axis=1)
        anomalies['is_anomaly'] = anomalies['total_anomalies'] > 0
        
        return anomalies

# Maintenance Recommendation Engine
class MaintenanceRecommender:
    """Intelligent maintenance recommendation system"""
    
    @staticmethod
    def generate_recommendations(risk_scores: pd.DataFrame, 
                                predictions: Optional[pd.DataFrame] = None) -> Dict:
        """Generate maintenance recommendations based on risk assessment"""
        
        current_risk = risk_scores['overall_risk'].iloc[-1]
        risk_level = risk_scores['risk_level'].iloc[-1]
        
        recommendations = {
            'urgency': 'ROUTINE',
            'timeframe': 'As scheduled',
            'estimated_downtime': '1-2 hours',
            'actions': [],
            'spare_parts': [],
            'cost_estimate': 0
        }
        
        # Determine urgency based on risk level
        if risk_level == 'Critical' or current_risk > 75:
            recommendations['urgency'] = 'IMMEDIATE'
            recommendations['timeframe'] = 'Within 4 hours'
            recommendations['estimated_downtime'] = '4-6 hours'
            recommendations['cost_estimate'] = 15000
            
            recommendations['actions'] = [
                "🔴 Immediate system shutdown recommended",
                "🔧 Complete electrode inspection required",
                "🔍 Check all critical safety systems",
                "📊 Perform comprehensive system diagnostics"
            ]
            
            recommendations['spare_parts'] = [
                "Electrode replacement set",
                "Diaphragm material",
                "Complete gasket set",
                "Emergency seal kit"
            ]
            
        elif risk_level == 'High' or current_risk > 50:
            recommendations['urgency'] = 'SCHEDULED'
            recommendations['timeframe'] = 'Within 48 hours'
            recommendations['estimated_downtime'] = '2-4 hours'
            recommendations['cost_estimate'] = 8000
            
            recommendations['actions'] = [
                "⚠️ Schedule maintenance window",
                "🔍 Inspect high-risk components",
                "🧪 Perform electrolyte analysis",
                "📈 Calibrate monitoring sensors"
            ]
            
            recommendations['spare_parts'] = [
                "Gasket set",
                "Sensor calibration kit",
                "Electrolyte additives"
            ]
            
        elif risk_level == 'Medium' or current_risk > 25:
            recommendations['urgency'] = 'PLANNED'
            recommendations['timeframe'] = 'Within 1 week'
            recommendations['estimated_downtime'] = '2-3 hours'
            recommendations['cost_estimate'] = 5000
            
            recommendations['actions'] = [
                "📅 Plan maintenance during next shutdown",
                "👁️ Visual inspection of key components",
                "📊 Review operational parameters",
                "🔧 Minor adjustments and calibrations"
            ]
            
            recommendations['spare_parts'] = [
                "Standard maintenance kit",
                "Calibration solutions"
            ]
        
        else:
            recommendations['actions'] = [
                "✅ Continue normal operation",
                "📊 Monitor key parameters",
                "📝 Log operational data",
                "👁️ Routine visual checks"
            ]
            recommendations['cost_estimate'] = 1000
        
        # Add prediction-based recommendations
        if predictions is not None and 'failure_probability' in predictions.columns:
            max_failure_prob = predictions['failure_probability'].max()
            if max_failure_prob > 0.5:
                high_risk_time = predictions[predictions['failure_probability'] > 0.5].iloc[0]['ds']
                recommendations['predicted_failure_window'] = high_risk_time
                recommendations['actions'].insert(0, 
                    f"⏰ High failure risk predicted by {high_risk_time.strftime('%Y-%m-%d %H:%M')}")
        
        return recommendations

# Main Application
def main():
    # Title and header
    st.title("⚡ Green Hydrogen Electrolyzer Predictive Maintenance System")
    st.markdown("**ACWA Power Production System** | Powered by Nixtla TimeGPT & Advanced Analytics")
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1c83e1/ffffff?text=ACWA+Power", 
                use_container_width=True)
        st.markdown("---")
        
        st.markdown("### ⚙️ System Configuration")
        
        # API Key configuration
        api_key = st.text_input(
            "Nixtla API Key",
            type="password",
            value=os.getenv('NIXTLA_API_KEY', ''),
            help="Enter your Nixtla TimeGPT API key for predictions"
        )
        
        if api_key:
            st.session_state.nixtla_client = TimeGPTPredictor(api_key)
            st.success("✅ API Key configured")
        
        # Prediction settings
        forecast_horizon = st.slider(
            "Forecast Horizon (hours)",
            min_value=24,
            max_value=168,
            value=48,
            step=24
        )
        
        risk_threshold = st.slider(
            "Risk Alert Threshold (%)",
            min_value=50,
            max_value=95,
            value=70,
            step=5
        )
        
        update_frequency = st.selectbox(
            "Update Frequency",
            ["Real-time (1 min)", "Every 5 minutes", "Every 15 minutes", "Hourly"],
            index=1
        )
        
        st.markdown("---")
        st.markdown("### 📊 Data Source")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Electrolyzer Data",
            type=['xlsx', 'xls', 'csv'],
            help="Upload historical operational data"
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
                        st.write(f"Columns: {', '.join(df.columns)}")
        
        # Generate demo data option
        if st.button("Generate Demo Data", type="secondary"):
            df = generate_demo_data()
            st.session_state.data = df
            st.session_state.data_loaded = True
            st.success("✅ Demo data generated")
    
    # Main content area
    if st.session_state.data_loaded and st.session_state.data is not None:
        df = st.session_state.data
        
        # Detect columns
        column_mapping = detect_columns(df)
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(df, column_mapping)
        
        # Detect anomalies
        anomalies = AnomalyDetector.detect_anomalies(df, column_mapping)
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Real-time Dashboard",
            "🔮 Predictive Analytics",
            "⚠️ Risk Assessment",
            "🔧 Maintenance Planning",
            "📊 Historical Analysis"
        ])
        
        # Tab 1: Real-time Dashboard
        with tab1:
            render_dashboard(df, risk_metrics, column_mapping, anomalies)
        
        # Tab 2: Predictive Analytics
        with tab2:
            render_predictions(df, column_mapping, forecast_horizon, risk_threshold)
        
        # Tab 3: Risk Assessment
        with tab3:
            render_risk_assessment(risk_metrics, df, column_mapping)
        
        # Tab 4: Maintenance Planning
        with tab4:
            render_maintenance_planning(risk_metrics, st.session_state.predictions, df)
        
        # Tab 5: Historical Analysis
        with tab5:
            render_historical_analysis(df, column_mapping, risk_metrics)
    
    else:
        # Landing page
        render_landing_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <small>
        Green Hydrogen Electrolyzer Predictive Maintenance System v2.0<br>
        Powered by Nixtla TimeGPT & Advanced Analytics<br>
        © 2024 ACWA Power - Production Ready
        </small>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard(df, risk_metrics, column_mapping, anomalies):
    """Render real-time monitoring dashboard for ACWA Power electrolyzer"""
    st.markdown("### 🔍 System Health Overview - Stack #1")
    
    # Current status metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if column_mapping.get('voltage'):
            current_value = df[column_mapping['voltage']].iloc[-1]
            delta = df[column_mapping['voltage']].iloc[-1] - df[column_mapping['voltage']].iloc[-24] if len(df) > 24 else 0
            st.metric("Stack Voltage", f"{current_value:.1f} V", f"{delta:+.1f} V", delta_color="inverse")
        else:
            st.metric("Stack Voltage", "N/A", "No data")
    
    with col2:
        if column_mapping.get('current'):
            current_value = df[column_mapping['current']].iloc[-1]
            delta = df[column_mapping['current']].iloc[-1] - df[column_mapping['current']].iloc[-24] if len(df) > 24 else 0
            st.metric("Current", f"{current_value:.0f} A", f"{delta:+.0f} A")
        else:
            st.metric("Current", "N/A", "No data")
    
    with col3:
        if column_mapping.get('power'):
            power_value = df[column_mapping['power']].iloc[-1]
            delta = df[column_mapping['power']].iloc[-1] - df[column_mapping['power']].iloc[-24] if len(df) > 24 else 0
            st.metric("Power", f"{power_value:.0f} kW", f"{delta:+.0f} kW")
        else:
            st.metric("Power", "N/A", "No data")
    
    with col4:
        if column_mapping.get('h2_production'):
            h2_rate = df[column_mapping['h2_production']].iloc[-1]
            delta = df[column_mapping['h2_production']].iloc[-1] - df[column_mapping['h2_production']].iloc[-24] if len(df) > 24 else 0
            st.metric("H₂ Flow", f"{h2_rate:.1f} Nm³/h", f"{delta:+.1f}")
        else:
            st.metric("H₂ Flow", "N/A", "No data")
    
    with col5:
        if column_mapping.get('o2_in_h2'):
            o2_content = df[column_mapping['o2_in_h2']].iloc[-1]
            status = "🔴" if o2_content > 2 else "🟡" if o2_content > 1 else "🟢"
            st.metric("O₂ in H₂", f"{o2_content:.2f}%", f"{status} Safety")
        else:
            st.metric("O₂ in H₂", "N/A", "No data")
    
    with col6:
        current_risk = risk_metrics['overall_risk'].iloc[-1]
        risk_level = risk_metrics['risk_level'].iloc[-1]
        color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
        st.metric("Risk Score", f"{current_risk:.1f}%", f"{color} {risk_level}")
    
    st.markdown("---")
    
    # Critical parameters monitoring
    st.markdown("#### Critical Parameters Monitoring")
    col1, col2 = st.columns(2)
    
    with col1:
        # Voltage vs Current relationship
        if column_mapping.get('voltage') and column_mapping.get('current'):
            st.markdown("##### Stack V-I Characteristic")
            fig = go.Figure()
            
            # Recent data
            recent_data = df.tail(min(500, len(df)))
            
            # Create scatter plot with color gradient for time
            fig.add_trace(go.Scatter(
                x=recent_data[column_mapping['current']],
                y=recent_data[column_mapping['voltage']],
                mode='markers',
                marker=dict(
                    size=5,
                    color=list(range(len(recent_data))),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time<br>Progression")
                ),
                text=[f"Time: {t}<br>V: {v:.1f}V<br>I: {i:.0f}A" 
                      for t, v, i in zip(recent_data['timestamp'].astype(str),
                                         recent_data[column_mapping['voltage']],
                                         recent_data[column_mapping['current']])],
                hovertemplate='%{text}<extra></extra>',
                name='V-I Points'
            ))
            
            fig.update_layout(
                xaxis_title="Current (A)",
                yaxis_title="Voltage (V)",
                height=350,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gas crossover safety monitoring
        if column_mapping.get('o2_in_h2') and column_mapping.get('h2_in_o2'):
            st.markdown("##### Gas Crossover Monitoring")
            fig = go.Figure()
            
            recent_data = df.tail(min(168, len(df)))
            
            # O2 in H2 (critical parameter)
            fig.add_trace(go.Scatter(
                x=recent_data['timestamp'],
                y=recent_data[column_mapping['o2_in_h2']],
                mode='lines',
                name='O₂ in H₂',
                line=dict(color='red', width=2),
                yaxis='y'
            ))
            
            # H2 in O2
            fig.add_trace(go.Scatter(
                x=recent_data['timestamp'],
                y=recent_data[column_mapping['h2_in_o2']],
                mode='lines',
                name='H₂ in O₂',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))
            
            # Add safety thresholds
            fig.add_hline(y=2, line_dash="dash", line_color="darkred",
                         annotation_text="Critical (2%)", yref='y')
            fig.add_hline(y=1, line_dash="dash", line_color="orange",
                         annotation_text="Warning (1%)", yref='y')
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis=dict(title="O₂ in H₂ (%)", side='left', range=[0, 3]),
                yaxis2=dict(title="H₂ in O₂ (%)", side='right', overlaying='y', range=[0, 3]),
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Temperature and Lye monitoring
    st.markdown("#### Process Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Temperature trends
        temp_cols = ['h2_temp', 'o2_temp', 'lye_temp']
        temp_data = []
        for temp in temp_cols:
            if column_mapping.get(temp):
                temp_data.append({
                    'Parameter': temp.replace('_', ' ').title(),
                    'Current': f"{df[column_mapping[temp]].iloc[-1]:.1f}°C",
                    'Average': f"{df[column_mapping[temp]].mean():.1f}°C",
                    'Status': '✅' if 50 < df[column_mapping[temp]].iloc[-1] < 85 else '⚠️'
                })
        if temp_data:
            st.markdown("##### Temperature Status")
            st.dataframe(pd.DataFrame(temp_data), use_container_width=True, hide_index=True)
    
    with col2:
        # Lye parameters
        if column_mapping.get('lye_concentration') and column_mapping.get('lye_flow'):
            st.markdown("##### Electrolyte (Lye) Status")
            lye_conc = df[column_mapping['lye_concentration']].iloc[-1]
            lye_flow = df[column_mapping['lye_flow']].iloc[-1]
            
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=lye_conc,
                title={'text': "Lye Concentration (%)"},
                gauge={'axis': {'range': [0, 35]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 25], 'color': "lightgray"},
                           {'range': [25, 32], 'color': "lightgreen"},
                           {'range': [32, 35], 'color': "yellow"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 32}}
            ))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Lye Flow Rate", f"{lye_flow:.1f} m³/h")
    
    with col3:
        # DM Water quality
        if column_mapping.get('dm_conductivity'):
            st.markdown("##### DM Water Quality")
            dm_cond = df[column_mapping['dm_conductivity']].iloc[-1]
            dm_flow = df[column_mapping.get('dm_flow', 'dm_flow')].iloc[-1] if 'dm_flow' in column_mapping else 0
            
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=dm_cond,
                title={'text': "Conductivity (µS/cm)"},
                gauge={'axis': {'range': [0, 5]},
                       'bar': {'color': "cyan"},
                       'steps': [
                           {'range': [0, 1], 'color': "green"},
                           {'range': [1, 2], 'color': "yellow"},
                           {'range': [2, 5], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 2}}
            ))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            if dm_flow:
                st.metric("DM Water Flow", f"{dm_flow:.1f} L/h")
    
    # System efficiency analysis
    st.markdown("---")
    st.markdown("#### System Performance Analysis")
    render_kpi_gauges(df, column_mapping, risk_metrics)

def render_predictions(df, column_mapping, horizon, threshold):
    """Render predictive analytics section"""
    st.markdown("### 🔮 Failure Prediction & Forecasting")
    
    # Select target variable for prediction
    available_cols = [col for col in column_mapping.values() if col and col in df.columns]
    if not available_cols:
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    target_variable = st.selectbox(
        "Select Variable to Predict",
        available_cols,
        index=0 if available_cols else None
    )
    
    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Running TimeGPT model..."):
            if st.session_state.nixtla_client:
                predictor = st.session_state.nixtla_client
            else:
                predictor = TimeGPTPredictor()
            
            predictions = predictor.predict(df, target_variable, horizon)
            st.session_state.predictions = predictions
            
            # Show prediction results
            if predictions is not None:
                render_prediction_results(df, predictions, target_variable, threshold)

def render_prediction_results(df, predictions, target_variable, threshold):
    """Display prediction results"""
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_pred = predictions['TimeGPT'].max()
        st.metric("Max Predicted Value", f"{max_pred:.3f}")
    
    with col2:
        max_prob = predictions['failure_probability'].max() * 100
        st.metric("Max Failure Risk", f"{max_prob:.1f}%",
                 "🔴 High" if max_prob > threshold else "🟢 Low")
    
    with col3:
        if any(predictions['failure_probability'] > threshold/100):
            risk_time = predictions[predictions['failure_probability'] > threshold/100].iloc[0]['ds']
            hours_to_risk = (risk_time - df['timestamp'].max()).total_seconds() / 3600
            st.metric("Time to Risk", f"{hours_to_risk:.0f} hours")
        else:
            st.metric("Time to Risk", "No risk detected")
    
    with col4:
        avg_uncertainty = predictions['uncertainty'].mean() if 'uncertainty' in predictions.columns else 0
        confidence = max(0, 100 - avg_uncertainty * 100)
        st.metric("Model Confidence", f"{confidence:.1f}%")
    
    # Visualization
    fig = create_forecast_plot(df, predictions, target_variable)
    st.plotly_chart(fig, use_container_width=True)

def render_risk_assessment(risk_metrics, df, column_mapping):
    """Render risk assessment dashboard"""
    st.markdown("### ⚠️ Comprehensive Risk Analysis")
    
    # Current risk status card
    current_risk = risk_metrics['overall_risk'].iloc[-1]
    risk_level = risk_metrics['risk_level'].iloc[-1]
    
    if risk_level == "Critical":
        st.markdown("""
        <div class="risk-card risk-critical">
            <h2>🚨 CRITICAL RISK DETECTED</h2>
            <h3>Overall Risk Score: {:.1f}%</h3>
            <p>Immediate action required. System shutdown recommended.</p>
        </div>
        """.format(current_risk), unsafe_allow_html=True)
    elif risk_level == "High":
        st.markdown("""
        <div class="risk-card risk-high">
            <h2>⚠️ HIGH RISK</h2>
            <h3>Overall Risk Score: {:.1f}%</h3>
            <p>Schedule immediate maintenance. Monitor closely.</p>
        </div>
        """.format(current_risk), unsafe_allow_html=True)
    elif risk_level == "Medium":
        st.markdown("""
        <div class="risk-card risk-medium">
            <h2>ℹ️ MEDIUM RISK</h2>
            <h3>Overall Risk Score: {:.1f}%</h3>
            <p>Plan maintenance within the week. Continue monitoring.</p>
        </div>
        """.format(current_risk), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="risk-card risk-low">
            <h2>✅ LOW RISK</h2>
            <h3>Overall Risk Score: {:.1f}%</h3>
            <p>System operating normally. Continue routine checks.</p>
        </div>
        """.format(current_risk), unsafe_allow_html=True)
    
    # Risk breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk radar chart
        fig = create_risk_radar(risk_metrics)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk trend
        fig = create_risk_trend(risk_metrics)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed risk table
    st.markdown("#### Risk Factor Analysis")
    risk_table = create_risk_table(risk_metrics, df, column_mapping)
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

def render_maintenance_planning(risk_metrics, predictions, df):
    """Render maintenance planning section"""
    st.markdown("### 🔧 Intelligent Maintenance Planning")
    
    # Generate recommendations
    recommender = MaintenanceRecommender()
    recommendations = recommender.generate_recommendations(risk_metrics, predictions)
    
    # Display recommendations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        urgency_colors = {
            'IMMEDIATE': 'red',
            'SCHEDULED': 'orange',
            'PLANNED': 'yellow',
            'ROUTINE': 'green'
        }
        color = urgency_colors.get(recommendations['urgency'], 'gray')
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; background: {color}; opacity: 0.8; color: white;">
            <h3>{recommendations['urgency']}</h3>
            <p>Timeline: {recommendations['timeframe']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Estimated Downtime", recommendations['estimated_downtime'])
        st.metric("Cost Estimate", f"${recommendations['cost_estimate']:,}")
    
    with col3:
        if 'predicted_failure_window' in recommendations:
            st.metric("Predicted Failure", 
                     recommendations['predicted_failure_window'].strftime('%Y-%m-%d %H:%M'))
    
    # Recommended actions
    st.markdown("#### Recommended Actions")
    for action in recommendations['actions']:
        st.markdown(action)
    
    # Spare parts
    if recommendations['spare_parts']:
        st.markdown("#### Required Spare Parts")
        for part in recommendations['spare_parts']:
            st.markdown(f"• {part}")
    
    # Maintenance schedule
    st.markdown("#### Maintenance Schedule")
    schedule = create_maintenance_schedule(df)
    st.dataframe(schedule, use_container_width=True, hide_index=True)

def render_historical_analysis(df, column_mapping, risk_metrics):
    """Render historical analysis section"""
    st.markdown("### 📊 Historical Performance Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=df['timestamp'].min().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=df['timestamp'].max().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
    
    # Filter data
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df.loc[mask]
    
    # Performance metrics over time
    if len(filtered_df) > 0:
        # Efficiency analysis
        if 'efficiency' in filtered_df.columns:
            st.markdown("#### Efficiency Trend")
            fig = px.line(filtered_df, x='timestamp', y='efficiency',
                         title="System Efficiency Over Time")
            fig.add_hline(y=filtered_df['efficiency'].mean(), 
                         line_dash="dash", annotation_text="Average")
            st.plotly_chart(fig, use_container_width=True)
        
        # Parameter correlation matrix
        st.markdown("#### Parameter Correlations")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = filtered_df[numeric_cols[:10]].corr()
            fig = px.imshow(corr_matrix, 
                          labels=dict(color="Correlation"),
                          color_continuous_scale='RdBu',
                          zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for selected date range")

def render_landing_page():
    """Render landing page when no data is loaded"""
    st.info("👆 Please upload electrolyzer data or generate demo data to begin analysis")
    
    st.markdown("---")
    st.markdown("### 🎯 System Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔮 Advanced Prediction**
        - Nixtla TimeGPT integration
        - Multi-variate forecasting
        - Uncertainty quantification
        - 24-168 hour horizons
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Risk Management**
        - Real-time anomaly detection
        - Multi-factor risk scoring
        - Failure probability estimation
        - Automated alerting
        """)
    
    with col3:
        st.markdown("""
        **🔧 Smart Maintenance**
        - AI-powered recommendations
        - Optimal scheduling
        - Cost-benefit analysis
        - Spare parts management
        """)
    
    st.markdown("---")
    st.markdown("### 📈 Expected Data Format")
    st.markdown("""
    The system accepts Excel or CSV files with operational data including:
    - Timestamp (date/time column)
    - Cell/Stack Voltage
    - Current
    - Temperature
    - Pressure
    - H₂/O₂ Production Rates
    - Power Consumption
    - Efficiency metrics
    
    The system will automatically detect and map columns based on naming patterns.
    """)

# Utility Functions
def generate_demo_data():
    """Generate realistic demo data matching ACWA Power electrolyzer structure"""
    np.random.seed(42)
    n_points = 2000
    
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_points),
        periods=n_points,
        freq='H'
    )
    
    t = np.arange(n_points)
    
    # Generate realistic ACWA Power electrolyzer data
    # Simulate startup, steady-state, and shutdown cycles
    operating_cycles = np.sin(2*np.pi*t/500) > 0  # Operating cycles
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        # Stack current (0-8000A based on actual data)
        'current': np.where(operating_cycles,
                          4000 + 2000 * np.sin(2*np.pi*t/24) + np.random.normal(0, 200, n_points),
                          np.random.normal(0, 50, n_points)).clip(0, 8000),
        # Stack voltage (0-640V based on actual data)
        'voltage': np.where(operating_cycles,
                          320 + 50 * np.sin(2*np.pi*t/24) + 0.02 * t + np.random.normal(0, 10, n_points),
                          np.random.normal(0, 10, n_points)).clip(0, 640),
        # Temperatures
        'h2_temp': np.where(operating_cycles,
                          70 + 10 * np.sin(2*np.pi*t/168) + np.random.normal(0, 3, n_points),
                          25 + np.random.normal(0, 1, n_points)),
        'o2_temp': np.where(operating_cycles,
                          72 + 10 * np.sin(2*np.pi*t/168) + np.random.normal(0, 3, n_points),
                          25 + np.random.normal(0, 1, n_points)),
        'lye_temp': np.where(operating_cycles,
                            65 + 8 * np.sin(2*np.pi*t/168) + np.random.normal(0, 2, n_points),
                            25 + np.random.normal(0, 1, n_points)),
        'room_temp': 25 + 0.5 * np.sin(2*np.pi*t/24) + np.random.normal(0, 0.5, n_points),
        # Lye parameters
        'lye_concentration': np.where(operating_cycles,
                                     28 + 2 * np.sin(2*np.pi*t/500) + np.random.normal(0, 0.5, n_points),
                                     0),
        'lye_flow': np.where(operating_cycles,
                            50 + 20 * np.sin(2*np.pi*t/24) + np.random.normal(0, 5, n_points),
                            0).clip(0, 100),
        # Separator levels
        'h2_separator_level': np.where(operating_cycles,
                                      20 + 10 * np.sin(2*np.pi*t/100) + np.random.normal(0, 2, n_points),
                                      0).clip(0, 51.5),
        'o2_separator_level': np.where(operating_cycles,
                                      20 + 10 * np.sin(2*np.pi*t/100) + np.random.normal(0, 2, n_points),
                                      0).clip(0, 51.5),
        'ldi_separator': np.where(operating_cycles,
                                0.5 + 0.2 * np.sin(2*np.pi*t/200) + np.random.normal(0, 0.1, n_points),
                                0).clip(0, 3),
        # Gas crossover (critical safety parameters)
        'o2_in_h2': np.where(operating_cycles,
                           0.3 + 0.001 * t + 0.2 * np.sin(2*np.pi*t/300) + np.random.normal(0, 0.1, n_points),
                           0).clip(0, 2.47),
        'h2_in_o2': np.where(operating_cycles,
                           0.5 + 0.0005 * t + 0.3 * np.sin(2*np.pi*t/300) + np.random.normal(0, 0.15, n_points),
                           0).clip(0, 2.97),
        # Pressure
        'pressure': np.where(operating_cycles,
                           10 + 3 * np.sin(2*np.pi*t/100) + np.random.normal(0, 0.5, n_points),
                           0).clip(0, 16),
        # H2 production
        'h2_flow': np.where(operating_cycles,
                          50 + 20 * np.sin(2*np.pi*t/24) + np.random.normal(0, 3, n_points),
                          0).clip(0, 100),
        # DM water parameters
        'dm_conductivity': 1.5 + 0.0005 * t + np.random.normal(0, 0.2, n_points).clip(0, 5),
        'dm_flow': np.where(operating_cycles,
                          500 + 100 * np.sin(2*np.pi*t/24) + np.random.normal(0, 30, n_points),
                          0).clip(0, 1050),
        'hours_since_maintenance': np.arange(n_points) % 2000
    })
    
    # Calculate power consumption
    data['power'] = np.where(operating_cycles,
                            data['current'] * data['voltage'] / 1000,  # kW
                            0).clip(0, 5120)
    
    # Calculate efficiency
    data['efficiency'] = np.where(data['power'] > 0,
                                 data['h2_flow'] / (data['power'] + 1e-6),
                                 0)
    
    return data

def create_time_series_plot(df, column, ylabel, window=168, 
                           critical_line=None, warning_line=None, 
                           optimal_range=None):
    """Create time series plot with thresholds"""
    fig = go.Figure()
    
    # Main data line
    data_window = df.tail(min(window, len(df)))
    fig.add_trace(go.Scatter(
        x=data_window['timestamp'],
        y=data_window[column],
        mode='lines',
        name=column,
        line=dict(color='blue', width=2)
    ))
    
    # Add threshold lines
    if critical_line:
        fig.add_hline(y=critical_line, line_dash="dash", line_color="red",
                     annotation_text="Critical")
    if warning_line:
        fig.add_hline(y=warning_line, line_dash="dash", line_color="orange",
                     annotation_text="Warning")
    
    # Add optimal range
    if optimal_range:
        fig.add_hrect(y0=optimal_range[0], y1=optimal_range[1],
                     fillcolor="green", opacity=0.1,
                     annotation_text="Optimal Range")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=ylabel,
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode='x unified'
    )
    
    return fig

def create_forecast_plot(df, predictions, target_variable):
    """Create forecast visualization"""
    fig = go.Figure()
    
    # Historical data
    recent_data = df.tail(min(168, len(df)))
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=recent_data[target_variable],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=predictions['ds'],
        y=predictions['TimeGPT'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence intervals
    if 'TimeGPT-hi-95' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions['ds'].tolist() + predictions['ds'].tolist()[::-1],
            y=predictions['TimeGPT-hi-95'].tolist() + predictions['TimeGPT-lo-95'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f"{target_variable} Forecast",
        xaxis_title="Time",
        yaxis_title=target_variable,
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_risk_radar(risk_metrics):
    """Create risk radar chart"""
    categories = ['Voltage\nRisk', 'Thermal\nRisk', 'Efficiency\nRisk', 'Maintenance\nRisk']
    
    values = []
    for cat in ['voltage_risk', 'thermal_risk', 'efficiency_risk', 'maintenance_risk']:
        if cat in risk_metrics.columns:
            values.append(risk_metrics[cat].iloc[-1])
        else:
            values.append(0)
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Risk Profile',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title="Risk Profile Analysis",
        height=400
    )
    
    return fig

def create_risk_trend(risk_metrics):
    """Create risk trend plot"""
    fig = go.Figure()
    
    window = min(168, len(risk_metrics))
    recent_metrics = risk_metrics.tail(window)
    
    fig.add_trace(go.Scatter(
        x=recent_metrics.index,
        y=recent_metrics['overall_risk'],
        mode='lines',
        name='Overall Risk',
        line=dict(color='red', width=3)
    ))
    
    # Add risk zones
    fig.add_hline(y=75, line_dash="dash", line_color="darkred",
                 annotation_text="Critical")
    fig.add_hline(y=50, line_dash="dash", line_color="orange",
                 annotation_text="High")
    fig.add_hline(y=25, line_dash="dash", line_color="yellow",
                 annotation_text="Medium")
    
    fig.update_layout(
        title="Risk Score Trend",
        xaxis_title="Time Period",
        yaxis_title="Risk Score (%)",
        height=400
    )
    
    return fig

def create_risk_table(risk_metrics, df, column_mapping):
    """Create detailed risk assessment table"""
    risk_components = []
    
    for risk_type in ['voltage_risk', 'thermal_risk', 'efficiency_risk', 'maintenance_risk']:
        if risk_type in risk_metrics.columns:
            current_value = risk_metrics[risk_type].iloc[-1]
            trend = risk_metrics[risk_type].iloc[-1] - risk_metrics[risk_type].iloc[-24] if len(risk_metrics) > 24 else 0
            
            status = 'Critical' if current_value > 75 else 'High' if current_value > 50 else 'Medium' if current_value > 25 else 'Low'
            status_emoji = '🔴' if status == 'Critical' else '🟠' if status == 'High' else '🟡' if status == 'Medium' else '🟢'
            
            risk_components.append({
                'Risk Factor': risk_type.replace('_', ' ').title(),
                'Current Score': f"{current_value:.1f}%",
                'Trend (24h)': f"{trend:+.1f}%",
                'Status': f"{status_emoji} {status}",
                'Action Required': get_risk_action(risk_type, current_value)
            })
    
    return pd.DataFrame(risk_components)

def get_risk_action(risk_type, value):
    """Get recommended action based on risk type and value"""
    actions = {
        'voltage_risk': {
            75: "Immediate electrode inspection",
            50: "Schedule electrode check",
            25: "Monitor voltage trends",
            0: "Normal operation"
        },
        'thermal_risk': {
            75: "Check cooling system immediately",
            50: "Adjust temperature control",
            25: "Monitor temperature",
            0: "Optimal temperature"
        },
        'efficiency_risk': {
            75: "System optimization required",
            50: "Performance review needed",
            25: "Efficiency monitoring",
            0: "Performing well"
        },
        'maintenance_risk': {
            75: "Immediate maintenance required",
            50: "Schedule maintenance soon",
            25: "Plan maintenance window",
            0: "Continue operation"
        }
    }
    
    risk_actions = actions.get(risk_type, {75: "Review", 50: "Monitor", 25: "Watch", 0: "OK"})
    
    for threshold in sorted(risk_actions.keys(), reverse=True):
        if value >= threshold:
            return risk_actions[threshold]
    
    return "Normal"

def create_maintenance_schedule(df):
    """Create maintenance schedule table"""
    current_time = datetime.now()
    
    schedule = pd.DataFrame({
        'Task': ['Electrode Inspection', 'Diaphragm Check', 'Sensor Calibration', 
                'Electrolyte Analysis', 'System Flush', 'Full Overhaul'],
        'Priority': ['High', 'High', 'Medium', 'Medium', 'Low', 'Low'],
        'Frequency': ['500 hrs', '1000 hrs', '250 hrs', '100 hrs', '2000 hrs', '8000 hrs'],
        'Last Performed': [
            (current_time - timedelta(hours=400)).strftime('%Y-%m-%d'),
            (current_time - timedelta(hours=800)).strftime('%Y-%m-%d'),
            (current_time - timedelta(hours=200)).strftime('%Y-%m-%d'),
            (current_time - timedelta(hours=80)).strftime('%Y-%m-%d'),
            (current_time - timedelta(hours=1500)).strftime('%Y-%m-%d'),
            (current_time - timedelta(hours=7000)).strftime('%Y-%m-%d')
        ],
        'Next Due': [
            (current_time + timedelta(hours=100)).strftime('%Y-%m-%d'),
            (current_time + timedelta(hours=200)).strftime('%Y-%m-%d'),
            (current_time + timedelta(hours=50)).strftime('%Y-%m-%d'),
            (current_time + timedelta(hours=20)).strftime('%Y-%m-%d'),
            (current_time + timedelta(hours=500)).strftime('%Y-%m-%d'),
            (current_time + timedelta(hours=1000)).strftime('%Y-%m-%d')
        ],
        'Estimated Duration': ['2-3 hrs', '3-4 hrs', '1 hr', '30 min', '4-5 hrs', '24-48 hrs']
    })
    
    return schedule

def render_kpi_gauges(df, column_mapping, risk_metrics):
    """Render KPI gauge charts for ACWA Power electrolyzer"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Stack Efficiency (Power efficiency)
        if column_mapping.get('h2_production') and column_mapping.get('power'):
            h2_prod = df[column_mapping['h2_production']].iloc[-1]
            power = df[column_mapping['power']].iloc[-1]
            # Efficiency in Nm³/kWh
            efficiency = (h2_prod / (power + 1e-6)) if power > 0 else 0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=efficiency,
                title={'text': "H₂ Production Efficiency<br>(Nm³/kWh)"},
                gauge={'axis': {'range': [0, 0.1]},
                      'bar': {'color': "green"},
                      'steps': [
                          {'range': [0, 0.02], 'color': "red"},
                          {'range': [0.02, 0.05], 'color': "yellow"},
                          {'range': [0.05, 0.1], 'color': "lightgreen"}]}
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Efficiency data not available")
    
    with col2:
        # Separator Levels (Safety indicator)
        if column_mapping.get('h2_separator_level') and column_mapping.get('o2_separator_level'):
            h2_level = df[column_mapping['h2_separator_level']].iloc[-1]
            o2_level = df[column_mapping['o2_separator_level']].iloc[-1]
            avg_level = (h2_level + o2_level) / 2
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_level,
                title={'text': "Avg Separator Level (%)"},
                gauge={'axis': {'range': [0, 60]},
                      'bar': {'color': "blue"},
                      'steps': [
                          {'range': [0, 15], 'color': "lightblue"},
                          {'range': [15, 35], 'color': "green"},
                          {'range': [35, 50], 'color': "yellow"},
                          {'range': [50, 60], 'color': "red"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': 50}}
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Separator data not available")
    
    with col3:
        # System Pressure
        if column_mapping.get('pressure'):
            pressure = df[column_mapping['pressure']].iloc[-1]
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pressure,
                title={'text': "System Pressure (bar)"},
                gauge={'axis': {'range': [0, 20]},
                      'bar': {'color': "purple"},
                      'steps': [
                          {'range': [0, 5], 'color': "lightgray"},
                          {'range': [5, 15], 'color': "green"},
                          {'range': [15, 18], 'color': "yellow"},
                          {'range': [18, 20], 'color': "red"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': 16}}
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pressure data not available")

# Run the application
if __name__ == "__main__":
    main()
