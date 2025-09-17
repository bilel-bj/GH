"""
Green Hydrogen Electrolyzer Predictive Maintenance App
----------------------------------------------------

This Streamlit application provides a simple yet complete predictive
maintenance dashboard for an alkaline water electrolysis system. It
allows operators to upload their own data or explore the built‑in
ACWA dataset, forecast key process variables using a lightweight
forecasting model and visualise the expected behaviour over a user
defined horizon. In addition, it computes an ad‑hoc risk score
relative to a configurable threshold and identifies when that risk
may exceed acceptable limits. The user interface takes inspiration
from the provided wireframe and includes a sidebar for system
configuration as well as a main area for prediction and risk
visualisation.

Features
--------
* Upload your own Excel/CSV file or fall back to the included ACWA
  dataset.
* Select which process variable to predict and adjust the forecast
  horizon (in hours).
* Configure a risk alert threshold to flag elevated values in the
  forecast.
* Uses an additive Holt–Winters exponential smoothing model from
  ``statsmodels`` to fit historical data and produce forecasts.
* Resamples minute‑level data to hourly averages to align the
  forecast horizon with the slider.
* Calculates a simple risk metric based on the predicted value
  relative to a baseline derived from the historical mean and
  standard deviation.
* Displays the forecast with confidence bands alongside the most
  recent history using Plotly, and summarises key metrics such as
  maximum predicted value, maximum risk and estimated time to risk.
* Shows a gauge indicator to visualise the overall risk compared
  against the user’s alert threshold.

Note
----
This example avoids any external API calls (for example, Nixtla
TimeGPT) and instead relies on local modelling using the
``statsmodels`` library. It is intended as a starting point for
exploration and can be extended with more sophisticated models or
additional tabs (risk assessment, maintenance planning, etc.) as
needed.
"""

import io
import os
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def load_dataset(file: Optional[st.uploaded_file_manager.UploadedFile]) -> pd.DataFrame:
    """Load an uploaded Excel/CSV file into a DataFrame.

    Parameters
    ----------
    file: UploadedFile or None
        File object provided by Streamlit's uploader. If None, the
        builtin ACWA dataset is loaded from disk.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a `timestamp` column (datetime) and
        numeric sensor columns. Unnamed columns are dropped.
    """
    if file is not None:
        name = (file.name or "").lower()
        try:
            if name.endswith((".xlsx", ".xlsm")):
                df = pd.read_excel(file, engine="openpyxl")
            elif name.endswith(".xls"):
                df = pd.read_excel(file, engine="xlrd")
            elif name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                try:
                    df = pd.read_excel(file)
                except Exception:
                    df = pd.read_csv(file)
        except Exception as exc:
            st.error(f"Failed to read the uploaded file: {exc}")
            return pd.DataFrame()
    else:
        default_path = os.path.join(os.path.dirname(__file__), "ACWA Power 2 Processed.xlsx")
        if not os.path.exists(default_path):
            st.error("Default dataset not found. Please upload a file instead.")
            return pd.DataFrame()
        df = pd.read_excel(default_path, engine="openpyxl")

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    time_col = None
    for col in df.columns:
        if col.lower().startswith("timestamp") or col.lower().startswith("time"):
            time_col = col
            break
    if time_col is None:
        st.error("Could not find a timestamp column in the data. Make sure your file contains a 'Timestamp' column.")
        return pd.DataFrame()
    try:
        df['timestamp'] = pd.to_datetime(df[time_col])
    except Exception:
        base_date = datetime(2023, 1, 1)
        df['timestamp'] = pd.to_timedelta(df[time_col].astype(str)) + base_date
    if time_col != 'timestamp':
        df = df.drop(columns=[time_col])

    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in df.columns:
        if col == 'timestamp':
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('timestamp')
    hourly = df.resample('1H').mean()
    hourly = hourly.interpolate(method='time', limit_direction='both', limit=2)
    hourly = hourly.reset_index()
    return hourly


def fit_holt_forecast(series: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    series = series.ffill().bfill()
    if len(series) < 24:
        raise ValueError("Not enough historical data to fit the model. Need at least 24 observations.")
    model = ExponentialSmoothing(series, trend='add', damped_trend=True, seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(horizon)
    residuals = series - fit.fittedvalues
    sigma = residuals.std()
    lower = forecast - 1.96 * sigma
    upper = forecast + 1.96 * sigma
    return forecast, lower, upper


def compute_risk_metrics(prediction: pd.Series, baseline: float, threshold_percent: float) -> Tuple[float, float, Optional[int]]:
    if baseline == 0:
        baseline = 1e-6
    risk_perc = (prediction / baseline) * 100
    max_pred = prediction.max()
    max_risk = risk_perc.max()
    exceed_indices = np.where(risk_perc > threshold_percent)[0]
    first_exceed_idx = int(exceed_indices[0]) if len(exceed_indices) > 0 else None
    return float(max_pred), float(max_risk), first_exceed_idx


def main() -> None:
    st.set_page_config(
        page_title="Green Hydrogen Electrolyzer Predictive Maintenance",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("⚡ Green Hydrogen Electrolyzer Predictive Maintenance")
    st.markdown(
        "This application forecasts key process variables for alkaline water electrolysis and "
        "assesses potential failure risk. Upload your own plant data or explore the built‑in "
        "dataset and adjust the configuration in the sidebar."
    )
    with st.sidebar:
        st.header("System Configuration")
        st.text_input("Nixtla API Key (unused in this demo)", type="password")
        forecast_hours = st.slider(
            "Forecast Horizon (hours)", min_value=6, max_value=72, value=24, step=6
        )
        risk_threshold = st.slider(
            "Risk Alert Threshold (%)", min_value=50, max_value=95, value=75, step=5
        )
        update_freq = st.selectbox(
            "Update Frequency", ["Every 5 minutes", "Every 15 minutes", "Hourly", "Daily"], index=1
        )
        st.markdown("---")
        st.subheader("Data Source")
        uploaded_file = st.file_uploader(
            "Upload Electrolyzer Data (Excel or CSV)",
            type=['xlsx', 'xls', 'csv'],
            accept_multiple_files=False
        )
        use_demo = st.checkbox("Use built‑in ACWA demo data", value=True)

    df = None
    if use_demo and uploaded_file is None:
        df = load_dataset(None)
    elif uploaded_file is not None:
        df = load_dataset(uploaded_file)
    if df is None or df.empty:
        st.info("Please upload a valid dataset or enable the demo dataset from the sidebar.")
        return
    hourly_df = resample_hourly(df)
    numeric_cols = [c for c in hourly_df.columns if c != 'timestamp' and pd.api.types.is_numeric_dtype(hourly_df[c])]
    if not numeric_cols:
        st.error("No numeric columns found for prediction.")
        return
    selected_var = st.selectbox("Select variable to forecast", numeric_cols, index=0)
    with st.expander("Latest observations", expanded=False):
        st.dataframe(hourly_df[['timestamp', selected_var]].tail(24).rename(columns={selected_var: 'value'}))
    if st.button("Generate Predictions", type="primary"):
        series = hourly_df.set_index('timestamp')[selected_var].astype(float)
        horizon = forecast_hours
        try:
            forecast, lower, upper = fit_holt_forecast(series, horizon)
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            return
        last_timestamp = series.index[-1]
        future_dates = [last_timestamp + timedelta(hours=i+1) for i in range(horizon)]
        pred_df = pd.DataFrame({
            'timestamp': future_dates,
            'prediction': forecast.values,
            'lower': lower.values,
            'upper': upper.values
        })
        baseline = series.mean() + series.std()
        max_pred, max_risk, first_exceed_idx = compute_risk_metrics(pred_df['prediction'], baseline, risk_threshold)
        time_to_risk = None
        if first_exceed_idx is not None:
            time_to_risk = pred_df['timestamp'].iloc[first_exceed_idx]
        try:
            back_train = series.iloc[:-12]
            back_test = series.iloc[-12:]
            back_forecast, _, _ = fit_holt_forecast(back_train, 12)
            mape = np.mean(np.abs((back_test.values - back_forecast.values) / np.maximum(1e-6, back_test.values)))
            model_confidence = max(0.0, 100 * (1 - mape))
        except Exception:
            model_confidence = 0.0
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Max Predicted Value", f"{max_pred:.3f}")
        mcol2.metric("Max Failure Risk", f"{max_risk:.1f}%")
        mcol3.metric("Model Confidence", f"{model_confidence:.1f}%")
        if time_to_risk is not None:
            st.success(f"Risk threshold exceeds {risk_threshold}% at {time_to_risk}")
        else:
            st.info("Risk threshold not exceeded in the forecast horizon.")
        st.subheader(f"{selected_var} Forecast")
        fig = go.Figure()
        hist_window = 72
        hist_data = series.tail(hist_window)
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=pred_df['timestamp'],
            y=pred_df['prediction'],
            mode='lines',
            name='Forecast',
            line=dict(color='#d62728')
        ))
        fig.add_trace(go.Scatter(
            x=list(pred_df['timestamp']) + list(pred_df['timestamp'][::-1]),
            y=list(pred_df['upper']) + list(pred_df['lower'][::-1]),
            fill='toself',
            fillcolor='rgba(255, 152, 150, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='95% CI'
        ))
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title=selected_var,
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Overall Risk Gauge")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", value=max_risk,
            title={'text': 'Risk %'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#d62728'},
                'steps': [
                    {'range': [0, 50], 'color': '#2ecc71'},
                    {'range': [50, 75], 'color': '#f1c40f'},
                    {'range': [75, 90], 'color': '#e67e22'},
                    {'range': [90, 100], 'color': '#e74c3c'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 2},
                    'thickness': 0.75,
                    'value': risk_threshold
                }
            }
        ))
        gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge_fig, use_container_width=False)


if __name__ == "__main__":
    main()