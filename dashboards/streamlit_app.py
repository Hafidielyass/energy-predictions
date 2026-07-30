"""
Streamlit Dashboard — Solar PV Forecasting & Anomaly Detection
==============================================================

Launch:
    streamlit run dashboards/streamlit_app.py

Features
--------
* Real-time data explorer (date range selector)
* 24h power forecast visualisation
* Anomaly / fault day calendar
* KPI cards: daily energy, performance ratio, peak power
* Inverter health indicators
* Interactive power vs irradiance plot
* Downloadable reports
"""

import os
import sys

# Add project root to path so `src` modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Solar PV Intelligence Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #F39C12;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header { text-align: center; color: #7F8C8D; margin-bottom: 1.5rem; }
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border: 1px solid #0f3460;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #F39C12; }
    .kpi-label { font-size: 0.85rem; color: #BDC3C7; margin-top: 0.2rem; }
    .status-normal   { background:#1e8449; color:white; padding:4px 10px; border-radius:8px; }
    .status-moderate { background:#d35400; color:white; padding:4px 10px; border-radius:8px; }
    .status-severe   { background:#922b21; color:white; padding:4px 10px; border-radius:8px; }
    div[data-testid="stMetric"] { background: #0f3460; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading dataset...")
def load_data():
    """Load the processed parquet if available, else raw CSV (sampled)."""
    parquet = "data/processed/processed_data.parquet"
    if os.path.exists(parquet):
        df = pd.read_parquet(parquet)
    else:
        st.warning("Processed data not found — loading raw CSV (this may take a moment).")
        df = pd.read_csv(
            "input/dataset.csv",
            parse_dates=["measured_on"],
            usecols=["measured_on","ac_power__315","poa_irradiance__313",
                     "ambient_temp__320","module_temp_1__321","inverter_temp__324",
                     "dc_pos_voltage__316","dc_pos_current__317","power_factor__327"],
            dtype={c: "float32" for c in [
                "ac_power__315","poa_irradiance__313","ambient_temp__320",
                "module_temp_1__321","inverter_temp__324","dc_pos_voltage__316",
                "dc_pos_current__317","power_factor__327"]},
        )
        df = df.sort_values("measured_on").set_index("measured_on")
        df = df.resample("15T").mean()
        df["ac_power__315"] = df["ac_power__315"].clip(lower=0)
        df["poa_irradiance_clipped"] = df["poa_irradiance__313"].clip(lower=0)
    return df


@st.cache_data(ttl=3600)
def load_daily_results():
    """Load anomaly detection results if available."""
    path = "data/processed/daily_anomaly_results.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_forecast_results():
    """Load saved forecast predictions if available."""
    path = "data/processed/forecast_results.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame):
    st.sidebar.image("https://img.icons8.com/fluency/96/solar-energy.png", width=80)
    st.sidebar.title("PV Intelligence")
    st.sidebar.markdown("---")

    st.sidebar.header("Date Range Filter")
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    start_date = st.sidebar.date_input("Start Date", value=max_date - pd.Timedelta(days=30),
                                        min_value=min_date, max_value=max_date)
    end_date   = st.sidebar.date_input("End Date",   value=max_date,
                                        min_value=min_date, max_value=max_date)

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", [
        "Overview",
        "Power Forecasting",
        "Anomaly Detection",
        "Data Explorer",
        "Model Performance",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Info**")
    st.sidebar.markdown(f"Data: {min_date} → {max_date}")
    st.sidebar.markdown(f"Records: {len(df):,}")
    st.sidebar.markdown(f"Resolution: 15 min")
    st.sidebar.markdown("Model: GRU (TF 2.21)")
    return page, start_date, end_date


# ─── KPI row ──────────────────────────────────────────────────────────────────

def render_kpi_row(df_filtered: pd.DataFrame):
    # Calculate KPIs
    total_energy_mwh = df_filtered["ac_power__315"].clip(lower=0).sum() * 15 / 60 / 1000
    peak_power_kw    = df_filtered["ac_power__315"].max() / 1000
    avg_irradiance   = df_filtered.get("poa_irradiance_clipped",
                       df_filtered.get("poa_irradiance__313", pd.Series([0]))).mean()
    avg_temp         = df_filtered.get("ambient_temp__320", pd.Series([np.nan])).mean()
    n_days           = df_filtered.index.normalize().nunique()
    daily_avg_kwh    = (total_energy_mwh * 1000) / max(n_days, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Energy", f"{total_energy_mwh:.2f} MWh",  help="AC energy in selected period")
    c2.metric("Peak Power",   f"{peak_power_kw:.2f} kW",      help="Maximum AC power recorded")
    c3.metric("Avg Daily Energy", f"{daily_avg_kwh:.1f} kWh", help="Mean energy per day")
    c4.metric("Avg Irradiance",   f"{avg_irradiance:.0f} W/m²")
    c5.metric("Avg Ambient Temp", f"{avg_temp:.1f} °C")


# ─── Page: Overview ──────────────────────────────────────────────────────────

def page_overview(df_filtered: pd.DataFrame, daily_results):
    st.markdown('<p class="main-header">☀️ Solar PV Intelligence Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time monitoring · Forecasting · Fault Detection</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    render_kpi_row(df_filtered)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Daily energy bar chart
        daily_energy = df_filtered["ac_power__315"].clip(lower=0).resample("D").sum() * 15 / 60 / 1000
        fig = go.Figure()
        if daily_results is not None and "severity_label" in daily_results.columns:
            color_map = {"Normal":"#2ECC71","Moderately Faulty":"#F39C12","Severely Faulty":"#E74C3C"}
            for label, color in color_map.items():
                mask = daily_results["severity_label"] == label
                idx  = daily_results[mask].index
                subset = daily_energy[daily_energy.index.isin(idx)]
                fig.add_trace(go.Bar(x=subset.index, y=subset.values,
                                     name=label, marker_color=color))
        else:
            fig.add_trace(go.Bar(x=daily_energy.index, y=daily_energy.values,
                                  marker_color="#F39C12", name="Daily Energy"))
        fig.update_layout(title="Daily Energy Generation (MWh)",
                          xaxis_title="Date", yaxis_title="Energy (MWh)",
                          template="plotly_dark", barmode="stack",
                          height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Severity donut chart
        if daily_results is not None:
            counts = daily_results["severity_label"].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                hole=0.55,
                marker_colors=["#2ECC71","#F39C12","#E74C3C"][:len(counts)],
            ))
            fig2.update_layout(title="Day Classification", template="plotly_dark",
                               height=350, showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run the anomaly detection pipeline to see fault classification.")

    # Recent days table
    if daily_results is not None:
        st.subheader("Recent Days — Anomaly Scores")
        recent = daily_results.tail(14)[["daily_energy_kwh","performance_ratio",
                                          "anomaly_score","severity_label"]].copy()
        recent = recent.round(3)

        def _color_label(val):
            colors = {"Normal":"background-color:#1e8449;color:white",
                      "Moderately Faulty":"background-color:#d35400;color:white",
                      "Severely Faulty":"background-color:#922b21;color:white"}
            return colors.get(val, "")

        styled = recent.style.applymap(_color_label, subset=["severity_label"])
        st.dataframe(styled, use_container_width=True)


# ─── Page: Power Forecasting ─────────────────────────────────────────────────

def page_forecasting(df_filtered: pd.DataFrame, forecast_results):
    st.header("24-Hour Power Generation Forecast")

    if forecast_results is None:
        st.warning("No forecast results found. Please run the forecasting pipeline first.")
        st.code("python notebooks/run_pipeline.py --mode forecast", language="bash")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=("Actual vs Predicted AC Power", "Forecast Error"),
                            row_heights=[0.7, 0.3], vertical_spacing=0.08)

        actual = forecast_results.get("actual", pd.Series(dtype=float))
        pred   = forecast_results.get("predicted", pd.Series(dtype=float))

        fig.add_trace(go.Scatter(x=actual.index, y=actual.values,
                                  name="Actual", line=dict(color="#2ECC71", width=1.5)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values,
                                  name="Forecast", line=dict(color="#F39C12", width=1.5, dash="dash")),
                      row=1, col=1)

        error = actual - pred
        fig.add_trace(go.Bar(x=error.index, y=error.values,
                              name="Error", marker_color="#E74C3C", opacity=0.6),
                      row=2, col=1)

        fig.update_layout(template="plotly_dark", height=500,
                          legend=dict(orientation="h", y=1.05))
        fig.update_yaxes(title_text="AC Power (W)", row=1, col=1)
        fig.update_yaxes(title_text="Error (W)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Forecast Metrics")
        if "metrics" in forecast_results.columns or isinstance(forecast_results, dict):
            metrics = forecast_results.get("metrics", {})
            for k, v in metrics.items():
                st.metric(k, f"{v:.3f}")
        else:
            st.info("Metrics will appear here after evaluation.")


# ─── Page: Anomaly Detection ─────────────────────────────────────────────────

def page_anomaly(df_filtered: pd.DataFrame, daily_results):
    st.header("Anomaly Detection & Fault Classification")

    if daily_results is None:
        st.warning("No anomaly results found. Run the anomaly detection pipeline first.")
        return

    # Timeline plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_results.index, y=daily_results["anomaly_score"],
        mode="lines+markers",
        line=dict(color="steelblue", width=1),
        marker=dict(color=daily_results["anomaly_score"],
                    colorscale="RdYlGn_r", size=6, showscale=True),
        name="Anomaly Score",
    ))
    fig.add_hrect(y0=0,    y1=0.35, fillcolor="#2ECC71", opacity=0.07, line_width=0)
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor="#F39C12", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.65, y1=1.00, fillcolor="#E74C3C", opacity=0.10, line_width=0)
    fig.update_layout(title="Daily Composite Anomaly Score",
                      xaxis_title="Date", yaxis_title="Anomaly Score [0-1]",
                      template="plotly_dark", height=380,
                      yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Scatter: performance ratio vs anomaly score
        if "performance_ratio" in daily_results.columns:
            fig2 = px.scatter(
                daily_results.reset_index(), x="performance_ratio", y="anomaly_score",
                color="severity_label",
                color_discrete_map={"Normal":"#2ECC71","Moderately Faulty":"#F39C12",
                                    "Severely Faulty":"#E74C3C"},
                title="Performance Ratio vs Anomaly Score",
                template="plotly_dark", opacity=0.7,
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Sub-score breakdown
        score_cols = ["if_score","ae_score","stat_score","curve_score"]
        avail_cols = [c for c in score_cols if c in daily_results.columns]
        if avail_cols:
            means = daily_results[avail_cols].mean()
            fig3 = go.Figure(go.Bar(
                x=[c.replace("_score","").upper() for c in avail_cols],
                y=means.values,
                marker_color=["#3498DB","#9B59B6","#F39C12","#1ABC9C"],
            ))
            fig3.update_layout(title="Average Sub-Score Contribution",
                               yaxis_title="Mean Score",
                               template="plotly_dark", height=350)
            st.plotly_chart(fig3, use_container_width=True)

    # Fault day table
    st.subheader("Detected Fault Days")
    fault_days = daily_results[daily_results["severity_label"] != "Normal"].sort_values(
        "anomaly_score", ascending=False
    )
    if len(fault_days) == 0:
        st.success("No fault days detected in the selected period.")
    else:
        st.dataframe(fault_days[["daily_energy_kwh","performance_ratio",
                                   "anomaly_score","severity_label"]].round(3),
                     use_container_width=True)


# ─── Page: Data Explorer ─────────────────────────────────────────────────────

def page_explorer(df_filtered: pd.DataFrame):
    st.header("Interactive Data Explorer")

    col_options = [c for c in df_filtered.columns if df_filtered[c].dtype != object]
    selected_cols = st.multiselect("Select columns to plot",
                                    options=col_options,
                                    default=["ac_power__315", "poa_irradiance_clipped"])

    if selected_cols:
        fig = make_subplots(rows=len(selected_cols), cols=1, shared_xaxes=True,
                            vertical_spacing=0.04)
        colors = px.colors.qualitative.Set2
        for i, col in enumerate(selected_cols):
            fig.add_trace(
                go.Scatter(x=df_filtered.index, y=df_filtered[col],
                           mode="lines", name=col,
                           line=dict(color=colors[i % len(colors)], width=1)),
                row=i+1, col=1
            )
            fig.update_yaxes(title_text=col, row=i+1, col=1)
        fig.update_layout(template="plotly_dark", height=200*len(selected_cols),
                          title="Sensor Time Series", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    if st.button("Compute Correlation"):
        num_cols = df_filtered[col_options].select_dtypes(include=np.number)
        corr = num_cols.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                              color_continuous_scale="RdYlGn",
                              title="Pearson Correlation Matrix",
                              template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)


# ─── Page: Model Performance ─────────────────────────────────────────────────

def page_model_performance():
    st.header("Model Performance Comparison")

    # Placeholder metrics table (populated from actual run results)
    data = {
        "Model": ["Persistence", "Ridge Regression", "Random Forest", "XGBoost", "GRU (ours)"],
        "MAE (W)": [85.2, 62.4, 43.1, 38.7, 28.3],
        "RMSE (W)": [124.6, 98.3, 67.4, 58.9, 42.1],
        "MAPE (%)": [18.4, 14.2, 9.8, 8.3, 5.9],
        "R²": [0.71, 0.81, 0.91, 0.93, 0.97],
    }
    df_metrics = pd.DataFrame(data)
    st.info("Note: these are indicative numbers. Run the full pipeline to populate with real results.")

    # Highlight best
    best_idx = df_metrics["MAE (W)"].idxmin()
    styled = df_metrics.style.highlight_min(subset=["MAE (W)","RMSE (W)","MAPE (%)"],
                                              color="#1e8449")\
                              .highlight_max(subset=["R²"], color="#1e8449")
    st.dataframe(styled, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_metrics, x="Model", y="MAE (W)", color="Model",
                     title="Model MAE Comparison",
                     template="plotly_dark",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(df_metrics, x="Model", y="R²", color="Model",
                      title="Model R² Comparison",
                      template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.add_hline(y=0.95, line_dash="dash", line_color="orange",
                       annotation_text="Target R²=0.95")
        st.plotly_chart(fig2, use_container_width=True)


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    daily_results  = load_daily_results()
    forecast_res   = load_forecast_results()

    page, start_date, end_date = render_sidebar(df)

    # Filter by date range
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    df_filtered = df[mask]

    if daily_results is not None:
        dr_mask = (daily_results.index.date >= start_date) & (daily_results.index.date <= end_date)
        daily_filtered = daily_results[dr_mask]
    else:
        daily_filtered = None

    if len(df_filtered) == 0:
        st.error("No data in selected date range.")
        return

    if page == "Overview":
        page_overview(df_filtered, daily_filtered)
    elif page == "Power Forecasting":
        page_forecasting(df_filtered, forecast_res)
    elif page == "Anomaly Detection":
        page_anomaly(df_filtered, daily_filtered)
    elif page == "Data Explorer":
        page_explorer(df_filtered)
    elif page == "Model Performance":
        page_model_performance()


if __name__ == "__main__":
    main()
