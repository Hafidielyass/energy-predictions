"""
Professional visualisation library for the PV forecasting project.

All functions return Matplotlib Figure objects so callers can
save / embed them independently. Style is set once at module load.
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for notebooks + scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.8,
    "lines.linewidth": 1.6,
})

PALETTE = sns.color_palette("Set2", 10)
SEVERITY_COLOURS = {
    "Normal": "#2ECC71",
    "Moderately Faulty": "#F39C12",
    "Severely Faulty": "#E74C3C",
}

SAVE_DIR = "reports/figures"


def _save(fig: plt.Figure, name: str) -> str:
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = f"{SAVE_DIR}/{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    return path


# ─── 1. Daily generation profiles ────────────────────────────────────────────

def plot_daily_profiles(
    df: pd.DataFrame,
    target_col: str = "ac_power__315",
    n_days: int = 7,
    start_date: Optional[str] = None,
) -> plt.Figure:
    """Plot hourly power profiles for n_days starting from start_date."""
    if start_date:
        df = df[df.index >= start_date]

    unique_days = pd.Series(df.index.date).unique()[:n_days]
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_days)))

    for i, day in enumerate(unique_days):
        day_data = df[df.index.date == day][target_col]
        ax.plot(day_data.index.hour + day_data.index.minute / 60,
                day_data.values, alpha=0.8, color=cmap[i], label=str(day))

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("AC Power (W)")
    ax.set_title("Daily AC Power Generation Profiles")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(0, 24)
    fig.tight_layout()
    _save(fig, "daily_profiles")
    return fig


# ─── 2. Seasonal trends ──────────────────────────────────────────────────────

def plot_seasonal_trends(df: pd.DataFrame, target_col: str = "ac_power__315") -> plt.Figure:
    """Monthly box plots showing seasonal generation distribution."""
    df_copy = df.copy()
    df_copy["month"]       = df_copy.index.month
    df_copy["month_label"] = df_copy.index.to_period("M").astype(str)

    monthly = df_copy.groupby("month")[target_col].median().reset_index()
    season_map = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",
                  11:"Autumn",12:"Winter"}
    monthly["season"] = monthly["month"].map(season_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: median power per month
    colors = [PALETTE[i % len(PALETTE)] for i in range(12)]
    axes[0].bar(monthly["month"], monthly[target_col], color=colors, edgecolor="white")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Median AC Power (W)")
    axes[0].set_title("Monthly Median Power Generation")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)

    # Right: seasonal box plot
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_colours = {"Winter":"#3498DB","Spring":"#2ECC71","Summer":"#F39C12","Autumn":"#E67E22"}
    df_copy["season"] = df_copy["month"].map(season_map)
    df_copy = df_copy[df_copy[target_col] > 10]   # daylight only
    groups = [df_copy[df_copy["season"] == s][target_col].dropna().values
              for s in season_order]
    bp = axes[1].boxplot(groups, labels=season_order, patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
    for patch, s in zip(bp["boxes"], season_order):
        patch.set_facecolor(season_colours[s])
    axes[1].set_ylabel("AC Power (W)")
    axes[1].set_title("Seasonal Power Distribution (Daytime only)")

    fig.tight_layout()
    _save(fig, "seasonal_trends")
    return fig


# ─── 3. Power vs Irradiance ───────────────────────────────────────────────────

def plot_power_vs_irradiance(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of AC power vs POA irradiance with density colouring."""
    sample = df[(df["ac_power__315"] > 0) & (df["poa_irradiance_clipped"] > 10)].sample(
        min(15000, len(df)), random_state=42
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    sc = axes[0].scatter(
        sample["poa_irradiance_clipped"], sample["ac_power__315"],
        alpha=0.2, s=4, c=sample.index.hour, cmap="plasma"
    )
    plt.colorbar(sc, ax=axes[0], label="Hour of Day")
    axes[0].set_xlabel("POA Irradiance (W/m²)")
    axes[0].set_ylabel("AC Power (W)")
    axes[0].set_title("Power vs Irradiance (coloured by hour)")

    # Monthly mean efficiency
    df_day = df[(df["ac_power__315"] > 0) & (df["poa_irradiance_clipped"] > 10)].copy()
    df_day["efficiency"] = df_day["ac_power__315"] / (df_day["poa_irradiance_clipped"] + 1e-6)
    df_day["month"] = df_day.index.month
    eff_monthly = df_day.groupby("month")["efficiency"].mean()
    axes[1].plot(eff_monthly.index, eff_monthly.values, "o-", color=PALETTE[1], linewidth=2)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Mean Power/Irradiance Ratio")
    axes[1].set_title("Monthly Mean PV Efficiency Proxy")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)

    fig.tight_layout()
    _save(fig, "power_vs_irradiance")
    return fig


# ─── 4. Temperature effects ──────────────────────────────────────────────────

def plot_temperature_effects(df: pd.DataFrame) -> plt.Figure:
    """Show how module temperature correlates with power output."""
    sample = df[(df["ac_power__315"] > 10) & (df["poa_irradiance_clipped"] > 50)].sample(
        min(10000, len(df)), random_state=42
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = axes[0].scatter(
        sample["module_temp_1__321"], sample["ac_power__315"],
        alpha=0.2, s=4,
        c=sample["poa_irradiance_clipped"], cmap="YlOrRd"
    )
    plt.colorbar(sc1, ax=axes[0], label="Irradiance (W/m²)")
    axes[0].set_xlabel("Module Temperature (°C)")
    axes[0].set_ylabel("AC Power (W)")
    axes[0].set_title("Power vs Module Temperature\n(controlled for irradiance)")

    # Temperature distribution by season
    df2 = df.copy()
    season_map = {1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3,12:0}
    df2["season"] = df2.index.month.map(season_map)
    season_names = ["Winter","Spring","Summer","Autumn"]
    season_cols  = ["#3498DB","#2ECC71","#F39C12","#E67E22"]
    for s, name, col in zip(range(4), season_names, season_cols):
        vals = df2[df2["season"] == s]["module_temp_1__321"].dropna()
        axes[1].hist(vals, bins=50, alpha=0.6, label=name, color=col, density=True)
    axes[1].set_xlabel("Module Temperature (°C)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Module Temperature Distribution by Season")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, "temperature_effects")
    return fig


# ─── 5. Correlation heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Pearson correlation heatmap of sensor columns."""
    sensor_cols = [c for c in df.columns if not c.startswith(("lag_", "roll_", "hour",
                   "day_", "month", "week", "season", "is_", "efficiency", "dc_ac",
                   "temp_delta", "ac_apparent", "poa_irradiance_clipped"))]
    corr = df[sensor_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Sensor Correlation Matrix", pad=15)
    fig.tight_layout()
    _save(fig, "correlation_heatmap")
    return fig


# ─── 6. Forecast results ─────────────────────────────────────────────────────

def plot_forecast_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    n_days: int = 3,
    title: str = "GRU 24h Forecast vs Actual",
) -> plt.Figure:
    """Plot actual vs predicted for the first n_days of the test set."""
    # Each row is a 96-step (24h) forecast
    periods_per_day = 96
    n_plot = min(n_days * periods_per_day, len(y_true) * y_true.shape[1])

    actual = y_true[:n_days * periods_per_day].ravel()[:n_plot]
    pred   = y_pred[:n_days * periods_per_day].ravel()[:n_plot]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    x = np.arange(len(actual))
    axes[0].plot(x, actual, label="Actual", color=PALETTE[0], alpha=0.9)
    axes[0].plot(x, pred,   label="Forecast", color=PALETTE[1], alpha=0.9, linestyle="--")
    axes[0].set_xlabel("Time Steps (15-min periods)")
    axes[0].set_ylabel("AC Power (W)")
    axes[0].set_title(title)
    axes[0].legend()

    # Error
    error = actual - pred
    axes[1].fill_between(x, error, 0, where=(error >= 0),
                         alpha=0.4, color=PALETTE[2], label="Over-forecast")
    axes[1].fill_between(x, error, 0, where=(error < 0),
                         alpha=0.4, color=PALETTE[3], label="Under-forecast")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("Forecast Error (W)")
    axes[1].set_title("Forecast Residuals")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, "forecast_results")
    return fig


# ─── 7. Forecast error analysis ──────────────────────────────────────────────

def plot_forecast_errors(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Distribution of forecast errors and by-hour RMSE."""
    errors = (y_true - y_pred).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Error histogram
    axes[0].hist(errors, bins=80, color=PALETTE[0], edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Forecast Error (W)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Forecast Error Distribution")

    # Scatter: actual vs pred
    lim = max(y_true.max(), y_pred.max())
    axes[1].scatter(y_true.ravel(), y_pred.ravel(), alpha=0.05, s=2, color=PALETTE[1])
    axes[1].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect forecast")
    axes[1].set_xlabel("Actual AC Power (W)")
    axes[1].set_ylabel("Predicted AC Power (W)")
    axes[1].set_title("Actual vs Predicted")
    axes[1].legend()

    # Horizon-wise MAE (per step in 24h forecast)
    horizon = y_true.shape[1] if y_true.ndim > 1 else 1
    if y_true.ndim > 1:
        mae_per_step = np.mean(np.abs(y_true - y_pred), axis=0)
        hours = np.arange(horizon) * 15 / 60
        axes[2].plot(hours, mae_per_step, color=PALETTE[2], linewidth=2)
        axes[2].set_xlabel("Forecast Horizon (hours)")
        axes[2].set_ylabel("MAE (W)")
        axes[2].set_title("MAE vs Forecast Horizon")
    else:
        axes[2].axis("off")

    fig.tight_layout()
    _save(fig, "forecast_errors")
    return fig


# ─── 8. Normal vs Abnormal days ──────────────────────────────────────────────

def plot_normal_vs_abnormal(
    df: pd.DataFrame,
    daily_result: pd.DataFrame,
    target_col: str = "ac_power__315",
    n_per_class: int = 3,
) -> plt.Figure:
    """Overlay sample power curves coloured by severity label."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    classes = ["Normal", "Moderately Faulty", "Severely Faulty"]

    for ax, cls in zip(axes, classes):
        days_of_class = daily_result[daily_result["severity_label"] == cls].index
        sample_days = days_of_class[:n_per_class]
        col = SEVERITY_COLOURS[cls]
        for day in sample_days:
            day_data = df[df.index.date == day.date()][target_col]
            if len(day_data) == 0:
                continue
            ax.plot(day_data.index.hour + day_data.index.minute / 60,
                    day_data.values, alpha=0.7, color=col,
                    label=str(day.date()))
        ax.set_title(f"{cls}\n(n={len(days_of_class)} days)", color=col)
        ax.set_xlabel("Hour of Day")
        ax.set_xlim(4, 22)
    axes[0].set_ylabel("AC Power (W)")
    fig.suptitle("Daily Power Profiles: Normal vs Faulty Days", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, "normal_vs_abnormal")
    return fig


# ─── 9. Anomaly scores ───────────────────────────────────────────────────────

def plot_anomaly_scores(daily_result: pd.DataFrame) -> plt.Figure:
    """Time-series of the composite anomaly score with severity bands."""
    fig, ax = plt.subplots(figsize=(16, 5))
    score = daily_result["anomaly_score"].dropna()

    ax.plot(score.index, score.values, color="steelblue", linewidth=1.0, alpha=0.8, zorder=2)

    # Shade severity zones
    ax.axhspan(0,    0.35, alpha=0.08, color="#2ECC71",  label="Normal zone")
    ax.axhspan(0.35, 0.65, alpha=0.10, color="#F39C12",  label="Moderate zone")
    ax.axhspan(0.65, 1.00, alpha=0.12, color="#E74C3C",  label="Severe zone")

    # Mark severe days
    severe = daily_result[daily_result["severity_label"] == "Severely Faulty"]
    ax.scatter(severe.index, severe["anomaly_score"], color="#E74C3C",
               zorder=5, s=30, label="Severe day")

    ax.set_xlabel("Date")
    ax.set_ylabel("Composite Anomaly Score")
    ax.set_title("Daily Anomaly Score Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "anomaly_scores")
    return fig


# ─── 10. Model comparison bar chart ──────────────────────────────────────────

def plot_model_comparison(results: Dict[str, Dict[str, float]]) -> plt.Figure:
    """Bar chart comparing MAE, RMSE, MAPE across models."""
    models  = list(results.keys())
    metrics = ["MAE", "RMSE", "MAPE"]
    n_metrics = len(metrics)
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5))
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        vals = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(x, vals, color=[PALETTE[j] for j in range(len(models))],
                      edgecolor="white", width=0.6)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Model Comparison — {metric}")

    fig.tight_layout()
    _save(fig, "model_comparison")
    return fig


# ─── 11. SHAP feature importance ─────────────────────────────────────────────

def plot_shap_importance(shap_values: np.ndarray, feature_names: List[str],
                          max_display: int = 20) -> plt.Figure:
    """Bar chart of mean |SHAP| importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[-max_display:][::-1]
    top_features = [feature_names[i] for i in idx]
    top_values   = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))[::-1]
    ax.barh(range(len(top_features)), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {max_display} Feature Importances (SHAP)")
    fig.tight_layout()
    _save(fig, "shap_importance")
    return fig
