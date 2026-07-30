"""
src.visualization — Professional Plot Library
==============================================

Modules
-------
plots : 11 publication-ready Matplotlib/Seaborn chart functions.
        All functions return a ``matplotlib.figure.Figure`` and save a PNG
        to ``reports/figures/``.

Available plots
---------------
plot_daily_profiles(df, target_col, n_days, start_date)
    Overlay hourly power curves for n consecutive days.

plot_seasonal_trends(df, target_col)
    Monthly median bar chart + seasonal box plots.

plot_power_vs_irradiance(df)
    Scatter: AC power vs POA irradiance, coloured by hour / month.

plot_temperature_effects(df)
    Power vs module temperature (irradiance-controlled) + seasonal histogram.

plot_correlation_heatmap(df)
    Pearson correlation matrix of all sensor columns.

plot_forecast_results(y_true, y_pred, timestamps, n_days, title)
    Actual vs GRU forecast overlay + residual panel.

plot_forecast_errors(y_true, y_pred)
    Error histogram + scatter actual/predicted + horizon-wise MAE.

plot_normal_vs_abnormal(df, daily_result, target_col, n_per_class)
    Side-by-side power profiles for Normal / Moderate / Severe days.

plot_anomaly_scores(daily_result)
    Time-series of composite anomaly score with severity shading bands.

plot_model_comparison(results)
    Bar charts comparing MAE / RMSE / MAPE across all models.

plot_shap_importance(shap_values, feature_names, max_display)
    Horizontal bar chart of mean |SHAP| feature importances.
"""

from src.visualization.plots import (
    plot_daily_profiles,
    plot_seasonal_trends,
    plot_power_vs_irradiance,
    plot_temperature_effects,
    plot_correlation_heatmap,
    plot_forecast_results,
    plot_forecast_errors,
    plot_normal_vs_abnormal,
    plot_anomaly_scores,
    plot_model_comparison,
    plot_shap_importance,
)

__all__ = [
    "plot_daily_profiles",
    "plot_seasonal_trends",
    "plot_power_vs_irradiance",
    "plot_temperature_effects",
    "plot_correlation_heatmap",
    "plot_forecast_results",
    "plot_forecast_errors",
    "plot_normal_vs_abnormal",
    "plot_anomaly_scores",
    "plot_model_comparison",
    "plot_shap_importance",
]
