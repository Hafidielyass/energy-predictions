"""
src.anomaly_detection — Hybrid Fault Detection & Classification
================================================================

Modules
-------
anomaly_detector : HybridAnomalyDetector — 4-signal composite scoring

Methodology
-----------
No ground-truth fault labels exist, so we combine four complementary
unsupervised signals into a single composite anomaly score [0, 1]:

  Score = 0.30 × Isolation Forest score
        + 0.30 × Autoencoder reconstruction error
        + 0.25 × Statistical threshold violations
        + 0.15 × Power curve shape deviation

Severity mapping (configurable in configs/config.yaml):
  score < 0.35          → Normal
  0.35 ≤ score < 0.65   → Moderately Faulty
  score ≥ 0.65          → Severely Faulty

Detected fault patterns
-----------------------
* Inverter trips          — zero power mid-day with high irradiance
* Partial shading         — suppressed bell curve shape
* Soiling / degradation   — declining performance ratio over time
* Sensor faults           — erratic irradiance-power mismatch
* Wiring issues           — anomalous DC-AC conversion ratio

Evaluation metrics (clustering quality)
----------------------------------------
* Silhouette Score        — intra-cluster cohesion vs inter-cluster separation
* Davies-Bouldin Index    — compactness / separation ratio (lower = better)

Key classes / functions
-----------------------
HybridAnomalyDetector   — fit(), predict(), evaluate_clustering(), save(), load()
build_autoencoder(n_features, encoding_dims)  → (autoencoder, encoder)
SEVERITY_LABELS         — {0: 'Normal', 1: 'Moderately Faulty', 2: 'Severely Faulty'}
"""

from src.anomaly_detection.anomaly_detector import (
    HybridAnomalyDetector,
    build_autoencoder,
    SEVERITY_LABELS,
)

__all__ = ["HybridAnomalyDetector", "build_autoencoder", "SEVERITY_LABELS"]
