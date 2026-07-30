# Solar PV Forecasting & Anomaly Detection

End-to-end ML pipeline for solar photovoltaic energy forecasting and anomaly detection using TensorFlow, FastAPI, and Streamlit.

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows
# source venv/bin/activate # Linux

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Core ML pipeline: preprocessing, features, forecasting, anomaly detection, utils, visualization |
| `api/` | FastAPI REST API (`main.py`) |
| `dashboards/` | Streamlit dashboards |
| `configs/` | YAML configuration |
| `data/` | Raw & processed data |
| `models/` | Trained model artifacts |
| `notebooks/` | Jupyter notebooks |
| `tests/` | Unit/integration tests |
| `reports/` | Generated reports |
| `experiments/` | Experiment tracking |

## Usage

- **API**: `uvicorn api.main:app --reload`
- **Dashboard**: `streamlit run dashboards/main.py`
- **Notebooks**: `jupyter notebook`

## Components

- **Preprocessing** (`src/preprocessing/`): Data cleaning, normalization, splitting
- **Feature Engineering** (`src/features/`): Weather features, time-based features, lag variables
- **Forecasting** (`src/forecasting/`): LSTM, GRU, Transformer, XGBoost models
- **Anomaly Detection** (`src/anomaly_detection/`): Reconstruction error, residual analysis
- **Deployment** (`src/deployment/`): Model serving utilities
- **Visualization** (`src/visualization/`): Charts, plots, dashboards
