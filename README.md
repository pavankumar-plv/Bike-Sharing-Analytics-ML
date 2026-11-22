# 🚲 BikeShare Analytics_ML — Streamlit Demand Dashboard

An interactive bike-sharing demand analytics and prediction app built from the supplied project notebook and dataset.

## Features
- Historical demand KPIs and interactive EDA charts
- Filters for season, weather and month
- Hourly bike-rental demand scenario simulator
- Random Forest regression with a chronological date-based holdout
- MAE, RMSE and R² evaluation plus prediction diagnostics
- Original notebook and project reference documents

## Project structure
```text
bikeshare-ai-streamlit/
├── app.py
├── requirements.txt
├── data/bike_sharing.csv
├── notebooks/BikeShare_AI_Interactive_Project.ipynb
└── docs/ (project reference documents)
```

## Run locally
Python 3.10+ is recommended.
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects the supplied dataset at `data/bike_sharing.csv`.

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**, select this repository and branch `main`.
3. Set the main file path to `app.py`, then deploy.

## Modeling notes
- Target: `cnt` (total hourly rentals).
- `casual`, `registered`, and `instant` are excluded from predictors.
- The final 20% of unique dates are held out chronologically.
- Missing predictors are imputed inside the scikit-learn pipeline.
- The app uses a Random Forest model for prediction.
