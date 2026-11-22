import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="BikeShare AI | Demand Intelligence", page_icon="🚲", layout="wide")
DATA_PATH = Path(__file__).parent / "bike_sharing.csv"

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH, dtype=str).replace("?", np.nan)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["dteday"] = pd.to_datetime(raw["dteday"], format="%d-%m-%Y", errors="coerce")
    numeric = ["instant", "yr", "mnth", "hr", "weekday", "temp", "atemp", "hum",
               "windspeed", "casual", "registered", "cnt"]
    for col in numeric:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["dteday", "cnt"]).copy()
    raw["year_from_date"] = raw["dteday"].dt.year
    raw["month_from_date"] = raw["dteday"].dt.month
    raw["day_of_month"] = raw["dteday"].dt.day
    raw["day_of_week_from_date"] = (raw["dteday"].dt.dayofweek + 1) % 7
    raw["is_weekend"] = raw["day_of_week_from_date"].isin([0, 6]).astype(float)
    return raw.sort_values(["dteday", "hr"]).reset_index(drop=True)

@st.cache_resource
def train_model(data):
    categorical = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
                   "weathersit", "hr", "month_from_date", "day_of_week_from_date"]
    excluded = ["cnt", "casual", "registered", "instant", "dteday"]
    features = [c for c in data.columns if c not in excluded]
    numeric = [c for c in features if c not in categorical]
    X, y = data[features], data["cnt"].astype(float)
    unique_dates = np.array(sorted(data["dteday"].dropna().unique()))
    split_at = max(1, int(len(unique_dates) * .8))
    test_start = pd.Timestamp(unique_dates[split_at])
    train_mask = data["dteday"] < test_start
    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_test = y.loc[train_mask], y.loc[~train_mask]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([
        ("preprocessor", pre),
        ("model", RandomForestRegressor(n_estimators=180, max_features=1.0,
                                         min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"MAE": mean_absolute_error(y_test, pred),
               "RMSE": mean_squared_error(y_test, pred) ** .5,
               "R²": r2_score(y_test, pred)}
    return model, features, metrics, y_test, pred

df = load_data()
model, feature_columns, metrics, y_test, test_pred = train_model(df)

st.title("🚲 BikeShare AI")
st.caption("Hourly bike-sharing demand intelligence · Explore historical patterns and simulate rental demand")
st.info("Business objective: predict hourly bike demand to improve bike availability, reduce customer waiting time, and support operational planning.")

with st.sidebar:
    st.header("Explore the data")
    seasons = ["All"] + sorted(df["season"].dropna().astype(str).unique().tolist())
    weather = ["All"] + sorted(df["weathersit"].dropna().astype(str).unique().tolist())
    selected_season = st.selectbox("Season", seasons)
    selected_weather = st.selectbox("Weather", weather)
    month = st.selectbox("Month", ["All"] + list(range(1, 13)))

filtered = df.copy()
if selected_season != "All":
    filtered = filtered[filtered["season"] == selected_season]
if selected_weather != "All":
    filtered = filtered[filtered["weathersit"] == selected_weather]
if month != "All":
    filtered = filtered[filtered["mnth"] == month]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hourly records", f"{len(df):,}")
c2.metric("Average rentals / hour", f"{df.cnt.mean():,.0f}")
c3.metric("Median rentals / hour", f"{df.cnt.median():,.0f}")
peak_hour = int(df.groupby("hr")["cnt"].mean().idxmax())
c4.metric("Peak average hour", f"{peak_hour:02d}:00")

tab1, tab2, tab3 = st.tabs(["📊 Demand explorer", "🤖 Demand predictor", "🧪 Model performance"])
with tab1:
    st.subheader("Demand patterns")
    left, right = st.columns(2)
    with left:
        hourly = filtered.groupby("hr", as_index=False)["cnt"].mean()
        st.plotly_chart(px.line(hourly, x="hr", y="cnt", markers=True,
            title="Average rentals by hour", labels={"hr": "Hour", "cnt": "Average rentals"}), use_container_width=True)
    with right:
        monthly = filtered.groupby("mnth", as_index=False)["cnt"].mean()
        st.plotly_chart(px.bar(monthly, x="mnth", y="cnt", title="Average rentals by month",
            labels={"mnth": "Month", "cnt": "Average rentals"}), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.box(filtered, x="season", y="cnt", color="season",
            title="Demand by season", labels={"season": "Season", "cnt": "Hourly rentals"}), use_container_width=True)
    with right:
        st.plotly_chart(px.box(filtered, x="weathersit", y="cnt", color="weathersit",
            title="Demand by weather", labels={"weathersit": "Weather", "cnt": "Hourly rentals"}), use_container_width=True)
    st.plotly_chart(px.histogram(filtered, x="cnt", nbins=40, title="Distribution of hourly rentals"),
                    use_container_width=True)
    st.caption(f"Charts reflect {len(filtered):,} records after applying the sidebar filters.")

with tab2:
    st.subheader("Scenario simulator")
    st.write("Choose conditions to estimate total bike rentals for one hour. Temperature, humidity and wind speed use the dataset's normalized scales (0–1).")
    a, b, c = st.columns(3)
    with a:
        scenario_date = st.date_input("Scenario date", value=pd.Timestamp("2012-08-07").date())
        hour = st.slider("Hour of day", 0, 23, 8)
        season_value = st.selectbox("Season", sorted(df.season.dropna().unique().tolist()))
        weather_value = st.selectbox("Weather", sorted(df.weathersit.dropna().unique().tolist()))
    with b:
        holiday = st.selectbox("Holiday", sorted(df.holiday.dropna().unique().tolist()))
        workingday = st.selectbox("Working day", sorted(df.workingday.dropna().unique().tolist()))
        temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.60, 0.01)
        atemp = st.slider("Feels-like temperature (normalized)", 0.0, 1.0, 0.60, 0.01)
    with c:
        humidity = st.slider("Humidity (normalized)", 0.0, 1.0, 0.55, 0.01)
        wind = st.slider("Wind speed (normalized)", 0.0, 1.0, 0.20, 0.01)
        st.write(" ")
        st.write(" ")
    if st.button("Predict hourly rentals", type="primary", use_container_width=True):
        weekday_code = (pd.Timestamp(scenario_date).dayofweek + 1) % 7
        row = {
            "season": season_value, "yr": scenario_date.year, "mnth": scenario_date.month,
            "hr": hour, "holiday": holiday, "weekday": weekday_code,
            "workingday": workingday, "weathersit": weather_value,
            "temp": temp, "atemp": atemp, "hum": humidity, "windspeed": wind,
            "year_from_date": scenario_date.year, "month_from_date": scenario_date.month,
            "day_of_month": scenario_date.day, "day_of_week_from_date": weekday_code,
            "is_weekend": float(weekday_code in [0, 6])
        }
        scenario = pd.DataFrame([row], columns=feature_columns)
        estimate = max(0, float(model.predict(scenario)[0]))
        st.success(f"Estimated demand: **{estimate:,.0f} bike rentals** in this hour.")
        if scenario_date.year not in [2011, 2012]:
            st.warning("This date is outside the model's 2011–2012 training period; treat the estimate cautiously.")
        st.caption("This is a model estimate, not a guarantee of the number of bikes available.")

with tab3:
    st.subheader("Chronological holdout evaluation")
    st.write("The final 20% of unique dates are held out for testing so that future dates do not leak into training.")
    a, b, c = st.columns(3)
    a.metric("MAE", f"{metrics['MAE']:.1f} rentals")
    b.metric("RMSE", f"{metrics['RMSE']:.1f} rentals")
    c.metric("R²", f"{metrics['R²']:.3f}")
    diagnostics = pd.DataFrame({"Actual": y_test, "Predicted": test_pred})
    st.plotly_chart(px.scatter(diagnostics, x="Actual", y="Predicted",
        title="Actual vs predicted demand", opacity=.45), use_container_width=True)
    diagnostics["Residual"] = diagnostics["Actual"] - diagnostics["Predicted"]
    st.plotly_chart(px.histogram(diagnostics, x="Residual", nbins=45,
        title="Prediction residual distribution"), use_container_width=True)
    st.caption("The Streamlit app trains a Random Forest pipeline on first load. The notebook contains the fuller comparison and tuning workflow for Decision Tree, Random Forest and Gradient Boosting.")

st.divider()
st.caption("BikeShare AI · ExcelR Data Science Project · Historical dataset: 2011–2012")
