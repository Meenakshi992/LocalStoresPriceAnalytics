import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# ---------------------------
# Part-4
# ---------------------------

# This module generates 90-day historical price series, 
# engineers calendar/flyer cycle features, trains a 
# Ridge Regression time-series model, evaluates MAE, 
# and forecasts future daily prices.
def generate_historical_prices(stores_df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """Generates 90 days of historical daily prices with weekly flyer discount cycles."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    unique_items = stores_df[["Item", "Store", "Price"]].drop_duplicates()

    records = []
    for _, row in unique_items.iterrows():
        base = row["Price"]
        for i, d in enumerate(dates):
            # Weekly flyer discount (Thu=3, Fri=4, Sat=5, Sun=6)
            promo_effect = -0.06 * base if d.dayofweek in [3, 4, 5, 6] else 0.02 * base
            inflation_trend = 0.0015 * i
            noise = np.random.normal(0, 0.08)
            daily_price = max(0.99, round(base + promo_effect + inflation_trend + noise, 2))
            records.append({
                "Date": d,
                "Store": row["Store"],
                "Item": row["Item"],
                "Price": daily_price
            })
    return pd.DataFrame(records)


def _build_time_features(df: pd.DataFrame, min_date: pd.Timestamp) -> pd.DataFrame:
    """Creates calendar and cyclical promotional features for ML training."""
    feat = df.copy()
    feat["day_idx"] = (feat["Date"] - min_date).dt.days
    feat["dayofweek"] = feat["Date"].dt.dayofweek
    feat["is_flyer_weekend"] = feat["dayofweek"].isin([3, 4, 5, 6]).astype(int)
    for dow in range(7):
        feat[f"dow_{dow}"] = (feat["dayofweek"] == dow).astype(int)
    return feat


def train_and_forecast_prices(hist_df: pd.DataFrame, store: str, item: str, forecast_days: int = 7) -> dict:
    """Trains Ridge Regression model, evaluates validation MAE, and predicts future daily prices."""
    subset = hist_df[(hist_df["Store"] == store) & (hist_df["Item"] == item)].sort_values("Date").copy()
    min_date = subset["Date"].min()
    feat_df = _build_time_features(subset, min_date)

    feature_cols = ["day_idx", "is_flyer_weekend"] + [f"dow_{dow}" for dow in range(7)]

    # Train/Validation Split (Last 14 days for MAE evaluation)
    train_df = feat_df.iloc[:-14]
    val_df = feat_df.iloc[-14:]

    model = Ridge(alpha=1.0)
    model.fit(train_df[feature_cols], train_df["Price"])
    val_preds = model.predict(val_df[feature_cols])
    mae = round(mean_absolute_error(val_df["Price"], val_preds), 3)

    # Retrain on 100% of historical data for future forecast
    model.fit(feat_df[feature_cols], feat_df["Price"])

    last_date = subset["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    future_raw = pd.DataFrame({"Date": future_dates})
    future_feat = _build_time_features(future_raw, min_date)
    future_feat["Predicted_Price"] = model.predict(future_feat[feature_cols]).round(2)

    return {
        "history": subset.tail(30),
        "forecast": future_feat[["Date", "Predicted_Price"]],
        "mae": mae,
        "best_day": future_feat.loc[future_feat["Predicted_Price"].idxmin()]
    }
