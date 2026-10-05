from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train-test.csv"
VALID_PATH = ROOT / "validation.csv"
PRED_TEMPLATE = ROOT / "validation-predictions-template.csv"
DECEMBER_TEMPLATE = ROOT / "december-chart-inputs.csv"
OUT_PREDICTIONS = ROOT / "validation_predictions.csv"
OUT_DECEMBER = ROOT / "december_chart_inputs.csv"
DATA_DECEMBER = ROOT / "data" / "december_chart_inputs.csv"
METRICS_PATH = ROOT / "validation_metrics.txt"

CAT_FEATURES = ["equipment", "pickup", "delivery"]
NUM_FEATURES = [
    "distance",
    "log_distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "week",
    "is_weekend",
    "dist_x_market",
    "dist_x_quote",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["log_distance"] = np.log1p(out["distance"].astype(float))
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofyear"] = out["date"].dt.dayofyear
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["dist_x_market"] = out["distance"] * out["market_index"]
    out["dist_x_quote"] = out["distance"] * out["quote_signal"]
    for col in CAT_FEATURES:
        out[col] = out[col].astype("category")
    return out


def city_coords(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    for city, lat, lon in zip(train["pickup"], train["pickup_lat"], train["pickup_lon"]):
        coords[city] = (float(lat), float(lon))
    for city, lat, lon in zip(train["delivery"], train["delivery_lat"], train["delivery_lon"]):
        coords[city] = (float(lat), float(lon))
    return coords


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE%": mape,
        "R2": r2_score(y_true, y_pred),
    }


def format_metrics(name: str, scores: dict[str, float]) -> str:
    return (
        f"{name}: MAE={scores['MAE']:.2f}  RMSE={scores['RMSE']:.2f}  "
        f"MAPE={scores['MAPE%']:.2f}%  R2={scores['R2']:.4f}"
    )


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=500,
        max_depth=7,
        max_leaf_nodes=63,
        min_samples_leaf=25,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        categorical_features=CAT_FEATURES,
    )


def predict_positive(model: HistGradientBoostingRegressor, X: pd.DataFrame) -> np.ndarray:
    preds = np.expm1(model.predict(X[ALL_FEATURES]))
    return np.clip(preds, 50.0, None)


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    valid_raw = pd.read_csv(VALID_PATH)
    train = add_features(train_raw)
    valid = add_features(valid_raw)

    y = np.log1p(train["posted_rate"].astype(float).to_numpy())
    X = train[ALL_FEATURES]

    holdout_mask = train["date"] >= "2025-10-01"
    model_holdout = make_model()
    model_holdout.fit(X.loc[~holdout_mask], y[~holdout_mask])
    holdout_pred = predict_positive(model_holdout, X.loc[holdout_mask])
    holdout_true = train.loc[holdout_mask, "posted_rate"].to_numpy()
    holdout_scores = metrics(holdout_true, holdout_pred)

    mid_mask = train["date"] >= "2025-09-01"
    model_mid = make_model()
    model_mid.fit(X.loc[~mid_mask], y[~mid_mask])
    mid_pred = predict_positive(model_mid, X.loc[mid_mask])
    mid_true = train.loc[mid_mask, "posted_rate"].to_numpy()
    mid_scores = metrics(mid_true, mid_pred)

    model = make_model()
    model.fit(X, y)
    in_sample = metrics(train["posted_rate"].to_numpy(), predict_positive(model, X))

    valid_pred = predict_positive(model, valid)
    template = pd.read_csv(PRED_TEMPLATE)
    pred_map = dict(zip(valid["load_id"].astype(str), valid_pred))
    template["predicted_rate"] = (
        template["load_id"].astype(str).map(pred_map).astype(float).round(2)
    )
    if template["predicted_rate"].isna().any():
        raise SystemExit("Missing predictions for some load_id values")
    template.to_csv(OUT_PREDICTIONS, index=False)

    coords = city_coords(train_raw)
    lex_lat, lex_lon = coords["Lexington"]
    fw_lat, fw_lon = coords["Fort Wayne"]
    daily = (
        valid.groupby(valid["date"].dt.normalize())[["market_index", "quote_signal"]]
        .mean()
        .rename_axis("date")
        .reset_index()
    )

    december = pd.read_csv(DECEMBER_TEMPLATE)
    december["date"] = pd.to_datetime(december["date"])
    december = december.merge(daily, on="date", how="left")
    december["pickup_lat"] = lex_lat
    december["pickup_lon"] = lex_lon
    december["delivery_lat"] = fw_lat
    december["delivery_lon"] = fw_lon
    december_feat = add_features(december)
    december_rates = predict_positive(model, december_feat)

    december_out = pd.read_csv(DECEMBER_TEMPLATE)
    december_out["predicted_rate"] = np.round(december_rates, 2)
    december_out.to_csv(OUT_DECEMBER, index=False)
    DATA_DECEMBER.parent.mkdir(parents=True, exist_ok=True)
    december_out.to_csv(DATA_DECEMBER, index=False)

    lines = [
        format_metrics("Holdout Oct 2025 (trained on Jan-Sep)", holdout_scores),
        format_metrics("Holdout Sep-Oct 2025 (trained on Jan-Aug)", mid_scores),
        format_metrics("In-sample full train", in_sample),
        f"Trees used on final model: {model.n_iter_}",
        f"Validation predictions: {len(template)}",
        f"December mean predicted rate: {float(np.mean(december_rates)):.2f}",
    ]
    METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {OUT_PREDICTIONS}")
    print(f"Wrote {OUT_DECEMBER}")


if __name__ == "__main__":
    main()
