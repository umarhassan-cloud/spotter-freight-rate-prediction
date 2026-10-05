# Freight Rate Prediction Report

## Task

Predict `posted_rate` for 12,000 unseen loads in `validation.csv` (Nov–Dec 2025) and produce a 31-day December 2025 rate path for a fixed Lexington → Fort Wayne Dry Van load (360 miles, 32,000 lb).

## Data

| File | Rows | Role |
|---|---|---|
| `train-test.csv` | 48,000 | Labeled loads, 2025-01-01 to 2025-10-31 |
| `validation.csv` | 12,000 | Unlabeled loads, 2025-11-01 to 2025-12-31 |
| `december-chart-inputs.csv` | 31 | Fixed lane; only `date` changes |

Target: `posted_rate`. Distance is the strongest driver (correlation ≈ 0.91). Equipment type shifts rate-per-mile (Reefer > Flatbed > Dry Van). `weight` and `market_index` have missing values in both train and validation; they are left as missing so the tree model can learn a missingness pattern.

Validation includes a few cities that never appear in train (for example Chicago, Laredo). Coordinates are still present, so geography is modeled with lat/lon plus city categories.

## Split approach

This is a time-series problem: validation sits entirely after the training window. Random K-fold would leak future market conditions.

1. **Primary holdout:** train on Jan–Sep 2025, score October 2025.
2. **Sensitivity holdout:** train on Jan–Aug 2025, score Sep–Oct 2025.
3. **Final model:** refit on all 48,000 labeled rows, then predict November–December.

October is the closest calendar month to the true scoring window, so it is the main estimate of generalization.

## Model

`HistGradientBoostingRegressor` on `log1p(posted_rate)`, then `expm1` and a floor of $50 so every rate is positive.

Features:

- Distance, `log(distance)`, weight, market index, quote signal
- Pickup/delivery coordinates
- Equipment, pickup city, delivery city (native categorical features)
- Calendar: month, day, day-of-week, day-of-year, week, weekend flag
- Interactions: `distance × market_index`, `distance × quote_signal`

Missing `weight` and `market_index` are handled natively by histogram gradient boosting.

## Internal validation results

| Split | MAE ($) | RMSE ($) | MAPE | R² |
|---|---|---|---|---|
| Train Jan–Sep, test Oct | 135.38 | 650.47 | 6.41% | 0.819 |
| Train Jan–Aug, test Sep–Oct | 136.33 | 640.95 | 5.92% | 0.824 |
| Full-train in-sample | 101.09 | 552.04 | 4.36% | 0.862 |

MAE and MAPE are the more useful numbers here. RMSE is inflated by a small number of extreme loads (rate-per-mile well above $5). Those outliers exist in train and are left in, because they may also appear in the hidden test.

Spotter computes the official validation metric after submission.

## December chart

The December file has no `market_index`, `quote_signal`, or coordinates. Those fields are filled as follows:

- Lexington / Fort Wayne coordinates from train
- Daily mean `market_index` and `quote_signal` from validation loads on the same calendar date

That keeps the lane fixed while still letting the date-driven market move, which is why the chart shows a weekly-looking wave around **$821–$851** (mean **$840.31**). Historical Lexington → Fort Wayne Dry Van loads in train sat in a similar band (roughly $760–$970).

Chart file: `scorer_results/candidate_december.png`

## How to reproduce

```bash
python -m pip install -r requirements.txt
python train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Outputs:

- `validation_predictions.csv` — 12,000 `load_id`, `predicted_rate` rows
- `december_chart_inputs.csv` and `data/december_chart_inputs.csv` — 31 filled December rows
- `scorer_results/candidate_december.png` — official chart
