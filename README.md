# ⚡ Electricity Bill Predictor

A machine learning project that forecasts **daily household electricity consumption (kWh)** from historical usage and converts the forecast into an **estimated electricity bill**.

Two models are compared: **Linear Regression** and a **Decision Tree Regressor**, plus a naive "yesterday" baseline. Performance is measured with **Mean Absolute Error (MAE)**.

## Project Overview

| Step | What happens |
|------|--------------|
| 1. Load | Raw UCI `.txt` file is converted to CSV; `?` becomes `NaN` |
| 2. Clean | Duplicates removed, missing readings handled, minute data aggregated to daily kWh |
| 3. Explore | Trend, monthly seasonality, weekday pattern, distribution plots |
| 4. Features | Date features (day, month, weekday, year, weekend flag, day-of-year) + lag features (1, 2, 3, 7, 14 days) + 7-day rolling mean |
| 5. Split | **Chronological** 80/20 split, no shuffling |
| 6. Train | Linear Regression vs Decision Tree |
| 7. Evaluate | MAE (plus RMSE, R²), actual-vs-predicted plots |
| 8. Forecast | Recursive multi-day forecast beyond the last date in the data |
| 9. Bill | Slab-tariff bill estimate for the forecast period |

## Project Structure

```
electricity-bill-predictor/
├── main.py             # full pipeline (heavily commented)
├── requirements.txt    # dependencies
├── README.md
├── .gitignore
├── data/               # dataset goes here (auto-downloaded)
└── outputs/            # generated plots, metrics, forecasts
```

## Setup

```bash
# 1. clone and enter the project
git clone <your-repo-url>
cd electricity-bill-predictor

# 2. (optional) virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt
```

## Run

```bash
python main.py                                  # downloads dataset automatically
python main.py --data data/household_power_consumption.txt   # use a local file
python main.py --forecast-days 60 --test-size 0.25
python main.py --synthetic                      # offline demo with fake data
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--data` | none | Path to raw `.txt` or converted `.csv` |
| `--data-dir` | `data` | Dataset folder |
| `--output-dir` | `outputs` | Where results are saved |
| `--test-size` | `0.2` | Fraction of the latest days used for testing |
| `--forecast-days` | `30` | Days to forecast past the dataset end |
| `--synthetic` | off | Use generated demo data (not real results) |

## Dataset

**Individual Household Electric Power Consumption** (UCI Machine Learning Repository)

- One household in Sceaux, France, Dec 2006 to Nov 2010
- About 2.07 million minute-level readings
- Fields: date, time, global active power (kW), global reactive power, voltage, global intensity, three sub-metering values (Wh)
- About 1.25% of rows are missing, stored as `?`
- Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

**Processing used here**

- Daily energy (kWh) = mean `Global_active_power` (kW) over the day × 24
- Days with less than 50% valid readings are marked missing and filled by time interpolation
- Leading/trailing empty days are dropped

## Methodology Notes

- **No data leakage:** lag and rolling features use only past values (`shift`), and the split is chronological.
- **Recursive forecasting:** each predicted day is fed back as input for the next day's lags.
- **Tree limitation:** Decision Trees cannot extrapolate beyond the value range seen in training, so Linear Regression often handles trend better. Check the metrics.
- **Baseline:** the naive "yesterday" predictor shows whether the models add real value.

## Bill Estimation

An illustrative slab tariff is applied per 30-day period (edit `TARIFF_SLABS`, `FIXED_CHARGE`, `CURRENCY` in `main.py` to match a real tariff):

| Units (kWh) | Rate per kWh |
|-------------|--------------|
| First 100 | 3.5 |
| 101 to 300 | 5.5 |
| Above 300 | 7.5 |
| Fixed charge | 50 per period |

## Evaluation Metrics Summary

Running `main.py` prints and saves `outputs/metrics.csv`. Paste your results here after running on the real dataset:

| Model | MAE (kWh/day) | RMSE | R² |
|-------|---------------|------|----|
| Naive (yesterday) | _fill in_ | _fill in_ | _fill in_ |
| Linear Regression | _fill in_ | _fill in_ | _fill in_ |
| Decision Tree | _fill in_ | _fill in_ | _fill in_ |

The bill error (mean absolute difference between actual and predicted bill per 30-day period) is printed and saved in `outputs/summary.json`.

## Outputs

| File | Description |
|------|-------------|
| `01_eda_patterns.png` | Trend, monthly, weekday, distribution |
| `02_actual_vs_predicted.png` | Test-set time series: actual vs both models |
| `03_scatter_actual_vs_predicted.png` | Prediction accuracy scatter |
| `04_future_forecast.png` | Forecast beyond the dataset |
| `05_bill_comparison.png` | Actual vs predicted bill per period |
| `metrics.csv`, `summary.json` | Metrics, bills, forecast summary |
| `forecast_next_days.csv` | Day-by-day forecast |
| `daily_consumption_clean.csv` | Cleaned daily series |

## Possible Improvements

- Add weather and holiday features
- Try Random Forest / Gradient Boosting / SARIMA
- Time-series cross-validation (`TimeSeriesSplit`) and hyperparameter tuning
- Use sub-metering columns for appliance-level insights

## License

MIT (or your institution's choice).
