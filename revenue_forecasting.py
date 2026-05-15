import pandas as pd
import sqlite3
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# 1. Load & Clean Data
df = pd.read_excel("Online Retail.xlsx")

data = df.copy()

data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
data = data.dropna(subset=["InvoiceDate"])

data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
data["Revenue"] = data["Quantity"] * data["UnitPrice"]


# 2. Store in SQLite & Aggregate
conn = sqlite3.connect("retail.db")
data.to_sql("retail", conn, if_exists="replace", index=False)

monthly_revenue = pd.read_sql_query(
    """
    SELECT
        strftime('%Y-%m', InvoiceDate) AS month,
        SUM(Revenue) AS monthly_revenue
    FROM retail
    GROUP BY month
    ORDER BY month;
    """,
    conn
)

conn.close()

monthly_revenue["month"] = pd.to_datetime(monthly_revenue["month"])
monthly_revenue["monthly_revenue"] = monthly_revenue["monthly_revenue"].astype(float)

monthly_revenue = monthly_revenue.sort_values("month")

# Drop last point if needed
ts_data = monthly_revenue.iloc[:-1].copy()


# 3. Plot Monthly Revenue
plt.figure(figsize=(10, 4))
plt.plot(ts_data["month"], ts_data["monthly_revenue"], marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()


# 4. Baseline Forecasts
ts_data = ts_data.set_index("month")

# Naive forecast
ts_data["naive_forecast"] = ts_data["monthly_revenue"].shift(1)

# 3-month moving average
ts_data["ma_3"] = ts_data["monthly_revenue"].rolling(window=3, min_periods=3).mean()

plt.figure(figsize=(10, 4))
plt.plot(ts_data.index, ts_data["monthly_revenue"], label="Actual", marker="o")
plt.plot(ts_data.index, ts_data["naive_forecast"], label="Naive Forecast", linestyle="--")
plt.plot(ts_data.index, ts_data["ma_3"], label="3-Month MA", linestyle="--")
plt.legend()
plt.title("Baseline Forecasts vs Actual")
plt.grid(True)
plt.show()

# 5. Train-Test Split
train = ts_data.iloc[:-3].copy()
test = ts_data.iloc[-3:].copy()

# 6. Holt’s Linear Trend Model
holt_model = ExponentialSmoothing(
    train["monthly_revenue"],
    trend="add",
    seasonal=None,
    initialization_method="estimated"
)

holt_fit = holt_model.fit()
holt_forecast = holt_fit.forecast(len(test))

# 7. Plot Holt Forecast
plt.figure(figsize=(10, 4))
plt.plot(train.index, train["monthly_revenue"], label="Train")
plt.plot(test.index, test["monthly_revenue"], label="Actual", marker="o")
plt.plot(test.index, holt_forecast, label="Holt Forecast", linestyle="--")
plt.legend()
plt.title("Holt's Linear Trend Forecast vs Actual")
plt.grid(True)
plt.show()


# 8. Evaluation Metrics
def eval_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted))
    }

holt_metrics = eval_metrics(test["monthly_revenue"], holt_forecast)
naive_metrics = eval_metrics(test["monthly_revenue"], test["naive_forecast"])
ma_metrics = eval_metrics(test["monthly_revenue"], test["ma_3"])

print("Holt:", holt_metrics)
print("Naive:", naive_metrics)
print("3-MA:", ma_metrics)

# 9. Future Forecast (3-Month MA)
last_ma = ts_data["monthly_revenue"].iloc[-3:].mean()
forecast_values = [last_ma] * 3

forecast_index = pd.date_range(
    start=ts_data.index[-1] + pd.offsets.MonthBegin(1),
    periods=3,
    freq="MS"
)

ma_forecast = pd.Series(forecast_values, index=forecast_index)

# 10. Tableau-Ready Output
actuals = ts_data[["monthly_revenue"]].copy()
actuals["type"] = "Actual"

forecast_df = pd.DataFrame({
    "monthly_revenue": ma_forecast,
    "type": "Forecast"
})

tableau_df = pd.concat([actuals, forecast_df])
tableau_df = tableau_df.reset_index().rename(columns={"index": "month"})

print(tableau_df.head())
print(tableau_df.tail())
