import os
import sys
import pandas as pd
import numpy as np
from arch import arch_model

# ---------------- PATH SETUP ---------------- #

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_loader import load_data


# ---------------- GARCH MODEL ---------------- #

def run_garch_all():
    """
    Fit GARCH(1,1) model for each stock
    using last 500 trading days.
    Returns volatility forecast values.
    """

    data_dict = load_data()

    results = {}

    for stock, df in data_dict.items():

        try:

            print(f"Estimating GARCH for {stock}")

            if 'Returns' not in df.columns:
                df['Returns'] = df['Close'].pct_change()

            df = df.dropna()

            # Use last 500 days
            if len(df) > 500:
                df = df.tail(500)

            returns = df['Returns'] * 100

            if len(returns) < 50:
                print(f"Skipping {stock}: insufficient data")
                continue

            # Fit GARCH model
            model = arch_model(
                returns,
                vol='Garch',
                p=1,
                q=1,
                dist='normal'
            )

            fitted = model.fit(disp="off")

            # Forecast next 30 days volatility
            forecast = fitted.forecast(horizon=30)

            variance_forecast = forecast.variance.iloc[-1].values

            # Save variance path
            results[stock] = variance_forecast.tolist()

        except Exception as e:

            print(f"GARCH error for {stock}: {e}")

            continue

    return results
