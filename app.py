from flask import Flask, render_template, jsonify, request
import numpy as np
import os
import joblib
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models")


# ---------------- LOAD MODELS ---------------- #

def load_models():
    try:
        lstm = joblib.load(os.path.join(MODEL_PATH, "lstm_results.pkl"))
        garch = joblib.load(os.path.join(MODEL_PATH, "garch_results.pkl"))

        if not isinstance(lstm, pd.DataFrame):
            lstm = pd.DataFrame(lstm)

        lstm.columns = [c.strip() for c in lstm.columns]

        return lstm, garch

    except Exception as e:
        print("Model loading error:", e)
        return pd.DataFrame(), {}


lstm_data, garch_variances = load_models()


# ---------------- ALPHA SCORE ---------------- #

def compute_alpha_score(exp_ret, current_vol, current_price, predicted_price):

    rf_daily = 0.07 / 252

    excess_return = exp_ret - rf_daily
    vol_decimal = max(current_vol / 100, 1e-4)

    sharpe_raw = excess_return / vol_decimal
    sharpe_score = np.tanh(sharpe_raw * 0.5)

    if predicted_price > 0 and current_price > 0:
        log_ret = np.log(predicted_price / current_price)
    else:
        log_ret = 0

    momentum_score = np.tanh(log_ret * 30)

    if current_vol <= 1.5:
        vol_score = 1
    elif current_vol >= 4:
        vol_score = 0
    else:
        vol_score = 1 - ((current_vol - 1.5) / (4 - 1.5))

    abs_ret = abs(exp_ret)

    if abs_ret < 0.005:
        confidence_score = abs_ret / 0.005
    else:
        confidence_score = min(1, 0.5 + abs_ret * 20)

    if exp_ret < 0:
        confidence_score = -confidence_score

    composite = (
        0.40 * sharpe_score +
        0.25 * momentum_score +
        0.20 * vol_score +
        0.15 * confidence_score
    )

    return round(composite, 5)


# ---------------- HOME PAGE ---------------- #

@app.route("/")
def index():

    stocks = sorted(lstm_data["Stock"].unique().tolist()) if not lstm_data.empty else []

    return render_template("index.html", stocks=stocks)


# ---------------- DASHBOARD API ---------------- #

@app.route("/api/dashboard_data")
def get_dashboard_data():

    selected_stock = request.args.get('stock')

    rf_daily = 0.07 / 252

    raw_rankings = []

    for _, row in lstm_data.iterrows():

        ticker = str(row.get("Stock", "Unknown"))

        exp_ret = float(row.get("Expected_Return", 0))
        curr_price = float(row.get("Current_Price", 0))
        pred_price = float(row.get("Predicted_Price", curr_price))

        target_price = curr_price * (1 + exp_ret)

        # -------- GARCH VOLATILITY -------- #

        garch_raw = garch_variances.get(ticker)

        if garch_raw is None or len(garch_raw) == 0:
            vol_path = [1.5]
        else:
            vol_path = np.sqrt(np.maximum(np.array(garch_raw).flatten(), 1e-8)) * 100
            vol_path = vol_path.tolist()

        current_vol = round(vol_path[-1], 3)

        current_vol = max(0.3, min(current_vol, 12))

        display_history = vol_path[-63:] if len(vol_path) > 1 else [current_vol] * 30

        # -------- ALPHA SCORE -------- #

        alpha = compute_alpha_score(exp_ret, current_vol, curr_price, pred_price)

        excess_return = exp_ret - rf_daily
        vol_decimal = max(current_vol / 100, 1e-4)

        sharpe = round(excess_return / vol_decimal, 3)

        # -------- SIGNAL -------- #

        if exp_ret > 0.02 and current_vol < 2.5:
            signal = "STRONG BUY"
        elif exp_ret > 0.01:
            signal = "BUY"
        elif exp_ret < -0.01:
            signal = "SELL"
        elif exp_ret < 0:
            signal = "WEAK SELL"
        else:
            signal = "HOLD"

        raw_rankings.append({
            "Stock": ticker,
            "Price": round(curr_price, 2),
            "Target": round(target_price, 2),
            "Predicted": round(pred_price, 2),
            "Ret": round(exp_ret * 100, 3),
            "Vol": round(current_vol, 3),
            "Alpha": alpha,
            "Sharpe": sharpe,
            "Signal": signal,
            "Vol_History": display_history
        })

    # -------- NORMALISE ALPHA SCORE -------- #

    alphas = [r["Alpha"] for r in raw_rankings]

    a_min = np.min(alphas)
    a_max = np.max(alphas)

    a_range = max(a_max - a_min, 1e-6)

    for item in raw_rankings:
        item["Score"] = round(((item["Alpha"] - a_min) / a_range) * 100, 1)

    raw_rankings.sort(key=lambda x: x["Score"], reverse=True)

    for i, item in enumerate(raw_rankings):
        item["Rank"] = i + 1

    target = next((i for i in raw_rankings if i["Stock"] == selected_stock), raw_rankings[0])

    # -------- OHLC SIMULATION -------- #

    seed = int(abs(hash(target["Stock"])) % 99999)

    rng = np.random.default_rng(seed)

    current_price = target["Price"]
    predicted_price = target["Predicted"]

    daily_vol = np.clip(target["Vol"] / 100, 0.003, 0.08)

    DAYS = 30

    prices = [current_price]

    for _ in range(DAYS - 1):

        move = rng.normal(0, daily_vol)

        prices.insert(0, round(prices[0] / (1 + move), 2))

    trading_days = []

    d = datetime.now() - timedelta(days=1)

    while len(trading_days) < DAYS:

        if d.weekday() < 5:
            trading_days.insert(0, d)

        d -= timedelta(days=1)

    ohlc_history = []

    for i, price in enumerate(prices):

        wick = abs(rng.normal(0, daily_vol * 0.6))
        spread = abs(rng.normal(0, daily_vol * 0.4))

        op = round(price * (1 + rng.uniform(-spread, spread)), 2)

        cl = price

        hi = round(max(op, cl) * (1 + wick), 2)
        lo = round(min(op, cl) * (1 - wick), 2)

        ohlc_history.append({
            "time": trading_days[i].strftime("%Y-%m-%d"),
            "open": op,
            "high": hi,
            "low": lo,
            "close": cl
        })

    next_day = datetime.now()

    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    predicted_candle = {
        "time": next_day.strftime("%Y-%m-%d"),
        "open": current_price,
        "high": round(max(current_price, predicted_price) * 1.004, 2),
        "low": round(min(current_price, predicted_price) * 0.996, 2),
        "close": predicted_price
    }

    return jsonify({

        "rankings": raw_rankings,
        "selected": target,
        "ohlc": ohlc_history,
        "predicted_candle": predicted_candle,

        "market_stats": {
            "avg_ret": round(float(np.mean([r["Ret"] for r in raw_rankings])), 2),
            "avg_vol": round(float(np.mean([r["Vol"] for r in raw_rankings])), 2),
            "total": len(raw_rankings),
            "bullish": sum(1 for r in raw_rankings if r["Ret"] > 0)
        }
    })


# ---------------- RUN SERVER ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
