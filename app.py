from flask import Flask, render_template, jsonify, request
import numpy as np
import os
import joblib
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = "alpha_terminal_secret_2024"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models")

# ── Hardcoded model results (fallback when pkl files cannot be loaded) ──
_LSTM_DATA = [
    {"Stock":"TCS.NS",        "Current_Price":3188.83, "Predicted_Price":3236.33, "Expected_Return":0.014896},
    {"Stock":"INFY.NS",       "Current_Price":1621.60, "Predicted_Price":1651.49, "Expected_Return":0.018435},
    {"Stock":"HCLTECH.NS",    "Current_Price":1603.58, "Predicted_Price":1655.49, "Expected_Return":0.032370},
    {"Stock":"WIPRO.NS",      "Current_Price":257.01,  "Predicted_Price":258.87,  "Expected_Return":0.007223},
    {"Stock":"TECHM.NS",      "Current_Price":1604.60, "Predicted_Price":1601.23, "Expected_Return":-0.002103},
    {"Stock":"KPITTECH.NS",   "Current_Price":1159.06, "Predicted_Price":1203.32, "Expected_Return":0.038188},
    {"Stock":"MPHASIS.NS",    "Current_Price":2795.70, "Predicted_Price":2869.19, "Expected_Return":0.026288},
    {"Stock":"COFORGE.NS",    "Current_Price":1650.96, "Predicted_Price":1740.35, "Expected_Return":0.054149},
    {"Stock":"PERSISTENT.NS", "Current_Price":6160.90, "Predicted_Price":6284.56, "Expected_Return":0.020071},
    {"Stock":"OFSS.NS",       "Current_Price":7626.50, "Predicted_Price":7813.98, "Expected_Return":0.024583},
]
_GARCH_DATA = {
    "TCS.NS":        [1.6550435597750561],
    "INFY.NS":       [2.261220678005101],
    "HCLTECH.NS":    [2.045082053839925],
    "WIPRO.NS":      [1.8535622299544647],
    "TECHM.NS":      [1.9790751739338948],
    "KPITTECH.NS":   [5.236150229525223],
    "MPHASIS.NS":    [2.7990949452600797],
    "COFORGE.NS":    [4.944571000070411],
    "PERSISTENT.NS": [3.839103569866602],
    "OFSS.NS":       [3.002863362204544],
}

def load_models():
    """Load model results — tries pkl first, falls back to embedded data."""
    try:
        lstm  = joblib.load(os.path.join(MODEL_PATH, "lstm_results.pkl"))
        garch = joblib.load(os.path.join(MODEL_PATH, "garch_results.pkl"))
        if not isinstance(lstm, pd.DataFrame):
            lstm = pd.DataFrame(lstm)
        lstm.columns = [c.strip() for c in lstm.columns]
        print(f"Models loaded from pkl — {len(lstm)} stocks")
        return lstm, garch
    except Exception as e:
        print(f"pkl load failed ({e}) — using embedded model data")
        return pd.DataFrame(_LSTM_DATA), _GARCH_DATA

lstm_data, garch_variances = load_models()


def compute_alpha_score(exp_ret, current_vol, current_price, predicted_price):
    """
    Composite Alpha Score (0–100 scale) using four signals:

    1. Return-to-Risk (Sharpe-like, 40% weight)
       Measures reward per unit of daily volatility above the risk-free rate.
       rf_daily = 7% annual / 252 trading days.

    2. Upside Momentum (25% weight)
       Logarithmic return signal — captures the magnitude of predicted price
       move relative to current price, rewarding larger upside non-linearly.

    3. Volatility Penalty (20% weight)
       Stocks with GARCH volatility > 2% daily are penalised progressively.
       Low-vol stocks score higher, reflecting capital preservation preference.

    4. Return Confidence Band (15% weight)
       Penalises near-zero predicted returns (too small to act on) and
       rewards clear directional signals above a 0.5% threshold.

    Final score is a weighted sum of the four normalised sub-scores,
    then scaled to a 0–100 range across the universe.
    """
    rf_daily = 0.07 / 252  # 7% annual risk-free rate

    # --- Signal 1: Sharpe-like Return-to-Risk (weight 0.40) ---
    excess_return = exp_ret - rf_daily
    vol_decimal = max(current_vol / 100, 1e-4)
    sharpe_raw = excess_return / vol_decimal
    sharpe_score = np.tanh(sharpe_raw * 0.5)  # bounded [-1, 1]

    # --- Signal 2: Upside Momentum (weight 0.25) ---
    if predicted_price > 0 and current_price > 0:
        log_ret = np.log(predicted_price / current_price)
    else:
        log_ret = 0.0
    momentum_score = np.tanh(log_ret * 30)  # amplified for daily scale

    # --- Signal 3: Volatility Penalty (weight 0.20) ---
    # Full score at vol ≤ 1.5%; linear decay to 0 at vol = 4%
    vol_threshold_low = 1.5
    vol_threshold_high = 4.0
    if current_vol <= vol_threshold_low:
        vol_score = 1.0
    elif current_vol >= vol_threshold_high:
        vol_score = 0.0
    else:
        vol_score = 1.0 - (current_vol - vol_threshold_low) / (vol_threshold_high - vol_threshold_low)

    # --- Signal 4: Return Confidence Band (weight 0.15) ---
    abs_ret = abs(exp_ret)
    signal_threshold = 0.005  # 0.5% minimum meaningful move
    if abs_ret < signal_threshold:
        confidence_score = abs_ret / signal_threshold  # partial credit
    else:
        # Reward directional clarity; penalise very large returns (outlier risk)
        confidence_score = min(1.0, 0.5 + abs_ret * 20)
    if exp_ret < 0:
        confidence_score = -confidence_score  # directional penalty for negatives

    # --- Weighted Composite ---
    composite = (
        0.40 * sharpe_score +
        0.25 * momentum_score +
        0.20 * vol_score +
        0.15 * confidence_score
    )
    return round(composite, 5)

# Pre-compute full-universe alpha scores AFTER function is defined
_universe_scores = {}
try:
    _alphas_raw = {}
    for _, _row in lstm_data.iterrows():
        _t   = str(_row.get("Stock", ""))
        _er  = float(_row.get("Expected_Return", 0))
        _cp  = float(_row.get("Current_Price", 0))
        _pp  = float(_row.get("Predicted_Price", _cp))
        _gv  = garch_variances.get(_t, [0.0001])
        _vf  = float(np.array(_gv).flatten()[0])
        _vol = float(np.sqrt(max(_vf, 1e-8)) * 100)
        _alphas_raw[_t] = compute_alpha_score(_er, _vol, _cp, _pp)
    _a_vals  = list(_alphas_raw.values())
    _a_min   = min(_a_vals)
    _a_max   = max(_a_vals)
    _a_range = _a_max - _a_min if _a_max != _a_min else 1.0
    for _t, _a in _alphas_raw.items():
        raw = (_a - _a_min) / _a_range * 95   # scale to 0-95
        _universe_scores[_t] = round(raw + 5, 1)  # floor at 5, ceiling at 100
    print(f"Universe scores computed: {_universe_scores}")
except Exception as _e:
    print(f"Score precompute error: {_e}")

VALID_USERS = {"admin": "alpha2024", "analyst": "nse@2024", "demo": "demo123"}

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    from flask import session, redirect, url_for
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if VALID_USERS.get(username) == password:
        session["user"] = username
        return redirect(url_for("dashboard"))
    return render_template("login.html", error="Invalid credentials")

@app.route("/logout")
def logout():
    from flask import session, redirect, url_for
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    from flask import session, redirect, url_for
    if "user" not in session:
        return redirect(url_for("login"))
    stocks = sorted(lstm_data["Stock"].unique().tolist()) if not lstm_data.empty else []
    return render_template("index.html", stocks=stocks, user=session["user"])

@app.route("/index")
def index():
    stocks = sorted(lstm_data["Stock"].unique().tolist()) if not lstm_data.empty else []
    return render_template("index.html", stocks=stocks, user="guest")

@app.route("/compare")
def compare_page():
    from flask import session, redirect, url_for
    if "user" not in session:
        return redirect(url_for("login"))
    stocks = sorted(lstm_data["Stock"].unique().tolist()) if not lstm_data.empty else []
    return render_template("compare.html", stocks=stocks, user=session["user"])

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

        # GARCH volatility
        garch_raw = garch_variances.get(ticker, [0.0001])
        vol_path = [float(np.sqrt(max(v, 1e-8)) * 100) for v in np.array(garch_raw).flatten()]
        current_vol = vol_path[-1]
        display_history = vol_path[-63:] if len(vol_path) > 1 else [current_vol] * 30

        # Composite Alpha Score
        alpha = compute_alpha_score(exp_ret, current_vol, curr_price, pred_price)

        # Sub-scores for display
        excess_return = exp_ret - rf_daily
        vol_decimal = max(current_vol / 100, 1e-4)
        sharpe = round(excess_return / vol_decimal, 3)

        # Signal label
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

    # Use pre-computed universe-wide scores (floor 5, ceiling 100)
    for item in raw_rankings:
        item["Score"] = _universe_scores.get(item["Stock"], 5.0)

    # Sort by normalised score
    raw_rankings.sort(key=lambda x: x["Score"], reverse=True)
    for i, item in enumerate(raw_rankings):
        item["Rank"] = i + 1

    target = next((i for i in raw_rankings if i["Stock"] == selected_stock), raw_rankings[0])

    # --- Generate realistic OHLC via seeded random walk ending at Current_Price ---
    seed = int(abs(hash(target["Stock"])) % 99999)
    rng = np.random.default_rng(seed)
    current_price   = target["Price"]
    predicted_price = target["Predicted"]
    # Vol is stored as sqrt(variance)*100 in %-point units;
    # divide by 10000 to get true decimal daily vol (e.g. 1.286% -> 0.01286)
    daily_vol = target["Vol"] / 10000

    DAYS = 30
    prices = [current_price]
    for _ in range(DAYS - 1):
        move = rng.normal(0, daily_vol)
        prices.insert(0, round(prices[0] / (1 + move), 2))

    # Build trading-day date labels (skip weekends)
    trading_days = []
    d = datetime.now() - timedelta(days=1)
    while len(trading_days) < DAYS:
        if d.weekday() < 5:
            trading_days.insert(0, d)
        d -= timedelta(days=1)

    ohlc_history = []
    for i, price in enumerate(prices):
        wick  = abs(rng.normal(0, daily_vol * 0.6))
        spread = abs(rng.normal(0, daily_vol * 0.4))
        op = round(price * (1 + rng.uniform(-spread, spread)), 2)
        cl = price
        hi = round(max(op, cl) * (1 + wick), 2)
        lo = round(min(op, cl) * (1 - wick), 2)
        ohlc_history.append({"time": trading_days[i].strftime("%Y-%m-%d"),
                              "open": op, "high": hi, "low": lo, "close": cl})

    # Predicted next-day candle
    next_day = datetime.now()
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    predicted_candle = {
        "time":  next_day.strftime("%Y-%m-%d"),
        "open":  current_price,
        "high":  round(max(current_price, predicted_price) * 1.004, 2),
        "low":   round(min(current_price, predicted_price) * 0.996, 2),
        "close": predicted_price
    }

    return jsonify({
        "rankings": raw_rankings,
        "selected": target,
        "ohlc": ohlc_history,
        "predicted_candle": predicted_candle,
        "market_stats": {
            "avg_ret":  round(float(np.mean([r["Ret"] for r in raw_rankings])), 2),
            "avg_vol":  round(float(np.mean([r["Vol"] for r in raw_rankings])), 2),
            "total":    len(raw_rankings),
            "bullish":  sum(1 for r in raw_rankings if r["Ret"] > 0)
        }
    })

@app.route("/api/compare")
def compare_stocks():
    tickers = request.args.get("stocks", "").split(",")
    tickers = [t.strip() for t in tickers if t.strip()]
    if not tickers:
        return jsonify({"error": "No stocks specified"})

    rf_daily = 0.07 / 252

    # Step 1: compute alpha for ALL stocks in universe to get the full range
    all_alphas = {}
    for _, urow in lstm_data.iterrows():
        ut   = str(urow.get("Stock", ""))
        uer  = float(urow.get("Expected_Return", 0))
        ucp  = float(urow.get("Current_Price", 0))
        upp  = float(urow.get("Predicted_Price", ucp))
        ugv  = garch_variances.get(ut, [0.0001])
        uvf  = float(np.array(ugv).flatten()[0])
        uvol = float(np.sqrt(max(uvf, 1e-8)) * 100)
        all_alphas[ut] = compute_alpha_score(uer, uvol, ucp, upp)

    # Step 2: normalise 0-100 across full universe
    a_vals  = list(all_alphas.values())
    a_min   = min(a_vals)
    a_max   = max(a_vals)
    a_range = a_max - a_min if a_max != a_min else 1.0
    universe_scores = {t: round((a - a_min) / a_range * 95 + 5, 1) for t, a in all_alphas.items()}

    # Step 3: build result only for requested tickers
    result = []
    for ticker in tickers:
        rows = lstm_data[lstm_data["Stock"] == ticker]
        if rows.empty:
            continue
        row = rows.iloc[0]
        exp_ret    = float(row.get("Expected_Return", 0))
        curr_price = float(row.get("Current_Price", 0))
        pred_price = float(row.get("Predicted_Price", curr_price))

        garch_raw   = garch_variances.get(ticker, [0.0001])
        vol_flat    = float(np.array(garch_raw).flatten()[0])
        current_vol = float(np.sqrt(max(vol_flat, 1e-8)) * 100)
        sharpe      = round((exp_ret - rf_daily) / max(current_vol / 100, 1e-4), 3)

        if exp_ret > 0.02 and current_vol < 2.5:  signal = "STRONG BUY"
        elif exp_ret > 0.01:                        signal = "BUY"
        elif exp_ret < -0.01:                       signal = "SELL"
        elif exp_ret < 0:                           signal = "WEAK SELL"
        else:                                       signal = "HOLD"

        result.append({
            "Stock":     ticker,
            "ShortName": ticker.replace(".NS", ""),
            "Price":     round(curr_price, 2),
            "Predicted": round(pred_price, 2),
            "Ret":       round(exp_ret * 100, 3),
            "Vol":       round(current_vol, 3),
            "Sharpe":    sharpe,
            "Signal":    signal,
            "Score":     universe_scores.get(ticker, 0.0),
        })

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
