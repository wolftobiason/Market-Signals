from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import math
import os

app = Flask(__name__)


def bs_d1_d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    phi = norm.pdf(d1)
    sqrt_T = math.sqrt(T)
    disc = math.exp(-r * T)

    gamma = phi / (S * sigma * sqrt_T)
    vega = S * phi * sqrt_T / 100  # per 1% vol move

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * phi * sigma / (2 * sqrt_T) - r * K * disc * norm.cdf(d2)) / 365
        rho = K * T * disc * norm.cdf(d2) / 100  # per 1% rate move
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * phi * sigma / (2 * sqrt_T) + r * K * disc * norm.cdf(-d2)) / 365
        rho = -K * T * disc * norm.cdf(-d2) / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "rho": round(rho, 4),
    }


def implied_vol(market_price, S, K, T, r, option_type="call"):
    if T <= 0:
        return None
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price <= intrinsic:
        return None
    try:
        iv = brentq(
            lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price,
            1e-6,
            10.0,
            xtol=1e-6,
            maxiter=500,
        )
        return round(iv * 100, 2)  # return as percentage
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/spx")
def get_spx():
    result = {}
    errors = []

    for symbol, key, fallback in [
        ("SPY",  "price", 530.00),  # SPY spot — used as underlying
        ("^IRX", "rate",  4.30),    # 13-week T-bill annualized yield, already in %
        ("^VIX", "vix",  15.00),    # 30-day ATM implied vol, already in %
    ]:
        try:
            result[key] = round(float(yf.Ticker(symbol).fast_info.last_price), 2)
        except Exception as e:
            result[key] = fallback
            errors.append(f"{symbol}: {e}")

    result["status"] = "fallback" if errors else "live"
    if errors:
        result["errors"] = errors
    return jsonify(result)


@app.route("/api/price", methods=["POST"])
def price_option():
    data = request.json
    try:
        S = float(data["spot"])
        K = float(data["strike"])
        r = float(data["rate"]) / 100
        sigma = float(data["vol"]) / 100
        expiry = datetime.strptime(data["expiry"], "%Y-%m-%d")
        T = max((expiry - datetime.today()).days / 365.25, 0)
        opt = data.get("type", "call")

        price = black_scholes(S, K, T, r, sigma, opt)
        greeks = compute_greeks(S, K, T, r, sigma, opt)
        iv = None
        if data.get("market_price"):
            iv = implied_vol(float(data["market_price"]), S, K, T, r, opt)

        moneyness = "ATM"
        pct = (S - K) / K * 100
        if opt == "call":
            moneyness = "ITM" if S > K else ("ATM" if S == K else "OTM")
        else:
            moneyness = "ITM" if S < K else ("ATM" if S == K else "OTM")

        return jsonify(
            {
                "price": round(price, 2),
                "greeks": greeks,
                "dte": round(T * 365.25),
                "implied_vol": iv,
                "moneyness": moneyness,
                "pct_from_spot": round(pct, 2),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
