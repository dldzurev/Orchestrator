#!/usr/bin/env python3
"""
Bitcoin Price Tracker using Finnhub API
Fetches and displays crypto prices via Finnhub WebSocket + FastAPI.
"""

import finnhub
import time
from datetime import datetime
import json
import ssl
import random
from typing import Optional, Tuple, List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread, Lock
from collections import deque

try:
    # websocket-client library
    from websocket import WebSocketApp
except Exception:  # pragma: no cover
    WebSocketApp = None  # type: ignore

# Import indicators
from indicators import indicator_calc

# -----------------------
# Configuration / Globals
# -----------------------

# NOTE: per your request, the API key remains hard-coded here.
API_KEY = "d32dvl1r01qn0gi3ief0d32dvl1r01qn0gi3iefg"
finnhub_client = finnhub.Client(api_key=API_KEY)

# Supported crypto tickers for Finnhub API
CRYPTO_TICKERS = {
    "Bitcoin": "BINANCE:BTCUSDT",
    "Ethereum": "BINANCE:ETHUSDT",
    "BNB": "BINANCE:BNBUSDT",
    "XRP": "BINANCE:XRPUSDT",
    "Cardano": "BINANCE:ADAUSDT",
    "Solana": "BINANCE:SOLUSDT",
    "Dogecoin": "BINANCE:DOGEUSDT",
    "Polygon": "BINANCE:MATICUSDT",
    "Litecoin": "BINANCE:LTCUSDT",
    "Avalanche": "BINANCE:AVAXUSDT",
}

# Current selection
current_ticker = "BINANCE:BTCUSDT"
current_crypto_name = "Bitcoin"

# Global variable to store current price data
current_price = {
    "price": None,
    "timestamp": None,
    "delta": None,
    "percent_change": None,
    "crypto": "Bitcoin",
}

# Ring buffer of last printed log lines (exactly what is printed to terminal)
log_lines = deque(maxlen=500)
log_lock = Lock()

# WebSocket state for (un)subscribe on-the-fly
ws_app = None
ws_lock = Lock()
current_subscribed_symbol = None  # what the WS is currently subscribed to


def log_line(text: str) -> None:
    """Print to terminal and store in memory for the frontend."""
    print(text)
    with log_lock:
        log_lines.append(text)


# -----------
# FastAPI app
# -----------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_crypto_price(ticker: str) -> Optional[float]:
    """
    Fetch the current crypto price using Finnhub's crypto quote endpoint.
    Returns the current price as float or None on error.
    """
    try:
        quote = finnhub_client.crypto_quote(ticker)
        return quote.get("c")
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{timestamp}] Error fetching {ticker} price: {e}")
        return None


def _generate_sample_candles(
    *,
    minutes: int,
    resolution: str,
    seed: int = 42,
    base_price_hint: Optional[float] = None,
    crypto_name: str = "Bitcoin",
) -> Dict:
    """
    Generate realistic-looking OHLC candles with a mild trend and bounded intra-candle ranges.
    Returns {points: [{t ISO, p}], candles: [{t sec, o,h,l,c}], interval, resolution, num_candles, source}
    """
    # Base prices used if we don't have a current_price yet
    base_prices = {
        "Bitcoin": 115700.0,
        "Ethereum": 4200.0,
        "BNB": 600.0,
        "XRP": 1.20,
        "Cardano": 2.50,
        "Solana": 180.0,
        "Dogecoin": 0.35,
        "Polygon": 2.10,
        "Litecoin": 180.0,
        "Avalanche": 85.0,
    }
    current_crypto_price = base_price_hint or base_prices.get(crypto_name, 100.0)
    if current_price.get("price"):
        try:
            current_crypto_price = float(current_price["price"])
        except Exception:
            pass

    # Map resolution to seconds and default vol
    if resolution == "1":
        candle_interval = 60
        volatility = 0.002  # 0.2% per minute
    elif resolution == "5":
        candle_interval = 300
        volatility = 0.005
    elif resolution == "30":
        candle_interval = 1800
        volatility = 0.015
    elif resolution == "60":
        candle_interval = 3600
        volatility = 0.025
    elif resolution == "D":
        candle_interval = 86400
        volatility = 0.08
    else:
        candle_interval = 60
        volatility = 0.002

    total_seconds = max(1, minutes) * 60
    num_candles = max(1, total_seconds // candle_interval)

    # Cap lengths by timeframe
    if resolution == "D":
        num_candles = min(num_candles, 730)  # ~2y
    elif resolution == "60":
        num_candles = min(num_candles, 4320)  # ~6mo hourly
    elif resolution == "30":
        num_candles = min(num_candles, 2880)  # ~2mo 30min
    else:
        num_candles = min(num_candles, 500)  # keep UI snappy

    random.seed(seed)
    now_ts = int(time.time())
    base_price = current_crypto_price * (0.95 if resolution == "D" else 0.995)

    points = []
    candles = []
    for i in range(num_candles):
        candle_ts = now_ts - ((num_candles - i) * candle_interval)
        trend = random.uniform(-0.001, 0.002)  # slight upward bias
        change = random.uniform(-volatility, volatility) + trend
        base_price *= (1 + change)

        price_range = base_price * (volatility * 0.5)
        o = base_price + random.uniform(-price_range, price_range)
        c = base_price + random.uniform(-price_range, price_range)
        h = max(o, c) + random.uniform(0, price_range)
        l = min(o, c) - random.uniform(0, price_range)

        iso = datetime.fromtimestamp(candle_ts).strftime("%Y-%m-%dT%H:%M:%S")
        points.append({"t": iso, "p": c})
        candles.append({"t": candle_ts, "o": o, "h": h, "l": l, "c": c})

    return {
        "points": points,
        "candles": candles,
        "resolution": resolution,
        "interval": candle_interval,
        "num_candles": num_candles,
        "source": "sample",
    }


def _ws_resubscribe(old_symbol: Optional[str], new_symbol: str) -> None:
    """
    Attempt to (un)subscribe the active Finnhub WS to a new symbol without restart.
    Safe if ws isn't connected yet (no-op).
    """
    global ws_app, current_subscribed_symbol
    with ws_lock:
        if ws_app is None:
            # No active ws yet; on next open we'll subscribe to current_ticker
            return
        try:
            if old_symbol and old_symbol != new_symbol:
                ws_app.send(json.dumps({"type": "unsubscribe", "symbol": old_symbol}))
            ws_app.send(json.dumps({"type": "subscribe", "symbol": new_symbol}))
            current_subscribed_symbol = new_symbol
            log_line(f"WS resubscribed: {old_symbol} → {new_symbol}")
        except Exception as e:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line(f"[{ts}] WS resubscribe error: {e}")


# -----------------
# FastAPI endpoints
# -----------------

@app.get("/api/price")
async def get_price():
    """Return current crypto price data."""
    return JSONResponse(content=current_price)


@app.get("/api/cryptos")
async def get_cryptos():
    """Return list of supported cryptocurrencies."""
    return JSONResponse(
        content={"cryptos": list(CRYPTO_TICKERS.keys()), "current": current_crypto_name}
    )


@app.post("/api/change-crypto")
async def change_crypto(request: Request):
    """Change the current cryptocurrency and update WS subscription live if connected."""
    global current_ticker, current_crypto_name, current_subscribed_symbol

    crypto_data = await request.json()
    crypto_name = crypto_data.get("crypto")
    if crypto_name not in CRYPTO_TICKERS:
        return JSONResponse(content={"success": False, "error": "Unsupported crypto"})

    old_ticker = current_ticker
    current_ticker = CRYPTO_TICKERS[crypto_name]
    current_crypto_name = crypto_name

    # Reset price data for new crypto
    current_price.update(
        {
            "price": None,
            "timestamp": None,
            "delta": None,
            "percent_change": None,
            "crypto": crypto_name,
        }
    )

    # Live resubscribe if possible
    _ws_resubscribe(old_ticker, current_ticker)

    return JSONResponse(
        content={"success": True, "crypto": crypto_name, "ticker": current_ticker}
    )


@app.get("/api/logs")
async def get_logs(limit: int = 200):
    """Return the most recent printed lines, newest last (like a terminal)."""
    capped = max(1, min(limit, 500))
    with log_lock:
        lines = list(log_lines)[-capped:]
    return JSONResponse(content={"lines": lines})


@app.get("/api/indicators")
async def get_indicators():
    """Return current indicator data for charting (server-side calc)."""
    try:
        indicator_data = indicator_calc.get_indicator_data_for_chart()
        # Return empty shape if no data yet
        if not indicator_data:
            return JSONResponse(
                content={
                    "ema_12": {"timestamps": [], "values": []},
                    "ema_26": {"timestamps": [], "values": []},
                    "bb_upper": {"timestamps": [], "values": []},
                    "bb_middle": {"timestamps": [], "values": []},
                    "bb_lower": {"timestamps": [], "values": []},
                    "rsi": {"timestamps": [], "values": []},
                    "realized_vol": {"timestamps": [], "values": []},
                    "vpin": 0,
                    "hawkes_intensity": 0,
                    "last_price": 0,
                }
            )
        return JSONResponse(content=indicator_data)
    except Exception as e:
        log_line(f"Indicator error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/test-candles")
async def test_candles():
    """Test endpoint to verify sample data generation works."""
    data = _generate_sample_candles(minutes=10, resolution="1", seed=42, crypto_name=current_crypto_name)
    return JSONResponse(content=data)


@app.get("/api/history")
async def get_history(minutes: int = 60, resolution: str = "1"):
    """
    Return recent OHLC candles. Currently uses a deterministic sample generator.
    Response: { points: [{ t: ISO8601, p: float }], candles: [{ t: unix_sec, o,h,l,c }], interval, resolution, num_candles }
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line(f"[{ts}] get_history called: minutes={minutes}, resolution={resolution}")

    # For now: skip CoinGecko and use local synthetic generator (good for dev)
    try:
        minutes = max(1, int(minutes))
    except Exception:
        minutes = 60

    try:
        data = _generate_sample_candles(
            minutes=minutes,
            resolution=resolution,
            seed=42,
            base_price_hint=None,
            crypto_name=current_crypto_name,
        )
        return JSONResponse(content=data)
    except Exception as e:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{ts}] History fetch error: {e}; falling back to short sample")

        # Fallback: short 60-candle, 1-min set
        data = _generate_sample_candles(
            minutes=min(minutes, 60),
            resolution="1",
            seed=43,
            crypto_name=current_crypto_name,
        )
        data["source"] = "fallback"
        return JSONResponse(content=data)

@app.post("/api/backtest")
async def backtest_pl(request: Request):
    """
    Compute simple P/L% for one strategy by fetching historical candles from Finnhub.
    Assumptions:
      - Use first coin only
      - Indicator condition means: indicator >= value
      - Buy at first bar meeting all buy conditions; sell at first later bar meeting all sell conditions
      - 60-minute candles
    Body:
      {
        "coin": "Bitcoin",        # or "coins": ["Bitcoin", ...]
        "buy":  [ { "key":"price","value":50000 }, ... ],
        "sell": [ { "key":"price","value":52000 }, ... ],
        "start": "2025-09-10T12:00:00Z",
        "end":   "2025-09-12T12:00:00Z"
      }
    Resp: { "pl_pct": 3.21 } or { "pl_pct": None, "reason": "..." }
    """
    import math
    from datetime import datetime as _dt
    import numpy as _np
    import pandas as _pd

    body = await request.json()
    coin = body.get("coin") or (body.get("coins") or [None])[0]
    buy_conds = body.get("buy", []) or []
    sell_conds = body.get("sell", []) or []
    start_iso = body.get("start")
    end_iso = body.get("end")

    if not coin:
        return JSONResponse({"pl_pct": None, "reason": "no coin"}, status_code=400)
    symbol = CRYPTO_TICKERS.get(coin)
    if not symbol:
        return JSONResponse({"pl_pct": None, "reason": f"unsupported coin {coin}"}, status_code=400)
    if not start_iso or not end_iso:
        return JSONResponse({"pl_pct": None, "reason": "missing start/end"}, status_code=400)

    # Parse ISO to unix seconds
    def _parse_iso(s: str) -> int:
        s = s.replace("Z", "+00:00")
        dt = _dt.fromisoformat(s)
        return int(dt.timestamp())

    start_ts = _parse_iso(start_iso)
    end_ts = _parse_iso(end_iso)
    if end_ts <= start_ts:
        return JSONResponse({"pl_pct": None, "reason": "end<=start"}, status_code=400)

    # Fixed 60-min resolution for simplicity
    resolution = "60"
    try:
        resp = finnhub_client.crypto_candles(symbol, resolution, start_ts, end_ts)
    except Exception as e:
        ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{ts}] backtest candles error: {e}")
        return JSONResponse({"pl_pct": None, "reason": "finnhub error"}, status_code=500)

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        return JSONResponse({"pl_pct": None, "reason": "no_data"}, status_code=200)

    # Build dataframe
    t = resp.get("t", [])
    o = resp.get("o", [])
    h = resp.get("h", [])
    l = resp.get("l", [])
    c = resp.get("c", [])
    if not t or not c:
        return JSONResponse({"pl_pct": None, "reason": "empty"}, status_code=200)

    df = _pd.DataFrame({
        "time": _pd.to_datetime(_pd.Series(t, dtype="int64"), unit="s", utc=True),
        "open": _pd.Series(o, dtype="float"),
        "high": _pd.Series(h, dtype="float"),
        "low":  _pd.Series(l, dtype="float"),
        "close":_pd.Series(c, dtype="float"),
    }).dropna()
    if df.empty:
        return JSONResponse({"pl_pct": None, "reason": "empty"}, status_code=200)
    df.set_index("time", inplace=True)

    prices = df["close"].astype(float)

    # Indicators (match frontend math)
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    sma20 = prices.rolling(window=20).mean()
    std20 = prices.rolling(window=20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    # z-score of log returns (20)
    logret = _np.log(prices / prices.shift(1))
    zr_mean = logret.rolling(20).mean()
    zr_std  = logret.rolling(20).std().replace(0.0, _np.nan)
    zscore  = (logret - zr_mean) / zr_std

    # realized vol annualized (20) using fixed 60-min cadence
    seconds_per_year = 365 * 24 * 3600
    periods_per_year = seconds_per_year / (60 * 60)
    rv = logret.rolling(20).std() * math.sqrt(periods_per_year)

    series_map = {
        "price": prices,
        "ema12": ema12,
        "ema26": ema26,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "zscore": zscore,
        "rv": rv,
    }

    def _meets(row_idx, conds):
        # all indicators >= value at this index
        for cond in conds:
            key = cond.get("key")
            thr = cond.get("value")
            if key not in series_map or thr is None:
                return False
            val = series_map[key].iloc[row_idx]
            if _pd.isna(val) or not _np.isfinite(val):
                return False
            if float(val) < float(thr):
                return False
        return True

    buy_idx = None
    for i in range(len(df)):
        if _meets(i, buy_conds):
            buy_idx = i
            break

    if buy_idx is None:
        return JSONResponse({"pl_pct": None, "reason": "no_buy_trigger"}, status_code=200)

    sell_idx = None
    for j in range(buy_idx + 1, len(df)):
        if _meets(j, sell_conds):
            sell_idx = j
            break

    if sell_idx is None:
        return JSONResponse({"pl_pct": None, "reason": "no_sell_trigger"}, status_code=200)

    buy_price = float(prices.iloc[buy_idx])
    sell_price = float(prices.iloc[sell_idx])
    if buy_price <= 0:
        return JSONResponse({"pl_pct": None, "reason": "bad_buy_price"}, status_code=200)

    pl_pct = (sell_price - buy_price) / buy_price * 100.0
    return JSONResponse({"pl_pct": pl_pct})

# -----------------------------
# WebSocket streaming to Finnhub
# -----------------------------

def stream_bitcoin_prices() -> None:
    """
    Stream real-time trade prices using Finnhub WebSocket.
    Updates global price data for the web interface and feeds indicators.
    """
    if WebSocketApp is None:
        log_line("websocket-client is not installed. Install it with: pip3 install websocket-client")
        return

    url = f"wss://ws.finnhub.io?token={API_KEY}"
    last_price: Optional[float] = None

    def on_open(ws):  # type: ignore
        """Subscribe to the current symbol on open."""
        global ws_app, current_subscribed_symbol
        with ws_lock:
            ws_app = ws
        try:
            subscribe_msg = {"type": "subscribe", "symbol": current_ticker}
            ws.send(json.dumps(subscribe_msg))
            current_subscribed_symbol = current_ticker
            log_line(f"Subscribed to {current_crypto_name} trade stream on Finnhub")
        except Exception as e:
            log_line(f"WS on_open subscribe error: {e}")

    def on_message(ws, message):  # type: ignore
        nonlocal last_price
        global current_price
        try:
            payload = json.loads(message)
        except Exception:
            return

        if payload.get("type") != "trade":
            return

        data = payload.get("data", [])
        now_ts = time.time()
        for trade in data:
            price = trade.get("p")
            ts_ms = trade.get("t")
            size = trade.get("v", 1)  # Default size if not provided
            if not isinstance(price, (int, float)):
                continue

            ts_str = datetime.fromtimestamp((ts_ms or int(now_ts * 1000)) / 1000.0).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            delta = None
            percent_change = None
            if isinstance(last_price, (int, float)) and last_price > 0:
                delta = price - last_price
                percent_change = (delta / last_price) * 100.0
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
                log_line(f"[{ts_str}] {current_crypto_name} ${price:.2f}  {arrow} {delta:+.2f} ({percent_change:+.2f}%)")
            else:
                log_line(f"[{ts_str}] {current_crypto_name} ${price:.2f}")

            # Update global price data
            current_price.update(
                {
                    "price": price,
                    "timestamp": ts_str,
                    "delta": delta,
                    "percent_change": percent_change,
                    "crypto": current_crypto_name,
                }
            )

            # Update indicators with trade data
            try:
                indicator_calc.add_trade(
                    price=price,
                    size=size,
                    timestamp=ts_ms or int(now_ts * 1000),
                    symbol=current_crypto_name,
                )
            except Exception as e:
                log_line(f"[{ts_str}] Indicator update error: {e}")

            last_price = price

    def on_error(ws, error):  # type: ignore
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{ts}] WebSocket error for {current_crypto_name}: {error}")

    def on_close(ws, status_code, msg):  # type: ignore
        global ws_app, current_subscribed_symbol
        with ws_lock:
            ws_app = None
            current_subscribed_symbol = None
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{ts}] WebSocket closed for {current_crypto_name} (code={status_code}, msg={msg})")

    backoff_seconds = 1
    while True:
        try:
            _app = WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            # run_forever handles ping/pong and reconnects on transient network issues.
            _app.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED})
        except KeyboardInterrupt:
            raise
        except Exception as e:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line(
                f"[{ts}] WebSocket exception for {current_crypto_name}: {e}. Reconnecting in {backoff_seconds}s..."
            )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 60)
            continue
        # If clean close, wait briefly and reconnect
        time.sleep(min(backoff_seconds, 5))


def main():
    """
    Start price streaming and FastAPI server.
    """
    print("Bitcoin Price Tracker - Starting...")
    # Start price streaming in background thread
    streaming_thread = Thread(target=stream_bitcoin_prices, daemon=True)
    streaming_thread.start()

    print("FastAPI server starting on http://localhost:8000")
    print("API endpoint: http://localhost:8000/api/price")

    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
