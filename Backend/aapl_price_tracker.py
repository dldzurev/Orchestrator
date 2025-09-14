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

# --- ensure streaming thread starts whether run via `python file.py` or `uvicorn module:app` ---
streaming_started = False
def _start_streaming_thread_once():
    """Idempotently start the WebSocket streaming thread."""
    global streaming_started
    if streaming_started:
        return
    t = Thread(target=stream_bitcoin_prices, daemon=True)
    t.start()
    streaming_started = True

@app.on_event("startup")
async def _on_startup():
    _start_streaming_thread_once()


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
    Backtest P/L% with trade path details.
    - Uses Finnhub candles (multi-res) or synthetic fallback for the same window.
    - Normalizes keys (price/ema/bb/zscore/rv).
    - Infers SELL direction (target vs stop).
    - Treats small numbers as percent for price-like keys when price scale is large.
    - Returns entry/exit snapshots so the UI can show what actually triggered.
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
    end_iso   = body.get("end")

    if not coin:
        return JSONResponse({"pl_pct": None, "reason": "no_coin"}, status_code=400)
    symbol = CRYPTO_TICKERS.get(coin)
    if not symbol:
        return JSONResponse({"pl_pct": None, "reason": f"unsupported_coin:{coin}"}, status_code=400)
    if not start_iso or not end_iso:
        return JSONResponse({"pl_pct": None, "reason": "missing_start_end"}, status_code=400)
    if len(buy_conds) == 0:
        return JSONResponse({"pl_pct": None, "reason": "no_buy_conditions"}, status_code=200)
    if len(sell_conds) == 0:
        return JSONResponse({"pl_pct": None, "reason": "no_sell_conditions"}, status_code=200)

    def _parse_iso(s: str) -> int:
        s = s.replace("Z", "+00:00")
        dt = _dt.fromisoformat(s)
        return int(dt.timestamp())

    start_ts = _parse_iso(start_iso)
    end_ts   = _parse_iso(end_iso)
    if end_ts <= start_ts:
        return JSONResponse({"pl_pct": None, "reason": "end<=start"}, status_code=400)

    # ---- fetch candles (fallback to synthetic) ----
    _RES_ORDER = [("60", 3600), ("30", 1800), ("5", 300), ("1", 60)]
    resp, used_sec = None, None
    for res, sec in _RES_ORDER:
        try:
            r = finnhub_client.crypto_candles(symbol, res, start_ts, end_ts)
            if isinstance(r, dict) and r.get("s") == "ok" and r.get("t"):
                resp, used_sec = r, sec
                break
        except Exception as e:
            log_line(f"[{_dt.now():%Y-%m-%d %H:%M:%S}] backtest candles error (res={res}): {e}")

    source_used = "finnhub"
    if resp is None:
        source_used = "sample"
        minutes = max(1, int((end_ts - start_ts) // 60))
        if minutes > 24*60:
            res_choice, used_sec = "60", 3600
        elif minutes > 6*60:
            res_choice, used_sec = "30", 1800
        elif minutes > 60:
            res_choice, used_sec = "5", 300
        else:
            res_choice, used_sec = "1", 60
        sample = _generate_sample_candles(
            minutes=minutes, resolution=res_choice, seed=(start_ts % 100_000), crypto_name=coin
        )
        resp = {
            "s": "ok",
            "t": [c["t"] for c in sample["candles"]],
            "o": [c["o"] for c in sample["candles"]],
            "h": [c["h"] for c in sample["candles"]],
            "l": [c["l"] for c in sample["candles"]],
            "c": [c["c"] for c in sample["candles"]],
        }

    t, o, h, l, c = resp.get("t", []), resp.get("o", []), resp.get("h", []), resp.get("l", []), resp.get("c", [])
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

    # ---- indicators (match frontend) ----
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    sma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    bb_upper = sma20 + 2*std20
    bb_lower = sma20 - 2*std20
    logret  = _np.log(prices / prices.shift(1))
    zr_mean = logret.rolling(20).mean()
    zr_std  = logret.rolling(20).std().replace(0.0, _np.nan)
    zscore  = (logret - zr_mean) / zr_std
    seconds_per_year = 365 * 24 * 3600
    cadence_sec = int(used_sec or 3600)
    periods_per_year = seconds_per_year / cadence_sec
    rv = logret.rolling(20).std() * math.sqrt(periods_per_year)

    def _canon(k: str):
        if not k: return None
        s = str(k).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        return {
            "price": "price", "close": "price",
            "ema12": "ema12", "ema012": "ema12",
            "ema26": "ema26", "ema026": "ema26",
            "bbupper": "bb_upper", "bbupperband": "bb_upper",
            "bblower": "bb_lower", "bblowerband": "bb_lower",
            "zscore": "zscore",
            "rv": "rv", "realizedvol": "rv", "realisedvol": "rv",
        }.get(s)

    series_map = {
        "price": prices,
        "ema12": ema12,
        "ema26": ema26,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "zscore": zscore,
        "rv": rv,
    }

    price_like = {"price", "ema12", "ema26", "bb_upper", "bb_lower"}
    median_price = float(prices.median()) if len(prices) else 0.0

    # Compare sell vs buy for direction inference
    buy_thr_map_raw = {}
    for cond in buy_conds:
        k = _canon(cond.get("key")); v = cond.get("value")
        if k is not None and v is not None:
            try: buy_thr_map_raw[k] = float(v)
            except Exception: pass

    def _sell_mode_for_key(key, sell_thr):
        try: st = float(sell_thr)
        except Exception: return "up"
        b = buy_thr_map_raw.get(key)
        if b is None: return "up"
        return "up" if st >= b else "down"

    def _norm_thr(key, thr, mode):
        if thr is None: return None
        try: x = float(thr)
        except Exception: return None
        if key in price_like and median_price >= 1000 and abs(x) <= 100:
            return median_price * (1.0 + (x/100.0)) if mode == "up" else median_price * (1.0 - abs(x)/100.0)
        return x

    def _get_val(key, idx):
        ser = series_map.get(key)
        if ser is None: return None
        v = ser.iloc[idx]
        if _pd.isna(v) or not _np.isfinite(v): return None
        return float(v)

    def _cross_up(i, key, thr):
        if i <= 0: return False
        vp, vn = _get_val(key, i-1), _get_val(key, i)
        return (vp is not None and vn is not None and vp < thr <= vn)

    def _cross_down(i, key, thr):
        if i <= 0: return False
        vp, vn = _get_val(key, i-1), _get_val(key, i)
        return (vp is not None and vn is not None and vp > thr >= vn)

    def _edge_ge(i, key, thr):
        if i == 0: return (_get_val(key, 0) or -_np.inf) >= thr
        return not ((_get_val(key, i-1) or -_np.inf) >= thr) and ((_get_val(key, i) or -_np.inf) >= thr)

    def _edge_le(i, key, thr):
        if i == 0: return (_get_val(key, 0) or _np.inf) <= thr
        return not ((_get_val(key, i-1) or _np.inf) <= thr) and ((_get_val(key, i) or _np.inf) <= thr)

    # --- scans (record methods & snapshots) ---
    def _buy_cross_ok(i):
        for cnd in buy_conds:
            key = _canon(cnd.get("key")); thr = _norm_thr(key, cnd.get("value"), "up")
            if key is None or thr is None or not _cross_up(i, key, thr): return False
        return True

    def _buy_edge_ok(i):
        for cnd in buy_conds:
            key = _canon(cnd.get("key")); thr = _norm_thr(key, cnd.get("value"), "up")
            if key is None or thr is None or not _edge_ge(i, key, thr): return False
        return True

    def _sell_cross_ok(i):
        for cnd in sell_conds:
            key = _canon(cnd.get("key")); raw = cnd.get("value")
            mode = _sell_mode_for_key(key, raw); thr = _norm_thr(key, raw, mode)
            ok = _cross_up(i, key, thr) if mode == "up" else _cross_down(i, key, thr)
            if key is None or thr is None or not ok: return False
        return True

    def _sell_edge_ok(i):
        for cnd in sell_conds:
            key = _canon(cnd.get("key")); raw = cnd.get("value")
            mode = _sell_mode_for_key(key, raw); thr = _norm_thr(key, raw, mode)
            ok = _edge_ge(i, key, thr) if mode == "up" else _edge_le(i, key, thr)
            if key is None or thr is None or not ok: return False
        return True

    # BUY
    buy_idx, buy_method = None, None
    for i in range(1, len(df)):
        if _buy_cross_ok(i): buy_idx, buy_method = i, "cross_up"; break
    if buy_idx is None:
        for i in range(len(df)):
            if _buy_edge_ok(i): buy_idx, buy_method = i, "edge_ge"; break
    if buy_idx is None:
        return JSONResponse({"pl_pct": None, "reason": "no_buy_trigger", "source": source_used}, status_code=200)

    # SELL
    sell_idx, sell_method = None, None
    for j in range(max(buy_idx+1, 1), len(df)):
        if _sell_cross_ok(j): sell_idx, sell_method = j, "cross"; break
    if sell_idx is None:
        for j in range(buy_idx+1, len(df)):
            if _sell_edge_ok(j): sell_idx, sell_method = j, "edge"; break
    if sell_idx is None:
        return JSONResponse({"pl_pct": None, "reason": "no_sell_trigger", "source": source_used}, status_code=200)

    buy_price, sell_price = float(prices.iloc[buy_idx]), float(prices.iloc[sell_idx])
    if buy_price <= 0:
        return JSONResponse({"pl_pct": None, "reason": "bad_buy_price", "source": source_used}, status_code=200)

    # snapshots for UI
    def _snap(idx, cond, side, method):
        raw_key = cond.get("key"); key = _canon(raw_key); raw_thr = cond.get("value")
        mode = "up" if side == "buy" else _sell_mode_for_key(key, raw_thr)
        thr  = _norm_thr(key, raw_thr, mode)
        val_now  = _get_val(key, idx)
        val_prev = _get_val(key, idx-1) if idx > 0 else None
        return {
            "raw_key": raw_key,
            "key": key,
            "dir": mode if side == "sell" else "up",
            "method": method,
            "threshold": thr,
            "raw_threshold": raw_thr,
            "value_now": val_now,
            "value_prev": val_prev,
        }

    entry = {
        "index": int(buy_idx),
        "time": df.index[buy_idx].isoformat(),
        "price": buy_price,
        "conditions": [_snap(buy_idx, c, "buy", buy_method) for c in buy_conds],
    }
    exit_ = {
        "index": int(sell_idx),
        "time": df.index[sell_idx].isoformat(),
        "price": sell_price,
        "conditions": [_snap(sell_idx, c, "sell", sell_method) for c in sell_conds],
    }

    pl_pct = (sell_price - buy_price) / buy_price * 100.0
    out = {
        "pl_pct": pl_pct,
        "source": source_used,
        "resolution_sec": cadence_sec,
        "duration_sec": int((sell_idx - buy_idx) * cadence_sec),
        "entry": entry,
        "exit": exit_,
    }
    # Helpful line in /api/logs so you can verify server returned details
    log_line(f"Backtest {coin}: entry@{entry['time']} {entry['price']:.2f} -> exit@{exit_['time']} {exit_['price']:.2f} = {pl_pct:.2f}%")
    return JSONResponse(out)





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
    # Start price streaming (idempotent)
    _start_streaming_thread_once()

    print("FastAPI server starting on http://localhost:8000")
    print("API endpoint: http://localhost:8000/api/price")

    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
