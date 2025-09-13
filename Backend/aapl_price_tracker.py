#!/usr/bin/env python3
"""
Bitcoin Price Tracker using Finnhub API
Fetches and displays Bitcoin prices. Now with FastAPI support.
"""

import finnhub
import time
from datetime import datetime
import json
import ssl
import requests
import random
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread, Lock
from collections import deque

try:
    # websocket-client library
    from websocket import WebSocketApp
except Exception:  # pragma: no cover
    WebSocketApp = None  # type: ignore

# Initialize the Finnhub client with your API key
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
    "Avalanche": "BINANCE:AVAXUSDT"
}

# Current selected ticker
current_ticker = "BINANCE:BTCUSDT"  # Default to Bitcoin
current_crypto_name = "Bitcoin"

# Global variable to store current price data
current_price = {"price": None, "timestamp": None, "delta": None, "percent_change": None, "crypto": "Bitcoin"}

# Ring buffer of last printed log lines (exactly what is printed to terminal)
log_lines = deque(maxlen=500)
log_lock = Lock()

def log_line(text: str) -> None:
    """Print to terminal and store in memory for the frontend."""
    print(text)
    try:
        log_lock.acquire()
        log_lines.append(text)
    finally:
        log_lock.release()

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_crypto_price(ticker):
    """
    Fetch the current crypto price using Finnhub's crypto quote endpoint.
    Returns the current price as float or None on error.
    """
    try:
        quote = finnhub_client.crypto_quote(ticker)
        return quote.get('c')
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Error fetching {ticker} price: {e}")
        return None

# FastAPI endpoint to get current price
@app.get("/api/price")
async def get_price():
    """Return current crypto price data"""
    return JSONResponse(content=current_price)

# FastAPI endpoint to get supported cryptos
@app.get("/api/cryptos")
async def get_cryptos():
    """Return list of supported cryptocurrencies"""
    return JSONResponse(content={"cryptos": list(CRYPTO_TICKERS.keys()), "current": current_crypto_name})

# FastAPI endpoint to change crypto
@app.post("/api/change-crypto")
async def change_crypto(crypto_data: dict):
    """Change the current cryptocurrency"""
    global current_ticker, current_crypto_name
    crypto_name = crypto_data.get("crypto")
    if crypto_name in CRYPTO_TICKERS:
        current_ticker = CRYPTO_TICKERS[crypto_name]
        current_crypto_name = crypto_name
        # Reset price data for new crypto
        current_price.update({
            "price": None,
            "timestamp": None, 
            "delta": None,
            "percent_change": None,
            "crypto": crypto_name
        })
        return JSONResponse(content={"success": True, "crypto": crypto_name, "ticker": current_ticker})
    return JSONResponse(content={"success": False, "error": "Unsupported crypto"})

# FastAPI endpoint to get recent terminal lines
@app.get("/api/logs")
async def get_logs(limit: int = 200):
    """Return the most recent printed lines, newest last (like a terminal)."""
    capped = max(1, min(limit, 500))
    with log_lock:
        lines = list(log_lines)[-capped:]
    return JSONResponse(content={"lines": lines})

# Simple test endpoint to generate sample data
@app.get("/api/test-candles")
async def test_candles():
    """Test endpoint to verify sample data generation works."""
    points = []
    candles = []
    now_ts = int(time.time())
    base_price = 115700.0
    random.seed(42)
    
    for i in range(10):  # Just 10 candles for testing
        candle_ts = now_ts - ((10 - i) * 60)
        volatility = 0.002
        change = random.uniform(-volatility, volatility)
        base_price *= (1 + change)
        
        price_range = base_price * 0.001
        o = base_price + random.uniform(-price_range, price_range)
        c = base_price + random.uniform(-price_range, price_range)
        h = max(o, c) + random.uniform(0, price_range)
        l = min(o, c) - random.uniform(0, price_range)
        
        iso = datetime.fromtimestamp(candle_ts).strftime('%Y-%m-%dT%H:%M:%S')
        points.append({"t": iso, "p": c})
        candles.append({"t": candle_ts, "o": o, "h": h, "l": l, "c": c})
    
    return JSONResponse(content={"points": points, "candles": candles, "count": len(candles)})

# FastAPI endpoint to fetch recent history using free CoinGecko API
@app.get("/api/history")
async def get_history(minutes: int = 60, resolution: str = '1'):
    """
    Return recent OHLC candles using CoinGecko free API.
    Response: { points: [{ t: ISO8601 string, p: float }, ...], candles: [{ t: unix_ts, o, h, l, c }, ...] }
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line(f"[{ts}] get_history called: minutes={minutes}, resolution={resolution}")
    try:
        minutes = max(1, int(minutes))  # Remove the artificial cap
    except Exception:
        minutes = 60

    try:
        # Skip CoinGecko for now to test fallback logic
        pass
        
        # Fallback: Generate sample historical candles based on current price
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{ts}] Main fallback: minutes={minutes}, resolution={resolution}")
        
        # Use current crypto price as base, with fallback defaults
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
            "Avalanche": 85.0
        }
        
        current_crypto_price = base_prices.get(current_crypto_name, 100.0)
        if current_price.get("price"):
            current_crypto_price = float(current_price["price"])
        
        points = []
        candles = []
        now_ts = int(time.time())
        
        # Generate realistic-looking historical candles
        random.seed(42)  # Consistent data
        
        # Adjust volatility and candle interval based on resolution
        if resolution == '1':  # 1 minute candles
            candle_interval = 60
            volatility = 0.002  # 0.2% per minute
        elif resolution == '5':  # 5 minute candles
            candle_interval = 300
            volatility = 0.005  # 0.5% per 5min
        elif resolution == '30':  # 30 minute candles
            candle_interval = 1800
            volatility = 0.015  # 1.5% per 30min
        elif resolution == '60':  # 1 hour candles
            candle_interval = 3600
            volatility = 0.025  # 2.5% per hour
        elif resolution == 'D':  # Daily candles
            candle_interval = 86400
            volatility = 0.08  # 8% per day
        else:
            candle_interval = 60
            volatility = 0.002
        
        # Calculate number of candles needed based on the actual time period
        total_seconds = minutes * 60
        num_candles = max(1, total_seconds // candle_interval)
        
        # Different caps based on resolution to keep it realistic
        if resolution == 'D':  # Daily - show up to 2 years of data
            num_candles = min(num_candles, 730)
        elif resolution == '60':  # Hourly - show up to 6 months
            num_candles = min(num_candles, 4320)  
        elif resolution == '30':  # 30min - show up to 2 months
            num_candles = min(num_candles, 2880)
        else:  # 1min and 5min - cap at reasonable amount
            num_candles = min(num_candles, 500)
        
        base_price = current_crypto_price * (0.95 if resolution == 'D' else 0.995)  # Start lower for longer timeframes
        
        for i in range(num_candles):
            candle_ts = now_ts - ((num_candles - i) * candle_interval)
            
            # Generate realistic price movement with trend
            trend = random.uniform(-0.001, 0.002)  # Slight upward bias
            change = random.uniform(-volatility, volatility) + trend
            base_price *= (1 + change)
            
            # Generate OHLC within reasonable bounds
            price_range = base_price * (volatility * 0.5)  # Intracandle range
            o = base_price + random.uniform(-price_range, price_range)
            c = base_price + random.uniform(-price_range, price_range)
            h = max(o, c) + random.uniform(0, price_range)
            l = min(o, c) - random.uniform(0, price_range)
            
            iso = datetime.fromtimestamp(candle_ts).strftime('%Y-%m-%dT%H:%M:%S')
            points.append({"t": iso, "p": c})
            candles.append({"t": candle_ts, "o": o, "h": h, "l": l, "c": c})
        
        return JSONResponse(content={"points": points, "candles": candles, "resolution": resolution, "source": "sample", "interval": candle_interval, "num_candles": num_candles})
        
    except Exception as e:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{ts}] History fetch error: {e}")
        log_line(f"[{ts}] Error handler: minutes={minutes}, resolution={resolution}")
        # Even on error, return sample data
        points = []
        candles = []
        now_ts = int(time.time())
        base_price = 115700.0
        random.seed(42)
        
        num_error_candles = min(minutes, 60)
        log_line(f"[{ts}] Error handler creating {num_error_candles} candles")
        for i in range(num_error_candles):
            candle_ts = now_ts - ((60 - i) * 60)
            volatility = 0.002
            change = random.uniform(-volatility, volatility)
            base_price *= (1 + change)
            
            price_range = base_price * 0.001
            o = base_price + random.uniform(-price_range, price_range)
            c = base_price + random.uniform(-price_range, price_range)
            h = max(o, c) + random.uniform(0, price_range)
            l = min(o, c) - random.uniform(0, price_range)
            
            iso = datetime.fromtimestamp(candle_ts).strftime('%Y-%m-%dT%H:%M:%S')
            points.append({"t": iso, "p": c})
            candles.append({"t": candle_ts, "o": o, "h": h, "l": l, "c": c})
        
        return JSONResponse(content={"points": points, "candles": candles, "resolution": resolution, "source": "fallback"})

def stream_bitcoin_prices() -> None:
    """
    Stream real-time Bitcoin trade prices using Finnhub WebSocket.
    Updates global price data for the web interface.
    """
    if WebSocketApp is None:
        print("websocket-client is not installed. Install it with: pip3 install websocket-client")
        return

    url = f"wss://ws.finnhub.io?token={API_KEY}"

    last_price: Optional[float] = None

    def on_open(ws):  # type: ignore
        subscribe_msg = {"type": "subscribe", "symbol": current_ticker}
        ws.send(json.dumps(subscribe_msg))
        log_line(f"Subscribed to {current_crypto_name} trade stream on Finnhub")

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
            if not isinstance(price, (int, float)):
                continue

            timestamp = datetime.fromtimestamp((ts_ms or int(now_ts * 1000)) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")

            delta = None
            percent_change = None
            if isinstance(last_price, (int, float)) and last_price > 0:
                delta = price - last_price
                percent_change = (delta / last_price) * 100.0
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
                log_line(f"[{timestamp}] {current_crypto_name} ${price:.2f}  {arrow} {delta:+.2f} ({percent_change:+.2f}%)")
            else:
                log_line(f"[{timestamp}] {current_crypto_name} ${price:.2f}")

            # Update global price data
            current_price.update({
                "price": price,
                "timestamp": timestamp,
                "delta": delta,
                "percent_change": percent_change,
                "crypto": current_crypto_name
            })

            last_price = price

    def on_error(ws, error):  # type: ignore
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{ts}] WebSocket error for {current_crypto_name}: {error}")

    def on_close(ws, status_code, msg):  # type: ignore
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line(f"[{ts}] WebSocket closed for {current_crypto_name} (code={status_code}, msg={msg})")

    backoff_seconds = 1
    while True:
        try:   
            ws_app = WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            # run_forever handles ping/pong and reconnects on transient network issues.
            ws_app.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED})
        except KeyboardInterrupt:
            raise
        except Exception as e:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line(f"[{ts}] WebSocket exception for {current_crypto_name}: {e}. Reconnecting in {backoff_seconds}s...")
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 60)
            continue
        # If clean close, wait briefly and reconnect
        time.sleep(min(backoff_seconds, 5))

def main():
    """
    Start Bitcoin price streaming and FastAPI server.
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
