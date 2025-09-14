"""
Technical Indicators Module
Based on the reference code for real-time indicator calculations
"""

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from scipy.stats import zscore
import threading
import time

# Configuration
TRADE_BUFFER_SIZE = 10000
VPIN_BUCKET_VOLUME = 5000
VPIN_WINDOW_BUCKETS = 50
EMA_PENALTY = 0.9

# Hawkes-like params
HAWKES_ALPHA = 0.8
HAWKES_BETA = 1.2
HAWKES_MU = 0.1

class IndicatorCalculator:
    def __init__(self):
        # Data buffers
        self.trade_lock = threading.Lock()
        self.trades = deque(maxlen=TRADE_BUFFER_SIZE)
        self.quote_lock = threading.Lock()
        self.latest_quote = {}
        
        # VPIN state
        self.vpin_buckets = deque(maxlen=VPIN_WINDOW_BUCKETS)
        self.current_bucket_vol = 0
        self.current_bucket_signed = 0
        
        # Hawkes state
        self.last_event_ts = None
        self.hawkes_state = 0.0
        
        # Trade/quote counters
        self.trade_count_1s = 0
        self.quote_count_1s = 0
        self.trade_count_lock = threading.Lock()
        
        # Start counter reset thread
        self._start_counter_reset()
    
    def _start_counter_reset(self):
        """Reset trade/quote counters every second"""
        def reset_loop():
            while True:
                time.sleep(1.0)
                with self.trade_count_lock:
                    self.trade_count_1s = 0
                    self.quote_count_1s = 0
        
        t = threading.Thread(target=reset_loop, daemon=True)
        t.start()
    
    def add_trade(self, price, size, timestamp, symbol, sign=None):
        """Add a new trade to the buffer"""
        with self.trade_lock:
            # Determine trade sign if not provided
            if sign is None:
                if len(self.trades) > 0:
                    last_price = self.trades[-1]["price"]
                    if price > last_price:
                        sign = 1
                    elif price < last_price:
                        sign = -1
                    else:
                        sign = self.trades[-1].get("sign", 1)
                else:
                    sign = 1
            
            trade = {
                "price": price,
                "size": size,
                "ts": timestamp,
                "symbol": symbol,
                "sign": sign
            }
            self.trades.append(trade)
        
        # Update counters
        with self.trade_count_lock:
            self.trade_count_1s += 1
        
        # Update VPIN and Hawkes
        self._update_vpin(size, sign)
        self._update_hawkes(timestamp)
    
    def update_quote(self, bid, ask, bid_size=None, ask_size=None, timestamp=None):
        """Update the latest quote data"""
        with self.quote_lock:
            self.latest_quote.update({
                "bid": bid,
                "ask": ask,
                "bidSize": bid_size,
                "askSize": ask_size,
                "timestamp": timestamp
            })
        
        with self.trade_count_lock:
            self.quote_count_1s += 1
    
    def _update_vpin(self, size, sign):
        """Update VPIN buckets"""
        self.current_bucket_vol += size
        self.current_bucket_signed += sign * size
        
        if self.current_bucket_vol >= VPIN_BUCKET_VOLUME:
            self.vpin_buckets.append(self.current_bucket_signed)
            self.current_bucket_vol = 0
            self.current_bucket_signed = 0
    
    def _update_hawkes(self, ts_ms):
        """Update Hawkes intensity"""
        t = ts_ms / 1000.0
        if self.last_event_ts is None:
            self.hawkes_state = HAWKES_ALPHA
            self.last_event_ts = t
            return
        
        # Decay existing state
        dt = t - self.last_event_ts
        decay = np.exp(-HAWKES_BETA * dt)
        self.hawkes_state = self.hawkes_state * decay + HAWKES_ALPHA
        self.last_event_ts = t
    
    def compute_indicators(self):
        """Compute all indicators from current data"""
        with self.trade_lock:
            df = pd.DataFrame(list(self.trades))
        with self.quote_lock:
            q = dict(self.latest_quote)
        
        out = {}
        if df.empty:
            return out
        
        # Convert timestamps to datetime
        df = df.copy()
        df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('dt', inplace=True)
        
        prices = df['price']
        sizes = df['size']
        signs = df['sign']
        
        # 1) Moving Averages (EMA 12 and EMA 26)
        out['ema_12'] = prices.ewm(span=12, adjust=False).mean()
        out['ema_26'] = prices.ewm(span=26, adjust=False).mean()
        
        # 2) Bollinger Bands (20-period, 2 standard deviations)
        bb_period = 20
        bb_std = 2
        sma_20 = prices.rolling(window=bb_period).mean()
        std_20 = prices.rolling(window=bb_period).std()
        out['bb_upper'] = sma_20 + (std_20 * bb_std)
        out['bb_middle'] = sma_20
        out['bb_lower'] = sma_20 - (std_20 * bb_std)
        
        # 3) RSI (Relative Strength Index)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        out['rsi'] = calculate_rsi(prices, 14)
        
        # 4) Realized volatility
        ret = prices.pct_change().dropna()
        rv_window = 20
        out['realized_vol'] = ret.rolling(window=rv_window).std() * np.sqrt(252 * 6.5 * 3600)
        
        # 10) Last price
        out['last_price'] = prices.iloc[-1] if not prices.empty else np.nan
        
        return out
    
    def get_indicator_data_for_chart(self, max_points=1000):
        """Get indicator data formatted for charting"""
        indicators = self.compute_indicators()
        if not indicators:
            return {}
        
        chart_data = {}
        
        def clean_float(value):
            """Clean float values to be JSON compliant"""
            if pd.isna(value) or np.isinf(value) or np.isnan(value):
                return 0.0
            return float(value)
        
        # Convert pandas Series to lists for JSON serialization
        for key, value in indicators.items():
            if isinstance(value, pd.Series):
                if not value.empty:
                    # Get last max_points and clean values
                    recent_data = value.tail(max_points)
                    cleaned_values = [clean_float(v) for v in recent_data.tolist()]
                    chart_data[key] = {
                        'timestamps': [int(ts.timestamp() * 1000) for ts in recent_data.index],
                        'values': cleaned_values
                    }
                else:
                    chart_data[key] = {'timestamps': [], 'values': []}
            else:
                # Single values - clean them
                chart_data[key] = clean_float(value)
        
        return chart_data

# Global indicator calculator instance
indicator_calc = IndicatorCalculator()
