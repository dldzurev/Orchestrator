"""
Technical Indicators Module
Real-time indicator calculations with thread-safe trade/quote ingestion.
"""

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import threading

# Configuration
TRADE_BUFFER_SIZE = 10000
VPIN_BUCKET_VOLUME = 5000
VPIN_WINDOW_BUCKETS = 50

# Hawkes-like params
HAWKES_ALPHA = 0.8
HAWKES_BETA = 1.2


class IndicatorCalculator:
    def __init__(self):
        # Data buffers
        self.trade_lock = threading.Lock()
        self.trades = deque(maxlen=TRADE_BUFFER_SIZE)

        self.quote_lock = threading.Lock()
        self.latest_quote = {}

        # VPIN state
        self.vpin_buckets = deque(maxlen=VPIN_WINDOW_BUCKETS)
        self.current_bucket_vol = 0.0
        self.current_bucket_signed = 0.0

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
        """Reset trade/quote counters every second."""
        def reset_loop():
            import time
            while True:
                time.sleep(1.0)
                with self.trade_count_lock:
                    self.trade_count_1s = 0
                    self.quote_count_1s = 0

        t = threading.Thread(target=reset_loop, daemon=True)
        t.start()

    def add_trade(self, price, size, timestamp, symbol, sign=None):
        """Add a new trade to the buffer."""
        with self.trade_lock:
            # Determine trade sign if not provided (tick rule)
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
                "price": float(price),
                "size": float(size),
                "ts": int(timestamp),
                "symbol": symbol,
                "sign": int(sign),
            }
            self.trades.append(trade)

        # Update counters
        with self.trade_count_lock:
            self.trade_count_1s += 1

        # Update VPIN and Hawkes
        self._update_vpin(trade["size"], trade["sign"])
        self._update_hawkes(trade["ts"])

    def update_quote(self, bid, ask, bid_size=None, ask_size=None, timestamp=None):
        """Update the latest quote data."""
        with self.quote_lock:
            self.latest_quote.update(
                {
                    "bid": bid,
                    "ask": ask,
                    "bidSize": bid_size,
                    "askSize": ask_size,
                    "timestamp": timestamp,
                }
            )
        with self.trade_count_lock:
            self.quote_count_1s += 1

    def _update_vpin(self, size, sign):
        """Update VPIN buckets by aggregating to a fixed volume per bucket."""
        self.current_bucket_vol += float(size)
        self.current_bucket_signed += float(sign) * float(size)

        if self.current_bucket_vol >= VPIN_BUCKET_VOLUME:
            # Store signed imbalance of this bucket
            self.vpin_buckets.append(self.current_bucket_signed)
            self.current_bucket_vol = 0.0
            self.current_bucket_signed = 0.0

    def _update_hawkes(self, ts_ms):
        """Update simple Hawkes-like intensity with exponential decay."""
        import numpy as _np
        t = ts_ms / 1000.0
        if self.last_event_ts is None:
            self.hawkes_state = HAWKES_ALPHA
            self.last_event_ts = t
            return

        dt = max(0.0, t - self.last_event_ts)
        decay = _np.exp(-HAWKES_BETA * dt)
        self.hawkes_state = self.hawkes_state * decay + HAWKES_ALPHA
        self.last_event_ts = t

    def _estimate_periods_per_year(self, index: pd.DatetimeIndex, fallback_seconds: float = 60.0) -> float:
        """
        Estimate sampling frequency (periods per year) from timestamps.
        Uses mean dt (in seconds) over the most recent part of the series.
        """
        if len(index) < 2:
            return (365 * 24 * 3600) / fallback_seconds
        dt = pd.Series(index).diff().dt.total_seconds()
        # Focus on most recent portion to reflect current feed cadence
        recent_dt = dt.tail(min(2 * 20, max(2, len(dt) // 3)))  # ~last 40 intervals or last third
        mean_dt = float(recent_dt[recent_dt > 0].mean()) if (recent_dt > 0).any() else fallback_seconds
        mean_dt = mean_dt if (mean_dt and np.isfinite(mean_dt) and mean_dt > 0) else fallback_seconds
        return (365.0 * 24.0 * 3600.0) / mean_dt

    def compute_indicators(self):
        """Compute indicators from current data."""
        with self.trade_lock:
            df = pd.DataFrame(list(self.trades))
        with self.quote_lock:
            q = dict(self.latest_quote)

        out = {}
        if df.empty:
            return out

        # Convert timestamps to datetime index
        df = df.copy()
        df["dt"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("dt", inplace=True)

        prices = df["price"].astype(float)
        sizes = df["size"].astype(float)
        signs = df["sign"].astype(int)

        # 1) EMAs
        out["ema_12"] = prices.ewm(span=12, adjust=False).mean()
        out["ema_26"] = prices.ewm(span=26, adjust=False).mean()

        # 2) Bollinger Bands (20, 2)
        bb_period = 20
        bb_std = 2
        sma_20 = prices.rolling(window=bb_period).mean()
        std_20 = prices.rolling(window=bb_period).std()
        out["bb_upper"] = sma_20 + (std_20 * bb_std)
        out["bb_middle"] = sma_20
        out["bb_lower"] = sma_20 - (std_20 * bb_std)

        # 3) RSI(14)
        def _rsi(prices_ser: pd.Series, period: int = 14) -> pd.Series:
            delta = prices_ser.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        out["rsi"] = _rsi(prices, 14)

        # 4) Realized vol (annualized) using estimated sampling frequency
        ret = prices.pct_change().dropna()
        rv_window = 20
        periods_per_year = self._estimate_periods_per_year(df.index)
        out["realized_vol"] = ret.rolling(window=rv_window).std() * np.sqrt(periods_per_year)

        # 5) VPIN (mean absolute signed bucket / bucket volume over window)
        if len(self.vpin_buckets) > 0:
            vpin_vals = np.array([abs(x) for x in self.vpin_buckets], dtype=float)
            out["vpin"] = float(np.mean(vpin_vals) / float(VPIN_BUCKET_VOLUME))
        else:
            out["vpin"] = float(0.0)

        # 6) Hawkes intensity (current state)
        out["hawkes_intensity"] = float(self.hawkes_state)

        # 7) Last price
        out["last_price"] = float(prices.iloc[-1]) if not prices.empty else float("nan")

        return out

    def get_indicator_data_for_chart(self, max_points=1000):
        """Package indicators for JSON transport."""
        indicators = self.compute_indicators()
        if not indicators:
            return {}

        chart_data = {}

        def clean_float(value):
            if value is None:
                return 0.0
            try:
                f = float(value)
                if np.isnan(f) or np.isinf(f):
                    return 0.0
                return f
            except Exception:
                return 0.0

        # Convert pandas Series to lists for JSON serialization
        for key, value in indicators.items():
            if isinstance(value, pd.Series):
                if not value.empty:
                    recent = value.tail(max_points)
                    cleaned_values = [clean_float(v) for v in recent.tolist()]
                    chart_data[key] = {
                        "timestamps": [int(ts.timestamp() * 1000) for ts in recent.index],
                        "values": cleaned_values,
                    }
                else:
                    chart_data[key] = {"timestamps": [], "values": []}
            else:
                chart_data[key] = clean_float(value)

        return chart_data


# Global indicator calculator instance
indicator_calc = IndicatorCalculator()
