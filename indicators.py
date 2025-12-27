from __future__ import annotations

import pandas as pd
import numpy as np


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    "Cumulative" VWAP across the full dataframe window.
    (This does NOT reset at session boundaries.)
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].cumsum().replace(0, np.nan)


def session_vwap(df: pd.DataFrame, tz: str = "America/New_York") -> pd.Series:
    """
    VWAP that resets each trading day/session (ET).
    Assumes df.index is intraday timestamps. If tz-naive, we treat as ET.
    """
    if df.empty:
        return pd.Series(dtype="float64")

    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx_et = idx.tz_localize(tz)
    else:
        idx_et = idx.tz_convert(tz)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    dates = pd.Series(idx_et.date, index=df.index)

    out = pd.Series(index=df.index, dtype="float64")
    for d, g in df.groupby(dates):
        gtp = tp.loc[g.index]
        gpv = pv.loc[g.index]
        out.loc[g.index] = gpv.cumsum() / g["volume"].cumsum().replace(0, np.nan)
    return out


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rolling_swing_lows(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    s = series
    is_low = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        window = s.iloc[i - left: i + right + 1]
        if s.iloc[i] == window.min():
            is_low.iloc[i] = True
    return is_low


def rolling_swing_highs(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    s = series
    is_high = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        window = s.iloc[i - left: i + right + 1]
        if s.iloc[i] == window.max():
            is_high.iloc[i] = True
    return is_high


def detect_fvg(df: pd.DataFrame):
    """
    Simple Fair Value Gap (FVG) detector (3-candle gap).
    Returns latest bullish and bearish FVG zones (if any) as tuples:
      bullish: (low_of_gap, high_of_gap) where low_of_gap = high[i-2], high_of_gap = low[i]
      bearish: (low_of_gap, high_of_gap) where low_of_gap = high[i], high_of_gap = low[i-2]
    """
    if len(df) < 3:
        return None, None
    h = df["high"].values
    l = df["low"].values
    bull = None
    bear = None
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            bull = (float(h[i - 2]), float(l[i]))
        if h[i] < l[i - 2]:
            bear = (float(h[i]), float(l[i - 2]))
    return bull, bear


def find_order_block(df: pd.DataFrame, atr_series: pd.Series, side: str = "bull", lookback: int = 30):
    """
    Lightweight "order block" approximation:
    - Bullish OB: last bearish candle before displacement up
    - Bearish OB: last bullish candle before displacement down

    Displacement:
      Within next 1-3 candles, close moves > 1.0*ATR and closes beyond candle high/low.

    Returns: (zone_low, zone_high, index_timestamp) or (None, None, None)
    """
    if len(df) < 10:
        return None, None, None
    df = df.tail(lookback).copy()
    atr_series = atr_series.reindex(df.index).ffill()

    opens = df["open"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    at = atr_series.values
    idx = df.index

    if side == "bull":
        for i in range(len(df) - 4, -1, -1):
            if closes[i] < opens[i]:
                atr_i = at[i] if np.isfinite(at[i]) else None
                if not atr_i:
                    continue
                for j in range(i + 1, min(i + 4, len(df))):
                    if closes[j] > highs[i] and (closes[j] - closes[i]) > 1.0 * atr_i:
                        zone_low = float(lows[i])
                        zone_high = float(opens[i])
                        return min(zone_low, zone_high), max(zone_low, zone_high), idx[i]
    else:
        for i in range(len(df) - 4, -1, -1):
            if closes[i] > opens[i]:
                atr_i = at[i] if np.isfinite(at[i]) else None
                if not atr_i:
                    continue
                for j in range(i + 1, min(i + 4, len(df))):
                    if closes[j] < lows[i] and (closes[i] - closes[j]) > 1.0 * atr_i:
                        zone_high = float(highs[i])
                        zone_low = float(opens[i])
                        return min(zone_low, zone_high), max(zone_low, zone_high), idx[i]
    return None, None, None


def find_breaker_block(df: pd.DataFrame, atr_series: pd.Series, side: str = "bull", lookback: int = 60):
    """
    Scalping-friendly breaker block approximation:

    Bullish breaker:
      1) Find a recent *bearish order block* zone (last bullish candle before displacement down).
      2) Price later breaks ABOVE that zone (close > zone_high + 0.2*ATR).
      3) Current price is retesting inside/near that zone from ABOVE (acts as support).

    Bearish breaker:
      1) Find a recent *bullish order block* zone (last bearish candle before displacement up).
      2) Price later breaks BELOW that zone (close < zone_low - 0.2*ATR).
      3) Current price is retesting inside/near that zone from BELOW (acts as resistance).

    Returns: (zone_low, zone_high, index_timestamp) or (None, None, None)
    """
    if len(df) < 20:
        return None, None, None

    df = df.tail(lookback).copy()
    atr_series = atr_series.reindex(df.index).ffill()
    atr_last = float(atr_series.iloc[-1]) if np.isfinite(atr_series.iloc[-1]) else 0.0
    pad = 0.2 * atr_last if atr_last else 0.0

    # Under the hood: breaker uses the *opposite* order block type
    if side == "bull":
        # bearish OB zone, then break up, then retest from above
        zone = find_order_block(df, atr_series, side="bear", lookback=lookback)
        zl, zh, ts = zone
        if zl is None:
            return None, None, None
        # did we break above?
        broke_up = (df["close"] > (zh + pad)).any()
        if not broke_up:
            return None, None, None
        # retest from above: last close is within zone (or just above)
        last = float(df["close"].iloc[-1])
        if (last >= (zl - pad)) and (last <= (zh + pad)):
            return float(zl), float(zh), ts
        return None, None, None
    else:
        # bullish OB zone, then break down, then retest from below
        zone = find_order_block(df, atr_series, side="bull", lookback=lookback)
        zl, zh, ts = zone
        if zl is None:
            return None, None, None
        broke_dn = (df["close"] < (zl - pad)).any()
        if not broke_dn:
            return None, None, None
        last = float(df["close"].iloc[-1])
        if (last >= (zl - pad)) and (last <= (zh + pad)):
            return float(zl), float(zh), ts
        return None, None, None


def in_zone(price: float, zone_low: float, zone_high: float, buffer: float = 0.0) -> bool:
    lo = zone_low - buffer
    hi = zone_high + buffer
    return (price >= lo) and (price <= hi)
