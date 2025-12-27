from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from indicators import (
    vwap as calc_vwap,
    session_vwap as calc_session_vwap,
    atr as calc_atr,
    ema as calc_ema,
    rolling_swing_lows,
    rolling_swing_highs,
    detect_fvg,
    find_order_block,
    find_breaker_block,
    in_zone,
)
from sessions import classify_session


@dataclass
class SignalResult:
    symbol: str
    bias: str                      # "LONG", "SHORT", "NEUTRAL"
    setup_score: int               # 0..100
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target_1r: Optional[float]
    target_2r: Optional[float]
    last_price: Optional[float]
    timestamp: Optional[pd.Timestamp]
    session: str                   # OPENING/MIDDAY/POWER/OFF
    extras: Dict[str, Any]         # Pro diagnostics / context


PRESETS: Dict[str, Dict[str, float]] = {
    "Fast scalp": {
        "min_actionable_score": 70,
        "vol_multiplier": 1.15,
        "require_volume": 0,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
    "Cleaner signals": {
        "min_actionable_score": 80,
        "vol_multiplier": 1.35,
        "require_volume": 1,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
}


def _fib_levels(hi: float, lo: float, direction: str) -> List[Tuple[str, float]]:
    """
    direction:
      - "up"   => retracement levels from hi down toward lo: hi - r*(hi-lo)
      - "down" => retracement levels from lo up toward hi: lo + r*(hi-lo)
    """
    ratios = [0.382, 0.5, 0.618, 0.786]
    levels = []
    rng = hi - lo
    if rng <= 0:
        return levels
    for r in ratios:
        if direction == "up":
            levels.append((f"Fib {r:g}", hi - r * rng))
        else:
            levels.append((f"Fib {r:g}", lo + r * rng))
    return levels


def _closest_level(price: float, levels: List[Tuple[str, float]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not levels:
        return None, None, None
    best = min(levels, key=lambda x: abs(price - x[1]))
    name, lvl = best
    dist = abs(price - lvl)
    return name, float(lvl), float(dist)


def compute_scalp_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series,
    macd_hist: pd.Series,
    *,
    mode: str = "Cleaner signals",
    pro_mode: bool = False,
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    lookback_bars: int = 180,
    vwap_logic: str = "session",     # "session" or "cumulative"
    fib_lookback_bars: int = 120,    # used for fib context/scoring
) -> SignalResult:
    if len(ohlcv) < 60:
        return SignalResult(symbol, "NEUTRAL", 0, "Not enough data", None, None, None, None, None, None, "OFF", {})

    cfg = PRESETS.get(mode, PRESETS["Cleaner signals"])

    df = ohlcv.copy().tail(int(lookback_bars)).copy()
    df["vwap_cum"] = calc_vwap(df)
    df["vwap_sess"] = calc_session_vwap(df)
    df["atr14"] = calc_atr(df, 14)
    df["ema20"] = calc_ema(df["close"], 20)
    df["ema50"] = calc_ema(df["close"], 50)

    rsi_fast = rsi_fast.reindex(df.index).ffill()
    rsi_slow = rsi_slow.reindex(df.index).ffill()
    macd_hist = macd_hist.reindex(df.index).ffill()

    close = df["close"]
    vol = df["volume"]
    vwap_use = df["vwap_sess"] if vwap_logic == "session" else df["vwap_cum"]
    vwap_sess = df["vwap_sess"]
    vwap_cum = df["vwap_cum"]
    atr14 = df["atr14"]
    ema20 = df["ema20"]
    ema50 = df["ema50"]

    last_ts = df.index[-1]
    session = classify_session(last_ts)

    allowed = (
        (session == "OPENING" and allow_opening)
        or (session == "MIDDAY" and allow_midday)
        or (session == "POWER" and allow_power)
    )
    last_price = float(close.iloc[-1])

    extras: Dict[str, Any] = {
        "vwap_logic": vwap_logic,
        "vwap_session": float(vwap_sess.iloc[-1]) if np.isfinite(vwap_sess.iloc[-1]) else None,
        "vwap_cumulative": float(vwap_cum.iloc[-1]) if np.isfinite(vwap_cum.iloc[-1]) else None,
        "ema20": float(ema20.iloc[-1]) if np.isfinite(ema20.iloc[-1]) else None,
        "ema50": float(ema50.iloc[-1]) if np.isfinite(ema50.iloc[-1]) else None,
        "fib_lookback_bars": int(fib_lookback_bars),
    }

    if not allowed:
        return SignalResult(symbol, "NEUTRAL", 0, f"Filtered by time-of-day ({session})", None, None, None, None, last_price, last_ts, session, extras)

    # BASIC events (using chosen VWAP)
    was_below_vwap = (close.shift(3) < vwap_use.shift(3)).iloc[-1] or (close.shift(5) < vwap_use.shift(5)).iloc[-1]
    reclaim_vwap = (close.iloc[-1] > vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] <= vwap_use.shift(1).iloc[-1])

    was_above_vwap = (close.shift(3) > vwap_use.shift(3)).iloc[-1] or (close.shift(5) > vwap_use.shift(5)).iloc[-1]
    reject_vwap = (close.iloc[-1] < vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] >= vwap_use.shift(1).iloc[-1])

    rsi5 = float(rsi_fast.iloc[-1])
    rsi14 = float(rsi_slow.iloc[-1])

    rsi_snap = (rsi5 >= 30 and float(rsi_fast.shift(1).iloc[-1]) < 30) or (rsi5 >= 25 and float(rsi_fast.shift(1).iloc[-1]) < 25)
    rsi_downshift = (rsi5 <= 70 and float(rsi_fast.shift(1).iloc[-1]) > 70) or (rsi5 <= 75 and float(rsi_fast.shift(1).iloc[-1]) > 75)

    macd_turn_up = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] > macd_hist.shift(2).iloc[-1])
    macd_turn_down = (macd_hist.iloc[-1] < macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] < macd_hist.shift(2).iloc[-1])

    vol_med = vol.rolling(30, min_periods=10).median().iloc[-1]
    vol_ok = (vol.iloc[-1] >= float(cfg["vol_multiplier"]) * vol_med) if np.isfinite(vol_med) else False

    # Swings for stops + liquidity reference
    swing_low_mask = rolling_swing_lows(df["low"], left=3, right=3)
    recent_swing_lows = df.loc[swing_low_mask, "low"].tail(6)
    recent_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(12).min())

    swing_high_mask = rolling_swing_highs(df["high"], left=3, right=3)
    recent_swing_highs = df.loc[swing_high_mask, "high"].tail(6)
    recent_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(12).max())

    atr_last = float(atr14.iloc[-1]) if np.isfinite(atr14.iloc[-1]) else 0.0
    buffer = 0.25 * atr_last if atr_last else 0.0
    extras["atr14"] = atr_last

    # Trend context
    trend_long_ok = bool((close.iloc[-1] >= ema20.iloc[-1]) and (ema20.iloc[-1] >= ema50.iloc[-1]))
    trend_short_ok = bool((close.iloc[-1] <= ema20.iloc[-1]) and (ema20.iloc[-1] <= ema50.iloc[-1]))
    extras["trend_long_ok"] = trend_long_ok
    extras["trend_short_ok"] = trend_short_ok

    # Fib context (recent swing range)
    seg = df.tail(int(min(max(60, fib_lookback_bars), len(df))))
    hi = float(seg["high"].max())
    lo = float(seg["low"].min())
    rng = hi - lo
    fib_bias = "none"
    fib_levels_up = []
    fib_levels_dn = []
    fib_name = fib_level = fib_dist = None
    fib_near_long = fib_near_short = False

    if rng > 0:
        # Determine directional context by where price sits in the range.
        pos = (last_price - lo) / rng
        if pos >= 0.60:
            fib_bias = "up"   # treat as an up-impulse context; look for pullback levels (from hi downward)
            fib_levels_up = _fib_levels(hi, lo, "up")
            fib_name, fib_level, fib_dist = _closest_level(last_price, fib_levels_up)
        elif pos <= 0.40:
            fib_bias = "down" # treat as down-impulse context; look for bounce levels (from lo upward) for short entries
            fib_levels_dn = _fib_levels(hi, lo, "down")
            fib_name, fib_level, fib_dist = _closest_level(last_price, fib_levels_dn)
        else:
            fib_bias = "range"

    extras["fib_hi"] = hi if rng > 0 else None
    extras["fib_lo"] = lo if rng > 0 else None
    extras["fib_bias"] = fib_bias
    extras["fib_closest"] = {"name": fib_name, "level": fib_level, "dist": fib_dist}

    if fib_level is not None and fib_dist is not None:
        near = fib_dist <= max(buffer, 0.0) if atr_last else (fib_dist <= (0.002 * last_price))
        if near:
            if fib_bias == "up":
                fib_near_long = True
            elif fib_bias == "down":
                fib_near_short = True
    extras["fib_near_long"] = fib_near_long
    extras["fib_near_short"] = fib_near_short

    # Liquidity sweeps
    prior_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(30).max())
    prior_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(30).min())
    bull_sweep = bool((df["low"].iloc[-1] < prior_swing_low) and (df["close"].iloc[-1] > prior_swing_low))
    bear_sweep = bool((df["high"].iloc[-1] > prior_swing_high) and (df["close"].iloc[-1] < prior_swing_high))
    extras["prior_swing_high"] = prior_swing_high
    extras["prior_swing_low"] = prior_swing_low
    extras["bull_liquidity_sweep"] = bull_sweep
    extras["bear_liquidity_sweep"] = bear_sweep

    # FVG + OB
    bull_fvg, bear_fvg = detect_fvg(df.tail(60))
    extras["bull_fvg"] = bull_fvg
    extras["bear_fvg"] = bear_fvg

    ob_bull = find_order_block(df, atr14, side="bull", lookback=35)
    ob_bear = find_order_block(df, atr14, side="bear", lookback=35)
    extras["bull_ob"] = ob_bull
    extras["bear_ob"] = ob_bear

    bull_ob_retest = bool(ob_bull[0] is not None and in_zone(last_price, ob_bull[0], ob_bull[1], buffer=buffer))
    bear_ob_retest = bool(ob_bear[0] is not None and in_zone(last_price, ob_bear[0], ob_bear[1], buffer=buffer))
    extras["bull_ob_retest"] = bull_ob_retest
    extras["bear_ob_retest"] = bear_ob_retest

    # Breaker blocks
    brk_bull = find_breaker_block(df, atr14, side="bull", lookback=60)
    brk_bear = find_breaker_block(df, atr14, side="bear", lookback=60)
    extras["bull_breaker"] = brk_bull
    extras["bear_breaker"] = brk_bear

    bull_breaker_retest = bool(brk_bull[0] is not None and in_zone(last_price, brk_bull[0], brk_bull[1], buffer=buffer))
    bear_breaker_retest = bool(brk_bear[0] is not None and in_zone(last_price, brk_bear[0], brk_bear[1], buffer=buffer))
    extras["bull_breaker_retest"] = bull_breaker_retest
    extras["bear_breaker_retest"] = bear_breaker_retest

    last_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
    displacement = bool(atr_last and last_range >= 1.5 * atr_last)
    extras["displacement"] = displacement

    # Scoring
    long_points = 0
    long_reasons = []
    if was_below_vwap and reclaim_vwap:
        long_points += 35; long_reasons.append(f"VWAP reclaim ({vwap_logic})")
    if rsi_snap and rsi14 < 60:
        long_points += 20; long_reasons.append("RSI-5 snapback (RSI-14 ok)")
    if macd_turn_up:
        long_points += 20; long_reasons.append("MACD hist turning up")
    if vol_ok:
        long_points += 15; long_reasons.append("Volume confirmation")
    if df["low"].tail(12).iloc[-1] > df["low"].tail(12).min():
        long_points += 10; long_reasons.append("Higher-low micro structure")

    short_points = 0
    short_reasons = []
    if was_above_vwap and reject_vwap:
        short_points += 35; short_reasons.append(f"VWAP rejection ({vwap_logic})")
    if rsi_downshift and rsi14 > 40:
        short_points += 20; short_reasons.append("RSI-5 downshift (RSI-14 ok)")
    if macd_turn_down:
        short_points += 20; short_reasons.append("MACD hist turning down")
    if vol_ok:
        short_points += 15; short_reasons.append("Volume confirmation")
    if df["high"].tail(12).iloc[-1] < df["high"].tail(12).max():
        short_points += 10; short_reasons.append("Lower-high micro structure")

    # Fib scoring (applies in both basic & pro because it's a passive confluence)
    # Stronger weighting for 0.5/0.618 proximity.
    if fib_near_long and fib_name is not None:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        long_points += add
        long_reasons.append(f"Near {fib_name}")
    if fib_near_short and fib_name is not None:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        short_points += add
        short_reasons.append(f"Near {fib_name}")

    if pro_mode:
        if bull_sweep:
            long_points += 20; long_reasons.append("Liquidity sweep (low)")
        if bear_sweep:
            short_points += 20; short_reasons.append("Liquidity sweep (high)")
        if bull_ob_retest:
            long_points += 15; long_reasons.append("Bullish order block retest")
        if bear_ob_retest:
            short_points += 15; short_reasons.append("Bearish order block retest")
        if bull_fvg is not None:
            long_points += 10; long_reasons.append("Bullish FVG present")
        if bear_fvg is not None:
            short_points += 10; short_reasons.append("Bearish FVG present")

        # Breaker blocks: stronger than OB retest for scalps
        if bull_breaker_retest:
            long_points += 20; long_reasons.append("Bullish breaker retest")
        if bear_breaker_retest:
            short_points += 20; short_reasons.append("Bearish breaker retest")

        if displacement:
            long_points += 5; short_points += 5

        if not trend_long_ok and not (was_below_vwap and reclaim_vwap):
            long_points = max(0, long_points - 15)
        if not trend_short_ok and not (was_above_vwap and reject_vwap):
            short_points = max(0, short_points - 15)

    # Requirements
    if int(cfg["require_vwap_event"]) == 1:
        if not ((was_below_vwap and reclaim_vwap) or (was_above_vwap and reject_vwap)):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No VWAP reclaim/rejection event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_rsi_event"]) == 1:
        if not (rsi_snap or rsi_downshift):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No RSI-5 snap/downshift event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_macd_turn"]) == 1:
        if not (macd_turn_up or macd_turn_down):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No MACD histogram turn event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_volume"]) == 1 and not vol_ok:
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No volume confirmation", None, None, None, None, last_price, last_ts, session, extras)

    if pro_mode:
        if not (bull_sweep or bear_sweep or bull_ob_retest or bear_ob_retest or bull_breaker_retest or bear_breaker_retest):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "Pro mode: no sweep / OB / breaker trigger", None, None, None, None, last_price, last_ts, session, extras)

    min_score = int(cfg["min_actionable_score"])

    # Entry/stop adjustments when breaker/fib confluence is active
    def _long_entry_stop(entry_px: float):
        stop_px = float(min(recent_swing_low, entry_px - max(atr_last, 0.0) * 0.8))
        if bull_breaker_retest and brk_bull[0] is not None:
            stop_px = float(min(stop_px, brk_bull[0] - buffer))
        # If near fib and we have a valid fib_level, treat that as structural support
        if fib_near_long and fib_level is not None:
            stop_px = float(min(stop_px, fib_level - buffer))
        return entry_px, stop_px

    def _short_entry_stop(entry_px: float):
        stop_px = float(max(recent_swing_high, entry_px + max(atr_last, 0.0) * 0.8))
        if bear_breaker_retest and brk_bear[1] is not None:
            stop_px = float(max(stop_px, brk_bear[1] + buffer))
        if fib_near_short and fib_level is not None:
            stop_px = float(max(stop_px, fib_level + buffer))
        return entry_px, stop_px

    if long_points >= min_score and long_points > short_points:
        entry, stop = _long_entry_stop(last_price)
        risk = max(entry - stop, 0.01)
        return SignalResult(symbol, "LONG", min(100, int(long_points)), ", ".join(long_reasons[:10]), entry, stop, entry + risk, entry + 2 * risk, last_price, last_ts, session, extras)

    if short_points >= min_score and short_points > long_points:
        entry, stop = _short_entry_stop(last_price)
        risk = max(stop - entry, 0.01)
        return SignalResult(symbol, "SHORT", min(100, int(short_points)), ", ".join(short_reasons[:10]), entry, stop, entry - risk, entry - 2 * risk, last_price, last_ts, session, extras)

    reason = f"LongScore={long_points} ({', '.join(long_reasons)}); ShortScore={short_points} ({', '.join(short_reasons)})"
    return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), reason, None, None, None, None, last_price, last_ts, session, extras)
