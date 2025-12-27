# Ztockly Scalping Scanner v6 (Fib Scoring + Session VWAP + Breaker Blocks)

What’s new vs v5:
- **Fib retracement confluence** is now part of scoring + stop logic.
  - Uses a recent swing range (default **120 bars**, adjustable in UI).
  - If price is *near* a key Fib level (0.382 / 0.5 / 0.618 / 0.786) it adds points.
  - Stronger weighting for 0.5 / 0.618.
  - If near a Fib, stop can tighten to just beyond that level (ATR-buffered).

Still included:
- Session VWAP (ET reset) + cumulative VWAP
- VWAP logic selector for signals + dual VWAP chart toggle
- Pro mode: liquidity sweeps + order blocks + breaker blocks + FVG + EMA context
- In-app alerts with cooldown + time-of-day filters

## Run
```bash
pip install -r requirements.txt
export ALPHAVANTAGE_API_KEY="YOUR_KEY"
streamlit run app.py
```
