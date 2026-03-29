"""
Leif Soreide - High Tight Flag (HTF) Backtest
==============================================
Full implementation of O'Neil/Soreide HTF criteria including
6-component scoring system.

Scoring weights:
  Pole       25%  - explosive advance, volume, clean up days
  Flag       25%  - tight consolidation, above 50MA, volume dry-up
  Volume     20%  - pole volume, flag volume, breakout surge
  Technical  15%  - RS, new highs, MA positioning
  Breakout   10%  - O'Neil pivot rule, R:R ratio
  Catalyst    5%  - price/volume action proxy
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import sys
import time

print(f"pandas   {pd.__version__}")
print(f"yfinance {yf.__version__}")
print(f"numpy    {np.__version__}\n")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    # ── Pole criteria ──
    "pole_min_gain":          0.90,   # >= 90% gain
    "pole_max_days":          40,     # <= 40 trading days (~8 weeks)
    "pole_min_vol_ratio":     1.40,   # pole avg volume >= 1.4x prior 20d avg
    "pole_min_up_day_pct":    0.55,   # >= 55% of pole days must be up days

    # ── Flag criteria ──
    "flag_min_drawdown":      0.10,   # >= 10% pullback (too shallow = weak flag)
    "flag_max_drawdown":      0.25,   # <= 25% pullback
    "flag_min_days":          5,      # minimum 1 week
    "flag_max_days":          25,     # max 5 weeks (3 weeks ideal)
    "flag_must_be_above_50ma":True,   # flag lows must stay above 50-day MA
    "flag_vol_dry_ratio":     0.75,   # flag avg vol <= 75% of pole avg vol

    # ── Breakout criteria ──
    "breakout_min_buffer":    0.10,   # price > flag_high + $0.10 (O'Neil rule)
    "volume_ratio":           1.50,   # breakout volume >= 1.5x 20d avg (50%+ above)

    # ── Filters ──
    "rs_percentile":          80,     # RS >= 80th percentile
    "rs_lookback":            63,     # 63-day RS lookback
    "min_price":              5.0,
    "min_score":              5.0,    # minimum composite score (0-10) to take trade

    # ── Trade management ──
    "trailing_stop_pct":      0.15,
    "max_hold_days":          40,

    # ── Backtest settings ──
    "start_date":             "2010-01-01",
    "end_date":               "2024-12-31",
    "initial_capital":        100_000,
    "risk_per_trade":         0.02,
    "max_concurrent":         5,
    "batch_size":             50,
    "min_trading_days":       200,
}

# ─────────────────────────────────────────────
# UNIVERSE — expanded to 160 stocks for more signals
# ─────────────────────────────────────────────
UNIVERSE = [
    # Mega-cap tech
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","AVGO","ORCL",
    # Semiconductors
    "LRCX","KLAC","AMAT","MRVL","MPWR","ON","ACLS","ONTO","RMBS","SLAB",
    "WOLF","SITM","AMBA","POWI","IREN",
    # Software / Cloud
    "CRWD","PANW","FTNT","SNOW","DDOG","NET","ZS","ANET","NOW","WDAY",
    "INTU","ADBE","CRM","VEEV","HUBS","BILL","MDB","GTLB","ESTC","DOMO",
    "APPN","MNDY","SMAR","DOCN","CFLT",
    # Biotech / Pharma
    "MRNA","BNTX","LLY","NVO","REGN","VRTX","ABBV","IDXX","DXCM","PODD",
    "ISRG","EXEL","HALO","DVAX","TMDX","RGEN","ACAD","VKTX","PCVX","RXRX",
    "NVCR","FOLD","ARWR","IONS","KRYS","PRAX","BEAM","INVA","RARE","ALNY",
    # Consumer / Retail
    "LULU","CELH","ELF","MNST","ORLY","DECK","ONON","CROX","WING","CAVA",
    "SFM","DUOL","MEDP","BOOT","CHWY","RH","FIVE","CVNA","DRVN","SHAK",
    "BIRK","TPVG","MODG","YETI","XPOF",
    # Financials / Fintech
    "IBKR","COIN","MARA","RIOT","HOOD","AFRM","SOFI","PYPL","UPST","GDOT",
    "CLSK","HUT","MSTR","BKNG","ABNB",
    # Industrials / Defence / Energy
    "GNRC","TDG","HEI","AXON","ODFL","SAIA","ENPH","FSLR","SEDG","PLUG",
    "HIMS","DOCS","RELY","ASTS","RKLB","LUNR","CORZ","SMCI","PLTR","SOUN",
    # International ADRs
    "MELI","SE","SHOP","BABA","NIO","XPEV","LI","GRAB","DKNG","RBLX",
    "NU","TCOM","PDD","TIGR","FUTU",
    # Market filter — must stay last
    "SPY",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def progress(current, total, label="", width=35):
    pct  = current / max(total, 1)
    done = int(width * pct)
    bar  = "█" * done + "░" * (width - done)
    sys.stdout.write(f"\r  [{bar}] {current}/{total} {label}   ")
    sys.stdout.flush()
    if current >= total:
        print()


def get_index(df, date):
    pos = df.index.searchsorted(date, side="right") - 1
    return int(pos) if pos >= 0 else None


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data(tickers, start, end, batch_size=50, min_days=200):
    prices     = {}
    n          = len(tickers)
    n_batches  = (n + batch_size - 1) // batch_size
    failed     = 0
    first_diag = True

    print(f"Downloading {n} tickers in {n_batches} batches of {batch_size}...\n")

    for i in range(n_batches):
        batch = tickers[i * batch_size:(i + 1) * batch_size]
        progress(i + 1, n_batches, f"| loaded {len(prices)} | failed {failed}")

        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False,
                              threads=True, timeout=60)
        except Exception as e:
            print(f"\n  Batch {i+1} exception: {e}")
            failed += len(batch)
            time.sleep(2)
            continue

        if raw is None or raw.empty:
            print(f"\n  Batch {i+1}: empty response")
            failed += len(batch)
            time.sleep(1)
            continue

        is_multi = isinstance(raw.columns, pd.MultiIndex)
        if first_diag:
            first_diag = False
            print(f"\n  [diag] raw.shape  = {raw.shape}")
            print(f"  [diag] col type   = {'MultiIndex' if is_multi else 'flat'}")
            print(f"  [diag] col sample = {list(raw.columns[:4])}")

        for tk in batch:
            try:
                if not is_multi:
                    df = raw.copy()
                else:
                    level0 = set(raw.columns.get_level_values(0))
                    fields = {"Close","Open","High","Low","Volume"}
                    if fields & level0:
                        df = raw.xs(tk, axis=1, level=1)
                    else:
                        df = raw.xs(tk, axis=1, level=0)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                df.columns = [str(c) for c in df.columns]
                df = df.dropna(how="all")

                if not {"Close","Volume","High","Low"}.issubset(df.columns):
                    failed += 1
                    continue
                df = df[df["Close"].notna() & df["Volume"].notna()]
                if len(df) < min_days:
                    failed += 1
                    continue
                prices[tk] = df

            except Exception:
                failed += 1
                if tk == "SPY":
                    print(f"\n  [error] SPY failed!")

        time.sleep(0.5)

    print(f"\n  Loaded {len(prices)} tickers | {failed} failed/filtered")
    for tk in (["SPY"] + list(prices.keys())[:3]):
        if tk not in prices:
            continue
        df = prices[tk]
        print(f"  [diag] {tk}: rows={len(df)}, last_close={float(df['Close'].iloc[-1]):.2f}")
    print()
    return prices


# ─────────────────────────────────────────────
# RELATIVE STRENGTH
# ─────────────────────────────────────────────
def calc_rs_ratings(prices, date, lookback=63):
    returns = {}
    for tk, df in prices.items():
        if tk == "SPY":
            continue
        idx = get_index(df, date)
        if idx is None or idx < lookback:
            continue
        try:
            p0 = float(df["Close"].iloc[idx - lookback])
            p1 = float(df["Close"].iloc[idx])
            if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
                returns[tk] = (p1 - p0) / p0
        except Exception:
            pass
    if not returns:
        return {}
    s = pd.Series(returns)
    return (s.rank(pct=True) * 100).to_dict()


# ─────────────────────────────────────────────
# MARKET FILTER
# ─────────────────────────────────────────────
def market_is_green(spy_df, date):
    idx = get_index(spy_df, date)
    if idx is None or idx < 50:
        return False
    try:
        sma50 = float(spy_df["Close"].iloc[idx - 50:idx].mean())
        price = float(spy_df["Close"].iloc[idx])
        return price > sma50
    except Exception:
        return False


# ─────────────────────────────────────────────
# 6-COMPONENT SCORING  (returns 0-10 score)
# ─────────────────────────────────────────────
def score_pattern(closes, volumes, pole_start, pole_end, flag_end, n, rs_val, cfg):
    """
    Scores the HTF pattern across 6 dimensions, each 0–10.
    Weighted composite returned as final score (0–10).

    Components:
      pole_score      25%
      flag_score      25%
      volume_score    20%
      technical_score 15%
      breakout_score  10%
      catalyst_score   5%
    """
    scores = {}

    # ── 1. POLE SCORE (25%) ─────────────────────────────────────
    pole_gain      = (closes[pole_end] - closes[pole_start]) / closes[pole_start]
    pole_days      = pole_end - pole_start

    # Gain sub-score: 90%=5, 100%=7, 120%+=10
    if pole_gain >= 1.20:
        gain_sub = 10.0
    elif pole_gain >= 1.00:
        gain_sub = 7.0 + (pole_gain - 1.00) / 0.20 * 3.0
    elif pole_gain >= 0.90:
        gain_sub = 5.0 + (pole_gain - 0.90) / 0.10 * 2.0
    else:
        gain_sub = pole_gain / 0.90 * 5.0

    # Speed sub-score: faster is better (20 days=10, 40 days=5)
    speed_sub = max(0, 10 - (pole_days - 20) * 0.25)

    # Up-day ratio sub-score
    pole_slice  = closes[pole_start:pole_end + 1]
    up_days     = sum(1 for i in range(1, len(pole_slice)) if pole_slice[i] > pole_slice[i-1])
    up_day_pct  = up_days / max(len(pole_slice) - 1, 1)
    upday_sub   = min(10.0, up_day_pct * 10.0 / 0.7)   # 70% up days = 10

    scores["pole"] = (gain_sub * 0.5 + speed_sub * 0.3 + upday_sub * 0.2)

    # ── 2. FLAG SCORE (25%) ─────────────────────────────────────
    flag_seg      = closes[pole_end:n]           # excludes today
    flag_high_v   = float(flag_seg.max())
    flag_low_v    = float(flag_seg.min())
    flag_drawdown = (flag_high_v - flag_low_v) / flag_high_v
    flag_days     = n - pole_end

    # Drawdown sub-score: 10-15% = 10, 15-20% = 7, 20-25% = 5
    if flag_drawdown <= 0.15:
        dd_sub = 10.0
    elif flag_drawdown <= 0.20:
        dd_sub = 7.0 + (0.20 - flag_drawdown) / 0.05 * 3.0
    elif flag_drawdown <= 0.25:
        dd_sub = 5.0 + (0.25 - flag_drawdown) / 0.05 * 2.0
    else:
        dd_sub = 0.0

    # Duration sub-score: 5-15 days = 10, 15-20 = 7, 20-25 = 5
    if flag_days <= 15:
        dur_sub = 10.0
    elif flag_days <= 20:
        dur_sub = 7.0
    else:
        dur_sub = 5.0

    # Tightness: low volatility during flag (std of daily returns)
    if len(flag_seg) > 2:
        flag_returns = np.diff(flag_seg) / flag_seg[:-1]
        flag_vol     = float(np.std(flag_returns))
        tight_sub    = max(0, 10.0 - flag_vol * 200)   # 5% daily std = 0
    else:
        tight_sub = 5.0

    scores["flag"] = (dd_sub * 0.4 + dur_sub * 0.3 + tight_sub * 0.3)

    # ── 3. VOLUME SCORE (20%) ────────────────────────────────────
    pre_pole_vol   = volumes[max(pole_start - 20, 0):pole_start]
    pre_pole_avg   = float(np.nanmean(pre_pole_vol)) if len(pre_pole_vol) else 1
    pole_vol_avg   = float(np.nanmean(volumes[pole_start:pole_end + 1]))
    flag_vol_avg   = float(np.nanmean(volumes[pole_end:n]))
    breakout_vol   = float(volumes[n])
    pre_break_avg  = float(np.nanmean(volumes[max(n - 20, 0):n]))

    # Pole volume sub-score: should be 40-100%+ above prior avg
    if pre_pole_avg > 0:
        pole_vol_ratio = pole_vol_avg / pre_pole_avg
        pvol_sub = min(10.0, max(0, (pole_vol_ratio - 1.0) / 1.0 * 10))  # 2x = 10
    else:
        pvol_sub = 5.0

    # Flag dry-up sub-score: flag vol should be well below pole vol
    if pole_vol_avg > 0:
        dry_ratio = flag_vol_avg / pole_vol_avg
        dry_sub   = max(0, 10.0 - dry_ratio * 10)   # 0% of pole vol = 10
    else:
        dry_sub = 5.0

    # Breakout volume sub-score: 1.5x = 5, 2x = 8, 3x+ = 10
    if pre_break_avg > 0:
        bvol_ratio = breakout_vol / pre_break_avg
        bvol_sub   = min(10.0, max(0, (bvol_ratio - 1.0) / 2.0 * 10))
    else:
        bvol_sub = 5.0

    scores["volume"] = (pvol_sub * 0.3 + dry_sub * 0.4 + bvol_sub * 0.3)

    # ── 4. TECHNICAL SCORE (15%) ─────────────────────────────────
    # RS sub-score
    rs_sub = min(10.0, max(0, (rs_val - 50) / 50 * 10))

    # Near 52-week high sub-score
    high_52 = float(np.nanmax(closes[max(n - 252, 0):n + 1]))
    near_high_pct = closes[n] / high_52 if high_52 > 0 else 0
    high_sub = min(10.0, near_high_pct * 10)   # at 52wk high = 10

    # Above key MAs (50, 20)
    ma50  = float(np.nanmean(closes[max(n - 50, 0):n]))
    ma20  = float(np.nanmean(closes[max(n - 20, 0):n]))
    ma_sub = 0.0
    if closes[n] > ma50:
        ma_sub += 5.0
    if closes[n] > ma20:
        ma_sub += 5.0

    scores["technical"] = (rs_sub * 0.4 + high_sub * 0.3 + ma_sub * 0.3)

    # ── 5. BREAKOUT SCORE (10%) ──────────────────────────────────
    flag_high_v = float(closes[pole_end:n].max())
    pivot       = flag_high_v + cfg["breakout_min_buffer"]
    today_close = closes[n]

    # Proximity: just above pivot = best; too far = chasing
    excess_pct  = (today_close - pivot) / pivot if pivot > 0 else 0
    if excess_pct < 0:
        prox_sub = 0.0   # not a breakout
    elif excess_pct <= 0.02:
        prox_sub = 10.0  # within 2% of pivot = ideal
    elif excess_pct <= 0.05:
        prox_sub = 7.0
    else:
        prox_sub = max(0, 10.0 - excess_pct * 100)

    # R:R ratio: target = pole_gain * 0.5 from breakout, risk = flag drawdown
    rr_target   = today_close * (1 + pole_gain * 0.5)
    rr_risk     = today_close - flag_low_v
    rr_reward   = rr_target - today_close
    rr_ratio    = rr_reward / rr_risk if rr_risk > 0 else 0
    rr_sub      = min(10.0, rr_ratio * 10 / 3)   # 3:1 = 10

    scores["breakout"] = (prox_sub * 0.6 + rr_sub * 0.4)

    # ── 6. CATALYST SCORE (5%) ───────────────────────────────────
    # Proxy: gap-up at start of pole (price jumped > 5% on high volume)
    if pole_start > 0:
        gap_pct = (closes[pole_start] - closes[pole_start - 1]) / closes[pole_start - 1]
        gap_vol = volumes[pole_start] / max(float(np.nanmean(volumes[max(pole_start-20,0):pole_start])), 1)
        if gap_pct >= 0.10 and gap_vol >= 2.0:
            cat_sub = 10.0   # strong gap + volume = clear catalyst
        elif gap_pct >= 0.05:
            cat_sub = 7.0
        else:
            cat_sub = 4.0    # no identifiable catalyst
    else:
        cat_sub = 4.0

    scores["catalyst"] = cat_sub

    # ── WEIGHTED COMPOSITE ──────────────────────────────────────
    weights = {
        "pole":      0.25,
        "flag":      0.25,
        "volume":    0.20,
        "technical": 0.15,
        "breakout":  0.10,
        "catalyst":  0.05,
    }
    composite = sum(scores[k] * weights[k] for k in weights)
    return round(composite, 2), {k: round(v, 1) for k, v in scores.items()}


# ─────────────────────────────────────────────
# HTF PATTERN DETECTION (full criteria)
# ─────────────────────────────────────────────
def find_htf_breakout(closes, volumes, n, cfg, rs_val=0):
    """
    Full O'Neil/Soreide HTF detection. Returns pattern dict or None.

    Pole segment:  pole_start → pole_end
    Flag segment:  pole_end   → n-1   (excludes today)
    Breakout:      n          = today
    """
    # Pre-compute 50-day MA for flag-above-MA check
    ma50_at_n = float(np.nanmean(closes[max(n - 50, 0):n]))

    for pole_end in range(n - cfg["flag_min_days"],
                          max(n - cfg["flag_max_days"] - 1, 0), -1):

        flag_days = n - pole_end
        if flag_days < cfg["flag_min_days"] or flag_days > cfg["flag_max_days"]:
            continue

        # ── Flag validation ────────────────────────────────────
        flag_seg    = closes[pole_end:n]
        flag_seg_c  = flag_seg[~np.isnan(flag_seg)]
        if len(flag_seg_c) < 2:
            continue

        flag_high_v = float(flag_seg_c.max())
        flag_low_v  = float(flag_seg_c.min())
        if flag_high_v <= 0:
            continue

        flag_drawdown = (flag_high_v - flag_low_v) / flag_high_v

        # Drawdown must be 10-25%
        if flag_drawdown < cfg["flag_min_drawdown"] or flag_drawdown > cfg["flag_max_drawdown"]:
            continue

        # Flag lows must stay above 50-day MA
        if cfg["flag_must_be_above_50ma"] and flag_low_v < ma50_at_n:
            continue

        # ── Breakout validation ────────────────────────────────
        today_close = closes[n]
        pivot       = flag_high_v + cfg["breakout_min_buffer"]
        if np.isnan(today_close) or today_close < pivot:
            continue

        # Breakout volume: >= 1.5x 20-day avg
        vol_slice   = volumes[max(n - 20, 0):n]
        vol_slice   = vol_slice[~np.isnan(vol_slice)]
        if len(vol_slice) == 0:
            continue
        vol_avg    = float(vol_slice.mean())
        if vol_avg == 0:
            continue
        today_vol  = volumes[n]
        if np.isnan(today_vol) or today_vol / vol_avg < cfg["volume_ratio"]:
            continue

        # ── Pole validation ────────────────────────────────────
        for pole_start in range(pole_end - 1,
                                max(pole_end - cfg["pole_max_days"] - 1, 0), -1):
            pole_low = closes[pole_start]
            if np.isnan(pole_low) or pole_low <= 0:
                continue

            gain = (flag_high_v - pole_low) / pole_low
            if gain < cfg["pole_min_gain"]:
                continue

            # Pole volume: avg during pole must be >= 1.4x pre-pole avg
            pre_pole_vol = volumes[max(pole_start - 20, 0):pole_start]
            pre_pole_avg = float(np.nanmean(pre_pole_vol)) if len(pre_pole_vol) else 0
            pole_vol_avg = float(np.nanmean(volumes[pole_start:pole_end + 1]))
            if pre_pole_avg > 0 and pole_vol_avg / pre_pole_avg < cfg["pole_min_vol_ratio"]:
                continue

            # Up-day ratio during pole
            pole_closes = closes[pole_start:pole_end + 1]
            up_days     = sum(1 for i in range(1, len(pole_closes))
                              if pole_closes[i] > pole_closes[i - 1])
            up_day_pct  = up_days / max(len(pole_closes) - 1, 1)
            if up_day_pct < cfg["pole_min_up_day_pct"]:
                continue

            # Flag volume dry-up: flag avg vol <= 75% of pole avg vol
            flag_vol_avg = float(np.nanmean(volumes[pole_end:n]))
            if pole_vol_avg > 0 and flag_vol_avg / pole_vol_avg > cfg["flag_vol_dry_ratio"]:
                continue

            # ── Score the pattern ──────────────────────────────
            composite, component_scores = score_pattern(
                closes, volumes, pole_start, pole_end, pole_end, n, rs_val, cfg
            )

            return {
                "pole_gain":      round(gain * 100, 1),
                "pole_days":      pole_end - pole_start,
                "flag_days":      flag_days,
                "flag_drawdown":  round(flag_drawdown * 100, 1),
                "flag_low":       flag_low_v,
                "flag_high":      flag_high_v,
                "vol_ratio":      round(today_vol / vol_avg, 2),
                "up_day_pct":     round(up_day_pct * 100, 1),
                "score":          composite,
                "score_pole":     component_scores["pole"],
                "score_flag":     component_scores["flag"],
                "score_volume":   component_scores["volume"],
                "score_tech":     component_scores["technical"],
                "score_breakout": component_scores["breakout"],
                "score_catalyst": component_scores["catalyst"],
            }

    return None


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(prices, cfg):
    spy_df = prices.get("SPY")
    if spy_df is None:
        print("ERROR: SPY not loaded.")
        return pd.DataFrame(), pd.DataFrame()

    tickers      = [t for t in prices if t != "SPY"]
    start, end   = cfg["start_date"], cfg["end_date"]
    trading_days = [d for d in spy_df.index if start <= str(d.date()) <= end]
    n_days       = len(trading_days)

    if n_days == 0:
        print("ERROR: No trading days in range.")
        return pd.DataFrame(), pd.DataFrame()

    # Pre-extract numpy arrays for speed
    ticker_arrays = {}
    for tk in tickers:
        df = prices[tk]
        ticker_arrays[tk] = {
            "index":   df.index,
            "closes":  np.array(df["Close"].values,  dtype=float),
            "volumes": np.array(df["Volume"].values, dtype=float),
        }

    print(f"Backtesting {n_days:,} trading days | {len(tickers)} stocks\n")

    capital        = float(cfg["initial_capital"])
    open_positions = {}
    closed_trades  = []
    equity_curve   = []
    rs_cache       = {}
    green_days     = 0
    dbg_rs_pass    = 0
    dbg_pattern    = 0
    dbg_score_fail = 0

    for day_num, date in enumerate(trading_days):

        if day_num % 50 == 0:
            progress(day_num, n_days,
                     f"| open={len(open_positions)} "
                     f"| trades={len(closed_trades)} "
                     f"| ${capital:,.0f}")

        # ── Exits ─────────────────────────────────────────────
        to_close = []
        for tk, pos in open_positions.items():
            arr  = ticker_arrays.get(tk)
            if arr is None:
                continue
            didx = get_index(prices[tk], date)
            if didx is None:
                continue
            price = arr["closes"][didx]
            if np.isnan(price):
                continue
            pos["peak"] = max(pos["peak"], price)
            trail_stop  = pos["peak"] * (1 - cfg["trailing_stop_pct"])
            days_held   = (date - pos["entry_date"]).days
            reason = None
            if price <= pos["stop"]:
                reason = "stop_loss"
            elif price <= trail_stop and days_held > 5:
                reason = "trailing_stop"
            elif days_held >= int(cfg["max_hold_days"] * 1.4):
                reason = "time_stop"
            if reason:
                to_close.append((tk, reason, price, date))

        for tk, reason, exit_price, exit_date in to_close:
            pos     = open_positions.pop(tk)
            pnl     = (exit_price - pos["entry_price"]) * pos["shares"]
            ret     = (exit_price - pos["entry_price"]) / pos["entry_price"]
            capital += pos["position_value"] + pnl
            closed_trades.append({
                "ticker":         tk,
                "entry_date":     pos["entry_date"].date(),
                "exit_date":      exit_date.date(),
                "entry_price":    round(pos["entry_price"], 2),
                "exit_price":     round(exit_price, 2),
                "shares":         pos["shares"],
                "pnl":            round(pnl, 2),
                "return_pct":     round(ret * 100, 2),
                "days_held":      (exit_date - pos["entry_date"]).days,
                "exit_reason":    reason,
                "pole_gain":      pos["pole_gain"],
                "vol_ratio":      pos["vol_ratio"],
                "rs":             pos["rs"],
                "score":          pos["score"],
                "score_pole":     pos["score_pole"],
                "score_flag":     pos["score_flag"],
                "score_volume":   pos["score_volume"],
                "score_tech":     pos["score_tech"],
                "score_breakout": pos["score_breakout"],
                "score_catalyst": pos["score_catalyst"],
            })

        # ── Entries ───────────────────────────────────────────
        if len(open_positions) < cfg["max_concurrent"] and market_is_green(spy_df, date):
            green_days += 1

            if day_num % 5 == 0:
                rs_cache = calc_rs_ratings(prices, date, cfg["rs_lookback"])

            for tk in tickers:
                if len(open_positions) >= cfg["max_concurrent"]:
                    break
                if tk in open_positions:
                    continue

                arr = ticker_arrays.get(tk)
                if arr is None:
                    continue

                didx = get_index(prices[tk], date)
                if didx is None or didx < 80:
                    continue

                price = arr["closes"][didx]
                if np.isnan(price) or price < cfg["min_price"]:
                    continue

                rs_val = rs_cache.get(tk, 0)
                if rs_val < cfg["rs_percentile"]:
                    continue

                dbg_rs_pass += 1

                pattern = find_htf_breakout(
                    arr["closes"], arr["volumes"], didx, cfg, rs_val
                )
                if pattern is None:
                    continue

                dbg_pattern += 1

                # Score gate
                if pattern["score"] < cfg["min_score"]:
                    dbg_score_fail += 1
                    continue

                risk_per_share = price - pattern["flag_low"]
                if risk_per_share <= 0:
                    continue

                shares  = int((capital * cfg["risk_per_trade"]) / risk_per_share)
                if shares <= 0:
                    continue
                pos_val = shares * price
                if pos_val > capital * 0.25:
                    shares  = int(capital * 0.25 / price)
                    pos_val = shares * price
                if pos_val > capital or shares == 0:
                    continue

                capital -= pos_val
                open_positions[tk] = {
                    "entry_date":     date,
                    "entry_price":    price,
                    "shares":         shares,
                    "position_value": pos_val,
                    "stop":           pattern["flag_low"],
                    "peak":           price,
                    "pole_gain":      pattern["pole_gain"],
                    "vol_ratio":      pattern["vol_ratio"],
                    "rs":             round(rs_val, 1),
                    "score":          pattern["score"],
                    "score_pole":     pattern["score_pole"],
                    "score_flag":     pattern["score_flag"],
                    "score_volume":   pattern["score_volume"],
                    "score_tech":     pattern["score_tech"],
                    "score_breakout": pattern["score_breakout"],
                    "score_catalyst": pattern["score_catalyst"],
                }

        equity_curve.append({"date": date, "equity": capital})

    progress(n_days, n_days,
             f"| trades={len(closed_trades)} | ${capital:,.0f}")
    print()
    print(f"  Market green:      {green_days}/{n_days} days ({green_days/n_days*100:.0f}%)")
    print(f"  RS filter pass:    {dbg_rs_pass:,} ticker-days")
    print(f"  HTF patterns:      {dbg_pattern:,} detected")
    print(f"  Score filter fail: {dbg_score_fail:,} rejected (score < {cfg['min_score']})")

    # Force-close remaining at last price
    last_date = trading_days[-1]
    for tk, pos in open_positions.items():
        arr  = ticker_arrays.get(tk)
        if arr is None:
            continue
        didx = get_index(prices[tk], last_date)
        if didx is None:
            continue
        exit_price = arr["closes"][didx]
        if np.isnan(exit_price):
            continue
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
        closed_trades.append({
            "ticker":         tk,
            "entry_date":     pos["entry_date"].date(),
            "exit_date":      last_date.date(),
            "entry_price":    round(pos["entry_price"], 2),
            "exit_price":     round(exit_price, 2),
            "shares":         pos["shares"],
            "pnl":            round(pnl, 2),
            "return_pct":     round(ret * 100, 2),
            "days_held":      (last_date - pos["entry_date"]).days,
            "exit_reason":    "end_of_backtest",
            "pole_gain":      pos["pole_gain"],
            "vol_ratio":      pos["vol_ratio"],
            "rs":             pos["rs"],
            "score":          pos["score"],
            "score_pole":     pos["score_pole"],
            "score_flag":     pos["score_flag"],
            "score_volume":   pos["score_volume"],
            "score_tech":     pos["score_tech"],
            "score_breakout": pos["score_breakout"],
            "score_catalyst": pos["score_catalyst"],
        })

    return pd.DataFrame(closed_trades), pd.DataFrame(equity_curve)


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
def print_and_save_results(trades_df, equity_df, cfg):
    trades_df.to_csv("htf_trade_log.csv", index=False)
    equity_df.to_csv("htf_equity_curve.csv", index=False)

    if trades_df.empty:
        msg = "\n⚠  No trades generated. Check diagnostic output above.\n"
        print(msg)
        with open("htf_summary.txt", "w") as f:
            f.write(msg)
        return

    wins    = trades_df[trades_df["pnl"] > 0]
    loses   = trades_df[trades_df["pnl"] <= 0]
    total   = len(trades_df)
    wr      = len(wins) / total * 100
    avg_win = wins["return_pct"].mean()  if len(wins)  else 0
    avg_los = loses["return_pct"].mean() if len(loses) else 0
    avg_d   = trades_df["days_held"].mean()
    pnl     = trades_df["pnl"].sum()
    final   = cfg["initial_capital"] + pnl
    ret_pct = (final - cfg["initial_capital"]) / cfg["initial_capital"] * 100
    years   = (pd.to_datetime(cfg["end_date"]) -
               pd.to_datetime(cfg["start_date"])).days / 365.25
    pf      = (wins["pnl"].sum() / abs(loses["pnl"].sum())
               if len(loses) and loses["pnl"].sum() != 0 else float("inf"))
    eq      = equity_df["equity"]
    max_dd  = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    avg_score = trades_df["score"].mean()

    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
    yearly = trades_df.groupby("year").agg(
        trades    =("pnl","count"),
        wins      =("pnl", lambda x: (x>0).sum()),
        total_pnl =("pnl","sum"),
        avg_ret   =("return_pct","mean"),
        avg_score =("score","mean"),
    )
    yearly["win_rate"] = (yearly["wins"] / yearly["trades"] * 100).round(1)

    W = "═" * 60
    w = "─" * 60
    lines = [
        W,
        "  LEIF SOREIDE — HIGH TIGHT FLAG BACKTEST (FULL SCORING)",
        f"  {cfg['start_date']}  →  {cfg['end_date']}",
        W,
        f"  {'Initial Capital:':<34} ${cfg['initial_capital']:>12,.0f}",
        f"  {'Final Equity:':<34} ${final:>12,.0f}",
        f"  {'Total Return:':<34} {ret_pct:>11.1f}%",
        f"  {'Max Drawdown:':<34} {max_dd:>11.1f}%",
        f"  {'Profit Factor:':<34} {pf:>12.2f}",
        w,
        f"  {'Total Trades (signals):':<34} {total:>12,}",
        f"  {'Avg Trades / Year:':<34} {total/years:>12.1f}",
        f"  {'Win Rate:':<34} {wr:>11.1f}%",
        f"  {'Avg Win:':<34} {avg_win:>11.1f}%",
        f"  {'Avg Loss:':<34} {avg_los:>11.1f}%",
        f"  {'Avg Days Held:':<34} {avg_d:>11.1f}",
        f"  {'Avg Pattern Score (0-10):':<34} {avg_score:>11.1f}",
        W, "",
        "  SCORE BREAKDOWN (avg by component):",
        f"    Pole      (25%): {trades_df['score_pole'].mean():.1f}/10",
        f"    Flag      (25%): {trades_df['score_flag'].mean():.1f}/10",
        f"    Volume    (20%): {trades_df['score_volume'].mean():.1f}/10",
        f"    Technical (15%): {trades_df['score_tech'].mean():.1f}/10",
        f"    Breakout  (10%): {trades_df['score_breakout'].mean():.1f}/10",
        f"    Catalyst   (5%): {trades_df['score_catalyst'].mean():.1f}/10",
        "", "  EXIT BREAKDOWN:",
    ]
    for reason, count in trades_df["exit_reason"].value_counts().items():
        bar = "█" * int(count / total * 30)
        lines.append(f"    {reason:<22} {count:>5}  {bar}")

    lines += ["", "  YEARLY BREAKDOWN:",
              f"  {'Year':<6} {'Trades':>6} {'Win%':>6} {'Score':>6} {'Avg Ret':>8} {'P&L':>12}",
              f"  {w[:52]}"]
    for yr, row in yearly.iterrows():
        arrow = "▲" if row.total_pnl >= 0 else "▼"
        lines.append(f"  {yr:<6} {int(row.trades):>6} {row.win_rate:>5.0f}%"
                     f"  {row.avg_score:>5.1f}  {row.avg_ret:>6.1f}%"
                     f"  {arrow}${abs(row.total_pnl):>9,.0f}")

    lines += ["", "  TOP 10 TRADES:",
              f"  {'Ticker':<7} {'Entry':<12} {'Exit':<12} {'Ret%':>7} "
              f"{'Days':>5} {'Score':>6} {'Pole%':>6}",
              f"  {w[:58]}"]
    for _, r in trades_df.nlargest(10, "return_pct").iterrows():
        lines.append(
            f"  {r.ticker:<7} {str(r.entry_date):<12} {str(r.exit_date):<12}"
            f"  {r.return_pct:>+6.1f}%  {int(r.days_held):>4}d"
            f"  {r.score:>5.1f}  {r.pole_gain:>5.0f}%"
        )

    lines += ["", "  WORST 5 TRADES:", f"  {w[:58]}"]
    for _, r in trades_df.nsmallest(5, "return_pct").iterrows():
        lines.append(
            f"  {r.ticker:<7} {str(r.entry_date):<12} {str(r.exit_date):<12}"
            f"  {r.return_pct:>+6.1f}%  {int(r.days_held):>4}d"
            f"  {r.score:>5.1f}  [{r.exit_reason}]"
        )
    lines.append(f"\n{W}\n")

    report = "\n".join(lines)
    print(report)
    with open("htf_summary.txt", "w") as f:
        f.write(report)
    print("  ✓ htf_trade_log.csv  (includes all 6 component scores per trade)")
    print("  ✓ htf_equity_curve.csv")
    print("  ✓ htf_summary.txt")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t0      = time.time()
    tickers = [t for t in UNIVERSE if t != "SPY"] + ["SPY"]

    prices = load_data(tickers, CONFIG["start_date"], CONFIG["end_date"],
                       CONFIG["batch_size"], CONFIG["min_trading_days"])

    trades_df, equity_df = run_backtest(prices, CONFIG)
    print_and_save_results(trades_df, equity_df, CONFIG)
    print(f"\n  Total runtime: {(time.time() - t0) / 60:.1f} minutes")
