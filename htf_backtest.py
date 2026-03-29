"""
Leif Soreide - High Tight Flag (HTF) Backtest
==============================================
Strategy Rules:
  1. Flagpole: Stock gains >= 90% in <= 8 weeks (40 trading days)
  2. Flag:     Subsequent consolidation <= 25% drawdown over 3-5 weeks
  3. RS:       63-day return in top 15% of universe
  4. Volume:   Contraction during flag, expansion (1.5x) on breakout day
  5. Entry:    Buy at close of breakout day
  6. Stop:     Below flag low
  7. Exit:     15% trailing stop from peak, or 8-week time stop
  8. Market:   Only trade when SPY > 50-day SMA
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
    "pole_min_gain":       0.90,   # flagpole must gain >= 90%
    "pole_max_days":       40,     # flagpole completes within 40 trading days
    "flag_max_drawdown":   0.25,   # flag pullback <= 25% from pole top
    "flag_min_days":       8,      # flag must last at least 8 days
    "flag_max_days":       25,     # flag must resolve within 25 days
    "rs_percentile":       85,     # stock must be top 15% RS
    "rs_lookback":         63,     # RS measured over 63 trading days (~3 months)
    "volume_ratio":        1.5,    # breakout day volume >= 1.5x 20-day avg
    "min_price":           5.0,    # ignore sub-$5 stocks
    "trailing_stop_pct":   0.15,   # 15% trailing stop from peak close
    "max_hold_days":       40,     # max ~8 weeks in a trade
    "start_date":          "2010-01-01",
    "end_date":            "2024-12-31",
    "initial_capital":     100_000,
    "risk_per_trade":      0.02,   # risk 2% of capital per trade
    "max_concurrent":      5,      # max open positions at once
    "batch_size":          50,
    "min_trading_days":    200,
}

# ─────────────────────────────────────────────
# UNIVERSE — 130 liquid, high-momentum stocks
# ─────────────────────────────────────────────
UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","AVGO","ORCL",
    "LRCX","KLAC","AMAT","MRVL","MPWR","ON","ACLS","ONTO","RMBS","SLAB",
    "CRWD","PANW","FTNT","SNOW","DDOG","NET","ZS","ANET","NOW","WDAY",
    "INTU","ADBE","CRM","VEEV","HUBS","BILL","MDB","GTLB","ESTC","DOMO",
    "MRNA","BNTX","LLY","NVO","REGN","VRTX","ABBV","IDXX","DXCM","PODD",
    "ISRG","EXEL","HALO","DVAX","TMDX","RGEN","ACAD","VKTX","PCVX","RXRX",
    "LULU","CELH","ELF","MNST","ORLY","DECK","ONON","CROX","WING","CAVA",
    "SFM","DUOL","MEDP","BOOT","CHWY","RH","FIVE","CVNA","DRVN","SHAK",
    "IBKR","COIN","MARA","RIOT","HOOD","AFRM","SOFI","PYPL","UPST","GDOT",
    "GNRC","TDG","HEI","AXON","ODFL","SAIA","ENPH","FSLR","SEDG","PLUG",
    "IRTC","GERN","NVCR","FOLD","ARWR","IONS","KRYS","PRAX","BEAM","INVA",
    "MELI","SE","SHOP","BABA","NIO","XPEV","LI","GRAB","DKNG","RBLX",
    "PLTR","ASTS","RKLB","LUNR","CORZ","CLSK","SMCI","HIMS","DOCS","RELY",
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
                        # (field, ticker) layout
                        df = raw.xs(tk, axis=1, level=1)
                    else:
                        # (ticker, field) layout
                        df = raw.xs(tk, axis=1, level=0)

                # Flatten any residual MultiIndex on columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                df.columns = [str(c) for c in df.columns]
                df = df.dropna(how="all")

                if "Close" not in df.columns or "Volume" not in df.columns:
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
                    print(f"\n  [error] SPY failed to load!")

        time.sleep(0.5)

    print(f"\n  Loaded {len(prices)} tickers | {failed} failed/filtered")
    for tk in (["SPY"] + list(prices.keys())[:3]):
        if tk not in prices:
            continue
        df = prices[tk]
        c  = df["Close"]
        print(f"  [diag] {tk}: rows={len(df)}, "
              f"Close type={type(c).__name__}, dtype={c.dtype}, "
              f"last={float(c.iloc[-1]):.2f}")
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
# HTF PATTERN DETECTION  ← KEY FIX HERE
# ─────────────────────────────────────────────
def find_htf_breakout(closes, volumes, n, cfg):
    """
    Detects a High Tight Flag breakout ending on bar n.

    Layout (all indices are into the closes/volumes arrays):

        pole_start ... pole_end   = the flagpole (big run-up)
        pole_end+1 ... n-1        = the flag (consolidation)
        n                         = TODAY (breakout bar)

    Critical fix: the flag segment is closes[pole_end : n]
    i.e. it does NOT include bar n. That way flag_high_v is the
    pre-breakout high, and we can legitimately check closes[n] > flag_high_v.
    """

    # Scan backwards to find where the flag started (= pole top)
    for pole_end in range(n - cfg["flag_min_days"],
                          max(n - cfg["flag_max_days"] - 1, 0), -1):

        flag_days = n - pole_end          # days from pole top to today (excl. today)
        if flag_days < cfg["flag_min_days"] or flag_days > cfg["flag_max_days"]:
            continue

        # Flag segment: pole_end .. n-1  (does NOT include today)
        flag_seg = closes[pole_end : n]
        flag_seg = flag_seg[~np.isnan(flag_seg)]
        if len(flag_seg) < 2:
            continue

        flag_high_v = float(flag_seg.max())
        flag_low_v  = float(flag_seg.min())

        if flag_high_v <= 0:
            continue

        # Today must close ABOVE the flag high (the breakout)
        today_close = closes[n]
        if np.isnan(today_close) or today_close <= flag_high_v:
            continue

        # Flag drawdown must be within limits
        drawdown = (flag_high_v - flag_low_v) / flag_high_v
        if drawdown > cfg["flag_max_drawdown"]:
            continue

        # Now find the pole: a run from some pole_start up to pole_end
        # pole_end close IS the top of the pole = flag_high_v (approx)
        pole_high = flag_high_v  # top of pole = top of flag consolidation zone

        for pole_start in range(pole_end - 1,
                                max(pole_end - cfg["pole_max_days"] - 1, 0), -1):
            pole_low = closes[pole_start]
            if np.isnan(pole_low) or pole_low <= 0:
                continue

            gain = (pole_high - pole_low) / pole_low
            if gain >= cfg["pole_min_gain"]:

                # Volume: today's volume >= ratio x 20-day avg (pre-today)
                vol_slice = volumes[max(n - 20, 0) : n]
                vol_slice = vol_slice[~np.isnan(vol_slice)]
                if len(vol_slice) == 0:
                    break
                vol_avg = float(vol_slice.mean())
                if vol_avg == 0:
                    break

                today_vol = volumes[n]
                if np.isnan(today_vol):
                    break

                vol_ratio = today_vol / vol_avg
                if vol_ratio < cfg["volume_ratio"]:
                    break   # volume too low — skip this pole_end entirely

                return {
                    "pole_gain":     round(gain * 100, 1),
                    "pole_days":     pole_end - pole_start,
                    "flag_days":     flag_days,
                    "flag_drawdown": round(drawdown * 100, 1),
                    "flag_low":      flag_low_v,
                    "flag_high":     flag_high_v,
                    "vol_ratio":     round(vol_ratio, 2),
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

    spy_arrays = {
        "index":  spy_df.index,
        "closes": np.array(spy_df["Close"].values, dtype=float),
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

    for day_num, date in enumerate(trading_days):

        if day_num % 50 == 0:
            progress(day_num, n_days,
                     f"| open={len(open_positions)} "
                     f"| trades={len(closed_trades)} "
                     f"| ${capital:,.0f}")

        # ── Exits ─────────────────────────────────────────────────────
        to_close = []
        for tk, pos in open_positions.items():
            arr = ticker_arrays.get(tk)
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
                "ticker":      tk,
                "entry_date":  pos["entry_date"].date(),
                "exit_date":   exit_date.date(),
                "entry_price": round(pos["entry_price"], 2),
                "exit_price":  round(exit_price, 2),
                "shares":      pos["shares"],
                "pnl":         round(pnl, 2),
                "return_pct":  round(ret * 100, 2),
                "days_held":   (exit_date - pos["entry_date"]).days,
                "exit_reason": reason,
                "pole_gain":   pos["pole_gain"],
                "vol_ratio":   pos["vol_ratio"],
                "rs":          pos["rs"],
            })

        # ── Entries ───────────────────────────────────────────────────
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

                pattern = find_htf_breakout(arr["closes"], arr["volumes"], didx, cfg)
                if pattern is None:
                    continue

                dbg_pattern += 1

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
                }

        equity_curve.append({"date": date, "equity": capital})

    progress(n_days, n_days,
             f"| trades={len(closed_trades)} | ${capital:,.0f}")
    print()
    print(f"  Market green:    {green_days}/{n_days} days ({green_days/n_days*100:.0f}%)")
    print(f"  RS filter pass:  {dbg_rs_pass:,} ticker-days")
    print(f"  HTF patterns:    {dbg_pattern:,} detected")

    # Force-close remaining at last price
    last_date = trading_days[-1]
    for tk, pos in open_positions.items():
        arr = ticker_arrays.get(tk)
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
            "ticker":      tk,
            "entry_date":  pos["entry_date"].date(),
            "exit_date":   last_date.date(),
            "entry_price": round(pos["entry_price"], 2),
            "exit_price":  round(exit_price, 2),
            "shares":      pos["shares"],
            "pnl":         round(pnl, 2),
            "return_pct":  round(ret * 100, 2),
            "days_held":   (last_date - pos["entry_date"]).days,
            "exit_reason": "end_of_backtest",
            "pole_gain":   pos["pole_gain"],
            "vol_ratio":   pos["vol_ratio"],
            "rs":          pos["rs"],
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

    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
    yearly = trades_df.groupby("year").agg(
        trades    =("pnl","count"),
        wins      =("pnl", lambda x: (x>0).sum()),
        total_pnl =("pnl","sum"),
        avg_ret   =("return_pct","mean"),
    )
    yearly["win_rate"] = (yearly["wins"] / yearly["trades"] * 100).round(1)

    W = "═" * 58
    w = "─" * 58
    lines = [
        W,
        "  LEIF SOREIDE — HIGH TIGHT FLAG BACKTEST",
        f"  {cfg['start_date']}  →  {cfg['end_date']}",
        W,
        f"  {'Initial Capital:':<32} ${cfg['initial_capital']:>12,.0f}",
        f"  {'Final Equity:':<32} ${final:>12,.0f}",
        f"  {'Total Return:':<32} {ret_pct:>11.1f}%",
        f"  {'Max Drawdown:':<32} {max_dd:>11.1f}%",
        f"  {'Profit Factor:':<32} {pf:>12.2f}",
        w,
        f"  {'Total Trades (signals):':<32} {total:>12,}",
        f"  {'Avg Trades / Year:':<32} {total/years:>12.1f}",
        f"  {'Win Rate:':<32} {wr:>11.1f}%",
        f"  {'Avg Win:':<32} {avg_win:>11.1f}%",
        f"  {'Avg Loss:':<32} {avg_los:>11.1f}%",
        f"  {'Avg Days Held:':<32} {avg_d:>11.1f}",
        W, "",
        "  EXIT BREAKDOWN:",
    ]
    for reason, count in trades_df["exit_reason"].value_counts().items():
        bar = "█" * int(count / total * 30)
        lines.append(f"    {reason:<22} {count:>5}  {bar}")

    lines += ["", "  YEARLY BREAKDOWN:",
              f"  {'Year':<6} {'Trades':>6} {'Win%':>6} {'Avg Ret':>8} {'P&L':>12}",
              f"  {w[:44]}"]
    for yr, row in yearly.iterrows():
        arrow = "▲" if row.total_pnl >= 0 else "▼"
        lines.append(f"  {yr:<6} {int(row.trades):>6} {row.win_rate:>5.0f}%"
                     f"  {row.avg_ret:>6.1f}%  {arrow}${abs(row.total_pnl):>9,.0f}")

    lines += ["", "  TOP 10 TRADES:",
              f"  {'Ticker':<7} {'Entry':<12} {'Exit':<12} {'Ret%':>7} {'Days':>5} {'Pole%':>6}",
              f"  {w[:52]}"]
    for _, r in trades_df.nlargest(10, "return_pct").iterrows():
        lines.append(f"  {r.ticker:<7} {str(r.entry_date):<12} {str(r.exit_date):<12}"
                     f"  {r.return_pct:>+6.1f}%  {int(r.days_held):>4}d  {r.pole_gain:>5.0f}%")

    lines += ["", "  WORST 5 TRADES:", f"  {w[:52]}"]
    for _, r in trades_df.nsmallest(5, "return_pct").iterrows():
        lines.append(f"  {r.ticker:<7} {str(r.entry_date):<12} {str(r.exit_date):<12}"
                     f"  {r.return_pct:>+6.1f}%  {int(r.days_held):>4}d  [{r.exit_reason}]")
    lines.append(f"\n{W}\n")

    report = "\n".join(lines)
    print(report)
    with open("htf_summary.txt", "w") as f:
        f.write(report)
    print("  ✓ htf_trade_log.csv")
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
