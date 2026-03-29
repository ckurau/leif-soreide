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

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "pole_min_gain":       0.90,
    "pole_max_days":       40,
    "flag_max_drawdown":   0.25,
    "flag_min_days":       8,
    "flag_max_days":       25,
    "rs_percentile":       85,
    "rs_lookback":         63,
    "volume_ratio":        1.5,
    "min_price":           5.0,
    "trailing_stop_pct":   0.15,
    "max_hold_days":       40,
    "start_date":          "2010-01-01",
    "end_date":            "2024-12-31",
    "initial_capital":     100_000,
    "risk_per_trade":      0.02,
    "max_concurrent":      5,
    "batch_size":          50,
    "min_trading_days":    200,
}

# ─────────────────────────────────────────────
# UNIVERSE — 150 liquid stocks across sectors
# Chosen for diversity and likelihood of producing HTF patterns.
# Survivorship bias note: yfinance only has currently-listed stocks.
# ─────────────────────────────────────────────
UNIVERSE = [
    # Mega-cap tech
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","AVGO","ORCL",
    # Semiconductors
    "LRCX","KLAC","AMAT","MRVL","MPWR","ON","ACLS","ONTO","RMBS","SLAB",
    # Software / Cloud
    "CRWD","PANW","FTNT","SNOW","DDOG","NET","ZS","ANET","NOW","WDAY",
    "INTU","ADBE","CRM","VEEV","HUBS","BILL","MDB","GTLB","ESTC","DOMO",
    # Biotech / Pharma
    "MRNA","BNTX","LLY","NVO","REGN","VRTX","ABBV","IDXX","DXCM","PODD",
    "ISRG","EXEL","HALO","DVAX","TMDX","RGEN","ACAD","VKTX","PCVX","RXRX",
    # Consumer / Retail
    "LULU","CELH","ELF","MNST","ORLY","DECK","ONON","CROX","WING","CAVA",
    "SFM","DUOL","MEDP","BOOT","CHWY","PTON","RH","FIVE","CVNA","DRVN",
    # Financials / Fintech
    "IBKR","COIN","MARA","RIOT","HOOD","AFRM","SOFI","PYPL","SQ","UPST",
    # Industrials / Energy
    "GNRC","TDG","HEI","AXON","ODFL","SAIA","ENPH","FSLR","SEDG","PLUG",
    # Biotech small-cap
    "IRTC","GERN","NVCR","FOLD","ARWR","IONS","KRYS","INVA","PRAX","BEAM",
    # International ADRs
    "MELI","SE","SHOP","BABA","NIO","XPEV","LI","GRAB","DKNG","RBLX",
    # Misc high-momentum
    "PLTR","ASTS","RKLB","LUNR","CORZ","CLSK","SMCI","HIMS","DOCS","RELY",
    # Market filter — always keep last
    "SPY",
]

# ─────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────
def progress(current, total, label="", width=35):
    pct  = current / max(total, 1)
    done = int(width * pct)
    bar  = "█" * done + "░" * (width - done)
    sys.stdout.write(f"\r  [{bar}] {current}/{total} {label}   ")
    sys.stdout.flush()
    if current >= total:
        print()

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def get_index(df, date):
    """
    Pandas-version-safe way to get integer index position for a date.
    Replaces deprecated df.index.get_loc(date, method='ffill').
    """
    pos = df.index.searchsorted(date, side="right") - 1
    if pos < 0:
        return None
    return pos

def load_data(tickers, start, end, batch_size=50, min_days=200):
    prices   = {}
    n        = len(tickers)
    n_batches = (n + batch_size - 1) // batch_size
    failed   = 0

    print(f"Downloading {n} tickers in {n_batches} batches...")

    for i in range(n_batches):
        batch = tickers[i * batch_size:(i + 1) * batch_size]
        progress(i + 1, n_batches, f"| loaded {len(prices)} | failed {failed}")

        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=60,
            )
        except Exception as e:
            print(f"\n  Batch {i+1} failed: {e}")
            failed += len(batch)
            time.sleep(2)
            continue

        # Handle single vs multi-ticker download shape
        if isinstance(raw.columns, pd.MultiIndex):
            for tk in batch:
                try:
                    df = raw.xs(tk, axis=1, level=1).dropna(how="all")
                    if len(df) >= min_days:
                        prices[tk] = df.copy()
                    else:
                        failed += 1
                except Exception:
                    failed += 1
        else:
            # Single ticker batch
            tk = batch[0]
            df = raw.dropna(how="all")
            if len(df) >= min_days:
                prices[tk] = df.copy()
            else:
                failed += 1

        time.sleep(0.5)

    print(f"\n  Loaded {len(prices)} tickers ({failed} failed/insufficient data)\n")
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
        p0 = df["Close"].iloc[idx - lookback]
        p1 = df["Close"].iloc[idx]
        if p0 > 0:
            returns[tk] = (p1 - p0) / p0
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
    sma50 = spy_df["Close"].iloc[idx - 50:idx].mean()
    price = spy_df["Close"].iloc[idx]
    return bool(price > sma50)

# ─────────────────────────────────────────────
# HTF PATTERN DETECTION
# ─────────────────────────────────────────────
def find_htf_breakout(df, date_idx, cfg):
    """
    Scans backwards from date_idx for:
      Pole:  close-to-close gain >= pole_min_gain in <= pole_max_days
      Flag:  pullback <= flag_max_drawdown over flag_min..flag_max days
      Break: today's close > flag high on >= volume_ratio x 20d avg volume
    """
    closes  = df["Close"].values
    volumes = df["Volume"].values
    n       = date_idx

    for flag_end in range(n, max(n - cfg["flag_max_days"] - 1, 0), -1):
        pole_high = closes[flag_end]

        for pole_start in range(flag_end - 1,
                                max(flag_end - cfg["pole_max_days"] - 1, 0), -1):
            pole_low = closes[pole_start]
            if pole_low <= 0:
                continue
            gain = (pole_high - pole_low) / pole_low
            if gain < cfg["pole_min_gain"]:
                continue

            flag_days = n - flag_end
            if flag_days < cfg["flag_min_days"] or flag_days > cfg["flag_max_days"]:
                continue

            seg          = closes[flag_end:n + 1]
            flag_high_v  = float(seg.max())
            flag_low_v   = float(seg.min())
            drawdown     = (flag_high_v - flag_low_v) / flag_high_v
            if drawdown > cfg["flag_max_drawdown"]:
                continue

            # Close must exceed flag high today
            if closes[n] <= flag_high_v:
                continue

            # Volume expansion on breakout
            vol_slice = volumes[max(n - 20, 0):n]
            vol_avg   = float(vol_slice.mean()) if len(vol_slice) else 0
            if vol_avg == 0:
                continue
            vol_ratio = float(volumes[n]) / vol_avg
            if vol_ratio < cfg["volume_ratio"]:
                continue

            return {
                "pole_gain":     round(gain * 100, 1),
                "pole_days":     flag_end - pole_start,
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
        print("ERROR: SPY not loaded — cannot apply market filter.")
        return pd.DataFrame(), pd.DataFrame()

    tickers      = [t for t in prices if t != "SPY"]
    start, end   = cfg["start_date"], cfg["end_date"]
    all_dates    = spy_df.index
    trading_days = [d for d in all_dates if start <= str(d.date()) <= end]
    n_days       = len(trading_days)

    capital        = float(cfg["initial_capital"])
    open_positions = {}
    closed_trades  = []
    equity_curve   = []
    rs_cache       = {}

    print(f"Backtesting {n_days:,} trading days across {len(tickers)} stocks...\n")

    for day_num, date in enumerate(trading_days):

        if day_num % 50 == 0:
            progress(day_num, n_days,
                     f"| open={len(open_positions)} "
                     f"| trades={len(closed_trades)} "
                     f"| ${capital:,.0f}")

        # ── Manage exits ───────────────────────────────────────────────
        to_close = []
        for tk, pos in open_positions.items():
            df = prices.get(tk)
            if df is None:
                continue
            idx = get_index(df, date)
            if idx is None:
                continue
            if date not in df.index:
                continue

            price       = float(df["Close"].iloc[idx])
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

        # ── Scan for entries ───────────────────────────────────────────
        slots_free   = len(open_positions) < cfg["max_concurrent"]
        market_green = market_is_green(spy_df, date)

        if slots_free and market_green:
            if day_num % 5 == 0:
                rs_cache = calc_rs_ratings(prices, date, cfg["rs_lookback"])

            for tk in tickers:
                if len(open_positions) >= cfg["max_concurrent"]:
                    break
                if tk in open_positions:
                    continue

                df = prices.get(tk)
                if df is None or date not in df.index:
                    continue

                didx = get_index(df, date)
                if didx is None or didx < 80:
                    continue

                price = float(df["Close"].iloc[didx])
                if price < cfg["min_price"]:
                    continue
                if rs_cache.get(tk, 0) < cfg["rs_percentile"]:
                    continue

                pattern = find_htf_breakout(df, didx, cfg)
                if pattern is None:
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
                    "rs":             round(rs_cache.get(tk, 0), 1),
                }

        equity_curve.append({"date": date, "equity": capital})

    progress(n_days, n_days,
             f"| trades={len(closed_trades)} | ${capital:,.0f}")
    print()

    # Force-close remaining positions at last price
    last_date = trading_days[-1]
    for tk, pos in open_positions.items():
        df = prices.get(tk)
        if df is not None and last_date in df.index:
            exit_price = float(df.loc[last_date, "Close"])
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
# RESULTS & REPORTING
# ─────────────────────────────────────────────
def print_and_save_results(trades_df, equity_df, cfg):
    if trades_df.empty:
        print("\n⚠  No trades generated.")
        print("   Possible causes:")
        print("   - All signals filtered by RS, volume, or market condition")
        print("   - Try lowering pole_min_gain to 0.80 or rs_percentile to 75")
        # Still save empty files so the workflow artifact upload doesn't fail
        trades_df.to_csv("htf_trade_log.csv", index=False)
        equity_df.to_csv("htf_equity_curve.csv", index=False)
        return

    wins  = trades_df[trades_df["pnl"] > 0]
    loses = trades_df[trades_df["pnl"] <= 0]

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
        trades    =("pnl", "count"),
        wins      =("pnl", lambda x: (x > 0).sum()),
        total_pnl =("pnl", "sum"),
        avg_ret   =("return_pct", "mean"),
    )
    yearly["win_rate"] = (yearly["wins"] / yearly["trades"] * 100).round(1)

    W = "═" * 58
    w = "─" * 58

    lines = []
    lines += [
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
        W,
        "",
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

    # Save results
    trades_df.to_csv("htf_trade_log.csv", index=False)
    equity_df.to_csv("htf_equity_curve.csv", index=False)
    with open("htf_summary.txt", "w") as f:
        f.write(report)

    print("  ✓ htf_trade_log.csv")
    print("  ✓ htf_equity_curve.csv")
    print("  ✓ htf_summary.txt")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    tickers = [t for t in UNIVERSE if t != "SPY"] + ["SPY"]

    prices = load_data(
        tickers,
        CONFIG["start_date"],
        CONFIG["end_date"],
        batch_size=CONFIG["batch_size"],
        min_days=CONFIG["min_trading_days"],
    )

    trades_df, equity_df = run_backtest(prices, CONFIG)
    print_and_save_results(trades_df, equity_df, CONFIG)

    print(f"\n  Total runtime: {(time.time() - t0) / 60:.1f} minutes")
