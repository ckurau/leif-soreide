"""
Leif Soreide - High Tight Flag (HTF) Backtest
==============================================
Strategy Rules:
  1. Flagpole: Stock gains >= 90% in <= 8 weeks (40 trading days)
  2. Flag:     Subsequent consolidation <= 25% drawdown over 3-5 weeks (15-25 trading days)
  3. RS:       Stock's 63-day return must be in top 10% of all stocks in universe (RS >= 90)
  4. Volume:   Flag forms on contracting volume; breakout day volume >= 1.5x 20-day avg
  5. Entry:    Buy at close of breakout day (flag high exceeded)
  6. Stop:     Below flag low (exit if price closes below flag low)
  7. Target:   Trailing stop of 15% from highest close, or max 8-week hold
  8. Market:   Only trade when SPY is above its 50-day SMA (green market)

Universe: Full NYSE + NASDAQ (~6,000-8,000 tickers), fetched dynamically
NOTE: yfinance still has survivorship bias (no delisted stocks). For fully
      accurate results, use Norgate Data (~$270/yr).
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
import time
import sys

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    # HTF Pattern Rules
    "pole_min_gain":        0.90,   # >= 90% gain for flagpole
    "pole_max_days":        40,     # <= 40 trading days for pole
    "flag_max_drawdown":    0.25,   # <= 25% pullback in flag
    "flag_min_days":        8,      # min days consolidating
    "flag_max_days":        25,     # max days consolidating (~5 weeks)

    # Filters
    "rs_percentile":        85,     # RS rating >= 85 (top 15%)
    "rs_lookback":          63,     # ~3 months for RS calculation
    "volume_ratio":         1.5,    # breakout volume >= 1.5x 20-day avg
    "min_price":            5.0,    # ignore penny stocks
    "min_avg_dollar_vol":   500_000,# min avg daily dollar volume (liquidity filter)

    # Trade Management
    "trailing_stop_pct":    0.15,   # 15% trailing stop from peak
    "max_hold_days":        40,     # max 8 weeks in trade

    # Backtest Settings
    "start_date":           "2004-01-01",
    "end_date":             "2024-12-31",
    "initial_capital":      100_000,
    "risk_per_trade":       0.02,   # 2% of capital per trade
    "max_concurrent":       5,      # max 5 open positions

    # Download Settings
    "batch_size":           100,    # tickers per yfinance batch
    "min_trading_days":     200,    # drop tickers with less history than this
}

# ─────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────
def progress(current, total, label="", bar_width=35):
    pct  = current / max(total, 1)
    done = int(bar_width * pct)
    bar  = "█" * done + "░" * (bar_width - done)
    sys.stdout.write(f"\r  [{bar}] {current}/{total} {label}   ")
    sys.stdout.flush()
    if current >= total:
        print()

# ─────────────────────────────────────────────
# UNIVERSE: FULL NYSE + NASDAQ
# ─────────────────────────────────────────────
def get_full_universe():
    """
    Fetch all NYSE and NASDAQ tickers from NASDAQ's public data feed.
    Filters out warrants, units, rights, preferred shares, and ETFs.
    """
    print("Fetching full NYSE + NASDAQ universe...")
    tickers = set()

    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                lines = resp.read().decode("utf-8").splitlines()

            for line in lines[1:]:  # skip header row
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                tk = parts[0].strip()

                # Skip blank, test issues, warrants, units, rights, preferred
                if not tk:
                    continue
                if any(c in tk for c in ["$", "^", ".", "+"]):
                    continue
                if len(tk) > 5:
                    continue

                # Skip ETFs listed in otherlisted.txt (column index 4 = ETF flag)
                if "otherlisted" in url and len(parts) > 4 and parts[4].strip() == "Y":
                    continue

                tickers.add(tk)

        except Exception as e:
            print(f"  Warning: could not fetch {url}: {e}")

    tickers.add("SPY")  # always include for market filter
    result = sorted(tickers)
    print(f"  Found {len(result):,} tickers.\n")
    return result


# ─────────────────────────────────────────────
# DATA LOADING — BATCHED WITH PROGRESS
# ─────────────────────────────────────────────
def load_data(tickers, start, end, batch_size=100, min_days=200):
    """
    Download price data in batches. Filters by minimum history and liquidity.
    Expect this to take 20-40 minutes for the full universe.
    """
    prices    = {}
    n         = len(tickers)
    n_batches = (n + batch_size - 1) // batch_size
    failed    = 0

    print(f"Downloading {n:,} tickers in {n_batches} batches of {batch_size}.")
    print("This will take 20-40 minutes. Grab a coffee ☕\n")

    for i in range(n_batches):
        batch = tickers[i * batch_size : (i + 1) * batch_size]
        progress(i + 1, n_batches,
                 f"| loaded {len(prices):,} | failed/filtered {failed}")

        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=30,
            )
        except Exception:
            failed += len(batch)
            time.sleep(1)
            continue

        for tk in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                else:
                    df = raw.xs(tk, axis=1, level=1).dropna(how="all")

                if len(df) < min_days:
                    failed += 1
                    continue

                # Liquidity filter: skip stocks with tiny average dollar volume
                avg_vol   = df["Volume"].tail(60).mean()
                avg_price = df["Close"].tail(60).mean()
                if avg_vol * avg_price < CONFIG["min_avg_dollar_vol"]:
                    failed += 1
                    continue

                prices[tk] = df

            except Exception:
                failed += 1

        time.sleep(0.3)  # polite delay between batches

    print(f"\n  ✓ Loaded {len(prices):,} usable tickers ({failed} failed/filtered)\n")
    return prices


# ─────────────────────────────────────────────
# RELATIVE STRENGTH
# ─────────────────────────────────────────────
def calc_rs_ratings(prices, date, lookback=63):
    """Percentile rank of 63-day returns across all tickers."""
    returns = {}
    for tk, df in prices.items():
        if tk == "SPY":
            continue
        try:
            idx = df.index.get_loc(date, method="ffill")
            if idx < lookback:
                continue
            p0 = df["Close"].iloc[idx - lookback]
            p1 = df["Close"].iloc[idx]
            if p0 > 0:
                returns[tk] = (p1 - p0) / p0
        except Exception:
            pass
    if not returns:
        return {}
    series = pd.Series(returns)
    return (series.rank(pct=True) * 100).to_dict()


# ─────────────────────────────────────────────
# MARKET FILTER
# ─────────────────────────────────────────────
def market_is_green(spy_df, date):
    """True if SPY is above its 50-day SMA."""
    try:
        idx = spy_df.index.get_loc(date, method="ffill")
        if idx < 50:
            return False
        sma50 = spy_df["Close"].iloc[idx - 50:idx].mean()
        return spy_df["Close"].iloc[idx] > sma50
    except Exception:
        return False


# ─────────────────────────────────────────────
# HTF PATTERN DETECTION
# ─────────────────────────────────────────────
def find_htf_breakout(df, date_idx, cfg):
    """
    Scans backwards from date_idx for a valid HTF pattern:
      - Pole:  close-to-close gain >= pole_min_gain in <= pole_max_days
      - Flag:  consolidation <= flag_max_drawdown over flag_min/max_days
      - Break: today's close > flag high on expanded volume
    Returns a dict of pattern details, or None.
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

            # Valid pole — now validate the flag
            flag_days = n - flag_end
            if flag_days < cfg["flag_min_days"] or flag_days > cfg["flag_max_days"]:
                continue

            seg          = closes[flag_end : n + 1]
            flag_high_v  = seg.max()
            flag_low_v   = seg.min()
            drawdown     = (flag_high_v - flag_low_v) / flag_high_v
            if drawdown > cfg["flag_max_drawdown"]:
                continue

            # Breakout: today must close above flag high
            if closes[n] <= flag_high_v:
                continue

            # Volume expansion on breakout
            vol_avg = volumes[max(n - 20, 0):n].mean()
            if vol_avg == 0:
                continue
            vol_ratio = volumes[n] / vol_avg
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
        print("ERROR: SPY data not loaded.")
        return pd.DataFrame(), pd.DataFrame()

    tickers       = [t for t in prices if t != "SPY"]
    all_dates     = spy_df.index
    start, end    = cfg["start_date"], cfg["end_date"]
    trading_days  = [d for d in all_dates if start <= str(d.date()) <= end]
    n_days        = len(trading_days)

    capital        = cfg["initial_capital"]
    open_positions = {}
    closed_trades  = []
    equity_curve   = []
    rs_cache       = {}

    print(f"Running backtest: {n_days:,} trading days | {len(tickers):,} stocks\n")

    for day_num, date in enumerate(trading_days):

        if day_num % 50 == 0:
            progress(day_num, n_days,
                     f"| open={len(open_positions)} "
                     f"| trades={len(closed_trades)} "
                     f"| equity=${capital:,.0f}")

        # ── Manage exits ───────────────────────────────────────────────
        to_close = []
        for tk, pos in open_positions.items():
            df = prices.get(tk)
            if df is None or date not in df.index:
                continue
            price       = float(df.loc[date, "Close"])
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
            pos    = open_positions.pop(tk)
            pnl    = (exit_price - pos["entry_price"]) * pos["shares"]
            ret    = (exit_price - pos["entry_price"]) / pos["entry_price"]
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
        if (len(open_positions) < cfg["max_concurrent"]
                and market_is_green(spy_df, date)):

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

                price = float(df.loc[date, "Close"])
                if price < cfg["min_price"]:
                    continue
                if rs_cache.get(tk, 0) < cfg["rs_percentile"]:
                    continue

                try:
                    didx = df.index.get_loc(date, method="ffill")
                except Exception:
                    continue
                if didx < 80:
                    continue

                pattern = find_htf_breakout(df, didx, cfg)
                if pattern is None:
                    continue

                # Risk-based position sizing
                risk_per_share = price - pattern["flag_low"]
                if risk_per_share <= 0:
                    continue
                shares = int((capital * cfg["risk_per_trade"]) / risk_per_share)
                if shares <= 0:
                    continue

                # Cap at 25% of capital
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

    # Final progress tick
    progress(n_days, n_days,
             f"| trades={len(closed_trades)} | equity=${capital:,.0f}")
    print()

    # Force-close any still-open positions at last price
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
def print_results(trades_df, equity_df, cfg):
    if trades_df.empty:
        print("\nNo trades generated.")
        print("Try lowering rs_percentile, pole_min_gain, or volume_ratio in CONFIG.")
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
        trades    =("pnl","count"),
        wins      =("pnl", lambda x: (x>0).sum()),
        total_pnl =("pnl","sum"),
        avg_ret   =("return_pct","mean"),
    )
    yearly["win_rate"] = (yearly["wins"] / yearly["trades"] * 100).round(1)

    W = "═" * 58
    w = "─" * 58
    print(f"\n{W}")
    print(f"  LEIF SOREIDE — HIGH TIGHT FLAG BACKTEST")
    print(f"  {cfg['start_date']}  →  {cfg['end_date']}")
    print(W)
    print(f"  {'Initial Capital:':<32} ${cfg['initial_capital']:>12,.0f}")
    print(f"  {'Final Equity:':<32} ${final:>12,.0f}")
    print(f"  {'Total Return:':<32} {ret_pct:>11.1f}%")
    print(f"  {'Max Drawdown:':<32} {max_dd:>11.1f}%")
    print(f"  {'Profit Factor:':<32} {pf:>12.2f}")
    print(w)
    print(f"  {'Total Trades (signals):':<32} {total:>12,}")
    print(f"  {'Avg Trades / Year:':<32} {total/years:>12.1f}")
    print(f"  {'Win Rate:':<32} {wr:>11.1f}%")
    print(f"  {'Avg Win:':<32} {avg_win:>11.1f}%")
    print(f"  {'Avg Loss:':<32} {avg_los:>11.1f}%")
    print(f"  {'Avg Days Held:':<32} {avg_d:>11.1f}")
    print(W)

    print(f"\n  EXIT BREAKDOWN:")
    for reason, count in trades_df["exit_reason"].value_counts().items():
        bar = "█" * int(count / total * 30)
        print(f"    {reason:<22} {count:>5}  {bar}")

    print(f"\n  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Trades':>6} {'Win%':>6} {'Avg Ret':>8} {'P&L':>12}")
    print(f"  {w[:44]}")
    for yr, row in yearly.iterrows():
        arrow = "▲" if row.total_pnl >= 0 else "▼"
        print(f"  {yr:<6} {int(row.trades):>6} {row.win_rate:>5.0f}%"
              f"  {row.avg_ret:>6.1f}%  {arrow}${abs(row.total_pnl):>9,.0f}")

    print(f"\n  TOP 10 TRADES:")
    print(f"  {'Ticker':<7} {'Entry':<12} {'Exit':<12} {'Ret%':>7} {'Days':>5} {'Pole%':>6}")
    print(f"  {w[:50]}")
    for _, r in trades_df.nlargest(10, "return_pct").iterrows():
        print(f"  {r.ticker:<7} {str(r.entry_date):<12} {str(r.exit_date):<12}"
              f"  {r.return_pct:>+6.1f}%  {int(r.days_held):>4}d  {r.pole_gain:>5.0f}%")

    print(f"\n  WORST 5 TRADES:")
    print(f"  {w[:50]}")
    for _, r in trades_df.nsmallest(5, "return_pct").iterrows():
        print(f"  {r.ticker:<7} {str(r.entry_date):<12} {str(r.exit_date):<12}"
              f"  {r.return_pct:>+6.1f}%  {int(r.days_held):>4}d  [{r.exit_reason}]")

    print(f"\n{W}\n")


def save_results(trades_df, equity_df):
    trades_df.to_csv("htf_trade_log.csv", index=False)
    equity_df.to_csv("htf_equity_curve.csv", index=False)
    print("  ✓ Saved: htf_trade_log.csv")
    print("  ✓ Saved: htf_equity_curve.csv")
    print("  (Right-click files in the left panel → Download)\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    universe          = get_full_universe()
    prices            = load_data(universe, CONFIG["start_date"], CONFIG["end_date"],
                                  CONFIG["batch_size"], CONFIG["min_trading_days"])
    trades_df, eq_df  = run_backtest(prices, CONFIG)
    print_results(trades_df, eq_df, CONFIG)
    save_results(trades_df, eq_df)

    print(f"  Total runtime: {(time.time()-t0)/60:.1f} minutes")
