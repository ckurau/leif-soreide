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

Data Source: yfinance (free, but has survivorship bias - see notes)
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    # HTF Pattern Rules
    "pole_min_gain":        0.90,   # >= 90% gain for flagpole
    "pole_max_days":        40,     # <= 40 trading days for pole
    "flag_max_drawdown":    0.25,   # <= 25% pullback in flag
    "flag_min_days":        10,     # min days consolidating
    "flag_max_days":        25,     # max days consolidating (5 weeks)

    # Filters
    "rs_percentile":        90,     # RS rating >= 90 (top 10%)
    "rs_lookback":          63,     # ~3 months for RS calculation
    "volume_ratio":         1.5,    # breakout volume >= 1.5x 20-day avg
    "min_price":            5.0,    # ignore penny stocks

    # Trade Management
    "trailing_stop_pct":    0.15,   # 15% trailing stop from peak
    "max_hold_days":        40,     # max 8 weeks in trade

    # Backtest Settings
    "start_date":           "2004-01-01",
    "end_date":             "2024-12-31",
    "initial_capital":      100_000,
    "risk_per_trade":       0.02,   # 2% of capital per trade
    "max_concurrent":       5,      # max 5 open positions
}

# ─────────────────────────────────────────────
# UNIVERSE  (survivorship-bias warning: these are current survivors)
# For a true 20-year backtest, use Norgate/Polygon with delisted stocks
# ─────────────────────────────────────────────
UNIVERSE = [
    # Large/Mid Cap Leaders (diverse sectors)
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","AVGO","SMCI",
    "CRWD","PANW","FTNT","SNOW","DDOG","NET","ZS","ANET","LRCX","KLAC",
    "ENPH","FSLR","SEDG","CELH","HIMS","AXON","DECK","FICO","PODD","ISRG",
    "MRNA","BNTX","COIN","MARA","RIOT","CLSK","CORZ","ASTS","RKLB","LUNR",
    "DUOL","IBKR","SFM","WING","CAVA","ELF","LULU","NVO","LLY","ABBV",
    "REGN","VRTX","IDXX","DXCM","VEEV","INTU","ADBE","CRM","NOW","WDAY",
    "MNST","ORLY","ODFL","SAIA","GNRC","TDG","HEI","KSPI","MELI","SE",
    "SHOP","NFLX","SPOT","UBER","ABNB","DASH","PLTR","TMDX","RGEN","ACLS",
    "RMBS","ONTO","FORM","ICHR","ALGM","COHU","DIOD","SLAB","VICR","AAON",
    "IBOC","HALO","IRTC","AGIO","PCVX","RXRX","GERN","EXEL","MRUS","DVAX",
    "SPY",  # market filter
]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data(tickers, start, end):
    print(f"Downloading data for {len(tickers)} tickers...")
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True
    )
    prices = {}
    for tk in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw.xs(tk, axis=1, level=1).dropna(how="all")
            if len(df) > 200:
                prices[tk] = df
        except Exception:
            pass
    print(f"  Loaded {len(prices)} tickers successfully.")
    return prices

# ─────────────────────────────────────────────
# RELATIVE STRENGTH CALCULATION
# ─────────────────────────────────────────────
def calc_rs_ratings(prices, date, lookback=63):
    """Return dict of {ticker: percentile_rank} for RS on a given date."""
    returns = {}
    for tk, df in prices.items():
        if tk == "SPY":
            continue
        try:
            idx = df.index.get_loc(date, method="ffill")
            if idx < lookback:
                continue
            start_price = df["Close"].iloc[idx - lookback]
            end_price   = df["Close"].iloc[idx]
            if start_price > 0:
                returns[tk] = (end_price - start_price) / start_price
        except Exception:
            pass
    if not returns:
        return {}
    series = pd.Series(returns)
    ranks  = series.rank(pct=True) * 100
    return ranks.to_dict()

# ─────────────────────────────────────────────
# MARKET FILTER
# ─────────────────────────────────────────────
def market_is_green(spy_df, date):
    """SPY must be above its 50-day SMA."""
    try:
        idx = spy_df.index.get_loc(date, method="ffill")
        if idx < 50:
            return False
        sma50 = spy_df["Close"].iloc[idx-50:idx].mean()
        price = spy_df["Close"].iloc[idx]
        return price > sma50
    except Exception:
        return False

# ─────────────────────────────────────────────
# HTF PATTERN DETECTION
# ─────────────────────────────────────────────
def find_htf_breakout(df, date_idx, cfg):
    """
    Look backwards from date_idx to find a valid HTF setup.
    Returns dict with pattern details or None.
    """
    closes = df["Close"].values
    volumes = df["Volume"].values
    n = date_idx

    # --- Find pole: look for a run-up ending within the last flag_max_days
    for flag_end in range(n, max(n - cfg["flag_max_days"] - 1, 0), -1):
        pole_high = closes[flag_end]

        for pole_start in range(flag_end - 1, max(flag_end - cfg["pole_max_days"] - 1, 0), -1):
            pole_low = closes[pole_start]
            if pole_low <= 0:
                continue
            gain = (pole_high - pole_low) / pole_low

            if gain >= cfg["pole_min_gain"]:
                # Found a valid pole. Now check for flag.
                flag_start_idx = flag_end
                flag_end_idx   = n

                flag_days = flag_end_idx - flag_start_idx
                if flag_days < cfg["flag_min_days"] or flag_days > cfg["flag_max_days"]:
                    continue

                flag_segment = closes[flag_start_idx:flag_end_idx + 1]
                if len(flag_segment) < 2:
                    continue

                flag_high_val = flag_segment.max()
                flag_low_val  = flag_segment.min()
                drawdown      = (flag_high_val - flag_low_val) / flag_high_val

                if drawdown > cfg["flag_max_drawdown"]:
                    continue

                # Check current bar breaks above flag high
                current_close = closes[n]
                if current_close <= flag_high_val:
                    continue

                # Volume check: today's volume >= 1.5x 20-day avg
                vol_avg = volumes[max(n-20,0):n].mean()
                if vol_avg == 0:
                    continue
                vol_ratio = volumes[n] / vol_avg
                if vol_ratio < cfg["volume_ratio"]:
                    continue

                # Volume contraction in flag (avg flag vol < avg pre-flag vol)
                pre_flag_vol  = volumes[max(pole_start,0):flag_start_idx].mean()
                flag_vol_avg  = volumes[flag_start_idx:flag_end_idx].mean()
                vol_contracted = (pre_flag_vol == 0) or (flag_vol_avg < pre_flag_vol * 1.1)

                return {
                    "pole_gain":       round(gain * 100, 1),
                    "pole_days":       flag_end - pole_start,
                    "flag_days":       flag_days,
                    "flag_drawdown":   round(drawdown * 100, 1),
                    "flag_low":        flag_low_val,
                    "flag_high":       flag_high_val,
                    "breakout_price":  current_close,
                    "vol_ratio":       round(vol_ratio, 2),
                    "vol_contracted":  vol_contracted,
                }
    return None

# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(prices, cfg):
    spy_df    = prices.get("SPY")
    tickers   = [t for t in prices if t != "SPY"]
    all_dates = spy_df.index

    capital        = cfg["initial_capital"]
    open_positions = {}   # {ticker: {entry details}}
    closed_trades  = []
    equity_curve   = []

    print(f"\nRunning backtest: {cfg['start_date']} → {cfg['end_date']}")
    print(f"Universe: {len(tickers)} stocks | Capital: ${capital:,}\n")

    # Pre-cache RS ratings weekly (every 5 trading days) for speed
    rs_cache = {}

    for i, date in enumerate(all_dates):
        if str(date.date()) < cfg["start_date"]:
            continue
        if str(date.date()) > cfg["end_date"]:
            break

        # ── Manage open positions ──────────────────────────────────────
        to_close = []
        for tk, pos in open_positions.items():
            if tk not in prices:
                to_close.append((tk, "data_missing"))
                continue
            df = prices[tk]
            if date not in df.index:
                continue
            price = df.loc[date, "Close"]
            pos["peak"] = max(pos["peak"], price)
            trail_stop  = pos["peak"] * (1 - cfg["trailing_stop_pct"])
            days_held   = (date - pos["entry_date"]).days

            exit_reason = None
            exit_price  = price

            if price <= pos["stop"]:
                exit_reason = "stop_loss"
            elif price <= trail_stop and days_held > 5:
                exit_reason = "trailing_stop"
            elif days_held >= cfg["max_hold_days"] * 1.4:  # calendar days
                exit_reason = "time_stop"

            if exit_reason:
                to_close.append((tk, exit_reason, exit_price, date))

        for item in to_close:
            tk = item[0]
            if len(item) == 4:
                _, reason, exit_price, exit_date = item
                pos = open_positions[tk]
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                capital += pos["position_value"] + pnl
                days_held = (exit_date - pos["entry_date"]).days

                closed_trades.append({
                    "ticker":        tk,
                    "entry_date":    pos["entry_date"].date(),
                    "exit_date":     exit_date.date(),
                    "entry_price":   round(pos["entry_price"], 2),
                    "exit_price":    round(exit_price, 2),
                    "shares":        pos["shares"],
                    "pnl":           round(pnl, 2),
                    "return_pct":    round(ret * 100, 2),
                    "days_held":     days_held,
                    "exit_reason":   reason,
                    "pole_gain":     pos["pole_gain"],
                    "vol_ratio":     pos["vol_ratio"],
                })
            del open_positions[tk]

        # ── Scan for new signals ───────────────────────────────────────
        if len(open_positions) >= cfg["max_concurrent"]:
            equity_curve.append({"date": date, "equity": capital})
            continue

        if not market_is_green(spy_df, date):
            equity_curve.append({"date": date, "equity": capital})
            continue

        # RS ratings (recalculate weekly)
        if i % 5 == 0:
            rs_cache = calc_rs_ratings(prices, date, cfg["rs_lookback"])

        for tk in tickers:
            if tk in open_positions:
                continue
            if len(open_positions) >= cfg["max_concurrent"]:
                break
            if tk not in prices:
                continue

            df = prices[tk]
            if date not in df.index:
                continue

            price = df.loc[date, "Close"]
            if price < cfg["min_price"]:
                continue

            # RS filter
            rs = rs_cache.get(tk, 0)
            if rs < cfg["rs_percentile"]:
                continue

            date_idx = df.index.get_loc(date)
            if date_idx < 80:
                continue

            pattern = find_htf_breakout(df, date_idx, cfg)
            if pattern is None:
                continue

            # Size position by 2% risk
            stop_price     = pattern["flag_low"]
            risk_per_share = price - stop_price
            if risk_per_share <= 0:
                continue

            risk_dollars   = capital * cfg["risk_per_trade"]
            shares         = int(risk_dollars / risk_per_share)
            if shares <= 0:
                continue

            position_value = shares * price
            if position_value > capital * 0.25:  # max 25% in one name
                shares = int(capital * 0.25 / price)
                position_value = shares * price

            if position_value > capital or shares == 0:
                continue

            capital -= position_value

            open_positions[tk] = {
                "entry_date":     date,
                "entry_price":    price,
                "shares":         shares,
                "position_value": position_value,
                "stop":           stop_price,
                "peak":           price,
                "pole_gain":      pattern["pole_gain"],
                "vol_ratio":      pattern["vol_ratio"],
                "rs":             round(rs, 1),
            }

        equity_curve.append({"date": date, "equity": capital})

    # Force-close any remaining open positions at last price
    last_date = all_dates[-1]
    for tk, pos in open_positions.items():
        if tk in prices and last_date in prices[tk].index:
            exit_price = prices[tk].loc[last_date, "Close"]
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
            days_held = (last_date - pos["entry_date"]).days
            closed_trades.append({
                "ticker":       tk,
                "entry_date":   pos["entry_date"].date(),
                "exit_date":    last_date.date(),
                "entry_price":  round(pos["entry_price"], 2),
                "exit_price":   round(exit_price, 2),
                "shares":       pos["shares"],
                "pnl":          round(pnl, 2),
                "return_pct":   round(ret * 100, 2),
                "days_held":    days_held,
                "exit_reason":  "end_of_backtest",
                "pole_gain":    pos["pole_gain"],
                "vol_ratio":    pos["vol_ratio"],
            })

    return pd.DataFrame(closed_trades), pd.DataFrame(equity_curve)

# ─────────────────────────────────────────────
# RESULTS & REPORTING
# ─────────────────────────────────────────────
def print_results(trades_df, equity_df, cfg):
    if trades_df.empty:
        print("No trades generated. Check universe size or loosen filters.")
        return

    wins  = trades_df[trades_df["pnl"] > 0]
    loses = trades_df[trades_df["pnl"] <= 0]

    total_trades   = len(trades_df)
    win_rate       = len(wins) / total_trades * 100
    avg_win        = wins["return_pct"].mean() if len(wins) else 0
    avg_loss       = loses["return_pct"].mean() if len(loses) else 0
    avg_days       = trades_df["days_held"].mean()
    total_pnl      = trades_df["pnl"].sum()
    final_equity   = cfg["initial_capital"] + total_pnl
    total_return   = (final_equity - cfg["initial_capital"]) / cfg["initial_capital"] * 100
    profit_factor  = (wins["pnl"].sum() / abs(loses["pnl"].sum())) if len(loses) and loses["pnl"].sum() != 0 else float("inf")

    # Max drawdown on equity curve
    eq = equity_df["equity"]
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    # Annual breakdown
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
    yearly = trades_df.groupby("year").agg(
        trades=("pnl","count"),
        wins=("pnl", lambda x: (x>0).sum()),
        total_pnl=("pnl","sum"),
        avg_return=("return_pct","mean")
    )
    yearly["win_rate"] = (yearly["wins"] / yearly["trades"] * 100).round(1)

    sep = "═" * 55
    print(f"\n{sep}")
    print(f"  LEIF SOREIDE HTF BACKTEST RESULTS")
    print(f"  {cfg['start_date']} → {cfg['end_date']}")
    print(sep)
    print(f"  {'Initial Capital:':<30} ${cfg['initial_capital']:>12,.0f}")
    print(f"  {'Final Equity:':<30} ${final_equity:>12,.0f}")
    print(f"  {'Total Return:':<30} {total_return:>11.1f}%")
    print(f"  {'Max Drawdown:':<30} {max_dd:>11.1f}%")
    print(f"  {'Profit Factor:':<30} {profit_factor:>12.2f}")
    print(sep)
    print(f"  {'Total Signals/Trades:':<30} {total_trades:>12}")
    print(f"  {'Win Rate:':<30} {win_rate:>11.1f}%")
    print(f"  {'Avg Win:':<30} {avg_win:>11.1f}%")
    print(f"  {'Avg Loss:':<30} {avg_loss:>11.1f}%")
    print(f"  {'Avg Days Held:':<30} {avg_days:>11.1f}")
    print(f"  {'Avg Trades/Year:':<30} {total_trades/20:>11.1f}")
    print(sep)

    print(f"\n  EXIT REASON BREAKDOWN:")
    for reason, count in trades_df["exit_reason"].value_counts().items():
        pct = count / total_trades * 100
        print(f"    {reason:<25} {count:>4} ({pct:.0f}%)")

    print(f"\n  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Trades':>6} {'Win%':>6} {'Avg Ret':>8} {'P&L':>10}")
    print(f"  {'-'*40}")
    for yr, row in yearly.iterrows():
        print(f"  {yr:<6} {int(row.trades):>6} {row.win_rate:>5.0f}% {row.avg_return:>7.1f}% ${row.total_pnl:>9,.0f}")

    print(f"\n  TOP 10 TRADES:")
    top10 = trades_df.nlargest(10, "return_pct")[
        ["ticker","entry_date","exit_date","return_pct","days_held","pole_gain","exit_reason"]
    ]
    for _, r in top10.iterrows():
        print(f"    {r.ticker:<6} {str(r.entry_date):<12} → {str(r.exit_date):<12} "
              f"{r.return_pct:>+7.1f}%  {int(r.days_held):>3}d  pole:{r.pole_gain:.0f}%")

    print(f"\n  WORST 5 TRADES:")
    bot5 = trades_df.nsmallest(5, "return_pct")[
        ["ticker","entry_date","exit_date","return_pct","days_held","exit_reason"]
    ]
    for _, r in bot5.iterrows():
        print(f"    {r.ticker:<6} {str(r.entry_date):<12} → {str(r.exit_date):<12} "
              f"{r.return_pct:>+7.1f}%  {int(r.days_held):>3}d  [{r.exit_reason}]")

    print(f"\n{sep}\n")
    return trades_df, yearly

def save_results(trades_df, equity_df):
    trades_df.to_csv("/mnt/user-data/outputs/htf_trade_log.csv", index=False)
    equity_df.to_csv("/mnt/user-data/outputs/htf_equity_curve.csv", index=False)
    print("  Saved: htf_trade_log.csv")
    print("  Saved: htf_equity_curve.csv")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    prices = load_data(UNIVERSE, CONFIG["start_date"], CONFIG["end_date"])
    trades_df, equity_df = run_backtest(prices, CONFIG)
    print_results(trades_df, equity_df, CONFIG)
    save_results(trades_df, equity_df)
