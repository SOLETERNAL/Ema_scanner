import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import timedelta

# -----------------------------
# CONFIG & UI
# -----------------------------
st.set_page_config(page_title="EMA Breakout Scanner", layout="wide")
st.title("📈 EMA Breakthrough + Sentiment Scanner (Upgraded)")

with st.sidebar:
    st.subheader("Filters")
    min_delta_pct = st.slider("Min % above EMA40", 0.0, 5.0, 0.5, 0.1)
    require_vol_above_sma = st.checkbox("Require Volume > SMA20", True)
    min_atr_pct = st.slider("Min ATR% (ATR14 / Close * 100)", 0.0, 10.0, 1.0, 0.1)
    cooldown_days = st.slider("Signal cooldown (days)", 0, 10, 3, 1)
    lookback_days = st.slider("Lookback window (days)", 60, 250, 120, 10)

tickers_input = st.text_input(
    "Enter stock tickers (comma-separated):",
    value="AAPL,TSLA,NVDA,AMZN,MSFT"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# -----------------------------
# DATA FUNCTIONS
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlc(ticker: str, days: int) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or not set(["Close","High","Low","Volume"]).issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()

    # EMA40 (trading-style)
    df["EMA40"] = df["Close"].ewm(span=40, adjust=False).mean()

    # --- ATR(14) ---
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["ATRpct"] = (df["ATR14"] / df["Close"]) * 100

    # Volume SMA20
    df["VolSMA20"] = df["Volume"].rolling(20).mean()

    # Above/Signal
    df["above"] = df["Close"] > df["EMA40"]
    df["above_prev"] = df["above"].shift(1)
    df["CrossUp"] = df["above"] & (~df["above_prev"])  # strict cross up
    df["Delta"] = df["Close"] - df["EMA40"]
    df["DeltaPct"] = (df["Delta"] / df["EMA40"]) * 100
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stocktwits_preview(ticker: str):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return "Stocktwits error", 0
        data = r.json()
        msgs = data.get("messages") or []
        count = len(msgs)
        preview = "\n\n".join(
            f"{m.get('user',{}).get('username','?')}: {str(m.get('body',''))[:80]}..."
            for m in msgs[:5]
        )
        return (preview or "No recent messages"), count
    except Exception:
        return "API error", 0

def apply_filters(df: pd.DataFrame) -> pd.Series:
    cond = df["CrossUp"]
    if min_delta_pct > 0:
        cond &= df["DeltaPct"] >= min_delta_pct
    if require_vol_above_sma:
        cond &= df["Volume"] > df["VolSMA20"]
    if min_atr_pct > 0:
        cond &= df["ATRpct"] >= min_atr_pct
    return cond

def last_signal_with_cooldown(df: pd.DataFrame, cooldown_days: int) -> bool:
    """True if there is a valid signal today that isn't blocked by cooldown from a recent prior signal."""
    if df.empty:
        return False
    valid = apply_filters(df)
    # locate signals
    sig_dates = df.index[valid]
    if sig_dates.empty:
        return False
    last_date = sig_dates[-1]
    today = df.index[-1]
    if last_date != today:
        return False
    if cooldown_days <= 0:
        return True
    # was there a prior signal within cooldown window?
    prior = sig_dates[:-1]
    if len(prior) == 0:
        return True
    if (today - prior[-1]).days <= cooldown_days:
        return False
    return True

# -----------------------------
# SCAN
# -----------------------------
rows = []
per_ticker_signal_logs = {}  # for per-ticker CSV export

for t in tickers:
    df = fetch_ohlc(t, lookback_days)
    if df.empty:
        st.warning(f"No valid data for {t}")
        continue

    # Determine today's signal with cooldown
    signal_today = last_signal_with_cooldown(df, cooldown_days)

    # Rank features
    latest = df.iloc[-1]
    delta_pct = float(latest["DeltaPct"])
    atr_pct = float(latest["ATRpct"])
    vol_ok = bool(latest["Volume"] > latest["VolSMA20"]) if not np.isnan(latest["VolSMA20"]) else False

    # Sentiment preview (single call per ticker)
    preview, chatter = fetch_stocktwits_preview(t)

    # Composite strength score (simple z-ish scaling without importing sklearn)
    # Weight breakout % more than ATR, then chatter volume (log scaled)
    strength = (
        (delta_pct) * 1.2 +
        (atr_pct) * 0.6 +
        (np.log1p(chatter)) * 2.0 +
        (2.0 if vol_ok else 0.0)
    )

    rows.append({
        "Ticker": t,
        "Price": round(float(latest["Close"]), 2),
        "EMA40": round(float(latest["EMA40"]), 2),
        "Δ (P-EMA)": round(float(latest["Delta"]), 2),
        "Δ%": round(delta_pct, 2),
        "ATR%": round(atr_pct, 2),
        "Vol>20SMA": vol_ok,
        "Signal": "✅ BUY" if signal_today else "—",
        "Chatter Vol.": int(chatter),
        "StrengthScore": round(float(strength), 3),
        "Sentiment Preview": preview
    })

    # Build a per-ticker filtered signal log (for export)
    sig_mask = apply_filters(df)
    log = df.loc[sig_mask, ["Close","EMA40","Delta","DeltaPct","ATR14","ATRpct","Volume","VolSMA20"]].copy()
    log.rename(columns={
        "Delta":"Δ (P-EMA)", "DeltaPct":"Δ%", "ATR14":"ATR14", "ATRpct":"ATR%", "VolSMA20":"VolSMA20"
    }, inplace=True)
    per_ticker_signal_logs[t] = log

if rows:
    df_results = pd.DataFrame(rows).sort_values(
        by=["Signal", "StrengthScore", "Chatter Vol.", "Δ%"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    st.subheader("📊 Scan Results")
    st.dataframe(
        df_results[["Ticker","Price","EMA40","Δ (P-EMA)","Δ%","ATR%","Vol>20SMA","Signal","Chatter Vol.","StrengthScore"]],
        use_container_width=True
    )

    # ---- Export all results CSV ----
    csv_all = df_results.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Scan CSV", data=csv_all, file_name="ema_breakout_scan.csv", mime="text/csv")

    # ---- Detail view ----
    choice = st.selectbox("Choose a stock to view chart, signals & sentiment:", df_results["Ticker"].tolist())
    chart_df = fetch_ohlc(choice, lookback_days)

    if chart_df.empty:
        st.info("No chart data available.")
    else:
        st.subheader(f"📉 {choice} Price + EMA40 + Signals")
        fig, ax = plt.subplots(figsize=(12, 4))
        chart_df["Close"].plot(ax=ax, label="Price")
        chart_df["EMA40"].plot(ax=ax, label="EMA40")
        # plot valid buy signals (after filters)
        ok_mask = apply_filters(chart_df)
        buy_pts = chart_df[ok_mask]
        if not buy_pts.empty:
            ax.scatter(buy_pts.index, buy_pts["Close"], label="Filtered BUY", marker="^")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

        st.subheader(f"🗓️ Signal Log for {choice}")
        log = per_ticker_signal_logs.get(choice, pd.DataFrame())
        if log.empty:
            st.write("No filtered signals in lookback window.")
        else:
            st.dataframe(log.tail(50), use_container_width=True)
            csv_log = log.to_csv().encode("utf-8")
            st.download_button(f"⬇️ Download {choice} Signal Log CSV", data=csv_log,
                               file_name=f"{choice}_signals.csv", mime="text/csv")

        st.subheader(f"💬 Recent Stocktwits for {choice}")
        st.text(df_results.loc[df_results["Ticker"]==choice, "Sentiment Preview"].item())
else:
    st.info("No valid tickers or signals found.")
