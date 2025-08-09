# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import timedelta
from yahooquery import Ticker

# -------------------------------------
# THEME: "Robinhood vibe" but purple
# -------------------------------------
PURPLE = "#7C3AED"      # primary accent
PURPLE_DARK = "#5B21B6" # hover/active
BG_DARK = "#0F1115"
TEXT_LIGHT = "#E5E7EB"

st.set_page_config(page_title="EMA Breakthrough Scanner", layout="wide")

st.markdown(
    f"""
    <style>
      :root {{
        --primary-color: {PURPLE};
        --text-color: {TEXT_LIGHT};
        --bg-color: {BG_DARK};
      }}
      html, body, [data-testid="stApp"] {{
        background: var(--bg-color);
        color: var(--text-color);
      }}
      /* Titles */
      h1, h2, h3, h4, h5, h6 {{
        letter-spacing: 0.2px;
      }}
      .title-hero {{
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        color: var(--text-color);
      }}
      .subtitle {{
        opacity: 0.8;
        margin-bottom: 1rem;
      }}
      /* Buttons */
      .stButton > button {{
        background: {PURPLE};
        color: white;
        border-radius: 10px;
        border: 1px solid {PURPLE_DARK};
        transition: all .15s ease;
      }}
      .stButton > button:hover {{
        background: {PURPLE_DARK};
        border-color: {PURPLE_DARK};
      }}
      /* Inputs */
      .stTextInput > div > div > input {{
        background: #1A1D24;
        color: var(--text-color);
        border-radius: 10px;
        border: 1px solid #252a33;
      }}
      /* Sidebar */
      section[data-testid="stSidebar"] {{
        background: #0b0d12;
        border-right: 1px solid #1c2030;
      }}
      /* Dataframe tweaks */
      div[data-testid="stDataFrame"] div[role="table"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #23283b;
      }}
      /* Badges */
      .badge {{
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.8rem;
      }}
      .badge-buy {{ background: {PURPLE}; color: white; }}
      .badge-none {{ background: #2a2f40; color: #b6b9c6; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title-hero">📈 EMA Breakthrough + Sentiment Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Purple-flavored, Robinhood-ish layout • Daily OHLC via yahooquery</div>', unsafe_allow_html=True)

# -------------------------------------
# Sidebar Filters
# -------------------------------------
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

# -------------------------------------
# Data Functions
# -------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlc(ticker: str, days: int) -> pd.DataFrame:
    """
    Daily OHLCV via yahooquery, adjusted, robust across weekends/holidays.
    Returns columns used downstream: Close, High, Low, Volume, EMA40, ATR14, ATRpct, VolSMA20, CrossUp, Delta, DeltaPct
    """
    try:
        tq = Ticker(ticker)
        df = tq.history(period=f"{days}d", interval="1d", adj_ohlc=True)
        if isinstance(df, pd.Series) or df is None or len(df) == 0:
            return pd.DataFrame()

        # Handle MultiIndex (symbol, date)
        if isinstance(df.index, pd.MultiIndex) and 'symbol' in df.index.names:
            try:
                df = df.xs(ticker, level='symbol')
            except Exception:
                df = df.loc[df.index.get_level_values('symbol') == ticker]

        df = df.copy()
        df.rename(columns={
            'close':'Close','high':'High','low':'Low','open':'Open','volume':'Volume','adjclose':'AdjClose'
        }, inplace=True)

        needed = {"Close","High","Low","Volume"}
        if not needed.issubset(df.columns):
            return pd.DataFrame()

        df = df.loc[~df["Close"].isna()].tail(days)

        # Indicators
        df["EMA40"] = df["Close"].ewm(span=40, adjust=False).mean()
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift(1)).abs()
        lc = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["ATR14"] = tr.rolling(14).mean()
        df["ATRpct"] = (df["ATR14"] / df["Close"]) * 100
        df["VolSMA20"] = df["Volume"].rolling(20).mean()

        df["above"] = df["Close"] > df["EMA40"]
        df["above_prev"] = df["above"].shift(1)
        df["CrossUp"] = df["above"] & (~df["above_prev"])

        df["Delta"] = df["Close"] - df["EMA40"]
        df["DeltaPct"] = (df["Delta"] / df["EMA40"]) * 100
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

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
    if df.empty:
        return False
    valid = apply_filters(df)
    sig_dates = df.index[valid]
    if sig_dates.empty:
        return False
    last_date = sig_dates[-1]
    today = df.index[-1]
    if last_date != today:
        return False
    if cooldown_days <= 0:
        return True
    prior = sig_dates[:-1]
    if len(prior) == 0:
        return True
    return (today - prior[-1]).days > cooldown_days

# -------------------------------------
# Scan loop
# -------------------------------------
rows = []
per_ticker_signal_logs = {}

for t in tickers:
    df = fetch_ohlc(t, lookback_days)
    if df.empty:
        st.warning(f"No valid data for {t}")
        continue

    signal_today = last_signal_with_cooldown(df, cooldown_days)
    latest = df.iloc[-1]

    delta_pct = float(latest["DeltaPct"])
    atr_pct = float(latest["ATRpct"])
    vol_ok = bool(latest["Volume"] > latest["VolSMA20"]) if not np.isnan(latest["VolSMA20"]) else False

    preview, chatter = fetch_stocktwits_preview(t)

    # Strength score (simple weighted sum)
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

    # filtered signal log
    sig_mask = apply_filters(df)
    log = df.loc[sig_mask, ["Close","EMA40","Delta","DeltaPct","ATR14","ATRpct","Volume","VolSMA20"]].copy()
    log.rename(columns={
        "Delta":"Δ (P-EMA)", "DeltaPct":"Δ%", "ATRpct":"ATR%", "VolSMA20":"VolSMA20"
    }, inplace=True)
    per_ticker_signal_logs[t] = log

# -------------------------------------
# UI: Results, Exports, Chart, Sentiment
# -------------------------------------
if rows:
    df_results = pd.DataFrame(rows).sort_values(
        by=["Signal", "StrengthScore", "Chatter Vol.", "Δ%"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    st.subheader("📊 Scan Results")

    # Make the Signal column look like a purple badge
    df_show = df_results.copy()
    df_show["Signal"] = df_show["Signal"].map(lambda x: f'<span class="badge {"badge-buy" if "BUY" in x else "badge-none"}">{x}</span>')
    st.write(
        df_show[["Ticker","Price","EMA40","Δ (P-EMA)","Δ%","ATR%","Vol>20SMA","Signal","Chatter Vol.","StrengthScore"]]
        .to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    # Export all results
    csv_all = df_results.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Scan CSV", data=csv_all, file_name="ema_breakout_scan.csv", mime="text/csv")

    # Detail view
    choice = st.selectbox("Choose a stock to view chart, signals & sentiment:", df_results["Ticker"].tolist())
    chart_df = fetch_ohlc(choice, lookback_days)

    if chart_df.empty:
        st.info("No chart data available.")
    else:
        st.subheader(f"📉 {choice} Price + EMA40 + Signals")
        fig, ax = plt.subplots(figsize=(12, 4))
        chart_df["Close"].plot(ax=ax, label="Price")
        chart_df["EMA40"].plot(ax=ax, label="EMA40")
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
            st.download_button(
                f"⬇️ Download {choice} Signal Log CSV",
                data=csv_log,
                file_name=f"{choice}_signals.csv",
                mime="text/csv"
            )

        st.subheader(f"💬 Recent Stocktwits for {choice}")
        st.text(df_results.loc[df_results["Ticker"]==choice, "Sentiment Preview"].item())
else:
    st.info("No valid tickers or signals found.")
