# src/news_generator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)


def _is_earnings_day(dt: pd.Timestamp):
    # Simple synthetic quarterly schedule:
    # Feb (Q3), May (Q4), Aug (Q1), Nov (Q2) — use 1st business day of month
    mo = dt.month
    if mo in (2, 5, 8, 11):
        return True
    return False


def _is_dividend_window(dt: pd.Timestamp):
    # Synthetic annual dividend around June 15 ±7 days
    if dt.month == 6 and abs(dt.day - 15) <= 7:
        return True
    return False


def _regime_label(code: int):
    # map numeric regime to human label
    mapping = {
        0: "low-volatility sideways",
        1: "moderate / trend-neutral",
        2: "clear uptrend",
        3: "high-volatility / chaotic"
    }
    return mapping.get(code, "neutral")


def _compose_headline(dt, regime, brent_sig, fx_sig, price_sig, events):
    parts = []
    # regime lead
    if regime == 2:
        parts.append("Deepak Nitrite posts upbeat signals amid market strength")
    elif regime == 0:
        parts.append("Deepak Nitrite holds steady in quiet trading")
    elif regime == 3:
        parts.append("Deepak Nitrite sees volatile session as markets wobble")
    else:
        parts.append("Deepak Nitrite shows mixed cues in recent trading")

    # input-cost / brent
    if brent_sig > 0.12:
        parts.append("rising crude pressure dims margin outlook")
    elif brent_sig < -0.12:
        parts.append("easing crude prices support margin expansion")
    # fx
    if fx_sig > 0.08:
        parts.append("rupee weakness raises import costs")
    elif fx_sig < -0.08:
        parts.append("rupee strength eases forex pressure")

    # event
    if events.get("earnings"):
        parts.append("quarterly results expected to steer sentiment")
    if events.get("dividend"):
        parts.append("dividend announcement draws investor attention")

    # join for a concise headline
    headline = " — ".join(parts[:3])
    return headline[:200]


def _compose_article(dt, regime, brent_ret, fx_ret, price_ret, events, extra_noise=0.0):
    # Build 3-5 sentences mixing signals
    sentences = []

    regime_txt = _regime_label(regime)
    sentences.append(
        f"On {dt.date()}, Deepak Nitrite traded in a {regime_txt} environment."
    )

    # Price sentence
    if price_ret > 0.02:
        sentences.append("Shares climbed on strong demand signals and positive downstream trends.")
    elif price_ret < -0.02:
        sentences.append("Shares fell as selling pressure outpaced any short-term buying interest.")
    else:
        sentences.append("Price action remained subdued with limited directional conviction.")

    # Brent sentence
    if brent_ret > 0.02:
        sentences.append("Rising Brent crude increased feedstock costs, creating margin headwinds.")
    elif brent_ret < -0.02:
        sentences.append("Declining crude eased input-cost pressure and improved operating leverage prospects.")
    else:
        sentences.append("Brent crude movements were muted and unlikely to materially affect margins.")

    # FX sentence
    if fx_ret > 0.02:
        sentences.append("A weaker rupee amplified import costs and added near-term currency concerns.")
    elif fx_ret < -0.02:
        sentences.append("A stronger rupee helped reduce the cost of imported raw materials.")
    else:
        sentences.append("FX fluctuations were modest and did not alter cost dynamics materially.")

    # Events sentence (earnings/dividend)
    if events.get("earnings") and events.get("dividend"):
        sentences.append("Quarterly results and a dividend decision are both on the radar, keeping the stock in focus.")
    elif events.get("earnings"):
        sentences.append("Investors await quarterly results which could provide fresh guidance.")
    elif events.get("dividend"):
        sentences.append("Dividend-related activity is boosting short-term investor interest.")

    # Add tiny editorial color / uncertainty
    tone = np.tanh((price_ret + (-brent_ret) * 0.7 - fx_ret * 0.5) * 5.0)
    if abs(tone) < 0.07:
        sentences.append("Analysts note that near-term outlook remains uncertain and more data is needed.")
    elif tone > 0:
        sentences.append("Analysts view the near-term outlook as constructive.")
    else:
        sentences.append("Analysts remain cautious given mixed indicators.")

    # Join 3-5 sentences, trim to 4 sentences for brevity
    article = " ".join(sentences[:5])
    # add small extra noise phrase
    if extra_noise > 0:
        article += " " + random.choice([
            "Market participants are watching volumes closely.",
            "Trading volumes remain a key watchpoint.",
            "Liquidity conditions could amplify price moves."
        ])
    return article


def _compute_compound(price_ret, brent_ret, fx_ret, regime, events):
    # weights: price 0.5, brent 0.2, fx 0.15, regime 0.15, event boosts
    w_price = 0.5
    w_brent = 0.2
    w_fx = 0.15
    w_reg = 0.15

    # convert regime to directional sign
    reg_sign = 0.0
    if regime == 2:
        reg_sign = 0.6
    elif regime == 0:
        reg_sign = -0.2
    elif regime == 3:
        reg_sign = -0.4
    else:
        reg_sign = 0.0

    base = (
        w_price * np.tanh(price_ret * 10.0) +
        w_brent * (-np.tanh(brent_ret * 8.0)) +
        w_fx * (-np.tanh(fx_ret * 8.0)) +
        w_reg * reg_sign
    )

    # event adjustments
    if events.get("earnings"):
        base += 0.15 * np.sign(price_ret)  # earnings amplify direction
    if events.get("dividend"):
        base += 0.10

    # clip and add small randomness
    base = float(np.clip(base + np.random.normal(0, 0.05), -1.0, 1.0))
    return base


def generate_full_synthetic_articles(price_df, brent_df, usd_df, regimes=None, out_path=None):
    """
    price_df: master price DataFrame with Date, Close
    brent_df, usd_df: DataFrames with Date, Close
    regimes: optional pd.Series aligned with price_df index with ints {0,1,2,3}
    """
    print("Generating enhanced synthetic articles...")

    # align dates
    df = price_df[["Date", "Close"]].copy().sort_values("Date").reset_index(drop=True)
    br = brent_df[["Date", "Close"]].rename(columns={"Close": "brent_close"}).sort_values("Date").reset_index(drop=True)
    fx = usd_df[["Date", "Close"]].rename(columns={"Close": "usd_close"}).sort_values("Date").reset_index(drop=True)

    # merge forward-fill to align values
    df = df.merge(br, on="Date", how="left").merge(fx, on="Date", how="left")
    df["brent_close"].ffill(inplace=True)
    df["usd_close"].ffill(inplace=True)

    # compute returns
    df["price_ret"] = df["Close"].pct_change().fillna(0)
    df["brent_ret"] = df["brent_close"].pct_change().fillna(0)
    df["fx_ret"] = df["usd_close"].pct_change().fillna(0)

    # simple regimes fallback if not provided
    if regimes is None:
        # use quantile-based quick regime like before but make 4 states: 0 low,1 neutral,2 up,3 volatile
        q_low = df["price_ret"].quantile(0.25)
        q_high = df["price_ret"].quantile(0.75)
        vol = df["price_ret"].rolling(20).std().fillna(df["price_ret"].std())
        regimes = []
        for r, v in zip(df["price_ret"], vol):
            if v > vol.quantile(0.85):
                regimes.append(3)
            elif r > q_high:
                regimes.append(2)
            elif r < q_low:
                regimes.append(0)
            else:
                regimes.append(1)
        regimes = pd.Series(regimes, index=df.index)

    records = []
    for idx, row in df.iterrows():
        dt = row["Date"]
        price_ret = float(row["price_ret"])
        brent_ret = float(row["brent_ret"])
        fx_ret = float(row["fx_ret"])
        reg = int(regimes.iloc[idx])

        events = {
            "earnings": _is_earnings_day(pd.to_datetime(dt)),
            "dividend": _is_dividend_window(pd.to_datetime(dt))
        }

        # headline, article, sentiment
        headline = _compose_headline(pd.to_datetime(dt), reg, brent_ret, fx_ret, price_ret, events)
        article = _compose_article(pd.to_datetime(dt), reg, brent_ret, fx_ret, price_ret, events, extra_noise=0.2)
        compound = _compute_compound(price_ret, brent_ret, fx_ret, reg, events)

        records.append({
            "Date": pd.to_datetime(dt),
            "headline": headline,
            "article": article,
            "compound": compound,
            "regime": reg
        })

    out = pd.DataFrame(records)
    if out_path:
        out.sort_values("Date").to_csv(out_path, index=False)
    print("Synthetic articles generated")
    return out
