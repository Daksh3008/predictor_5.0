# src/real_news_fetcher.py

import requests
import feedparser
import pandas as pd
from datetime import datetime
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from pathlib import Path
from .utils import ensure_dir
from rapidfuzz import fuzz


KEYWORDS_STRICT = [
    "deepak nitrite", "deepak nitrate", "deepak nitrtie", "deepak nitrite ltd",
    "deepak nitrite limited", "deepak nitrite share", "deepak nitrite stock",
    "deepak nitrite shares", "deepak nitrite stock price",
    "deepak fertilisers", "deepakntr", "deepak group"
]

KEYWORDS_CONTEXT = [
    "profit", "loss", "revenue", "increase", "decrease", "drop", "rise",
    "results", "quarter", "expansion", "blasts", "malfunction",
    "problems", "q1", "q2", "q3", "q4", "share", "stock",
    "price", "market", "cap", "capitalisation", "plant", "fire",
    "shutdown", "capacity", "board", "dividend", "announcement",
    "nse", "bse", "brent", "crude", "oil", "currency", "rupee"
]

def headline_matches_company(text: str) -> bool:
    """Return True only if headline is likely about Deepak Nitrite."""
    if not isinstance(text, str):
        return False

    t = text.lower()

    # Strict keywords
    for kw in KEYWORDS_STRICT:
        if kw in t:
            return True

    # Fuzzy matching on company name
    if fuzz.partial_ratio(t, "deepak nitrite") >= 75:
        return True

    # Context keywords: must mention "deepak" or match fuzzy
    for kw in KEYWORDS_CONTEXT:
        if kw in t and (
            "deepak" in t or "nitrite" in t or fuzz.partial_ratio(t, "deepak") > 60
        ):
            return True

    return False



def _safe_request(url, timeout=10):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        if r.status_code == 200:
            return r.text
        return None
    except:
        return None


# ---------------------------------------------------------
# Yahoo Finance Headlines (HTML scrape)
# ---------------------------------------------------------
def fetch_yahoo_headlines(ticker="DEEPAKNTR.BO", limit=50):
    print("Fetching Yahoo Finance headlines...")

    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    html = _safe_request(url)
    if not html:
        print("⚠ Yahoo HTML fetch failed.")
        return pd.DataFrame()

    soup = BeautifulSoup(html, "lxml")
    items = soup.select("h3 a")

    rows = []
    for a in items[:limit]:
        headline = a.get_text(strip=True)
        href = a.get("href")
        if not headline or not href:
            continue
        if href.startswith("/"):
            href = "https://finance.yahoo.com" + href

        dt = datetime.utcnow()

        rows.append({"date": dt, "title": headline, "summary": "", "link": href, "source": "Yahoo"})

    print(f"✔ Yahoo: {len(rows)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------
def fetch_google_news(limit=70):
    print("Fetching Google News RSS...")

    q = quote_plus("Deepak Nitrite when:30d")
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en&gl=IN&ceid=IN:en"

    feed = feedparser.parse(rss_url)
    rows = []
    for e in feed.entries[:limit]:
        title = e.get("title", "").strip()
        link = e.get("link", "")
        published = e.get("published", None)

        if not title:
            continue

        try:
            dt = pd.to_datetime(published, utc=True).tz_convert(None)
        except:
            dt = datetime.utcnow()

        rows.append({"date": dt, "title": title, "summary": "", "link": link, "source": "Google RSS"})

    print(f"✔ Google RSS: {len(rows)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# MoneyControl Headlines
# ---------------------------------------------------------
def fetch_moneycontrol(limit=50):
    print("Fetching MoneyControl headlines...")

    url = "https://www.moneycontrol.com/news/tags/deepak-nitrite.html"
    html = _safe_request(url)
    if not html:
        print("⚠ MoneyControl fetch failed.")
        return pd.DataFrame()

    soup = BeautifulSoup(html, "lxml")
    cards = soup.select("div.article") or soup.select("li")

    rows = []
    for c in cards[:limit]:
        a = c.find("a")
        if not a:
            continue
        headline = a.get_text(strip=True)
        href = a.get("href")
        dt = datetime.utcnow()

        rows.append({"date": dt, "title": headline, "summary": "", "link": href, "source": "MoneyControl"})

    print(f"✔ MoneyControl: {len(rows)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# MAIN FETCH + SAVE
# ---------------------------------------------------------
def fetch_and_save_real_news(out_path="data/raw_data/news_real.csv"):
    print("\nFetching REAL news (Yahoo + Google RSS + MoneyControl)...")

    yahoo_df = fetch_yahoo_headlines()
    google_df = fetch_google_news()
    mc_df = fetch_moneycontrol()

    dfs = [df for df in [yahoo_df, google_df, mc_df] if not df.empty]

    if len(dfs) == 0:
        print("⚠ No real news found from ANY source.")
        final = pd.DataFrame(columns=["date", "title", "summary", "link"])
    else:
        final = pd.concat(dfs, ignore_index=True)
        
    # new fuzzy filter here
    before = len(final)
    final = final[final['title'].apply(headline_matches_company)]
    after = len(final)

    print(f"✔ Filtered real news: {after}/{before} relevant articles kept.")

    ensure_dir(str(Path(out_path).parent))
    final.to_csv(out_path, index=False)

    print(f"✔ Saved REAL news → {out_path} | rows={len(final)}")
    return final
