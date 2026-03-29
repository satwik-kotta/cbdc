import pickle
import numpy as np
import pandas as pd
import nltk
import trafilatura
import feedparser
import requests
import urllib.parse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from newsapi import NewsApiClient
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)

# ==========================
# LOAD MODELS
# ==========================
with open("sentiment_model.pkl","rb") as f:
    sentiment_model = pickle.load(f)

with open("vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

newsapi = NewsApiClient(api_key="67f062a32b7945578c2c69473ac3eff1")

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

query_embedding = semantic_model.encode([
    "central bank digital currency cbdc digital rupee digital yuan enaira jamdex"
])

# ==========================
# QUERIES
# ==========================
COUNTRIES = {
    "Jamaica":["JAMDEX","Jamaica digital currency","CBDC Jamaica"]
}

# ==========================
# GOOGLE URL FIX
# ==========================
def clean_google_url(url):
    try:
        r = requests.get(url, allow_redirects=True, timeout=5)
        return r.url
    except:
        return url

# ==========================
# NEWSAPI
# ==========================
def newsapi_collect(country):
    urls, titles = [], []

    for q in COUNTRIES[country]:
        try:
            res = newsapi.get_everything(q=q, language="en", page_size=100)
            for a in res.get("articles", []):
                urls.append(a["url"])
                titles.append(a["title"])
        except:
            continue

    return urls, titles

# ==========================
# GOOGLE NEWS
# ==========================
def google_news_collect(country):
    urls, titles = [], []

    for q in COUNTRIES[country]:
        query = urllib.parse.quote(q)
        url = f"https://news.google.com/rss/search?q={query}"

        feed = feedparser.parse(url)

        for entry in feed.entries:
            real_url = clean_google_url(entry.link)
            urls.append(real_url)
            titles.append(entry.title)

    return urls, titles

# ==========================
# GDELT FIXED
# ==========================
def gdelt_collect(country):
    urls, titles = [], []

    for q in COUNTRIES[country]:
        try:
            url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={urllib.parse.quote(q)}&mode=ArtList&maxrecords=50&format=json"
            r = requests.get(url, timeout=10)

            if r.status_code != 200:
                continue

            data = r.json()

            for a in data.get("articles", []):
                urls.append(a["url"])
                titles.append(a["title"])

        except:
            continue

    return urls, titles

# ==========================
# COLLECT
# ==========================
def collect_articles(country):

    u1,t1 = newsapi_collect(country)
    u2,t2 = google_news_collect(country)
    u3,t3 = gdelt_collect(country)

    urls = u1+u2+u3
    titles = t1+t2+t3

    df = pd.DataFrame({"url":urls,"title":titles}).drop_duplicates()

    return df.reset_index(drop=True)

# ==========================
# EXTRACT
# ==========================
def extract_text(url):
    try:
        downloaded = trafilatura.fetch_url(url, timeout=10)
        if not downloaded:
            return None

        text = trafilatura.extract(downloaded)

        if not text or len(text) < 50:   # relaxed
            return None

        return text
    except:
        return None

# ==========================
# RELEVANCE
# ==========================
def is_relevant(sentence):
    emb = semantic_model.encode([sentence])
    sim = cosine_similarity(emb, query_embedding)
    return sim[0][0] > 0.25   # relaxed

# ==========================
# SENTIMENT
# ==========================
def score_sentence(sentence):
    X = vectorizer.transform([sentence])
    s = sentiment_model.predict(X)[0]
    return float(max(0, min(1, s)))

# ==========================
# SCORE DOC
# ==========================
def score_document(url):
    text = extract_text(url)

    if not text:
        return None

    sentences = sent_tokenize(text)
    relevant = [s for s in sentences if is_relevant(s)]

    if not relevant:
        return None

    scores = []
    weights = []

    for s in relevant:
        sc = score_sentence(s)
        scores.append(sc)
        weights.append(len(s.split()))

    return np.average(scores, weights=weights)

# ==========================
# ANALYZE
# ==========================
def analyze(country):

    df = collect_articles(country)

    print("Articles collected:", len(df))

    results = []

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(score_document, row["url"]): row for _,row in df.iterrows()}

        for f in as_completed(futures):
            s = f.result()
            if s is not None:
                r = futures[f]
                results.append({
                    "title": r["title"],
                    "url": r["url"],
                    "sentiment": s
                })

    return pd.DataFrame(results)

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    df = analyze("Jamaica")

    if len(df) == 0:
        print("❌ still nothing, but unlikely now")
    else:
        print("✅ DONE")
        print("Articles used:", len(df))
        print("Mean sentiment:", df["sentiment"].mean())

        df.to_csv("output.csv", index=False)