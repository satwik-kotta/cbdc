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
from datetime import datetime

from newsapi import NewsApiClient
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)

# ==========================
# LOAD SENTIMENT MODEL
# ==========================
try:
    with open("sentiment_model.pkl","rb") as f:
        sentiment_model = pickle.load(f)
    logger.info("✓ Sentiment model loaded")
except Exception as e:
    logger.error(f"✗ Failed to load sentiment model: {e}")
    sys.exit(1)

try:
    with open("vectorizer.pkl","rb") as f:
        vectorizer = pickle.load(f)
    logger.info("✓ Vectorizer loaded")
except Exception as e:
    logger.error(f"✗ Failed to load vectorizer: {e}")
    sys.exit(1)

# ==========================
# NEWS API (YOUR KEY ADDED)
# ==========================
try:
    newsapi = NewsApiClient(api_key="67f062a32b7945578c2c69473ac3eff1")
    logger.info("✓ NewsAPI initialized")
except Exception as e:
    logger.error(f"✗ NewsAPI initialization failed: {e}")
    newsapi = None

# ==========================
# SEMANTIC MODEL
# ==========================
try:
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("✓ Semantic model loaded")
except Exception as e:
    logger.error(f"✗ Semantic model failed: {e}")
    sys.exit(1)

query_embedding = semantic_model.encode([
    "central bank digital currency cbdc digital rupee digital yuan enaira jamdex"
], show_progress_bar=False)

# ==========================
# SEARCH QUERIES
# ==========================
COUNTRIES = {

    "India":[
        "digital rupee",
        "RBI digital currency",
        "India central bank currency",
        "Indian blockchain currency"
    ],

    "China":[
        "digital yuan",
        "e-CNY",
        "China blockchain currency",
        "PBOC digital payment"
    ],

    "Nigeria":[
        "Nigeria digital currency",
        "Nigeria blockchain",
        "Nigerian central bank currency",
        "eNaira payment",
        "Nigeria digital payment",
        "CBN blockchain"
    ],

    "Jamaica":[
        "Jamaica fintech",
        "Jamaica blockchain",
        "Caribbean digital currency",
        "Jamaica central bank"
    ],

    "Singapore":[
        "Singapore digital currency",
        "Singapore blockchain",
        "MAS fintech",
        "Singapore monetary authority",
        "Singapore payment system",
        "Asian digital currency"
    ],

    "Australia":[
        "Australia digital currency",
        "Australia blockchain",
        "RBA fintech",
        "Australian payment system",
        "Reserve Bank Australia",
        "Australian monetary policy"
    ],

    "Japan":[
        "digital yen",
        "Japan blockchain",
        "Bank of Japan currency",
        "BOJ digital",
        "Japan payment system",
        "Japanese fintech"
    ],

    "USA":[
        "digital dollar",
        "Federal Reserve digital",
        "US blockchain",
        "US CBDC",
        "American digital currency",
        "Fed digital payment"
    ]
}

# ==========================
# CBDC STATUS PER COUNTRY
# ==========================
CBDC_STATUS = {
    "India":     "Pilot Stage — RBI launched the retail Digital Rupee (e₹) pilot in December 2022, currently expanding across cities.",
    "China":     "Advanced Rollout — e-CNY is operational in 26+ cities with large-scale trials; not yet a full national rollout.",
    "Nigeria":   "Implemented — eNaira was launched in October 2021, making Nigeria the first African country with a live CBDC.",
    "Jamaica":   "Implemented — JAMDEX launched in 2022 and is fully operational as legal tender.",
    "Singapore": "Research/Pilot — MAS is running Project Orchid for wholesale CBDC and purpose-bound money experiments.",
    "Australia": "Research/Pilot — RBA completed the eAUD pilot (Project Acacia) in 2023; no live CBDC yet.",
    "Japan":     "Pilot Stage — Bank of Japan completed Phase 2 CBDC pilot in 2023; a decision on launch is pending.",
    "USA":       "Exploratory/Research — No official CBDC. The Fed completed Project Hamilton research; political debate ongoing."
}

# ==========================
# COUNTRY SELECT
# ==========================
def select_country():

    countries = list(COUNTRIES.keys())

    print("\nSelect Country\n")

    for i,c in enumerate(countries):
        print(i+1,c)

    choice = int(input("\nEnter number: "))

    return countries[choice-1]


# ==========================
# NEWSAPI COLLECTION (100 ARTICLES MAX - FREE PLAN LIMIT)
# ==========================
def newsapi_collect(country, from_date, to_date):

    urls = []
    titles = []
    articles_collected = 0

    for q in COUNTRIES[country]:

        logger.info(f"NewsAPI search: {q}")

        try:
            response = newsapi.get_everything(
                q=q,
                language="en",
                page_size=100,
                page=1,
                from_param=from_date,
                to=to_date
            )

            if response.get("status") != "ok":
                logger.warning(f"NewsAPI error for '{q}': {response.get('message', 'Unknown error')}")
                continue

            if "articles" not in response:
                logger.warning(f"NewsAPI error for '{q}': No articles field in response")
                continue

            articles = response["articles"]
            
            for article in articles:
                urls.append(article["url"])
                titles.append(article["title"])
                articles_collected += 1

            logger.info(f"NewsAPI '{q}': collected {len(articles)} articles")

        except Exception as e:
            logger.error(f"NewsAPI error for '{q}': {e}")
            continue

    logger.info(f"NewsAPI total: {articles_collected} URLs collected")
    return urls, titles


# ==========================
# CLEAN GOOGLE NEWS URL
# ==========================
def clean_google_url(url):
    """Resolve Google News redirect URLs to actual article URLs."""
    if not url or "news.google.com" not in url:
        return url
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        response = requests.get(url, allow_redirects=True, timeout=15, headers=headers)
        
        if response.status_code == 200 and "news.google.com" not in response.url:
            logger.debug(f"Resolved URL: {response.url}")
            return response.url
        else:
            logger.debug(f"Could not resolve Google News URL: {url}")
            return url
            
    except requests.exceptions.Timeout:
        logger.debug(f"Timeout resolving URL: {url}")
        return url
    except requests.exceptions.RequestException as e:
        logger.debug(f"Request error resolving URL {url}: {e}")
        return url
    except Exception as e:
        logger.debug(f"Error resolving URL {url}: {e}")
        return url


# ==========================
# GOOGLE NEWS RSS
# ==========================
def google_news_collect(country, from_date, to_date):

    urls = []
    titles = []
    articles_collected = 0

    for q in COUNTRIES[country]:

        query = urllib.parse.quote(q)

        url = f"https://news.google.com/rss/search?q={query}+after:{from_date}+before:{to_date}"

        try:
            feed = feedparser.parse(url)
            
            if not feed.entries:
                logger.debug(f"Google News: no results for '{q}'")
                continue

            for entry in feed.entries:
                resolved_url = clean_google_url(entry.link)
                urls.append(resolved_url)
                titles.append(entry.title)
                articles_collected += 1
            
            logger.info(f"Google News '{q}': collected {len(feed.entries)} articles")
        except Exception as e:
            logger.error(f"Google News error for '{q}': {e}")
            continue

    logger.info(f"Google News total: {articles_collected} URLs collected")
    return urls, titles


# ==========================
# GDELT COLLECTION
# ==========================
def gdelt_collect(country, from_date, to_date):

    urls = []
    titles = []
    articles_collected = 0

    start_dt = from_date.replace("-", "") + "000000"
    end_dt   = to_date.replace("-", "")   + "235959"

    for q in COUNTRIES[country]:

        try:

            url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={urllib.parse.quote(q)}&mode=ArtList&maxrecords=100&format=json&startdatetime={start_dt}&enddatetime={end_dt}"

            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                logger.debug(f"GDELT: HTTP {response.status_code} for '{q}'")
                continue

            try:
                data = response.json()
            except Exception as e:
                logger.debug(f"GDELT: JSON parsing failed for '{q}': {e}")
                continue

            if "articles" not in data:
                logger.debug(f"GDELT: no articles for '{q}'")
                continue

            for article in data["articles"]:

                urls.append(article["url"])
                titles.append(article["title"])
                articles_collected += 1

            logger.debug(f"GDELT '{q}': {len(data['articles'])} articles")

        except Exception as e:
            logger.error(f"GDELT error for '{q}': {e}")
            continue

    logger.info(f"GDELT total: {articles_collected} URLs collected")
    return urls,titles


# ==========================
# COMBINE ALL SOURCES
# ==========================
def collect_articles(country, from_date, to_date):

    urls = []
    titles = []

    logger.info("\n" + "="*50)
    logger.info(f"Collecting articles for {country} ({from_date} to {to_date})")
    logger.info("="*50)

    # NewsAPI Collection
    logger.info("\nCollecting from NewsAPI...")
    try:
        u, t = newsapi_collect(country, from_date, to_date)
        urls += u
        titles += t
        logger.info(f"✓ NewsAPI: {len(u)} articles collected")
    except Exception as e:
        logger.error(f"✗ NewsAPI collection failed: {e}")
        logger.info("Continuing with other sources...")

    # Google News Collection - DISABLED
    # (Google News RSS returns unresolvable redirect URLs that cannot be extracted)
    logger.info("\nGoogle News collection disabled (technical limitation)")
    # try:
    #     u, t = google_news_collect(country, from_date, to_date)
    #     urls += u
    #     titles += t
    #     logger.info(f"✓ Google News: {len(u)} articles collected")
    # except Exception as e:
    #     logger.error(f"✗ Google News collection failed: {e}")
    #     logger.info("Continuing with other sources...")

    # GDELT Collection
    logger.info("\nCollecting from GDELT...")
    try:
        u, t = gdelt_collect(country, from_date, to_date)
        urls += u
        titles += t
        logger.info(f"✓ GDELT: {len(u)} articles collected")
    except Exception as e:
        logger.error(f"✗ GDELT collection failed: {e}")
        logger.info("Continuing with available articles...")

    logger.info(f"\nTotal articles collected from all sources: {len(urls)}")

    if len(urls) == 0:
        logger.warning("No articles collected from any source")
        return pd.DataFrame()

    df = pd.DataFrame({
        "url": urls,
        "title": titles
    })

    before_url_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    url_dedup_removed = before_url_dedup - len(df)
    logger.info(f"After URL dedup: {len(df)} articles (removed {url_dedup_removed} duplicates)")

    if len(df) == 0:
        logger.warning("All articles removed during URL deduplication")
        return df

    try:
        before_semantic_dedup = len(df)
        df = deduplicate_articles(df)
        semantic_dedup_removed = before_semantic_dedup - len(df)
        logger.info(f"After semantic dedup: {len(df)} articles (removed {semantic_dedup_removed} duplicates)")
    except Exception as e:
        logger.error(f"Semantic deduplication failed: {e}")
        logger.info("Continuing with URL-deduplicated articles...")

    return df


# ==========================
# SEMANTIC DEDUPLICATION
# ==========================
def deduplicate_articles(df):

    if len(df) == 0:
        return df

    logger.info("Running semantic deduplication...")

    titles     = df["title"].tolist()
    embeddings = semantic_model.encode(titles, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    keep    = []
    dropped = 0

    for i in range(len(df)):
        if any(sim_matrix[i][j] > 0.9 for j in keep):
            dropped += 1
        else:
            keep.append(i)

    logger.info(f"Removed {dropped} near-duplicate articles")

    return df.iloc[keep].reset_index(drop=True)


# ==========================
# ARTICLE EXTRACTION
# ==========================
def extract_text(url):

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            logger.debug(f"Failed to fetch: {url} (HTTP {response.status_code})")
            return None

        text = trafilatura.extract(response.text)

        if text is None:
            logger.debug(f"Extraction failed: {url}")
            return None

        if len(text) < 200:
            logger.debug(f"Text too short ({len(text)} chars): {url}")
            return None

        return text

    except requests.exceptions.Timeout:
        logger.debug(f"Timeout fetching URL: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.debug(f"Request error for {url}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Extraction error for {url}: {e}")
        return None


# ==========================
# CBDC RELEVANCE
# ==========================
def filter_relevant_sentences(sentences):

    if not sentences:
        return []

    embeddings = semantic_model.encode(sentences, show_progress_bar=False)
    similarities = cosine_similarity(embeddings, query_embedding)
    relevant_indices = np.where(similarities[:, 0] > 0.15)[0]

    return [sentences[i] for i in relevant_indices]


# ==========================
# SENTIMENT SCORE
# ==========================
def score_sentence(sentence):

    try:
        X = vectorizer.transform([sentence])
        score = sentiment_model.predict(X)[0]
        score = float(score)
        score = max(0, min(1, score))
        return score

    except Exception as e:
        logger.debug(f"Sentiment scoring error: {e}")
        return None


# ==========================
# SCORE DOCUMENT
# ==========================
def score_document(url, index, total):

    text = extract_text(url)

    if text is None:
        logger.info(f"[{index}/{total}] Skipped: extraction failed - {url[:60]}...")
        return None

    logger.debug(f"Extracted length: {len(text)}")

    sentences = sent_tokenize(text)
    relevant = filter_relevant_sentences(sentences)

    if len(relevant) == 0:
        logger.info(f"[{index}/{total}] Skipped: no CBDC content")
        return None

    scores = []
    weights = []

    for s in relevant:
        sc = score_sentence(s)
        if sc is None:
            continue
        scores.append(sc)
        weights.append(len(s.split()))

    if len(scores) == 0:
        logger.info(f"[{index}/{total}] Skipped: sentiment scoring failed")
        return None

    final_score = np.average(scores, weights=weights)
    logger.info(f"[{index}/{total}] ✓ Scored: {final_score:.3f} ({len(relevant)} relevant sentences)")
    return final_score


# ==========================
# ANALYZE COUNTRY
# ==========================
def _process_row(args):
    i, total, row, country = args
    score = score_document(row["url"], i, total)
    if score is None:
        return None
    return {
        "country":    country,
        "title":      row["title"],
        "url":        row["url"],
        "sentiment":  score
    }

def analyze_country(country, from_date, to_date):

    articles = collect_articles(country, from_date, to_date)

    if len(articles) == 0:
        logger.error("No articles collected!")
        return pd.DataFrame()

    logger.info(f"\nStarting sentiment analysis on {len(articles)} articles...")

    results = []
    total = len(articles)

    for i, (_, row) in enumerate(articles.iterrows(), start=1):
        result = _process_row((i, total, row, country))
        if result is not None:
            results.append(result)

    logger.info(f"\n{'='*50}")
    logger.info(f"Extraction Summary:")
    logger.info(f"  Total articles analyzed: {total}")
    logger.info(f"  Successfully scored: {len(results)}")
    logger.info(f"  Skipped: {total - len(results)}")
    logger.info(f"{'='*50}\n")

    return pd.DataFrame(results)


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    country = select_country()

    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    from_date = start_date
    to_date = end_date

    df = analyze_country(country, from_date, to_date)

    if len(df) == 0:
        logger.error("\n❌ No valid articles found. Check logs above for details.")
        logger.error("Common causes:")
        logger.error("  1. Missing dependencies (trafilatura, newsapi, sentence-transformers)")
        logger.error("  2. Invalid NewsAPI key")
        logger.error("  3. Network issues fetching articles")
        logger.error("  4. All articles filtered (relevance threshold too strict)")
        sys.exit(1)

    else:
        country_score = df["sentiment"].mean()

        logger.info("\n" + "="*50)
        logger.info(f"✓ ANALYSIS COMPLETE")
        logger.info("="*50)
        logger.info(f"Country Sentiment: {round(country_score, 3)}")
        logger.info(f"Articles analyzed: {len(df)}")
        logger.info(f"Sentiment range: [{df['sentiment'].min():.3f}, {df['sentiment'].max():.3f}]")

        filename = f"{country}_{start_date.split('-')[0]}.csv"

        df.to_csv(filename, index=False)

        logger.info(f"\n✓ Dataset saved: {filename}")
        logger.info("="*50)