import pickle
import numpy as np
import pandas as pd
import nltk
import trafilatura
import requests
import logging
import sys
import os
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RETRY = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.8,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
)
HTTP = requests.Session()
HTTP.mount("https://", HTTPAdapter(max_retries=RETRY))
HTTP.mount("http://", HTTPAdapter(max_retries=RETRY))

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
# EXTERNAL NEWS API KEYS
# ==========================
MEDIASTACK_API_KEY = os.getenv("MEDIASTACK_API_KEY", "9276e5a5fa84bf5527e1580b8d5ec98d")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if MEDIASTACK_API_KEY:
    logger.info("✓ MediaStack API key loaded")
else:
    logger.warning("MediaStack API key missing. Set MEDIASTACK_API_KEY to enable this source.")

OPENAI_CLIENT = None
OPENAI_DISABLED = False
if OPENAI_API_KEY and OpenAI is not None:
    try:
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=20)
        logger.info("✓ OpenAI API key loaded")
    except Exception as e:
        logger.warning(f"OpenAI client initialization failed: {e}")
elif OPENAI_API_KEY and OpenAI is None:
    logger.warning("OpenAI key found but openai package is not installed.")
else:
    logger.warning("OpenAI API key missing. Set OPENAI_API_KEY to enable OpenAI analysis.")

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

    while True:
        try:
            choice = int(input("\nEnter number: "))
            if 1 <= choice <= len(countries):
                return countries[choice-1]
            print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid number.")


def normalize_url(url):
    if not url:
        return None

    url = url.strip()
    if not url.startswith("http"):
        return None

    parsed = urlparse(url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_query = [
        (k, v) for k, v in query_pairs
        if not k.lower().startswith("utm_") and k.lower() not in {"ocid", "fbclid", "gclid", "mc_cid", "mc_eid"}
    ]

    normalized = parsed._replace(query=urlencode(filtered_query, doseq=True), fragment="")
    return urlunparse(normalized)


def normalize_title(title):
    if isinstance(title, dict):
        if "eng" in title and title["eng"]:
            title = title["eng"]
        else:
            title = next((v for v in title.values() if v), None)

    if not title:
        return None

    return str(title).strip()


# ==========================
# MEDIASTACK COLLECTION
# ==========================
def mediastack_collect(country, from_date, to_date):

    if not MEDIASTACK_API_KEY:
        logger.warning("MediaStack collection skipped (missing API key)")
        return [], []

    urls = []
    titles = []
    articles_collected = 0

    query_candidates = [
        f"{country} central bank",
        f"cbdc {country}",
        f"{country} digital currency",
    ]

    for query in query_candidates:
        logger.info(f"MediaStack search: {query}")
        try:
            params = {
                "access_key": MEDIASTACK_API_KEY,
                "keywords": query,
                "languages": "en",
                "sort": "published_desc",
                "limit": 100,
                "offset": 0,
            }

            response = HTTP.get(
                "http://api.mediastack.com/v1/news",
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.warning(f"MediaStack API error: {data['error']}")
                if data["error"].get("code") == "rate_limit_reached":
                    logger.warning("MediaStack rate limit reached. Using collected articles so far.")
                    break
                continue

            articles = data.get("data", [])
            for article in articles:
                url = normalize_url(article.get("url"))
                title = normalize_title(article.get("title"))
                if not url or not title:
                    continue
                urls.append(url)
                titles.append(title)
                articles_collected += 1

            if articles_collected > 0:
                # Stop early to preserve low MediaStack plan quota.
                break
        except Exception as e:
            logger.error(f"MediaStack error: {e}")
            continue

    logger.info(f"MediaStack total: {articles_collected} URLs collected")
    return urls, titles


# ==========================
# COMBINE ALL SOURCES
# ==========================
def collect_articles(country, from_date, to_date):

    urls = []
    titles = []

    logger.info("\n" + "="*50)
    logger.info(f"Collecting articles for {country} ({from_date} to {to_date})")
    logger.info("="*50)

    if not MEDIASTACK_API_KEY:
        logger.error("No source API key is configured. Set MEDIASTACK_API_KEY.")
        return pd.DataFrame()

    # MediaStack Collection
    logger.info("\nCollecting from MediaStack...")
    try:
        u, t = mediastack_collect(country, from_date, to_date)
        urls += u
        titles += t
        logger.info(f"✓ MediaStack: {len(u)} articles collected")
    except Exception as e:
        logger.error(f"✗ MediaStack collection failed: {e}")
        logger.info("Continuing with other sources...")

    logger.info(f"\nTotal articles collected from all sources: {len(urls)}")

    if len(urls) == 0:
        logger.warning("No articles collected from any source")
        return pd.DataFrame()

    df = pd.DataFrame({
        "url": urls,
        "title": titles
    })

    df = df.dropna(subset=["url", "title"])

    before_url_dedup = len(df)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
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
        response = HTTP.get(url, headers=headers, timeout=20)

        if response.status_code != 200:
            logger.debug(f"Failed to fetch: {url} (HTTP {response.status_code})")
            return None

        text = trafilatura.extract(response.text, include_comments=False, include_tables=False)

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


def score_document_openai(text):

    global OPENAI_DISABLED

    if OPENAI_DISABLED:
        return None

    if OPENAI_CLIENT is None:
        return None

    prompt = (
        "You are scoring sentiment of text about Central Bank Digital Currencies (CBDC). "
        "Return ONLY a number between 0 and 1. "
        "Interpretation: 0=very negative, 0.5=neutral, 1=very positive. "
        "If the text has no CBDC relevance, return exactly -1.\n\n"
        f"Text:\n{text[:12000]}"
    )

    try:
        response = OPENAI_CLIENT.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0,
            max_output_tokens=32,
        )

        raw = (response.output_text or "").strip()
        val = float(raw)
        if val == -1:
            return None
        return max(0.0, min(1.0, val))
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        if status_code == 429:
            OPENAI_DISABLED = True
            logger.warning("OpenAI rate limit reached (429). Disabling OpenAI scoring for this run and using local model fallback.")
        logger.debug(f"OpenAI scoring error: {e}")
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

    openai_score = score_document_openai(text)
    if openai_score is not None:
        logger.info(f"[{index}/{total}] ✓ Scored (OpenAI): {openai_score:.3f}")
        return openai_score

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
        logger.error("  1. Missing dependencies (trafilatura, sentence-transformers)")
        logger.error("  2. Missing/invalid MEDIASTACK_API_KEY")
        logger.error("  3. Network issues fetching articles")
        logger.error("  4. All articles filtered or extraction blocked")
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