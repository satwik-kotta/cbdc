import os
import pickle
import sys
import logging
from datetime import datetime
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    logger.error("NLTK punkt tokenizer is missing. Install it once offline into this environment.")
    logger.error("Example (one-time): python -m nltk.downloader punkt")
    sys.exit(1)

# Supported local formats
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules"}
DEFAULT_COUNTRY = "Jamaica"
DEFAULT_ARTICLES_SUBDIR = Path("cbdc") / "BB_Docs_20260325_082641"

# Relevance/scoring tuning to avoid overly compressed article scores.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.22"))
TOP_K_RELEVANT = int(os.getenv("TOP_K_RELEVANT", "80"))
MIN_SENTENCE_WORDS = int(os.getenv("MIN_SENTENCE_WORDS", "6"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "1"))

# Semantic anchors broaden topic understanding beyond direct keyword mentions.
CBDC_QUERY_TEXTS = [
    "central bank digital currency cbdc digital rupee digital yuan enaira jamdex",
    "digital payments modernization cashless economy mobile wallet adoption",
    "financial inclusion instant payments reduced transaction costs",
    "retail payments government transfers merchant acceptance",
    "secure transparent traceable digital money infrastructure",
]


# ==========================
# LOAD SENTIMENT MODEL
# ==========================
try:
    with open("sentiment_model.pkl", "rb") as f:
        sentiment_model = pickle.load(f)
    logger.info("Sentiment model loaded")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    sys.exit(1)

try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Vectorizer loaded")
except Exception as e:
    logger.error(f"Failed to load vectorizer: {e}")
    sys.exit(1)


# ==========================
# SEMANTIC MODEL
# ==========================
try:
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    logger.info("Semantic model loaded")
except Exception as e:
    logger.error(f"Semantic model failed (local-only mode): {e}")
    logger.error("Ensure all-MiniLM-L6-v2 is already cached locally for this environment.")
    sys.exit(1)

query_vectors = semantic_model.encode(CBDC_QUERY_TEXTS, show_progress_bar=False)
query_embedding = np.mean(query_vectors, axis=0, keepdims=True)


# ==========================
# SEARCH QUERIES (USED FOR LOCAL FILE NAME MATCHING)
# ==========================
COUNTRIES = {
    "India": [
        "digital rupee",
        "RBI digital currency",
        "India central bank currency",
        "Indian blockchain currency"
    ],
    "China": [
        "digital yuan",
        "e-CNY",
        "China blockchain currency",
        "PBOC digital payment"
    ],
    "Nigeria": [
        "Nigeria digital currency",
        "Nigeria blockchain",
        "Nigerian central bank currency",
        "eNaira payment",
        "Nigeria digital payment",
        "CBN blockchain"
    ],
    "Jamaica": [
        "JAM-DEX",
        "Jam-Dex",
        "Jamaica CBDC",
        "Bank of Jamaica digital currency",
        "Jamaica central bank digital currency"
    ],
    "Singapore": [
        "Singapore digital currency",
        "Singapore blockchain",
        "MAS fintech",
        "Singapore monetary authority",
        "Singapore payment system",
        "Asian digital currency"
    ],
    "Australia": [
        "Australia digital currency",
        "Australia blockchain",
        "RBA fintech",
        "Australian payment system",
        "Reserve Bank Australia",
        "Australian monetary policy"
    ],
    "Japan": [
        "digital yen",
        "Japan blockchain",
        "Bank of Japan currency",
        "BOJ digital",
        "Japan payment system",
        "Japanese fintech"
    ],
    "USA": [
        "digital dollar",
        "Federal Reserve digital",
        "US blockchain",
        "US CBDC",
        "American digital currency",
        "Fed digital payment"
    ]
}


# ==========================
# COUNTRY SELECT
# ==========================
def select_country():
    env_country = os.getenv("COUNTRY", "").strip()
    if env_country in COUNTRIES:
        logger.info(f"Country selected from COUNTRY env: {env_country}")
        return env_country

    logger.info(
        f"Defaulting to {DEFAULT_COUNTRY}. Set COUNTRY env var to override."
    )
    return DEFAULT_COUNTRY


# ==========================
# LOCAL DOCUMENT COLLECTION
# ==========================
def _walk_supported_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                yield Path(dirpath) / name


def collect_local_documents(country):
    base_dir = Path(__file__).resolve().parent
    default_local_root = base_dir / DEFAULT_ARTICLES_SUBDIR
    local_root = Path(os.getenv("LOCAL_ARTICLES_DIR", str(default_local_root)))

    if not local_root.exists() or not local_root.is_dir():
        logger.error(f"Local articles directory not found: {local_root}")
        return pd.DataFrame(columns=["path", "title"])

    seen = set()
    files = []

    for file_path in _walk_supported_files(local_root):
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        files.append(file_path)

    logger.info(
        f"Local collection: found {len(files)} files for {country} in {local_root}"
    )

    if not files:
        return pd.DataFrame(columns=["path", "title"])

    return pd.DataFrame({
        "path": [str(p) for p in files],
        "title": [p.stem for p in files]
    })


# ==========================
# SEMANTIC DEDUPLICATION
# ==========================
def deduplicate_articles(df):
    if len(df) == 0:
        return df

    logger.info("Running semantic deduplication on file titles...")

    titles = df["title"].fillna("").tolist()
    embeddings = semantic_model.encode(titles, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    keep = []
    dropped = 0

    for i in range(len(df)):
        if any(sim_matrix[i][j] > 0.9 for j in keep):
            dropped += 1
        else:
            keep.append(i)

    logger.info(f"Removed {dropped} near-duplicate files")
    return df.iloc[keep].reset_index(drop=True)


# ==========================
# FILE TEXT EXTRACTION
# ==========================
def extract_pdf_text(path: Path):
    try:
        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks.append(page_text)
        text = "\n".join(chunks).strip()
        return text if text else None
    except Exception as e:
        logger.debug(f"PDF extraction error for {path}: {e}")
        return None


def extract_image_text(path: Path):
    try:
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img)
        text = (text or "").strip()
        return text if text else None
    except Exception as e:
        logger.debug(f"Image OCR error for {path}: {e}")
        return None


def extract_text_from_file(file_path):
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        text = extract_pdf_text(path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        text = extract_image_text(path)
    else:
        return None

    if text is None:
        logger.debug(f"Extraction failed: {path}")
        return None

    if len(text) < 120:
        logger.debug(f"Text too short ({len(text)} chars): {path}")
        return None

    return text


# ==========================
# CBDC RELEVANCE
# ==========================
def filter_relevant_sentences(sentences):
    if not sentences:
        return []

    # Drop very short/generic lines before semantic relevance filtering.
    candidate_sentences = [
        s.strip() for s in sentences
        if len(s.split()) >= MIN_SENTENCE_WORDS
    ]

    if not candidate_sentences:
        return []

    embeddings = semantic_model.encode(candidate_sentences, show_progress_bar=False)
    similarities = cosine_similarity(embeddings, query_embedding)
    similarity_scores = similarities[:, 0]

    relevant_indices = np.where(similarity_scores >= RELEVANCE_THRESHOLD)[0]
    if len(relevant_indices) == 0:
        return []

    ranked_indices = sorted(
        relevant_indices,
        key=lambda i: similarity_scores[i],
        reverse=True
    )
    core_selected = ranked_indices[:TOP_K_RELEVANT]

    expanded_indices = set(core_selected)
    for idx in core_selected:
        for delta in range(1, CONTEXT_WINDOW + 1):
            if idx - delta >= 0:
                expanded_indices.add(idx - delta)
            if idx + delta < len(candidate_sentences):
                expanded_indices.add(idx + delta)

    selected = sorted(
        expanded_indices,
        key=lambda i: similarity_scores[i],
        reverse=True,
    )[: TOP_K_RELEVANT * (CONTEXT_WINDOW + 1)]

    return [(candidate_sentences[i], float(similarity_scores[i])) for i in selected]


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
def score_document(file_path, index, total):
    text = extract_text_from_file(file_path)

    if text is None:
        logger.info(f"[{index}/{total}] Skipped: extraction failed - {str(file_path)[:60]}...")
        return None

    sentences = sent_tokenize(text)
    relevant = filter_relevant_sentences(sentences)

    if len(relevant) == 0:
        logger.info(f"[{index}/{total}] Skipped: no CBDC content")
        return None

    scores = []
    weights = []

    for sentence, relevance in relevant:
        sc = score_sentence(sentence)
        if sc is None:
            continue
        scores.append(sc)
        # Emphasize semantically stronger CBDC statements in aggregation.
        weights.append(len(sentence.split()) * max(relevance, 0.05))

    if len(scores) == 0:
        logger.info(f"[{index}/{total}] Skipped: sentiment scoring failed")
        return None

    final_score = np.average(scores, weights=weights)
    logger.info(f"[{index}/{total}] Scored: {final_score:.3f} ({len(relevant)} relevant sentences)")
    return final_score


# ==========================
# ANALYZE COUNTRY
# ==========================
def _process_row(args):
    i, total, row, country = args
    score = score_document(row["path"], i, total)
    if score is None:
        return None

    return {
        "country": country,
        "title": row["title"],
        "source_file": row["path"],
        "sentiment": score
    }


def analyze_country(country):
    files_df = collect_local_documents(country)

    if len(files_df) == 0:
        logger.error("No local documents found!")
        return pd.DataFrame()

    before_dedup = len(files_df)
    files_df = files_df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    logger.info(f"After path dedup: {len(files_df)} files (removed {before_dedup - len(files_df)})")

    files_df = deduplicate_articles(files_df)

    logger.info(f"Starting sentiment analysis on {len(files_df)} files...")

    results = []
    total = len(files_df)

    for i, (_, row) in enumerate(files_df.iterrows(), start=1):
        result = _process_row((i, total, row, country))
        if result is not None:
            results.append(result)

    logger.info("=" * 50)
    logger.info("Extraction Summary:")
    logger.info(f"  Total files analyzed: {total}")
    logger.info(f"  Successfully scored: {len(results)}")
    logger.info(f"  Skipped: {total - len(results)}")
    logger.info("=" * 50)

    return pd.DataFrame(results)


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    country = select_country()

    df = analyze_country(country)

    if len(df) == 0:
        logger.error("No valid local documents found for scoring. Check logs above.")
        logger.error("Common causes:")
        logger.error("  1. No pdf/png/jpg/jpeg files matched country or keywords")
        logger.error("  2. OCR engine unavailable (install tesseract binary)")
        logger.error("  3. Document text extraction failed")
        logger.error("  4. All documents filtered (relevance threshold too strict)")
        sys.exit(1)

    country_score = df["sentiment"].mean()

    logger.info("=" * 50)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Country Sentiment: {round(country_score, 3)}")
    logger.info(f"Files analyzed: {len(df)}")
    logger.info(f"Sentiment range: [{df['sentiment'].min():.3f}, {df['sentiment'].max():.3f}]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{country}_{timestamp}.csv"

    df.to_csv(filename, index=False)
    logger.info(f"Dataset saved: {filename}")
    logger.info("=" * 50)
