from newspaper import Article
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import numpy as np
import pandas as pd

# -----------------------------
# STEP 1: Extract text from URL
# -----------------------------
def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# -----------------------------
# STEP 2: Sentence Split
# -----------------------------
def split_into_sentences(text):
    return sent_tokenize(text)

# -----------------------------
# STEP 3: Load Sentiment Model
# -----------------------------
print("Loading sentiment model... (first run takes time)")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# -----------------------------
# STEP 4: Normalize score to 0-1
# -----------------------------
def normalize_sentiment(output):
    label = output['label']
    score = output['score']

    if label == "NEGATIVE":
        return 1 - score
    elif label == "NEUTRAL":
        return 0.5
    else:
        return score

# -----------------------------
# STEP 5: Score Sentences
# -----------------------------
def score_sentences(sentences):

    results = []

    for s in sentences:
        if len(s.strip()) < 5:
            continue

        out = sentiment_model(s)[0]
        score = normalize_sentiment(out)

        results.append({
            "sentence": s,
            "score": score
        })

    return results

# -----------------------------
# STEP 6: Document Score
# -----------------------------
def compute_document_score(scored_sentences):

    scores = [item['score'] for item in scored_sentences]
    weights = [len(item['sentence'].split()) for item in scored_sentences]

    return np.average(scores, weights=weights)

# -----------------------------
# STEP 7: MASTER PIPELINE
# -----------------------------
def analyze_document(input_data, is_url=True):

    if is_url:
        text = extract_text_from_url(input_data)
    else:
        text = input_data

    sentences = split_into_sentences(text)

    scored = score_sentences(sentences)

    doc_score = compute_document_score(scored)

    return {
        "document_score": doc_score,
        "sentences": scored
    }

# -----------------------------
# STEP 8: RUN PROGRAM
# -----------------------------
if __name__ == "__main__":

    url = "PASTE_A_GHANA_OR_JAMAICA_CBDC_ARTICLE_LINK_HERE"

    result = analyze_document(url, is_url=True)

    print("\nDOCUMENT SCORE:", result["document_score"])

    for item in result["sentences"]:
        print(item["score"], "->", item["sentence"])

    # Save CSV Output
    df = pd.DataFrame(result["sentences"])
    df.to_csv("output_scores.csv", index=False)
