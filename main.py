from newspaper import Article
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import pickle

# -----------------------------
# LOAD YOUR CUSTOM MODEL
# -----------------------------
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Custom sentiment model loaded.")

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
# STEP 3: YOUR CUSTOM SCORING
# -----------------------------
def score_sentence_custom(sentence):
    X = vectorizer.transform([sentence])
    score = model.predict(X)[0]

    # Clamp score between 0 and 1
    return max(0, min(1, score))

# -----------------------------
# STEP 4: Score Sentences
# -----------------------------
def score_sentences(sentences):

    results = []

    for s in sentences:
        if len(s.strip()) < 5:
            continue

        score = score_sentence_custom(s)

        results.append({
            "sentence": s,
            "score": score
        })

    return results

# -----------------------------
# STEP 5: Document Score
# -----------------------------
def compute_document_score(scored_sentences):

    scores = [item['score'] for item in scored_sentences]
    weights = [len(item['sentence'].split()) for item in scored_sentences]

    return np.average(scores, weights=weights)

# -----------------------------
# STEP 6: MASTER PIPELINE
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
# STEP 7: RUN PROGRAM
# -----------------------------
if __name__ == "__main__":

    url = "https://www.jamaicaobserver.com/2024/09/04/boj-update-jam-dex/?utm_source=chatgpt.com#google_vignette"

    result = analyze_document(url, is_url=True)

    print("\nDOCUMENT SCORE:", result["document_score"])

    for item in result["sentences"]:
        print(item["score"], "->", item["sentence"])

    # Save CSV Output
    df = pd.DataFrame(result["sentences"])
    df.to_csv("output_scores.csv", index=False)
