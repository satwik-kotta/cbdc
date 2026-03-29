from main import analyze_document
import pandas as pd

# --------------------------------
# STEP 1: Define your document set
# --------------------------------

documents = [

    # 🇯🇲 Jamaica documents
    {"country": "Jamaica", "url": "https://www.ledgerinsights.com/sole-jamaican-cbdc-wallet-provider-questions-jam-dex-progress/"},
    {"country": "Jamaica", "url": "https://www.jamaicaobserver.com/2024/09/04/boj-update-jam-dex/?utm_source=chatgpt.com"},

    # 🇬🇭 Ghana documents
    {"country": "Ghana", "url": "https://gna.org.gh/2023/10/bog-governor-discuss-ghanas-digital-currency-initiative-at-imf-meetings/#utm_source=chatgpt.com"},
    {"country": "Ghana", "url": "https://www.newsghana.com.gh/ghana-ready-for-controlled-ecedi-rollout-says-ecobank-executive/?utm_source=chatgpt.com"},
]

# --------------------------------
# STEP 2: Analyze all documents
# --------------------------------

results = []

for doc in documents:

    print("\nAnalyzing:", doc["url"])

    try:
        output = analyze_document(doc["url"], is_url=True)

        results.append({
            "country": doc["country"],
            "url": doc["url"],
            "document_score": output["document_score"]
        })

    except Exception as e:
        print("Error:", e)

# --------------------------------
# STEP 3: Convert to DataFrame
# --------------------------------

df = pd.DataFrame(results)

print("\nDocument Results:")
print(df)

# --------------------------------
# STEP 4: Compute Country Score
# --------------------------------

country_scores = df.groupby("country")["document_score"].mean().reset_index()

print("\nCountry Sentiment Index:")
print(country_scores)

# --------------------------------
# STEP 5: Save Outputs
# --------------------------------

df.to_csv("document_scores.csv", index=False)
country_scores.to_csv("country_sentiment.csv", index=False)

print("\nSaved document_scores.csv and country_sentiment.csv")
