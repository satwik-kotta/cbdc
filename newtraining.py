import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("final_dataset .csv")

sentences = df["Sentence"]
scores = df["FinalScore"]

print("Dataset loaded:", len(df), "sentences")

# -----------------------------
# 2. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    sentences,
    scores,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 3. Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorization complete")

# -----------------------------
# 4. Train Model
# -----------------------------
model = Ridge()

model.fit(X_train_vec, y_train)

print("Model training complete")

# -----------------------------
# 5. Evaluate Model
# -----------------------------
preds = model.predict(X_test_vec)

rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

print("\nModel Evaluation")
print("RMSE:", rmse)
print("R2:", r2)

# -----------------------------
# 6. Save Model
# -----------------------------
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved successfully")
print("Files created:")
print("sentiment_model.pkl")
print("vectorizer.pkl")