import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load your dataset
# -----------------------------
df = pd.read_csv("training_data.csv")

print("Dataset size:", len(df))

X_text = df["Sentence"]
y = df["Score"]

# -----------------------------
# Convert text → numeric features
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,2),
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

print("Feature matrix shape:", X.shape)

# -----------------------------
# Split data (80/20)
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train YOUR custom model
# -----------------------------
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

print("Model trained.")

# -----------------------------
# Validate how well it mimics YOU
# -----------------------------
preds = model.predict(X_val)

mae = mean_absolute_error(y_val, preds)
print("Validation MAE:", round(mae,3))

# -----------------------------
# Save model + vectorizer
# -----------------------------
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully.")
