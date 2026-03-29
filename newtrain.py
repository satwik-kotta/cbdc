import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

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
# 4. Define Models
# -----------------------------
models = {
    "Ridge": Ridge(),
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(n_estimators=200),
    "GradientBoosting": GradientBoostingRegressor()
}


best_model = None
best_r2 = -999
best_name = ""


# -----------------------------
# 5. Train and Evaluate
# -----------------------------
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("RMSE:", rmse)
    print("R2:", r2)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name


# -----------------------------
# 6. Best Model
# -----------------------------
print("\nBest Model:", best_name)
print("Best R2:", best_r2)


# -----------------------------
# 7. Save Best Model
# -----------------------------
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nBest model saved")
print("Files created:")
print("sentiment_model.pkl")
print("vectorizer.pkl")