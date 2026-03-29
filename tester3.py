import pandas as pd
import numpy as np
import krippendorff
import pingouin as pg

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("removed_sentences_dataset.csv")

# Rename columns to simpler names
df = df.rename(columns={
    "Prof. Sathye": "Prof",
    "Manoranjan Dash sir": "Dash",
    "Score": "You"
})

print("\nDataset Loaded:")
print(df.head())

print("\nColumns in dataset:")
print(df.columns)

# -----------------------------
# 2. Create Final Consensus Score
# -----------------------------
df["FinalScore"] = df[["LLM","Prof","Dash","You"]].mean(axis=1)

print("\nConsensus Score Created")

# -----------------------------
# 3. Compute MAE for each annotator
# -----------------------------
df["Ref_LLM"] = df[["Prof","Dash","You"]].mean(axis=1)
df["Ref_Prof"] = df[["LLM","Dash","You"]].mean(axis=1)
df["Ref_Dash"] = df[["LLM","Prof","You"]].mean(axis=1)
df["Ref_You"] = df[["LLM","Prof","Dash"]].mean(axis=1)

mae_llm = abs(df["LLM"] - df["Ref_LLM"]).mean()
mae_prof = abs(df["Prof"] - df["Ref_Prof"]).mean()
mae_dash = abs(df["Dash"] - df["Ref_Dash"]).mean()
mae_you = abs(df["You"] - df["Ref_You"]).mean()

print("\nMean Absolute Error (MAE):")
print("LLM:", mae_llm)
print("Prof:", mae_prof)
print("Dash:", mae_dash)
print("You:", mae_you)

# -----------------------------
# 4. Bias Analysis
# -----------------------------
bias_llm = (df["LLM"] - df["Ref_LLM"]).mean()
bias_prof = (df["Prof"] - df["Ref_Prof"]).mean()
bias_dash = (df["Dash"] - df["Ref_Dash"]).mean()
bias_you = (df["You"] - df["Ref_You"]).mean()

print("\nBias Analysis:")
print("LLM:", bias_llm)
print("Prof:", bias_prof)
print("Dash:", bias_dash)
print("You:", bias_you)

# -----------------------------
# 5. Correlation Matrix
# -----------------------------
print("\nAnnotator Correlation Matrix:")
corr_matrix = df[["LLM","Prof","Dash","You"]].corr()
print(corr_matrix)

# -----------------------------
# 6. Krippendorff's Alpha
# -----------------------------
data = df[["LLM","Prof","Dash","You"]].to_numpy().T
alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='interval')

print("\nKrippendorff Alpha:", alpha)

# -----------------------------
# 7. Intraclass Correlation (ICC)
# -----------------------------
print("\nIntraclass Correlation Coefficient (ICC):")

# Convert dataset to long format for ICC
icc_data = df.reset_index().melt(
    id_vars=['index'],
    value_vars=['LLM','Prof','Dash','You'],
    var_name='Rater',
    value_name='Score'
)

icc_data = icc_data.rename(columns={'index':'SentenceID'})

icc_result = pg.intraclass_corr(
    data=icc_data,
    targets='SentenceID',
    raters='Rater',
    ratings='Score'
)

print(icc_result)

# -----------------------------
# 8. Sentence-level Disagreement
# -----------------------------
df["StdDev"] = df[["LLM","Prof","Dash","You"]].std(axis=1)

print("\nTop Disagreement Sentences:")
print(df.sort_values("StdDev", ascending=False)[["Sentence","StdDev"]].head(10))

# -----------------------------
# 9. Save Final Dataset
# -----------------------------
df.to_csv("final_cbdc_sentiment_dataset.csv", index=False)

print("\nFinal dataset saved as 'final_cbdc_ICC.csv'")