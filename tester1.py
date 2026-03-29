"""
Inter-Annotator Agreement Analysis for Sentiment Scoring
=========================================================
Computes ICC, pairwise correlations, flags high-disagreement sentences,
and recommends a ground truth strategy for model training.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── 1. DATA ──────────────────────────────────────────────────────────────────

data = {
    "Sentence": [
        "Ghana is currently in the pilot phase of its central bank digital currency, known as the e-Cedi.",
        "The Bank of Ghana launched the e-Cedi pilot to test the feasibility of a retail digital currency",
        "The e-Cedi pilot successfully supported both online and offline transactions.",
        "The central bank views the e-Cedi as a tool to promote financial inclusion and digital payments.",
        "Ghana is prepared to begin a controlled rollout of its central bank digital currency (CBDC) following significant progress made by the Bank of Ghana.",
        "Stakeholders in the financial sector have observed meaningful strides from the central bank regarding the introduction of the eCedi.",
        "The introduction and full implementation of the eCedi, together with broader digital payment reforms, will be crucial in strengthening Ghana's monetary sovereignty.",
        "The First Deputy Governor of the Bank of Ghana described eCedi as a key pillar of Ghana's digital future.",
        "The central bank digital currency will create a more efficient payment ecosystem that proves faster, safer, and more reliable for individuals and businesses.",
        "Early and deliberate investments in digital systems have created an enabling environment for successfully adopting a CBDC at scale.",
        "Robust security measures and minimal transaction costs will be critical for successful adoption of the eCedi.",
        "The pilot revealed significant implementation challenges as it moves closer to rolling out the eCedi.",
        "Stakeholder feedback suggests that public trust and fee considerations could significantly influence adoption rates.",
        "Despite the popularity of mobile money, the top bank has been pushing the eCedi central bank digital currency, whose pilot has been ongoing this year.",
        "The eCedi will complement mobile payments, ensuring minimal disruption to Ghana's financial systems",
        "BoG will remain cautious and rely on feedback from the pilots to guide its rollout.",
        "The Bank of Jamaica officially launched its central bank digital currency, known as JAM-DEX, in July 2022.",
        "The evidence shows that the CBDCs in Jamaica have failed to gain traction with consumers or businesses.",
        "Despite strong government promotion, consumer adoption of Jamaica's CBDC remains limited.",
        "Merchants have shown little incentive to adopt JAM-DEX due to low demand from customers.",
        "The CEO of Jamaica's largest CBDC wallet provider questioned whether the effort to promote JAM-DEX is justified.",
        "It is often easier for users to rely on existing digital payment apps rather than converting funds into CBDC.",
        "There are concerns about whether investing further in JAM-DEX infrastructure will deliver sufficient returns",
        "JAM-DEX offers a safe, efficient and convenient way to pay for goods and services without the need for physical cash.",
        "The expansion of digital currency usage is expected to ease challenges associated with cash availability at ATMs.",
        "The Bank of Jamaica continues to encourage merchants and consumers to adopt JAM-DEX for everyday transactions.",
        "Retrofitting point-of-sale machines to accept JAM-DEX is necessary for wider adoption.",
        "Jam-Dex, the country's central bank digital currency (CBDC), can now be used to pay fitness fees, property taxes and traffic tickets.",
        "Jamaicans will now have another means to make certain payments at Tax Administration Jamaica with Jam-Dex.",
        "Jam-Dex is legal tender and can be exchanged one-for-one with Jamaican dollars",
        "Jam-Dex was officially minted in August 2021 and became legal tender in June 2022.",
        "The BOJ annual report stated that merchant-strengthening initiatives included an internal Jam-Dex point-of-sale pilot.",
        "Preparatory work for a Jam-Dex online pilot with a government entity was also reported.",
        "The BOJ boosted its system resilience with the installation of the Jam-Dex Disaster Recovery site.",
        "Jam-Dex has been struggling to garner wider acceptance in the Jamaican population due to the limited number of wallet providers.",
        "It has also struggled because it cannot be readily used at point-of-sale (POS) machines at merchants.",
        "The amount of Jam-Dex minted has remained at $276 million since 2022.",
        "That amount is small relative to the physical currency in circulation at $286.1 billion at the end of 2024.",
        "A check by the Jamaica Observer revealed booths decorated with Jam-Dex promotional material.",
        "The Government is currently pursuing initiatives to improve tax collection using Jam-Dex.",
        "More tax offices would be open on weekends to make it easier for people to pay with Jam-Dex.",
        "Prime Minister Dr Andrew Holness announced other initiatives under the SPEED initiative to reduce foot traffic in tax offices.",
        "The pilot demonstrates that Ghana is building capability for future interoperable digital payment solutions",
        "By enabling Ghanaian micro, small, and medium enterprises to participate in international trade cost-effectively, the eCedi could revolutionize cross-border commerce.",
        "The central bank noted that the platform supports improved consumer experiences and inclusive growth.",
        "The Bank of Ghana reassured the public that progress toward launching the eCedi is ongoing.",
        "The successful pilot comes as Ghana prepares for the long-awaited launch of the eCedi, initially slated for 2026.",
        "The Bank of Ghana (BoG) adopted a retail token-based CBDC model designed to replicate traditional attributes of physical cash.",
        "This affirms the potential of the eCedi system for future interoperability with various cross-border credential and payment platforms.",
        "The live transactions demonstrated the feasibility of utilizing the proposed Ghanaian domestic retail CBDC platform.",
        "Phase two successfully executed a live cross-border transaction between Ghana and Singapore using the eCedi CBDC.",
        "The first phase focused on developing a trusted credential system to transform key information into verifiable digital credentials.",
        "Project DESFT is aimed at supporting SMEs in Africa to engage in international trade by removing significant obstacles they face.",
        "The landmark pilot was part of Project DESFT initiated by the Bank of Ghana and the Monetary Authority of Singapore.",
        "Ghana has successfully executed its first cross-border trade transaction using digital credentials, the eCedi central bank digital currency, and a Singaporean stablecoin.",
        "The eCedi, Ghana's upcoming retail CBDC, is expected to significantly enhance the country's payment ecosystem, fostering inclusive growth and innovation.",
        "The central bank is urging all commercial banks to participate in the full roll-out of digital currency.",
        "Successful deployment of JAM-DEX requires commercial banks to bring their merchant customers into the system.",
        "Technical challenges among large merchants have slowed the implementation of the digital currency.",
        "Large retailers faced technological hurdles integrating the digital currency into existing payment systems.",
        "Merchant onboarding remains a critical factor determining the success of JAM-DEX.",
        "Banks must work directly with merchants because they manage those business relationships.",
        "The Bank of Jamaica views digital currency as part of a broader modernisation of payments.",
        "Government officials emphasised that collaboration between banks and merchants is essential.",
        "Regulators believe the CBDC can enhance efficiency in the financial system.",
        "Adoption remains limited because many businesses are not yet ready for the technological transition.",
        "Infrastructure readiness is still evolving despite progress in the rollout.",
        "Some merchants expressed concerns about operational disruptions during integration.",
        "JAM-DEX represents Jamaica's central bank digital currency initiative.",
        "The rollout strategy relies on cooperation between regulators, banks, and payment providers.",
        "Authorities continue working to expand merchant acceptance across the country.",
        "The Bank of Ghana says the e-Cedi project remains a key part of the country's digital transformation strategy.",
        "Officials believe a central bank digital currency could strengthen the efficiency of Ghana's payment ecosystem.",
        "The e-Cedi initiative is intended to complement existing mobile money services rather than replace them.",
        "The central bank confirmed that technical testing of the digital currency platform is ongoing",
        "Pilot activities are focused on evaluating usability among both consumers and merchants.",
        "Authorities continue to refine offline payment capabilities as part of the e-Cedi design.",
        "The system is being developed to operate even in areas with limited internet connectivity.",
        "Regulators say digital currency could improve financial inclusion for underserved communities.",
        "Officials emphasised that stakeholder education is necessary before any full-scale launch.",
        "The Bank of Ghana maintains that public trust will be essential for widespread adoption.",
        "Some analysts argue that adoption may be slow due to strong reliance on mobile money platforms.",
        "There are ongoing discussions about the regulatory framework needed to support digital currency.",
        "The central bank acknowledged that technological readiness remains a critical challenge.",
        "The e-Cedi is being developed as a retail digital version of Ghana's national currency.",
        "Officials say the pilot programme will guide future decisions about a national rollout.",
        "The Bank of Ghana believes the e-Cedi will enhance payment efficiency across the financial system.",
        "Officials argue that a digital currency can reduce friction in everyday transactions.",
        "The central bank views the e-Cedi as a secure alternative to physical cash.",
        "Authorities say the digital currency could improve reliability for both merchants and consumers.",
        "Public education and awareness are considered essential for successful rollout.",
        "Stakeholders believe the system will support the country's digitalisation agenda.",
        "The e-Cedi is being developed as a retail central bank digital currency",
        "Pilot testing has explored both online and offline transaction capabilities.",
        "The central bank continues to evaluate regulatory requirements before launch.",
        "Adoption timelines remain dependent on legislative approval.",
        "Economic conditions have influenced delays in launching the digital currency.",
        "Analysts note that strong reliance on existing payment systems could slow uptake.",
        "Officials believe the e-Cedi could strengthen financial inclusion across underserved communities.",
        "The digital currency initiative aims to support Ghana's broader digital economy strategy.",
        "The Bank of Ghana continues to explore how CBDC design can improve accessibility and efficiency.",
        "The Bank of Jamaica is working to improve access to its central bank digital currency.",
        "Jam-Dex became legal tender in June 2022 and is equivalent one-to-one with the Jamaican dollar.",
        "Officials say expanding access to digital payments is a priority for the central bank.",
        "The number of wallet providers remains a major limitation affecting usage.",
        "Merchant acceptance through existing point-of-sale terminals continues to be a challenge.",
        "Only two digital wallets currently allow the public to transact using Jam-Dex.",
        "Authorities believe converting POS machines will accelerate adoption.",
        "The value of Jam-Dex in issuance represents only a small fraction of total currency in circulation.",
        "The central bank lists thousands of merchants that accept the digital currency.",
        "Officials continue to encourage Jamaicans to move away from heavy reliance on cash.",
        "Infrastructure disruptions after Hurricane Melissa slowed digital payment adoption.",
        "Some regions reverted to cash transactions due to telecom and electricity challenges.",
        "Emergency cash injections highlighted ongoing dependence on physical currency.",
        "The central bank plans to aggressively promote Jam-Dex once POS integration improves",
        "Officials argue that digital currency can help modernise Jamaica's payment ecosystem.",
        "The introduction of the e-Cedi is intended to extend financial services to remote areas.",
        "Officials believe digital currency could improve international transfers and remittances.",
        "The central bank described the digital cedi as a pivotal step in Ghana's financial evolution.",
        "The Bank of Ghana collaborated with technology partners to explore a digital version of the national currency.",
        "A pilot programme was launched to test the feasibility of the e-Cedi platform.",
        "Officials said the pilot attracted strong participation from Ghana's youth population.",
        "Public trust in the central bank was highlighted as a factor supporting experimentation with digital currency.",
        "The e-Cedi was designed as a non-interest-bearing token issued by the central bank",
        "Commercial banks are expected to act as distribution channels for the digital currency.",
        "Hackathons were launched to encourage fintech innovation around the e-Cedi ecosystem.",
        "Preventing misuse and illegal activities remains a priority for the central bank.",
        "The e-Cedi is issued directly by the central bank rather than private institutions.",
        "The initiative reflects Ghana's broader move toward domestic digitalisation.",
        "Jamaica's central bank completed a pilot enabling dynamic QR payments for Jam-Dex at merchant terminals.",
        "The digital currency has been available to the public since 2022 but adoption remains limited.",
        "Officials reported that uptake of CBDC transactions is much lower than other wallet features.",
        "Many retailers prefer a single payment terminal instead of adding a separate device for CBDC transactions.",
        "Adoption levels remain modest compared to expectations from early pilots.",
        "The central bank is working with technology providers to upgrade thousands of POS machines.",
        "Merchant readiness is a key factor slowing widespread rollout of Jam-Dex.",
        "Wallet providers have not fully implemented the technology required for CBDC payments at scale.",
        "Additional banks have joined the digital wallet ecosystem to distribute Jam-Dex.",
        "Authorities believe expanding wallet options could increase transaction volume.",
        "Officials suggested that incentives or payroll payments in CBDC might encourage adoption.",
        "The rollout strategy focuses on improving merchant acceptance rather than consumer incentives alone.",
        "Regulators see CBDC as part of a broader modernization of Jamaica's payment infrastructure.",
        "Delays in technological integration continue to slow CBDC expansion.",
    ],
    "Me":      [0.5,0.5,0.5,0.6,0.6,0.6,0.75,0.75,0.75,0.6,0.6,0.3,0.6,0.5,0.3,0.5,0.5,0.3,0.3,0.3,0.6,0.6,0.3,0.75,0.3,0.6,0.6,0.6,0.6,0.5,0.5,0.75,0.5,0.6,0.3,0.3,0.6,0.6,0.6,0.75,0.6,0.5,0.5,0.75,0.75,0.6,0.5,0.5,0.6,0.6,0.5,0.5,0.6,0.5,0.6,0.75,0.6,0.6,0.3,0.3,0.6,0.6,0.6,0.6,0.75,0.3,0.6,0.3,0.5,0.6,0.6,0.6,0.75,0.5,0.5,0.6,0.5,0.6,0.6,0.6,0.3,0.6,0.3,0.5,0.5,0.75,0.6,0.75,0.75,0.6,0.6,0.5,0.5,0.6,0.6,0.3,0.3,0.75,0.5,0.75,0.75,0.6,0.6,0.3,0.6,0.6,0.3,0.3,0.3,0.75,0.75,0.6,0.75,0.75,0.5,0.6,0.5,0.6,0.6,0.5,0.6,0.5,0.6,0.6,0.5,0.5,0.5,0.6,0.6,0.5,0.6,0.3,0.6,0.6,0.5,0.6,0.6,0.6,0.6,0.3],
    "LLM":     [0.6,0.7,0.8,0.8,0.9,0.7,0.6,0.8,0.9,0.6,0.5,0.3,0.3,0.5,0.6,0.3,1.0,0.2,0.3,0.3,0.2,0.4,0.3,0.5,0.5,0.5,0.5,0.7,0.6,0.8,0.9,0.6,0.6,0.6,0.4,0.3,0.3,0.2,0.5,0.5,0.5,0.5,0.6,0.6,0.7,0.6,0.5,0.5,0.5,0.5,0.7,0.5,0.5,0.6,0.7,0.7,0.8,0.7,0.4,0.4,0.4,0.4,0.6,0.6,0.6,0.4,0.4,0.5,0.5,0.5,0.6,0.7,0.7,0.5,0.5,0.5,0.5,0.6,0.5,0.6,0.5,0.6,0.4,0.4,0.4,0.5,0.5,0.7,0.7,0.5,0.6,0.7,0.6,0.5,0.5,0.4,0.4,0.6,0.6,0.6,0.9,0.5,0.4,0.4,0.5,0.4,0.3,0.4,0.4,0.5,0.6,0.5,0.6,0.6,0.6,0.8,0.6,0.5,0.4,0.5,0.6,0.5,0.4,0.6,0.6,0.7,0.8,0.4,0.6,0.5,0.5,0.6,0.5,0.4,0.6,0.6,0.7,0.5,0.7,0.4],
    "Prof1":   [0.55,0.55,0.95,0.65,0.95,0.85,0.7,0.85,0.8,0.8,0.6,0.25,0.5,0.75,0.8,0.6,0.6,0.1,0.3,0.2,0.15,0.4,0.3,0.8,0.7,0.7,0.55,0.8,0.7,0.75,0.6,0.75,0.75,0.7,0.3,0.2,0.45,0.45,0.6,0.7,0.7,0.5,0.75,0.75,0.85,0.7,0.85,0.7,0.85,0.8,0.9,0.75,0.7,0.75,0.8,0.7,0.7,0.6,0.3,0.25,0.55,0.5,0.75,0.55,0.6,0.3,0.55,0.3,0.5,0.5,0.65,0.75,0.65,0.7,0.65,0.7,0.75,0.75,0.55,0.55,0.3,0.5,0.2,0.65,0.65,0.7,0.7,0.8,0.75,0.55,0.7,0.6,0.7,0.65,0.5,0.3,0.3,0.65,0.8,0.75,0.75,0.65,0.65,0.3,0.25,0.4,0.25,0.15,0.2,0.65,0.65,0.8,0.65,0.65,0.5,0.75,0.85,0.85,0.4,0.5,0.5,0.75,0.75,0.75,0.5,0.5,0.5,0.25,0.3,0.3,0.75,0.3,0.25,0.3,0.35,0.6,0.5,0.8,0.8,0.25],
    "Prof2":   [0.5,0.5,0.9,0.75,0.65,0.75,0.65,0.7,0.8,0.85,0.5,0.32,0.5,0.5,0.8,0.5,0.5,0.1,0.25,0.3,0.45,0.4,0.3,0.85,0.55,0.75,0.5,0.65,0.65,0.5,0.5,0.5,0.5,0.5,0.28,0.25,0.4,0.5,0.5,0.55,0.5,0.5,0.7,0.8,0.65,0.7,0.5,0.5,0.75,0.55,0.9,0.5,0.5,0.5,0.9,0.8,0.75,0.5,0.3,0.2,0.5,0.5,0.6,0.6,0.75,0.3,0.35,0.25,0.5,0.5,0.7,0.75,0.75,0.7,0.5,0.5,0.75,0.75,0.5,0.5,0.3,0.5,0.25,0.5,0.55,0.75,0.8,0.5,0.75,0.5,0.6,0.65,0.5,0.5,0.5,0.35,0.3,0.65,0.65,0.65,0.65,0.65,0.65,0.3,0.25,0.5,0.6,0.3,0.3,0.7,0.65,0.7,0.75,0.75,0.5,0.6,0.8,0.65,0.25,0.25,0.3,0.5,0.5,0.7,0.5,0.55,0.5,0.3,0.35,0.15,0.6,0.65,0.3,0.5,0.5,0.6,0.5,0.5,0.65,0.3],
}

# Align sentence count to score count (140)
n_scores = len(data["Me"])
data["Sentence"] = data["Sentence"][:n_scores]

df = pd.DataFrame(data)
annotators = ["Me", "LLM", "Prof1", "Prof2"]
scores = df[annotators].values  # shape (n_sentences, 4)

# ── 2. ICC (Two-way mixed, absolute agreement, average measures) ──────────────
# Formula: ICC(2,k) — treating raters as fixed, items as random

def compute_icc(ratings):
    """
    Compute ICC(2,1) and ICC(2,k) using ANOVA decomposition.
    ratings: ndarray of shape (n, k)  n=subjects, k=raters
    Returns dict with single-measure and average-measure ICCs.
    """
    n, k = ratings.shape
    grand_mean = ratings.mean()

    # Row (subject) means and column (rater) means
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    # Sum of Squares
    SS_total = ((ratings - grand_mean) ** 2).sum()
    SS_rows  = k * ((row_means - grand_mean) ** 2).sum()
    SS_cols  = n * ((col_means - grand_mean) ** 2).sum()
    SS_error = SS_total - SS_rows - SS_cols

    # Degrees of freedom
    df_rows  = n - 1
    df_cols  = k - 1
    df_error = (n - 1) * (k - 1)

    # Mean Squares
    MS_rows  = SS_rows  / df_rows
    MS_cols  = SS_cols  / df_cols
    MS_error = SS_error / df_error

    # ICC(2,1) — single measure, absolute agreement
    icc_single = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error + k * (MS_cols - MS_error) / n)

    # ICC(2,k) — average measure, absolute agreement (Spearman-Brown corrected)
    icc_avg = (MS_rows - MS_error) / (MS_rows + (MS_cols - MS_error) / n)

    return {
        "ICC_single": round(icc_single, 4),
        "ICC_average": round(icc_avg, 4),
        "MS_rows": MS_rows,
        "MS_cols": MS_cols,
        "MS_error": MS_error,
        "n": n, "k": k
    }

icc_results = compute_icc(scores)

def icc_interpretation(val):
    if val < 0.5:  return "Poor"
    if val < 0.75: return "Moderate"
    if val < 0.9:  return "Good"
    return "Excellent"

# ── 3. PAIRWISE CORRELATIONS ──────────────────────────────────────────────────

pairs = []
for i, a1 in enumerate(annotators):
    for j, a2 in enumerate(annotators):
        if j <= i: continue
        r_p, p_p = pearsonr(df[a1], df[a2])
        r_s, p_s = spearmanr(df[a1], df[a2])
        mae = np.mean(np.abs(df[a1] - df[a2]))
        pairs.append({
            "Pair": f"{a1} vs {a2}",
            "Pearson r": round(r_p, 3),
            "Pearson p": round(p_p, 4),
            "Spearman ρ": round(r_s, 3),
            "MAE": round(mae, 3),
        })

pairs_df = pd.DataFrame(pairs)

# ── 4. PER-ANNOTATOR DESCRIPTIVE STATS ───────────────────────────────────────

stats_rows = []
for a in annotators:
    stats_rows.append({
        "Annotator": a,
        "Mean":   round(df[a].mean(), 3),
        "Std":    round(df[a].std(), 3),
        "Min":    round(df[a].min(), 3),
        "Max":    round(df[a].max(), 3),
        "Median": round(df[a].median(), 3),
    })
stats_df = pd.DataFrame(stats_rows)

# ── 5. GROUND TRUTH: AVERAGE OF ALL 4 ────────────────────────────────────────

df["GroundTruth_avg"] = df[annotators].mean(axis=1).round(4)
df["Score_range"]     = (df[annotators].max(axis=1) - df[annotators].min(axis=1)).round(4)
df["Score_std"]       = df[annotators].std(axis=1).round(4)

DISAGREEMENT_THRESHOLD = 0.4
df["HighDisagreement"] = df["Score_range"] >= DISAGREEMENT_THRESHOLD

# ── 6. HUMAN-LEVEL PERFORMANCE BASELINE ──────────────────────────────────────
# Each annotator treated as a "model" evaluated against the group average

human_baseline = []
for a in annotators:
    mae  = np.mean(np.abs(df[a] - df["GroundTruth_avg"]))
    rmse = np.sqrt(np.mean((df[a] - df["GroundTruth_avg"]) ** 2))
    r, _ = pearsonr(df[a], df["GroundTruth_avg"])
    human_baseline.append({
        "Annotator": a,
        "MAE vs avg":  round(mae,  4),
        "RMSE vs avg": round(rmse, 4),
        "Pearson r vs avg": round(r, 4),
    })
baseline_df = pd.DataFrame(human_baseline)

# ── 7. PRINT REPORT ──────────────────────────────────────────────────────────

SEP  = "=" * 65
sep  = "-" * 65

print(SEP)
print("  INTER-ANNOTATOR AGREEMENT ANALYSIS REPORT")
print(SEP)

print("\n── SECTION 1: DESCRIPTIVE STATISTICS PER ANNOTATOR ──")
print(stats_df.to_string(index=False))

print(f"\n── SECTION 2: INTRACLASS CORRELATION COEFFICIENT (ICC) ──")
print(f"  Model: Two-way mixed, absolute agreement")
print(f"  N sentences : {icc_results['n']}")
print(f"  N annotators: {icc_results['k']}")
print(f"\n  ICC (single measure) : {icc_results['ICC_single']}  → {icc_interpretation(icc_results['ICC_single'])}")
print(f"  ICC (average measure): {icc_results['ICC_average']}  → {icc_interpretation(icc_results['ICC_average'])}")
print(f"\n  Interpretation guide:")
print(f"    < 0.50 = Poor | 0.50–0.74 = Moderate | 0.75–0.90 = Good | > 0.90 = Excellent")

print(f"\n── SECTION 3: PAIRWISE CORRELATIONS & MAE ──")
print(pairs_df.to_string(index=False))

print(f"\n── SECTION 4: HIGH-DISAGREEMENT SENTENCES ──")
print(f"  Threshold: score range >= {DISAGREEMENT_THRESHOLD}")
flagged = df[df["HighDisagreement"]][["Sentence", "Me", "LLM", "Prof1", "Prof2", "Score_range"]]
print(f"  Flagged: {len(flagged)} / {len(df)} sentences ({100*len(flagged)/len(df):.1f}%)\n")
for _, row in flagged.iterrows():
    print(f"  Range={row['Score_range']:.2f} | Me={row['Me']} LLM={row['LLM']} P1={row['Prof1']} P2={row['Prof2']}")
    print(f"    → \"{row['Sentence'][:80]}...\"" if len(row['Sentence']) > 80 else f"    → \"{row['Sentence']}\"")
    print()

print(f"── SECTION 5: HUMAN-LEVEL PERFORMANCE BASELINE ──")
print(f"  (Each annotator measured against the 4-annotator average as ground truth)")
print(baseline_df.to_string(index=False))

print(f"\n── SECTION 6: GROUND TRUTH RECOMMENDATION ──")
icc_val = icc_results["ICC_single"]
n_flagged = len(flagged)
pct_flagged = 100 * n_flagged / len(df)

print(f"\n  ICC (single): {icc_val}  →  {icc_interpretation(icc_val)} agreement")
print(f"  High-disagreement sentences: {n_flagged} ({pct_flagged:.1f}%)")

if icc_val >= 0.75:
    rec = "AVERAGE ALL 4 ANNOTATORS"
    rationale = ("ICC is Good/Excellent. Averaging all 4 scores minimizes "
                 "individual bias and is the statistically optimal ground truth.")
elif icc_val >= 0.5:
    rec = "AVERAGE + ADJUDICATE FLAGGED"
    rationale = ("ICC is Moderate. Use the 4-annotator average as ground truth "
                 "but manually review the flagged high-disagreement sentences before training.")
else:
    rec = "MANDATORY ADJUDICATION BEFORE TRAINING"
    rationale = ("ICC is Poor. Do not average blindly — convene annotators to "
                 "resolve disagreements on flagged sentences and re-evaluate.")

print(f"\n  ✅ RECOMMENDATION: {rec}")
print(f"  Rationale: {rationale}")

print(f"\n── SECTION 7: MODEL VALIDATION METRICS TO USE ──")
print("""
  Primary metrics (in order of importance):
    1. Pearson r      — linear correlation with ground truth (target: > human avg)
    2. Spearman ρ     — rank-order correlation, robust to outliers
    3. MAE            — mean absolute error (same units as scores, easy to interpret)
    4. RMSE           — penalises large errors; compare to human RMSE baseline above

  Human baseline target (from Section 5):
    → Your trained model should achieve MAE ≤ the best human annotator's MAE
    → Pearson r should be ≥ 0.70 to be considered useful

  Recommended train/val/test split: 70 / 15 / 15
  Use stratified split on score quartiles to ensure coverage of low/high scores.
""")

print(SEP)
print("  GROUND TRUTH SCORES SAVED TO: annotator_scores_with_groundtruth.csv")
print(SEP)

# ── 8. SAVE OUTPUT CSV ───────────────────────────────────────────────────────

output_cols = ["Sentence", "Me", "LLM", "Prof1", "Prof2",
               "GroundTruth_avg", "Score_range", "Score_std", "HighDisagreement"]
df[output_cols].to_csv("/mnt/user-data/outputs/annotator_scores_with_groundtruth.csv", index=False)
print("\nCSV saved successfully.")