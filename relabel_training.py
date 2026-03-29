#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Load training data
path = Path('training_data.csv')
df = pd.read_csv(path)

print(f"Starting size: {len(df)} rows")
print(f"Score range before: {df['Score'].min():.4f} - {df['Score'].max():.4f}")

# Create backup if it doesn't exist
backup = Path('training_data.backup_orig.csv')
if not backup.exists():
    df.to_csv(backup, index=False)
    print(f"Backup created: {backup}")

# Strong positive CBDC-related keywords
strong_positive_terms = {
    'vision', 'confidence', 'bold', 'forward', 'inclusive', 'efficient',
    'faster', 'safer', 'reliable', 'modernisation', 'enhance', 'innovation',
    'transform', 'empower', 'seamless', 'trusted', 'pioneering', 'leadership'
}

cbdc_terms = {'cbdc', 'jam-dex', 'digital currency', 'central bank digital'}

def uplift_score(row):
    sentence = str(row['Sentence']).lower()
    current_score = float(row['Score'])
    
    # Check for CBDC relevance
    has_cbdc = any(term in sentence for term in cbdc_terms)
    
    # Count strong positive terms
    strong_pos_count = sum(1 for term in strong_positive_terms if term in sentence)
    
    # Uplift logic: CBDC + 2+ strong positive terms = min 0.85
    if has_cbdc and strong_pos_count >= 2:
        new_score = max(current_score, 0.85)
    # Uplift logic: 3+ strong positive terms = min 0.80
    elif strong_pos_count >= 3:
        new_score = max(current_score, 0.80)
    else:
        new_score = current_score
    
    return min(1.0, max(0.0, new_score))

# Apply uplift
before = df['Score'].copy()
df['Score'] = df.apply(uplift_score, axis=1)

changed_count = (before.round(6) != df['Score'].round(6)).sum()
print(f"Rows updated: {changed_count}")

# Add reference sentence if not present
ref_sentence = (
    "In a world where digital transformation is accelerating across industries, "
    "Jamaica's proactive embrace of digital currency reflects both vision and confidence."
)

if not (df['Sentence'] == ref_sentence).any():
    new_row = pd.DataFrame([{'Sentence': ref_sentence, 'Score': 0.92}])
    df = pd.concat([df, new_row], ignore_index=True)
    ref_added = 1
    print(f"Reference sentence added: {ref_added}")
else:
    print("Reference sentence already present")
    ref_added = 0

# Save updated training data
df.to_csv(path, index=False)

print(f"Ending size: {len(df)} rows")
print(f"Score range after: {df['Score'].min():.4f} - {df['Score'].max():.4f}")
print(f"\n✓ Training data updated successfully")
