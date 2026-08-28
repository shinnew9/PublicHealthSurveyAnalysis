"""
Linguistic marker sensitivity check (Section 5, Construct Validity).

Tests whether GPT-generated profiles respond systematically to input
covariates (insurance status, age group) via five linguistic markers, rather
than producing input-invariant text.
"""
import re
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = "nsduh_analysis_outputs"
master = pd.read_csv(f"{OUT_DIR}/df_corrected_7970_with_gpt_profiles_embeddings.csv")


def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def linguistic_features(text):
    text = str(text) if not pd.isna(text) else ""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    return pd.Series({
        "word_count": n_words,
        "words_per_sentence": n_words / n_sentences,
        "avg_word_length": np.mean([len(w) for w in words]) if words else np.nan,
        "syllables_per_word": np.mean([count_syllables(w) for w in words]) if words else np.nan,
        "unique_word_ratio": len(set(w.lower() for w in words)) / n_words if n_words else np.nan,
    })


feats = master["predictive_profile"].apply(linguistic_features)
df = pd.concat([master[["IRINSUR4", "AGE3"]], feats], axis=1)

MARKERS = ["word_count", "words_per_sentence", "avg_word_length", "syllables_per_word", "unique_word_ratio"]

# Insurance status: Mann-Whitney U
insured_mask = df["IRINSUR4"].str.startswith("1")
uninsured_mask = df["IRINSUR4"].str.startswith("2")

print("=== Insured vs Uninsured ===")
rows = []
for marker in MARKERS:
    g1 = df.loc[insured_mask, marker].dropna()
    g2 = df.loc[uninsured_mask, marker].dropna()
    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    rows.append(dict(marker=marker, insured_mean=g1.mean(), uninsured_mean=g2.mean(),
                      diff=g1.mean() - g2.mean(), p_value=p))
insurance_result = pd.DataFrame(rows)
print(insurance_result.to_string(index=False))

# Age group (ordinal): Spearman correlation
age_order = {
    "1 - Respondent is 12 or 13 years old": 1,
    "2 - Respondent is 14 or 15 years old": 2,
    "3 - Respondent is 16 or 17 years old": 3,
    "4 - Respondent is between 18 and 20 years old": 4,
    "5 - Respondent is between 21 and 23 years old": 5,
    "6 - Respondent is 24 or 25 years old": 6,
    "7 - Respondent is between 26 and 29 years old": 7,
    "8 - Respondent is between 30 and 34 years old": 8,
    "9 - Respondent is between 35 and 49 years old": 9,
    "10 - Respondent is between 50 and 64 years old": 10,
    "11 - Respondent is 65 years old or older": 11,
}
df["age_ordinal"] = df["AGE3"].map(age_order)

print("\n=== Age group correlation (Spearman) ===")
rows = []
for marker in MARKERS:
    valid = df[["age_ordinal", marker]].dropna()
    rho, p = stats.spearmanr(valid["age_ordinal"], valid[marker])
    rows.append(dict(marker=marker, spearman_rho=rho, p_value=p))
age_result = pd.DataFrame(rows)
print(age_result.to_string(index=False))

insurance_result.to_csv(f"{OUT_DIR}/linguistic_marker_insurance.csv", index=False)
age_result.to_csv(f"{OUT_DIR}/linguistic_marker_age.csv", index=False)
print(f"\nsaved: {OUT_DIR}/linguistic_marker_insurance.csv, {OUT_DIR}/linguistic_marker_age.csv")
