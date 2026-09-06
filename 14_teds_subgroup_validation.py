"""
External validity check against TEDS-A (Section 8).

Compares NSDUH-predicted cost-barrier probability (from the GPT profile
embedding model) against an independently observed TEDS-A admission rate,
across demographic subgroups (age group, race/ethnicity, insurance status).

Design (per Prof. Lee, 2026-09-06):
  - TEDS admission rate = TEDS-A admissions in a subgroup / Census population
    estimate for that subgroup (Option 1). The Census denominator includes
    people without substance use disorders -- an acknowledged limitation of
    this standard epidemiological approach.
  - Primary result: Spearman rank correlation between NSDUH-predicted
    probability and TEDS admission rate, pooled across all subgroup
    categories (age + race/ethnicity + insurance status).
  - Expected direction (stated before running, per instruction): NEGATIVE.
    Subgroups with higher predicted cost barriers are, by construction,
    subgroups who report being unable to afford treatment, so they should
    show LOWER TEDS admission rates (barrier -> less treatment access).
  - Confound to flag explicitly: subgroups with higher underlying substance
    use disorder (SUD) prevalence may show both higher cost barriers AND
    higher admission volume simultaneously, which would push the observed
    correlation in the POSITIVE direction. A result in either direction is
    interpretable as long as this confound is discussed.
  - Insurance status crosswalk caveat: TEDS-A records PRIMPAY (primary source
    of payment), not insurance status directly. We map PRIMPAY to an
    insured/uninsured proxy (documented below) rather than treating it as
    equivalent to NSDUH's IRINSUR4.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

NSDUH_DATA_DIR = "nsduh_analysis_outputs"
TEDS_DIR = "TEDS_A_2021_2023_csv"
CENSUS_DIR = "census_data"

AGE_BINS = ["12-17", "18-20", "21-29", "30-34", "35-49", "50-64", "65+"]
RACE_GROUPS = [
    "NonHisp White", "NonHisp Black/Afr Am", "NonHisp Native Am/AK Native",
    "NonHisp Native HI/Other Pac Isl", "NonHisp Asian",
    "NonHisp more than one race", "Hispanic",
]

# ============================================================
# 1. NSDUH: 5-fold cross-validated predicted probability of cost_barrier
#    from the GPT profile embedding (logistic regression, standardized).
#    Cross-validated (rather than in-sample) predictions avoid overfitting
#    bias when aggregating to subgroup means; the full n=7,970 sample is
#    used (rather than just the 20% test split) so small subgroups still
#    have enough respondents for a stable mean.
# ============================================================
print("=== Step 1: NSDUH cross-validated predicted probabilities ===")
nsduh = pd.read_csv(f"{NSDUH_DATA_DIR}/df_corrected_7970_with_gpt_profiles_embeddings.csv")

CACHE_PATH = f"{NSDUH_DATA_DIR}/nsduh_pred_proba_cache.csv"
import os
if os.path.exists(CACHE_PATH):
    print(f"Loading cached predicted probabilities from {CACHE_PATH}")
    cache = pd.read_csv(CACHE_PATH)
    nsduh["pred_proba"] = cache["pred_proba"].values
else:
    X = np.vstack(nsduh["predictive_embedding"].apply(json.loads).values).astype("float32")
    y = nsduh["cost_barrier"].astype(int).values

    pred_proba = np.zeros(len(nsduh))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(solver="saga", class_weight="balanced", max_iter=2000, random_state=42)
        clf.fit(X_train, y[train_idx])
        pred_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    nsduh["pred_proba"] = pred_proba
    pd.DataFrame({"pred_proba": pred_proba}).to_csv(CACHE_PATH, index=False)
    print(f"Cached predicted probabilities to {CACHE_PATH}")

print("Mean predicted probability:", nsduh["pred_proba"].mean())

# --- NSDUH subgroup assignment ---
NSDUH_AGE_MAP = {
    "1 - Respondent is 12 or 13 years old": "12-17",
    "2 - Respondent is 14 or 15 years old": "12-17",
    "3 - Respondent is 16 or 17 years old": "12-17",
    "4 - Respondent is between 18 and 20 years old": "18-20",
    "5 - Respondent is between 21 and 23 years old": "21-29",
    "6 - Respondent is 24 or 25 years old": "21-29",
    "7 - Respondent is between 26 and 29 years old": "21-29",
    "8 - Respondent is between 30 and 34 years old": "30-34",
    "9 - Respondent is between 35 and 49 years old": "35-49",
    "10 - Respondent is between 50 and 64 years old": "50-64",
    "11 - Respondent is 65 years old or older": "65+",
}
NSDUH_RACE_MAP = {
    "1 - NonHisp White": "NonHisp White",
    "2 - NonHisp Black/Afr Am": "NonHisp Black/Afr Am",
    "3 - NonHisp Native Am/AK Native": "NonHisp Native Am/AK Native",
    "4 - NonHisp Native HI/Other Pac Isl": "NonHisp Native HI/Other Pac Isl",
    "5 - NonHisp Asian": "NonHisp Asian",
    "6 - NonHisp more than one race": "NonHisp more than one race",
    "7 - Hispanic": "Hispanic",
}

nsduh["age_bin"] = nsduh["AGE3"].map(NSDUH_AGE_MAP)
nsduh["race_group"] = nsduh["NEWRACE2"].map(NSDUH_RACE_MAP)
nsduh["insured"] = nsduh["IRINSUR4"].str.startswith("1").map({True: "Insured", False: "Uninsured"})

nsduh_by_age = nsduh.groupby("age_bin")["pred_proba"].mean()
nsduh_by_race = nsduh.groupby("race_group")["pred_proba"].mean()
nsduh_by_insurance = nsduh.groupby("insured")["pred_proba"].mean()

# ============================================================
# 2. TEDS-A 2022+2023: admissions per subgroup category
# ============================================================
print("\n=== Step 2: TEDS-A admissions by subgroup ===")
teds_2022 = pd.read_csv(f"{TEDS_DIR}/TEDS_A_2022.csv", low_memory=False)
teds_2023 = pd.read_csv(f"{TEDS_DIR}/TEDS_A_2023.csv", low_memory=False)
teds = pd.concat([teds_2022, teds_2023], ignore_index=True)
print("TEDS-A 2022+2023 admissions:", len(teds))

TEDS_AGE_MAP = {
    "12-14 years": "12-17", "15-17 years": "12-17",
    "18-20 years": "18-20",
    "21-24 years": "21-29", "25-29 years": "21-29",
    "30-34 years": "30-34",
    "35-39 years": "35-49", "40-44 years": "35-49", "45-49 years": "35-49",
    "50-54 years": "50-64", "55-64 years": "50-64",
    "65 years and older": "65+",
}
# TEDS-A's AGE labels contain a non-standard en-dash byte (0x96) between the
# bounds (e.g. "12\x9614 years") instead of a plain hyphen; normalize before
# mapping so the crosswalk above can use readable keys.
teds["age_bin"] = teds["AGE"].str.replace("\x96", "-", regex=False).map(TEDS_AGE_MAP)
n_age_unmapped = teds["age_bin"].isna().sum()
print(f"TEDS age unmapped (excluded): {n_age_unmapped} ({n_age_unmapped/len(teds):.1%})")

# Race/ethnicity crosswalk: NSDUH's NEWRACE2 is a single combined
# race/Hispanic-origin variable; TEDS-A records RACE and ETHNIC separately.
# Anyone with a Hispanic ETHNIC code is assigned "Hispanic" regardless of
# RACE, matching NEWRACE2's convention. "Asian or Pacific Islander" (a
# combined pre-1997-OMB TEDS code) and "Other single race" have no clean
# NSDUH analogue and are left unmapped (excluded from the race/ethnicity
# subgroup analysis) rather than guessed into a bucket -- a data limitation
# to state explicitly in the manuscript.
HISPANIC_ETHNIC_CODES = {
    "Mexican", "Puerto Rican", "Cuban or other specific Hispanic",
    "Hispanic or Latino, specific origin not specified",
}
TEDS_RACE_MAP = {
    "White": "NonHisp White",
    "Black or African American": "NonHisp Black/Afr Am",
    "American Indian (other than Alaska Native)": "NonHisp Native Am/AK Native",
    "Alaska Native (Aleut, Eskimo)": "NonHisp Native Am/AK Native",
    "Asian": "NonHisp Asian",
    "Native Hawaiian or Other Pacific Islander": "NonHisp Native HI/Other Pac Isl",
    "Two or more races": "NonHisp more than one race",
}


def assign_race_group(row):
    if row["ETHNIC"] in HISPANIC_ETHNIC_CODES:
        return "Hispanic"
    return TEDS_RACE_MAP.get(row["RACE"])


teds["race_group"] = teds.apply(assign_race_group, axis=1)

# Insurance crosswalk: TEDS-A records PRIMPAY (primary source of payment for
# this treatment episode), not insurance coverage directly. We treat
# Medicaid/Medicare/private insurance/other government payments as an
# "Insured" proxy, and self-pay/no-charge(charity) as an "Uninsured" proxy;
# "Other" and missing/unknown are excluded as not classifiable. This is an
# approximation, not a direct measurement of insurance status, and should be
# reported as a limitation.
TEDS_INSURED_MAP = {
    "Medicaid": "Insured",
    "Medicare": "Insured",
    "Private insurance (Blue Cross/Blue Shield, other health insurance, workers compensation)": "Insured",
    "Other government payments": "Insured",
    "Self-pay": "Uninsured",
    "No charge (free, charity, special research, teaching)": "Uninsured",
}
teds["insured"] = teds["PRIMPAY"].map(TEDS_INSURED_MAP)

n_race_unmapped = teds["race_group"].isna().sum()
n_insured_unmapped = teds["insured"].isna().sum()
print(f"TEDS race/ethnicity unmapped (excluded): {n_race_unmapped} ({n_race_unmapped/len(teds):.1%})")
print(f"TEDS insurance unmapped (excluded): {n_insured_unmapped} ({n_insured_unmapped/len(teds):.1%})")

teds_admissions_by_age = teds.groupby("age_bin").size()
teds_admissions_by_race = teds.groupby("race_group").size()
teds_admissions_by_insurance = teds.groupby("insured").size()

# ============================================================
# 3. Census population denominators (2022+2023 combined)
# ============================================================
print("\n=== Step 3: Census population denominators ===")
census_age_race = pd.concat([
    pd.read_csv(f"{CENSUS_DIR}/nc-est2023-alldata-r-2022jul.csv"),
    pd.read_csv(f"{CENSUS_DIR}/nc-est2023-alldata-r-2023jul.csv"),
])
census_age_race = census_age_race[(census_age_race["MONTH"] == 7) & (census_age_race["AGE"] != 999)]

AGE_BIN_RANGES = {
    "12-17": range(12, 18), "18-20": range(18, 21), "21-29": range(21, 30),
    "30-34": range(30, 35), "35-49": range(35, 50), "50-64": range(50, 65),
    "65+": range(65, 101),
}
census_pop_by_age = {}
for age_bin, ages in AGE_BIN_RANGES.items():
    census_pop_by_age[age_bin] = census_age_race.loc[
        census_age_race["AGE"].isin(list(ages)), "TOT_POP"
    ].sum()

RACE_CENSUS_COLS = {
    "NonHisp White": ("NHWA_MALE", "NHWA_FEMALE"),
    "NonHisp Black/Afr Am": ("NHBA_MALE", "NHBA_FEMALE"),
    "NonHisp Native Am/AK Native": ("NHIA_MALE", "NHIA_FEMALE"),
    "NonHisp Native HI/Other Pac Isl": ("NHNA_MALE", "NHNA_FEMALE"),
    "NonHisp Asian": ("NHAA_MALE", "NHAA_FEMALE"),
    "NonHisp more than one race": ("NHTOM_MALE", "NHTOM_FEMALE"),
    "Hispanic": ("H_MALE", "H_FEMALE"),
}
census_pop_by_race = {}
for race_group, (male_col, female_col) in RACE_CENSUS_COLS.items():
    census_pop_by_race[race_group] = census_age_race[male_col].sum() + census_age_race[female_col].sum()

# Insurance: U.S. Census Bureau, Table HI05_ACS ("United States, All people"
# row), ACS 1-year estimates, civilian noninstitutionalized population
# (thousands), summed across the 2022 and 2023 data years to match the
# combined TEDS/NSDUH sample:
#   2022: total=328,300k, insured=301,900k -> uninsured=26,400k
#   2023: total=330,000k, insured=303,800k -> uninsured=26,200k
census_pop_by_insurance = {
    "Insured": (301_900 + 303_800) * 1000,
    "Uninsured": (26_400 + 26_200) * 1000,
}

# ============================================================
# 4. Build subgroup comparison table and compute Spearman correlation
# ============================================================
print("\n=== Step 4: Subgroup comparison ===")


def build_table(pred_series, admissions_series, pop_dict, categories):
    rows = []
    for cat in categories:
        pred = pred_series.get(cat, np.nan)
        admissions = admissions_series.get(cat, 0)
        pop = pop_dict.get(cat, np.nan)
        rows.append({
            "category": cat,
            "nsduh_pred_proba": pred,
            "teds_admissions": admissions,
            "census_population": pop,
            "teds_rate_per_1000": admissions / pop * 1000 if pop else np.nan,
        })
    return pd.DataFrame(rows)


age_table = build_table(nsduh_by_age, teds_admissions_by_age, census_pop_by_age, AGE_BINS)
race_table = build_table(nsduh_by_race, teds_admissions_by_race, census_pop_by_race, RACE_GROUPS)
insurance_table = build_table(nsduh_by_insurance, teds_admissions_by_insurance, census_pop_by_insurance,
                               ["Insured", "Uninsured"])

for name, table in [("Age group", age_table), ("Race/ethnicity", race_table), ("Insurance status", insurance_table)]:
    print(f"\n--- {name} ---")
    print(table.to_string(index=False))

age_table.to_csv(f"{NSDUH_DATA_DIR}/teds_subgroup_age.csv", index=False)
race_table.to_csv(f"{NSDUH_DATA_DIR}/teds_subgroup_race.csv", index=False)
insurance_table.to_csv(f"{NSDUH_DATA_DIR}/teds_subgroup_insurance.csv", index=False)

# Primary result: pooled Spearman correlation across all subgroup categories
pooled = pd.concat([age_table, race_table, insurance_table], ignore_index=True).dropna(
    subset=["nsduh_pred_proba", "teds_rate_per_1000"]
)
rho, p = spearmanr(pooled["nsduh_pred_proba"], pooled["teds_rate_per_1000"])
print(f"\n=== PRIMARY RESULT: pooled Spearman correlation (n={len(pooled)} subgroups) ===")
print(f"rho = {rho:.4f}, p = {p:.4f}")

# Secondary: within-variable-type breakdowns (age n=7, race n=7 both usable;
# insurance n=2 is too small for a meaningful rank correlation on its own
# and is reported descriptively instead).
for name, table in [("Age group", age_table), ("Race/ethnicity", race_table)]:
    sub = table.dropna(subset=["nsduh_pred_proba", "teds_rate_per_1000"])
    rho_sub, p_sub = spearmanr(sub["nsduh_pred_proba"], sub["teds_rate_per_1000"])
    print(f"{name} only (n={len(sub)}): rho = {rho_sub:.4f}, p = {p_sub:.4f}")

pooled.to_csv(f"{NSDUH_DATA_DIR}/teds_subgroup_pooled.csv", index=False)
print(f"\nsaved: {NSDUH_DATA_DIR}/teds_subgroup_{{age,race,insurance,pooled}}.csv")
