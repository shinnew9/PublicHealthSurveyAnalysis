"""
03: (1) Wilcoxon signed-rank test on cross-year transfer drop differences (5 classifiers)
    (2) 2022 vs 2023 asymmetry investigation:
        - sample sizes & class balance
        - per-variable distribution tests (chi-square + Cramer's V)
        - adversarial validation (can a model tell 2022 from 2023?)
Outputs go to ./nsduh_analysis_outputs/
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest, chi2_contingency

OUT_DIR = Path("./nsduh_analysis_outputs")

# ---------------------------------------------------------------
# Part 1: Wilcoxon on transfer drop differences (GPT drop - Raw drop)
# ---------------------------------------------------------------
print("=" * 70)
print("PART 1: Wilcoxon signed-rank test on transfer drop differences")
print("=" * 70)

summary = pd.read_csv(OUT_DIR / "cross_year_drop_summary_repeated_within_year_5seeds_by_model.csv")
pivot = pd.read_csv(OUT_DIR / "cross_year_drop_comparison_repeated_within_year_5seeds_pivot.csv")

rows = []

def run_tests(diffs, label):
    diffs = np.asarray(diffs, dtype=float)
    # Hypothesis: GPT embeddings degrade LESS than raw -> differences < 0
    w_less = wilcoxon(diffs, alternative="less", zero_method="wilcox")
    w_two = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
    n_neg = int(np.sum(diffs < 0))
    n_nonzero = int(np.sum(diffs != 0))
    sign = binomtest(k=n_neg, n=n_nonzero, p=0.5, alternative="greater")
    rows.append({
        "Comparison": label,
        "n": len(diffs),
        "mean_diff": diffs.mean(),
        "median_diff": np.median(diffs),
        "n_negative(favor GPT)": n_neg,
        "wilcoxon_stat": w_less.statistic,
        "wilcoxon_p_one_sided_less": w_less.pvalue,
        "wilcoxon_p_two_sided": w_two.pvalue,
        "sign_test_p_one_sided": sign.pvalue,
    })
    print(f"\n{label} (n={len(diffs)})")
    print(f"  diffs: {np.round(diffs, 4).tolist()}")
    print(f"  one-sided (GPT drops less) p = {w_less.pvalue:.4f}")
    print(f"  two-sided               p = {w_two.pvalue:.4f}")
    print(f"  sign test one-sided     p = {sign.pvalue:.4f}")

# Averaged over both directions (as in the summary table)
run_tests(summary["GPT_drop_minus_Raw_drop_mean"], "GPT vs Raw drop, mean of both directions")
run_tests(summary["Combined_drop_minus_Raw_drop_mean"], "Combined vs Raw drop, mean of both directions")

# Each direction separately
for exp, g in pivot.groupby("Experiment"):
    run_tests(g["GPT_drop_minus_Raw_drop"], f"GPT vs Raw drop, {exp}")

# Excluding Logistic Regression (tree-based only) as a sensitivity check
tree = summary[summary["Model"] != "Logistic Regression"]
run_tests(tree["GPT_drop_minus_Raw_drop_mean"], "GPT vs Raw drop, tree-based only (n=4)")

res1 = pd.DataFrame(rows)
res1.to_csv(OUT_DIR / "transfer_drop_wilcoxon_results.csv", index=False)
print(f"\nSaved: {OUT_DIR/'transfer_drop_wilcoxon_results.csv'}")

# NOTE on power: with n=5 the smallest achievable one-sided exact p is 1/32 = 0.03125,
# and two-sided is 0.0625. Report honestly.

# ---------------------------------------------------------------
# Part 2: 2022 vs 2023 asymmetry investigation
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: 2022 vs 2023 asymmetry investigation")
print("=" * 70)

cols = ["year", "AGE3", "NEWRACE2", "IRINSUR4",
        "substance_peer_support", "mental_health_peer_support", "cost_barrier"]
df = pd.read_csv(OUT_DIR / "df_corrected_7970_with_gpt_profiles_embeddings.csv", usecols=cols)
print(f"\nLoaded {len(df)} rows")

# 2a. Sample sizes and class balance
print("\n--- Sample size and target (cost_barrier) prevalence by year ---")
bal = df.groupby("year").agg(
    n=("cost_barrier", "size"),
    positives=("cost_barrier", "sum"),
    prevalence=("cost_barrier", "mean"),
)
print(bal)
# chi-square on target by year
ct_target = pd.crosstab(df["year"], df["cost_barrier"])
chi2, p_target, dof, _ = chi2_contingency(ct_target)
print(f"\nChi-square target-by-year: chi2={chi2:.3f}, p={p_target:.4f}")
bal.to_csv(OUT_DIR / "year_asymmetry_class_balance.csv")

# 2b. Per-variable distribution comparison
def cramers_v(ct):
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

print("\n--- Variable distribution differences between years ---")
var_rows = []
for col in ["AGE3", "NEWRACE2", "IRINSUR4", "substance_peer_support",
            "mental_health_peer_support", "cost_barrier"]:
    ct = pd.crosstab(df["year"], df[col])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = cramers_v(ct)
    # max absolute proportion difference across categories
    prop = ct.div(ct.sum(axis=1), axis=0)
    max_prop_diff = float((prop.loc[2022] - prop.loc[2023]).abs().max())
    var_rows.append({"variable": col, "chi2": chi2, "dof": dof, "p_value": p,
                     "cramers_v": v, "max_category_prop_diff": max_prop_diff})
    print(f"{col:30s} chi2={chi2:8.2f} p={p:.2e} CramersV={v:.4f} maxPropDiff={max_prop_diff:.4f}")

res2 = pd.DataFrame(var_rows)
res2.to_csv(OUT_DIR / "year_asymmetry_variable_tests.csv", index=False)

# Detailed proportion tables for variables that differ
print("\n--- Category proportions by year (all variables) ---")
prop_tables = []
for col in ["AGE3", "NEWRACE2", "IRINSUR4", "substance_peer_support",
            "mental_health_peer_support"]:
    prop = pd.crosstab(df["year"], df[col], normalize="index").T
    prop.columns = [f"prop_{c}" for c in prop.columns]
    prop["abs_diff"] = (prop.iloc[:, 0] - prop.iloc[:, 1]).abs()
    prop = prop.sort_values("abs_diff", ascending=False)
    prop.insert(0, "variable", col)
    prop_tables.append(prop.reset_index(names="category"))
pd.concat(prop_tables).to_csv(OUT_DIR / "year_asymmetry_category_proportions.csv", index=False)
print(f"Saved: {OUT_DIR/'year_asymmetry_category_proportions.csv'}")

# 2c. Adversarial validation: predict year from raw structured variables
print("\n--- Adversarial validation: can a model distinguish 2022 vs 2023? ---")
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

X_raw = df[["AGE3", "NEWRACE2", "IRINSUR4", "substance_peer_support",
            "mental_health_peer_support"]]
X = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(X_raw)
y = (df["year"] == 2023).astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
adv_rows = []
for name, clf in [
    ("Logistic Regression", LogisticRegression(max_iter=2000)),
    ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
]:
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    adv_rows.append({"model": name, "adv_auc_mean": aucs.mean(), "adv_auc_std": aucs.std()})
    print(f"{name:25s} year-classification AUC = {aucs.mean():.4f} (+/- {aucs.std():.4f})")

# feature importances for interpretation
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
Xf = enc.fit_transform(X_raw)
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xf, y)
imp = pd.Series(rf.feature_importances_, index=enc.get_feature_names_out()).sort_values(ascending=False)
print("\nTop 10 features distinguishing years:")
print(imp.head(10))
imp.to_csv(OUT_DIR / "year_asymmetry_adversarial_feature_importance.csv")
pd.DataFrame(adv_rows).to_csv(OUT_DIR / "year_asymmetry_adversarial_validation.csv", index=False)

print("\nDone. All outputs saved to nsduh_analysis_outputs/")
