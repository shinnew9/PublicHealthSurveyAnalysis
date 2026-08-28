"""
Cross-respondent embedding diversity check (Section 5, Construct Validity).

Compares within-respondent reliability (repeated GPT rewrites of the same
respondent) against cross-respondent diversity (pairwise similarity across
distinct respondents), to test whether profiles collapse into a generic
template rather than responding to each respondent's input fields.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "nsduh_analysis_outputs"

# Within-respondent: reuse the reliability re-validation results (n=300, 10 reruns each)
reliability_df = pd.read_csv(f"{OUT_DIR}/reliability_revalidation_results.csv")
within_resp = reliability_df["mean_pairwise_cosine"].dropna().values
print("Within-respondent: n =", len(within_resp), "mean =", within_resp.mean())

# Cross-respondent: pairwise similarity across 300 distinct respondents
master = pd.read_csv(f"{OUT_DIR}/df_corrected_7970_with_gpt_profiles_embeddings.csv")
sample = master.sample(n=300, random_state=42)
X = np.vstack(sample["predictive_embedding"].apply(json.loads).values).astype("float32")

Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
sim_matrix = Xn @ Xn.T
iu = np.triu_indices(len(sample), k=1)
cross_resp = sim_matrix[iu]
print("Cross-respondent: n_pairs =", len(cross_resp), "mean =", cross_resp.mean())

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

axes[0].hist(within_resp, bins=30, color="#4C72B0", alpha=0.8)
axes[0].set_title(f"Within-respondent\n(mean = {within_resp.mean():.4f}, n={len(within_resp)})")
axes[0].set_xlabel("Cosine similarity")
axes[0].set_ylabel("Count")

axes[1].hist(cross_resp, bins=30, color="#C44E52", alpha=0.8)
axes[1].set_title(f"Cross-respondent\n(mean = {cross_resp.mean():.4f}, n={len(cross_resp)})")
axes[1].set_xlabel("Cosine similarity")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/diversity_check_histogramn.png", dpi=150)
print(f"saved: {OUT_DIR}/diversity_check_histogramn.png")
