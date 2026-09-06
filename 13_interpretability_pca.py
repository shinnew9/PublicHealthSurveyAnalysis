"""
PCA interpretability analysis (Section 7).

Computes explained variance per principal component of the GPT profile
embeddings and tests which components are most associated with cost_barrier,
to determine where cost-barrier-related signal lives in the embedding space.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, mannwhitneyu
from sklearn.decomposition import PCA

OUT_DIR = "nsduh_analysis_outputs"

df = pd.read_csv(f"{OUT_DIR}/df_corrected_7970_with_gpt_profiles_embeddings.csv")
X = np.vstack(df["predictive_embedding"].apply(json.loads).values).astype("float32")
y = df["cost_barrier"].astype(int).values

N_COMPONENTS = 20
pca = PCA(n_components=N_COMPONENTS, random_state=42)
pcs = pca.fit_transform(X)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

results = []
for i in range(N_COMPONENTS):
    r, p_corr = pointbiserialr(y, pcs[:, i])
    u_stat, p_mwu = mannwhitneyu(pcs[y == 1, i], pcs[y == 0, i])
    results.append({
        "PC": i + 1,
        "explained_variance_ratio": explained[i],
        "cumulative_variance": cumulative[i],
        "point_biserial_r": r,
        "p_value_corr": p_corr,
        "p_value_mwu": p_mwu,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/interpretability_pca_variance.csv", index=False)
print(results_df.to_string(index=False))

print("\nTop PC most associated with cost_barrier (by |r|):")
top = results_df.reindex(results_df["point_biserial_r"].abs().sort_values(ascending=False).index)
print(top.head(5).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, N_COMPONENTS + 1), explained * 100, color="#4C72B0")
ax.plot(range(1, N_COMPONENTS + 1), cumulative * 100, color="#C44E52", marker="o", label="Cumulative")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance (%)")
ax.set_title("PCA Explained Variance — GPT Profile Embeddings (n=7,970)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/interpretability_pca_scree.png", dpi=150)
print(f"saved: {OUT_DIR}/interpretability_pca_scree.png")
