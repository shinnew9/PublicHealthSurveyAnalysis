import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap

warnings.filterwarnings("ignore")

df = pd.read_csv("nsduh_analysis_outputs/df_corrected_7970_with_gpt_profiles_embeddings.csv")
X = np.vstack(df["predictive_embedding"].apply(json.loads).values).astype("float32")
y = df["cost_barrier"].astype(int).values

pca_2d = PCA(n_components=2, random_state=42).fit_transform(X)
pca_sil = silhouette_score(pca_2d, y)
print("PCA silhouette:", pca_sil)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
umap_2d = reducer.fit_transform(X)
umap_sil = silhouette_score(umap_2d, y)
print("UMAP silhouette:", umap_sil)

tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
tsne_sil = silhouette_score(tsne_2d, y)
print("t-SNE silhouette:", tsne_sil)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
projections = [("PCA", pca_2d, pca_sil), ("UMAP", umap_2d, umap_sil), ("t-SNE", tsne_2d, tsne_sil)]

scatter = None
for ax, (name, proj, sil) in zip(axes, projections):
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=y, cmap="coolwarm", alpha=0.5, s=8)
    ax.set_title(f"{name} (silhouette = {sil:.4f})")
    ax.set_xlabel(f"{name}-1")
    ax.set_ylabel(f"{name}-2")

fig.colorbar(scatter, ax=axes, label="Cost Barrier", shrink=0.8, ticks=[0, 1])
plt.savefig("nsduh_analysis_outputs/construct_validity_projections_7970.png", dpi=150, bbox_inches="tight")
print("saved: nsduh_analysis_outputs/construct_validity_projections_7970.png")
