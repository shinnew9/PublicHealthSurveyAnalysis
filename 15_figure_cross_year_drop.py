"""
Regenerates the cross-year AUC drop figures (Figures 4-6) with two fixes
requested during review:
  - the zero-degradation reference line is now solid and higher-contrast
    (was a low-contrast dashed line), so the positive-vs-negative direction
    is immediately visible.
  - the per-transfer-direction panels (Figure 6) use larger fonts and more
    label rotation, since they are displayed at half-textwidth and the
    original labels were too small to read at that size.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "nsduh_analysis_outputs"
ZERO_LINE_KW = dict(color="black", linestyle="-", linewidth=1.5, zorder=3)

drop_repeated_summary = pd.read_csv(f"{OUT_DIR}/cross_year_drop_summary_repeated_within_year_5seeds_by_model.csv")
drop_repeated_pivot = pd.read_csv(f"{OUT_DIR}/cross_year_drop_comparison_repeated_within_year_5seeds_pivot.csv")

# --- Figure 4: mean cross-year AUC degradation by model (grouped bars) ---
plot_df = drop_repeated_summary.copy()
models = plot_df["Model"].tolist()
x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 5.5))
plt.bar(x - width, plot_df["Raw_drop_mean"], width, label="Raw structured")
plt.bar(x, plot_df["GPT_drop_mean"], width, label="GPT profile embedding")
plt.bar(x + width, plot_df["Combined_drop_mean"], width, label="Raw + GPT profile embedding")
plt.axhline(0, **ZERO_LINE_KW)
plt.xticks(x, models, rotation=30, ha="right", fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel("AUC drop: within-year mean - cross-year", fontsize=12)
plt.title("Mean cross-year AUC degradation by model\nRepeated 5-seed within-year baseline", fontsize=13)
plt.legend(fontsize=11)
plt.tight_layout()
save_path = f"{OUT_DIR}/cross_year_auc_drop_by_model_repeated_5seeds.png"
plt.savefig(save_path, dpi=300)
plt.close()
print("Saved:", save_path)

# --- Figure 5: GPT-minus-raw drop difference, pooled across seeds ---
plot_df = drop_repeated_summary.copy().sort_values("GPT_drop_minus_Raw_drop_mean")

plt.figure(figsize=(9, 5.5))
plt.bar(plot_df["Model"], plot_df["GPT_drop_minus_Raw_drop_mean"])
plt.axhline(0, **ZERO_LINE_KW)
plt.xticks(rotation=30, ha="right", fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel("GPT drop - Raw drop", fontsize=12)
plt.title("Cross-year AUC degradation difference by model\nRepeated 5-seed within-year baseline", fontsize=13)
plt.tight_layout()
save_path = f"{OUT_DIR}/cross_year_gpt_minus_raw_drop_difference_by_model.png"
plt.savefig(save_path, dpi=300)
plt.close()
print("Saved:", save_path)

# --- Figure 6: GPT-minus-raw drop difference, split by transfer direction ---
# Displayed at 0.45\textwidth each (two side by side) in the manuscript, so
# labels need to be larger and more rotated than a full-width figure would.
plot_df = drop_repeated_pivot.copy()
for exp_name in plot_df["Experiment"].unique():
    temp = plot_df[plot_df["Experiment"] == exp_name].copy()
    temp = temp.sort_values("GPT_drop_minus_Raw_drop")

    plt.figure(figsize=(10, 7))
    plt.bar(temp["Model"], temp["GPT_drop_minus_Raw_drop"])
    plt.axhline(0, **ZERO_LINE_KW)
    plt.xticks(rotation=45, ha="right", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("GPT drop - Raw drop", fontsize=17)
    plt.title(f"Cross-year AUC degradation difference\n{exp_name}", fontsize=17)
    plt.tight_layout()

    safe_name = exp_name.replace(" ", "_").replace("->", "to").replace(">", "to").lower()
    save_path = f"{OUT_DIR}/cross_year_gpt_minus_raw_drop_difference_{safe_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)
