import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("nsduh_analysis_outputs/df_corrected_7970_with_gpt_profiles_embeddings.csv")

age_labels = {
    "1 - Respondent is 12 or 13 years old": "12-13",
    "2 - Respondent is 14 or 15 years old": "14-15",
    "3 - Respondent is 16 or 17 years old": "16-17",
    "4 - Respondent is between 18 and 20 years old": "18-20",
    "5 - Respondent is between 21 and 23 years old": "21-23",
    "6 - Respondent is 24 or 25 years old": "24-25",
    "7 - Respondent is between 26 and 29 years old": "26-29",
    "8 - Respondent is between 30 and 34 years old": "30-34",
    "9 - Respondent is between 35 and 49 years old": "35-49",
    "10 - Respondent is between 50 and 64 years old": "50-64",
    "11 - Respondent is 65 years old or older": "65+",
}
df["age_label"] = df["AGE3"].map(age_labels)
order = ["12-13", "14-15", "16-17", "18-20", "21-23", "24-25",
         "26-29", "30-34", "35-49", "50-64", "65+"]

ct_count = pd.crosstab(df["age_label"], df["cost_barrier"]).reindex(order)
ct_prop = pd.crosstab(df["age_label"], df["cost_barrier"], normalize="index").reindex(order)
rates = ct_prop[1]
counts = ct_count.sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(order, rates.values * 100, color="#4C72B0")
for i, (rate, n) in enumerate(zip(rates.values, counts.values)):
    ax.text(i, rate * 100 + 1.5, f"{rate*100:.1f}%\n(n={n})", ha="center", fontsize=8)

ax.set_ylabel("Cost Barrier Rate (%)")
ax.set_xlabel("Age Group")
ax.set_title("Cost Barrier Rate by Age Group (n = 7,970)")
ax.set_ylim(0, 80)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("nsduh_analysis_outputs/age_group_cost_barrier_7970.png", dpi=150)
print("saved: nsduh_analysis_outputs/age_group_cost_barrier_7970.png")
