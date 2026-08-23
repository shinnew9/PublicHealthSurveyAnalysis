"""
Reliability re-validation on the FINAL leak-free pipeline (predictive_text, no cost_barrier)
using the ACTUAL corpus generation models (gpt-4.1-nano 90.7% / gpt-4.1-mini 9.3%),
not the old n=300 leaky respondent_text used in the original May 2026 pilot.
"""
import os
import json
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI

warnings.filterwarnings("ignore")

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")

OUT_DIR = Path("./nsduh_analysis_outputs")
MASTER_PATH = OUT_DIR / "df_corrected_7970_with_gpt_profiles_embeddings.csv"
MISSING743_PATH = OUT_DIR / "df_missing_743_with_gpt_profiles_embeddings.csv"
CHECKPOINT_PATH = OUT_DIR / "reliability_revalidation_checkpoint.csv"
OUTPUT_PATH = OUT_DIR / "reliability_revalidation_results.csv"

N_SAMPLE = 300
N_RUNS = 10
RANDOM_STATE = 42
EMBED_MODEL = "text-embedding-3-small"
N_WORKERS = 8

# Exact system prompt used for the actual final corpus (profile_generate_for_whole.ipynb / 00 notebook)
SYSTEM_PROMPT = """
You are assisting with an academic machine learning study using NSDUH survey variables.

Your task is to rewrite structured survey information into a concise, neutral respondent profile.

Important rules:
- Do NOT predict the target label.
- Do NOT mention whether the respondent has or does not have a cost barrier.
- Do NOT add facts that are not provided.
- Do NOT infer sensitive traits beyond the given survey categories.
- Preserve uncertainty when information is unavailable, missing, or not collected.
- Keep the profile concise, factual, and suitable for downstream text embedding.
- Write in 2 to 4 sentences.
""".strip()


def build_user_prompt(predictive_text):
    return f"""
Structured survey information:

{predictive_text}

Rewrite this into a concise predictive profile for downstream machine learning.
"""


client = OpenAI(timeout=60.0, max_retries=2)


def generate_profile(predictive_text, model_name, max_retries=5):
    user_prompt = build_user_prompt(str(predictive_text))
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_output_tokens=180,
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"[WARN] generate attempt {attempt+1}: {e}")
            time.sleep(min(2 ** attempt, 30))
    return None


def get_embedding(text, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=[text], timeout=60)
            return resp.data[0].embedding
        except Exception as e:
            print(f"[WARN] embed attempt {attempt+1}: {e}")
            time.sleep(min(2 ** attempt, 30))
    return None


def process_respondent(row):
    orig_idx = row["original_index"]
    model_name = row["source_model"]
    predictive_text = row["predictive_text"]

    profiles = []
    for _ in range(N_RUNS):
        p = generate_profile(predictive_text, model_name)
        profiles.append(p)

    embeddings = []
    for p in profiles:
        if p is None:
            embeddings.append(None)
        else:
            embeddings.append(get_embedding(p))

    valid = [e for e in embeddings if e is not None]
    if len(valid) < 2:
        mean_cos = np.nan
    else:
        X = np.array(valid)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        sim_matrix = Xn @ Xn.T
        iu = np.triu_indices(len(valid), k=1)
        mean_cos = sim_matrix[iu].mean()

    return {
        "original_index": orig_idx,
        "source_model": model_name,
        "n_valid_runs": len(valid),
        "mean_pairwise_cosine": mean_cos,
    }


def main():
    df = pd.read_csv(MASTER_PATH)
    missing743 = pd.read_csv(MISSING743_PATH)
    mini_idx = set(missing743["original_index"])
    df["source_model"] = df["original_index"].apply(
        lambda x: "gpt-4.1-mini" if x in mini_idx else "gpt-4.1-nano"
    )

    sample = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE)
    print(f"[INFO] Sampled {len(sample)} respondents")
    print(sample["source_model"].value_counts())

    if CHECKPOINT_PATH.exists():
        done = pd.read_csv(CHECKPOINT_PATH)
        done_idx = set(done["original_index"])
        print(f"[INFO] Resuming, {len(done_idx)} already done")
    else:
        done = pd.DataFrame(columns=["original_index", "source_model", "n_valid_runs", "mean_pairwise_cosine"])
        done_idx = set()

    todo = sample[~sample["original_index"].isin(done_idx)]
    print(f"[INFO] {len(todo)} respondents remaining")

    results = list(done.to_dict("records"))
    completed = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_respondent, row): row["original_index"] for _, row in todo.iterrows()}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1
            if completed % 10 == 0:
                pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
                print(f"[CHECKPOINT] {completed}/{len(todo)} saved")

    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_PATH, index=False)
    final_df.to_csv(CHECKPOINT_PATH, index=False)

    print("\n=== Overall ===")
    print("Mean cosine similarity:", final_df["mean_pairwise_cosine"].mean())
    print("Across-respondent SD:", final_df["mean_pairwise_cosine"].std())
    print("\n=== By source model ===")
    print(final_df.groupby("source_model")["mean_pairwise_cosine"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
