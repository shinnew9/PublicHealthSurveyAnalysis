"""
Open-weight replication with LLaMA-3.1-8B-Instruct (Section 3.9 / 6.5).

Replaces the proprietary GPT-4.1-family generator with an open-weight model,
holding the rest of the pipeline constant, then evaluates the resulting
profiles under both an open-weight embedding model (BGE-large) and the same
embedding model used for the GPT profiles (text-embedding-3-small), to
isolate whether any performance difference reflects the generator model or
the embedding model.

Requires HF_TOKEN (with access to meta-llama/Llama-3.1-8B-Instruct) and
OPENAI_API_KEY set as environment variables. Run in three stages:
    python 10_replication_on_LLaMA3.py generate   # profile generation (GPU)
    python 10_replication_on_LLaMA3.py embed       # BGE + OpenAI embedding
    python 10_replication_on_LLaMA3.py evaluate    # classifiers + Wilcoxon tests
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

OUT_DIR = "nsduh_analysis_outputs"
DATA_PATH = f"{OUT_DIR}/df_corrected_7970_with_gpt_profiles_embeddings.csv"
CHECKPOINT_PATH = f"{OUT_DIR}/llama_profiles_checkpoint.csv"

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 180
SAVE_EVERY = 5  # batches

SYSTEM_PROMPT = """You are assisting with an academic machine learning study using NSDUH survey variables.

Your task is to rewrite structured survey information into a concise, neutral respondent profile.

Important rules:
- Do NOT predict the target label.
- Do NOT mention whether the respondent has or does not have a cost barrier.
- Do NOT add facts that are not provided.
- Do NOT infer sensitive traits beyond the given survey categories.
- Preserve uncertainty when information is unavailable, missing, or not collected.
- Keep the profile concise, factual, and suitable for downstream text embedding.
- Write in 2 to 4 sentences."""


def build_user_prompt(predictive_text):
    return f"""Structured survey information:

    {predictive_text}

    Rewrite this into a concise predictive profile for downstream machine learning."""


# ============================================================
# Stage 1: profile generation (LLaMA-3.1-8B-Instruct)
# ============================================================
def generate_profiles():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Set the HF_TOKEN environment variable before running this stage.")
    login(token=os.environ["HF_TOKEN"])

    print("[INFO] Loading LLaMA model...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "10GiB", 1: "10GiB", 2: "10GiB", 3: "10GiB", "cpu": "0GiB"},
    )
    model.eval()

    def generate_batch(predictive_texts):
        prompts = []
        for pt in predictive_texts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(pt)},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id)

        results = []
        for i in range(len(predictive_texts)):
            gen_tokens = outputs[i][inputs["input_ids"].shape[1]:]
            results.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
        return results

    df = pd.read_csv(DATA_PATH)
    print("[INFO] Total respondents:", len(df))

    if os.path.exists(CHECKPOINT_PATH):
        done_df = pd.read_csv(CHECKPOINT_PATH)
        done_idx = set(done_df["original_index"])
        print(f"[INFO] Resuming, {len(done_idx)} already done")
    else:
        done_df = pd.DataFrame(columns=["original_index", "llama_profile"])
        done_idx = set()

    todo = df[~df["original_index"].isin(done_idx)].reset_index(drop=True)
    print(f"[INFO] {len(todo)} remaining")

    results = list(done_df.to_dict("records"))
    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, start in enumerate(tqdm(range(0, len(todo), BATCH_SIZE), total=n_batches, desc="Generating profiles")):
        batch = todo.iloc[start:start + BATCH_SIZE]
        profiles = generate_batch(batch["predictive_text"].tolist())
        for orig_idx, profile in zip(batch["original_index"], profiles):
            results.append({"original_index": orig_idx, "llama_profile": profile})

        if (batch_num + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            print(f"[CHECKPOINT] {len(results)}/{len(df)} saved")

    pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
    print("[DONE] All profiles generated")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Stage 2: embed the generated profiles (BGE-large, then OpenAI)
# ============================================================
def embed_profiles():
    from sentence_transformers import SentenceTransformer

    profiles_df = pd.read_csv(CHECKPOINT_PATH)
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")

    texts = profiles_df["llama_profile"].fillna("").tolist()
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True)

    profiles_df["llama_embedding"] = [json.dumps(e.tolist()) for e in embeddings]
    profiles_df.to_csv(f"{OUT_DIR}/llama_profiles_with_embeddings.csv", index=False)
    print("[DONE] Saved llama_profiles_with_embeddings.csv")

    # Re-embed with the same model used for the GPT profile embeddings, to
    # isolate the generator model's effect from the embedding model's.
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this stage.")
    from openai import OpenAI

    EMBED_MODEL = "text-embedding-3-small"
    openai_client = OpenAI(timeout=120.0, max_retries=2)

    def get_embeddings_batch(texts, max_retries=5):
        clean = ["" if pd.isna(t) else str(t) for t in texts]
        for attempt in range(max_retries):
            try:
                resp = openai_client.embeddings.create(model=EMBED_MODEL, input=clean, timeout=120)
                return [d.embedding for d in resp.data]
            except Exception as e:
                print(f"[WARN] embed attempt {attempt + 1}: {e}")
                time.sleep(min(2 ** attempt, 60))
        return [None] * len(texts)

    llama_df = pd.read_csv(CHECKPOINT_PATH)
    texts = llama_df["llama_profile"].tolist()
    embs, BATCH = [], 256
    for i in range(0, len(texts), BATCH):
        embs.extend(get_embeddings_batch(texts[i:i + BATCH]))
        print(f"  embedded {min(i + BATCH, len(texts))} / {len(texts)}")

    assert all(e is not None for e in embs), "some embeddings failed"

    llama_df["llama_embedding_openai"] = [json.dumps(e) for e in embs]
    llama_df.to_csv(f"{OUT_DIR}/llama_profiles_with_openai_embeddings.csv", index=False)
    print("[DONE] Saved llama_profiles_with_openai_embeddings.csv")


# ============================================================
# Stage 3: evaluate (7 classifiers, embedding model held constant across
# generators as the primary comparison; BGE retained as a robustness check)
# ============================================================
RAW_COLS = ["year", "AGE3", "NEWRACE2", "IRINSUR4", "substance_peer_support", "mental_health_peer_support"]


def make_model(name):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", random_state=42))
    if name == "Linear SVM":
        return make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=5000, random_state=42))
    if name == "Ridge Classifier":
        return make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced", random_state=42))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42)
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.01, random_state=42)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)
    raise ValueError(name)


MODELS = ["Logistic Regression", "Linear SVM", "Ridge Classifier", "Random Forest",
          "Extra Trees", "HistGradientBoosting"] + (["XGBoost"] if HAS_XGB else [])


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def wilx(res, a, b, alternative="greater"):
    p = res.pivot(index="model", columns="feature", values="auc")
    d = (p[a] - p[b]).values
    stat, pval = wilcoxon(d, alternative=alternative)
    return dict(comparison=f"{a} - {b} ({alternative})", n=len(d), mean_diff=d.mean(),
                n_pos=int((d > 0).sum()), wilcoxon_stat=stat, p_one_sided=pval)


def evaluate():
    master = pd.read_csv(DATA_PATH)
    llama_openai_df = pd.read_csv(f"{OUT_DIR}/llama_profiles_with_openai_embeddings.csv")
    llama_bge_df = pd.read_csv(f"{OUT_DIR}/llama_profiles_with_embeddings.csv")

    df = master.merge(
        llama_openai_df[["original_index", "llama_embedding_openai"]], on="original_index", how="inner"
    ).merge(
        llama_bge_df[["original_index", "llama_embedding"]], on="original_index", how="inner"
    )
    print("Merged shape:", df.shape)
    assert len(df) == len(master), "rows dropped during merge -- check original_index"

    y = df["cost_barrier"].astype(int).values
    X_raw = pd.get_dummies(df[RAW_COLS].astype(str), drop_first=False).values.astype("float32")
    X_gpt = np.vstack(df["predictive_embedding"].apply(json.loads).values).astype("float32")
    X_llama_openai = np.vstack(df["llama_embedding_openai"].apply(json.loads).values).astype("float32")
    X_llama_bge = np.vstack(df["llama_embedding"].apply(json.loads).values).astype("float32")

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    feature_sets = {
        "Raw structured": X_raw,
        "GPT profile embedding": X_gpt,
        "LLaMA profile (text-embedding-3-small)": X_llama_openai,
        "LLaMA profile (BGE, supplementary)": X_llama_bge,
    }

    rows = []
    for m in MODELS:
        for fname, X in feature_sets.items():
            model = make_model(m)
            model.fit(X[train_idx], y[train_idx])
            score = get_score(model, X[test_idx])
            pred = model.predict(X[test_idx])
            rows.append(dict(model=m, feature=fname,
                              auc=roc_auc_score(y[test_idx], score),
                              f1=f1_score(y[test_idx], pred),
                              accuracy=accuracy_score(y[test_idx], pred)))

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT_DIR}/llama_replication_results_v2.csv", index=False)
    print("\n=== AUC pivot ===")
    print(res.pivot(index="model", columns="feature", values="auc").round(4).to_string())

    comp = pd.DataFrame([
        wilx(res, "LLaMA profile (text-embedding-3-small)", "Raw structured"),
        wilx(res, "GPT profile embedding", "Raw structured"),  # sanity check: should reproduce p=0.039
        wilx(res, "LLaMA profile (text-embedding-3-small)", "GPT profile embedding"),
        wilx(res, "LLaMA profile (text-embedding-3-small)", "LLaMA profile (BGE, supplementary)"),
    ])
    comp.to_csv(f"{OUT_DIR}/llama_replication_wilcoxon_v2.csv", index=False)
    print("\n=== Wilcoxon (greater) ===")
    print(comp.to_string(index=False))

    # All observed differences are negative, so re-test in the direction the
    # data actually point (see Methods 3.6 for the minimum-achievable-p note).
    comp_correct_direction = pd.DataFrame([
        wilx(res, "LLaMA profile (text-embedding-3-small)", "Raw structured", "less"),
        wilx(res, "LLaMA profile (text-embedding-3-small)", "GPT profile embedding", "less"),
        wilx(res, "LLaMA profile (text-embedding-3-small)", "LLaMA profile (BGE, supplementary)", "less"),
    ])
    print("\n=== Wilcoxon (correct direction) ===")
    print(comp_correct_direction.to_string(index=False))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else None
    if stage == "generate":
        generate_profiles()
    elif stage == "embed":
        embed_profiles()
    elif stage == "evaluate":
        evaluate()
    else:
        print(__doc__)
