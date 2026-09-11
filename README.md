# LLM-Generated Respondent Representations for Public Health Survey Analysis

Code accompanying the manuscript *"LLM-Generated Respondent Representations for
Public Health Survey Analysis"* (Lee, Shin, Chuah). This repository evaluates
whether rewriting structured NSDUH survey responses into short GPT-generated
narrative profiles improves prediction of self-reported healthcare cost
barriers, relative to raw structured features and to a direct text-embedding
control, and replicates the result with an open-weight generator model
(LLaMA-3.1-8B-Instruct).

## Pipeline

Notebooks/scripts are numbered in the order they are meant to be run. Each
assumes the outputs of earlier steps are available in `nsduh_analysis_outputs/`
(created automatically by the scripts).

| File | Purpose |
|---|---|
| `00_build_corrected_7970_dataset_and_within_year_ml.ipynb` | Builds the final 7,970-respondent analytic sample (2022+2023 NSDUH waves), generates GPT profiles/embeddings, and runs the within-year 7-classifier comparison. |
| `01_crossyear_2022_2023.ipynb` | Cross-year (2022→2023, 2023→2022) transfer evaluation and repeated within-year baselines across 5 seeds. |
| `02_wilcoxon_classifier_significance_test.ipynb` | One-sided Wilcoxon signed-rank test of the within-year GPT-vs-raw comparison (headline result, p=0.039). |
| `03_transfer_wilcoxon_and_year_asymmetry.py` | Wilcoxon test on cross-year AUC drop, plus the year-asymmetry diagnostics (class balance, covariate distributions, adversarial validation). |
| `04_downsampling_matched_train_size.ipynb` | Rules out training-set-size as an explanation for the 2022/2023 asymmetry by matching training size (n=3,008) across transfer directions. |
| `05_text_embedding.ipynb` | Direct text embedding control: embeds the raw `predictive_text` template without the GPT rewriting step, to isolate the value of the rewriting step itself. |
| `06_reliability_revalidation.py` | Reliability check: 10 repeated GPT rewrites per respondent (n=300) on the final leak-free pipeline, reporting pairwise cosine similarity. |
| `07_figure_age_group.py` | Generates the cost-barrier-by-age-group figure. |
| `08_figure_construct_validity.py` | Generates the PCA/UMAP/t-SNE embedding-space projection figure and silhouette scores. |
| `09_checking_external_validity.ipynb` | TEDS-A data preparation (extracts the Stata files and converts them to CSV) for the subgroup-level external-validity comparison in `14_teds_subgroup_validation.py`. |
| `10_replication_on_LLaMA3.ipynb` | Open-weight replication: regenerates profiles with LLaMA-3.1-8B-Instruct and re-evaluates under both an open-weight embedding model (BGE-large) and the same embedding model used for the GPT profiles (`text-embedding-3-small`). |
| `11_diversity_check.py` | Cross-respondent embedding diversity check: compares within-respondent reliability against pairwise similarity across distinct respondents, to test whether profiles collapse into a generic template. |
| `12_linguistic_marker_check.py` | Linguistic marker sensitivity check: tests whether profile language (word count, sentence length, vocabulary complexity, etc.) responds systematically to insurance status and age group. |
| `13_interpretability_pca.py` | PCA on the GPT profile embeddings: explained variance per component and each component's association with `cost_barrier`, to locate where the predictive signal lives in the embedding space. |
| `14_teds_subgroup_validation.py` | External validity check: cross-validated NSDUH-predicted cost-barrier probability vs. TEDS-A admission rate (per Census population estimate), compared across age, race/ethnicity, and insurance-status subgroups. Requires `09` to have been run first, and the population estimates in `census_data/`. |
| `15_figure_cross_year_drop.py` | Regenerates the cross-year AUC drop figures (mean degradation by model, GPT-minus-raw drop pooled and by transfer direction) from the saved `01` outputs, with a solid high-contrast zero-reference line and larger, more legible axis labels. |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 was used for development; other 3.10+ versions should work.

### Credentials

This code calls the OpenAI API and, for `10_replication_on_LLaMA3.ipynb`, downloads
a gated Hugging Face model. Set both as environment variables before running —
**do not hardcode them in the notebooks**:

```bash
export OPENAI_API_KEY="your-key-here"
export HF_TOKEN="your-huggingface-token-here"
```

`HF_TOKEN` requires access to `meta-llama/Llama-3.1-8B-Instruct` (accept the
license on the model page, and enable "Read access to contents of all public
gated repos" if using a fine-grained token).

## Data

This repository does not include any survey microdata. To reproduce the
analysis, download the following from SAMHSA and place them in this directory:

- **NSDUH** (2021–2024 public use file): [SAMHSA NSDUH data page](https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health)
- **TEDS-A** (2021–2023 public use files, for the external-validity check in `14`): [SAMHSA TEDS data page](https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set)

`09_checking_external_validity.ipynb` reads its data directory from the
`NSDUH_DATA_DIR` environment variable (defaulting to the current directory),
so it runs from the repo root without manual editing:

```bash
export NSDUH_DATA_DIR="."
```

`census_data/` contains the U.S. Census Bureau population denominators used
by `14_teds_subgroup_validation.py` (national Population Estimates Program
vintage-2023 estimates by single-year age/race/Hispanic origin for July 2022
and July 2023, and ACS 1-year Table HI05 national health-insurance-coverage
totals for the same two years). These are small, aggregate, publicly
released statistics (not survey microdata), so they are checked directly
into this repository for reproducibility rather than requiring a separate
download.

## Notes on reproducibility

- All classifier comparisons use a fixed, stratified 80/20 split (`random_state=42`)
  applied identically across feature sets, and one-sided Wilcoxon signed-rank
  tests across the 7 paired classifier AUCs.
- `HistGradientBoostingClassifier` has known internal non-determinism under
  multi-threaded histogram construction; AUCs for this classifier may vary by
  <0.01 across otherwise-identical reruns. This does not affect any reported
  conclusion (see manuscript footnote in Section 6.5).
- The final analytic corpus's GPT profiles were generated with a mix of
  `gpt-4.1-nano` (90.7%) and `gpt-4.1-mini` (9.3%, a later supplemental batch);
  both are handled identically throughout.

## License

Code is provided for reproducibility of the accompanying manuscript. See
`LICENSE` for terms.
