# TM Analyzer

Topic modeling pipeline for bibliographic and abstract data, with a built-in model
evaluation stage.

The script takes a CSV of documents — typically the output of `ab_extractor.py` — fits
LDA / Sentence-BERT / LDA+SBERT topic models over a range of topic counts, keeps the most
coherent one, and exports every figure and table needed to report the result.

---

## Pipeline

| Step | What happens |
|------|--------------|
| 1. Load | Read the CSV and concatenate the selected text columns (Title / Keywords / Abstract) into one field per row |
| 2. Preprocess | Sentence cleanup and optional language filter, then tokenize, keep nouns only, drop stop words |
| 3. Model | Fit one model per candidate topic count `k`, score each with c_v coherence, keep the best `k` |
| 4. Visualize | UMAP projection colored by topic, per-topic word clouds, temporal dynamics by publication year |
| 5. Evaluate | Score the winning model with `tm_topic_metrics.py` — coherence, diversity, separation, silhouette, DBCV, assignment confidence, outliers |

Step 5 refits nothing: it reuses the vectors and labels produced in step 3.

---

## Installation

```bash
pip install -r requirements_tm_analyzer.txt
```

Python 3.9 or newer is required (verified on Python 3.11).

Two notes on the dependencies:

- **NumPy must stay below 2.x.** gensim 4.x does not yet build against NumPy 2.
- **`sentence-transformers` is only needed for `--method BERT` and `--method LDA_BERT`**
  (the default). Running with `--method LDA` needs neither it nor a model download.

NLTK resources (`punkt`, `punkt_tab`, `averaged_perceptron_tagger_eng`) are downloaded
automatically on first run, and only if they are missing.

---

## Usage

### Basic usage

```bash
python tm_analyzer.py --input-csv <documents.csv> --output-dir <output folder>
```

### Examples

```bash
# 1) Default hybrid model, scanning k = 2, 4, ..., 14
python tm_analyzer.py \
    --input-csv pubmed_abstracts_filtered.csv \
    --custom-stopwords-csv custom_stopwords.csv \
    --output-dir ./tm_output \
    --text-columns Title Keywords Abstract \
    --method LDA_BERT --start-k 2 --end-k 14 --step-k 2

# 2) Plain LDA - much faster, no SBERT model download
python tm_analyzer.py \
    --input-csv pubmed_abstracts_filtered.csv \
    --output-dir ./tm_output_lda \
    --method LDA --start-k 4 --end-k 10 --step-k 2

# 3) Fixed number of topics (k = 6), evaluation stage skipped
python tm_analyzer.py \
    --input-csv pubmed_abstracts_filtered.csv \
    --output-dir ./tm_output_k6 \
    --start-k 6 --end-k 6 --step-k 1 \
    --skip-evaluation
```

### Use from a Python script

```python
from tm_analyzer import (
    ensure_nltk_resources, load_custom_stopwords, load_data,
    preprocess, topic_model_coherence_generator, plot_umap_projection,
)
import numpy as np

ensure_nltk_resources()
load_custom_stopwords("custom_stopwords.csv")

meta, documents = load_data("pubmed_abstracts_filtered.csv",
                            ["Title", "Keywords", "Abstract"])
sentences, token_lists, idx_in, idx_out = preprocess(documents.tolist())

models, scores = topic_model_coherence_generator(
    method="LDA", sentences=sentences, token_lists=token_lists,
    output_dir="./tm_output", start_k=2, end_k=8, step_k=2,
)
best_model = models[int(np.argmax(scores))]
meta = plot_umap_projection(meta, idx_out, best_model, "./tm_output")
```

---

## Command line options

### Input and output

| Option | Default | Description |
|--------|---------|-------------|
| `--input-csv` | (required) | CSV of documents |
| `--custom-stopwords-csv` | none | Single-column CSV of extra stop words |
| `--output-dir` | `./tm_output` | Directory for all figures and tables |
| `--encoding` | `utf-8` | Encoding of the input CSV; cp1252 and latin-1 are tried as fallbacks |
| `--text-columns` | `Title Keywords Abstract` | Columns concatenated into the modeled text |

### Model

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `LDA_BERT` | `LDA`, `BERT`, or `LDA_BERT` |
| `--gamma` | `14.0` | Weight of the LDA block inside an LDA_BERT vector |
| `--start-k` | `2` | Smallest number of topics to try |
| `--end-k` | `14` | Largest number of topics to try |
| `--step-k` | `2` | Step between candidate topic counts |
| `--random-state` | `100` | Seed shared by LDA, k-means and UMAP |
| `--sbert-model` | `bert-base-nli-max-tokens` | sentence-transformers model name |

### Preprocessing

| Option | Default | Description |
|--------|---------|-------------|
| `--disable-language-filter` | off | Skip the English/French/Spanish/Chinese filter |
| `--use-stemming` | off | Porter-stem tokens |
| `--use-typo-correction` | off | Correct typos with SymSpell (slow) |

### Figures

| Option | Default | Description |
|--------|---------|-------------|
| `--wordcloud-font` | bundled font | Path to a TTF font for the word clouds |
| `--wordcloud-top-n` | `30` | Terms shown per topic word cloud |

### Evaluation stage

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-evaluation` | off | Do not run the `tm_topic_metrics` stage |
| `--eval-top-n-words` | `10` | Words per topic used for coherence and diversity |
| `--eval-temperature` | `0.1` | Softmax temperature of the assignment-confidence proxy |
| `--eval-low-confidence` | `0.50` | Assignment probability below which a document is flagged |
| `--eval-outlier-z` | `-1.5` | Within-topic z-score below which a document is an outlier |

---

## Input requirements

The input CSV needs the columns named in `--text-columns` (`Title`, `Keywords`,
`Abstract` by default). Two more columns are used when present:

| Column | Used for |
|--------|----------|
| `Publication_Year` | The temporal dynamics figure. Without it that figure is skipped. |
| `Abstract_ID` | Copied into `document_level_metrics.csv` as a document identifier. |

`ab_extractor.py` writes all of these, so its `*_filtered.csv` output can be fed
straight into this script.

---

## Generated files

```
<output-dir>/
├── coherence_scores.csv / .html            # c_v coherence for every candidate k
├── data_topic.csv                          # input rows + assigned `topic` column
├── UMAP.html                               # 2D projection colored by topic
├── temporal_dynamics_topics.html           # topic prevalence by publication year
├── wordcloud_topic_<n>.pdf                 # one word cloud per topic
├── wordcloud_terms_weights.csv             # top terms and weights per topic
│
│   # written by the evaluation stage (step 5)
├── model_level_metrics.csv                 # one row summarizing the whole model
├── topic_level_metrics.csv                 # one row per topic
├── document_level_metrics.csv              # one row per modeled document
├── document_level_summary.csv              # aggregate of the document metrics
├── topic_similarity_matrix.csv             # topic-to-topic cosine similarity
├── fig_topic_sizes_and_coherence.png
├── fig_topic_similarity_heatmap.png
├── fig_silhouette_distribution.png
├── fig_assignment_probability_distribution.png
└── fig_topic_embedding_scatter.png
```

### The `topic` column

`data_topic.csv` is the input table with one extra column:

- `topic` in `[1, k]` — the topic assigned to that document.
- `topic = 0` — the document was dropped during preprocessing (no nouns survived
  the stop word filter, or it failed the language filter) and was never modeled.

Topics are 1-indexed everywhere in the outputs, so the figures and tables all agree.

---

## Methods

| Method | Document representation | Notes |
|--------|-------------------------|-------|
| `LDA` | LDA topic-probability vector | Fastest, fully interpretable term weights |
| `BERT` | Sentence-BERT embedding | Captures semantics, no term weights |
| `LDA_BERT` | `[LDA × gamma, SBERT]` concatenated | Default. Sees both corpus-level topic signal and sentence-level semantics |

k-means is fit for every method, so `topic` always comes from the same clustering step.
`--gamma` exists because the SBERT block is far higher-dimensional than the LDA block;
without scaling, the LDA signal would be drowned out during clustering.

Only LDA exposes probabilistic term weights. For `BERT` and `LDA_BERT`,
`wordcloud_terms_weights.csv` ranks terms by frequency within the cluster and records a
weight of `1.0`.

---

## Implementation notes

- **Nouns only.** The word-level filter keeps tokens tagged `NN*`. Topics then read as
  concepts rather than as verb/adjective mixtures. This is the single most influential
  preprocessing choice in the pipeline.
- **Stemming and typo correction are off by default.** Stems hurt readability in the
  figures (`detection` → `detect`), and typo correction costs a dictionary lookup per
  word while rarely helping edited abstracts. Both remain available as flags.
- **Custom stop words matter.** Domain terms that appear in nearly every document
  (`food`, `supply`, `chain` in a food-supply corpus) dominate every topic and make them
  indistinguishable. Run once, look at `wordcloud_terms_weights.csv`, add the offenders
  to the custom stop word CSV, and run again.
- **One palette across all figures.** A topic number keeps the same color in the UMAP
  scatter, the temporal dynamics chart and its word cloud, so figures can be read
  together. The palette cycles beyond 8 topics.
- **Reproducibility.** `--random-state` seeds LDA, k-means and UMAP together. The
  evaluation stage records the method, seed, gamma and SBERT model name in
  `model_level_metrics.csv`.

---

## Notes and limitations

1. `--method LDA_BERT` with a wide `k` scan is expensive: every candidate `k` refits LDA
   over the whole corpus, and SBERT encodes it once per model. Start with `--method LDA`
   to find a sensible `k` range, then switch to the hybrid.
2. Coherence is maximized, not validated. The highest c_v score is a starting point, not
   proof that `k` is right — read the topic-level metrics and the word clouds before
   settling on a topic count.
3. Documents with `topic = 0` are excluded from every figure and metric. If that count is
   large, the stop word list is probably too aggressive.
4. `--end-k` must be smaller than the number of documents that survive preprocessing;
   the script stops with a message rather than fitting more topics than documents.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'gensim'`
```bash
pip install -r requirements_tm_analyzer.txt
```

### `AttributeError: module 'numpy' has no attribute 'float_'` (or similar, inside gensim)
NumPy 2.x is installed. Downgrade:
```bash
pip install "numpy<2"
```

### `Evaluation stage skipped: 'tm_topic_metrics.py' was not found`
That companion module must sit next to `tm_analyzer.py`. It ships alongside this script;
if it was moved, copy it back, or pass `--skip-evaluation` to run without step 5.

### `KeyError: Column(s) [...] not found`
The `--text-columns` names do not match the CSV header. The error message lists the
columns that are actually available.

### Every topic contains the same few words
The corpus vocabulary is dominated by domain terms. Add them to
`--custom-stopwords-csv` and re-run.

### The run is very slow
Use `--method LDA`, narrow the `k` scan (`--start-k` / `--end-k` / `--step-k`), and leave
`--use-typo-correction` off.

---

## Related scripts

| Script | Role |
|--------|------|
| `ab_extractor.py` | Produces the input CSV from a PubMed query |
| `tm_topic_metrics.py` | Metric suite used by the evaluation stage (step 5) |

`tm_topic_metrics.py` is carried over unmodified from the project's `functions/` folder;
it is a dependency of this script rather than a separate entry point.

---

## License

MIT — see [LICENSE](LICENSE) in the repository root.
