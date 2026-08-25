# RT-Tracker

**Research Trend Tracker** — a pipeline for mapping the state and trajectory of a research
field from its published literature.

Give it a PubMed query. It retrieves the matching literature, discovers the topics inside
it, and then measures the field from three angles: **who** works on it (co-authorship
networks), **what** it talks about (term co-occurrence networks), and **where it is
going** (topic dynamics over time and vocabulary growth curves).

Every stage is a standalone command-line script. Run the whole chain, or start in the
middle with data you already have.

---

## Pipeline

```
                    PubMed query
                          │
                          ▼
            ┌───────────────────────────┐
            │     ab_extractor.py       │  search + download
            └───────────────────────────┘
                          │
                pubmed_abstracts_filtered.csv
                          │
        ┌─────────────────┴──────────────────┐
        ▼                                    ▼
┌────────────────────┐            ┌────────────────────────┐
│   tm_analyzer.py   │            │   tdm_generator.py     │
│  topic modeling    │            │ term-document matrices │
└────────────────────┘            └────────────────────────┘
        │                              │                │
   data_topic.csv                  D_tdm.csv        D_tdm.csv
        │                              │                │
        ▼                              ▼                ▼
┌──────────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│ post_hoc_analyzer.py │   │ cooc_nt_builder.py │   │ pan_core_analyzer.py │
│  author networks     │   │   term networks    │   │  vocabulary growth   │
└──────────────────────┘   └────────────────────┘   └──────────────────────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
            ┌───────────────────────────┐
            │   network_analyzer.py     │  structural analysis
            └───────────────────────────┘
```

`tm_analyzer.py` also supplies the chronological document order that
`pan_core_analyzer.py` can use, and the per-topic split that `tdm_generator.py` can build
separate matrices from.

---

## The scripts

| Script | What it does | Documentation |
|--------|--------------|---------------|
| **`ab_extractor.py`** | Searches PubMed via NCBI E-utilities, downloads records in batches, writes a CSV of bibliographic metadata and abstracts, and builds first-pass co-authorship networks and publication statistics. | [AB_EXTRACTOR_README.md](AB_EXTRACTOR_README.md) |
| **`tm_analyzer.py`** | Fits LDA / Sentence-BERT / LDA+SBERT topic models across a range of topic counts, keeps the most coherent one, assigns a topic to every document, and evaluates the result with a full metric suite. | [TM_ANALYZER_README.md](TM_ANALYZER_README.md) |
| **`post_hoc_analyzer.py`** | Turns the topic-labelled table into weighted co-authorship networks (directed and undirected) plus author and publication statistics. | [POST_HOC_ANALYZER_README.md](POST_HOC_ANALYZER_README.md) |
| **`tdm_generator.py`** | Builds term-document matrices from the text: full-text, keyword-words, and keyword-phrases. | [TDM_GENERATOR_README.md](TDM_GENERATOR_README.md) |
| **`cooc_nt_builder.py`** | Builds a term co-occurrence network from a term-document matrix, with an interactive view and an analysis report. | [COOC_NT_BUILDER_README.md](COOC_NT_BUILDER_README.md) |
| **`pan_core_analyzer.py`** | Measures vocabulary growth with Heaps' law and a permutation test — is the field still introducing new terminology, or saturating? | [PAN_CORE_ANALYZER_README.md](PAN_CORE_ANALYZER_README.md) |
| **`network_analyzer.py`** | Structural analysis of any network the pipeline produces: degree distributions, power-law fit, clustering, communities, PageRank. | [NETWORK_ANALYZER_README.md](NETWORK_ANALYZER_README.md) |
| `tm_topic_metrics.py` | Metric suite used by `tm_analyzer.py`'s evaluation stage. Not an entry point. | — |

---

## Installation

```bash
git clone <this-repository>
cd <this-repository>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Install everything at once:

```bash
pip install -r requirements.txt
```

Each script also has its own requirements file, so you can install only what one stage
needs:

```bash
pip install -r requirements_ab_extractor.txt
```

**Python 3.9 or newer.** The whole pipeline is verified on Python 3.11.

Two constraints worth knowing before you start:

- **NumPy must stay below 2.x.** gensim 4.x does not yet build against NumPy 2.
- **`sentence-transformers` is only needed for the default `--method LDA_BERT`** in
  `tm_analyzer.py`. Running with `--method LDA` needs neither it nor a model download.

`tm_analyzer.py` and `tdm_generator.py` download a few NLTK resources on first run and
reuse them afterwards.

---

## NCBI credentials

NCBI requires a **contact e-mail address** for every E-utilities request. Register it once:

```bash
# Windows PowerShell
$env:NCBI_EMAIL = "you@example.com"
$env:NCBI_API_KEY = "your_api_key"      # optional

# Linux / macOS
export NCBI_EMAIL="you@example.com"
export NCBI_API_KEY="your_api_key"      # optional
```

An API key is free from the [NCBI account settings page](https://www.ncbi.nlm.nih.gov/account/settings/)
and raises the request limit from 3 to 10 requests per second, which matters for large
harvests.

---

## Quick start

A complete run over a small corpus. Validate the query with a small `--max-results`
first, then raise it.

```bash
# 1. Retrieve the literature
python ab_extractor.py \
    --query "nontuberculous[title] NOT review[pt]" \
    --year "after 1990" \
    --output-dir ./run/abstracts \
    --email you@example.com

# 2. Discover the topics
python tm_analyzer.py \
    --input-csv ./run/abstracts/pubmed_abstracts_filtered.csv \
    --output-dir ./run/topics \
    --method LDA --start-k 4 --end-k 10 --step-k 2

# 3. Who works on it
python post_hoc_analyzer.py \
    --input-file ./run/topics/data_topic.csv \
    --output-dir ./run/authors

python network_analyzer.py \
    ./run/authors/author_information/author_cooccur_undirected_nt.graphml \
    ./run/authors/network_analysis

# 4. What it talks about
python tdm_generator.py \
    --input-csv ./run/abstracts/pubmed_abstracts_filtered.csv \
    --output-dir ./run/tdm

python cooc_nt_builder.py \
    --input-tdm ./run/tdm/D_tdm.csv \
    --output-dir ./run/terms \
    --top-terms 100

# 5. Where it is going
python pan_core_analyzer.py \
    --input-tdm ./run/tdm/D_tdm.csv \
    --output-dir ./run/vocabulary \
    --document-order-csv ./run/topics/data_topic.csv \
    --order-column Abstract_ID \
    --order-sort-by Publication_Year Abstract_ID
```

---

## Data flow

Each stage hands the next one a named file. You can enter the pipeline anywhere, as long
as the input has the columns the stage needs.

| Stage | Reads | Writes |
|-------|-------|--------|
| `ab_extractor.py` | a PubMed query | `pubmed_abstracts_filtered.csv` |
| `tm_analyzer.py` | that CSV | `data_topic.csv` (input rows + a `topic` column) |
| `post_hoc_analyzer.py` | `data_topic.csv` | `author_cooccur_*_nt.{gml,graphml,gexf}`, `author_affiliations.csv` |
| `tdm_generator.py` | either CSV | `D_tdm.csv`, `D_keywords_all_tdm.csv`, `D_keywords_tdm.csv` |
| `cooc_nt_builder.py` | a TDM | `term_co_occurrence_network.{gml,graphml,gexf}`, co-occurrence matrices |
| `pan_core_analyzer.py` | a TDM (+ optional order) | `pan_core_statistics.csv`, accumulation curves |
| `network_analyzer.py` | any `.graphml`/`.gml`/`.gexf` | figures + `network_statistics.csv` |

### Key columns

The pipeline is held together by a handful of column names. Any table carrying these can
be fed in from outside.

| Column | Used by | Purpose |
|--------|---------|---------|
| `Abstract_ID` | all downstream stages | Unique document identifier; becomes the TDM column labels |
| `Title`, `Abstract`, `Keywords` | `tm_analyzer`, `tdm_generator` | The modeled text |
| `Authors`, `Corresponding_Author`, `Affiliations` | `post_hoc_analyzer` | Co-authorship networks |
| `Publication_Year` | `tm_analyzer`, `post_hoc_analyzer`, `pan_core_analyzer` | Temporal analysis and chronological ordering |
| `Journal_Name` | `post_hoc_analyzer` | Journal statistics |
| `topic` | `post_hoc_analyzer` | Topic coverage per author. `0` means the document was dropped during preprocessing |

---

## Reproducibility

- **Seeds are explicit.** `tm_analyzer.py` (`--random-state`) seeds LDA, k-means and UMAP
  together; `pan_core_analyzer.py` (`--seed`) seeds the permutation RNG and records the
  seed in its statistics output.
- **Parameters are recorded.** `pan_core_analyzer.py` writes `software_versions.csv` and
  the full fit configuration; `tm_analyzer.py`'s evaluation stage records the method,
  seed, gamma and SBERT model name in `model_level_metrics.csv`.
- **Every stage writes a statistics table**, not only figures, so numbers quoted in a
  manuscript can be traced back to a file.

---

## Interpreting the output

Each script's README has a section on reading its figures. Three cautions that apply
across the pipeline:

1. **Custom stop words are the highest-leverage setting.** Domain terms that appear in
   nearly every document (`food`, `supply`, `chain` in a food-supply corpus) dominate
   every topic and every network, making them indistinguishable. Run once, inspect
   `wordcloud_terms_weights.csv` and the co-occurrence hub list, add the offenders to a
   custom stop word CSV, and run again. This iteration matters more than any model
   parameter.
2. **Coherence is maximized, not validated.** The highest c_v score is a starting point
   for choosing a topic count, not proof that it is right. Read the topic-level metrics
   and the word clouds before settling.
3. **A good fit is not evidence of a law.** Both the Heaps' law exponent and the
   power-law degree exponent come from least-squares fits. Check the R² and the
   permutation test before quoting an exponent.

---

## Repository layout

```
.
├── README.md                          # this file
├── LICENSE                            # MIT
├── requirements.txt                   # every dependency, whole pipeline
│
├── ab_extractor.py                    # 1. literature retrieval
├── tm_analyzer.py                     # 2. topic modeling
├── tm_topic_metrics.py                #    (metric suite used by tm_analyzer)
├── post_hoc_analyzer.py               # 3. author networks
├── tdm_generator.py                   # 4. term-document matrices
├── cooc_nt_builder.py                 # 5. term co-occurrence networks
├── pan_core_analyzer.py               # 6. vocabulary growth
├── network_analyzer.py                # 7. structural network analysis
│
├── AB_EXTRACTOR_README.md             # per-script documentation
├── TM_ANALYZER_README.md
├── POST_HOC_ANALYZER_README.md
├── TDM_GENERATOR_README.md
├── COOC_NT_BUILDER_README.md
├── PAN_CORE_ANALYZER_README.md
├── NETWORK_ANALYZER_README.md
│
└── requirements_*.txt                 # per-script dependencies
```

---

## Getting help

Every script prints its full option list:

```bash
python <script>.py --help
```

Each per-script README ends with a troubleshooting section covering the errors that
actually come up.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify and redistribute,
including commercially, provided the copyright notice is kept.

If you use RT-Tracker in published work, a citation is appreciated but not required.
