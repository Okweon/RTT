# TDM Generator

Build term-document matrices (TDMs) from a bibliographic table.

A TDM has **terms in rows and documents in columns**, each cell holding how many times
that term occurs in that document. It is the input format for `pan_core_analyzer.py` and
for the word cloud and co-occurrence network tools.

---

## Where it fits

```
ab_extractor.py  →  tdm_generator.py  →  pan_core_analyzer.py
   abstracts          term-document matrices    accumulation curves
                                           →  word cloud / co-occurrence tools
```

---

## The three matrices

Every run produces three matrices from the same table, because they answer different
questions.

| File | Source | Terms are | Preprocessing |
|------|--------|-----------|---------------|
| `D_tdm.csv` | Title + Abstract + Keywords | single words | Full: nouns only, lemmatized, stop words removed |
| `D_keywords_all_tdm.csv` | Keywords | single words | Tokenized and lowercased only |
| `D_keywords_tdm.csv` | Keywords | **whole phrases** | Split on commas, lowercased |

`D_tdm.csv` is the general-purpose matrix. The keyword matrices are useful when you want
what the authors themselves called the work: `D_keywords_tdm.csv` keeps
`food supply chain` as one term, while `D_keywords_all_tdm.csv` splits it into three.

Note that the two keyword matrices skip the full preprocessing — no lemmatization and no
NLTK stop word removal — because author keywords are already curated phrases. Custom stop
words are still applied to all three.

---

## Installation

```bash
pip install -r requirements_tdm_generator.txt
```

Python 3.9 or newer is required (verified on Python 3.11).

The first run downloads three NLTK resources (`wordnet`, `stopwords`,
`averaged_perceptron_tagger_eng`) and later runs reuse them. On an offline machine, fetch
them in advance:

```bash
python -c "import nltk; [nltk.download(r) for r in ('wordnet','stopwords','averaged_perceptron_tagger_eng')]"
```

---

## Usage

### Basic usage

```bash
python tdm_generator.py --input-csv <documents.csv> --output-dir <output folder>
```

### Examples

```bash
# 1) Standard run on the output of ab_extractor.py
python tdm_generator.py \
    --input-csv pubmed_abstracts_filtered.csv \
    --output-dir ./tdm \
    --custom-stopwords-csv custom_stopwords.csv

# 2) Title only, no abstracts
python tdm_generator.py \
    --input-csv pubmed_abstracts_filtered.csv \
    --output-dir ./tdm_title \
    --text-columns Title

# 3) A table that identifies documents by a different column
python tdm_generator.py \
    --input-csv scopus_export.csv \
    --output-dir ./tdm_scopus \
    --id-column DOI --encoding cp1252
```

### Use from a Python script

```python
from tdm_generator import tdm_generator, tdm_generator_df
import pandas as pd

# From a file
shapes = tdm_generator("pubmed_abstracts_filtered.csv",
                       "custom_stopwords.csv", "./tdm")
print(shapes)   # {'D_tdm.csv': (4821, 1204), ...}

# From a DataFrame, one set of matrices per topic
data = pd.read_csv("./tm_output/data_topic.csv")
for topic, group in data.groupby("topic"):
    tdm_generator_df(topic, group, "custom_stopwords.csv", "./tdm_by_topic")
```

`tdm_generator_df` prefixes every output filename with the label you pass, so per-topic
matrices can share one directory (`3_D_tdm.csv`, `4_D_tdm.csv`, ...). This is the entry
point `word_cloud_network_generator.py` uses.

---

## Command line options

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input-csv` | (required) | Bibliographic table |
| `-o`, `--output-dir` | (required) | Directory for the generated matrices |
| `-s`, `--custom-stopwords-csv` | none | Single-column CSV of domain stop words |
| `--text-columns` | `Title Abstract Keywords` | Columns concatenated into the full-text field |
| `--id-column` | `Abstract_ID` | Column used as the matrix column labels |
| `--encoding` | `utf-8` | Encoding of the input CSV; cp1252 is tried as a fallback |
| `--log-level` | `INFO` | Console verbosity |

---

## Input requirements

The input table needs the columns named in `--text-columns` plus the `--id-column`.
`ab_extractor.py` writes all of them, so its `*_filtered.csv` output can be used directly.

Document IDs become the **column labels** of every matrix, so they must be unique — and
they must match whatever downstream tools expect. `pan_core_analyzer.py` matches its
`--order-column` against them.

---

## The preprocessing pipeline

`D_tdm.csv` is built from text that passes through these steps, in order:

1. Blank out the `Not available` placeholder, per column.
2. Concatenate the text columns.
3. Strip HTML tags.
4. Replace `/` with a space, so `and/or` becomes two words.
5. Drop every character that is not alphanumeric, whitespace, or a hyphen.
6. Lowercase.
7. Remove English stop words, custom stop words, and tokens shorter than 3 characters.
8. Remove tokens containing digits — measurements and identifiers are not vocabulary.
9. **Keep only nouns.** This is the most influential choice in the pipeline: topics and
   term networks read as concepts, and nouns carry those. To keep verbs and adjectives as
   well, add `"VB"` and `"JJ"` to `KEPT_POS_PREFIXES` near the top of the script.
10. Lemmatize (`studies` → `study`).

After vectorization, each matrix additionally drops terms that are purely numeric, blank,
shorter than 3 characters, absent from every document, or in the custom stop word list.

---

## Implementation notes

- **Hyphenated compounds survive.** The tokenizer treats `non-tuberculous` and `covid-19`
  as single terms rather than splitting them.
- **Everything is lowercased at tokenization.** `CountVectorizer` runs with
  `lowercase=False` because the tokenizers handle casing themselves; without that,
  `Food` and `food` would be counted as two different keyword terms.
- **NLTK resources are checked before downloading, packed or unpacked.** A downloaded
  corpus often stays as `corpora/wordnet.zip` and is perfectly usable in that form, so
  checking only the unpacked path would re-download it on every run.
- **Missing values become empty strings, not the text `nan`.** `astype(str)` on a column
  with blanks produces the literal string `"nan"`, which is three characters, contains no
  digits, and is tagged as a noun — so it would otherwise end up as a term in the matrix.

---

## Notes and limitations

1. The matrices are **dense CSV files**. A corpus of 5,000 documents with 20,000 terms
   produces a 100-million-cell file. Narrow the vocabulary with a good custom stop word
   list before scaling up.
2. Lemmatization uses WordNet's default noun mode. Irregular forms outside WordNet are
   left unchanged.
3. Part-of-speech tagging runs per document and dominates the runtime on large corpora.
4. The output filenames (`D_tdm.csv` and the two keyword matrices) are fixed, since
   downstream scripts refer to them by name. Use separate output directories, or
   `tdm_generator_df`'s prefix, to keep several runs apart.

---

## Troubleshooting

### `Missing required column(s) [...]`
The `--text-columns` or `--id-column` names do not match the CSV header. The error lists
the columns that are actually available.

### `LookupError: Resource averaged_perceptron_tagger_eng not found`
The automatic download did not complete, usually because the machine is offline. Fetch
the resources manually with the command in the Installation section.

### The matrix contains terms that are clearly noise
Add them to the custom stop word CSV (one term per line, no header) and re-run. Sort
`D_tdm.csv` by row sum to find the worst offenders quickly.

### The matrices are empty or nearly so
Either the text columns are blank, or the custom stop word list is too aggressive. The
log line `Preprocessing N documents` confirms how many rows were read.

### A `UnicodeDecodeError` on the input file
Pass `--encoding cp1252` (or `latin-1`). Files exported from Excel are usually cp1252.

---

## Related scripts

| Script | Role |
|--------|------|
| `ab_extractor.py` | Produces the bibliographic table |
| `tm_analyzer.py` | Assigns a topic to every document, for per-topic matrices |
| `pan_core_analyzer.py` | Consumes `D_tdm.csv` for accumulation analysis |

---

## License

MIT — see [LICENSE](LICENSE) in the repository root.
