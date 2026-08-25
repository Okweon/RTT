# Post Hoc Analyzer

Post-hoc analysis of a topic-labelled bibliographic table.

This script runs at the end of the pipeline, once every document has been assigned a
topic. It takes the table produced by `tm_analyzer.py` (`data_topic.csv` or `.xlsx`) and
derives two families of output:

- **Author information** — weighted co-authorship networks (directed and undirected)
  exported in GML / GraphML / GEXF, plus a table of every author with their publication
  count, affiliation and topic coverage.
- **Publication information** — article counts by journal, by year, and by
  journal-and-year, as CSV tables and interactive charts.

---

## Where it fits

```
ab_extractor.py  →  tm_analyzer.py  →  post_hoc_analyzer.py
   abstracts          data_topic          networks + statistics
```

---

## Installation

```bash
pip install -r requirements_post_hoc_analyzer.txt
```

Python 3.9 or newer is required (verified on Python 3.11).

`openpyxl` is only needed to read `.xlsx` input. `kaleido` is only needed for the
optional `--png` export.

---

## Usage

### Basic usage

```bash
python post_hoc_analyzer.py --input-file <data_topic.csv> --output-dir <output folder>
```

### Examples

```bash
# 1) Standard run on the output of tm_analyzer.py
python post_hoc_analyzer.py \
    --input-file ./tm_output/data_topic.csv \
    --output-dir ./post_hoc_analysis

# 2) Excel input, and static PNG copies of every chart
python post_hoc_analyzer.py \
    --input-file ./tm_output/data_topic.xlsx \
    --output-dir ./post_hoc_analysis \
    --png

# 3) A database that separates author names with semicolons
python post_hoc_analyzer.py \
    --input-file ./scopus_topics.csv \
    --output-dir ./post_hoc_scopus \
    --author-separator ";"
```

### Use from a Python script

```python
from post_hoc_analyzer import post_hoc_analyzer

author_df = post_hoc_analyzer(
    input_file="./tm_output/data_topic.csv",
    output_dir="./post_hoc_analysis",
)
print(author_df.sort_values("Frequency", ascending=False).head(10))
```

---

## Command line options

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input-file` | (required) | Topic-labelled table: `data_topic.csv` or `data_topic.xlsx` |
| `-o`, `--output-dir` | (required) | Directory for all output files |
| `--author-separator` | `,` | Character separating names in the `Authors` column |
| `--png` | off | Also export the interactive charts as static PNGs (requires `kaleido`) |

---

## Input requirements

The input table must contain these six columns. Every other column is ignored, so extra
columns are harmless.

| Column | Used for |
|--------|----------|
| `Authors` | Author identity and both networks |
| `Corresponding_Author` | Direction of the edges in the directed network |
| `Affiliations` | The `Affiliation` column of the author table |
| `Journal_Name` | Journal statistics |
| `Publication_Year` | Yearly statistics |
| `topic` | Topic coverage per author |

`tm_analyzer.py` writes all six, so its `data_topic.csv` can be used directly. If a
column is missing, the script stops and lists what the file actually contains.

**Author separator.** PubMed exports separate author names with `,`, which is the
default. Several other databases use `;` — pass `--author-separator ";"` for those, or
every record collapses into a single bogus author name.

---

## Generated files

```
<output-dir>/
├── author_information/
│   ├── author_affiliations.csv                    # one row per author
│   ├── author_cooccur_directed_nt.gml/.graphml/.gexf
│   └── author_cooccur_undirected_nt.gml/.graphml/.gexf
├── publication_information/
│   ├── articles_per_journal.csv
│   ├── yearly_publications.csv
│   └── articles_per_year_per_journal.csv
└── figures/
    ├── yearly_publications.png                    # 300 DPI, print-ready
    ├── yearly_publications_interactive.html
    ├── stacked_articles_per_year_per_journal.html
    ├── percentage_stacked_articles_per_year_per_journal.html
    └── articles_per_journal.html
```

### `author_affiliations.csv`

| Column | Description |
|--------|-------------|
| `ID` | Integer used as the node ID in the exported networks |
| `Author` | Author name |
| `Frequency` | Number of articles the author appears on |
| `Affiliation` | Affiliation (see the note below) |
| `Topic` | Every topic the author published in, comma separated |
| `Topic_color` | Single value for coloring a network: the author's topic if they published in only one, otherwise `max topic + 1` — a shared "multi-topic" category. `0` means the author has no assigned topic. |

### The two networks

| Network | Edge meaning |
|---------|--------------|
| `author_cooccur_directed_nt` | Corresponding author → each co-author. Shows who a lab's lead author publishes with. |
| `author_cooccur_undirected_nt` | Every pair of authors on a paper. Plain co-occurrence. |

Both are **weighted**: an edge's `weight` is the number of papers the pair shares.

Nodes are integer IDs, not names — names are noisy (accents, mojibake, inconsistent
casing), so the ID keeps the graph stable across runs. Each node carries three
attributes so the graph stays readable:

| Attribute | Description |
|-----------|-------------|
| `label` | Author name — what Gephi displays by default |
| `frequency` | Number of articles |
| `topics` | Topics the author published in |

Three formats are written for every graph: **GML**, **GraphML** (best for Gephi and for
`network_analyzer.py`) and **GEXF** (preserves the node attributes most faithfully).

---

## Implementation notes

- **Text repair before matching.** Author and affiliation strings pass through
  `normalize_all_text()`, which unescapes HTML entities, attempts a latin-1 → UTF-8
  round-trip to undo mojibake (`Ã¶` → `ö`), and applies NFKC normalization. Without this
  the same author appears under several spellings and the network fragments. The repair
  cannot recover text that was already lost — if the source file was decoded with the
  wrong codec and characters became `?`, that damage is permanent.
- **Author IDs are alphabetical.** IDs are assigned in sorted order, so the same input
  always produces the same graph and two runs can be compared node by node.
- **Affiliations are approximate.** PubMed stores affiliations as one `;`-joined field
  per article, not per author, so every author on a paper inherits the first listed
  affiliation, taken from the first article they appear in. Treat the `Affiliation`
  column as indicative rather than authoritative.
- **Topic 0 is excluded.** `tm_analyzer.py` assigns `topic = 0` to documents dropped
  during preprocessing. Those rows say nothing about an author's subject area, so they
  do not contribute to `Topic` or `Topic_color`.
- **Rows without a usable year** are dropped from the year-based tables and counted in a
  notice; they still count toward the per-journal totals.

---

## Troubleshooting

### `Error: Missing required column(s) [...]`
The input table is not a `tm_analyzer.py` output, or the wrong file was passed. The
message lists the columns the file actually has.

### Only one author was found, with a very long name
The author separator is wrong. Try `--author-separator ";"`.

### `ModuleNotFoundError: No module named 'openpyxl'`
Reading `.xlsx` needs it:
```bash
pip install openpyxl
```

### `Could not write ....png` when using `--png`
Static image export needs kaleido:
```bash
pip install kaleido
```
The HTML charts are written either way, so the run still completes.

### Author names look garbled in Gephi
Open the **GEXF** file rather than the GML one, and make sure Gephi is set to display the
`label` attribute.

---

## Related scripts

| Script | Role |
|--------|------|
| `ab_extractor.py` | Produces the abstract table from a PubMed query |
| `tm_analyzer.py` | Assigns a topic to every document (`data_topic.csv`) |

---

## License

MIT — see [LICENSE](LICENSE) in the repository root.
