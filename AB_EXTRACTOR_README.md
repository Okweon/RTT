# AB Extractor

PubMed abstract extractor and co-authorship network builder.

Given a PubMed search query and a publication-year filter, the script retrieves the
matching records through the NCBI E-utilities (Entrez) API and produces
(1) a CSV file of bibliographic metadata and abstracts, (2) co-authorship networks
as GML files and interactive HTML pages, and (3) descriptive publication statistics
as CSV tables and interactive HTML figures.

---

## Installation

```bash
pip install -r requirements_ab_extractor.txt
```

Python 3.9 or newer is required (verified on Python 3.11).

## NCBI credentials

NCBI requires a **contact e-mail address** for every E-utilities request. Pass it with
`--email`, or register it once as an environment variable.

```bash
# Windows PowerShell
$env:NCBI_EMAIL = "you@example.com"
$env:NCBI_API_KEY = "your_api_key"      # optional

# Linux / macOS
export NCBI_EMAIL="you@example.com"
export NCBI_API_KEY="your_api_key"      # optional
```

An API key is free from the [NCBI account settings page](https://www.ncbi.nlm.nih.gov/account/settings/)
and raises the request limit from 3 to 10 requests per second, which makes large
harvests considerably faster.

---

## Usage

### Basic usage

```bash
python ab_extractor.py --query "<PubMed query>" --year "<year filter>" --output-dir <output folder>
```

### Examples

```bash
# 1) All non-review articles with "nontuberculous" in the title
python ab_extractor.py \
    --query "nontuberculous[title] NOT review[pt]" \
    --year "after 1800" \
    --output-dir ./output/NTM \
    --email you@example.com

# 2) Keep a long query in a text file
python ab_extractor.py \
    --query-file food_supply_chain.txt \
    --year 2010-2024 \
    --output-dir ./output/FSC

# 3) CSV only - skip the figures and network pages for a fast run
python ab_extractor.py \
    --query "oral microbiome[title]" \
    --year 2015-2024 \
    --output-dir ./output/oral \
    --skip-networks --skip-plots
```

### Use from a Python script

```python
from ab_extractor import (
    configure_entrez, search_pubmed, fetch_details,
    save_abstracts_and_create_network,
)

configure_entrez("you@example.com")                      # required by NCBI
pmids = search_pubmed("nontuberculous[title]", "2010-2024", max_results=5000)
articles = fetch_details(pmids)                          # downloaded in batches of 200
df = save_abstracts_and_create_network(articles, "./output/NTM")

print(df.shape)
```

---

## Command line options

| Option | Default | Description |
|--------|---------|-------------|
| `-q`, `--query` | (required) | PubMed query string. Mutually exclusive with `--query-file` |
| `--query-file` | (required) | Path to a UTF-8 text file holding the query |
| `-y`, `--year` | no restriction | Publication year filter (see table below) |
| `-o`, `--output-dir` | (required) | Directory for all output files |
| `-e`, `--email` | `$NCBI_EMAIL` | Contact e-mail required by NCBI |
| `--api-key` | `$NCBI_API_KEY` | NCBI API key for a higher rate limit |
| `--max-results` | `10000` | Maximum number of records to retrieve |
| `--batch-size` | `200` | PMIDs per download request |
| `--csv-filename` | `pubmed_abstracts.csv` | Name of the main abstract CSV |
| `--top-nodes` | `300` | Authors kept in the interactive networks |
| `--skip-networks` | off | Do not render the interactive network HTML files |
| `--skip-plots` | off | Do not render the plotly figures |
| `--log-level` | `INFO` | Console verbosity (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |

### Year filter formats

| Input | Meaning |
|-------|---------|
| `2014` | That single year |
| `2001-2020` | An inclusive range |
| `before 2000` | Everything up to that year |
| `after 2023` | Everything from that year on |
| (omitted) | No date restriction |

---

## Generated files

```
<output-dir>/
├── pubmed_abstracts.csv                   # one row per article (all records)
├── pubmed_abstracts_filtered.csv          # only articles that carry an abstract
├── summary_pubmedSearch.csv               # record counts of this run
├── author_information/
│   ├── author_affiliations.csv            # articles per author + affiliations
│   ├── author_cooccur_directed_nt.gml     # corresponding author -> co-authors
│   ├── author_cooccur_undirected_nt.gml   # author co-occurrence
│   ├── interactive_directed_network.html
│   └── interactive_undirected_network.html
└── publication_information/
    ├── articles_per_journal.csv / .html
    ├── yearly_publications.csv / .html
    ├── articles_per_year_per_journal.csv / .html
    ├── stacked_articles_per_year_per_journal.html
    └── percentage_stacked_articles_per_year_per_journal.html
```

### Columns of the main CSV

| Column | Description |
|--------|-------------|
| `Index` | Position in the result set (oldest to newest) |
| `Publication_Year` | Publication year, or `Unknown` when it cannot be determined |
| `Abstract_ID` | Unique identifier of the form `<Index>_<year>` |
| `Journal_Name` | Journal title |
| `Title` | Article title |
| `Authors` | Author list, separated by `, ` |
| `Affiliations` | Affiliation list, separated by `; ` (duplicates removed) |
| `Keywords` | PubMed keywords |
| `Abstract` | Full abstract text, or `Not available` |
| `DOI` | DOI |
| `URL` | PubMed link |
| `Corresponding_Author` | Corresponding author (inferred) |
| `Publication_Type` | Article type (Journal Article, Review, ...) |

---

## Implementation notes

- **Corresponding author is inferred.** PubMed has no dedicated field for it. The script
  first looks for an author whose affiliation string contains `corresponding`; if none is
  found it falls back to the **last named author**, which is the usual convention. Treat
  this column as an estimate rather than a fact.
- **Abstracts are merged in full.** Structured abstracts (BACKGROUND / METHODS / RESULTS ...)
  arrive as several fragments. All fragments are concatenated together with their labels.
  (The original `ab_extractor_test_1.py` stored only the first fragment, so most of the
  body text was lost.)
- **Year fallback.** Older records carry a free-text `MedlineDate` (e.g. `1998 Nov-Dec`)
  instead of a `Year` element. The leading four digits are used in that case; if no year
  can be derived the value becomes `Unknown`. `Unknown` rows are excluded from the
  year-based figures only - they remain in the CSV files.
- **Large harvests.** esearch is paged in blocks of 10,000 records and efetch downloads
  `--batch-size` PMIDs at a time (200 by default). Failed requests are retried up to
  three times.
- **Visualisation scope.** Author networks can reach tens of thousands of nodes, which no
  browser renders usefully, so the HTML pages draw only the top `--top-nodes` authors
  (300 by default). **The GML files always contain the complete network**, so the full
  graph can still be analysed in Gephi or similar tools.

---

## Notes and limitations

1. Registering an e-mail address is mandatory under the NCBI usage policy. Sending heavy
   traffic without an API key may get the client throttled or temporarily blocked; use an
   API key for large harvests.
2. A large `--max-results` value can make a run take tens of minutes. Validate the query
   with a small value first (for example 100) before starting the full harvest.
3. `Publication_Year` may reflect the electronic publication date, so it can differ by
   about a year from the range given in `--year`. This is normal PubMed behaviour.
4. The interactive HTML pages load vis.js from a CDN, so **an internet connection is
   required to view them**.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'Bio'`
```bash
pip install -r requirements_ab_extractor.txt
```

### `An e-mail address is required`
Pass `--email`, or set the `NCBI_EMAIL` environment variable.

### `efetch ... failed after 3 attempts`
A transient NCBI error or too many requests. Lower `--batch-size` to about 100, set an
API key, and run again.

### Records are retrieved but most abstracts are empty
The result set contains many item types that have no abstract (Editorial, Comment,
Letter). Add `NOT (editorial[pt] OR comment[pt] OR letter[pt])` to the query.

### The HTML network pages are too heavy
Lower `--top-nodes` to 100 or less, or skip them with `--skip-networks` / `--skip-plots`
and open the GML files in Gephi instead.

---

## License

MIT — see [LICENSE](LICENSE) in the repository root.
