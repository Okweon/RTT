# Co-occurrence Network Builder

Build a term co-occurrence network from a term-document matrix.

Two terms co-occur when they appear in the same document. Counting those co-occurrences
across a corpus produces a network in which the edges show which concepts are discussed
together, and the communities show how a field organises its vocabulary.

---

## Where it fits

```
tdm_generator.py  →  cooc_nt_builder.py  →  network_analyzer.py
   D_tdm.csv          co-occurrence matrices     degree / clustering figures
                      + term network (GML/GraphML/GEXF)
                      + interactive HTML
```

---

## Installation

```bash
pip install -r requirements_cooc_nt_builder.txt
```

Python 3.9 or newer is required (verified on Python 3.11).

---

## Usage

### Basic usage

```bash
python cooc_nt_builder.py --input-tdm <tdm.csv> --output-dir <output folder>
```

### Examples

```bash
# 1) Network of the 100 most frequent terms
python cooc_nt_builder.py \
    --input-tdm ./tdm/D_tdm.csv \
    --output-dir ./cooc \
    --top-terms 100

# 2) Author keywords instead of full text, phrases kept whole
python cooc_nt_builder.py \
    --input-tdm ./tdm/D_keywords_tdm.csv \
    --output-dir ./cooc_keywords \
    --top-terms 60

# 3) Large vocabulary: network only, no whole-vocabulary matrices
python cooc_nt_builder.py \
    --input-tdm ./tdm/D_tdm.csv \
    --output-dir ./cooc_big \
    --top-terms 150 --skip-full-matrices
```

### Use from a Python script

```python
from cooc_nt_builder import cooccurrence_nt_maker

info = cooccurrence_nt_maker(
    "./tdm/D_tdm.csv", "./cooc",
    tf_threshold=0, top_terms=100,
)
print(info["Number of Edges"], info["Density"])
```

---

## Command line options

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input-tdm` | (required) | CSV term-document matrix: terms in rows, documents in columns |
| `-o`, `--output-dir` | (required) | Directory for all output files |
| `--tf-threshold` | `0` | Minimum total corpus frequency for a term to be eligible |
| `--top-terms` | `100` | Number of most frequent terms forming the network |
| `--hub-degree-threshold` | `--top-terms` − 3 | Degree above which a node is drawn as a hub |
| `--max-full-matrix-terms` | `5000` | Skip the whole-vocabulary matrices above this term count |
| `--skip-full-matrices` | off | Never write the whole-vocabulary matrices |
| `--log-level` | `INFO` | Console verbosity |

---

## Generated files

```
<output-dir>/
├── cooc_network.csv                        # count-product matrix (all terms)
├── weighted_cooc_nt.csv                    # document co-occurrence matrix (all terms)
├── binary_cooc_nt.csv                      # presence/absence matrix (all terms)
├── term_co_occurrence_network.gml/.graphml/.gexf   # network of the top terms
├── term_co_occurrence_network.html         # interactive network
├── term_cooccurrence_network_report.html   # statistics table + degree histogram
└── network_statistics.csv                  # the same statistics as a table
```

### The three matrices

All three cover the **whole vocabulary** and all three have a zeroed diagonal, so none of
them implies a self-loop.

| File | Cell value | When to use it |
|------|-----------|----------------|
| `weighted_cooc_nt.csv` | Number of documents containing **both** terms | The usual choice. Interpretable and robust |
| `cooc_network.csv` | Σ over documents of `count(a) × count(b)` | Emphasises documents where both terms are frequent, not merely present |
| `binary_cooc_nt.csv` | `1` when the two terms ever co-occur | Structure only, ignoring strength |

### The network

The graph is built from the `--top-terms` most frequent terms, **not** the whole
vocabulary: a full-vocabulary network is too dense to read and too large to lay out. Its
edge weights are the count product between the two terms.

Isolated terms — those that never share a document with any other selected term — are
removed before export.

---

## Reading the interactive network

- **Node size** is degree: how many other selected terms this one co-occurs with.
- **Node colour** marks hubs. A node is orange when its degree exceeds
  `--hub-degree-threshold`, which defaults to `--top-terms − 3` — that is, terms that
  co-occur with almost every other selected term. These are usually the field's generic
  vocabulary and are good candidates for the custom stop word list.
- **Edge thickness** is the co-occurrence weight.
- The select menu and neighbourhood highlighting isolate one term's neighbours.

**High density is expected.** With the 100 most frequent terms of a focused corpus,
densities above 0.9 are normal — frequent terms co-occur with nearly everything. If the
goal is a readable structure rather than a hairball, lower `--top-terms`, raise
`--tf-threshold`, or prune weak edges in Gephi after export.

---

## Implementation notes

- **The co-occurrence product runs on sparse matrices.** A term-document matrix is mostly
  zeros, so `csr_matrix(X) @ csr_matrix(X).T` is far faster and lighter than the dense
  equivalent. The result is densified only to save it.
- **Whole-vocabulary matrices are guarded by size.** A dense term × term matrix costs n²
  cells: 20,000 terms is 400 million cells, roughly 3 GB per file. Above
  `--max-full-matrix-terms` they are skipped with a notice, and the network is still
  built.
- **Self-loops are removed from all three matrices**, including the binary one.
- **Edge weights reach the interactive page**, so edge thickness reflects co-occurrence
  strength rather than every edge rendering identically.
- **Term selection happens on one table.** Frequency filtering and top-N selection both
  operate on the filtered frame, so a term can never be selected and then found missing.

---

## Notes and limitations

1. **Raw co-occurrence favours frequent terms.** All three matrices count co-occurrence
   without normalising for how often each term appears on its own. For association
   strength rather than raw counts, normalise afterwards (PMI, Jaccard, or a
   cosine/correlation measure) using `weighted_cooc_nt.csv` as the input.
2. The network reflects the vocabulary of the TDM it was given. Garbage terms that
   survived preprocessing become hubs — check the hub list and feed the offenders back
   into `tdm_generator.py`'s custom stop word CSV.
3. `--top-terms` is a frequency cut, not a significance cut. A term that is rare overall
   but central to one sub-community will not appear.
4. The interactive page loads vis.js from a CDN, and the report loads plotly from a CDN,
   so both need an internet connection to view.

---

## Troubleshooting

### `No term reaches the frequency threshold of N`
`--tf-threshold` is above every term's total frequency. The error reports the highest
frequency actually present in the matrix.

### `No term co-occurs with any other selected term; the network is empty`
The matrix is extremely sparse, or `--top-terms` is very small. Raise `--top-terms`.

### `Skipping the whole-vocabulary matrices: N terms would produce ...`
Expected on a large vocabulary. Raise `--max-full-matrix-terms` if you really want those
files and have the memory, or leave them out — the network is built either way.

### The network is a solid blob
See "High density is expected" above: lower `--top-terms`, raise `--tf-threshold`, or
prune edges after export.

### The HTML page is blank
Both HTML outputs need an internet connection for their CDN scripts. Check the browser
console.

---

## Related scripts

| Script | Role |
|--------|------|
| `tdm_generator.py` | Produces the term-document matrix this script consumes |
| `network_analyzer.py` | Analyzes the exported `.graphml` structurally |

---

## License

MIT — see [LICENSE](LICENSE) in the repository root.
