# Network Analyzer

Structural analysis of a co-authorship network — or any other network — producing
publication-quality figures and a summary statistics table.

The script reads a network file, detects whether it is directed or undirected, and runs
the analysis appropriate to that type. Every figure is saved at 300 DPI with large fonts,
sized to stay legible in a single journal column.

---

## Where it fits

```
post_hoc_analyzer.py  →  network_analyzer.py
   .graphml / .gml / .gexf     degree, clustering, community, PageRank figures
```

The three network formats `post_hoc_analyzer.py` writes are all accepted directly.

---

## Installation

```bash
pip install -r requirements_network_analyzer.txt
```

Python 3.9 or newer is required (verified on Python 3.11).

---

## Usage

### Basic usage

```bash
python network_analyzer.py <input_network> <output_directory>
```

### Examples

```bash
# Undirected co-authorship network
python network_analyzer.py \
    ./post_hoc_analysis/author_information/author_cooccur_undirected_nt.graphml \
    ./network_analysis/undirected

# Directed network, figures as vector PDFs
python network_analyzer.py \
    ./post_hoc_analysis/author_information/author_cooccur_directed_nt.gexf \
    ./network_analysis/directed \
    --format pdf

# Large network: skip the metrics that scale poorly
python network_analyzer.py big_network.graphml ./analysis \
    --max-nodes-slow-metrics 1000
```

### Use from a Python script

```python
from network_analyzer import analyze_network

stats = analyze_network(
    "author_cooccur_undirected_nt.graphml",
    "./network_analysis",
)
print(stats["average_degree"], stats.get("powerlaw_gamma"))
```

---

## Command line options

| Option | Default | Description |
|--------|---------|-------------|
| `input_file` | (required) | Input network: `.graphml`, `.gml` or `.gexf` |
| `output_dir` | (required) | Directory for the figures and statistics |
| `--format` | `png` | Image format: `png`, `pdf` or `svg` |
| `--max-nodes-slow-metrics` | `5000` | Skip average path length, diameter and community detection above this node count |

The two positional arguments come first, so existing command lines keep working.

---

## Generated files

### Undirected network

| File | Content |
|------|---------|
| `undirected_degree_distribution_loglog` | Degree distribution on log-log axes with a power-law fit |
| `undirected_degree_histogram` | Degree distribution histogram |
| `undirected_degree_cumulative_distribution` | Number of nodes with degree ≥ k |
| `undirected_clustering_vs_threshold` | Average clustering of the subgraph of hubs |
| `undirected_clustering_vs_degree_scatter` | Per-node clustering against degree |
| `undirected_community_size_distribution` | Community sizes (greedy modularity) |
| `network_statistics.csv` | Every summary statistic below |

### Directed network

| File | Content |
|------|---------|
| `directed_degree_distribution_loglog` | Total-degree distribution with a power-law fit |
| `directed_in_degree_histogram` | In-degree distribution |
| `directed_out_degree_histogram` | Out-degree distribution |
| `directed_in_cumulative_distribution` | Number of nodes with in-degree ≥ k |
| `directed_out_cumulative_distribution` | Number of nodes with out-degree ≥ k |
| `directed_pagerank_histogram` | PageRank score distribution |
| `directed_pagerank_cumulative` | PageRank score against author rank |
| `directed_clustering_vs_threshold` | Average clustering of the subgraph of hubs |
| `directed_clustering_vs_degree_scatter` | Per-node clustering against degree |
| `directed_community_size_distribution` | Community sizes (greedy modularity) |
| `network_statistics.csv` | Every summary statistic below |

---

## Summary statistics

`network_statistics.csv` holds one `metric,value` row per entry, and the same values are
printed to the console.

### Undirected

`nodes`, `edges`, `average_degree`, `average_clustering`, `transitivity`,
`n_communities`, `powerlaw_gamma`, `powerlaw_gamma_std_err`, `powerlaw_r_squared`, plus
either `average_path_length` and `diameter` (connected network) or
`largest_component_nodes` and `largest_component_fraction` (disconnected).

### Directed

`nodes`, `edges`, `average_in_degree`, `average_out_degree`, `n_communities`, the
power-law fields, `is_weakly_connected`, `is_strongly_connected`, and the size of the
largest weak or strong component when the network is not connected.

---

## Reading the figures

- **Log-log degree distribution.** A straight line indicates a scale-free degree
  distribution, `P(k) ~ k^-gamma`. Read the **R²** in the annotation before quoting the
  exponent: a low R² means the straight line does not describe the data, and the
  exponent is then meaningless. In a co-authorship network gamma is typically 2–3.
- **Cumulative degree distribution.** Each point `(k, N)` says that `N` nodes have degree
  **k or higher**. This view is far less sensitive to binning than a histogram, which is
  why it is the standard way to judge a heavy tail.
- **Clustering versus degree threshold.** A rising curve means the well-connected nodes
  form a denser core than the network at large — the signature of a hub-and-spoke
  collaboration structure.
- **Clustering versus degree scatter.** A downward trend means hubs bridge otherwise
  separate groups rather than sitting inside one dense cluster (a hierarchical structure).
- **PageRank on a corresponding-author network** reads as author influence: a node scores
  highly when it is named as a co-author by authors who are themselves frequently named.

---

## Implementation notes

- **The analysis type is chosen from the file**, not from a flag: a directed graph gets
  the in/out-degree and PageRank treatment, an undirected one the clustering and
  community treatment.
- **Clustering and community detection always run on an undirected view.** Both metrics
  are defined on undirected structure, so a directed input is converted first.
- **Isolated nodes are excluded from the power-law fit.** Degree 0 has no logarithm. They
  still appear in the histogram and in the node count.
- **Slow metrics are guarded.** Average shortest path length, diameter and greedy
  modularity all scale worse than linearly and can run for hours on a large network, so
  they are skipped above `--max-nodes-slow-metrics` with a notice rather than left
  running silently.
- **Degenerate inputs degrade gracefully.** An empty network, or one with too few
  distinct degrees to fit a power law, produces a notice per figure instead of a
  traceback.

---

## Notes and limitations

1. The power-law exponent comes from a least-squares fit to the binned degree histogram.
   This is the conventional quick estimate, not a rigorous one — for a publication claim
   about scale-freeness, use a maximum-likelihood estimator with a Kolmogorov-Smirnov
   goodness-of-fit test (the `powerlaw` package implements Clauset et al.'s method).
2. `directed_pagerank_cumulative` is a rank plot (score against rank), not a cumulative
   distribution. The filename is kept for continuity with earlier output; the axis labels
   describe what is actually plotted.
3. Community detection uses greedy modularity maximisation, which is deterministic but
   tends to produce a few large communities. Other algorithms (Louvain, Leiden) may split
   the same network differently.
4. Path length and diameter are only defined on a connected network. When the input is
   disconnected, the size of the largest component is reported instead.

---

## Troubleshooting

### `Error: Unsupported network format '.csv'`
Only `.graphml`, `.gml` and `.gexf` are supported. `post_hoc_analyzer.py` writes all
three for every network it builds.

### `Skipping log-log degree distribution: fewer than 3 distinct non-zero degrees`
The network is too small or too regular to fit a power law. Every other figure is still
produced.

### The run seems to hang
Community detection or the path metrics are running on a large network. Interrupt it and
re-run with a lower `--max-nodes-slow-metrics`.

### Figures look cramped when placed in a manuscript
Use `--format pdf` (or `svg`) for vector output that scales without pixelation.

### `ImportError` on `scipy` or `networkx`
```bash
pip install -r requirements_network_analyzer.txt
```

---

## Related scripts

| Script | Role |
|--------|------|
| `ab_extractor.py` | Produces the abstract table from a PubMed query |
| `tm_analyzer.py` | Assigns a topic to every document |
| `post_hoc_analyzer.py` | Builds the co-authorship networks this script analyzes |

---

## License

MIT — see [LICENSE](LICENSE) in the repository root.
