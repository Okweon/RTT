#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cooc_nt_builder.py
===========================================================================
Build a term co-occurrence network from a term-document matrix.

Two terms co-occur when they appear in the same document. Counting those
co-occurrences across a corpus produces a network in which the edges show
which concepts are discussed together, and the communities show how a field
organises its vocabulary.

The script produces three co-occurrence matrices, a network built from the
most frequent terms, and an interactive report.

Matrices (all terms)
--------------------
==========================  ===============================================
``cooc_network.csv``        Raw product of term counts: sum over documents
                            of ``count(a) * count(b)``. Weights emphasise
                            documents where both terms are frequent.
``weighted_cooc_nt.csv``    Document co-occurrence: how many documents
                            contain both terms. The usual choice.
``binary_cooc_nt.csv``      Presence/absence: 1 when the two terms ever
                            co-occur.
==========================  ===============================================

Network (top terms only)
------------------------
The graph is built from the ``--top-terms`` most frequent terms, because a
full-vocabulary network is too dense to read. It is exported as GML,
GraphML and GEXF, plus an interactive HTML page and an analysis report.

Usage
-----
    python cooc_nt_builder.py --input-tdm D_tdm.csv --output-dir ./cooc \\
        --top-terms 100

Requirements
------------
    pip install -r requirements_cooc_nt_builder.txt
"""

# ===========================================================================
# 0. Imports
# ===========================================================================
import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

import plotly.graph_objects as go
from pyvis.network import Network
from jinja2 import Template
from tqdm import tqdm

LOGGER = logging.getLogger("cooc_nt_builder")


# ===========================================================================
# 1. Constants
# ===========================================================================
#: A dense term x term matrix costs n^2 cells. 5,000 terms is already
#: 25 million cells (~200 MB as float64); beyond that the full matrices are
#: skipped rather than exhausting memory. Override with
#: --max-full-matrix-terms.
DEFAULT_MAX_FULL_MATRIX_TERMS = 5000

#: Number of most frequent terms kept for the network itself.
DEFAULT_TOP_TERMS = 100

#: Node colours in the interactive network: hubs and everything else.
HUB_COLOR = "rgb(255, 165, 0)"
NODE_COLOR = "rgb(173, 216, 230)"


# ===========================================================================
# 2. Co-occurrence matrices
# ===========================================================================
def _cooccurrence(matrix_values, labels):
    """
    Multiply a term-document matrix by its transpose.

    The multiplication runs in sparse form: a term-document matrix is mostly
    zeros, so the sparse product is far faster and lighter than the dense
    equivalent. The result is densified only for saving.

    :param matrix_values: 2-D array with terms in rows, documents in columns.
    :param labels: term labels for the rows and columns of the result.
    :return: a square DataFrame of co-occurrence counts.
    """
    sparse = csr_matrix(matrix_values)
    product = sparse.dot(sparse.T)
    return pd.DataFrame(product.toarray(), index=labels, columns=labels)


def _remove_self_loops(matrix):
    """Zero the diagonal: a term co-occurring with itself is not an edge."""
    np.fill_diagonal(matrix.values, 0)
    return matrix


def build_cooccurrence_matrices(df):
    """
    Build the three whole-vocabulary co-occurrence matrices.

    All three have their diagonal zeroed, so no matrix implies a self-loop.

    :param df: term-document matrix, terms in rows.
    :return: ``(count_product, document_cooccurrence, binary)``.
    """
    LOGGER.info("Computing the count-product co-occurrence matrix ...")
    count_product = _remove_self_loops(_cooccurrence(df.values, df.index))

    LOGGER.info("Computing the document co-occurrence matrix ...")
    presence = (df.values > 0).astype(np.int32)
    document_cooccurrence = _remove_self_loops(_cooccurrence(presence, df.index))

    binary = (document_cooccurrence > 0).astype(np.int8)

    return count_product, document_cooccurrence, binary


def save_cooccurrence_matrices(count_product, document_cooccurrence, binary,
                               output_dir):
    """Write the three whole-vocabulary matrices to CSV."""
    LOGGER.info("Saving co-occurrence matrices ...")
    count_product.to_csv(os.path.join(output_dir, "cooc_network.csv"))
    document_cooccurrence.to_csv(os.path.join(output_dir, "weighted_cooc_nt.csv"))
    binary.to_csv(os.path.join(output_dir, "binary_cooc_nt.csv"))


# ===========================================================================
# 3. Network construction
# ===========================================================================
def select_top_terms(df, tf_threshold=0, top_terms=DEFAULT_TOP_TERMS):
    """
    Keep the most frequent terms, which are the ones the network is built on.

    Terms are first filtered by total corpus frequency, then the highest
    ranking ones are kept. Both steps happen on the same filtered table, so
    a term can never be selected and then found missing.

    :param df: term-document matrix, terms in rows.
    :param tf_threshold: minimum total frequency a term must reach.
    :param top_terms: how many of the remaining terms to keep.
    :return: the reduced term-document matrix.
    :raises ValueError: when nothing survives the threshold.
    """
    term_frequencies = df.sum(axis=1)
    surviving = term_frequencies[term_frequencies >= tf_threshold]
    if surviving.empty:
        raise ValueError(
            f"No term reaches the frequency threshold of {tf_threshold}. "
            f"The highest term frequency in this matrix is "
            f"{term_frequencies.max()}.")

    selected = surviving.sort_values(ascending=False).head(top_terms).index
    LOGGER.info("Selected %d of %d terms (frequency >= %s, top %d).",
                len(selected), len(df), tf_threshold, top_terms)
    return df.loc[selected]


def build_network(top_term_df):
    """
    Build the co-occurrence graph of the selected terms.

    Edge weights are the count product between the two terms. Isolated
    nodes - terms that never share a document with any other selected term -
    are removed, because they carry no information in a co-occurrence
    network and only clutter the layout.

    :param top_term_df: reduced term-document matrix.
    :return: ``(graph, adjacency_matrix)``.
    """
    adjacency = _remove_self_loops(
        _cooccurrence(top_term_df.values, top_term_df.index))

    graph = nx.from_pandas_adjacency(adjacency)

    isolated = list(nx.isolates(graph))
    if isolated:
        graph.remove_nodes_from(isolated)
        adjacency = adjacency.drop(index=isolated, columns=isolated)
        LOGGER.info("Removed %d isolated term(s) from the network.", len(isolated))

    LOGGER.info("Network built: %d nodes, %d edges.",
                graph.number_of_nodes(), graph.number_of_edges())
    return graph, adjacency


def save_network(graph, output_dir):
    """Write the graph as GML, GraphML and GEXF."""
    stem = os.path.join(output_dir, "term_co_occurrence_network")
    nx.write_gml(graph, f"{stem}.gml")
    nx.write_graphml(graph, f"{stem}.graphml")
    nx.write_gexf(graph, f"{stem}.gexf")
    LOGGER.info("Network saved as GML, GraphML and GEXF.")


# ===========================================================================
# 4. Interactive network
# ===========================================================================
def build_interactive_network(graph, output_dir, hub_degree_threshold):
    """
    Render the network as a browsable HTML page.

    Nodes are sized by degree and coloured in two bands, so the hubs - the
    terms that co-occur with almost everything else - stand out from the
    periphery at a glance.

    :param hub_degree_threshold: nodes with a degree above this are drawn in
        the hub colour.
    """
    LOGGER.info("Creating the interactive network (hub degree > %d) ...",
                hub_degree_threshold)

    net = Network(height="750px", width="100%", bgcolor="white",
                  font_color="black", cdn_resources="remote",
                  neighborhood_highlight=True, select_menu=True)
    net.toggle_hide_edges_on_drag(True)
    # The repulsion solver keeps a dense term network readable.
    net.repulsion(node_distance=150, central_gravity=0.2,
                  spring_length=100, spring_strength=0.05)

    for node in tqdm(graph.nodes(), desc="Adding nodes"):
        degree = graph.degree(node)
        net.add_node(node, title=f"{node} (Degree: {degree})",
                     color=HUB_COLOR if degree > hub_degree_threshold else NODE_COLOR,
                     value=degree)

    # Edges are added explicitly with their weights. Calling from_nx() here
    # as well would add every edge a second time.
    for source, target, data in tqdm(graph.edges(data=True), desc="Adding edges"):
        net.add_edge(source, target, value=float(data.get("weight", 1)))

    out_path = os.path.join(output_dir, "term_co_occurrence_network.html")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(net.generate_html())
    LOGGER.info("Interactive network saved to %s", out_path)


# ===========================================================================
# 5. Report
# ===========================================================================
_REPORT_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Co-occurrence Network Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>
<body style="font-family: sans-serif; max-width: 900px; margin: 40px auto;">
    <h1>Co-occurrence Network Analysis Report</h1>
    <h2>Basic Network Information</h2>
    <table border="1" cellpadding="6" style="border-collapse: collapse;">
        <tr><th>Metric</th><th>Value</th></tr>
        {% for key, value in network_info.items() %}
        <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
        {% endfor %}
    </table>
    <h2>Degree Distribution</h2>
    {{ degree_distribution_html | safe }}
</body>
</html>
""")


def compute_network_statistics(graph):
    """
    Summarise the network.

    :return: an ordered dict of metric name to value, used by the report and
        written to ``network_statistics.csv``.
    """
    degrees = [degree for _, degree in graph.degree()]
    return {
        "Number of Nodes": graph.number_of_nodes(),
        "Number of Edges": graph.number_of_edges(),
        "Average Degree": float(np.mean(degrees)) if degrees else 0.0,
        "Maximum Degree": max(degrees) if degrees else 0,
        "Number of Modules (Connected Components)":
            nx.number_connected_components(graph),
        "Density": nx.density(graph),
        "Average Clustering Coefficient": nx.average_clustering(graph),
    }


def build_report(graph, network_info, output_dir):
    """
    Write the HTML analysis report and the statistics CSV.

    The report pairs the summary table with a degree histogram, so the shape
    of the network can be judged next to its headline numbers.
    """
    degrees = [degree for _, degree in graph.degree()]

    figure = go.Figure(data=[go.Histogram(
        x=degrees,
        nbinsx=max(1, min(50, max(degrees) if degrees else 1)),
        marker=dict(color="rgba(100, 149, 237, 0.7)",
                    line=dict(color="black", width=1)))])
    figure.update_layout(title="Degree Distribution", xaxis_title="Degree",
                         yaxis_title="Frequency", bargap=0.1,
                         template="ggplot2", width=770)

    html = _REPORT_TEMPLATE.render(
        network_info=network_info,
        degree_distribution_html=figure.to_html(full_html=False))

    report_path = os.path.join(output_dir, "term_cooccurrence_network_report.html")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write(html)

    pd.DataFrame(list(network_info.items()), columns=["metric", "value"]).to_csv(
        os.path.join(output_dir, "network_statistics.csv"), index=False)

    LOGGER.info("Report saved to %s", report_path)


# ===========================================================================
# 6. Pipeline
# ===========================================================================
def cooccurrence_nt_maker(input_file_path, output_dir, tf_threshold=0,
                          top_terms=DEFAULT_TOP_TERMS,
                          hub_degree_threshold=None,
                          max_full_matrix_terms=DEFAULT_MAX_FULL_MATRIX_TERMS,
                          skip_full_matrices=False):
    """
    Build the co-occurrence matrices, network, figures and report.

    :param input_file_path: CSV term-document matrix, terms in rows.
    :param output_dir: directory that receives every output file.
    :param tf_threshold: minimum total frequency for a term to be eligible.
    :param top_terms: how many of the most frequent terms form the network.
    :param hub_degree_threshold: degree above which a node is drawn as a hub;
        defaults to ``top_terms - 3``, i.e. terms that co-occur with almost
        every other selected term.
    :param max_full_matrix_terms: skip the whole-vocabulary matrices above
        this term count.
    :param skip_full_matrices: skip them regardless of size.
    :return: the network statistics dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    LOGGER.info("Loading term-document matrix from %s ...", input_file_path)
    df = pd.read_csv(input_file_path, index_col=0)
    LOGGER.info("TDM loaded: %d terms x %d documents.", df.shape[0], df.shape[1])

    if df.empty:
        raise ValueError("The term-document matrix is empty.")

    # ---- whole-vocabulary matrices ------------------------------------- #
    n_terms = df.shape[0]
    if skip_full_matrices:
        LOGGER.info("Skipping the whole-vocabulary matrices (--skip-full-matrices).")
    elif n_terms > max_full_matrix_terms:
        estimated_bytes = n_terms * n_terms * 8
        size = (f"{estimated_bytes / 1e9:.1f} GB" if estimated_bytes >= 1e9
                else f"{estimated_bytes / 1e6:.0f} MB")
        LOGGER.warning(
            "Skipping the whole-vocabulary matrices: %d terms would produce a "
            "%d x %d dense matrix (~%s in memory per file). Raise "
            "--max-full-matrix-terms to force it.",
            n_terms, n_terms, n_terms, size)
    else:
        count_product, document_cooccurrence, binary = \
            build_cooccurrence_matrices(df)
        save_cooccurrence_matrices(count_product, document_cooccurrence,
                                   binary, output_dir)

    # ---- network over the top terms -------------------------------------- #
    top_term_df = select_top_terms(df, tf_threshold, top_terms)
    graph, _ = build_network(top_term_df)

    if graph.number_of_nodes() == 0:
        raise ValueError("No term co-occurs with any other selected term; "
                         "the network is empty. Try a larger --top-terms.")

    save_network(graph, output_dir)

    if hub_degree_threshold is None:
        hub_degree_threshold = max(0, top_terms - 3)
    build_interactive_network(graph, output_dir, hub_degree_threshold)

    network_info = compute_network_statistics(graph)
    build_report(graph, network_info, output_dir)

    LOGGER.info("All outputs saved to %s", os.path.abspath(output_dir))
    return network_info


# ===========================================================================
# 7. Command line interface
# ===========================================================================
def parse_args(argv=None):
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        prog="cooc_nt_builder.py",
        description="Build a term co-occurrence network from a "
                    "term-document matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-tdm", required=True,
                        help="CSV term-document matrix: terms in rows, "
                             "documents in columns.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory for all output files.")

    parser.add_argument("--tf-threshold", type=int, default=0,
                        help="Minimum total corpus frequency for a term to be "
                             "eligible for the network.")
    parser.add_argument("--top-terms", type=int, default=DEFAULT_TOP_TERMS,
                        help="Number of most frequent terms forming the network.")
    parser.add_argument("--hub-degree-threshold", type=int, default=None,
                        help="Degree above which a node is drawn as a hub. "
                             "Defaults to --top-terms minus 3, i.e. terms that "
                             "co-occur with almost every other selected term.")

    parser.add_argument("--max-full-matrix-terms", type=int,
                        default=DEFAULT_MAX_FULL_MATRIX_TERMS,
                        help="Skip the whole-vocabulary matrices above this "
                             "term count; they are dense and cost n^2 cells.")
    parser.add_argument("--skip-full-matrices", action="store_true",
                        help="Never write the whole-vocabulary matrices.")

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console log verbosity.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Entry point.

    :return: 0 on success, 1 when the input cannot be used.
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        network_info = cooccurrence_nt_maker(
            args.input_tdm, args.output_dir,
            tf_threshold=args.tf_threshold,
            top_terms=args.top_terms,
            hub_degree_threshold=args.hub_degree_threshold,
            max_full_matrix_terms=args.max_full_matrix_terms,
            skip_full_matrices=args.skip_full_matrices,
        )
    except (ValueError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print()
    for metric, value in network_info.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {metric:<44} {formatted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
