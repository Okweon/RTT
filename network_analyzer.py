#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network_analyzer.py
===========================================================================
Structural analysis of a co-authorship (or any other) network, producing
publication-quality figures and a summary statistics table.

The script reads a network file, detects whether it is directed or
undirected, and runs the analysis appropriate to that type:

**Undirected**
    degree distribution (log-log with a power-law fit, histogram,
    complementary cumulative), clustering coefficient versus degree
    threshold and versus degree, community size distribution.

**Directed**
    the same degree analyses split into in-degree and out-degree, plus
    PageRank distribution and rank plot.

Every figure is saved at 300 DPI with large fonts, sized for a journal
column. A ``network_statistics.csv`` summarising the network accompanies
them.

Usage
-----
    python network_analyzer.py <input_network> <output_directory>

    python network_analyzer.py author_cooccur_undirected_nt.graphml ./analysis
    python network_analyzer.py author_cooccur_directed_nt.gexf ./analysis --format pdf

Input formats: ``.graphml``, ``.gml``, ``.gexf`` - the three formats
``post_hoc_analyzer.py`` writes.

Requirements
------------
    pip install -r requirements_network_analyzer.txt
"""

# ===========================================================================
# 0. Imports
# ===========================================================================
import os
import csv
import sys
import argparse
import warnings
from collections import Counter

import numpy as np
import networkx as nx
from scipy import stats

import matplotlib
matplotlib.use("Agg")   # headless-safe backend; must be set before pyplot
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ===========================================================================
# 1. Figure style
#    Publication defaults: 300 DPI and font sizes that stay legible when a
#    10x8 inch figure is scaled down to a single journal column.
# ===========================================================================
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16

FIGURE_SIZE = (10, 8)
PRIMARY_COLOR = "steelblue"

#: Readers used for each supported input extension.
NETWORK_READERS = {
    ".graphml": nx.read_graphml,
    ".gml": nx.read_gml,
    ".gexf": nx.read_gexf,
}

#: Above this node count, metrics that scale worse than linearly (average
#: shortest path length, diameter, community detection) are skipped rather
#: than left running for hours. Override with --max-nodes-slow-metrics.
DEFAULT_MAX_NODES_SLOW_METRICS = 5000


# ===========================================================================
# 2. Small helpers
# ===========================================================================
def _save_figure(fig, output_dir, filename_stem, image_format="png"):
    """Save and close a figure, then report the path."""
    filename = f"{filename_stem}.{image_format}"
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def _style_axes(ax, xlabel, ylabel, title, grid_axis="both"):
    """Apply the shared label, title and grid styling to an axis."""
    ax.set_xlabel(xlabel, fontsize=20, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.grid(True, alpha=0.3, axis=grid_axis, linestyle="--")


def _degrees(graph, degree_type="degree"):
    """
    Return the degree sequence of a graph.

    :param degree_type: ``degree``, ``in`` or ``out``. The directed variants
        require a directed graph.
    :return: ``(degrees, label)`` where label names the quantity for axes.
    """
    if degree_type == "in":
        return [d for _, d in graph.in_degree()], "In-Degree"
    if degree_type == "out":
        return [d for _, d in graph.out_degree()], "Out-Degree"
    return [d for _, d in graph.degree()], "Degree"


def _as_undirected(graph):
    """Return an undirected view of the graph, for metrics that need one."""
    return graph.to_undirected() if graph.is_directed() else graph


def _histogram_bins(values, maximum=50):
    """
    Choose a bin count that suits the sample size.

    A fixed 50 bins leaves a small network's histogram mostly empty, so the
    count is capped by the number of distinct values.
    """
    distinct = len(set(values))
    return max(1, min(maximum, distinct))


# ===========================================================================
# 3. Loading
# ===========================================================================
def load_network(file_path):
    """
    Load a network from a GraphML, GML or GEXF file.

    :param file_path: path to the network file.
    :return: the NetworkX graph.
    :raises ValueError: on an unsupported extension.
    """
    extension = os.path.splitext(file_path)[1].lower()
    reader = NETWORK_READERS.get(extension)
    if reader is None:
        raise ValueError(
            f"Unsupported network format '{extension}'. "
            f"Supported: {', '.join(sorted(NETWORK_READERS))}")

    print(f"Loading network from {file_path}...")
    graph = reader(file_path)
    print(f"Network loaded: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")
    print(f"Network type: {'Directed' if graph.is_directed() else 'Undirected'}")
    return graph


# ===========================================================================
# 4. Degree distribution figures
# ===========================================================================
def plot_degree_distribution_loglog(graph, output_dir, network_type="undirected",
                                    image_format="png"):
    """
    Plot the degree distribution on log-log axes with a power-law fit.

    A straight line on log-log axes indicates a scale-free degree
    distribution, ``P(k) ~ k^-gamma``. The exponent comes from a linear
    regression of ``log10(count)`` on ``log10(degree)``; the R-squared value
    in the annotation says how well that straight line actually describes the
    data, and should be read before quoting the exponent.

    :return: ``{"gamma", "gamma_std_err", "r_squared"}``, or None when the
        network has too few distinct degrees to fit.
    """
    degrees, _ = _degrees(graph)
    if not degrees:
        print("Skipping log-log degree distribution: the network has no nodes.")
        return None

    degree_count = Counter(degrees)
    deg, cnt = (np.array(x) for x in zip(*sorted(degree_count.items())))

    # Zero degrees have no logarithm; isolated nodes are excluded from the fit.
    mask = (deg > 0) & (cnt > 0)
    deg_fit_input, cnt_fit_input = deg[mask], cnt[mask]

    if len(deg_fit_input) < 3:
        print("Skipping log-log degree distribution: fewer than 3 distinct "
              "non-zero degrees, too few to fit a power law.")
        return None

    log_deg = np.log10(deg_fit_input)
    log_cnt = np.log10(cnt_fit_input)
    slope, intercept, r_value, _, std_err = stats.linregress(log_deg, log_cnt)

    log_x = np.linspace(log_deg.min(), log_deg.max(), 100)
    fit_x = 10 ** log_x
    fit_y = 10 ** (slope * log_x + intercept)

    gamma = -slope                  # power-law exponent, reported as positive
    r_squared = r_value ** 2

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.scatter(deg, cnt, alpha=0.6, s=80, edgecolors="black", linewidth=0.7,
               label="Data", color=PRIMARY_COLOR, zorder=2)
    ax.plot(fit_x, fit_y, "r--", linewidth=2.5, label="Power-law fit", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_axes(ax, "Degree (k)", "Number of nodes", "Degree Distribution (Log-Log)")

    annotation = (f"P(k) ∝ k$^{{-{gamma:.2f}}}$\n"
                  f"γ = {gamma:.2f} ± {std_err:.2f}\n"
                  f"R² = {r_squared:.3f}")
    ax.text(0.95, 0.95, annotation, transform=ax.transAxes, fontsize=16,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.75), fontsize=16)

    _save_figure(fig, output_dir, f"{network_type}_degree_distribution_loglog",
                 image_format)
    print(f"  Power-law exponent gamma = {gamma:.2f} +/- {std_err:.2f}, "
          f"R^2 = {r_squared:.3f}")
    return {"gamma": gamma, "gamma_std_err": std_err, "r_squared": r_squared}


def plot_degree_histogram(graph, output_dir, network_type="undirected",
                          degree_type="degree", image_format="png"):
    """Plot the degree distribution as a histogram."""
    degrees, label = _degrees(graph, degree_type)
    if not degrees:
        print(f"Skipping {label} histogram: the network has no nodes.")
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(degrees, bins=_histogram_bins(degrees), edgecolor="black",
            alpha=0.7, color=PRIMARY_COLOR)
    _style_axes(ax, label, "Frequency", f"{label} Distribution", grid_axis="y")

    # "degree" is implied for an undirected network; avoid "degree_degree".
    stem = (f"{network_type}_degree_histogram" if degree_type == "degree"
            else f"{network_type}_{degree_type}_degree_histogram")
    _save_figure(fig, output_dir, stem, image_format)


def plot_cumulative_degree_distribution(graph, output_dir, network_type="undirected",
                                        degree_type="degree", image_format="png"):
    """
    Plot the complementary cumulative degree distribution.

    Each point ``(k, N)`` says that ``N`` nodes have degree **k or higher**.
    This view is far less sensitive to binning than a histogram, which is why
    it is the standard way to judge a heavy tail.
    """
    degrees, label = _degrees(graph, degree_type)
    if not degrees:
        print(f"Skipping cumulative {label} distribution: the network has no nodes.")
        return

    degree_count = Counter(degrees)
    deg, cnt = (np.array(x) for x in zip(*sorted(degree_count.items())))

    # Count nodes at or above each degree: sum the counts from the largest
    # degree downwards, then flip back to ascending order.
    cumulative = np.cumsum(cnt[::-1])[::-1]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(deg, cumulative, linewidth=3, color=PRIMARY_COLOR)
    _style_axes(ax, label, "Number of nodes with degree ≥ k",
                f"Cumulative {label} Distribution")

    _save_figure(fig, output_dir,
                 f"{network_type}_{degree_type}_cumulative_distribution", image_format)


# ===========================================================================
# 5. Clustering figures
# ===========================================================================
def plot_clustering_vs_degree_threshold(graph, output_dir, network_type="undirected",
                                        image_format="png"):
    """
    Plot the average clustering coefficient of the subgraph of hubs.

    At each threshold ``k`` the graph is restricted to nodes of degree ``>= k``
    and the average clustering of that subgraph is measured. A rising curve
    means the well-connected nodes form a denser core than the network at
    large - the signature of a hub-and-spoke collaboration structure.

    Directed input is converted to undirected first; clustering is defined on
    the undirected structure.
    """
    undirected = _as_undirected(graph)
    if undirected.number_of_nodes() == 0:
        print("Skipping clustering vs threshold: the network has no nodes.")
        return

    max_degree = max(d for _, d in undirected.degree())
    # Cap at 100: beyond that the subgraph holds only a handful of hubs and
    # the average becomes noise. Step 5 keeps the curve readable.
    thresholds = list(range(0, min(max_degree, 100) + 1, 5)) or [0]

    average_clustering = []
    for threshold in thresholds:
        nodes = [n for n, d in undirected.degree() if d >= threshold]
        subgraph = undirected.subgraph(nodes)
        if subgraph.number_of_nodes() > 0:
            average_clustering.append(nx.average_clustering(subgraph))
        else:
            average_clustering.append(0.0)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(thresholds, average_clustering, linewidth=3, color=PRIMARY_COLOR,
            marker="o", markersize=8)
    _style_axes(ax, "Degree Threshold (k)", "Average Clustering Coefficient",
                "Clustering vs Degree Threshold")

    _save_figure(fig, output_dir, f"{network_type}_clustering_vs_threshold",
                 image_format)


def plot_clustering_vs_degree_scatter(graph, output_dir, network_type="undirected",
                                      image_format="png"):
    """
    Scatter each node's clustering coefficient against its degree.

    A downward trend (high-degree nodes with low clustering) indicates a
    hierarchical structure: hubs bridge otherwise separate groups instead of
    sitting inside one dense cluster.
    """
    undirected = _as_undirected(graph)
    if undirected.number_of_nodes() == 0:
        print("Skipping clustering vs degree scatter: the network has no nodes.")
        return

    clustering = nx.clustering(undirected)
    degrees = dict(undirected.degree())

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.scatter([degrees[n] for n in clustering], list(clustering.values()),
               alpha=0.5, s=50, edgecolors="black", linewidth=0.5,
               color=PRIMARY_COLOR)
    _style_axes(ax, "Degree", "Clustering Coefficient",
                "Clustering Coefficient vs Degree")

    _save_figure(fig, output_dir, f"{network_type}_clustering_vs_degree_scatter",
                 image_format)


# ===========================================================================
# 6. Community and PageRank figures
# ===========================================================================
def plot_community_size_distribution(graph, output_dir, network_type="undirected",
                                     image_format="png",
                                     max_nodes=DEFAULT_MAX_NODES_SLOW_METRICS):
    """
    Detect communities by greedy modularity maximisation and plot their sizes.

    Greedy modularity is agglomerative and scales poorly, so the detection is
    skipped above ``max_nodes`` rather than left running.

    :return: the number of communities found, or None when skipped.
    """
    undirected = _as_undirected(graph)
    if undirected.number_of_nodes() == 0:
        print("Skipping community detection: the network has no nodes.")
        return None
    if undirected.number_of_nodes() > max_nodes:
        print(f"Skipping community detection: {undirected.number_of_nodes()} nodes "
              f"exceeds the --max-nodes-slow-metrics limit of {max_nodes}.")
        return None

    try:
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(undirected)
    except Exception as exc:
        print(f"Warning: community detection failed: {exc}")
        return None

    community_sizes = [len(c) for c in communities]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(community_sizes, bins=_histogram_bins(community_sizes, maximum=30),
            edgecolor="black", alpha=0.7, color=PRIMARY_COLOR)
    _style_axes(ax, "Community Size", "Frequency",
                f"Community Size Distribution (n={len(communities)})", grid_axis="y")

    _save_figure(fig, output_dir, f"{network_type}_community_size_distribution",
                 image_format)
    return len(communities)


def plot_pagerank_distribution(graph, output_dir, network_type="directed",
                               image_format="png"):
    """
    Plot the PageRank distribution of a directed network.

    Two figures are produced:

    - a histogram of the scores, and
    - a rank plot: every author's score against their rank, sorted from
      highest to lowest. A steep drop at the left means influence is
      concentrated in a few authors.

    PageRank on the corresponding-author network reads as author influence:
    a node scores highly when it is named as a co-author by authors who are
    themselves frequently named.
    """
    if not graph.is_directed():
        print("PageRank analysis applies only to directed networks.")
        return
    if graph.number_of_nodes() == 0:
        print("Skipping PageRank: the network has no nodes.")
        return

    pagerank = nx.pagerank(graph)
    scores = list(pagerank.values())

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(scores, bins=_histogram_bins(scores), edgecolor="black",
            alpha=0.7, color=PRIMARY_COLOR)
    _style_axes(ax, "PageRank Score", "Frequency", "PageRank Distribution",
                grid_axis="y")
    _save_figure(fig, output_dir, f"{network_type}_pagerank_histogram", image_format)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(range(1, len(scores) + 1), sorted(scores, reverse=True),
            linewidth=3, color=PRIMARY_COLOR)
    _style_axes(ax, "Author Rank", "PageRank Score", "PageRank by Author Rank")
    _save_figure(fig, output_dir, f"{network_type}_pagerank_cumulative", image_format)


# ===========================================================================
# 7. Analyses
# ===========================================================================
def analyze_undirected_network(graph, output_dir, image_format="png",
                               max_nodes_slow=DEFAULT_MAX_NODES_SLOW_METRICS):
    """
    Run the undirected analysis and return its summary statistics.

    :return: dict of statistics, also written to ``network_statistics.csv``.
    """
    print("\n" + "=" * 60)
    print("ANALYZING UNDIRECTED NETWORK")
    print("=" * 60)

    network_type = "undirected"
    fit = plot_degree_distribution_loglog(graph, output_dir, network_type, image_format)
    plot_degree_histogram(graph, output_dir, network_type, "degree", image_format)
    plot_cumulative_degree_distribution(graph, output_dir, network_type, "degree",
                                        image_format)
    plot_clustering_vs_degree_threshold(graph, output_dir, network_type, image_format)
    plot_clustering_vs_degree_scatter(graph, output_dir, network_type, image_format)
    n_communities = plot_community_size_distribution(graph, output_dir, network_type,
                                                     image_format, max_nodes_slow)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    statistics = {
        "network_type": "undirected",
        "nodes": n_nodes,
        "edges": n_edges,
        "average_degree": (sum(d for _, d in graph.degree()) / n_nodes) if n_nodes else 0.0,
        "average_clustering": nx.average_clustering(graph) if n_nodes else 0.0,
        "transitivity": nx.transitivity(graph) if n_nodes else 0.0,
        "n_communities": n_communities,
    }
    if fit:
        statistics.update({f"powerlaw_{k}": v for k, v in fit.items()})

    # Path-based metrics are only defined on a connected graph, and cost
    # O(nodes * edges) even then, so they are guarded on both counts.
    if n_nodes and nx.is_connected(graph):
        if n_nodes <= max_nodes_slow:
            statistics["average_path_length"] = nx.average_shortest_path_length(graph)
            statistics["diameter"] = nx.diameter(graph)
        else:
            print(f"Skipping path length and diameter: {n_nodes} nodes exceeds the "
                  f"--max-nodes-slow-metrics limit of {max_nodes_slow}.")
    elif n_nodes:
        largest = max(nx.connected_components(graph), key=len)
        statistics["largest_component_nodes"] = len(largest)
        statistics["largest_component_fraction"] = len(largest) / n_nodes

    _print_statistics(statistics)
    return statistics


def analyze_directed_network(graph, output_dir, image_format="png",
                             max_nodes_slow=DEFAULT_MAX_NODES_SLOW_METRICS):
    """
    Run the directed analysis and return its summary statistics.

    :return: dict of statistics, also written to ``network_statistics.csv``.
    """
    print("\n" + "=" * 60)
    print("ANALYZING DIRECTED NETWORK")
    print("=" * 60)

    network_type = "directed"
    fit = plot_degree_distribution_loglog(graph, output_dir, network_type, image_format)
    plot_degree_histogram(graph, output_dir, network_type, "in", image_format)
    plot_degree_histogram(graph, output_dir, network_type, "out", image_format)
    plot_cumulative_degree_distribution(graph, output_dir, network_type, "in",
                                        image_format)
    plot_cumulative_degree_distribution(graph, output_dir, network_type, "out",
                                        image_format)
    plot_pagerank_distribution(graph, output_dir, network_type, image_format)
    plot_clustering_vs_degree_threshold(graph, output_dir, network_type, image_format)
    plot_clustering_vs_degree_scatter(graph, output_dir, network_type, image_format)
    n_communities = plot_community_size_distribution(graph, output_dir, network_type,
                                                     image_format, max_nodes_slow)

    n_nodes = graph.number_of_nodes()
    statistics = {
        "network_type": "directed",
        "nodes": n_nodes,
        "edges": graph.number_of_edges(),
        "average_in_degree": (sum(d for _, d in graph.in_degree()) / n_nodes) if n_nodes else 0.0,
        "average_out_degree": (sum(d for _, d in graph.out_degree()) / n_nodes) if n_nodes else 0.0,
        "n_communities": n_communities,
    }

    if n_nodes:
        statistics["is_weakly_connected"] = nx.is_weakly_connected(graph)
        if not statistics["is_weakly_connected"]:
            largest = max(nx.weakly_connected_components(graph), key=len)
            statistics["largest_weak_component_nodes"] = len(largest)

        statistics["is_strongly_connected"] = nx.is_strongly_connected(graph)
        if statistics["is_strongly_connected"]:
            if n_nodes <= max_nodes_slow:
                statistics["average_path_length"] = nx.average_shortest_path_length(graph)
            else:
                print(f"Skipping path length: {n_nodes} nodes exceeds the "
                      f"--max-nodes-slow-metrics limit of {max_nodes_slow}.")
        else:
            largest = max(nx.strongly_connected_components(graph), key=len)
            statistics["largest_strong_component_nodes"] = len(largest)

    if fit:
        statistics.update({f"powerlaw_{k}": v for k, v in fit.items()})

    _print_statistics(statistics)
    return statistics


def _print_statistics(statistics):
    """Print the summary statistics as an aligned block."""
    print("\n" + "-" * 60)
    print("NETWORK STATISTICS")
    print("-" * 60)
    for key, value in statistics.items():
        if value is None:
            continue
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"{key.replace('_', ' ').capitalize():<38} {formatted}")


def write_statistics_csv(statistics, output_dir):
    """Write the summary statistics to ``network_statistics.csv``."""
    path = os.path.join(output_dir, "network_statistics.csv")
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        for key, value in statistics.items():
            if value is not None:
                writer.writerow([key, value])
    print("Saved: network_statistics.csv")


# ===========================================================================
# 8. Pipeline
# ===========================================================================
def analyze_network(input_file, output_dir, image_format="png",
                    max_nodes_slow=DEFAULT_MAX_NODES_SLOW_METRICS):
    """
    Analyze one network file end to end.

    The analysis performed is chosen from the file itself: a directed graph
    gets the in/out-degree and PageRank treatment, an undirected one the
    clustering and community treatment.

    :return: the summary statistics dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    graph = load_network(input_file)

    if graph.is_directed():
        statistics = analyze_directed_network(graph, output_dir, image_format,
                                              max_nodes_slow)
    else:
        statistics = analyze_undirected_network(graph, output_dir, image_format,
                                                max_nodes_slow)

    write_statistics_csv(statistics, output_dir)

    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE! All output saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)
    return statistics


# Backwards-compatible alias for code that called the previous entry point.
main = analyze_network


# ===========================================================================
# 9. Command line interface
# ===========================================================================
def parse_args(argv=None):
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        prog="network_analyzer.py",
        description="Analyze a network file and generate publication-quality "
                    "figures plus a summary statistics table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file",
                        help="Input network file (.graphml, .gml or .gexf).")
    parser.add_argument("output_dir",
                        help="Directory to save the figures and statistics.")
    parser.add_argument("--format", dest="image_format", default="png",
                        choices=["png", "pdf", "svg"],
                        help="Image format for the figures.")
    parser.add_argument("--max-nodes-slow-metrics", type=int,
                        default=DEFAULT_MAX_NODES_SLOW_METRICS,
                        help="Skip average path length, diameter and community "
                             "detection above this node count. These scale worse "
                             "than linearly and can run for hours on a large "
                             "network.")
    return parser.parse_args(argv)


def main_cli(argv=None):
    """
    Entry point.

    :return: 0 on success, 1 when the input file cannot be used.
    """
    args = parse_args(argv)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1

    try:
        analyze_network(args.input_file, args.output_dir,
                        image_format=args.image_format,
                        max_nodes_slow=args.max_nodes_slow_metrics)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
