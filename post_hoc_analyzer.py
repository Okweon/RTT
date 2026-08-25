#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_hoc_analyzer.py
===========================================================================
Post-hoc analysis of a topic-labelled bibliographic table.

This script runs at the end of the pipeline, once every document has been
assigned a topic. It takes the table produced by ``tm_analyzer.py``
(``data_topic.csv`` / ``.xlsx``) and derives two families of output:

1. **Author information** - weighted co-authorship networks (directed and
   undirected) exported in GML / GraphML / GEXF, plus a table of every
   author with their publication count, affiliation and topic coverage.
2. **Publication information** - article counts by journal, by year, and by
   journal-and-year, as CSV tables and interactive charts.

Authors are stored in the networks as integer IDs, with the author name
attached as a node attribute. Names are noisy (accents, mojibake, casing),
so the ID keeps the graph stable while the ``label`` attribute keeps it
readable in Gephi.

Usage
-----
    python post_hoc_analyzer.py \
        --input-file ./tm_output/data_topic.csv \
        --output-dir ./post_hoc_analysis

Requirements
------------
    pip install -r requirements_post_hoc_analyzer.txt
"""

# ===========================================================================
# 0. Imports
# ===========================================================================
import os
import sys
import html
import argparse
import unicodedata
from collections import Counter

import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")   # headless-safe backend; must be set before pyplot
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.io as pio


# ===========================================================================
# 1. Constants
# ===========================================================================
#: Columns the analysis actually reads. Anything else in the table is copied
#: through untouched, so extra columns are harmless.
REQUIRED_COLUMNS = [
    "Authors",              # comma/semicolon separated author list
    "Corresponding_Author",  # source of the directed network's edges
    "Affiliations",         # semicolon separated affiliation list
    "Journal_Name",         # journal statistics
    "Publication_Year",     # yearly statistics
    "topic",                # topic assigned by tm_analyzer.py
]

#: Columns whose text is repaired before use (see :func:`normalize_all_text`).
TEXT_COLUMNS = ["Authors", "Corresponding_Author", "Affiliations"]

#: Network file formats written for every graph.
NETWORK_WRITERS = {
    "gml": nx.write_gml,
    "graphml": nx.write_graphml,
    "gexf": nx.write_gexf,
}

NO_AFFILIATION = "No affiliation available"


# ===========================================================================
# 2. Text normalization
# ===========================================================================
def normalize_all_text(cell):
    """
    Clean one text cell so the same author always produces the same key.

    Three problems are fixed, in order:

    1. HTML entities left by the source database (``&amp;`` -> ``&``).
    2. Mojibake from a UTF-8 file that was once read as latin-1
       (``Ã¶`` -> ``ö``). The round-trip is attempted and silently skipped
       when the text was never mangled.
    3. Unicode variants of the same character, normalized to NFKC so that
       composed and decomposed accents compare equal.

    :param cell: any value; non-strings are converted, nulls become "".
    :return: the cleaned string.
    """
    if pd.isnull(cell):
        return ""
    if not isinstance(cell, str):
        cell = str(cell)

    cell = html.unescape(cell)
    try:
        cell = cell.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass  # text was not mojibake; leave it as it is

    return unicodedata.normalize("NFKC", cell).strip()


def split_names(value, separator=","):
    """
    Split an author or affiliation field into a list of trimmed names.

    :param value: the raw cell value.
    :param separator: the character separating entries.
    :return: list of non-empty names.
    """
    if pd.isnull(value):
        return []
    return [part.strip() for part in str(value).split(separator) if part.strip()]


# ===========================================================================
# 3. Data loading
# ===========================================================================
def load_table(input_file):
    """
    Read the topic-labelled table from CSV or Excel, based on its extension.

    :param input_file: path to ``data_topic.csv`` or ``data_topic.xlsx``.
    :return: the loaded DataFrame.
    :raises ValueError: on an unsupported extension or missing columns.
    """
    extension = os.path.splitext(input_file)[1].lower()

    if extension in (".xlsx", ".xls"):
        df = pd.read_excel(input_file)
    elif extension == ".csv":
        # tm_analyzer.py writes UTF-8; older exports may be cp1252.
        try:
            df = pd.read_csv(input_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding="cp1252")
            print("Note: UTF-8 decoding failed; loaded the CSV as cp1252.")
    else:
        raise ValueError(
            f"Unsupported input format '{extension}'. Use a .csv, .xlsx or .xls file.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in {input_file}. "
            f"Available columns: {list(df.columns)}")

    for column in TEXT_COLUMNS:
        df[column] = df[column].apply(normalize_all_text)

    # 'topic' arrives as float whenever the column contains blanks, which
    # would turn into unusable "1.0" strings further down.
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce").astype("Int64")

    print(f"Loaded {len(df)} rows from {input_file}")
    return df


# ===========================================================================
# 4. Author identity
# ===========================================================================
def build_author_ids(df, separator=","):
    """
    Assign a stable integer ID to every author name in the table.

    IDs are handed out in alphabetical order so a rerun over the same data
    produces the same graph.

    :return: ``(author_to_id, id_to_author)`` dictionaries.
    """
    all_authors = set()
    for value in df["Authors"]:
        all_authors.update(split_names(value, separator))

    author_to_id = {author: idx for idx, author in enumerate(sorted(all_authors), start=1)}
    id_to_author = {idx: author for author, idx in author_to_id.items()}

    print(f"Found {len(author_to_id)} distinct authors.")
    return author_to_id, id_to_author


# ===========================================================================
# 5. Co-authorship networks
# ===========================================================================
def build_networks(df, author_to_id, separator=","):
    """
    Build the directed and undirected co-authorship networks.

    - **Directed**: one edge from the corresponding author to each co-author,
      i.e. who a lab's lead author publishes with.
    - **Undirected**: one edge between every pair of authors on a paper,
      i.e. plain co-occurrence.

    Both are weighted: an edge's ``weight`` is the number of papers the pair
    shares.

    :return: ``(graph_directed, graph_undirected, author_counter)`` where the
        counter holds each author ID's publication count.
    """
    graph_directed = nx.DiGraph()
    graph_undirected = nx.Graph()
    author_counter = Counter()

    for _, row in df.iterrows():
        author_ids = [
            author_to_id[name]
            for name in split_names(row["Authors"], separator)
            if name in author_to_id
        ]
        corresponding_id = author_to_id.get(row["Corresponding_Author"])

        for author_id in author_ids:
            author_counter[author_id] += 1
            if corresponding_id and author_id != corresponding_id:
                _add_weighted_edge(graph_directed, corresponding_id, author_id)

        for i in range(len(author_ids)):
            for j in range(i + 1, len(author_ids)):
                _add_weighted_edge(graph_undirected, author_ids[i], author_ids[j])

    print(f"Directed network:   {graph_directed.number_of_nodes()} nodes, "
          f"{graph_directed.number_of_edges()} edges")
    print(f"Undirected network: {graph_undirected.number_of_nodes()} nodes, "
          f"{graph_undirected.number_of_edges()} edges")
    return graph_directed, graph_undirected, author_counter


def _add_weighted_edge(graph, source, target):
    """Add an edge, or increment its weight when it already exists."""
    if graph.has_edge(source, target):
        graph[source][target]["weight"] += 1
    else:
        graph.add_edge(source, target, weight=1)


def annotate_nodes(graph, id_to_author, author_counter, author_topics):
    """
    Attach readable attributes to every node of a graph.

    Without this the exported networks contain bare integers, which is
    unreadable in Gephi. ``label`` is the attribute Gephi displays by default.

    :param author_topics: ``{author_id: set of topic numbers}``.
    """
    for node in graph.nodes():
        graph.nodes[node]["label"] = id_to_author.get(node, str(node))
        graph.nodes[node]["frequency"] = int(author_counter.get(node, 0))
        graph.nodes[node]["topics"] = ", ".join(
            str(t) for t in sorted(author_topics.get(node, set())))


def save_networks(graph_directed, graph_undirected, author_dir):
    """Write both graphs in every format listed in :data:`NETWORK_WRITERS`."""
    for extension, writer in NETWORK_WRITERS.items():
        writer(graph_directed,
               os.path.join(author_dir, f"author_cooccur_directed_nt.{extension}"))
        writer(graph_undirected,
               os.path.join(author_dir, f"author_cooccur_undirected_nt.{extension}"))
    print(f"Networks saved to {author_dir} "
          f"({', '.join(NETWORK_WRITERS)} formats)")


# ===========================================================================
# 6. Author table
# ===========================================================================
def map_author_affiliations(df, separator=","):
    """
    Map each author name to one affiliation string.

    PubMed records store affiliations as a single ``;``-joined field per
    article rather than per author, so the first affiliation of the first
    article an author appears in is used. This is an approximation: for a
    paper written across several institutions, co-authors inherit the first
    listed affiliation.

    :return: ``{author name: affiliation}``.
    """
    affiliation_by_author = {}
    for _, row in df.iterrows():
        affiliations = split_names(row["Affiliations"], ";")
        affiliation = affiliations[0] if affiliations else NO_AFFILIATION
        for name in split_names(row["Authors"], separator):
            affiliation_by_author.setdefault(name, affiliation)
    return affiliation_by_author


def map_author_topics(df, author_to_id, separator=","):
    """
    Collect the set of topics each author has published in.

    Topic 0 marks a document ``tm_analyzer.py`` dropped during preprocessing,
    so it is excluded - it says nothing about the author's subject area.

    :return: ``{author_id: set of topic numbers}``.
    """
    topics_by_author = {}
    for _, row in df.iterrows():
        topic = row["topic"]
        if pd.isna(topic) or int(topic) <= 0:
            continue
        for name in split_names(row["Authors"], separator):
            author_id = author_to_id.get(name)
            if author_id is not None:
                topics_by_author.setdefault(author_id, set()).add(int(topic))
    return topics_by_author


def build_author_table(df, author_to_id, id_to_author, author_counter,
                       author_topics, author_dir, separator=","):
    """
    Write ``author_affiliations.csv``: one row per author.

    Columns:

    ==============  =========================================================
    ``ID``          Integer used as the node ID in the exported networks
    ``Author``      Author name
    ``Frequency``   Number of articles the author appears on
    ``Affiliation`` Affiliation (see :func:`map_author_affiliations`)
    ``Topic``       Every topic the author published in, comma separated
    ``Topic_color`` Single value for coloring a network: the author's topic
                    if they only published in one, otherwise
                    ``max topic + 1`` - a shared "multi-topic" category
    ==============  =========================================================

    :return: the author DataFrame.
    """
    affiliation_by_author = map_author_affiliations(df, separator)

    valid_topics = [int(t) for t in df["topic"].dropna().tolist() if int(t) > 0]
    max_topic = max(valid_topics) if valid_topics else 0
    multi_topic_color = max_topic + 1

    rows = []
    for author_id in sorted(author_counter):
        name = id_to_author[author_id]
        topics = author_topics.get(author_id, set())

        if len(topics) == 1:
            topic_color = next(iter(topics))
        elif len(topics) > 1:
            topic_color = multi_topic_color
        else:
            topic_color = 0        # no assigned topic at all

        rows.append({
            "ID": author_id,
            "Author": name,
            "Frequency": author_counter[author_id],
            "Affiliation": affiliation_by_author.get(name, NO_AFFILIATION),
            "Topic": ", ".join(str(t) for t in sorted(topics)),
            "Topic_color": topic_color,
        })

    author_df = pd.DataFrame(rows)
    out_path = os.path.join(author_dir, "author_affiliations.csv")
    author_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return author_df


# ===========================================================================
# 7. Publication statistics
# ===========================================================================
def build_publication_tables(df, publication_dir):
    """
    Write the three publication-count tables.

    Rows whose ``Publication_Year`` is not numeric are dropped from the
    year-based tables; they remain in the per-journal totals.

    :return: ``(articles_per_journal, yearly_publications, per_year_journal)``
    """
    # -- articles per journal --------------------------------------------
    articles_per_journal = df["Journal_Name"].value_counts().reset_index()
    articles_per_journal.columns = ["Journal_Name", "Number_of_Articles"]
    articles_per_journal.to_csv(
        os.path.join(publication_dir, "articles_per_journal.csv"), index=False)

    # -- restrict to rows with a usable year -------------------------------
    dated = df.copy()
    dated["Publication_Year"] = pd.to_numeric(dated["Publication_Year"], errors="coerce")
    dropped = int(dated["Publication_Year"].isna().sum())
    if dropped:
        print(f"{dropped} rows have no usable publication year "
              f"and are left out of the yearly tables.")
    dated = dated.dropna(subset=["Publication_Year"])
    dated["Publication_Year"] = dated["Publication_Year"].astype(int)

    # -- publications per year ----------------------------------------------
    yearly_publications = (
        dated["Publication_Year"].value_counts().sort_index().reset_index()
    )
    yearly_publications.columns = ["Publication_Year", "Number_of_Publications"]
    yearly_publications.to_csv(
        os.path.join(publication_dir, "yearly_publications.csv"), index=False)

    # -- publications per year and journal ------------------------------------
    per_year_journal = (
        dated.groupby(["Publication_Year", "Journal_Name"])
        .size()
        .reset_index(name="Number_of_Articles")
        .sort_values(by=["Publication_Year", "Journal_Name"])
    )
    per_year_journal.to_csv(
        os.path.join(publication_dir, "articles_per_year_per_journal.csv"), index=False)

    print(f"Publication tables saved to {publication_dir}")
    return articles_per_journal, yearly_publications, per_year_journal


# ===========================================================================
# 8. Figures
# ===========================================================================
def _write_figure(fig, figure_dir, name, write_png=False):
    """
    Save a plotly figure as HTML, and optionally as a static PNG.

    PNG export needs the ``kaleido`` package; a missing install produces a
    notice rather than aborting the run.
    """
    pio.write_html(fig, file=os.path.join(figure_dir, f"{name}.html"))
    if write_png:
        try:
            pio.write_image(fig, os.path.join(figure_dir, f"{name}.png"), scale=3)
        except Exception as exc:
            print(f"Could not write {name}.png ({exc}). Install 'kaleido' for "
                  f"static image export.")


def plot_yearly_publications_bar(yearly_publications, figure_dir):
    """
    Save the yearly publication counts as a print-ready matplotlib bar chart.

    This is the one figure meant for a manuscript, so it is a 300 DPI PNG with
    a fixed 6.5 inch width - a single journal column.
    """
    plt.figure(figsize=(6.5, 3))
    plt.bar(yearly_publications["Publication_Year"],
            yearly_publications["Number_of_Publications"],
            color="skyblue", edgecolor="black")
    plt.xlabel("Publication Year", fontsize=14)
    plt.ylabel("Number of Articles", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(figure_dir, "yearly_publications.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_interactive_figures(articles_per_journal, yearly_publications,
                             per_year_journal, figure_dir, write_png=False):
    """
    Save the four interactive publication charts.

    1. Yearly publication trend.
    2. Articles per year per journal, stacked.
    3. The same, normalized to a within-year percentage.
    4. Total articles per journal.
    """
    fig = px.bar(
        yearly_publications, x="Publication_Year", y="Number_of_Publications",
        title="Yearly Publications",
        labels={"Publication_Year": "Year",
                "Number_of_Publications": "Number of Publications"},
    )
    _write_figure(fig, figure_dir, "yearly_publications_interactive", write_png)

    fig = px.bar(
        per_year_journal, x="Publication_Year", y="Number_of_Articles",
        color="Journal_Name", title="Articles per Year per Journal",
        labels={"Number_of_Articles": "Number of Articles"},
    )
    _write_figure(fig, figure_dir, "stacked_articles_per_year_per_journal", write_png)

    # Share of each journal within a year (each year adds up to 100 %).
    per_year_journal = per_year_journal.copy()
    year_totals = per_year_journal.groupby("Publication_Year")["Number_of_Articles"] \
                                  .transform("sum")
    per_year_journal["Percentage"] = (
        per_year_journal["Number_of_Articles"] / year_totals * 100
    )
    fig = px.bar(
        per_year_journal, x="Publication_Year", y="Percentage",
        color="Journal_Name", title="Percentage of Articles per Year per Journal",
        labels={"Percentage": "Percentage (%)"},
    )
    _write_figure(fig, figure_dir, "percentage_stacked_articles_per_year_per_journal",
                  write_png)

    fig = px.bar(
        articles_per_journal, x="Journal_Name", y="Number_of_Articles",
        title="Total Articles per Journal",
        labels={"Number_of_Articles": "Total Articles"},
    )
    _write_figure(fig, figure_dir, "articles_per_journal", write_png)

    print(f"Interactive figures saved to {figure_dir}")


# ===========================================================================
# 9. Pipeline
# ===========================================================================
def post_hoc_analyzer(input_file, output_dir, author_separator=",", write_png=False):
    """
    Run the full post-hoc analysis.

    :param input_file: topic-labelled table (``data_topic.csv`` / ``.xlsx``).
    :param output_dir: directory that receives every output file.
    :param author_separator: character separating names in ``Authors``.
        PubMed exports use ``,``; several other databases use ``;``.
    :param write_png: also export the interactive charts as static PNGs
        (requires ``kaleido``).
    :return: the author DataFrame.
    """
    author_dir = os.path.join(output_dir, "author_information")
    publication_dir = os.path.join(output_dir, "publication_information")
    figure_dir = os.path.join(output_dir, "figures")
    for directory in (author_dir, publication_dir, figure_dir):
        os.makedirs(directory, exist_ok=True)

    # -- 1. Load -----------------------------------------------------------
    df = load_table(input_file)

    # -- 2. Author identity and networks ------------------------------------
    author_to_id, id_to_author = build_author_ids(df, author_separator)
    graph_directed, graph_undirected, author_counter = build_networks(
        df, author_to_id, author_separator)

    author_topics = map_author_topics(df, author_to_id, author_separator)
    for graph in (graph_directed, graph_undirected):
        annotate_nodes(graph, id_to_author, author_counter, author_topics)
    save_networks(graph_directed, graph_undirected, author_dir)

    # -- 3. Author table -----------------------------------------------------
    author_df = build_author_table(df, author_to_id, id_to_author, author_counter,
                                   author_topics, author_dir, author_separator)

    # -- 4. Publication statistics ---------------------------------------------
    articles_per_journal, yearly_publications, per_year_journal = \
        build_publication_tables(df, publication_dir)

    # -- 5. Figures --------------------------------------------------------------
    plot_yearly_publications_bar(yearly_publications, figure_dir)
    plot_interactive_figures(articles_per_journal, yearly_publications,
                             per_year_journal, figure_dir, write_png)

    print(f"\nAnalysis complete. Results saved to: {os.path.abspath(output_dir)}")
    return author_df


# ===========================================================================
# 10. Command line interface
# ===========================================================================
def parse_args(argv=None):
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        prog="post_hoc_analyzer.py",
        description="Derive co-authorship networks and publication statistics "
                    "from a topic-labelled bibliographic table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-file", required=True,
                        help="Topic-labelled table produced by tm_analyzer.py "
                             "(data_topic.csv or data_topic.xlsx).")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory for all output files.")
    parser.add_argument("--author-separator", default=",",
                        help="Character separating names in the 'Authors' column. "
                             "PubMed exports use ',', several other databases "
                             "use ';'.")
    parser.add_argument("--png", action="store_true",
                        help="Also export the interactive charts as static PNGs "
                             "(requires the 'kaleido' package).")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Entry point.

    :return: 0 on success, 1 when the input table cannot be used.
    """
    args = parse_args(argv)
    try:
        post_hoc_analyzer(args.input_file, args.output_dir,
                          author_separator=args.author_separator,
                          write_png=args.png)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
