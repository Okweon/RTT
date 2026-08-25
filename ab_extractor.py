#!/usr/bin/env python3
"""
ab_extractor.py
===============

PubMed abstract extractor and co-authorship network builder.

Given a PubMed search query and a publication-year filter, this script

1. searches PubMed through the NCBI E-utilities (Entrez) API,
2. downloads the matching records in safe batches,
3. writes one CSV row per article (bibliographic metadata + abstract),
4. builds directed / undirected co-authorship networks (GML + interactive HTML),
5. produces descriptive publication statistics (CSV + interactive HTML plots).

Typical use
-----------
    python ab_extractor.py \
        --query "nontuberculous[title] NOT review[pt]" \
        --year "after 1800" \
        --output-dir ./output/NTM \
        --email you@example.com

NCBI requires a contact e-mail address for every E-utilities request.  Provide
it with ``--email`` or by exporting ``NCBI_EMAIL``.  An optional API key
(``--api-key`` / ``NCBI_API_KEY``) raises the request rate limit from 3 to 10
requests per second.

Output layout
-------------
    <output-dir>/
        pubmed_abstracts.csv                 # one row per article (all records)
        pubmed_abstracts_filtered.csv        # articles that actually have an abstract
        summary_pubmedSearch.csv             # record counts of this run
        author_information/
            author_affiliations.csv
            author_cooccur_directed_nt.gml
            author_cooccur_undirected_nt.gml
            interactive_directed_network.html
            interactive_undirected_network.html
        publication_information/
            articles_per_journal.csv / .html
            yearly_publications.csv / .html
            articles_per_year_per_journal.csv / .html
            stacked_articles_per_year_per_journal.html
            percentage_stacked_articles_per_year_per_journal.html
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
from Bio import Entrez
from pyvis.network import Network

LOGGER = logging.getLogger("ab_extractor")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Column order of the main abstract CSV file.
CSV_FIELDNAMES: List[str] = [
    "Index", "Publication_Year", "Abstract_ID", "Journal_Name", "Title",
    "Authors", "Affiliations", "Keywords", "Abstract", "DOI", "URL",
    "Corresponding_Author", "Publication_Type",
]

#: NCBI refuses an esearch ``retmax`` above 10,000, so results are paged.
ESEARCH_PAGE_SIZE = 9_999

#: Number of PMIDs fetched per efetch call.  200-500 is the safe range;
#: sending thousands of IDs in one request makes NCBI drop the connection.
DEFAULT_BATCH_SIZE = 200

#: Number of nodes kept in the interactive network visualisations.
DEFAULT_TOP_NODES = 300

#: Retry policy applied to every Entrez request (NCBI throttles frequently).
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3.0

#: Placeholder written when a field is missing from the PubMed record.
NOT_AVAILABLE = "Not available"

#: Seconds to wait between consecutive Entrez requests.  Set by
#: :func:`configure_entrez` according to whether an API key is used.
_REQUEST_INTERVAL = 0.34


# --------------------------------------------------------------------------- #
# Entrez helpers
# --------------------------------------------------------------------------- #

def configure_entrez(email: str, api_key: Optional[str] = None) -> None:
    """
    Register the caller with NCBI E-utilities.

    Args:
        email: Contact address.  NCBI requires this for every request and may
            block anonymous traffic.
        api_key: Optional NCBI API key.  With a key the allowed request rate
            goes from 3/s to 10/s, so the polite delay is shortened as well.
    """
    global _REQUEST_INTERVAL

    Entrez.email = email
    Entrez.tool = "ab_extractor"
    if api_key:
        Entrez.api_key = api_key
        _REQUEST_INTERVAL = 0.11  # 10 requests/second
    else:
        _REQUEST_INTERVAL = 0.34  # 3 requests/second


def _entrez_request(operation, description: str, **kwargs) -> Any:
    """
    Run one Entrez call with retries and return the parsed response.

    Args:
        operation: ``Entrez.esearch`` or ``Entrez.efetch``.
        description: Short label used in log messages.
        **kwargs: Arguments forwarded to the Entrez function.

    Returns:
        The dict-like structure produced by ``Entrez.read``.

    Raises:
        RuntimeError: If every attempt fails.
    """
    last_error: Optional[BaseException] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            handle = operation(**kwargs)
            try:
                return Entrez.read(handle)
            finally:
                handle.close()
        except Exception as exc:  # network errors, HTTP 429/500, XML errors ...
            last_error = exc
            LOGGER.warning(
                "%s failed (attempt %d/%d): %s", description, attempt, MAX_RETRIES, exc
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"{description} failed after {MAX_RETRIES} attempts") from last_error


def format_year_query(year: Optional[str]) -> Optional[str]:
    """
    Translate a human-friendly year expression into a PubMed [PDAT] clause.

    Accepted forms:
        ``"2014"``          -> a single year
        ``"2001-2020"``     -> an inclusive range
        ``"before 2000"``   -> everything up to that year
        ``"after 2023"``    -> everything from that year on
        ``""`` / ``None``   -> no date restriction

    Args:
        year: The year expression.

    Returns:
        The PubMed date clause, or ``None`` when no restriction is wanted.
    """
    if not year or not year.strip():
        return None

    year = year.strip()

    if "-" in year:                                  # range, e.g. "2001-2020"
        start_year, end_year = (part.strip() for part in year.split("-", 1))
        return f"({start_year}[PDAT] : {end_year}[PDAT])"

    if year.lower().startswith("before "):           # e.g. "before 2000"
        return f"(1000[PDAT] : {year.split(' ', 1)[1].strip()}[PDAT])"

    if year.lower().startswith("after "):            # e.g. "after 2023"
        # 3000 is an arbitrary upper bound; PubMed has no later records.
        return f"({year.split(' ', 1)[1].strip()}[PDAT] : 3000[PDAT])"

    return f"{year}[PDAT]"                           # e.g. "2014"


def search_pubmed(query: str, year: Optional[str], max_results: int = 10_000) -> List[str]:
    """
    Search PubMed and return the matching PMIDs.

    Results are requested page by page because NCBI caps ``retmax`` at 10,000.

    Args:
        query: A PubMed query string (field tags and boolean operators allowed).
        year: Year expression understood by :func:`format_year_query`.
        max_results: Upper bound on the number of PMIDs collected.

    Returns:
        PMIDs ordered from oldest to newest publication date.
    """
    year_query = format_year_query(year)
    search_query = f"({query}) AND {year_query}" if year_query else query
    LOGGER.info("PubMed query: %s", search_query)

    pmids: List[str] = []
    retstart = 0
    total: Optional[int] = None

    while len(pmids) < max_results:
        retmax = min(ESEARCH_PAGE_SIZE, max_results - len(pmids))
        results = _entrez_request(
            Entrez.esearch,
            f"esearch [{retstart + 1}-{retstart + retmax}]",
            db="pubmed",
            term=search_query,
            sort="pub+date",
            retmode="xml",
            retstart=retstart,
            retmax=retmax,
        )

        if total is None:
            total = int(results.get("Count", 0))
            LOGGER.info("PubMed reports %d matching records.", total)

        batch = list(results.get("IdList", []))
        if not batch:
            break

        pmids.extend(batch)
        retstart += len(batch)
        if retstart >= total:
            break
        time.sleep(_REQUEST_INTERVAL)

    LOGGER.info("Collected %d PMIDs.", len(pmids))
    # esearch sorts by publication date descending; reverse to go oldest -> newest.
    return pmids[::-1]


def fetch_details(id_list: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[dict]:
    """
    Download the full PubMed records for a list of PMIDs.

    Args:
        id_list: PMIDs to fetch.
        batch_size: PMIDs per efetch request.

    Returns:
        A flat list of ``PubmedArticle`` records.  Book chapters
        (``PubmedBookArticle``) are counted in the log but not returned,
        because they do not carry the fields this script extracts.
    """
    articles: List[dict] = []
    total = len(id_list)

    for start in range(0, total, batch_size):
        chunk = list(id_list[start:start + batch_size])
        records = _entrez_request(
            Entrez.efetch,
            f"efetch [{start + 1}-{start + len(chunk)}]",
            db="pubmed",
            retmode="xml",
            id=",".join(chunk),
        )

        articles.extend(records.get("PubmedArticle", []))
        skipped_books = len(records.get("PubmedBookArticle", []))
        if skipped_books:
            LOGGER.info("Skipped %d book records in this batch.", skipped_books)

        LOGGER.info("Fetched %d/%d records.", min(start + batch_size, total), total)
        time.sleep(_REQUEST_INTERVAL)

    return articles


# --------------------------------------------------------------------------- #
# Record parsing
# --------------------------------------------------------------------------- #

def _extract_abstract(article_body: dict) -> str:
    """
    Join every section of an abstract into a single string.

    Structured abstracts are split into several ``AbstractText`` elements
    (BACKGROUND / METHODS / RESULTS / ...).  All of them are concatenated, each
    prefixed with its label when one exists, so no text is lost.

    Args:
        article_body: The ``MedlineCitation.Article`` sub-dictionary.

    Returns:
        The abstract text, or ``"Not available"`` when the record has none.
    """
    abstract = article_body.get("Abstract", {})
    sections = abstract.get("AbstractText", [])
    if not sections:
        return NOT_AVAILABLE

    parts: List[str] = []
    for section in sections:
        text = str(section).strip()
        if not text:
            continue
        label = getattr(section, "attributes", {}).get("Label")
        parts.append(f"{label}: {text}" if label else text)

    return " ".join(parts) if parts else NOT_AVAILABLE


def _extract_pub_year(article_body: dict) -> str:
    """
    Return the publication year as a string.

    Older records store a free-text ``MedlineDate`` (e.g. ``"1998 Nov-Dec"``)
    instead of a ``Year`` element; the leading four digits are used in that case.

    Args:
        article_body: The ``MedlineCitation.Article`` sub-dictionary.

    Returns:
        A four-digit year, or ``"Unknown"`` if none can be determined.
    """
    pub_date = article_body.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})

    year = pub_date.get("Year")
    if year:
        return str(year)

    medline_date = str(pub_date.get("MedlineDate", ""))
    if len(medline_date) >= 4 and medline_date[:4].isdigit():
        return medline_date[:4]

    return "Unknown"


def _extract_authors(authors_list: Iterable[dict]) -> Tuple[List[str], List[str], Dict[str, set]]:
    """
    Collect author names and affiliations from an ``AuthorList``.

    Group authors (``CollectiveName``) and authors without a full name are
    skipped, because they cannot be matched reliably across articles.

    Args:
        authors_list: The ``AuthorList`` of one article.

    Returns:
        A tuple ``(author_names, affiliations, affiliations_by_author)``.
    """
    author_names: List[str] = []
    affiliations: List[str] = []
    affiliations_by_author: Dict[str, set] = {}

    for author in authors_list:
        author_affiliations = [
            aff["Affiliation"]
            for aff in author.get("AffiliationInfo", [])
            if "Affiliation" in aff
        ]
        affiliations.extend(author_affiliations)

        if "LastName" not in author or "ForeName" not in author:
            continue

        name = f"{author['ForeName']} {author['LastName']}"
        author_names.append(name)
        affiliations_by_author.setdefault(name, set()).add(
            author_affiliations[0] if author_affiliations else "No affiliation available"
        )

    return author_names, affiliations, affiliations_by_author


def _find_corresponding_author(
    authors_list: Sequence[dict], author_names: Sequence[str]
) -> Optional[str]:
    """
    Guess the corresponding author of an article.

    PubMed has no dedicated field for this, so two heuristics are applied in
    order: (1) an affiliation string that mentions "corresponding", and
    (2) the last named author, which is the usual convention.

    Args:
        authors_list: The raw ``AuthorList``.
        author_names: Names already extracted by :func:`_extract_authors`.

    Returns:
        The author name, or ``None`` when the article lists no usable author.
    """
    for author in authors_list:
        if "LastName" not in author or "ForeName" not in author:
            continue
        if any(
            "corresponding" in aff.get("Affiliation", "").lower()
            for aff in author.get("AffiliationInfo", [])
        ):
            return f"{author['ForeName']} {author['LastName']}"

    return author_names[-1] if author_names else None


def _parse_article(
    article: dict, index: int
) -> Tuple[Dict[str, Any], List[str], Optional[str], Dict[str, set]]:
    """
    Turn one raw PubMed record into a flat CSV row.

    Args:
        article: A single ``PubmedArticle`` record.
        index: 1-based position of the article in the result set.

    Returns:
        A tuple ``(record, author_names, corresponding_author, affiliations_by_author)``.
    """
    citation = article["MedlineCitation"]
    body = citation["Article"]

    authors_list = body.get("AuthorList", [])
    author_names, affiliations, affiliations_by_author = _extract_authors(authors_list)
    corresponding_author = _find_corresponding_author(authors_list, author_names)

    keyword_list = citation.get("KeywordList", [])
    keywords = ", ".join(str(k) for k in keyword_list[0]) if keyword_list else NOT_AVAILABLE

    publication_types = body.get("PublicationTypeList", [])
    publication_type = (
        ", ".join(str(p) for p in publication_types) if publication_types else NOT_AVAILABLE
    )

    article_ids = article.get("PubmedData", {}).get("ArticleIdList", [])
    doi = next(
        (str(i) for i in article_ids if i.attributes["IdType"] == "doi"), "No DOI available"
    )
    pmid = next(
        (str(i) for i in article_ids if i.attributes["IdType"] == "pubmed"), "No PMID available"
    )

    pub_year = _extract_pub_year(body)

    record = {
        "Index": index,
        "Publication_Year": pub_year,
        "Abstract_ID": f"{index}_{pub_year}",
        "Journal_Name": body.get("Journal", {}).get("Title", NOT_AVAILABLE),
        "Title": str(body.get("ArticleTitle", NOT_AVAILABLE)),
        "Authors": ", ".join(author_names),
        "Affiliations": "; ".join(sorted(set(affiliations))),
        "Keywords": keywords,
        "Abstract": _extract_abstract(body),
        "DOI": doi,
        "URL": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "Corresponding_Author": corresponding_author,
        "Publication_Type": publication_type,
    }

    return record, author_names, corresponding_author, affiliations_by_author


# --------------------------------------------------------------------------- #
# Main extraction routine
# --------------------------------------------------------------------------- #

def save_abstracts_and_create_network(
    articles: Sequence[dict],
    output_dir: str,
    csv_filename: str = "pubmed_abstracts.csv",
    top_nodes: int = DEFAULT_TOP_NODES,
    build_networks: bool = True,
    build_plots: bool = True,
) -> pd.DataFrame:
    """
    Write the abstract CSV and derive author networks and publication plots.

    Args:
        articles: ``PubmedArticle`` records returned by :func:`fetch_details`.
        output_dir: Directory that receives every output file.
        csv_filename: Name of the main abstract CSV.
        top_nodes: Number of nodes kept in the interactive network HTML files.
        build_networks: Set to ``False`` to skip the interactive network HTML.
        build_plots: Set to ``False`` to skip the plotly figures.

    Returns:
        A DataFrame holding the same rows as the CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    graph_directed = nx.DiGraph()      # corresponding author -> co-authors
    graph_undirected = nx.Graph()      # author co-occurrence within an article

    author_affiliations: Dict[str, set] = {}
    author_counter: Counter = Counter()
    records: List[Dict[str, Any]] = []
    failures = 0

    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for index, article in enumerate(articles, 1):
            try:
                record, author_names, corresponding, affiliations = _parse_article(article, index)
            except Exception as exc:
                # One malformed record must not abort the whole run.
                failures += 1
                LOGGER.warning("Skipping article #%d: %s", index, exc)
                continue

            writer.writerow(record)
            records.append(record)

            # --- author bookkeeping ---------------------------------------- #
            for name in author_names:
                author_counter[name] += 1
            for name, affs in affiliations.items():
                author_affiliations.setdefault(name, set()).update(affs)

            # --- directed network: corresponding author -> each co-author --- #
            if corresponding:
                for name in author_names:
                    if name != corresponding:
                        graph_directed.add_edge(corresponding, name)

            # --- undirected network: every author pair of this article ------ #
            for i in range(len(author_names)):
                for j in range(i + 1, len(author_names)):
                    graph_undirected.add_edge(author_names[i], author_names[j])

    LOGGER.info("Wrote %d rows to %s (%d records skipped).", len(records), csv_path, failures)

    # ----------------------------------------------------------------- #
    # Author information
    # ----------------------------------------------------------------- #
    author_dir = os.path.join(output_dir, "author_information")
    os.makedirs(author_dir, exist_ok=True)

    nx.write_gml(graph_directed, os.path.join(author_dir, "author_cooccur_directed_nt.gml"))
    nx.write_gml(graph_undirected, os.path.join(author_dir, "author_cooccur_undirected_nt.gml"))

    affiliation_csv = os.path.join(author_dir, "author_affiliations.csv")
    with open(affiliation_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["Index", "Author", "Frequency", "Affiliation"])
        writer.writeheader()
        for index, (author, affs) in enumerate(author_affiliations.items(), 1):
            writer.writerow({
                "Index": index,
                "Author": author,
                "Frequency": author_counter[author],
                "Affiliation": "; ".join(sorted(affs)),
            })

    df = pd.DataFrame(records, columns=CSV_FIELDNAMES)

    # ----------------------------------------------------------------- #
    # Derived analyses
    # ----------------------------------------------------------------- #
    if build_plots and not df.empty:
        publication_dir = os.path.join(output_dir, "publication_information")
        os.makedirs(publication_dir, exist_ok=True)
        generate_analysis_and_plots(df, publication_dir)

    if build_networks:
        generate_interactive_networks(
            graph_directed, graph_undirected, author_counter, author_dir, top_nodes
        )

    return df


# --------------------------------------------------------------------------- #
# Descriptive statistics and plots
# --------------------------------------------------------------------------- #

def generate_analysis_and_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Write publication statistics as CSV tables and interactive plotly figures.

    Rows whose publication year is not numeric (``"Unknown"``) are excluded from
    the year-based figures but still counted in the per-journal totals.

    Args:
        df: DataFrame produced by :func:`save_abstracts_and_create_network`.
        output_dir: Destination directory (created by the caller).
    """
    # --- articles per journal ------------------------------------------ #
    articles_per_journal = df["Journal_Name"].value_counts().reset_index()
    articles_per_journal.columns = ["Journal_Name", "Number_of_Articles"]
    articles_per_journal.to_csv(
        os.path.join(output_dir, "articles_per_journal.csv"), index=False
    )

    fig = px.bar(
        articles_per_journal, x="Journal_Name", y="Number_of_Articles",
        title="Number of Articles per Journal",
        labels={"Journal_Name": "Journal", "Number_of_Articles": "Number of Articles"},
    )
    fig.write_html(os.path.join(output_dir, "articles_per_journal.html"))

    # Keep only records with a usable numeric year for the time series below.
    dated = df.copy()
    dated["Publication_Year"] = pd.to_numeric(dated["Publication_Year"], errors="coerce")
    dropped = int(dated["Publication_Year"].isna().sum())
    if dropped:
        LOGGER.info(
            "%d records have no usable year and are left out of the yearly plots.", dropped
        )
    dated = dated.dropna(subset=["Publication_Year"])
    if dated.empty:
        return
    dated["Publication_Year"] = dated["Publication_Year"].astype(int)

    # --- publications per year ------------------------------------------ #
    yearly = dated["Publication_Year"].value_counts().sort_index().reset_index()
    yearly.columns = ["Publication_Year", "Number_of_Publications"]
    yearly.to_csv(os.path.join(output_dir, "yearly_publications.csv"), index=False)

    fig = px.bar(
        yearly, x="Publication_Year", y="Number_of_Publications",
        title="Number of Yearly Publications",
        labels={"Publication_Year": "Year", "Number_of_Publications": "Number of Publications"},
    )
    fig.write_html(os.path.join(output_dir, "yearly_publications.html"))

    # --- publications per year and journal ------------------------------ #
    per_year_journal = (
        dated.groupby(["Publication_Year", "Journal_Name"])
        .size()
        .reset_index(name="Number_of_Articles")
        .sort_values(by=["Publication_Year", "Journal_Name"])
    )
    per_year_journal.to_csv(
        os.path.join(output_dir, "articles_per_year_per_journal.csv"), index=False
    )

    shared_labels = {
        "Publication_Year": "Year",
        "Number_of_Articles": "Number of Articles",
        "Journal_Name": "Journal",
    }

    fig = px.line(
        per_year_journal, x="Publication_Year", y="Number_of_Articles", color="Journal_Name",
        title="Number of Articles per Year for Each Journal", labels=shared_labels,
    )
    fig.write_html(os.path.join(output_dir, "articles_per_year_per_journal.html"))

    fig = px.bar(
        per_year_journal, x="Publication_Year", y="Number_of_Articles", color="Journal_Name",
        title="Stacked Bar Chart: Number of Articles per Year for Each Journal",
        labels=shared_labels, barmode="stack",
    )
    fig.write_html(os.path.join(output_dir, "stacked_articles_per_year_per_journal.html"))

    # Share of each journal within a year (each year adds up to 100 %).
    per_year_journal["Percentage_of_Articles"] = per_year_journal.groupby("Publication_Year")[
        "Number_of_Articles"
    ].transform(lambda x: x / x.sum() * 100)

    fig = px.bar(
        per_year_journal, x="Publication_Year", y="Percentage_of_Articles", color="Journal_Name",
        title="Percentage Stacked Bar Chart: Number of Articles per Year for Each Journal",
        labels={**shared_labels, "Percentage_of_Articles": "Percentage of Articles"},
        barmode="stack",
    )
    fig.write_html(
        os.path.join(output_dir, "percentage_stacked_articles_per_year_per_journal.html")
    )


# --------------------------------------------------------------------------- #
# Interactive networks
# --------------------------------------------------------------------------- #

def _new_pyvis_network(directed: bool = False) -> Network:
    """
    Create a PyVis canvas with the layout settings shared by both networks.

    Args:
        directed: Whether the canvas should draw directed edges.

    Returns:
        A configured, empty :class:`pyvis.network.Network`.
    """
    net = Network(
        height="750px",
        width="100%",
        bgcolor="white",
        font_color="black",
        cdn_resources="remote",
        neighborhood_highlight=True,
        select_menu=True,
        directed=directed,
    )
    net.toggle_hide_edges_on_drag(True)
    # The repulsion solver keeps large author networks readable.
    net.repulsion(node_distance=150, central_gravity=0.2, spring_length=100, spring_strength=0.05)
    return net


def _write_network_html(net: Network, path: str) -> None:
    """Render a PyVis network to a standalone, UTF-8 encoded HTML file."""
    with open(path, mode="w", encoding="utf-8") as fp:
        fp.write(net.generate_html())


def generate_interactive_networks(
    graph_directed: nx.DiGraph,
    graph_undirected: nx.Graph,
    author_counter: Counter,
    output_dir: str,
    top_nodes: int = DEFAULT_TOP_NODES,
) -> None:
    """
    Export browsable HTML versions of the two co-authorship networks.

    Only the most prominent ``top_nodes`` authors are drawn - the full author
    network of a large query has tens of thousands of nodes and no browser can
    render it usefully.  Ranking uses node degree for the directed network and
    publication count for the undirected one.

    Args:
        graph_directed: Corresponding-author -> co-author graph.
        graph_undirected: Author co-occurrence graph.
        author_counter: Number of articles per author.
        output_dir: Destination directory.
        top_nodes: Number of nodes to keep in each visualisation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- directed network: keep the highest-degree nodes ---------------- #
    ranked = sorted(graph_directed.degree, key=lambda item: item[1], reverse=True)[:top_nodes]
    subgraph_directed = graph_directed.subgraph([node for node, _ in ranked])

    net_directed = _new_pyvis_network(directed=True)
    for node, data in subgraph_directed.nodes(data=True):
        net_directed.add_node(node, **data)
    for source, target, data in subgraph_directed.edges(data=True):
        net_directed.add_edge(source, target, **data, arrows="to")
    _write_network_html(
        net_directed, os.path.join(output_dir, "interactive_directed_network.html")
    )

    # --- undirected network: keep the most published authors ------------ #
    frequent_authors = [name for name, _ in author_counter.most_common(top_nodes)]
    subgraph_undirected = graph_undirected.subgraph(
        [name for name in frequent_authors if name in graph_undirected]
    )

    net_undirected = _new_pyvis_network(directed=False)
    net_undirected.from_nx(subgraph_undirected)
    _write_network_html(
        net_undirected, os.path.join(output_dir, "interactive_undirected_network.html")
    )

    LOGGER.info("Interactive networks written to %s", output_dir)


# --------------------------------------------------------------------------- #
# Run summary
# --------------------------------------------------------------------------- #

def write_summary(output_dir: str, fetched: int, parsed: int, with_abstract: int) -> None:
    """
    Write ``summary_pubmedSearch.csv`` with the record counts of this run.

    Args:
        output_dir: Destination directory.
        fetched: Records returned by PubMed.
        parsed: Records successfully converted into CSV rows.
        with_abstract: Rows that actually contain an abstract.
    """
    path = os.path.join(output_dir, "summary_pubmedSearch.csv")
    fieldnames = [
        "No. of articles Fetched", "No. of error", "No. of articles",
        "No. of articels with no abstract", "No. of articles (filtered)",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "No. of articles Fetched": fetched,
            "No. of error": fetched - parsed,
            "No. of articles": parsed,
            "No. of articels with no abstract": parsed - with_abstract,
            "No. of articles (filtered)": with_abstract,
        })


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #

EPILOG = """\
examples:
  # every non-review article with "nontuberculous" in the title
  python ab_extractor.py --query "nontuberculous[title] NOT review[pt]" \\
      --year "after 1800" --output-dir ./output/NTM --email you@example.com

  # a long query kept in a text file, restricted to 2010-2024
  python ab_extractor.py --query-file food_supply_chain.txt \\
      --year 2010-2024 --output-dir ./output/FSC --email you@example.com

year formats:
  2014            a single year
  2001-2020       an inclusive range
  before 2000     everything up to that year
  after 2023      everything from that year on
  (omitted)       no date restriction
"""


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        prog="ab_extractor.py",
        description="Extract PubMed abstracts and build co-authorship networks.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("-q", "--query", help="PubMed query string.")
    query_group.add_argument(
        "--query-file", help="Path to a UTF-8 text file holding the query."
    )

    parser.add_argument(
        "-y", "--year", default="",
        help='Publication year filter, e.g. "2014", "2001-2020", "after 2023". '
             "Omit for no date restriction.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Directory for all output files."
    )
    parser.add_argument(
        "-e", "--email", default=os.environ.get("NCBI_EMAIL"),
        help="Contact e-mail required by NCBI (or set NCBI_EMAIL).",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("NCBI_API_KEY"),
        help="NCBI API key for a higher rate limit (or set NCBI_API_KEY).",
    )
    parser.add_argument(
        "--max-results", type=int, default=10_000,
        help="Maximum number of records to retrieve (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="PMIDs per download request (default: %(default)s).",
    )
    parser.add_argument(
        "--csv-filename", default="pubmed_abstracts.csv",
        help="Name of the main abstract CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--top-nodes", type=int, default=DEFAULT_TOP_NODES,
        help="Authors kept in the interactive networks (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-networks", action="store_true",
        help="Do not render the interactive network HTML files.",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Do not render the plotly publication figures.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity (default: %(default)s).",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the full extraction pipeline.

    Args:
        argv: Command line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` when the search returned nothing.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.email:
        parser.error("An e-mail address is required: use --email or set NCBI_EMAIL.")

    if args.query_file:
        with open(args.query_file, encoding="utf-8") as fp:
            query = fp.read().strip()
        if not query:
            parser.error(f"{args.query_file} is empty.")
    else:
        query = args.query

    configure_entrez(args.email, args.api_key)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Search ----------------------------------------------------------- #
    pmids = search_pubmed(query, args.year, max_results=args.max_results)
    if not pmids:
        LOGGER.error("No articles found with the given search criteria.")
        return 1

    # 2. Download --------------------------------------------------------- #
    articles = fetch_details(pmids, batch_size=args.batch_size)
    LOGGER.info("Number of articles fetched: %d", len(articles))

    # 3. Parse, write CSV, build networks and plots ----------------------- #
    df = save_abstracts_and_create_network(
        articles,
        args.output_dir,
        csv_filename=args.csv_filename,
        top_nodes=args.top_nodes,
        build_networks=not args.skip_networks,
        build_plots=not args.skip_plots,
    )

    # 4. Keep only records that carry an abstract ------------------------- #
    df_filtered = df[df["Abstract"] != NOT_AVAILABLE].reset_index(drop=True)
    stem = os.path.splitext(args.csv_filename)[0]
    filtered_path = os.path.join(args.output_dir, f"{stem}_filtered.csv")
    df_filtered.to_csv(filtered_path, index=False)

    # 5. Run summary ------------------------------------------------------ #
    write_summary(args.output_dir, len(articles), len(df), len(df_filtered))

    LOGGER.info(
        "Done. %d fetched / %d parsed / %d with abstract. Output: %s",
        len(articles), len(df), len(df_filtered), os.path.abspath(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
