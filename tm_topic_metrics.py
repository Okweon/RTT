#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tm_topic_metrics.py
===========================================================================
Shared topic-model evaluation metrics, figures, and visual style — used by
both ``tm_analyzer_publication.py`` (to score the model it just fit, right
after the best topic count k is chosen by coherence) and
``tm_topic_evaluation_publication.py`` (to score topic labels that already
exist in a CSV, e.g. after manual review).

This module has no dependency on either caller (it only takes plain arrays /
DataFrames), which keeps the import graph a simple one-way DAG:

    tm_topic_metrics.py  <---  tm_analyzer_publication.py
    tm_topic_metrics.py  <---  tm_topic_evaluation_publication.py  ---> tm_analyzer_publication.py

Metrics computed (model- / topic- / document-level)
---------------------------------------------------
Model-level (one row):
    Mean C_v coherence, Topic diversity@N, Mean nearest-topic cosine
    similarity, Outlier rate.
Topic-level (one row per topic):
    Top words, document count & share, C_v coherence, nearest topic,
    cosine similarity to the nearest topic, mean assignment probability,
    mean silhouette, outlier rate.
Document-level (one row per document + an aggregate summary):
    Silhouette, DBCV, assignment-probability proxy, low-confidence flag,
    outlier flag.

Two metrics are *proxies*, not a direct read of a probabilistic model's
internals (because k-means / SBERT clustering has no native notion of
"probability" or "noise point" the way LDA or HDBSCAN/BERTopic do):
  * assignment probability: softmax over each document's cosine similarity
    to every topic centroid (peakier with a lower --assignment-temperature).
  * outlier flag: per-topic z-score of a document's cosine similarity to
    its own topic's centroid, flagged if it falls below --outlier-z-threshold
    (or, if the topic column already encodes an explicit outlier/noise code,
    that code is used directly instead).

The top-level entry point most callers want is ``evaluate_topic_assignments``
(section 6): given a fitted document representation (`vec`), the topic label
already assigned to each row, and each topic's representative words, it
computes every metric above, writes the standard CSV tables + PNG figures to
``output_dir``, and returns the tables for further use (e.g. printing).
"""

# ===========================================================================
# 0. Imports
# ===========================================================================
import os
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, silhouette_samples

from gensim.models.coherencemodel import CoherenceModel
from umap import UMAP

# ===========================================================================
# 1. Shared visual style
#    One colorblind-safe categorical palette (fixed order, one hue per
#    topic, reused across every figure in the tm_* family) + a single
#    sequential blue ramp for magnitude (the similarity heatmap) + a status
#    red for thresholds (silhouette = 0, low-confidence cutoff).
# ===========================================================================
TOPIC_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
STATUS_CRITICAL = "#d03b3b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID_COLOR, "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY, "axes.grid": True,
    "grid.color": GRID_COLOR, "grid.linewidth": 0.8,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "savefig.facecolor": SURFACE, "savefig.dpi": 200,
})


def topic_color(topic_id, topics_sorted):
    """Fixed categorical color for a topic, keyed by its rank among all
    topic ids present (not by the raw id) so colors stay stable and dense
    even if some topic numbers were excluded upstream."""
    idx = topics_sorted.index(topic_id)
    return TOPIC_PALETTE[idx % len(TOPIC_PALETTE)]


# ===========================================================================
# 2. Topic-level metrics: diversity, centroids, inter-topic similarity
# ===========================================================================
def topic_diversity_at_n(top_words_by_topic, n):
    """Fraction of unique words among the union of all topics' top-N words.
    Close to 1.0 means topics use largely non-overlapping vocabulary
    (well-separated); close to 0 means topics are redundant."""
    all_words = [w for words in top_words_by_topic.values() for w in words[:n]]
    if not all_words:
        return float("nan")
    return len(set(all_words)) / len(all_words)


def compute_topic_centroids(vec, topic_labels, topics_sorted):
    """Mean document vector per topic."""
    labels_arr = np.asarray(topic_labels)
    return {t: vec[labels_arr == t].mean(axis=0) for t in topics_sorted}


def nearest_topic_similarity(centroids, topics_sorted):
    """Cosine similarity matrix between topic centroids, plus each topic's
    single nearest neighbor and that similarity (a proxy for topic
    separation: high nearest-neighbor similarity suggests two topics could
    be merged)."""
    matrix = np.vstack([centroids[t] for t in topics_sorted])
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, -np.inf)  # exclude self when finding the nearest topic
    nearest = {}
    for i, t in enumerate(topics_sorted):
        j = int(np.argmax(sim[i]))
        nearest[t] = (topics_sorted[j], float(sim[i, j]))
    np.fill_diagonal(sim, 1.0)
    return sim, nearest


# ===========================================================================
# 3. Document-level metrics: assignment probability, silhouette, DBCV,
#    outliers
# ===========================================================================
def assignment_probability_proxy(vec, topic_labels, centroids, topics_sorted, temperature):
    """Soft-assignment probability proxy: cosine similarity of each document
    to every topic centroid, softmax-normalized across topics. This is a
    reconstruction, not a genuine posterior probability — k-means / SBERT
    clustering makes a hard assignment and keeps no probability vector, so
    this proxy stands in for "how confidently does this document belong to
    its assigned topic vs. the runner-up topics"."""
    centroid_matrix = np.vstack([centroids[t] for t in topics_sorted])
    sims = cosine_similarity(vec, centroid_matrix)  # (n_docs, k)
    scaled = sims / max(temperature, 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(scaled)
    probs = exp / exp.sum(axis=1, keepdims=True)

    topic_to_col = {t: i for i, t in enumerate(topics_sorted)}
    own_prob = np.array([probs[i, topic_to_col[t]] for i, t in enumerate(topic_labels)])
    own_sim = np.array([sims[i, topic_to_col[t]] for i, t in enumerate(topic_labels)])
    return own_prob, own_sim


def silhouette_metrics(vec, topic_labels):
    """Overall + per-document silhouette score: how much closer each
    document is to its own topic's documents than to the nearest other
    topic's documents (positive = well-clustered, negative = likely
    mis-assigned)."""
    labels_arr = np.asarray(topic_labels)
    overall = silhouette_score(vec, labels_arr)
    per_doc = silhouette_samples(vec, labels_arr)
    return overall, per_doc


def dbcv_metric(vec, topic_labels):
    """Density-Based Clustering Validation index (Moulavi et al., 2014).
    Requires the optional `hdbscan` package; returns NaN with a console note
    if unavailable so the rest of the pipeline still runs."""
    try:
        from hdbscan.validity import validity_index
    except ImportError:
        print("NOTE: `hdbscan` is not installed, so DBCV will be NaN. "
              "Install it with `pip install hdbscan` to compute DBCV.")
        return float("nan")
    labels_arr = np.asarray(topic_labels).astype(np.intp)
    try:
        return float(validity_index(vec.astype(np.float64), labels_arr))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"NOTE: DBCV computation failed ({exc}); reporting NaN.")
        return float("nan")


def detect_outliers(topic_labels, own_topic_similarity, topics_sorted, z_threshold,
                     explicit_outlier_label=None):
    """Flag outliers per document.

    If `explicit_outlier_label` is given and present in `topic_labels`, those
    rows are the outliers (the topic column already encoded them, e.g.
    BERTopic's -1 "noise" topic).

    Otherwise, use a density proxy: within each topic, z-score each
    document's cosine similarity to its own topic centroid; documents whose
    z-score falls below `z_threshold` sit unusually far from their assigned
    topic's core and are flagged as outliers.
    """
    labels_arr = np.asarray(topic_labels)
    if explicit_outlier_label is not None and explicit_outlier_label in set(topic_labels):
        return labels_arr == explicit_outlier_label

    is_outlier = np.zeros(len(labels_arr), dtype=bool)
    for t in topics_sorted:
        mask = labels_arr == t
        sims = own_topic_similarity[mask]
        std = sims.std()
        if std == 0 or len(sims) < 3:
            continue
        z = (sims - sims.mean()) / std
        is_outlier[mask] = z < z_threshold
    return is_outlier


# ===========================================================================
# 4. Figures
# ===========================================================================
def plot_topic_sizes_and_coherence(topic_table, output_dir):
    """Two-panel figure: document count (+ share) per topic on top, C_v
    coherence per topic below — the top panel is the "abstracts per topic"
    bar chart, paired here with coherence for a quick at-a-glance read of
    which topics are both large and coherent."""
    topics = topic_table["topic"].tolist()
    colors = [topic_color(t, topics) for t in topics]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].bar([str(t) for t in topics], topic_table["n_documents"], color=colors)
    axes[0].set_ylabel("Documents")
    axes[0].set_title("Topic size and coherence")
    for i, pct in enumerate(topic_table["pct_documents"]):
        axes[0].text(i, topic_table["n_documents"].iloc[i], f"{pct:.1f}%",
                     ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    axes[1].bar([str(t) for t in topics], topic_table["c_v_coherence"], color=colors)
    axes[1].set_ylabel("C$_v$ coherence")
    axes[1].set_xlabel("Topic")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig_topic_sizes_and_coherence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_similarity_heatmap(sim_matrix, topics_sorted, output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    im = ax.imshow(sim_matrix, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(topics_sorted)))
    ax.set_yticks(range(len(topics_sorted)))
    ax.set_xticklabels(topics_sorted)
    ax.set_yticklabels(topics_sorted)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Topic")
    ax.set_title("Inter-topic cosine similarity")
    for i in range(len(topics_sorted)):
        for j in range(len(topics_sorted)):
            value = sim_matrix[i, j]
            text_color = "white" if value > 0.6 else INK_PRIMARY
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                     fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.85)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_topic_similarity_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_silhouette_distribution(per_doc_silhouette, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(per_doc_silhouette, bins=40, color=TOPIC_PALETTE[0], alpha=0.9)
    ax.axvline(0, color=STATUS_CRITICAL, linestyle="--", linewidth=1.5)
    negative_pct = (per_doc_silhouette < 0).mean() * 100
    ax.text(0, ax.get_ylim()[1] * 0.95, f"  {negative_pct:.1f}% negative",
            color=STATUS_CRITICAL, va="top", fontsize=10)
    ax.set_xlabel("Per-document silhouette")
    ax.set_ylabel("Documents")
    ax.set_title("Document silhouette distribution")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_silhouette_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_assignment_probability_distribution(assignment_prob, threshold, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(assignment_prob, bins=40, color=TOPIC_PALETTE[2], alpha=0.9)
    ax.axvline(threshold, color=STATUS_CRITICAL, linestyle="--", linewidth=1.5)
    low_conf_pct = (assignment_prob < threshold).mean() * 100
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f"  {low_conf_pct:.1f}% < {threshold:.2f}",
            color=STATUS_CRITICAL, va="top", fontsize=10)
    ax.set_xlabel("Assignment probability (proxy)")
    ax.set_ylabel("Documents")
    ax.set_title("Document assignment-confidence distribution")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_assignment_probability_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_embedding_scatter_umap(vec, topic_labels, is_outlier, topics_sorted,
                                 output_dir, random_state):
    """Static 2D UMAP projection of the document vectors, colored by topic
    (a PNG counterpart to the interactive UMAP.html tm_analyzer_publication.py
    produces, for embedding directly in a paper)."""
    print("Calculating UMAP projection...")
    reducer = UMAP(random_state=random_state)
    coords = reducer.fit_transform(vec)
    print("Calculating UMAP projection. Done!")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    labels_arr = np.asarray(topic_labels)
    for t in topics_sorted:
        mask = (labels_arr == t) & (~is_outlier)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.75,
                   color=topic_color(t, topics_sorted), label=f"Topic {t}", linewidths=0)
    ax.scatter(coords[is_outlier, 0], coords[is_outlier, 1], s=22, facecolors="none",
               edgecolors=INK_MUTED, marker="x", linewidths=1.0, label="Outlier (proxy)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Document embeddings by topic (UMAP projection)")
    ax.legend(fontsize=8, markerscale=1.3, frameon=False, loc="best")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig_topic_embedding_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ===========================================================================
# 5. Table assembly
# ===========================================================================
def build_metric_tables(topics_sorted, topic_labels, top_words_by_topic,
                         coherence_by_topic, mean_coherence, diversity_at_n,
                         top_n_words, nearest, mean_nearest_similarity,
                         assignment_prob, own_topic_sim, per_doc_silhouette,
                         dbcv, is_outlier, outlier_rate,
                         low_confidence_threshold, doc_id_table=None,
                         extra_model_fields=None):
    """Assemble the model-, topic-, and document-level metric DataFrames
    from already-computed per-document / per-topic arrays."""
    labels_arr = np.asarray(topic_labels)

    topic_rows = []
    for t in topics_sorted:
        mask = labels_arr == t
        nearest_t, nearest_sim = nearest[t]
        topic_rows.append({
            "topic": t,
            "top_words": ", ".join(top_words_by_topic[t]),
            "n_documents": int(mask.sum()),
            "pct_documents": round(100 * mask.sum() / len(topic_labels), 1),
            "c_v_coherence": round(coherence_by_topic[t], 4),
            "nearest_topic": nearest_t,
            "nearest_topic_cosine_similarity": round(nearest_sim, 4),
            "mean_assignment_probability": round(float(assignment_prob[mask].mean()), 4),
            "mean_silhouette": round(float(per_doc_silhouette[mask].mean()), 4),
            "outlier_rate": round(float(is_outlier[mask].mean()), 4),
        })
    topic_table = pd.DataFrame(topic_rows)

    model_row = {"n_topics": len(topics_sorted), "n_documents": len(topic_labels)}
    if extra_model_fields:
        model_row.update(extra_model_fields)
    model_row.update({
        "mean_c_v_coherence": round(mean_coherence, 4),
        f"topic_diversity_at_{top_n_words}": round(diversity_at_n, 4),
        "mean_nearest_topic_cosine_similarity": round(mean_nearest_similarity, 4),
        "outlier_rate": round(outlier_rate, 4),
    })
    model_table = pd.DataFrame([model_row])

    doc_table = doc_id_table.copy() if doc_id_table is not None \
        else pd.DataFrame(index=range(len(topic_labels)))
    doc_table["topic"] = topic_labels
    doc_table["silhouette"] = per_doc_silhouette
    doc_table["assignment_probability"] = assignment_prob
    doc_table["own_topic_cosine_similarity"] = own_topic_sim
    doc_table["is_low_confidence"] = assignment_prob < low_confidence_threshold
    doc_table["is_outlier"] = is_outlier

    doc_summary_table = pd.DataFrame([{
        "n_documents": len(topic_labels),
        "mean_silhouette": round(float(np.mean(per_doc_silhouette)), 4),
        "median_silhouette": round(float(np.median(per_doc_silhouette)), 4),
        "dbcv": round(dbcv, 4) if not math.isnan(dbcv) else np.nan,
        "pct_negative_silhouette": round(float((per_doc_silhouette < 0).mean() * 100), 2),
        "mean_assignment_probability": round(float(np.mean(assignment_prob)), 4),
        f"pct_low_confidence_below_{low_confidence_threshold:.2f}":
            round(float((assignment_prob < low_confidence_threshold).mean() * 100), 2),
        "outlier_rate_pct": round(outlier_rate * 100, 2),
    }])

    return model_table, topic_table, doc_table, doc_summary_table


# ===========================================================================
# 6. Top-level entry point
# ===========================================================================
def evaluate_topic_assignments(topics_sorted, topic_labels, vec, token_lists,
                                dictionary, top_words_by_topic, output_dir,
                                doc_id_table=None, top_n_words=10,
                                assignment_temperature=0.1,
                                low_confidence_threshold=0.50,
                                outlier_z_threshold=-1.5,
                                outlier_topic_label=None, random_state=100,
                                extra_model_fields=None, print_summary=True):
    """Compute the full model-/topic-/document-level metric suite for a set
    of already-assigned topic labels and save the standard CSV tables +
    PNG figures to ``output_dir``.

    :param topics_sorted: sorted list of distinct topic ids to score.
    :param topic_labels: topic id for each row of `vec` / `token_lists`
        (same order, one entry per document).
    :param vec: (n_docs, dim) dense document representation (e.g. a fitted
        TopicModel's LDA / SBERT / LDA+SBERT vector).
    :param token_lists: per-document tokenized text, same order as `vec`.
    :param dictionary: gensim Dictionary built from `token_lists`.
    :param top_words_by_topic: {topic_id: [word, ...]} representative words,
        already selected by the caller (e.g. via get_topic_words()).
    :param output_dir: directory the CSVs / figures are written to.
    :param doc_id_table: optional DataFrame (same row order as `vec`) of
        identifier columns copied into document_level_metrics.csv.
    :param extra_model_fields: optional dict merged into the model-level row
        (e.g. {"representation_method": "LDA_BERT", "gamma": 14.0}).
    :return: dict with keys "model_table", "topic_table", "doc_table",
        "doc_summary_table", "sim_matrix", "topics_sorted".
    """
    os.makedirs(output_dir, exist_ok=True)
    vec = np.asarray(vec)

    # -- per-topic coherence + vocabulary diversity --------------------------
    top_words_lists = [top_words_by_topic[t] for t in topics_sorted]
    cm = CoherenceModel(topics=top_words_lists, texts=token_lists,
                         dictionary=dictionary, coherence="c_v")
    per_topic_scores = cm.get_coherence_per_topic()
    coherence_by_topic = dict(zip(topics_sorted, per_topic_scores))
    mean_coherence = float(np.mean(per_topic_scores))
    diversity_at_n = topic_diversity_at_n(top_words_by_topic, top_n_words)

    # -- centroids, inter-topic separation ------------------------------------
    centroids = compute_topic_centroids(vec, topic_labels, topics_sorted)
    sim_matrix, nearest = nearest_topic_similarity(centroids, topics_sorted)
    mean_nearest_similarity = float(np.mean([sim for _, sim in nearest.values()]))

    # -- document-level: assignment probability, silhouette, DBCV, outliers --
    assignment_prob, own_topic_sim = assignment_probability_proxy(
        vec, topic_labels, centroids, topics_sorted, assignment_temperature)
    overall_silhouette, per_doc_silhouette = silhouette_metrics(vec, topic_labels)
    dbcv = dbcv_metric(vec, topic_labels)
    is_outlier = detect_outliers(topic_labels, own_topic_sim, topics_sorted,
                                  outlier_z_threshold, outlier_topic_label)
    outlier_rate = float(is_outlier.mean())

    # -- assemble + save tables ------------------------------------------------
    model_table, topic_table, doc_table, doc_summary_table = build_metric_tables(
        topics_sorted, topic_labels, top_words_by_topic, coherence_by_topic,
        mean_coherence, diversity_at_n, top_n_words, nearest,
        mean_nearest_similarity, assignment_prob, own_topic_sim,
        per_doc_silhouette, dbcv, is_outlier, outlier_rate,
        low_confidence_threshold, doc_id_table, extra_model_fields,
    )

    model_table.to_csv(os.path.join(output_dir, "model_level_metrics.csv"), index=False)
    topic_table.to_csv(os.path.join(output_dir, "topic_level_metrics.csv"), index=False)
    doc_table.to_csv(os.path.join(output_dir, "document_level_metrics.csv"), index=False)
    doc_summary_table.to_csv(os.path.join(output_dir, "document_level_summary.csv"), index=False)
    pd.DataFrame(sim_matrix, index=topics_sorted, columns=topics_sorted).to_csv(
        os.path.join(output_dir, "topic_similarity_matrix.csv"))
    print(f"\nSaved evaluation metric tables to {os.path.abspath(output_dir)}")

    # -- figures -----------------------------------------------------------------
    plot_topic_sizes_and_coherence(topic_table, output_dir)
    plot_similarity_heatmap(sim_matrix, topics_sorted, output_dir)
    plot_silhouette_distribution(per_doc_silhouette, output_dir)
    plot_assignment_probability_distribution(assignment_prob, low_confidence_threshold, output_dir)
    plot_embedding_scatter_umap(vec, topic_labels, is_outlier, topics_sorted,
                                 output_dir, random_state)

    if print_summary:
        print("\n=== Model-level summary ===")
        print(model_table.to_string(index=False))
        print("\n=== Document-level summary ===")
        print(doc_summary_table.to_string(index=False))
        print("\n=== Topic-level summary ===")
        print(topic_table.drop(columns=["top_words"]).to_string(index=False))

    return {
        "model_table": model_table, "topic_table": topic_table,
        "doc_table": doc_table, "doc_summary_table": doc_summary_table,
        "sim_matrix": sim_matrix, "topics_sorted": topics_sorted,
    }
