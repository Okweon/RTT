#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tm_analyzer.py
===========================================================================
Topic modeling pipeline for bibliographic / abstract data, with a built-in
model evaluation stage.

The script takes a CSV of documents (typically the output of
``ab_extractor.py``), fits LDA / Sentence-BERT / LDA+SBERT topic models over
a range of topic counts, keeps the most coherent one, and exports every
figure and table needed to report the result.

Pipeline overview
-----------------
1. Load a CSV of documents and concatenate the selected text columns
   (Title / Keywords / Abstract, ...) into one field per row.
2. Preprocess the text at the sentence level (cleanup, optional language
   filter) and at the word level (tokenize, keep nouns, drop stop words).
3. Fit a ``TopicModel`` for every candidate topic count k, score each with
   c_v coherence, and keep the best k.
4. Visualize the winning model: a 2D UMAP projection colored by topic (with
   the silhouette score in the title), per-topic word clouds, and a
   temporal-dynamics chart of topic prevalence by publication year.
5. Evaluate the winning model with the metric suite in
   ``tm_topic_metrics.py`` - per-topic coherence, topic diversity,
   inter-topic similarity, silhouette, DBCV, assignment confidence and
   outlier proxies. Nothing is refit: the metrics reuse the vectors and
   labels produced in step 3.

Usage
-----
    python tm_analyzer.py \
        --input-csv pubmed_abstracts_filtered.csv \
        --custom-stopwords-csv custom_stopwords.csv \
        --output-dir ./tm_output \
        --text-columns Title Keywords Abstract \
        --method LDA_BERT --start-k 2 --end-k 14 --step-k 2

Requirements
------------
    pip install -r requirements_tm_analyzer.txt

Step 5 additionally needs ``tm_topic_metrics.py`` next to this file. If that
module is missing the pipeline still completes - the evaluation stage is
skipped with a notice.
"""

# ===========================================================================
# 0. Imports
# ===========================================================================
# --- standard library --------------------------------------------------------
import os
import re
import sys
import argparse
import warnings
from datetime import datetime
from collections import Counter

# --- data handling -----------------------------------------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- NLP ---------------------------------------------------------------------
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words

try:
    # Niche package used only for the optional non-English sentence filter.
    from language_detector import detect_language
    _HAS_LANGUAGE_DETECTOR = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_LANGUAGE_DETECTOR = False

import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

# --- clustering / metrics ----------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP

# --- visualization -----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")   # headless-safe backend; must be set before pyplot
import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import plotly.offline as pyo

# --- optional evaluation stage (step 5) --------------------------------------
# Lives in tm_topic_metrics.py next to this script. Imported defensively so a
# missing companion file downgrades the run instead of aborting it.
try:
    import tm_topic_metrics as tmet
    _HAS_TOPIC_METRICS = True
except ImportError:  # pragma: no cover - optional companion module
    _HAS_TOPIC_METRICS = False

warnings.filterwarnings("ignore", category=Warning)


# ===========================================================================
# 1. Visual style
#    One categorical palette shared by every figure, so a given topic number
#    always maps to the same color across the UMAP scatter plot, the temporal
#    dynamics chart and the word clouds. Colorblind-safe order (blue, orange,
#    aqua, yellow, magenta, green, violet, red); cycles beyond 8 topics.
# ===========================================================================
TOPIC_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

PLOT_BACKGROUND = "#fcfcfb"
PLOT_GRIDCOLOR = "#e1e0d9"
PLOT_FONTCOLOR = "#0b0b0b"


def topic_color(topic_number_1_indexed):
    """Return the fixed categorical color for a 1-indexed topic number."""
    return TOPIC_PALETTE[(topic_number_1_indexed - 1) % len(TOPIC_PALETTE)]


def _hex_to_hsl_hue(hex_color):
    """
    Extract the hue (0-360) of a hex color.

    Used to tint each word cloud with the same identity color the topic has
    in the other figures.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if delta == 0:
        return 0.0
    if mx == r:
        return 60 * (((g - b) / delta) % 6)
    if mx == g:
        return 60 * (((b - r) / delta) + 2)
    return 60 * (((r - g) / delta) + 4)


# ===========================================================================
# 2. Command-line configuration
# ===========================================================================
def parse_args(argv=None):
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        prog="tm_analyzer.py",
        description="Fit and visualize LDA / SBERT / LDA+SBERT topic models over "
                    "a range of topic counts, export the best model, and score it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- input / output --------------------------------------------------
    parser.add_argument("--input-csv", required=True,
                        help="CSV of documents (e.g. journal abstracts).")
    parser.add_argument("--custom-stopwords-csv", default=None,
                        help="Optional single-column CSV of extra stop words "
                             "(domain terms that are uninformative for topics).")
    parser.add_argument("--output-dir", default="./tm_output",
                        help="Directory where all figures/tables are written.")
    parser.add_argument("--encoding", default="utf-8",
                        help="Text encoding of --input-csv. If decoding fails, "
                             "cp1252 and latin-1 are tried as fallbacks.")
    parser.add_argument("--text-columns", nargs="+",
                        default=["Title", "Keywords", "Abstract"],
                        help="Columns concatenated into the text to model.")

    # -- model ------------------------------------------------------------
    parser.add_argument("--method", choices=["LDA", "BERT", "LDA_BERT"],
                        default="LDA_BERT",
                        help="Vectorization method. 'BERT' uses Sentence-BERT "
                             "embeddings; 'LDA_BERT' concatenates LDA topic "
                             "probabilities (scaled by --gamma) with them.")
    parser.add_argument("--gamma", type=float, default=14.0,
                        help="Relative weight of the LDA vector block when "
                             "--method LDA_BERT concatenates it with SBERT.")
    parser.add_argument("--start-k", type=int, default=2,
                        help="Smallest number of topics to try.")
    parser.add_argument("--end-k", type=int, default=14,
                        help="Largest number of topics to try.")
    parser.add_argument("--step-k", type=int, default=2,
                        help="Step between candidate topic counts.")
    parser.add_argument("--random-state", type=int, default=100,
                        help="Seed shared by LDA, k-means and UMAP.")
    parser.add_argument("--sbert-model", default="bert-base-nli-max-tokens",
                        help="sentence-transformers model name.")

    # -- preprocessing -----------------------------------------------------
    parser.add_argument("--disable-language-filter", action="store_true",
                        help="Skip the English/French/Spanish/Chinese language "
                             "filter (needed only if `language-detector` is "
                             "installed and you want it switched off).")
    parser.add_argument("--use-stemming", action="store_true",
                        help="Porter-stem tokens. Off by default: stems hurt "
                             "topic-word readability in figures.")
    parser.add_argument("--use-typo-correction", action="store_true",
                        help="Correct typos with SymSpell (slow; requires the "
                             "`symspellpy` package).")

    # -- figures ------------------------------------------------------------
    parser.add_argument("--wordcloud-font", default=None,
                        help="Path to a TTF font for word clouds. Defaults to "
                             "the wordcloud package's bundled font.")
    parser.add_argument("--wordcloud-top-n", type=int, default=30,
                        help="Number of top terms shown per topic word cloud.")

    # -- evaluation stage (step 5) -------------------------------------------
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Do not run the tm_topic_metrics evaluation stage.")
    parser.add_argument("--eval-top-n-words", type=int, default=10,
                        help="Words per topic used for coherence and diversity "
                             "in the evaluation stage.")
    parser.add_argument("--eval-temperature", type=float, default=0.1,
                        help="Softmax temperature of the assignment-confidence "
                             "proxy. Lower = sharper probabilities.")
    parser.add_argument("--eval-low-confidence", type=float, default=0.50,
                        help="Documents below this assignment probability are "
                             "flagged as low confidence.")
    parser.add_argument("--eval-outlier-z", type=float, default=-1.5,
                        help="Within-topic z-score below which a document is "
                             "flagged as an outlier.")

    return parser.parse_args(argv)


# ===========================================================================
# 3. Data loading
# ===========================================================================
#: Encodings tried in order when the requested one fails. Bibliographic
#: exports are UTF-8 or cp1252 in practice; latin-1 never fails, so it is the
#: last resort that guarantees the file loads.
_ENCODING_FALLBACKS = ("utf-8", "cp1252", "latin-1")


def load_data(file_path, columns, encoding="utf-8"):
    """
    Load the source CSV and build a single ``combined_text`` field.

    :param file_path: path to the CSV of documents.
    :param columns: list of column names to concatenate per row.
    :param encoding: preferred file encoding; fallbacks are tried on failure.
    :return: (dataframe, combined_text Series)
    :raises KeyError: if a requested text column is not in the CSV.
    """
    candidates = [encoding] + [e for e in _ENCODING_FALLBACKS if e != encoding]
    meta = None
    for candidate in candidates:
        try:
            meta = pd.read_csv(file_path, encoding=candidate)
            if candidate != encoding:
                print(f"Note: '{encoding}' failed; loaded the CSV as '{candidate}'.")
            break
        except UnicodeDecodeError:
            continue
    if meta is None:
        raise UnicodeDecodeError(
            "csv", b"", 0, 1,
            f"Could not decode {file_path} with any of {candidates}.")

    missing = [c for c in columns if c not in meta.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in {file_path}. "
            f"Available columns: {list(meta.columns)}")

    if "Keywords" in meta.columns:
        # 'Not available' placeholders carry no topical signal. They are blanked
        # rather than dropped so the row (and its abstract) stays in the corpus.
        meta["Keywords"] = meta["Keywords"].replace("Not available", "")

    # Missing values become empty strings, not the literal "nan"/"<NA>" text
    # that astype(str) would otherwise inject into the modeled documents.
    meta["combined_text"] = (
        meta[columns].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    )
    return meta, meta["combined_text"]


# ===========================================================================
# 4. Text preprocessing
#    Sentence-level cleanup + optional language filter, then word-level
#    tokenization, POS filtering (nouns only) and stop-word removal. Typo
#    correction and stemming are implemented but OFF by default, to keep
#    topic words readable (stemming) and preprocessing fast (typo correction
#    needs a dictionary lookup per word and rarely helps edited abstracts).
# ===========================================================================
_CUSTOM_STOPWORDS = set()            # populated by load_custom_stopwords()
_BASE_STOPWORDS = set(get_stop_words("en"))
_SYMSPELL = None                     # lazily constructed, see _get_symspell()
_STEMMER = PorterStemmer()


def load_custom_stopwords(csv_path):
    """
    Populate the module-level custom stop word set from a 1-column CSV.

    :param csv_path: path to the CSV, or None to use no custom stop words.
    """
    global _CUSTOM_STOPWORDS
    if csv_path is None:
        _CUSTOM_STOPWORDS = set()
        return
    custom_df = pd.read_csv(csv_path, header=None)
    _CUSTOM_STOPWORDS = {str(w).strip().lower() for w in custom_df[0].tolist()}
    print(f"Loaded {len(_CUSTOM_STOPWORDS)} custom stop words from {csv_path}")


def ensure_nltk_resources():
    """Download the required NLTK resources, but only if they are missing."""
    required = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for name, path in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource '{name}' ...")
            nltk.download(name, quiet=True)


def _get_symspell():
    """
    Lazily build the SymSpell dictionary.

    Only needed when typo correction is enabled, so importing this module
    does not pay the dictionary-loading cost up front.
    """
    global _SYMSPELL
    if _SYMSPELL is None:
        import pkg_resources
        from symspellpy import SymSpell
        _SYMSPELL = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        _SYMSPELL.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return _SYMSPELL


def f_base(s):
    """
    Sentence-level normalization.

    Fixes missing delimiters, lowercases, collapses repeated characters and
    phrases, and strips parenthetical asides.
    """
    s = re.sub(r"([a-z])([A-Z])", r"\1. \2", s)    # "xxxThis" -> "xxx. This"
    s = s.lower()
    s = re.sub(r"&gt|&lt", " ", s)                 # stray HTML entities
    s = re.sub(r"([a-z])\1{2,}", r"\1", s)         # letter repeated 3+ times
    s = re.sub(r"([\W+])\1{1,}", r"\1", s)         # punctuation repeated
    s = re.sub(r"\*|\W\*|\*\W", ". ", s)           # '*' used as a delimiter
    s = re.sub(r"\(.*?\)", ". ", s)                # parenthetical asides
    s = re.sub(r"\W+?\.", ".", s)                  # "xxx?." -> "xxx."
    s = re.sub(r"(\.|\?|!)(\w)", r"\1 \2", s)      # space after sentence end
    s = re.sub(r" ing ", " ", s)                   # tokenization noise
    s = re.sub(r"product received for free[.| ]", " ", s)
    s = re.sub(r"(.{2,}?)\1{1,}", r"\1", s)        # repeated phrases
    return s.strip()


def f_lan(s, allowed_languages=("English", "French", "Spanish", "Chinese")):
    """
    Return True if the sentence's detected language is in the allowed set.

    Always True when the optional ``language-detector`` package is missing,
    so an unavailable dependency keeps documents rather than dropping them.
    """
    if not _HAS_LANGUAGE_DETECTOR:
        return True
    return detect_language(s) in set(allowed_languages)


def preprocess_sent(raw_text, use_language_filter=True):
    """Sentence-level preprocessing: cleanup plus the optional language filter."""
    s = f_base(str(raw_text))
    if use_language_filter and not f_lan(s):
        return None
    return s


def f_punct(w_list):
    """Drop tokens that are not purely alphabetic (numbers, punctuation, units)."""
    return [w for w in w_list if w.isalpha()]


def f_noun(w_list):
    """Keep only nouns - topics read as concepts, and nouns carry those."""
    return [w for w, pos in nltk.pos_tag(w_list) if pos[:2] == "NN"]


def f_one_letter(w_list):
    """Drop tokens shorter than 3 characters (units and acronyms, mostly noise)."""
    return [w for w in w_list if len(w) > 2]


def f_typo(w_list):
    """Optional: correct typos via SymSpell. Words with no match are dropped."""
    from symspellpy import Verbosity
    sym_spell = _get_symspell()
    fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            fixed.append(suggestions[0].term)
    return fixed


def f_stem(w_list):
    """Optional: Porter-stem tokens (e.g. 'detection' -> 'detect')."""
    return [_STEMMER.stem(w) for w in w_list]


def f_stopw(w_list):
    """Remove generic English stop words."""
    return [w for w in w_list if w not in _BASE_STOPWORDS]


def f_stopw_custom(w_list):
    """Remove domain-specific stop words loaded via load_custom_stopwords()."""
    return [w for w in w_list if w not in _CUSTOM_STOPWORDS]


def preprocess_word(sentence, use_typo_correction=False, use_stemming=False):
    """
    Word-level preprocessing.

    Tokenize -> keep alphabetic nouns -> drop short tokens -> (optionally)
    fix typos and stem -> drop stop words.

    :return: list of tokens, or None when the sentence was empty.
    """
    if not sentence:
        return None
    tokens = word_tokenize(sentence)
    tokens = f_punct(tokens)
    tokens = f_noun(tokens)
    tokens = f_one_letter(tokens)
    if use_typo_correction:
        tokens = f_typo(tokens)
    if use_stemming:
        tokens = f_stem(tokens)
    tokens = f_stopw(tokens)
    tokens = f_stopw_custom(tokens)
    return tokens


def preprocess(docs, use_language_filter=True, use_typo_correction=False,
               use_stemming=False):
    """
    Run the full sentence + word preprocessing pipeline over all documents.

    :param docs: list of raw document strings.
    :return: ``(sentences, token_lists, idx_in, idx_out)`` where ``idx_in`` /
        ``idx_out`` are the original positions kept / dropped. Dropped
        documents had no nouns left, or failed the language filter.
    """
    print("Preprocessing raw texts ...")
    sentences, token_lists, idx_in, idx_out = [], [], [], []
    n_docs = len(docs)
    for i, doc in enumerate(docs):
        sentence = preprocess_sent(doc, use_language_filter=use_language_filter)
        tokens = preprocess_word(sentence, use_typo_correction, use_stemming)
        if tokens:
            idx_in.append(i)
            sentences.append(sentence)
            token_lists.append(tokens)
        else:
            idx_out.append(i)
        print(f"{np.round((i + 1) / n_docs * 100, 2)} %", end="\r")
    print("\nPreprocessing raw texts. Done!")
    return sentences, token_lists, idx_in, idx_out


# ===========================================================================
# 5. Topic model
# ===========================================================================
class TopicModel:
    """
    Fit one of three topic-modeling methods and expose the resulting
    per-document vectors and cluster assignments.

    - ``LDA``:      classic Latent Dirichlet Allocation (gensim). Each
                    document is represented by its topic-probability vector.
    - ``BERT``:     Sentence-BERT sentence embeddings, clustered with k-means.
    - ``LDA_BERT``: concatenation of the (scaled) LDA vector and the SBERT
                    embedding, letting k-means see both the corpus-level
                    topic signal and sentence-level semantics.

    k-means is fit for every method, so ``cluster_model.labels_`` is always
    the document-to-topic assignment used downstream.
    """

    def __init__(self, k, method, gamma=14.0, random_state=100):
        """
        :param k: number of topics / clusters.
        :param method: one of ``LDA``, ``BERT``, ``LDA_BERT``.
        :param gamma: weight of the LDA block inside an LDA_BERT vector.
        :param random_state: seed shared by LDA and k-means.
        """
        if method not in {"LDA", "BERT", "LDA_BERT"}:
            raise ValueError(f"Invalid method: {method!r}")
        self.k = k
        self.method = method
        self.gamma = gamma
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.ldamodel = None
        self.cluster_model = None
        self.vec = {}          # cache of computed vector representations by method
        self.id = f"{method}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    # -- vectorization -----------------------------------------------------
    def vectorize(self, sentences, token_lists, method=None,
                  sbert_model_name="bert-base-nli-max-tokens"):
        """
        Compute the document vector representation for ``method``.

        :return: an ``(n_docs, dim)`` array.
        """
        method = method or self.method

        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(token_lists)
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == "LDA":
            print("Getting vector representations for LDA ...")
            if self.ldamodel is None:
                self.ldamodel = gensim.models.ldamodel.LdaModel(
                    self.corpus, num_topics=self.k, id2word=self.dictionary,
                    random_state=self.random_state, passes=100,
                    alpha="auto", eta="auto", iterations=50,
                )
            n_doc = len(self.corpus)
            vec_lda = np.zeros((n_doc, self.k))
            for i in range(n_doc):
                for topic, prob in self.ldamodel.get_document_topics(self.corpus[i]):
                    vec_lda[i, topic] = prob
            print("Getting vector representations for LDA. Done!")
            return vec_lda

        if method == "BERT":
            print("Getting vector representations for BERT ...")
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer(sbert_model_name)
            vec = np.array(sbert.encode(sentences, show_progress_bar=True))
            print("Getting vector representations for BERT. Done!")
            return vec

        # method == "LDA_BERT": concatenate the two representations above,
        # scaling the LDA block by gamma so it is not drowned out by the much
        # higher-dimensional SBERT block during clustering.
        vec_lda = self.vectorize(sentences, token_lists, method="LDA")
        vec_bert = self.vectorize(sentences, token_lists, method="BERT",
                                  sbert_model_name=sbert_model_name)
        return np.c_[vec_lda * self.gamma, vec_bert]

    # -- fitting -------------------------------------------------------------
    def fit(self, sentences, token_lists, m_clustering=KMeans,
            sbert_model_name="bert-base-nli-max-tokens"):
        """
        Fit the topic model end to end: vectorize, then cluster.

        :param m_clustering: clustering class with the sklearn KMeans-style
            constructor signature ``(n_clusters, random_state=...)``.
        """
        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(token_lists)
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        print("Clustering embeddings ...")
        self.vec[self.method] = self.vectorize(
            sentences, token_lists, sbert_model_name=sbert_model_name)
        self.cluster_model = m_clustering(self.k, random_state=self.random_state)
        self.cluster_model.fit(self.vec[self.method])
        print("Clustering embeddings. Done!")

    # -- prediction ------------------------------------------------------------
    def predict(self, corpus=None):
        """
        Return the 0-indexed topic label of each document.

        :param corpus: bag-of-words corpus; defaults to the training corpus.
        """
        corpus = corpus if corpus is not None else self.corpus
        if self.method == "LDA":
            return np.array([
                max(self.ldamodel.get_document_topics(bow), key=lambda x: x[1])[0]
                for bow in corpus
            ])
        return self.cluster_model.predict(self.vec[self.method])


# Backwards-compatible alias for code written against the original script.
Topic_Model = TopicModel


# ===========================================================================
# 6. Model selection (coherence scan)
# ===========================================================================
def get_topic_words(token_lists, labels, k=None, top_n=30):
    """
    Rank each cluster's vocabulary by raw frequency and keep the top words.

    This is the "topic as top words" input gensim's CoherenceModel needs for
    the non-LDA methods (BERT / LDA_BERT), which expose no per-topic word
    distribution of their own.

    :return: list of ``k`` word lists, ordered by topic index.
    """
    k = k if k is not None else len(np.unique(labels))
    topic_text = ["" for _ in range(k)]
    for tokens, label in zip(token_lists, labels):
        topic_text[label] += " " + " ".join(tokens)
    word_counts = [Counter(text.split()).most_common() for text in topic_text]
    return [[word for word, _ in counts[:top_n]] for counts in word_counts]


def get_coherence(model, token_lists, measure="c_v"):
    """
    Compute corpus-level topic coherence for a fitted :class:`TopicModel`.

    :param measure: any coherence measure gensim supports (``c_v``, ``u_mass``...).
    """
    topics = get_topic_words(token_lists, model.cluster_model.labels_, k=model.k)
    print("Top words per topic:")
    for i, words in enumerate(topics, start=1):
        print(f"  Topic {i}: {', '.join(words[:10])}")

    if model.method == "LDA":
        cm = CoherenceModel(model=model.ldamodel, texts=token_lists,
                            corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    else:
        cm = CoherenceModel(topics=topics, texts=token_lists,
                            corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    return cm.get_coherence()


def topic_model_coherence_generator(method, sentences, token_lists, output_dir,
                                    start_k=2, end_k=10, step_k=1,
                                    gamma=14.0, random_state=100,
                                    sbert_model_name="bert-base-nli-max-tokens"):
    """
    Fit one :class:`TopicModel` per candidate k and score each with coherence.

    Writes ``coherence_scores.csv`` and returns every fitted model so the
    caller can pick the best k without refitting.

    :return: ``(models, coherence_scores)`` in candidate-k order.
    """
    models, coherence_scores = [], []
    k_values = list(range(start_k, end_k + 1, step_k))

    for k in tqdm(k_values, desc="Scanning topic counts"):
        print(f"\nTraining model with {k} topics...")
        model = TopicModel(k=k, method=method, gamma=gamma,
                           random_state=random_state)
        model.fit(sentences, token_lists, sbert_model_name=sbert_model_name)
        models.append(model)

        score = get_coherence(model, token_lists, measure="c_v")
        coherence_scores.append(score)
        print(f"Coherence score for {k} topics: {score:.4f}")

    pd.DataFrame({"Number of Topics": k_values,
                  "Coherence Score": coherence_scores}) \
        .to_csv(os.path.join(output_dir, "coherence_scores.csv"), index=False)

    best_idx = int(np.argmax(coherence_scores))
    print(f"Optimal number of topics: {k_values[best_idx]} "
          f"with coherence score {coherence_scores[best_idx]:.4f}")
    return models, coherence_scores


# ===========================================================================
# 7. Visualization
# ===========================================================================
def _style_plotly_figure(fig):
    """Apply the shared background / font styling to a plotly figure."""
    fig.update_layout(
        title_x=0.5,
        font=dict(color=PLOT_FONTCOLOR, size=14),
        plot_bgcolor=PLOT_BACKGROUND,
        paper_bgcolor=PLOT_BACKGROUND,
    )
    fig.update_xaxes(gridcolor=PLOT_GRIDCOLOR, tickfont=dict(size=13))
    fig.update_yaxes(gridcolor=PLOT_GRIDCOLOR, tickfont=dict(size=13))
    return fig


def plot_coherence_topics(start_k, end_k, coherence_scores, output_dir, step_k=2):
    """Interactive line chart of c_v coherence versus the number of topics."""
    k_values = list(range(start_k, end_k + 1, step_k))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=coherence_scores, mode="lines+markers",
        line=dict(color=TOPIC_PALETTE[0], width=2),
        marker=dict(size=10, color=TOPIC_PALETTE[0]),
    ))
    fig.update_layout(
        title="Coherence Score (c_v) vs. Number of Topics",
        xaxis_title="Number of Topics", yaxis_title="Coherence Score (c_v)",
        width=900, height=600,
    )
    _style_plotly_figure(fig)
    out_path = os.path.join(output_dir, "coherence_scores.html")
    pyo.plot(fig, filename=out_path, auto_open=False)
    print(f"Saved {out_path}")


def temporal_dynamic_plot(meta, output_dir):
    """
    Plot topic prevalence over time as two stacked-bar panels.

    The upper panel shows absolute article counts per topic and year; the
    lower panel normalizes each year to 1.0, so volume growth and relative
    share can be read separately.

    Skipped with a notice when the input has no ``Publication_Year`` column.
    """
    if "Publication_Year" not in meta.columns:
        print("No 'Publication_Year' column: skipping the temporal dynamics plot.")
        return

    dated = meta[meta["topic"].notna()].copy()
    dated["Publication_Year"] = pd.to_numeric(dated["Publication_Year"],
                                              errors="coerce")
    dated = dated.dropna(subset=["Publication_Year"])
    if dated.empty:
        print("No usable publication years: skipping the temporal dynamics plot.")
        return
    dated["Publication_Year"] = dated["Publication_Year"].astype(int)

    counts = dated.groupby(["Publication_Year", "topic"]).size() \
                  .reset_index(name="counts")
    totals = counts.groupby("Publication_Year")["counts"].sum() \
                   .reset_index(name="total")
    normalized = counts.merge(totals, on="Publication_Year")
    normalized["fraction"] = normalized["counts"] / normalized["total"]

    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    topics_sorted = sorted(counts["topic"].unique(), key=int)

    for topic in topics_sorted:
        color = topic_color(int(topic))
        topic_counts = counts[counts["topic"] == topic]
        topic_fraction = normalized[normalized["topic"] == topic]
        fig.add_trace(
            go.Bar(x=topic_counts["Publication_Year"], y=topic_counts["counts"],
                   name=f"Topic {topic}", marker_color=color, opacity=0.9),
            row=1, col=1)
        fig.add_trace(
            go.Bar(x=topic_fraction["Publication_Year"], y=topic_fraction["fraction"],
                   name=f"Topic {topic}", marker_color=color, opacity=0.9,
                   showlegend=False),
            row=2, col=1)

    fig.update_layout(
        height=800, barmode="stack", title_text="Temporal Dynamics of Topics",
        yaxis_title="Number of Articles", yaxis2_title="Fraction",
        xaxis2_title="Publication Year",
    )
    _style_plotly_figure(fig)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    out_path = os.path.join(output_dir, "temporal_dynamics_topics.html")
    fig.write_html(out_path, full_html=True)
    print(f"Saved {out_path}")


def assign_topics_and_save(meta, filtered_idx_out, model, output_dir):
    """
    Map cluster labels back onto the original (pre-filter) dataframe rows.

    Documents dropped during preprocessing (``filtered_idx_out``) get
    ``topic = 0`` ("unassigned"); modeled documents get ``topic`` in
    ``[1, k]``, 1-indexed for readability in figures and tables.

    :return: ``(silhouette_score, updated meta dataframe)``
    """
    labels = model.cluster_model.labels_ + 1
    labels_with_gaps = labels.copy()
    for idx in sorted(filtered_idx_out):
        labels_with_gaps = np.insert(labels_with_gaps, idx, 0)

    meta = meta.copy()
    if len(labels_with_gaps) != len(meta):
        raise ValueError(
            f"Label count ({len(labels_with_gaps)}) does not match the number "
            f"of input rows ({len(meta)}); cannot align topic assignments.")
    meta["topic"] = labels_with_gaps

    output_path = os.path.join(output_dir, "data_topic.csv")
    meta.to_csv(output_path, index=False)
    print(f"Per-document topic assignments saved to {output_path}")

    temporal_dynamic_plot(meta, output_dir)

    score = silhouette_score(model.vec[model.method], model.cluster_model.labels_)
    return score, meta


def plot_umap_projection(meta, filtered_idx_out, model, output_dir):
    """
    Draw a 2D UMAP scatter of the fitted document vectors, colored by topic.

    The model's silhouette score is shown in the title, so cluster separation
    can be judged next to the projection it describes.

    :return: the meta dataframe with its new ``topic`` column.
    """
    print("Calculating UMAP projection...")
    reducer = UMAP(random_state=model.random_state)
    embedding_2d = reducer.fit_transform(model.vec[model.method])
    print("Calculating UMAP projection. Done!")

    silhouette, meta = assign_topics_and_save(meta, filtered_idx_out, model, output_dir)

    labels = model.cluster_model.labels_ + 1
    df_plot = pd.DataFrame({
        "UMAP_1": embedding_2d[:, 0],
        "UMAP_2": embedding_2d[:, 1],
        "Topic": labels.astype(str),
    }).sort_values("Topic", key=lambda s: s.astype(int))

    color_map = {str(t): topic_color(int(t)) for t in np.unique(labels)}
    fig = px.scatter(df_plot, x="UMAP_1", y="UMAP_2", color="Topic",
                     color_discrete_map=color_map, width=1000, height=700)
    fig.update_traces(marker=dict(size=11, line=dict(width=0)))
    fig.update_layout(
        title=f"UMAP projection of topic clusters (silhouette = {silhouette:.2f})",
        legend=dict(font=dict(size=13)),
    )
    _style_plotly_figure(fig)

    out_path = os.path.join(output_dir, "UMAP.html")
    fig.write_html(out_path, auto_open=False)
    print(f"Interactive UMAP plot saved to {out_path}")
    return meta


def get_wordcloud(model, token_lists, topic_number_0_indexed, output_dir,
                  font_path=None, top_n=30):
    """
    Render and save one topic's word cloud as a PDF.

    The cloud is tinted with that topic's identity color, shared with the
    UMAP scatter and the temporal-dynamics plot, so a reader can match a word
    cloud back to its cluster visually.
    """
    topic_number = topic_number_0_indexed + 1
    print(f"Rendering word cloud for topic {topic_number} ...")

    labels = model.cluster_model.labels_
    tokens = " ".join(
        " ".join(token_lists[i]) for i in range(len(labels))
        if labels[i] == topic_number_0_indexed
    )
    if not tokens.strip():
        print(f"Topic {topic_number} has no tokens; skipping its word cloud.")
        return

    base_hue = _hex_to_hsl_hue(topic_color(topic_number))

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        """Tint every word with the topic hue; larger words render lighter."""
        lightness = min(80, font_size ** (3 / 5) + 20)
        return f"hsl({base_hue:.0f}, 55%, {lightness:.0f}%)"

    wordcloud = WordCloud(
        font_path=font_path,        # None -> wordcloud's bundled default font
        width=900, height=900, background_color="white",
        random_state=100, collocations=False, min_font_size=20,
        max_words=top_n, color_func=color_func,
    ).generate(tokens)

    plt.figure(dpi=300, figsize=(9, 9))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    mpl.rcParams["pdf.fonttype"] = 42   # embed fonts as text, not curves
    mpl.rcParams["ps.fonttype"] = 42

    out_path = os.path.join(output_dir, f"wordcloud_topic_{topic_number}.pdf")
    plt.savefig(out_path, dpi=300, format="pdf")
    plt.close()
    print(f"Saved {out_path}")


def export_topic_terms(model, token_lists, output_dir, top_n=30):
    """
    Write the top terms of every topic, with weights, to a CSV.

    LDA exposes real probabilistic term weights; the embedding-based methods
    do not, so their terms are ranked by frequency and given a weight of 1.0.
    """
    rows = []
    for topic_idx in range(model.k):
        if model.method == "LDA" and model.ldamodel is not None:
            terms = model.ldamodel.show_topic(topic_idx, topn=top_n)
        else:
            words = get_topic_words(token_lists, model.cluster_model.labels_,
                                    k=model.k, top_n=top_n)[topic_idx]
            terms = [(w, 1.0) for w in words]
        rows.extend({"Topic": topic_idx + 1, "Term": term, "Weight": weight}
                    for term, weight in terms)

    if not rows:
        print("No topic terms to export.")
        return

    out_path = os.path.join(output_dir, "wordcloud_terms_weights.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


# ===========================================================================
# 8. Evaluation stage
#    Scores the selected model with tm_topic_metrics.py. Everything below
#    reuses the vectors, labels and dictionary produced by the coherence scan
#    - no model is refit here.
# ===========================================================================
#: Identifier columns copied into document_level_metrics.csv when present.
EVAL_ID_COLUMNS = ["Abstract_ID", "Publication_Year", "Title"]


def evaluate_model(model, meta, token_lists, idx_in, idx_out, output_dir, args):
    """
    Run the ``tm_topic_metrics`` suite on the selected model.

    Produces model-, topic- and document-level metric tables plus figures:
    per-topic coherence, topic diversity, inter-topic similarity, silhouette,
    DBCV, assignment confidence and outlier proxies.

    :return: the metrics dict from ``tm_topic_metrics``, or None if the stage
        was skipped.
    """
    if not _HAS_TOPIC_METRICS:
        print("\nEvaluation stage skipped: 'tm_topic_metrics.py' was not found "
              "next to this script. Copy it here (it lives in the project's "
              "functions/ folder) or pass --skip-evaluation to silence this.")
        return None

    print("\nEvaluating the selected model with tm_topic_metrics ...")

    topics_sorted = list(range(1, model.k + 1))
    topic_labels = (model.cluster_model.labels_ + 1).tolist()

    # Representative words per topic, sliced to the evaluation width so
    # coherence and diversity are computed on the same vocabulary size.
    top_words_lists = get_topic_words(token_lists, model.cluster_model.labels_,
                                      k=model.k, top_n=args.eval_top_n_words)
    top_words_by_topic = {t: top_words_lists[t - 1] for t in topics_sorted}

    # Identifier columns for the modeled documents only, in vector order.
    id_columns = [c for c in EVAL_ID_COLUMNS if c in meta.columns]
    doc_id_table = meta.iloc[idx_in][id_columns].reset_index(drop=True)

    # Provenance recorded alongside the metrics so a run can be reproduced.
    extra_model_fields = {
        "n_documents_dropped_in_preprocessing": len(idx_out),
        "representation_method": model.method,
    }
    if model.method in ("LDA", "LDA_BERT"):
        extra_model_fields["lda_random_state"] = model.random_state
    if model.method == "LDA_BERT":
        extra_model_fields["gamma"] = model.gamma
    if model.method in ("BERT", "LDA_BERT"):
        extra_model_fields["sbert_model"] = args.sbert_model

    return tmet.evaluate_topic_assignments(
        topics_sorted=topics_sorted,
        topic_labels=topic_labels,
        vec=model.vec[model.method],
        token_lists=token_lists,
        dictionary=model.dictionary,
        top_words_by_topic=top_words_by_topic,
        output_dir=output_dir,
        doc_id_table=doc_id_table,
        top_n_words=args.eval_top_n_words,
        assignment_temperature=args.eval_temperature,
        low_confidence_threshold=args.eval_low_confidence,
        outlier_z_threshold=args.eval_outlier_z,
        random_state=args.random_state,
        extra_model_fields=extra_model_fields,
    )


# ===========================================================================
# 9. Main pipeline
# ===========================================================================
def main(argv=None):
    """
    Run the full pipeline.

    :return: 0 on success, 1 when preprocessing left too few documents to model.
    """
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    ensure_nltk_resources()
    load_custom_stopwords(args.custom_stopwords_csv)

    # -- 1. Load and combine text ----------------------------------------
    meta, documents = load_data(args.input_csv, args.text_columns, args.encoding)
    print(f"Loaded {meta.shape[0]} rows, {meta.shape[1]} columns.")

    # -- 2. Preprocess -----------------------------------------------------
    sentences, token_lists, idx_in, idx_out = preprocess(
        documents.tolist(),
        use_language_filter=not args.disable_language_filter,
        use_typo_correction=args.use_typo_correction,
        use_stemming=args.use_stemming,
    )
    print(f"{len(sentences)} documents kept after preprocessing "
          f"({len(idx_out)} dropped: empty after filtering / non-target language).")

    if len(token_lists) <= args.end_k:
        print(f"Only {len(token_lists)} documents survived preprocessing, which is "
              f"too few for up to {args.end_k} topics. Lower --end-k or relax the "
              f"stop word list.")
        return 1

    # -- 3. Scan topic counts and keep the most coherent model ------------
    models, coherence_scores = topic_model_coherence_generator(
        method=args.method, sentences=sentences, token_lists=token_lists,
        output_dir=args.output_dir, start_k=args.start_k, end_k=args.end_k,
        step_k=args.step_k, gamma=args.gamma, random_state=args.random_state,
        sbert_model_name=args.sbert_model,
    )
    plot_coherence_topics(args.start_k, args.end_k, coherence_scores,
                          args.output_dir, args.step_k)

    best_model = models[int(np.argmax(coherence_scores))]
    print(f"Selected model: {best_model.k} topics "
          f"(method={args.method}, coherence={max(coherence_scores):.4f})")

    # -- 4. Visualize the winning model ------------------------------------
    meta = plot_umap_projection(meta, idx_out, best_model, args.output_dir)

    for topic_idx in range(best_model.k):
        try:
            get_wordcloud(best_model, token_lists, topic_idx, args.output_dir,
                          font_path=args.wordcloud_font, top_n=args.wordcloud_top_n)
        except Exception as exc:
            # One unrenderable topic must not cost the whole run.
            print(f"Error rendering word cloud for topic {topic_idx + 1}: {exc}")
    export_topic_terms(best_model, token_lists, args.output_dir,
                       top_n=args.wordcloud_top_n)

    # -- 5. Evaluate the winning model ---------------------------------------
    if not args.skip_evaluation:
        evaluate_model(best_model, meta, token_lists, idx_in, idx_out,
                       args.output_dir, args)

    print(f"\nAll outputs written to: {os.path.abspath(args.output_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
