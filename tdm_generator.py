#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tdm_generator.py
===========================================================================
Build term-document matrices (TDMs) from a bibliographic table.

A TDM has terms in rows and documents in columns, each cell holding how many
times that term occurs in that document. It is the input format for
``pan_core_analyzer.py`` and for the word cloud / co-occurrence tools.

Three matrices are produced from every input table, because they answer
different questions:

============================  =============================================
``D_tdm.csv``                 Full text (Title + Abstract + Keywords),
                              preprocessed: nouns only, lemmatized, stop
                              words removed. The general-purpose matrix.
``D_keywords_all_tdm.csv``    Author keywords split into individual words.
``D_keywords_tdm.csv``        Author keywords kept as whole phrases, so
                              "food supply chain" stays one term.
============================  =============================================

Usage
-----
    python tdm_generator.py \\
        --input-csv pubmed_abstracts_filtered.csv \\
        --output-dir ./tdm \\
        --custom-stopwords-csv custom_stopwords.csv

Requirements
------------
    pip install -r requirements_tdm_generator.txt

The first run downloads three NLTK resources (``wordnet``, ``stopwords``,
``averaged_perceptron_tagger_eng``); later runs reuse them.
"""

# ===========================================================================
# 0. Imports
# ===========================================================================
import os
import re
import sys
import logging
import argparse

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

LOGGER = logging.getLogger("tdm_generator")


# ===========================================================================
# 1. Constants
# ===========================================================================
#: Columns concatenated into the full-text field.
DEFAULT_TEXT_COLUMNS = ["Title", "Abstract", "Keywords"]

#: Column used as the document (column) label of the matrices.
DEFAULT_ID_COLUMN = "Abstract_ID"

#: NLTK resources and the sub-directory each one actually lives in. Getting
#: these paths right matters: a wrong prefix makes nltk.data.find() raise
#: LookupError every time, so the resource is re-downloaded on every run.
NLTK_RESOURCES = {
    "wordnet": "corpora/wordnet",
    "stopwords": "corpora/stopwords",
    # NLTK 3.9 renamed the English tagger; pos_tag() looks for the _eng name.
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}

#: Part-of-speech prefixes kept by the full-text preprocessing. Nouns only:
#: topics and term networks read as concepts, which nouns carry. Add "VB"
#: and "JJ" here to keep verbs and adjectives as well.
KEPT_POS_PREFIXES = ("NN",)

#: Placeholder written by ab_extractor.py for fields PubMed did not supply.
NOT_AVAILABLE = "not available"

#: Terms shorter than this are dropped: units and acronym fragments, mostly.
MIN_TERM_LENGTH = 3

_LEMMATIZER = None
_BASE_STOPWORDS = None
_NLTK_READY = False


# ===========================================================================
# 2. NLTK setup
# ===========================================================================
def _nltk_resource_present(path):
    """
    Report whether an NLTK resource is already available.

    Downloaded corpora often stay packed, so ``corpora/wordnet`` can be
    missing while ``corpora/wordnet.zip`` is present and perfectly usable.
    Checking only the unpacked path makes every run re-download the corpus.
    """
    for candidate in (path, f"{path}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def setup_nltk_resources():
    """Download the required NLTK resources, but only if they are missing."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    for name, path in NLTK_RESOURCES.items():
        if not _nltk_resource_present(path):
            LOGGER.info("Downloading NLTK resource '%s' ...", name)
            nltk.download(name, quiet=True)
    _NLTK_READY = True


def _get_lemmatizer():
    """Return the shared lemmatizer, building it on first use."""
    global _LEMMATIZER
    if _LEMMATIZER is None:
        setup_nltk_resources()
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def _get_base_stopwords():
    """Return the English stop word set, loading it on first use."""
    global _BASE_STOPWORDS
    if _BASE_STOPWORDS is None:
        setup_nltk_resources()
        _BASE_STOPWORDS = set(stopwords.words("english"))
    return _BASE_STOPWORDS


# ===========================================================================
# 3. Text preprocessing
# ===========================================================================
def preprocess_text(text, custom_stopwords=frozenset()):
    """
    Clean, filter and lemmatize one document's text.

    The steps, in order: strip HTML, split on slashes, drop everything that
    is not alphanumeric or a hyphen, lowercase, remove stop words and short
    tokens, remove tokens containing digits, keep only the parts of speech in
    :data:`KEPT_POS_PREFIXES`, then lemmatize.

    :param text: raw text; non-strings and the "Not available" placeholder
        return an empty string.
    :param custom_stopwords: domain-specific stop words to remove as well.
    :return: the processed text as a space-joined string.
    """
    if not isinstance(text, str) or text.strip().lower() == NOT_AVAILABLE:
        return ""

    text = re.sub(r"<[^>]*>", "", text)              # HTML tags
    text = text.replace("/", " ")                    # "and/or" -> two words
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)      # keep hyphens
    text = text.lower()

    words = [w for w in text.split()
             if w not in _get_base_stopwords()
             and w not in custom_stopwords
             and len(w) >= MIN_TERM_LENGTH]
    # Measurements and identifiers are not topical vocabulary.
    words = [w for w in words if not re.search(r"\d", w)]
    words = [w for w, pos in nltk.pos_tag(words) if pos[:2] in KEPT_POS_PREFIXES]

    lemmatizer = _get_lemmatizer()
    return " ".join(lemmatizer.lemmatize(w) for w in words)


def custom_tokenizer(text):
    """
    Split text into words, keeping hyphenated compounds intact.

    ``non-tuberculous`` stays one token rather than becoming two.
    Tokens are lowercased so that ``Food`` and ``food`` are the same term -
    the vectorizer runs with ``lowercase=False`` because these tokenizers
    handle casing themselves.
    """
    return [token.lower() for token in re.findall(r"\b\w+(?:-\w+)*\b", str(text))]


def csv_keyword_tokenizer(text):
    """
    Split a comma-separated keyword field, keeping each keyword whole.

    ``"food supply chain, cold chain"`` becomes two multi-word terms rather
    than five single words.
    """
    if not isinstance(text, str):
        return []
    return [kw.strip().lower() for kw in text.split(",")
            if kw.strip() and len(kw.strip()) >= MIN_TERM_LENGTH
            and not kw.strip().isdigit()]


def load_stopwords(filepath):
    """
    Load custom stop words from a single-column CSV.

    :param filepath: path to the CSV, or None.
    :return: a lowercase set; empty when no usable file was given.
    """
    if not filepath:
        return set()
    if not os.path.exists(filepath):
        LOGGER.warning("Custom stopwords file not found: %s. "
                       "Using the NLTK list only.", filepath)
        return set()

    stopwords_df = pd.read_csv(filepath, header=None)
    words = {str(w).strip().lower() for w in stopwords_df[0].tolist() if str(w).strip()}
    LOGGER.info("Loaded %d custom stop words from %s", len(words), filepath)
    return words


# ===========================================================================
# 4. Matrix construction
# ===========================================================================
def build_tdm(texts, doc_ids, tokenizer, custom_stopwords=frozenset()):
    """
    Vectorize a text column into a cleaned term-document matrix.

    Rows that survive: terms that are not purely numeric, not blank, at least
    :data:`MIN_TERM_LENGTH` characters, present in at least one document, and
    not in the custom stop word list.

    :param texts: iterable of document texts.
    :param doc_ids: document labels, used as the matrix columns.
    :param tokenizer: callable splitting one text into tokens.
    :return: DataFrame with terms in rows and documents in columns.
    """
    vectorizer = CountVectorizer(tokenizer=tokenizer, lowercase=False,
                                 token_pattern=None)
    matrix = vectorizer.fit_transform(texts)

    tdm = pd.DataFrame(matrix.toarray().T,
                       index=vectorizer.get_feature_names_out(),
                       columns=doc_ids)

    tdm = tdm[~tdm.index.str.isnumeric()]
    tdm = tdm[tdm.index.str.strip() != ""]
    tdm = tdm[tdm.index.str.len() >= MIN_TERM_LENGTH]
    tdm = tdm.loc[(tdm != 0).any(axis=1)]
    tdm = tdm[~tdm.index.str.lower().isin(custom_stopwords)]
    tdm = tdm[~tdm.index.str.lower().isin({NOT_AVAILABLE})]
    return tdm


def save_tdm_to_csv(tdm, output_path):
    """Write a term-document matrix to CSV and report its shape."""
    tdm.to_csv(output_path)
    LOGGER.info("TDM saved to %s (%d terms x %d documents)",
                output_path, tdm.shape[0], tdm.shape[1])


def generate_tdms(data, custom_stopwords, output_dir,
                  text_columns=None, id_column=DEFAULT_ID_COLUMN,
                  filename_prefix=""):
    """
    Build and write the three term-document matrices for one table.

    :param data: DataFrame of documents.
    :param custom_stopwords: set of domain stop words.
    :param output_dir: directory that receives the CSV files.
    :param text_columns: columns concatenated into the full-text field.
    :param id_column: column used as the matrix column labels.
    :param filename_prefix: prepended to every output filename, used to keep
        per-topic matrices apart.
    :return: ``{filename: (n_terms, n_documents)}``.
    :raises ValueError: when a required column is missing.
    """
    text_columns = text_columns or DEFAULT_TEXT_COLUMNS
    os.makedirs(output_dir, exist_ok=True)

    missing = [c for c in list(text_columns) + [id_column] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing}. "
                         f"Available columns: {list(data.columns)}")

    data = data.copy()

    # Blank out the placeholder per column, before concatenating: once the
    # columns are joined the whole-field comparison in preprocess_text() can
    # no longer match it. fillna("") keeps pandas from writing the literal
    # string "nan" into the text.
    for column in text_columns:
        series = data[column].fillna("").astype(str)
        data[column] = series.mask(series.str.strip().str.lower() == NOT_AVAILABLE, "")

    LOGGER.info("Preprocessing %d documents ...", len(data))
    data["Combined"] = data[text_columns].agg(" ".join, axis=1)
    data["Combined"] = data["Combined"].apply(
        lambda text: preprocess_text(text, custom_stopwords))

    # Each matrix answers a different question - see the module docstring.
    specifications = {
        "D_keywords_tdm.csv": ("Keywords", csv_keyword_tokenizer),
        "D_keywords_all_tdm.csv": ("Keywords", custom_tokenizer),
        "D_tdm.csv": ("Combined", custom_tokenizer),
    }

    doc_ids = data[id_column].astype(str)
    shapes = {}
    for filename, (column, tokenizer) in specifications.items():
        if column not in data.columns:
            LOGGER.warning("Column '%s' not present; skipping %s.", column, filename)
            continue

        tdm = build_tdm(data[column], doc_ids, tokenizer, custom_stopwords)
        output_name = f"{filename_prefix}{filename}"
        save_tdm_to_csv(tdm, os.path.join(output_dir, output_name))
        shapes[output_name] = tdm.shape

    LOGGER.info("All TDMs generated successfully.")
    return shapes


# ===========================================================================
# 5. Public wrappers
# ===========================================================================
def tdm_generator(input_csv_filename, custom_stopwords_csv_filename, output_dir,
                  text_columns=None, id_column=DEFAULT_ID_COLUMN, encoding="utf-8"):
    """
    Build the term-document matrices from a CSV file.

    :param input_csv_filename: bibliographic table, e.g. the output of
        ``ab_extractor.py``.
    :param custom_stopwords_csv_filename: single-column CSV of extra stop
        words, or None.
    :param output_dir: directory that receives the CSV files.
    :return: ``{filename: (n_terms, n_documents)}``.
    """
    try:
        data = pd.read_csv(input_csv_filename, encoding=encoding)
    except UnicodeDecodeError:
        data = pd.read_csv(input_csv_filename, encoding="cp1252")
        LOGGER.info("Note: '%s' decoding failed; loaded the CSV as cp1252.", encoding)
    LOGGER.info("Loaded %d rows from %s", len(data), input_csv_filename)

    custom_stopwords = load_stopwords(custom_stopwords_csv_filename)
    return generate_tdms(data, custom_stopwords, output_dir,
                         text_columns, id_column)


def tdm_generator_df(topic_num, data, custom_stopwords_csv_filename, output_dir,
                     text_columns=None, id_column=DEFAULT_ID_COLUMN):
    """
    Build the term-document matrices from an in-memory DataFrame.

    Output filenames are prefixed with ``topic_num`` so that per-topic
    matrices can share one directory. Used by
    ``word_cloud_network_generator.py``.

    :param topic_num: label prepended to every output filename.
    :return: ``{filename: (n_terms, n_documents)}``.
    """
    custom_stopwords = load_stopwords(custom_stopwords_csv_filename)
    return generate_tdms(data, custom_stopwords, output_dir,
                         text_columns, id_column,
                         filename_prefix=f"{topic_num}_")


# ===========================================================================
# 6. Command line interface
# ===========================================================================
def parse_args(argv=None):
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        prog="tdm_generator.py",
        description="Build term-document matrices from a bibliographic table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-csv", required=True,
                        help="Bibliographic table, e.g. the output of "
                             "ab_extractor.py.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory for the generated matrices.")
    parser.add_argument("-s", "--custom-stopwords-csv", default=None,
                        help="Single-column CSV of domain stop words.")
    parser.add_argument("--text-columns", nargs="+", default=DEFAULT_TEXT_COLUMNS,
                        help="Columns concatenated into the full-text field.")
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN,
                        help="Column used as the matrix column labels.")
    parser.add_argument("--encoding", default="utf-8",
                        help="Encoding of the input CSV; cp1252 is tried as a "
                             "fallback.")
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
        shapes = tdm_generator(args.input_csv, args.custom_stopwords_csv,
                               args.output_dir, args.text_columns,
                               args.id_column, args.encoding)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(f"\nMatrices written to {os.path.abspath(args.output_dir)}:")
    for filename, (n_terms, n_docs) in shapes.items():
        print(f"  {filename:<28} {n_terms:>7} terms x {n_docs:>6} documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
