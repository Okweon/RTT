#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pan_core_analyzer.py
===========================================================================
Pan-/core-term accumulation analysis with Heaps' law fitting.

Borrowing the pan-genome framework from comparative genomics, this script
treats a corpus as a population of documents and its vocabulary as a gene
pool:

- **pan terms**       - the union: every distinct term seen so far.
- **core terms**      - the strict intersection: terms present in *every*
                        document so far.
- **soft-core terms** - terms present in at least a given fraction of the
                        documents so far (95 % by default).
- **new terms**       - terms contributed for the first time by each
                        document, i.e. the first difference of the pan curve.

The pan curve is fitted with Heaps' law, ``n = kappa * N ** gamma``. The
exponent gamma says how fast the vocabulary keeps growing as documents
accumulate: a value near 1 means the field is still introducing new
terminology at an undiminished rate, while a low value means the vocabulary
is approaching saturation.

Two scenarios are run and compared:

1. **Ordered** - documents accumulated in chronological order, which is the
   quantity of interest.
2. **Randomized** - documents accumulated in random order, repeated over
   many permutations. This is the null model: it shows what the curve looks
   like when publication order carries no information.

A permutation test then compares the chronological exponent against the null
distribution, giving an empirical p-value for the claim that chronological
order matters.

Usage
-----
    python pan_core_analyzer.py --input-tdm D_tdm.csv --output-dir ./pan_core

    # with the chronological order taken from a metadata file
    python pan_core_analyzer.py \\
        --input-tdm D_tdm.csv --output-dir ./pan_core \\
        --document-order-csv data_topic.csv \\
        --order-column Abstract_ID \\
        --order-sort-by Publication_Year Abstract_ID

Requirements
------------
    pip install -r requirements_pan_core_analyzer.txt

Methodological notes
--------------------
*   The permutation RNG is explicitly seeded, so randomized results are
    exactly reproducible. The seed is recorded in the statistics output.
*   A single initial-value vector (``p0``) is used for every curve fit, and
    convergence is re-checked from alternative starting values.
*   No "open"/"closed" pangenome label is emitted. For a cumulative
    power-law fit gamma is bounded on (0, 1] by construction, so the sign of
    gamma carries no information and such a label is always "open".
    Interpretation rests on where gamma falls inside that interval and on
    the permutation test.
*   Two distinct randomized quantities are reported and never conflated:
    ``gamma_mean_of_fits`` (the mean of the per-permutation exponents) and
    ``gamma_fit_of_mean`` (a single fit to the averaged curve, which is the
    curve drawn in the figure).
*   Soft-core accumulation is tracked alongside the strict core, because the
    strict intersection collapses to zero for any corpus of realistic size
    and therefore carries no information on its own.
"""

# ===========================================================================
# 0. Imports
# ===========================================================================
import os
import sys
import logging
import argparse
import platform
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit

import plotly
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm

LOGGER = logging.getLogger("pan_core_analyzer")


# ===========================================================================
# 1. Defaults
# ===========================================================================
#: Any fixed value works; this one is the date of the original PubMed search.
DEFAULT_SEED = 20241203

#: (kappa, gamma) starting values, used for every fit so that the ordered and
#: randomized exponents are estimated under identical conditions.
DEFAULT_P0 = (1.0, 0.5)
DEFAULT_MAXFEV = 10000

#: Starting values used by check_p0_robustness() to confirm convergence.
ALTERNATIVE_P0 = ((1.0, 1.0), (10.0, 0.3), (100.0, 0.7))

#: A term is "soft core" when it appears in at least this fraction of the
#: documents accumulated so far.
SOFT_CORE_THRESHOLD = 0.95

#: Cap on the number of individual permutation curves drawn as grey
#: background lines; beyond this the HTML becomes unwieldy.
DEFAULT_MAX_BACKGROUND_CURVES = 100


# ===========================================================================
# 2. Heaps' law fitting
# ===========================================================================
def heaps(x, kappa, gamma):
    """Heaps' law: ``n = kappa * N ** gamma``."""
    return kappa * x ** gamma


@dataclass
class HeapsFit:
    """Result of one Heaps' law fit."""
    kappa: float
    gamma: float
    r_squared: float
    kappa_se: float
    gamma_se: float
    n_points: int


def fit_heaps(x: np.ndarray,
              y: np.ndarray,
              p0: Sequence[float] = DEFAULT_P0,
              maxfev: int = DEFAULT_MAXFEV) -> HeapsFit:
    """
    Fit ``n = kappa * N ** gamma`` by non-linear least squares.

    The fit is performed in untransformed (linear) space rather than
    log-log space. This weights the high-N portion of the accumulation curve
    more heavily, which should be stated explicitly in any Methods section
    reporting these numbers.

    :param x: number of documents accumulated (1, 2, ... N).
    :param y: the accumulation curve to fit, normally the pan-term count.
    :param p0: starting values for ``(kappa, gamma)``.
    :param maxfev: maximum number of function evaluations.
    :return: a :class:`HeapsFit` with the parameters, R-squared and
        asymptotic standard errors.
    :raises RuntimeError: if the optimizer fails to converge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    parameters, covariance = curve_fit(heaps, x, y, p0=list(p0), maxfev=maxfev)
    kappa, gamma = parameters

    residuals = y - heaps(x, kappa, gamma)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    with np.errstate(invalid="ignore"):
        errors = np.sqrt(np.diag(covariance))
    kappa_se, gamma_se = ((float(errors[0]), float(errors[1]))
                          if errors.size == 2 else (float("nan"), float("nan")))

    return HeapsFit(kappa=float(kappa), gamma=float(gamma),
                    r_squared=r_squared, kappa_se=kappa_se,
                    gamma_se=gamma_se, n_points=int(x.size))


def check_p0_robustness(x: np.ndarray,
                        y: np.ndarray,
                        reference: HeapsFit,
                        alternatives: Sequence[Sequence[float]] = ALTERNATIVE_P0,
                        tol: float = 1e-4,
                        maxfev: int = DEFAULT_MAXFEV) -> bool:
    """
    Re-fit from alternative starting values and confirm the same optimum.

    Non-linear least squares can settle in a different local minimum
    depending on where it starts. Passing this check supports the statement
    that equivalent estimates were obtained from alternative starting values.

    :return: True when every alternative converged to the same exponent.
    """
    for alternative in alternatives:
        try:
            alternative_fit = fit_heaps(x, y, p0=alternative, maxfev=maxfev)
        except Exception as exc:
            LOGGER.warning("Alternative p0 %s failed to converge: %s",
                           alternative, exc)
            return False
        if abs(alternative_fit.gamma - reference.gamma) > tol:
            LOGGER.warning(
                "Fit is sensitive to starting values: p0=%s gave gamma=%.6f "
                "vs reference gamma=%.6f",
                alternative, alternative_fit.gamma, reference.gamma)
            return False
    return True


# ===========================================================================
# 3. Accumulation
# ===========================================================================
def accumulate(binary_by_document: np.ndarray,
               order: np.ndarray,
               soft_core_threshold: float = SOFT_CORE_THRESHOLD
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk through the documents in ``order`` and record the curves.

    After each document the function records:

    ==============  =======================================================
    ``pan``         distinct terms observed so far (union)
    ``core``        terms present in *every* document so far (intersection)
    ``soft_core``   terms present in at least ``soft_core_threshold`` of the
                    documents so far
    ``new``         terms contributed for the first time by that document
    ==============  =======================================================

    All four are derived from one running per-term document-count vector, so
    they are mutually consistent by construction and far faster than the
    equivalent set operations.

    :param binary_by_document: presence/absence matrix with **documents in
        rows and terms in columns**. Row-major access keeps each step
        contiguous in memory.
    :param order: indices of the document rows, in accumulation order.
    :return: ``(n_docs_axis, pan, core, soft_core, new)``.
    """
    n_terms = binary_by_document.shape[1]
    n_docs = order.size

    counts = np.zeros(n_terms, dtype=np.int32)
    pan = np.empty(n_docs, dtype=np.int64)
    core = np.empty(n_docs, dtype=np.int64)
    soft_core = np.empty(n_docs, dtype=np.int64)
    new = np.empty(n_docs, dtype=np.int64)

    previous_pan = 0
    for i, row in enumerate(order):
        counts += binary_by_document[row]
        seen = i + 1

        pan_i = int(np.count_nonzero(counts))
        pan[i] = pan_i
        new[i] = pan_i - previous_pan
        previous_pan = pan_i

        core[i] = int(np.count_nonzero(counts == seen))
        cutoff = int(np.ceil(soft_core_threshold * seen))
        soft_core[i] = int(np.count_nonzero(counts >= cutoff))

    return np.arange(1, n_docs + 1, dtype=np.int64), pan, core, soft_core, new


# ===========================================================================
# 4. Input handling
# ===========================================================================
def load_tdm(input_tdm_file_path: str,
             document_order: Optional[Sequence] = None) -> pd.DataFrame:
    """
    Load the term-document matrix and put its columns in accumulation order.

    :param input_tdm_file_path: CSV with terms in rows and documents in
        columns; the first column holds the term labels.
    :param document_order: optional list of column labels giving the
        chronological order. When omitted the existing column order is used
        and a warning is logged, because the ordered scenario is only
        meaningful if that order really is chronological.
    :return: the reordered DataFrame.
    :raises ValueError: when requested columns are absent from the TDM.
    """
    LOGGER.info("Loading TDM from %s ...", input_tdm_file_path)
    df = pd.read_csv(input_tdm_file_path, index_col=0)
    LOGGER.info("TDM loaded: %d terms x %d documents.", df.shape[0], df.shape[1])

    if document_order is None:
        LOGGER.warning("No explicit document order supplied; the existing "
                       "column order of the TDM is assumed to be chronological. "
                       "Verify this before reporting the ordered scenario.")
        return df

    missing = [c for c in document_order if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} requested columns are not in the TDM "
            f"(first few: {missing[:5]})")

    df = df.loc[:, list(document_order)]
    LOGGER.info("Documents reordered according to the supplied chronological "
                "order (%d columns).", df.shape[1])
    return df


def read_document_order(order_csv: str, order_column: str,
                        sort_by: Optional[Sequence[str]] = None) -> list:
    """
    Read the chronological document order from a metadata CSV.

    :param order_csv: metadata file, e.g. ``data_topic.csv``.
    :param order_column: column holding the document IDs, which must match
        the TDM's column labels.
    :param sort_by: optional columns to sort by first, e.g.
        ``["Publication_Year", "Abstract_ID"]``. Without this the file's
        existing row order is used.
    :return: list of document IDs as strings.
    """
    meta = pd.read_csv(order_csv)

    for column in [order_column] + list(sort_by or []):
        if column not in meta.columns:
            raise ValueError(
                f"Column '{column}' not found in {order_csv}. "
                f"Available columns: {list(meta.columns)}")

    if sort_by:
        meta = meta.sort_values(list(sort_by))
        LOGGER.info("Metadata sorted by %s.", ", ".join(sort_by))

    order = meta[order_column].astype(str).tolist()
    LOGGER.info("Read a document order of %d entries from %s.",
                len(order), order_csv)
    return order


# ===========================================================================
# 5. Main analysis
# ===========================================================================
def pan_core_term_analyzer(x_title: str,
                           y_title: str,
                           input_tdm_file_path: str,
                           output_dir: str,
                           iterations: int = 100,
                           seed: int = DEFAULT_SEED,
                           p0: Sequence[float] = DEFAULT_P0,
                           maxfev: int = DEFAULT_MAXFEV,
                           soft_core_threshold: float = SOFT_CORE_THRESHOLD,
                           document_order: Optional[Sequence] = None,
                           max_background_curves: int = DEFAULT_MAX_BACKGROUND_CURVES
                           ) -> pd.DataFrame:
    """
    Run the ordered and randomized pan-/core-term analyses.

    :param x_title: name of the accumulation unit, e.g. ``document``.
    :param y_title: name of the accumulated item, e.g. ``term``.
    :param input_tdm_file_path: CSV term-document matrix.
    :param output_dir: directory that receives every output file.
    :param iterations: number of independent permutations for the null model.
    :param seed: seed for the permutation RNG; fixing it makes the randomized
        results exactly reproducible.
    :param p0: starting values used for every curve fit.
    :param maxfev: maximum function evaluations per fit.
    :param soft_core_threshold: fraction of documents a term must appear in
        to count as soft core.
    :param document_order: explicit chronological order of column labels.
    :param max_background_curves: how many permutation curves to draw behind
        the randomized figure.
    :return: a one-row DataFrame of summary statistics, also written to
        ``pan_core_statistics.csv``.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_tdm(input_tdm_file_path, document_order)

    # Documents in rows: each accumulation step then reads one contiguous row.
    binary_by_document = np.ascontiguousarray((df.to_numpy() > 0).T.astype(np.int32))
    n_docs = binary_by_document.shape[0]
    document_counts = binary_by_document.sum(axis=0)
    n_terms_total = int(np.count_nonzero(document_counts))

    if n_docs < 2:
        raise ValueError(f"The TDM has {n_docs} document column(s); "
                         f"accumulation needs at least 2.")

    # ---- static pan / core inventories -------------------------------- #
    write_term_inventories(df, document_counts, n_docs, y_title,
                           soft_core_threshold, output_dir)
    core_terms = df.index[document_counts == n_docs]
    soft_cutoff = int(np.ceil(soft_core_threshold * n_docs))
    soft_core_terms = df.index[document_counts >= soft_cutoff]

    LOGGER.info("Pan %s: %d | strict core: %d | soft-core (>=%.0f%%): %d",
                y_title, n_terms_total, len(core_terms),
                100 * soft_core_threshold, len(soft_core_terms))

    # ---- ordered (chronological) scenario ------------------------------ #
    LOGGER.info("Ordered (chronological) accumulation ...")
    x, pan, core, soft_core, new = accumulate(
        binary_by_document, np.arange(n_docs), soft_core_threshold)

    pd.DataFrame({
        f"Number of {x_title}": x,
        f"Pan {y_title} count": pan,
        f"Core {y_title} count": core,
        f"Soft-core {y_title} count": soft_core,
        f"New {y_title} count": new,
    }).to_csv(os.path.join(output_dir, "ordered_accumulation.csv"), index=False)

    ordered_fit = fit_heaps(x, pan, p0=p0, maxfev=maxfev)
    p0_robust = check_p0_robustness(x, pan, ordered_fit, maxfev=maxfev)
    LOGGER.info("Ordered fit: gamma = %.4f (SE %.4f), kappa = %.3f, "
                "R2 = %.4f, alpha = 1 - gamma = %.4f",
                ordered_fit.gamma, ordered_fit.gamma_se, ordered_fit.kappa,
                ordered_fit.r_squared, 1.0 - ordered_fit.gamma)

    # ---- randomized scenario (null model) ------------------------------- #
    LOGGER.info("Randomized accumulation, %d permutations, seed = %d ...",
                iterations, seed)
    rng = np.random.default_rng(seed)

    pan_stack = np.empty((iterations, n_docs), dtype=np.int64)
    core_stack = np.empty((iterations, n_docs), dtype=np.int64)
    soft_stack = np.empty((iterations, n_docs), dtype=np.int64)
    new_stack = np.empty((iterations, n_docs), dtype=np.int64)
    per_iteration = []
    n_failed_fits = 0

    for iteration in tqdm(range(iterations), desc="Permutations"):
        shuffled = rng.permutation(n_docs)
        _, pan_i, core_i, soft_i, new_i = accumulate(
            binary_by_document, shuffled, soft_core_threshold)
        pan_stack[iteration] = pan_i
        core_stack[iteration] = core_i
        soft_stack[iteration] = soft_i
        new_stack[iteration] = new_i

        # One non-converging permutation must not abort the whole null model.
        try:
            fit_i = fit_heaps(x, pan_i, p0=p0, maxfev=maxfev)
            record = {"kappa": fit_i.kappa, "gamma": fit_i.gamma,
                      "r_squared": fit_i.r_squared}
        except Exception as exc:
            n_failed_fits += 1
            LOGGER.warning("Permutation %d failed to converge: %s", iteration + 1, exc)
            record = {"kappa": float("nan"), "gamma": float("nan"),
                      "r_squared": float("nan")}
        per_iteration.append({"iteration": iteration + 1, **record})

    per_iteration_df = pd.DataFrame(per_iteration)
    per_iteration_df.to_csv(
        os.path.join(output_dir, "random_per_iteration_fits.csv"), index=False)

    if n_failed_fits:
        LOGGER.warning("%d of %d permutation fits failed to converge and are "
                       "excluded from the null distribution.",
                       n_failed_fits, iterations)

    gammas = per_iteration_df["gamma"].dropna().to_numpy()
    kappas = per_iteration_df["kappa"].dropna().to_numpy()
    r_squareds = per_iteration_df["r_squared"].dropna().to_numpy()
    if gammas.size == 0:
        raise RuntimeError("Every permutation fit failed to converge; "
                           "try different --p0 values or more --maxfev.")

    n_valid = gammas.size
    gamma_mean_of_fits = float(np.mean(gammas))
    gamma_sd_of_fits = float(np.std(gammas, ddof=1)) if n_valid > 1 else float("nan")
    kappa_mean_of_fits = float(np.mean(kappas))
    kappa_sd_of_fits = float(np.std(kappas, ddof=1)) if n_valid > 1 else float("nan")

    mean_pan_curve = pan_stack.mean(axis=0)
    fit_of_mean = fit_heaps(x, mean_pan_curve, p0=p0, maxfev=maxfev)

    pd.DataFrame({
        f"Number of {x_title}": x,
        f"Mean pan {y_title} count": mean_pan_curve,
        f"SD pan {y_title} count": (pan_stack.std(axis=0, ddof=1)
                                    if iterations > 1 else np.zeros(n_docs)),
        f"Mean core {y_title} count": core_stack.mean(axis=0),
        f"Mean soft-core {y_title} count": soft_stack.mean(axis=0),
        f"Mean new {y_title} count": new_stack.mean(axis=0),
    }).to_csv(os.path.join(output_dir, "random_mean_accumulation.csv"), index=False)

    LOGGER.info("Randomized: gamma (mean of %d fits) = %.4f +/- %.4f; "
                "gamma (fit to mean curve) = %.4f; mean R2 = %.4f",
                n_valid, gamma_mean_of_fits, gamma_sd_of_fits,
                fit_of_mean.gamma, float(np.mean(r_squareds)))

    # ---- permutation test of the chronological exponent ------------------ #
    # Add-one correction: with a finite number of permutations the p-value can
    # never legitimately be reported as exactly zero.
    n_ge = int(np.sum(gammas >= ordered_fit.gamma))
    empirical_p = (n_ge + 1) / (n_valid + 1)
    z_score = ((ordered_fit.gamma - gamma_mean_of_fits) / gamma_sd_of_fits
               if gamma_sd_of_fits and np.isfinite(gamma_sd_of_fits)
               and gamma_sd_of_fits > 0 else float("nan"))
    LOGGER.info("Permutation test: %d/%d permutation exponents >= observed; "
                "empirical p = %.4f; z = %.2f",
                n_ge, n_valid, empirical_p, z_score)

    # ---- consolidated statistics ------------------------------------------ #
    statistics = {
        "input_tdm": os.path.basename(input_tdm_file_path),
        "n_documents": n_docs,
        "n_terms_pan": n_terms_total,
        "n_terms_strict_core": int(len(core_terms)),
        "soft_core_threshold": soft_core_threshold,
        "n_terms_soft_core": int(len(soft_core_terms)),

        "ordered_kappa": ordered_fit.kappa,
        "ordered_kappa_se": ordered_fit.kappa_se,
        "ordered_gamma": ordered_fit.gamma,
        "ordered_gamma_se": ordered_fit.gamma_se,
        "ordered_alpha_1_minus_gamma": 1.0 - ordered_fit.gamma,
        "ordered_r_squared": ordered_fit.r_squared,

        "random_iterations": iterations,
        "random_iterations_converged": n_valid,
        "random_seed": seed,
        "random_kappa_mean_of_fits": kappa_mean_of_fits,
        "random_kappa_sd_of_fits": kappa_sd_of_fits,
        "random_gamma_mean_of_fits": gamma_mean_of_fits,
        "random_gamma_sd_of_fits": gamma_sd_of_fits,
        "random_gamma_min": float(np.min(gammas)),
        "random_gamma_max": float(np.max(gammas)),
        "random_alpha_mean_1_minus_gamma": 1.0 - gamma_mean_of_fits,
        "random_r_squared_mean_of_fits": float(np.mean(r_squareds)),
        "random_kappa_fit_of_mean_curve": fit_of_mean.kappa,
        "random_gamma_fit_of_mean_curve": fit_of_mean.gamma,
        "random_r_squared_fit_of_mean_curve": fit_of_mean.r_squared,

        "permutation_n_ge_observed": n_ge,
        "permutation_empirical_p": empirical_p,
        "permutation_z_score": z_score,

        "curve_fit_p0_kappa": p0[0],
        "curve_fit_p0_gamma": p0[1],
        "curve_fit_maxfev": maxfev,
        "curve_fit_space": "linear (untransformed) non-linear least squares",
        "p0_robustness_check_passed": bool(p0_robust),
    }
    statistics_df = pd.DataFrame([statistics])
    statistics_df.to_csv(os.path.join(output_dir, "pan_core_statistics.csv"),
                         index=False)

    write_software_versions(output_dir)

    # ---- figures ------------------------------------------------------------ #
    plot_scenario(output_dir, "ordered", x_title, y_title,
                  x, pan, core, soft_core, new,
                  fitted=heaps(x, ordered_fit.kappa, ordered_fit.gamma),
                  fit_label=(f"Heaps fit: gamma = {ordered_fit.gamma:.3f}, "
                             f"R2 = {ordered_fit.r_squared:.3f}"),
                  title_prefix="Ordered (chronological)")

    plot_scenario(output_dir, "random", x_title, y_title,
                  x, mean_pan_curve, core_stack.mean(axis=0),
                  soft_stack.mean(axis=0), new_stack.mean(axis=0),
                  fitted=heaps(x, fit_of_mean.kappa, fit_of_mean.gamma),
                  fit_label=(f"Fit to mean curve: gamma = {fit_of_mean.gamma:.3f}"
                             f" (mean of per-permutation fits = "
                             f"{gamma_mean_of_fits:.3f} +/- {gamma_sd_of_fits:.3f})"),
                  title_prefix=f"Randomized ({iterations} permutations)",
                  background_curves=pan_stack[:max_background_curves])

    plot_permutation_null(output_dir, gammas, ordered_fit.gamma,
                          empirical_p, y_title)

    LOGGER.info("Pan-core analysis complete. Output: %s",
                os.path.abspath(output_dir))
    return statistics_df


def write_term_inventories(df, document_counts, n_docs, y_title,
                           soft_core_threshold, output_dir):
    """
    Write the static pan / core / soft-core term lists.

    These are the whole-corpus inventories, independent of accumulation
    order: which terms exist, which appear in every document, and which
    appear in at least ``soft_core_threshold`` of them.
    """
    term_frequency = df.sum(axis=1)
    present = (term_frequency > 0).to_numpy()

    pd.DataFrame({
        "Index": np.arange(1, int(present.sum()) + 1),
        y_title: df.index[present],
        "Frequency": term_frequency[present].to_numpy(),
        "Document_count": document_counts[present],
    }).to_csv(os.path.join(output_dir, f"pan_{y_title}.csv"), index=False)

    pd.Series(df.index[document_counts == n_docs], name=y_title).to_csv(
        os.path.join(output_dir, f"core_{y_title}.csv"), index=False)

    soft_cutoff = int(np.ceil(soft_core_threshold * n_docs))
    pd.Series(df.index[document_counts >= soft_cutoff], name=y_title).to_csv(
        os.path.join(output_dir, f"soft_core_{y_title}.csv"), index=False)


def write_software_versions(output_dir: str) -> None:
    """Record library versions, to support a reproducibility statement."""
    rows = [
        ("python", platform.python_version()),
        ("numpy", np.__version__),
        ("pandas", pd.__version__),
        ("scipy", scipy.__version__),
        ("plotly", plotly.__version__),
    ]
    pd.DataFrame(rows, columns=["library", "version"]).to_csv(
        os.path.join(output_dir, "software_versions.csv"), index=False)


# ===========================================================================
# 6. Figures
# ===========================================================================
_AXIS_STYLE = dict(title_font={"size": 22}, tickfont=dict(color="black", size=18),
                   linecolor="black", linewidth=2)

_LAYOUT_STYLE = dict(template="plotly_white", height=600, width=1000,
                     paper_bgcolor="white", plot_bgcolor="white",
                     margin=dict(l=10, r=10, t=60, b=10), showlegend=True,
                     title_font=dict(size=24, color="black"), title_x=0.05,
                     title_y=0.97,
                     legend=dict(borderwidth=0.3, bordercolor="gray"))

#: Caption embedded in the figure page. The lower panel is the first
#: difference of the upper one, so a declining trend there is arithmetic and
#: not independent evidence of saturation - stating this in the figure keeps
#: the caption written from these panels honest.
_FIRST_DIFFERENCE_NOTE = (
    "Note: the lower panel plots the first differences of the cumulative curve "
    "above. For any exponent below 1 a declining trend is an arithmetic "
    "consequence of sublinear accumulation and is not independent evidence of "
    "lexical saturation."
)


def plot_scenario(output_dir, label, x_title, y_title,
                  x, pan, core, soft_core, new,
                  fitted, fit_label, title_prefix,
                  background_curves=None) -> None:
    """
    Write one scenario's two panels to a single HTML page.

    The upper panel shows the pan, soft-core and strict-core accumulation
    curves with the Heaps fit; the lower panel shows the per-document count
    of new terms on log-log axes.

    :param background_curves: individual permutation curves drawn faintly
        behind the mean, to show the spread of the null model.
    """
    figure = go.Figure()

    if background_curves is not None:
        for row in background_curves:
            figure.add_trace(go.Scatter(x=x, y=row, mode="lines",
                                        line=dict(color="lightgray", width=1),
                                        opacity=0.35, showlegend=False,
                                        hoverinfo="skip"))

    figure.add_trace(go.Scatter(x=x, y=pan, mode="markers",
                                marker=dict(size=6, color="royalblue", opacity=0.6),
                                name=f"Pan {y_title}"))
    figure.add_trace(go.Scatter(x=x, y=fitted, mode="lines",
                                line=dict(color="crimson", width=3),
                                name=fit_label))
    figure.add_trace(go.Scatter(x=x, y=soft_core, mode="lines",
                                line=dict(color="darkorange", width=2),
                                name=f"Soft-core {y_title}"))
    figure.add_trace(go.Scatter(x=x, y=core, mode="lines",
                                line=dict(color="seagreen", width=2, dash="dot"),
                                name=f"Strict core {y_title}"))
    figure.update_layout(
        title=f"{title_prefix} pan-core {y_title} accumulation",
        xaxis_title=f"Number of {x_title}",
        yaxis_title=f"{y_title.capitalize()} count",
        legend_title="Legend", **_LAYOUT_STYLE)
    figure.update_xaxes(**_AXIS_STYLE)
    figure.update_yaxes(**_AXIS_STYLE)

    new_figure = go.Figure()
    new_figure.add_trace(go.Scatter(x=x, y=new, mode="markers",
                                    marker=dict(size=5, color="black", opacity=0.6),
                                    name=f"New {y_title} per {x_title}"))
    new_figure.update_layout(
        title=f"New {y_title} count ({title_prefix.lower()})",
        xaxis_title=f"Number of {x_title}",
        yaxis_title=f"Number of new {y_title}",
        xaxis_type="log", yaxis_type="log",
        legend_title="Legend", **_LAYOUT_STYLE)
    new_figure.update_xaxes(**_AXIS_STYLE)
    new_figure.update_yaxes(**_AXIS_STYLE)

    out_path = os.path.join(output_dir, f"combined_{label}_analysis.html")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("<html><head><meta charset='utf-8'>"
                 f"<title>{title_prefix} analysis</title></head><body>")
        fp.write(pio.to_html(figure, full_html=False))
        fp.write("<div style='margin-top: 40px;'></div>")
        fp.write(pio.to_html(new_figure, full_html=False))
        fp.write("<p style='font-family:sans-serif;font-size:13px;color:#444;"
                 f"max-width:960px;margin-top:24px;'>{_FIRST_DIFFERENCE_NOTE}</p>")
        fp.write("</body></html>")
    LOGGER.info("Wrote %s", out_path)


def plot_permutation_null(output_dir, gammas, observed_gamma,
                          empirical_p, y_title) -> None:
    """
    Plot the permutation null distribution of gamma.

    The chronological exponent is marked on the same axis, so the reader can
    see directly whether it falls inside or outside the null distribution.
    """
    figure = go.Figure()
    figure.add_trace(go.Histogram(
        x=gammas, nbinsx=30,
        marker=dict(color="lightsteelblue",
                    line=dict(color="steelblue", width=1)),
        name="Permutation null"))
    figure.add_vline(x=observed_gamma, line=dict(color="crimson", width=3))
    figure.add_annotation(
        x=observed_gamma, y=1, yref="paper", yanchor="bottom",
        text=(f"Chronological gamma = {observed_gamma:.3f}<br>"
              f"empirical p = {empirical_p:.4f}"),
        showarrow=False, font=dict(size=15, color="crimson"))
    figure.update_layout(
        title=f"Permutation null distribution of the Heaps exponent ({y_title})",
        xaxis_title="Fitted exponent, gamma",
        yaxis_title="Number of permutations",
        legend_title="Legend", **_LAYOUT_STYLE)
    figure.update_xaxes(**_AXIS_STYLE)
    figure.update_yaxes(**_AXIS_STYLE)

    out_path = os.path.join(output_dir, "permutation_null_gamma.html")
    figure.write_html(out_path)
    LOGGER.info("Wrote %s", out_path)


# ===========================================================================
# 7. Command line interface
# ===========================================================================
def parse_args(argv=None):
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        prog="pan_core_analyzer.py",
        description="Pan-/core-term accumulation analysis with Heaps' law "
                    "fitting and a permutation test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", "--input-tdm", required=True,
                        help="CSV term-document matrix: terms in rows, "
                             "documents in columns, term labels in column 1.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory for all output files.")

    parser.add_argument("--x-title", default="document",
                        help="Name of the accumulation unit, used in labels.")
    parser.add_argument("--y-title", default="term",
                        help="Name of the accumulated item, used in labels "
                             "and in the output file names.")

    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of permutations for the null model.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Seed for the permutation RNG.")
    parser.add_argument("--p0", type=float, nargs=2, default=list(DEFAULT_P0),
                        metavar=("KAPPA", "GAMMA"),
                        help="Starting values for every curve fit.")
    parser.add_argument("--maxfev", type=int, default=DEFAULT_MAXFEV,
                        help="Maximum function evaluations per curve fit.")
    parser.add_argument("--soft-core-threshold", type=float,
                        default=SOFT_CORE_THRESHOLD,
                        help="Fraction of documents a term must appear in to "
                             "count as soft core.")

    parser.add_argument("--document-order-csv", default=None,
                        help="Metadata CSV giving the chronological document "
                             "order. Without it the TDM's existing column "
                             "order is assumed to be chronological.")
    parser.add_argument("--order-column", default="Abstract_ID",
                        help="Column of --document-order-csv holding the "
                             "document IDs; must match the TDM column labels.")
    parser.add_argument("--order-sort-by", nargs="+", default=None,
                        help="Columns to sort the metadata by first, e.g. "
                             "Publication_Year Abstract_ID.")

    parser.add_argument("--max-background-curves", type=int,
                        default=DEFAULT_MAX_BACKGROUND_CURVES,
                        help="Permutation curves drawn behind the randomized "
                             "figure. Lower this if the HTML gets too large.")
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

    LOGGER.info("Pan-core term analysis started.")

    try:
        document_order = None
        if args.document_order_csv:
            document_order = read_document_order(
                args.document_order_csv, args.order_column, args.order_sort_by)

        statistics = pan_core_term_analyzer(
            x_title=args.x_title,
            y_title=args.y_title,
            input_tdm_file_path=args.input_tdm,
            output_dir=args.output_dir,
            iterations=args.iterations,
            seed=args.seed,
            p0=tuple(args.p0),
            maxfev=args.maxfev,
            soft_core_threshold=args.soft_core_threshold,
            document_order=document_order,
            max_background_curves=args.max_background_curves,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(statistics.T.to_string(header=False))
    LOGGER.info("Pan-core term analysis completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
