# Pan Core Analyzer

Pan-/core-term accumulation analysis with Heaps' law fitting and a permutation test.

Borrowing the pan-genome framework from comparative genomics, this script treats a corpus
as a population of documents and its vocabulary as a gene pool. It answers one question:
**as documents accumulate, is the field still introducing new terminology, or is its
vocabulary saturating?**

---

## Concepts

| Term | Definition |
|------|------------|
| **Pan terms** | The union — every distinct term seen so far |
| **Core terms** | The strict intersection — terms present in *every* document so far |
| **Soft-core terms** | Terms present in at least a given fraction of the documents so far (95 % by default) |
| **New terms** | Terms contributed for the first time by each document — the first difference of the pan curve |

The pan curve is fitted with **Heaps' law**, `n = kappa * N ** gamma`. The exponent
`gamma` is the quantity of interest: a value near 1 means the vocabulary keeps growing at
an undiminished rate, a low value means it is approaching saturation.

Two scenarios are run and compared:

| Scenario | Meaning |
|----------|---------|
| **Ordered** | Documents accumulated in chronological order — the quantity of interest |
| **Randomized** | Documents accumulated in random order, over many permutations — the null model, showing what the curve looks like when publication order carries no information |

A **permutation test** then compares the chronological exponent against the null
distribution, giving an empirical p-value for the claim that chronological order matters.

---

## Where it fits

```
ab_extractor.py  →  tdm_generator  →  pan_core_analyzer.py
   abstracts        term-document matrix    accumulation curves + Heaps fit
```

The optional chronological ordering can be taken from `data_topic.csv` (written by
`tm_analyzer.py`) or from any metadata table that shares document IDs with the TDM.

---

## Installation

```bash
pip install -r requirements_pan_core_analyzer.txt
```

Python 3.9 or newer is required (verified on Python 3.11).

---

## Usage

### Basic usage

```bash
python pan_core_analyzer.py --input-tdm <tdm.csv> --output-dir <output folder>
```

### Examples

```bash
# 1) Default run: 100 permutations, TDM column order assumed chronological
python pan_core_analyzer.py \
    --input-tdm D_tdm.csv \
    --output-dir ./pan_core

# 2) Chronological order taken from a metadata file (recommended)
python pan_core_analyzer.py \
    --input-tdm D_tdm.csv \
    --output-dir ./pan_core \
    --document-order-csv data_topic.csv \
    --order-column Abstract_ID \
    --order-sort-by Publication_Year Abstract_ID

# 3) A larger null model, and keywords instead of terms in the labels
python pan_core_analyzer.py \
    --input-tdm D_keywords_all_tdm.csv \
    --output-dir ./pan_core_keywords \
    --y-title keyword --iterations 1000
```

### Use from a Python script

```python
from pan_core_analyzer import pan_core_term_analyzer

stats = pan_core_term_analyzer(
    x_title="document",
    y_title="term",
    input_tdm_file_path="D_tdm.csv",
    output_dir="./pan_core",
    iterations=100,
)
print(stats[["ordered_gamma", "random_gamma_mean_of_fits",
             "permutation_empirical_p"]])
```

---

## Command line options

### Input and output

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input-tdm` | (required) | CSV term-document matrix |
| `-o`, `--output-dir` | (required) | Directory for all output files |
| `--x-title` | `document` | Name of the accumulation unit, used in labels |
| `--y-title` | `term` | Name of the accumulated item, used in labels and file names |

### Model

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations` | `100` | Number of permutations for the null model |
| `--seed` | `20241203` | Seed for the permutation RNG |
| `--p0` | `1.0 0.5` | Starting values `(kappa, gamma)` for every curve fit |
| `--maxfev` | `10000` | Maximum function evaluations per fit |
| `--soft-core-threshold` | `0.95` | Fraction of documents a term must appear in to count as soft core |

### Chronological order

| Option | Default | Description |
|--------|---------|-------------|
| `--document-order-csv` | none | Metadata CSV giving the chronological document order |
| `--order-column` | `Abstract_ID` | Column holding the document IDs; must match the TDM column labels |
| `--order-sort-by` | none | Columns to sort the metadata by first, e.g. `Publication_Year Abstract_ID` |

### Other

| Option | Default | Description |
|--------|---------|-------------|
| `--max-background-curves` | `100` | Permutation curves drawn behind the randomized figure |
| `--log-level` | `INFO` | Console verbosity |

---

## Input requirements

The term-document matrix is a CSV with **terms in rows and documents in columns**, term
labels in the first column:

```
term,1_2013,2_2013,3_2014,...
mycobacterium,3,0,1,...
infection,1,2,0,...
```

Values are counts; anything above zero counts as presence.

**Chronological order matters.** The ordered scenario is only meaningful if the columns
really are in publication order. Without `--document-order-csv` the script assumes the
existing column order is chronological and logs a warning saying so. Supplying the order
explicitly removes that assumption.

---

## Generated files

```
<output-dir>/
├── pan_term.csv                        # every term with frequency and document count
├── core_term.csv                       # terms present in every document
├── soft_core_term.csv                  # terms present in ≥ threshold of documents
├── ordered_accumulation.csv            # chronological pan/core/soft-core/new curves
├── random_mean_accumulation.csv        # mean and SD of the permutation curves
├── random_per_iteration_fits.csv       # kappa, gamma, R² for every permutation
├── pan_core_statistics.csv             # one-row summary of the whole analysis
├── software_versions.csv               # library versions, for reproducibility
├── combined_ordered_analysis.html      # chronological accumulation + new-term panels
├── combined_random_analysis.html       # randomized accumulation + new-term panels
└── permutation_null_gamma.html         # null distribution with the observed value marked
```

The `_term` suffix follows `--y-title`, so `--y-title keyword` produces `pan_keyword.csv`
and so on.

### Key fields of `pan_core_statistics.csv`

| Field | Meaning |
|-------|---------|
| `ordered_gamma` | Heaps exponent of the chronological curve |
| `ordered_r_squared` | How well Heaps' law describes that curve |
| `random_gamma_mean_of_fits` | Mean of the per-permutation exponents — **the value to report in text** |
| `random_gamma_sd_of_fits` | Spread of the null distribution |
| `random_gamma_fit_of_mean_curve` | Single fit to the averaged curve — **the curve drawn in the figure** |
| `permutation_empirical_p` | Fraction of permutations with an exponent at least as large as the observed one |
| `permutation_z_score` | Observed exponent expressed in null standard deviations |
| `p0_robustness_check_passed` | Whether alternative starting values converged to the same exponent |
| `random_iterations_converged` | How many permutation fits actually converged |

---

## Methodological notes

These are deliberate choices, several of them made in response to peer review. They
should be reflected in any Methods section that reports these numbers.

- **The RNG is explicitly seeded.** The seed is recorded in the statistics output, so the
  randomized results are exactly reproducible.
- **One starting-value vector for every fit.** The ordered and randomized exponents are
  estimated under identical conditions, and `check_p0_robustness()` re-fits the ordered
  curve from three alternative starting values to confirm the optimizer did not settle in
  a different local minimum.
- **No "open"/"closed" pangenome label is emitted.** For a cumulative power-law fit gamma
  is bounded on (0, 1] by construction, so the sign of gamma carries no information and
  such a label is always "open". Interpretation rests on where gamma falls inside that
  interval and on the permutation test.
- **Two randomized quantities, never conflated.** `gamma_mean_of_fits` is the mean of the
  per-permutation exponents; `gamma_fit_of_mean` is a single fit to the averaged curve.
  They answer different questions and generally differ. The figure draws the second; the
  text should quote the first.
- **Soft-core is tracked alongside the strict core.** The strict intersection collapses to
  zero for any corpus of realistic size, so on its own it carries no information.
- **The fit is in linear, not log-log, space.** This weights the high-N portion of the
  accumulation curve more heavily. State it explicitly when reporting.
- **The new-term panel is a first difference.** For any exponent below 1 a declining
  trend there is an arithmetic consequence of sublinear accumulation, not independent
  evidence of saturation. This caveat is printed into the figure page itself so that
  captions written from those panels stay honest.

---

## Notes and limitations

1. **Runtime scales as iterations × documents × terms.** A 20,000-term × 2,000-document
   TDM with 100 permutations takes a while. Start with `--iterations 10` to check the
   setup, then raise it for the final run.
2. **The p-value resolution is bounded by the permutation count.** With 100 permutations
   the smallest reportable p-value is 1/101 ≈ 0.0099; the add-one correction means it can
   never be reported as exactly zero. Use `--iterations 1000` if you need finer
   resolution.
3. **A high R² does not confirm a power law.** It says only that Heaps' law fits better
   than a horizontal line. Compare against the permutation null before drawing
   conclusions.
4. Permutation fits that fail to converge are excluded from the null distribution and
   counted in `random_iterations_converged`; check that field before reporting.

---

## Troubleshooting

### `The TDM has 1 document column(s); accumulation needs at least 2`
The input was read with the wrong orientation, or the index column was misdetected. The
TDM needs terms in rows and documents in columns, with term labels in the first column.

### `Column 'X' not found in <metadata>`
The `--order-column` or `--order-sort-by` names do not match the metadata file. The error
lists the columns that are actually available.

### `N requested columns are not in the TDM`
The document IDs in the metadata do not match the TDM column labels. Check for trailing
whitespace, or for IDs written as numbers in one file and strings in the other.

### `Every permutation fit failed to converge`
Try different starting values (`--p0`) or raise `--maxfev`. A pan curve that is nearly
linear can also be hard to fit — inspect `ordered_accumulation.csv` first.

### `Fit is sensitive to starting values` warning
`p0_robustness_check_passed` is `False` in the output. The reported exponent is a local
optimum; do not report it without investigating.

### The randomized HTML file is very large
Lower `--max-background-curves`, which controls how many individual permutation curves
are drawn behind the mean.

---

## Related scripts

| Script | Role |
|--------|------|
| `ab_extractor.py` | Produces the abstract table from a PubMed query |
| `tm_analyzer.py` | Assigns a topic to every document, and can supply the chronological order |

---

## License

MIT — see [LICENSE](LICENSE) in the repository root.
