# Limestone Data Challenge 2026

> **We won the Limestone Challenge!**

**Team:** Tye Phoenix & Deven Pietrzak

> Full problem statement: [`data/limestone_data_challenge_2026.pdf`](data/limestone_data_challenge_2026.pdf)

![Problem Statement](data/challenge_pages.png)

A 3650×53 price matrix with ~50% missing values. Six columns are "index" columns (convex combinations of "farmer" columns). The challenge: identify indices, recover coefficients, complete the matrix, and build trading strategies.

### Challenge Overview

You run a bakery buying flour from *m* farmers at varying daily prices, published in a newspaper bulletin. Some prices are smudged (NaN). The bulletin also includes *n* index columns — convex combinations of farmer prices — but column identities are unknown.

| Problem | Task | Weight |
|---------|------|--------|
| 1a | Identify which columns are indices | 25% |
| 1b | Recover index decomposition coefficients | 5% |
| 2 | Fill all missing values in the matrix | 15% |
| 3 | Buy 100kg from NaN columns, minimize cost | 15% |
| 4 | Arbitrage: buy NaN, sell to index, maximize profit | 20% |
| 5 | Limit orders on NaN columns with price + quantity | 20% |

## Project Structure

```
Tower/
├── data/
│   ├── limestone_data_challenge_2026.data.csv   # raw input (3650 rows × 53 cols + time)
│   ├── limestone_data_challenge_2026.pdf        # full problem statement
│   └── challenge_pages.png                      # problem statement rendered as image
├── answers/                                      # generated outputs (suffixed by method)
│   ├── problem1a_answer-tye.csv                 # Tye's column classification
│   ├── problem1a_answer-deven.csv               # Deven's column classification
│   ├── problem1b_answer-tye.csv                 # Tye's decomposition coefficients
│   ├── problem1b_answer-deven.csv               # Deven's decomposition coefficients
│   ├── problem2_answer-tye.csv                  # completed matrix (Tye's P1 + Tye's P2)
│   └── problem2_answer-deven.csv                # completed matrix (Deven's P1 + Tye's P2)
├── problem-1/                                    # index detection — two independent methods
│   ├── main.py                                   # entry point, --method tye|deven (default tye)
│   ├── tye/                                      # Tye's row-residual + NNLS method
│   │   ├── candidates.py                         # phase 1: row-residual index detection
│   │   └── coefficients.py                       # phase 2: NNLS coefficient recovery
│   └── deven/                                    # Deven's 7-stage stability verification
│       ├── common.py                             # shared helpers + hardcoded decomposition constants
│       └── scripts/                              # 01–07 sequential verification stages
├── problem-2/                                    # full pipeline: P1 + matrix completion + EM refinement
│   ├── main.py                                   # entry point, --method tye|deven
│   ├── matrix.py                                 # iterative SVD completion (torch)
│   ├── trend.py                                  # periodic trend fitting (Lomb-Scargle + OLS harmonics)
│   ├── em.py                                     # standalone EM refinement (optional)
│   ├── intermediates/                            # cached results (--intermediates flag)
│   └── analysis/                                 # diagnostic plots (--intermediates flag)
├── problem-3_4/                                  # buying + arbitrage strategies
│   ├── buy.py                                    # trading_problem_3: buy 100kg cheapest
│   └── trade.py                                  # trading_problem_4: arbitrage
├── problem-5/                                    # limit-order buying
│   ├── main.py                                   # compute σ + backtest
│   ├── trade.py                                  # trading_problem_5: optimal bidding
│   ├── compute_sigma.py                          # standalone σ computation
│   └── intermediates/
│       └── sigma.json                            # precomputed per-column σ values
├── compact.py                                    # notebook generator (uses rep2nb)
├── limestone_data_challenge_2026.ipynb           # submission notebook (generated)
└── requirements.txt                              # Python dependencies
```

## Results

| Problem | Metric | Value |
|---------|--------|-------|
| **1a** | Indices detected | 6 identified (col_11, col_30, col_42, col_46, col_48, col_50) — **both methods agree** |
| **1b** | Cross-method agreement | 4 of 6 decompositions numerically identical (max Δ < 3e-6); col_30 and col_46 agree on majors (Δ < 0.03) |
| **1b** | Coefficient sum | 1.0000 for all 6 indices |
| **1b** | Verify RMSE (co-observed) | < 0.01 for proven indices |
| **2** | SVD obs RMSE | 0.0031 (rank 47, post-EM) |
| **3** | Cost vs oracle (3650 rows) | 56,169,176 / 56,169,176 — **100% oracle-hit rate** |
| **3** | Out-of-time cost/oracle | 106,095 / 100,927 — **5.1% above oracle** (7 synthetic rows) |
| **4** | Profit vs oracle (3650 rows) | 5,064,153 / 5,064,153 — **100% capture rate** |
| **4** | Out-of-time profit | 4,960 / 10,762 oracle (46%) |
| **5** | Fill rate (500 training rows) | 473/500 (**94.6%**) |
| **5** | Total score (profit) | 453,574 |
| **5** | Avg profit / day | 907.1 |

## Problem 1 — Two Independent Approaches

Tye and Deven attacked Problem 1 separately, using different methodologies. The two methods converged on the same 6 index columns and (for 4 of 6) numerically identical decomposition coefficients. The two remaining columns (col_30 and col_46) have multiple valid convex parameterizations; both methods agree on the dominant farmers and differ only on minor (< 0.03) weights.

Run each method independently:

```bash
cd problem-1
python3 main.py --method tye     # row-residual + NNLS  → problem1{a,b}_answer-tye.csv
python3 main.py --method deven   # 7-stage verification → problem1{a,b}_answer-deven.csv
```

### Method 1 — Tye: row-residual test + NNLS coefficient recovery

**Detection (1a).** For each column, greedily build a predictor set from its most correlated columns while maintaining sufficient co-observed rows. Fit a convex combination (non-negative weights summing to 1) on a train split, measure RMSE on a test split. True indices have near-zero test RMSE; farmers have high RMSE. A threshold of 3.5 cleanly separates the two groups.

**Coefficient recovery (1b).** Iterative rank-and-peel using multiple NNLS methods:
- Method A: NNLS on co-observed rows (original data)
- Method B: NNLS on constraint-filled data
- Method C: Greedy forward selection
- Method D: Farmer-only pool variants

Each method's result is verified via convex re-fit. Best result per column is accepted. An EM loop (SVD completion → re-regression) refines uncertain coefficients. Index-to-index dependencies are resolved via `distribute_through` to express everything in terms of pure farmer columns.

### Method 2 — Deven: 7-stage stability verification

A sequential pipeline that discovers and validates exact decompositions through aggressive constraint propagation and stepwise regression:

1. **Validate known decomps** — confirm exact-fit RMSE for each hypothesised decomposition on fully co-observed rows
2. **Stability validation** — bootstrap convex fits over candidate support sets; inspect weight means, stds, and coefficient-of-variation to separate stable indices from noise
3. **Farmer-only stepwise** — restrict the predictor pool to high/medium-confidence farmers; forward-stepwise on every column
4. **Fill and rerun** — use exact decomps to propagate fills across the matrix, then re-solve uncertain columns on the densified data
5. **Exact fill all** — apply all 8 known decomps round-robin until fixed point; rank every remaining column's stepwise fit
6. **Refine col_30 and col_46** — equality-constrained full-precision convex fits with dense post-fill data
7. **Classify remaining** — regress unclassified columns on the known-farmer pool and label as `LIKELY INDEX / UNCERTAIN / LIKELY FARMER`

The hardcoded `KNOWN_DECOMPS_EXACT` constants in `problem-1/deven/common.py` are the output of this verification — the scripts exist to *prove* those constants are correct.

### Convergence Comparison (Tye distributed-to-farmers vs Deven)

| Index | Tye # farmers | Deven # farmers | Shared | Max Δ | Σ\|Δ\| |
|-------|---------------|-----------------|--------|-------|--------|
| col_11 | 5 | 5 | 5 | **0.000001** | 0.000002 |
| col_42 | 2 | 2 | 2 | **0.000003** | 0.000003 |
| col_48 | 5 | 5 | 5 | **0.000000** | 0.000001 |
| col_50 | 3 | 3 | 3 | **0.000002** | 0.000003 |
| col_30 | 9 | 7 | 7 | 0.027 | 0.136 |
| col_46 | 15 | 9 | 8 | 0.013 | 0.101 |

Four columns match to 6+ decimal places. col_30 and col_46 have legitimately multi-valued support — the farmer pool has enough correlated redundancy that slightly different subsets can each reproduce the index. Both methods agree on the dominant farmers; they differ only in whether to attribute the last percent of variance to sparse or denser sets.

## Problem 2 — Matrix Completion

**Goal:** Fill all ~96,000 missing values.

**Approach:** Three-stage warm start followed by iterative SVD:
1. **Constraint propagation:** deterministically fill ~3,000 cells from proven index decompositions
2. **Trend model:** fit per-column periodic trends (Lomb-Scargle dominant period detection + OLS harmonic regression with up to 5 harmonics, BIC model selection for single vs double period)
3. **Iterative SVD:** rank-47 (= number of farmers) SVD completion using PyTorch, warm-started from steps 1+2
4. **EM refinement:** alternate between SVD completion and re-regression of uncertain coefficients until convergence

Final reconstruction enforces index constraints exactly and preserves all original observations.

`problem-2/main.py --method deven` uses Deven's decompositions as the input to the same completion machinery — producing an alternate completed matrix.

## Problem 3 — Buy 100kg

**Goal:** Given a row with NaN prices, buy exactly 100kg from NaN-priced columns to minimize cost.

**Approach:**
- **Historical rows (t ≤ 3649):** look up predicted prices from the Problem 2 completed matrix
- **Out-of-time rows:** algebraic fills from decomposition constraints, then KNN (k=20) + low-rank SVD projection (rank 12), blended 50/50

Buy all 100kg from the single cheapest predicted NaN column.

## Problem 4 — Arbitrage

**Goal:** Buy from a NaN column and sell to an index column to maximize profit, up to 100kg.

**Approach:** Same price prediction as Problem 3. Find the cheapest NaN column (source) and the most expensive index column (destination). If dest > src, trade 100kg.

## Problem 5 — Limit-Order Buying

**Goal:** Place limit orders (price + quantity) on NaN columns. Orders fill only if bid ≥ true price. Score = Σ(qty × (median - bid) × I{fill}).

**Approach:**
- **Per-column uncertainty (σ):** estimated via leave-one-out cross-validation — hide one observed value, predict it via KNN+low-rank, record error. σ_j = std(errors). Precomputed and saved to `intermediates/sigma.json`.
- **Cell classification:** each NaN cell is tagged as algebraic (σ ≈ 0.02) or SVD-imputed (σ capped at 3.0 for historical, inflated 1.5× for out-of-time).
- **Optimal bid:** maximize E[profit/kg] = (median − bid) × Φ((bid − p̂) / σ) via bounded scalar optimization.
- **Allocation:** 100kg on best column for historical rows; spread across top 3 for out-of-time.

**Backtest result:** 94.6% fill rate, avg profit 907/day on 500 training rows.

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Full pipeline (Problems 1 & 2 end-to-end)

```bash
cd problem-2
python3 main.py --intermediates               # Tye's method (default)
python3 main.py --method deven --intermediates # alternate: Deven's P1 + Tye's P2
```

This generates:
- `answers/problem1a_answer-<method>.csv` — column classifications
- `answers/problem1b_answer-<method>.csv` — decomposition coefficients (EM-refined)
- `answers/problem2_answer-<method>.csv` — completed matrix
- `problem-2/intermediates/*.json` — candidate lists, coefficient snapshots, SVD metadata
- `problem-2/analysis/*.png` — diagnostic plots

Runtime: ~30 seconds (tye), ~15 seconds (deven) depending on hardware.

### Problem 1 standalone (no imputation)

```bash
cd problem-1
python3 main.py                # Tye's method (default) — writes problem1{a,b}_answer-tye.csv
python3 main.py --method deven # runs Deven's 7-stage verification
```

### Problem 5 pipeline

```bash
cd problem-5
python3 main.py                  # compute σ in memory, run backtest
python3 main.py --intermediates  # also save intermediates/sigma.json
```

Runtime: ~15 seconds.

### Test trading strategies

```bash
cd problem-3_4
python3 buy.py          # Problem 3 — buy strategy
python3 trade.py        # Problem 4 — arbitrage

cd ../problem-5
python3 trade.py        # Problem 5 — limit-order buying
```

### Run everything from scratch

```bash
cd problem-2 && python3 main.py --intermediates
cd ../problem-5 && python3 main.py --intermediates
cd ../problem-3_4 && python3 buy.py && python3 trade.py
cd ../problem-5 && python3 trade.py
```

### Generate submission notebook

```bash
pip install rep2nb
python3 compact.py
```

Produces `limestone_data_challenge_2026.ipynb` — a fully executable notebook with correct dependency ordering, `!pip install` cell, and section isolation. The notebook contains only Tye's unified pipeline; Deven's verification suite is excluded to keep the notebook a single coherent submission. Powered by [rep2nb](https://github.com/tyephoenix/rep2nb).

## Key Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Linear algebra, array operations |
| pandas | Data I/O, DataFrames |
| scipy | NNLS, SLSQP, Lomb-Scargle, minimize_scalar, normal CDF |
| torch | GPU-accelerated iterative SVD (falls back to CPU) |
| rep2nb | Converts this repo into an executable notebook |

---

### Notebook Generation

The submission notebook for this project was generated using [**rep2nb**](https://github.com/tyephoenix/rep2nb) — a tool I built that converts an entire Python repository into a single executable Jupyter notebook. It handles dependency ordering, cross-file imports, sections, argparse, and more.

```bash
pip install rep2nb
```

Check it out: [GitHub](https://github.com/tyephoenix/rep2nb) | [PyPI](https://pypi.org/project/rep2nb/)
