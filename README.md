# prop-var
Code and data analysis for the bioRxiv preprint: "Proportional variability relative to lower abundances: An interpretable and efficient index of deep lows in population time series" (https://doi.org/10.1101/2022.02.19.481101)

# Proportional variability relative to lower abundances (PV<sub>lo</sub>)

Code accompanying the paper

> M. Schlemmer. *Proportional variability relative to lower abundances: An interpretable and efficient index of deep lows in population time series.* bioRxiv, doi: [10.1101/2022.02.19.481101](https://doi.org/10.1101/2022.02.19.481101)

The repository contains the simulation notebook that produces the numbers, tables and figure data of the paper, and a short self-contained implementation of the index.

## What the index is

For a time series of abundances, **PV<sub>lo</sub>** averages the proportional difference of all pairs of observations relative to the *lower* value of each pair and maps the result to the unit interval:

- pv<sub>lo</sub> = mean over all pairs of |z<sub>i</sub> − z<sub>j</sub>| / min(z<sub>i</sub>, z<sub>j</sub>)
- PV<sub>lo</sub> = pv<sub>lo</sub> / (1 + pv<sub>lo</sub>)

For positive abundances, 1 + pv<sub>lo</sub> is the average ratio of the higher to the lower value of a pair, so PV<sub>lo</sub> = 0.5 means an average ratio of 2 and PV<sub>lo</sub> = 0.9 an average ratio of 10. In a pair containing a zero, the lower value is treated as one individual, so zero counts need no data transformation. PV<sub>hi</sub> (Heath 2006), which scales relative to the *higher* value, is computed alongside.

Both indices are evaluated without enumerating the pairs (sorting and cumulative sums), so the effort is O(n log n) instead of O(n²). The index ignores the temporal order of the observations.

## Quick start

The function below is copied from the notebook and needs only NumPy.

```python
import numpy as np

def pv_indices(z):
    """Return pv_lo, PV_lo and PV_hi of a time series z (abundances >= 0)."""
    z = np.sort(np.asarray(z, dtype=float))
    n = z.size
    C = n * (n - 1) / 2                       # number of unique pairs
    n0 = int(np.sum(z == 0))                  # number of zeros
    pos = z[z > 0]                            # strictly positive abundances (sorted)
    m = pos.size
    if m >= 2:
        suffix = np.cumsum(pos[::-1])[::-1]
        sum_pd_lo = np.sum(suffix[1:] / pos[:-1]) - m * (m - 1) / 2
        prefix = np.cumsum(pos)
        sum_pd_hi = m * (m - 1) / 2 - np.sum(prefix[:-1] / pos[1:])
    else:
        sum_pd_lo = sum_pd_hi = 0.0
    sum_pd_lo += n0 * pos.sum()               # pairs (0, z): d_lo = z
    sum_pd_hi += n0 * m                       # pairs (0, z): d_hi = 1
    pv_lo = sum_pd_lo / C
    return pv_lo, pv_lo / (1 + pv_lo), sum_pd_hi / C

pv_lo, PV_lo, PV_hi = pv_indices([0, 0, 0, 0, 0, 10, 10, 20, 30, 30])
print(round(pv_lo, 2), round(PV_lo, 2), round(PV_hi, 2))   # 11.36 0.92 0.65
```

The example is series y7 of Table 2 in the paper. The notebook contains a pairwise reference implementation and checks the fast method against it (also for series with zeros).

## Notebook

`analysis.ipynb`: *Kernel → Restart & Run All*. All imports are in the first code cell.

**Requirements:** Python 3, NumPy, pandas, SciPy, IPython/Jupyter. Tested with Python 3.12.3, NumPy 2.4.4, pandas 3.0.2 and SciPy 1.17.1. A complete run takes roughly 10–15 minutes on one CPU core. The CSV files are written to `output/` next to the notebook.

### Contents

| Notebook section | Content | Output files | Paper |
|---|---|---|---|
| Settings, variability measures | PV<sub>lo</sub>, PV<sub>hi</sub>, CV, SDL, ADM; settings (`N_RUNS`, `N_STEPS`, `SEED`, `DDOF`, `SAVE_SERIES`) | | Section 2 |
| Self-tests | Efficient method versus pairwise definition; reproduction of Table 2 | | Sections 2.2, 3.1 |
| Simulation, summary | Normal, Cauchy, Rare and Bimodal regimes; mean, SD and correlation with ADM | `runs_*.csv`, `summary.csv` | Section 3.2, Table 3, Figs. 2–3 |
| Additional analysis (Rare) | What drives the indices; stable phase without crashes as reference | | Section 3.2 |
| Series length | Variance exponents and window-length dependence | | Section 4 (limitations) |
| Cells A, B | Decline regime: exploration of final density, onset and noise | `decline_grid.csv`, `decline_auc.csv`, `decline_detection.csv` | parameter choice for Section 3.3 |
| Cell C | Decline regime, final analysis (end of series, expanding window) | `decline_paper_*.csv` | Section 3.3, Tables 4–5 |
| Cell D | Crash series versus ordinary fluctuations of unknown amplitude (Gaussian and log-normal) | `crash_vs_ordinary_auc.csv` | not in the paper (log-normal variant) |
| Cell E | Data for the trajectory figure | `decline_trajectory_fig.csv` | Fig. 4 |
| Cell F | Controls: known versus unknown baseline amplitude, crash versus decline | `control_crash_auc.csv`, `control_decline_auc.csv` | Section 3.3, Table 6 |

### Reproducibility

All random numbers come from seeded `numpy.random.default_rng` generators. Every additional analysis uses its own generator, so the main results of Section 3.2 do not change when later cells are modified or skipped. Small deviations in the last digit are possible with other library versions. Setting `SAVE_SERIES = True` also writes every simulated series to the `runs_*.csv` files, which makes them large.

## Notes on use

- The index ignores the temporal order. It does not distinguish a crash from a recovery or a gradual decline from recurrent crashes. Series with trends should be detrended or analysed together with an order-sensitive index.
- For counts of individuals the unit of the zero convention is natural; for biomass or density the unit has to be chosen deliberately.
- Series of different length should be compared with caution (the dependence on series length is similar to that of SDL; see the paper).

## Version history

- **Current version:** The paper concentrates on PV<sub>lo</sub>. The combined index PV of the first version is no longer part of it. Pairs containing a zero are divided by one individual, the algorithm is derived in full, and the analyses cover series length, a gradual decline, separation of crashes from ordinary fluctuations and a comparison with the consecutive disparity index D.
- **First version (2022):** Combined index PV (mean of the indices relative to lower and higher abundances).

## Citation

Please cite the paper above. A `CITATION.cff` file can be added if desired.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

