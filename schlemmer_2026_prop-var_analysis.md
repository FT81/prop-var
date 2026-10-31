# Proportional variability – simulation study

Code for M. Schlemmer, *Proportional variability relative to lower abundances: An interpretable and efficient index of deep lows in population time series* (bioRxiv). The notebook computes CV, SDL, pv<sub>lo</sub>, PV<sub>lo</sub> and PV<sub>hi</sub> for simulated population dynamics and produces the numbers, tables and figure data of the paper. In the decline analyses the consecutive disparity index D (Fernández-Martínez et al. 2018) is included as an order-sensitive reference.

**Contents**
- *Self-tests*: efficient method versus pairwise definition; reproduction of Table 2 (example series).
- *Simulation* (Normal, Cauchy, Rare, Bimodal) and *summary* with the correlation with the average deviation from the mean (ADM): Section 3.2, Table 3, Figs. 2–3.
- *Additional analysis (Rare)* and *series length* (variance exponents): Section 3.2 and the limitations in Section 4.
- *Decline regime*: Cells A and B (exploration of the parameters, not reported in the paper), Cell C (final analysis, Section 3.3, Tables 4–5), Cell D (crash versus ordinary fluctuations, log-normal variant, not used in the paper), Cell E (data for Fig. 4), Cell F (controls with known and unknown baseline amplitude, Table 6).

**Notation**
- PV<sub>hi</sub>: proportional variability relative to the higher value of each pair (Heath 2006).
- pv<sub>lo</sub>: unstandardised proportional variability relative to the lower value of each pair.
- PV<sub>lo</sub> = 1 − 1 / (1 + pv<sub>lo</sub>) = pv<sub>lo</sub> / (1 + pv<sub>lo</sub>).
- Zero counts: in a pair containing a zero, the lower value is treated as 1 (one individual). Such a pair contributes 1 to PV<sub>hi</sub> and the positive value itself to pv<sub>lo</sub>; a pair of two zeros contributes 0 to both.

**Usage:** Kernel → *Restart & Run All* (roughly 10–15 minutes on one CPU core). All imports are in the first code cell; if it is skipped, `NameError`s appear. CSV files are written to the `output` folder next to the notebook.



```python
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
N_RUNS      = 50_000         # simulations per condition
N_STEPS     = 100            # time steps per time series
SEED        = 12345          # for reproducibility
OUT_DIR     = Path("output") # CSV files are written here (relative to the notebook folder)
SAVE_SERIES = False          # True: also write the 100 sorted raw values of each run to the CSV (large files!)
DDOF        = 1              # delta degrees of freedom for the SD in CV and SDL:
                             # 1 = sample SD (as in Table 2), 0 = population SD

# Rare condition after Heath (2006): stable N(100, 5), crash N(10, 5)
RARE_FREQ        = 0.05      # crash frequency
RARE_FIXED_COUNT = True      # True: exactly round(RARE_FREQ * N_STEPS) crashes per series;
                             # False: each time step crashes independently with probability RARE_FREQ

OUT_DIR.mkdir(exist_ok=True)

```


```python
# ------------------------------------------------------------------
# Variability measures
# ------------------------------------------------------------------
def pv_indices(z):
    """Return pv_lo, PV_lo and PV_hi of a time series z (abundances >= 0).

    Efficient method (sorting and cumulative sums, O(n log n)) instead of
    n(n-1)/2 pairwise comparisons. Pairs containing a zero are treated as if
    the lower value were 1 (one individual):
      d_hi(0, z>0) = 1                  -> contribution N0 * Npos
      d_lo(0, z>0) = |z - 0| / 1 = z    -> contribution N0 * sum(z>0)
    Pairs of two zeros contribute 0 to both measures.
    """
    z = np.sort(np.asarray(z, dtype=float))
    n = z.size
    C = n * (n - 1) / 2                         # number of unique pairs
    n0 = int(np.sum(z == 0))                    # number of zeros
    pos = z[z > 0]                              # strictly positive abundances (sorted)
    m = pos.size

    # Comparisons among the positive values
    if m >= 2:
        suffix = np.cumsum(pos[::-1])[::-1]     # suffix[i] = sum of pos[i], ..., pos[m-1]
        sum_pd_lo = np.sum(suffix[1:] / pos[:-1]) - m * (m - 1) / 2
        prefix = np.cumsum(pos)                 # prefix[i] = sum of pos[0], ..., pos[i]
        sum_pd_hi = m * (m - 1) / 2 - np.sum(prefix[:-1] / pos[1:])
    else:
        sum_pd_lo = sum_pd_hi = 0.0

    # Comparisons between zeros and positive values
    sum_pd_lo += n0 * pos.sum()
    sum_pd_hi += n0 * m

    pv_lo = sum_pd_lo / C
    PV_hi = sum_pd_hi / C                       # already bounded between 0 and 1
    PV_lo = 1 - 1 / (1 + pv_lo)
    return pv_lo, PV_lo, PV_hi


def cv_sdl_adm(z):
    """Return CV, SDL and ADM (average deviation from the mean) of a time series z."""
    z = np.asarray(z, dtype=float)
    mean = z.mean()
    cv = z.std(ddof=DDOF) / mean
    # log(0) is undefined: 1 is added to all abundances only if the series contains zeros
    zl = np.log(z + 1) if np.any(z == 0) else np.log(z)
    sdl = zl.std(ddof=DDOF)
    adm = np.mean(np.abs(z - mean))
    return cv, sdl, adm


def all_measures(z):
    cv, sdl, adm = cv_sdl_adm(z)
    pv_lo, PV_lo, PV_hi = pv_indices(z)
    return dict(CV=cv, SDL=sdl, pv_lo=pv_lo, PV_lo=PV_lo, PV_hi=PV_hi, ADM=adm)

```


```python
# ------------------------------------------------------------------
# Self-tests
# ------------------------------------------------------------------
def pv_bruteforce(z):
    """Pairwise definition from the paper (slow, only for checking): returns pv_lo and PV_hi."""
    z = [float(v) for v in z]
    C = len(z) * (len(z) - 1) / 2
    s_lo = s_hi = 0.0
    for a, b in itertools.combinations(z, 2):
        if a == b:
            continue                            # d_lo = d_hi = 0
        s_hi += abs(a - b) / max(a, b)
        if a > 0 and b > 0:
            s_lo += abs(a - b) / min(a, b)
        else:
            s_lo += abs(a - b)                  # pair with a zero: lower value treated as 1
    return s_lo / C, s_hi / C


# 1) Is the efficient method identical to the pairwise definition (also with zeros)?
_rng = np.random.default_rng(0)
_maxdiff = 0.0
for _ in range(200):
    _z = _rng.integers(0, 30, size=int(_rng.integers(5, 40)))
    _fast = pv_indices(_z)
    _slow = pv_bruteforce(_z)
    _maxdiff = max(_maxdiff, abs(_fast[0] - _slow[0]), abs(_fast[2] - _slow[1]))
print(f"max. deviation efficient vs. pairwise (200 random series, with zeros): {_maxdiff:.2e}")
assert _maxdiff < 1e-9

# 2) Reproduce Table 2 of the paper: CV, SDL, pv_lo, PV_lo, PV_hi
#    Reference values as printed in the paper (rounded to 2 decimals).
TABLE2 = {
    "y1":  ((1, 7, 8, 9, 10, 11, 12, 13, 14, 15),  (0.41, 0.79, 2.32,  0.70, 0.39)),
    "y2":  ((5, 5, 6, 6, 7, 7, 8, 8, 8, 40),       (1.06, 0.60, 1.25,  0.56, 0.32)),
    "y3":  ((5, 5, 8, 8, 10, 10, 12, 12, 15, 15),  (0.36, 0.40, 0.67,  0.40, 0.34)),
    "y4":  ((0, 7, 8, 9, 10, 11, 12, 13, 14, 15),  (0.44, 0.81, 2.52,  0.72, 0.41)),
    "y5":  ((0, 0, 1, 1, 1, 1, 2, 2, 2, 2),        (0.66, 0.42, 0.89,  0.47, 0.53)),
    "y6":  ((0, 0, 1, 3, 10, 12, 15, 17, 20, 22),  (0.85, 1.26, 7.23,  0.88, 0.70)),
    "y7":  ((0, 0, 0, 0, 0, 10, 10, 20, 30, 30),   (1.25, 1.59, 11.36, 0.92, 0.65)),
    "y8":  ((0, 0, 0, 0, 0, 1, 1, 2, 3, 3),        (1.25, 0.60, 1.36,  0.58, 0.65)),
    "y9":  ((0, 0, 0, 0, 0, 0, 0, 0, 1, 99),       (3.13, 1.45, 19.96, 0.95, 0.38)),
    "y10": ((0, 0, 0, 0, 0, 0, 0, 0, 1, 2),        (2.25, 0.39, 0.56,  0.36, 0.37)),
}
TABLE2_COLS = ["CV", "SDL", "pv_lo", "PV_lo", "PV_hi"]

_rows, _problems = [], []
for name, (series, ref) in TABLE2.items():
    cv, sdl, _ = cv_sdl_adm(series)
    pv_lo, PV_lo, PV_hi = pv_indices(series)
    calc = np.array([cv, sdl, pv_lo, PV_lo, PV_hi])
    # reference values are rounded to 2 decimals -> tolerance of half a unit in the last digit
    close = np.abs(calc - np.array(ref)) <= 0.005 + 1e-9
    if DDOF != 1:
        close[:2] = True                        # Table 2 uses the sample SD; CV and SDL not comparable
    _rows.append([name, *np.round(calc, 2), bool(close.all())])
    _problems += [f"{name}: {c} = {v:.4f} (paper: {r})"
                  for c, v, r, ok in zip(TABLE2_COLS, calc, ref, close) if not ok]

table2 = pd.DataFrame(_rows, columns=["series"] + TABLE2_COLS + ["matches paper"])
display(table2)
if DDOF != 1:
    print("Note: DDOF != 1, the CV and SDL columns were not compared with the paper.")
assert not _problems, "Table 2 differs from the paper:\n" + "\n".join(_problems)

# LaTeX rows for Table 2 (to copy into the paper)
print("\nLaTeX rows for Table 2:")
for name, (series, _) in TABLE2.items():
    row = table2.loc[table2.series == name, TABLE2_COLS].iloc[0]
    label = name[0] + "_" + (name[1:] if len(name) == 2 else "{" + name[1:] + "}")
    vec = ", ".join(str(v) for v in series)
    vals = " & ".join(f"{v:.2f}" for v in row)
    print(f"${label} = ({vec})$ & {vals} \\\\")

```

    max. deviation efficient vs. pairwise (200 random series, with zeros): 7.11e-15
    max. deviation efficient vs. pairwise (200 random series, with zeros): 7.11e-15
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>series</th>
      <th>CV</th>
      <th>SDL</th>
      <th>pv_lo</th>
      <th>PV_lo</th>
      <th>PV_hi</th>
      <th>matches paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>y1</td>
      <td>0.41</td>
      <td>0.79</td>
      <td>2.32</td>
      <td>0.70</td>
      <td>0.39</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>y2</td>
      <td>1.06</td>
      <td>0.60</td>
      <td>1.25</td>
      <td>0.56</td>
      <td>0.32</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>y3</td>
      <td>0.36</td>
      <td>0.40</td>
      <td>0.67</td>
      <td>0.40</td>
      <td>0.34</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>y4</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>2.52</td>
      <td>0.72</td>
      <td>0.41</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>y5</td>
      <td>0.66</td>
      <td>0.42</td>
      <td>0.89</td>
      <td>0.47</td>
      <td>0.53</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>y6</td>
      <td>0.85</td>
      <td>1.26</td>
      <td>7.23</td>
      <td>0.88</td>
      <td>0.70</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>y7</td>
      <td>1.25</td>
      <td>1.59</td>
      <td>11.36</td>
      <td>0.92</td>
      <td>0.65</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>y8</td>
      <td>1.25</td>
      <td>0.60</td>
      <td>1.36</td>
      <td>0.58</td>
      <td>0.65</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>y9</td>
      <td>3.13</td>
      <td>1.45</td>
      <td>19.96</td>
      <td>0.95</td>
      <td>0.38</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>y10</td>
      <td>2.25</td>
      <td>0.39</td>
      <td>0.56</td>
      <td>0.36</td>
      <td>0.37</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    LaTeX rows for Table 2:
    $y_1 = (1, 7, 8, 9, 10, 11, 12, 13, 14, 15)$ & 0.41 & 0.79 & 2.32 & 0.70 & 0.39 \\
    $y_2 = (5, 5, 6, 6, 7, 7, 8, 8, 8, 40)$ & 1.06 & 0.60 & 1.25 & 0.56 & 0.32 \\
    $y_3 = (5, 5, 8, 8, 10, 10, 12, 12, 15, 15)$ & 0.36 & 0.40 & 0.67 & 0.40 & 0.34 \\
    $y_4 = (0, 7, 8, 9, 10, 11, 12, 13, 14, 15)$ & 0.44 & 0.81 & 2.52 & 0.72 & 0.41 \\
    $y_5 = (0, 0, 1, 1, 1, 1, 2, 2, 2, 2)$ & 0.66 & 0.42 & 0.89 & 0.47 & 0.53 \\
    $y_6 = (0, 0, 1, 3, 10, 12, 15, 17, 20, 22)$ & 0.85 & 1.26 & 7.23 & 0.88 & 0.70 \\
    $y_7 = (0, 0, 0, 0, 0, 10, 10, 20, 30, 30)$ & 1.25 & 1.59 & 11.36 & 0.92 & 0.65 \\
    $y_8 = (0, 0, 0, 0, 0, 1, 1, 2, 3, 3)$ & 1.25 & 0.60 & 1.36 & 0.58 & 0.65 \\
    $y_9 = (0, 0, 0, 0, 0, 0, 0, 0, 1, 99)$ & 3.13 & 1.45 & 19.96 & 0.95 & 0.38 \\
    $y_{10} = (0, 0, 0, 0, 0, 0, 0, 0, 1, 2)$ & 2.25 & 0.39 & 0.56 & 0.36 & 0.37 \\
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>series</th>
      <th>CV</th>
      <th>SDL</th>
      <th>pv_lo</th>
      <th>PV_lo</th>
      <th>PV_hi</th>
      <th>matches paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>y1</td>
      <td>0.41</td>
      <td>0.79</td>
      <td>2.32</td>
      <td>0.70</td>
      <td>0.39</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>y2</td>
      <td>1.06</td>
      <td>0.60</td>
      <td>1.25</td>
      <td>0.56</td>
      <td>0.32</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>y3</td>
      <td>0.36</td>
      <td>0.40</td>
      <td>0.67</td>
      <td>0.40</td>
      <td>0.34</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>y4</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>2.52</td>
      <td>0.72</td>
      <td>0.41</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>y5</td>
      <td>0.66</td>
      <td>0.42</td>
      <td>0.89</td>
      <td>0.47</td>
      <td>0.53</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>y6</td>
      <td>0.85</td>
      <td>1.26</td>
      <td>7.23</td>
      <td>0.88</td>
      <td>0.70</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>y7</td>
      <td>1.25</td>
      <td>1.59</td>
      <td>11.36</td>
      <td>0.92</td>
      <td>0.65</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>y8</td>
      <td>1.25</td>
      <td>0.60</td>
      <td>1.36</td>
      <td>0.58</td>
      <td>0.65</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>y9</td>
      <td>3.13</td>
      <td>1.45</td>
      <td>19.96</td>
      <td>0.95</td>
      <td>0.38</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>y10</td>
      <td>2.25</td>
      <td>0.39</td>
      <td>0.56</td>
      <td>0.36</td>
      <td>0.37</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    
    LaTeX rows for Table 2:
    $y_1 = (1, 7, 8, 9, 10, 11, 12, 13, 14, 15)$ & 0.41 & 0.79 & 2.32 & 0.70 & 0.39 \\
    $y_2 = (5, 5, 6, 6, 7, 7, 8, 8, 8, 40)$ & 1.06 & 0.60 & 1.25 & 0.56 & 0.32 \\
    $y_3 = (5, 5, 8, 8, 10, 10, 12, 12, 15, 15)$ & 0.36 & 0.40 & 0.67 & 0.40 & 0.34 \\
    $y_4 = (0, 7, 8, 9, 10, 11, 12, 13, 14, 15)$ & 0.44 & 0.81 & 2.52 & 0.72 & 0.41 \\
    $y_5 = (0, 0, 1, 1, 1, 1, 2, 2, 2, 2)$ & 0.66 & 0.42 & 0.89 & 0.47 & 0.53 \\
    $y_6 = (0, 0, 1, 3, 10, 12, 15, 17, 20, 22)$ & 0.85 & 1.26 & 7.23 & 0.88 & 0.70 \\
    $y_7 = (0, 0, 0, 0, 0, 10, 10, 20, 30, 30)$ & 1.25 & 1.59 & 11.36 & 0.92 & 0.65 \\
    $y_8 = (0, 0, 0, 0, 0, 1, 1, 2, 3, 3)$ & 1.25 & 0.60 & 1.36 & 0.58 & 0.65 \\
    $y_9 = (0, 0, 0, 0, 0, 0, 0, 0, 1, 99)$ & 3.13 & 1.45 & 19.96 & 0.95 & 0.38 \\
    $y_{10} = (0, 0, 0, 0, 0, 0, 0, 0, 1, 2)$ & 2.25 & 0.39 & 0.56 & 0.36 & 0.37 \\
    


```python
# ------------------------------------------------------------------
# Generators for the simulated time series
# ------------------------------------------------------------------
def draw_normal_pos(rng, mean, sd, size):
    """Rounded normal distribution; values < 1 are redrawn (abundances = natural numbers)."""
    x = np.round(rng.normal(mean, sd, size))
    bad = x < 1
    while bad.any():
        x[bad] = np.round(rng.normal(mean, sd, bad.sum()))
        bad = x < 1
    return x


def gen_normal(rng):
    return draw_normal_pos(rng, 1000, 150, N_STEPS)


def gen_cauchy(rng):
    # Cauchy (location 100, scale 5), rounded; "natural numbers": absolute value, lower bound 1
    x = np.round(100 + 5 * rng.standard_cauchy(N_STEPS))
    return np.maximum(np.abs(x), 1)


def gen_rare(rng):
    # stable N(100, 5); crash N(10, 5) with frequency RARE_FREQ (Heath 2006)
    if RARE_FIXED_COUNT:
        n_crash = int(round(RARE_FREQ * N_STEPS))
        crash = np.zeros(N_STEPS, dtype=bool)
        crash[rng.choice(N_STEPS, n_crash, replace=False)] = True
    else:
        crash = rng.random(N_STEPS) < RARE_FREQ
    x = np.empty(N_STEPS)
    x[~crash] = draw_normal_pos(rng, 100, 5, (~crash).sum())
    x[crash] = draw_normal_pos(rng, 10, 5, crash.sum())
    return x


def gen_bimodal(rng):
    # first half N(100, 10), second half N(10, 1)
    half = N_STEPS // 2
    return np.concatenate([draw_normal_pos(rng, 100, 10, half),
                           draw_normal_pos(rng, 10, 1, N_STEPS - half)])


CONDITIONS = {"Normal": gen_normal, "Cauchy": gen_cauchy, "Rare": gen_rare, "Bimodal": gen_bimodal}

```


```python
# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
rng = np.random.default_rng(SEED)
runs = {}

for cond, gen in CONDITIONS.items():
    t0 = time.time()
    records, series = [], []
    for _ in range(N_RUNS):
        z = gen(rng)
        records.append(all_measures(z))
        if SAVE_SERIES:
            series.append(np.sort(z))
    df = pd.DataFrame.from_records(records)
    if SAVE_SERIES:
        df = pd.concat([df, pd.DataFrame(series, columns=[f"sortval{i}" for i in range(1, N_STEPS + 1)])], axis=1)
    runs[cond] = df
    df.to_csv(OUT_DIR / f"runs_{cond}.csv", index=False)
    print(f"{cond:8s}: {N_RUNS} runs in {time.time() - t0:5.1f} s  ->  {OUT_DIR / f'runs_{cond}.csv'}")

```

    Normal  : 50000 runs in   4.9 s  ->  output\runs_Normal.csv
    Normal  : 50000 runs in   4.9 s  ->  output\runs_Normal.csv
    Cauchy  : 50000 runs in   5.5 s  ->  output\runs_Cauchy.csv
    Cauchy  : 50000 runs in   5.5 s  ->  output\runs_Cauchy.csv
    Rare    : 50000 runs in   7.5 s  ->  output\runs_Rare.csv
    Rare    : 50000 runs in   7.5 s  ->  output\runs_Rare.csv
    Bimodal : 50000 runs in   6.4 s  ->  output\runs_Bimodal.csv
    Bimodal : 50000 runs in   6.4 s  ->  output\runs_Bimodal.csv
    


```python
# ------------------------------------------------------------------
# Summary: mean, SD and correlation with ADM (Pearson, Spearman, Kendall)
# ------------------------------------------------------------------
measures = ["CV", "SDL", "pv_lo", "PV_lo", "PV_hi"]
rows = []
for cond, df in runs.items():
    for m in measures:
        r, _   = stats.pearsonr(df[m], df["ADM"])
        rho, _ = stats.spearmanr(df[m], df["ADM"])
        tau, _ = stats.kendalltau(df[m], df["ADM"])
        rows.append(dict(condition=cond, measure=m, mean=df[m].mean(), sd=df[m].std(),
                         pearson_ADM=r, spearman_ADM=rho, kendall_ADM=tau))
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
display(summary.round(3))

# Compact views: means and SDs (bar chart of the variability indices), correlations with ADM
display(summary.pivot(index="measure", columns="condition", values="mean").loc[measures, list(CONDITIONS)].round(3))
display(summary.pivot(index="measure", columns="condition", values="sd").loc[measures, list(CONDITIONS)].round(3))
display(summary.pivot(index="measure", columns="condition", values="pearson_ADM").loc[measures, list(CONDITIONS)].round(3))
display(summary.pivot(index="measure", columns="condition", values="spearman_ADM").loc[measures, list(CONDITIONS)].round(3))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condition</th>
      <th>measure</th>
      <th>mean</th>
      <th>sd</th>
      <th>pearson_ADM</th>
      <th>spearman_ADM</th>
      <th>kendall_ADM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Normal</td>
      <td>CV</td>
      <td>0.150</td>
      <td>0.011</td>
      <td>0.917</td>
      <td>0.910</td>
      <td>0.740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Normal</td>
      <td>SDL</td>
      <td>0.154</td>
      <td>0.012</td>
      <td>0.863</td>
      <td>0.858</td>
      <td>0.672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Normal</td>
      <td>pv_lo</td>
      <td>0.200</td>
      <td>0.018</td>
      <td>0.917</td>
      <td>0.913</td>
      <td>0.745</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Normal</td>
      <td>PV_lo</td>
      <td>0.166</td>
      <td>0.012</td>
      <td>0.918</td>
      <td>0.913</td>
      <td>0.745</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Normal</td>
      <td>PV_hi</td>
      <td>0.151</td>
      <td>0.010</td>
      <td>0.948</td>
      <td>0.943</td>
      <td>0.795</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cauchy</td>
      <td>CV</td>
      <td>0.951</td>
      <td>1.351</td>
      <td>0.117</td>
      <td>0.976</td>
      <td>0.868</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cauchy</td>
      <td>SDL</td>
      <td>0.366</td>
      <td>0.132</td>
      <td>0.090</td>
      <td>0.707</td>
      <td>0.537</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cauchy</td>
      <td>pv_lo</td>
      <td>1.344</td>
      <td>30.939</td>
      <td>0.998</td>
      <td>0.696</td>
      <td>0.540</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cauchy</td>
      <td>PV_lo</td>
      <td>0.379</td>
      <td>0.177</td>
      <td>0.066</td>
      <td>0.696</td>
      <td>0.540</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cauchy</td>
      <td>PV_hi</td>
      <td>0.171</td>
      <td>0.027</td>
      <td>0.016</td>
      <td>0.602</td>
      <td>0.437</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Rare</td>
      <td>CV</td>
      <td>0.212</td>
      <td>0.005</td>
      <td>0.702</td>
      <td>0.686</td>
      <td>0.496</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rare</td>
      <td>SDL</td>
      <td>0.540</td>
      <td>0.063</td>
      <td>0.518</td>
      <td>0.516</td>
      <td>0.360</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Rare</td>
      <td>pv_lo</td>
      <td>1.287</td>
      <td>0.593</td>
      <td>0.398</td>
      <td>0.482</td>
      <td>0.334</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Rare</td>
      <td>PV_lo</td>
      <td>0.540</td>
      <td>0.091</td>
      <td>0.464</td>
      <td>0.482</td>
      <td>0.334</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Rare</td>
      <td>PV_hi</td>
      <td>0.136</td>
      <td>0.004</td>
      <td>0.878</td>
      <td>0.869</td>
      <td>0.683</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Bimodal</td>
      <td>CV</td>
      <td>0.832</td>
      <td>0.004</td>
      <td>0.612</td>
      <td>0.593</td>
      <td>0.419</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Bimodal</td>
      <td>SDL</td>
      <td>1.162</td>
      <td>0.010</td>
      <td>0.752</td>
      <td>0.734</td>
      <td>0.540</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Bimodal</td>
      <td>pv_lo</td>
      <td>4.664</td>
      <td>0.107</td>
      <td>0.742</td>
      <td>0.723</td>
      <td>0.531</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Bimodal</td>
      <td>PV_lo</td>
      <td>0.823</td>
      <td>0.003</td>
      <td>0.742</td>
      <td>0.723</td>
      <td>0.531</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Bimodal</td>
      <td>PV_hi</td>
      <td>0.505</td>
      <td>0.004</td>
      <td>0.142</td>
      <td>0.135</td>
      <td>0.090</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condition</th>
      <th>measure</th>
      <th>mean</th>
      <th>sd</th>
      <th>pearson_ADM</th>
      <th>spearman_ADM</th>
      <th>kendall_ADM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Normal</td>
      <td>CV</td>
      <td>0.150</td>
      <td>0.011</td>
      <td>0.917</td>
      <td>0.910</td>
      <td>0.740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Normal</td>
      <td>SDL</td>
      <td>0.154</td>
      <td>0.012</td>
      <td>0.863</td>
      <td>0.858</td>
      <td>0.672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Normal</td>
      <td>pv_lo</td>
      <td>0.200</td>
      <td>0.018</td>
      <td>0.917</td>
      <td>0.913</td>
      <td>0.745</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Normal</td>
      <td>PV_lo</td>
      <td>0.166</td>
      <td>0.012</td>
      <td>0.918</td>
      <td>0.913</td>
      <td>0.745</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Normal</td>
      <td>PV_hi</td>
      <td>0.151</td>
      <td>0.010</td>
      <td>0.948</td>
      <td>0.943</td>
      <td>0.795</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cauchy</td>
      <td>CV</td>
      <td>0.951</td>
      <td>1.351</td>
      <td>0.117</td>
      <td>0.976</td>
      <td>0.868</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cauchy</td>
      <td>SDL</td>
      <td>0.366</td>
      <td>0.132</td>
      <td>0.090</td>
      <td>0.707</td>
      <td>0.537</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cauchy</td>
      <td>pv_lo</td>
      <td>1.344</td>
      <td>30.939</td>
      <td>0.998</td>
      <td>0.696</td>
      <td>0.540</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cauchy</td>
      <td>PV_lo</td>
      <td>0.379</td>
      <td>0.177</td>
      <td>0.066</td>
      <td>0.696</td>
      <td>0.540</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cauchy</td>
      <td>PV_hi</td>
      <td>0.171</td>
      <td>0.027</td>
      <td>0.016</td>
      <td>0.602</td>
      <td>0.437</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Rare</td>
      <td>CV</td>
      <td>0.212</td>
      <td>0.005</td>
      <td>0.702</td>
      <td>0.686</td>
      <td>0.496</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rare</td>
      <td>SDL</td>
      <td>0.540</td>
      <td>0.063</td>
      <td>0.518</td>
      <td>0.516</td>
      <td>0.360</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Rare</td>
      <td>pv_lo</td>
      <td>1.287</td>
      <td>0.593</td>
      <td>0.398</td>
      <td>0.482</td>
      <td>0.334</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Rare</td>
      <td>PV_lo</td>
      <td>0.540</td>
      <td>0.091</td>
      <td>0.464</td>
      <td>0.482</td>
      <td>0.334</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Rare</td>
      <td>PV_hi</td>
      <td>0.136</td>
      <td>0.004</td>
      <td>0.878</td>
      <td>0.869</td>
      <td>0.683</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Bimodal</td>
      <td>CV</td>
      <td>0.832</td>
      <td>0.004</td>
      <td>0.612</td>
      <td>0.593</td>
      <td>0.419</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Bimodal</td>
      <td>SDL</td>
      <td>1.162</td>
      <td>0.010</td>
      <td>0.752</td>
      <td>0.734</td>
      <td>0.540</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Bimodal</td>
      <td>pv_lo</td>
      <td>4.664</td>
      <td>0.107</td>
      <td>0.742</td>
      <td>0.723</td>
      <td>0.531</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Bimodal</td>
      <td>PV_lo</td>
      <td>0.823</td>
      <td>0.003</td>
      <td>0.742</td>
      <td>0.723</td>
      <td>0.531</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Bimodal</td>
      <td>PV_hi</td>
      <td>0.505</td>
      <td>0.004</td>
      <td>0.142</td>
      <td>0.135</td>
      <td>0.090</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>Normal</th>
      <th>Cauchy</th>
      <th>Rare</th>
      <th>Bimodal</th>
    </tr>
    <tr>
      <th>measure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.150</td>
      <td>0.951</td>
      <td>0.212</td>
      <td>0.832</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.154</td>
      <td>0.366</td>
      <td>0.540</td>
      <td>1.162</td>
    </tr>
    <tr>
      <th>pv_lo</th>
      <td>0.200</td>
      <td>1.344</td>
      <td>1.287</td>
      <td>4.664</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.166</td>
      <td>0.379</td>
      <td>0.540</td>
      <td>0.823</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.151</td>
      <td>0.171</td>
      <td>0.136</td>
      <td>0.505</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>Normal</th>
      <th>Cauchy</th>
      <th>Rare</th>
      <th>Bimodal</th>
    </tr>
    <tr>
      <th>measure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.150</td>
      <td>0.951</td>
      <td>0.212</td>
      <td>0.832</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.154</td>
      <td>0.366</td>
      <td>0.540</td>
      <td>1.162</td>
    </tr>
    <tr>
      <th>pv_lo</th>
      <td>0.200</td>
      <td>1.344</td>
      <td>1.287</td>
      <td>4.664</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.166</td>
      <td>0.379</td>
      <td>0.540</td>
      <td>0.823</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.151</td>
      <td>0.171</td>
      <td>0.136</td>
      <td>0.505</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>Normal</th>
      <th>Cauchy</th>
      <th>Rare</th>
      <th>Bimodal</th>
    </tr>
    <tr>
      <th>measure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.011</td>
      <td>1.351</td>
      <td>0.005</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.012</td>
      <td>0.132</td>
      <td>0.063</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>pv_lo</th>
      <td>0.018</td>
      <td>30.939</td>
      <td>0.593</td>
      <td>0.107</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.012</td>
      <td>0.177</td>
      <td>0.091</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.010</td>
      <td>0.027</td>
      <td>0.004</td>
      <td>0.004</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>Normal</th>
      <th>Cauchy</th>
      <th>Rare</th>
      <th>Bimodal</th>
    </tr>
    <tr>
      <th>measure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.011</td>
      <td>1.351</td>
      <td>0.005</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.012</td>
      <td>0.132</td>
      <td>0.063</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>pv_lo</th>
      <td>0.018</td>
      <td>30.939</td>
      <td>0.593</td>
      <td>0.107</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.012</td>
      <td>0.177</td>
      <td>0.091</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.010</td>
      <td>0.027</td>
      <td>0.004</td>
      <td>0.004</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>Normal</th>
      <th>Cauchy</th>
      <th>Rare</th>
      <th>Bimodal</th>
    </tr>
    <tr>
      <th>measure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.917</td>
      <td>0.117</td>
      <td>0.702</td>
      <td>0.612</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.863</td>
      <td>0.090</td>
      <td>0.518</td>
      <td>0.752</td>
    </tr>
    <tr>
      <th>pv_lo</th>
      <td>0.917</td>
      <td>0.998</td>
      <td>0.398</td>
      <td>0.742</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.918</td>
      <td>0.066</td>
      <td>0.464</td>
      <td>0.742</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.948</td>
      <td>0.016</td>
      <td>0.878</td>
      <td>0.142</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>Normal</th>
      <th>Cauchy</th>
      <th>Rare</th>
      <th>Bimodal</th>
    </tr>
    <tr>
      <th>measure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.910</td>
      <td>0.976</td>
      <td>0.686</td>
      <td>0.593</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.858</td>
      <td>0.707</td>
      <td>0.516</td>
      <td>0.734</td>
    </tr>
    <tr>
      <th>pv_lo</th>
      <td>0.913</td>
      <td>0.696</td>
      <td>0.482</td>
      <td>0.723</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.913</td>
      <td>0.696</td>
      <td>0.482</td>
      <td>0.723</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.943</td>
      <td>0.602</td>
      <td>0.869</td>
      <td>0.135</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>Normal</th>
      <th>Cauchy</th>
      <th>Rare</th>
      <th>Bimodal</th>
    </tr>
    <tr>
      <th>measure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.917</td>
      <td>0.117</td>
      <td>0.702</td>
      <td>0.612</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.863</td>
      <td>0.090</td>
      <td>0.518</td>
      <td>0.752</td>
    </tr>
    <tr>
      <th>pv_lo</th>
      <td>0.917</td>
      <td>0.998</td>
      <td>0.398</td>
      <td>0.742</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.918</td>
      <td>0.066</td>
      <td>0.464</td>
      <td>0.742</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.948</td>
      <td>0.016</td>
      <td>0.878</td>
      <td>0.142</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>condition</th>
      <th>Normal</th>
      <th>Cauchy</th>
      <th>Rare</th>
      <th>Bimodal</th>
    </tr>
    <tr>
      <th>measure</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.910</td>
      <td>0.976</td>
      <td>0.686</td>
      <td>0.593</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.858</td>
      <td>0.707</td>
      <td>0.516</td>
      <td>0.734</td>
    </tr>
    <tr>
      <th>pv_lo</th>
      <td>0.913</td>
      <td>0.696</td>
      <td>0.482</td>
      <td>0.723</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.913</td>
      <td>0.696</td>
      <td>0.482</td>
      <td>0.723</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.943</td>
      <td>0.602</td>
      <td>0.869</td>
      <td>0.135</td>
    </tr>
  </tbody>
</table>
</div>



```python
# ------------------------------------------------------------------
# Additional analysis (Rare regime): what drives the indices, and what
# would they be without crashes?
# Requires the cells above (all_measures, draw_normal_pos, N_STEPS, np, pd, stats).
# A separate random number generator (rng_extra) is used, so the main
# results above are not affected.
# ------------------------------------------------------------------
N_EXTRA   = 20_000                       # simulated series (smaller than N_RUNS, sufficient for rank correlations)
N_CRASHES = round(RARE_FREQ * N_STEPS)   # 5 crashes per series, as in gen_rare()
rng_extra = np.random.default_rng(7)

# --- Part A: between-run drivers ----------------------------------------------
# For every simulated Rare series we record two features of the series:
#   sd_stable      : SD of the stable (non-crash) abundances -> scatter of the stable phase
#   mean_inv_crash : mean of 1/z over the crashed abundances -> proportional depth of the
#                    crashes (low values receive a large weight, as in pv_lo, Eq. 10 of the paper)
rows = []
for _ in range(N_EXTRA):
    crash = np.zeros(N_STEPS, dtype=bool)
    crash[rng_extra.choice(N_STEPS, N_CRASHES, replace=False)] = True
    z = np.empty(N_STEPS)
    z[~crash] = draw_normal_pos(rng_extra, 100, 5, (~crash).sum())    # stable phase  N(100, 5)
    z[crash]  = draw_normal_pos(rng_extra, 10, 5, crash.sum())        # crashes       N(10, 5)
    row = all_measures(z)
    row["mean_inv_crash"] = np.mean(1 / z[crash])
    row["sd_stable"]      = z[~crash].std(ddof=1)
    rows.append(row)
rare_extra = pd.DataFrame(rows)

# Spearman correlation of each index (and of ADM) with the two features, across the series.
# Spearman uses ranks only, so pv_lo and PV_lo give identical values.
drivers = pd.DataFrame({
    m: {f: stats.spearmanr(rare_extra[m], rare_extra[f])[0] for f in ["sd_stable", "mean_inv_crash"]}
    for m in ["CV", "SDL", "PV_lo", "PV_hi", "ADM"]
}).T
print("Spearman correlation of each index with features of the Rare series:")
display(drivers.round(2))

# --- Part B: reference without crashes ----------------------------------------
# Same stable population N(100, 5) as in the Rare regime, but without any crash.
# This is NOT the Normal regime of the paper (mean 1000, SD 150): the relative fluctuation
# is 5 % here (CV about 0.05) instead of 15 %, so the index values are lower.
# It shows how much of the Rare-regime scores is caused by the crashes themselves.
no_crash = pd.DataFrame([all_measures(draw_normal_pos(rng_extra, 100, 5, N_STEPS))
                         for _ in range(N_EXTRA)])
reference = pd.DataFrame({
    "no crash": no_crash[["CV", "SDL", "PV_lo", "PV_hi"]].mean(),
    "Rare (5 crashes)": runs["Rare"][["CV", "SDL", "PV_lo", "PV_hi"]].mean(),
})
reference["factor"] = reference["Rare (5 crashes)"] / reference["no crash"]
print("\nMean index values without crashes and in the Rare regime:")
display(reference.round(3))

```

    Spearman correlation of each index with features of the Rare series:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sd_stable</th>
      <th>mean_inv_crash</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.18</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.01</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.02</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.84</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>ADM</th>
      <td>0.66</td>
      <td>0.47</td>
    </tr>
  </tbody>
</table>
</div>


    Spearman correlation of each index with features of the Rare series:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sd_stable</th>
      <th>mean_inv_crash</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.18</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.01</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.02</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.84</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>ADM</th>
      <td>0.66</td>
      <td>0.47</td>
    </tr>
  </tbody>
</table>
</div>


    
    Mean index values without crashes and in the Rare regime:
    
    Mean index values without crashes and in the Rare regime:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no crash</th>
      <th>Rare (5 crashes)</th>
      <th>factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.050</td>
      <td>0.212</td>
      <td>4.242</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.050</td>
      <td>0.540</td>
      <td>10.776</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.056</td>
      <td>0.540</td>
      <td>9.669</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.054</td>
      <td>0.136</td>
      <td>2.504</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no crash</th>
      <th>Rare (5 crashes)</th>
      <th>factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CV</th>
      <td>0.050</td>
      <td>0.212</td>
      <td>4.242</td>
    </tr>
    <tr>
      <th>SDL</th>
      <td>0.050</td>
      <td>0.540</td>
      <td>10.776</td>
    </tr>
    <tr>
      <th>PV_lo</th>
      <td>0.056</td>
      <td>0.540</td>
      <td>9.669</td>
    </tr>
    <tr>
      <th>PV_hi</th>
      <td>0.054</td>
      <td>0.136</td>
      <td>2.504</td>
    </tr>
  </tbody>
</table>
</div>



```python
# ------------------------------------------------------------------
# OPTIONAL extra cell: dependence of the indices on the length of the time series
# ("more time - more variation"; Heath 2006, Inchausti & Halley 2002).
# Requires the cells above (all_measures, CONDITIONS, N_STEPS, np, pd).
# A separate random number generator is used, so the main results are not affected.
# ------------------------------------------------------------------
# Idea: a good index should not increase just because a series is observed for longer.
# For every simulated series we compute each index in windows of length k (k = 3 ... 100),
# average it over several random windows per k, and fit a straight line to
# ln(index) against ln(k). The slope is the variance exponent gamma:
#   gamma = 0  : index does not depend on series length (white-noise-like behaviour)
#   gamma > 0  : index increases with series length ("more time - more variation")
# Only the stationary regimes are used (Bimodal is non-stationary by construction).
N_SERIES   = 1000                                   # series per regime (as in Heath 2006)
N_WINDOWS  = 15                                     # random windows per window length k
KS         = [3, 4, 6, 9, 13, 20, 30, 45, 70, 100]  # window lengths (time steps)
MEASURES   = ["CV", "SDL", "PV_lo", "PV_hi"]
rng_exp    = np.random.default_rng(2024)

def variance_exponents(gen, n_series=N_SERIES):
    gammas = {m: [] for m in MEASURES}
    for _ in range(n_series):
        z = gen(rng_exp)                            # one simulated series of N_STEPS values
        mean_index = {m: [] for m in MEASURES}      # index averaged over windows, per k
        for k in KS:
            starts = rng_exp.integers(0, N_STEPS - k + 1, size=N_WINDOWS) if k < N_STEPS else [0]
            vals = [all_measures(z[s:s + k]) for s in starts]
            for m in MEASURES:
                mean_index[m].append(np.mean([v[m] for v in vals]))
        for m in MEASURES:                          # slope of ln(index) vs ln(k)
            gammas[m].append(np.polyfit(np.log(KS), np.log(mean_index[m]), 1)[0])
    return pd.DataFrame(gammas)

rows = {}
for cond in ["Normal", "Cauchy", "Rare"]:
    g = variance_exponents(CONDITIONS[cond])
    rows[cond] = {f"{m} mean": g[m].mean() for m in MEASURES} | {f"{m} SD": g[m].std() for m in MEASURES}
print("Variance exponents gamma (mean and SD across simulated series):")
display(pd.DataFrame(rows).T.round(3))
```

    Variance exponents gamma (mean and SD across simulated series):
    Variance exponents gamma (mean and SD across simulated series):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CV mean</th>
      <th>SDL mean</th>
      <th>PV_lo mean</th>
      <th>PV_hi mean</th>
      <th>CV SD</th>
      <th>SDL SD</th>
      <th>PV_lo SD</th>
      <th>PV_hi SD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>0.030</td>
      <td>0.035</td>
      <td>0.015</td>
      <td>0.003</td>
      <td>0.027</td>
      <td>0.029</td>
      <td>0.026</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>Cauchy</th>
      <td>0.334</td>
      <td>0.174</td>
      <td>0.185</td>
      <td>0.008</td>
      <td>0.254</td>
      <td>0.093</td>
      <td>0.124</td>
      <td>0.046</td>
    </tr>
    <tr>
      <th>Rare</th>
      <td>0.145</td>
      <td>0.287</td>
      <td>0.359</td>
      <td>0.022</td>
      <td>0.093</td>
      <td>0.121</td>
      <td>0.103</td>
      <td>0.077</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CV mean</th>
      <th>SDL mean</th>
      <th>PV_lo mean</th>
      <th>PV_hi mean</th>
      <th>CV SD</th>
      <th>SDL SD</th>
      <th>PV_lo SD</th>
      <th>PV_hi SD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>0.030</td>
      <td>0.035</td>
      <td>0.015</td>
      <td>0.003</td>
      <td>0.027</td>
      <td>0.029</td>
      <td>0.026</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>Cauchy</th>
      <td>0.334</td>
      <td>0.174</td>
      <td>0.185</td>
      <td>0.008</td>
      <td>0.254</td>
      <td>0.093</td>
      <td>0.124</td>
      <td>0.046</td>
    </tr>
    <tr>
      <th>Rare</th>
      <td>0.145</td>
      <td>0.287</td>
      <td>0.359</td>
      <td>0.022</td>
      <td>0.093</td>
      <td>0.121</td>
      <td>0.103</td>
      <td>0.077</td>
    </tr>
  </tbody>
</table>
</div>


## Decline regime: parameter exploration (Cells A and B)

Gradual-decline regime: the population fluctuates around 100 and, after a fraction `onset` of the series, declines gradually to `100 / factor` at the last time step (`"exp"`: constant per-capita rate, `"lin"`: linear). Scatter around the path is Gaussian with relative SD `noise` (values < 1 are redrawn, as elsewhere in this notebook). Cells A and B explore combinations of `factor`, `onset` and `noise`. They served to choose the parameters of Section 3.3 (main case: factor 10, onset 0.5, noise 0.15; stress test: noise 0.30; sensitivity: factor 100) and are not reported in the paper. The final analysis is in Cell C.

- **Cell A** (grid): index means, response relative to a stable reference series with the same noise, and rank discrimination (AUC) between decline depths.
- **Cell B** (expanding window): the series is revealed step by step; an index "detects" the decline when it exceeds the 95th percentile of the same index in stable reference series of the same length (one-sided, no multiplicity correction; exploratory).

The consecutive disparity index D (Fernández-Martínez et al. 2018) is included as an order-sensitive reference (ln-ratios of consecutive values; 1 is added to all values only if the series contains zeros).


```python
# ------------------------------------------------------------------
# Cell A: decline regime, grid over final density (factor), onset and noise
# Requires the cells above (all_measures, draw_normal_pos, N_STEPS, np, pd, stats, OUT_DIR).
# ------------------------------------------------------------------
DECL_FACTORS = [2, 5, 10, 20, 50, 100]   # final abundance = 100 / factor
DECL_ONSETS  = [0.3, 0.5, 0.7]           # decline starts after this fraction of the series
DECL_NOISES  = [0.05, 0.15, 0.30]        # relative SD of the scatter around the path (0.05: Rare stable phase, 0.15: Normal)
DECL_SHAPE   = "exp"                     # "exp": constant per-capita decline rate; "lin": linear decline
N_DECL       = 5000                      # simulated series per combination
rng_dec      = np.random.default_rng(4711)   # separate generator: main results are not affected

def d_index(z):
    """Consecutive disparity index D: mean |ln(z[t+1] / z[t])| (1 is added if the series contains zeros)."""
    z = np.asarray(z, dtype=float)
    if np.any(z == 0):
        z = z + 1
    return np.mean(np.abs(np.diff(np.log(z))))

def draw_around_path(rng, mean, rel_sd):
    """Rounded Gaussian scatter around a deterministic path; values < 1 are redrawn."""
    x = np.round(rng.normal(mean, rel_sd * mean))
    bad = x < 1
    while bad.any():
        x[bad] = np.round(rng.normal(mean[bad], rel_sd * mean[bad]))
        bad = x < 1
    return x

def gen_decline(rng, factor, onset, noise, n=N_STEPS, shape=DECL_SHAPE):
    t0 = int(round(onset * n))
    path = np.full(n, 100.0)
    m = n - t0
    if m > 0:
        s = np.arange(1, m + 1) / m             # 0 -> 1 over the decline phase
        path[t0:] = 100.0 * factor ** (-s) if shape == "exp" else 100.0 - (100.0 - 100.0 / factor) * s
    return draw_around_path(rng, path, noise)

def measures_with_d(z):
    r = all_measures(z)
    r["D"] = d_index(z)
    return r

DECL_MEASURES = ["CV", "SDL", "pv_lo", "PV_lo", "PV_hi", "D"]
RESP_MEASURES = ["CV", "SDL", "PV_lo", "PV_hi", "D"]

decl_runs, decl_ref, rows = {}, {}, []
for noise in DECL_NOISES:
    ref = pd.DataFrame([measures_with_d(draw_normal_pos(rng_dec, 100, 100 * noise, N_STEPS))
                        for _ in range(N_DECL)])
    decl_ref[noise] = ref
    for onset in DECL_ONSETS:
        for F in DECL_FACTORS:
            df = pd.DataFrame([measures_with_d(gen_decline(rng_dec, F, onset, noise)) for _ in range(N_DECL)])
            decl_runs[(noise, onset, F)] = df
            rows.append(dict(noise=noise, onset=onset, factor=F, **df[DECL_MEASURES].mean().to_dict()))
decl_grid = pd.DataFrame(rows).set_index(["noise", "onset", "factor"])
decl_grid.to_csv(OUT_DIR / "decline_grid.csv")

print("Mean index values (end of the series, n = %d):" % N_STEPS)
display(decl_grid.round(3))

# response relative to a stable series with the same noise (same logic as 'factor' in the Rare analysis)
resp = decl_grid.copy()
for noise in DECL_NOISES:
    ref_mean = decl_ref[noise][DECL_MEASURES].mean()
    resp.loc[noise] = decl_grid.loc[noise].values / ref_mean.values
resp = resp[RESP_MEASURES]
resp["PV_lo / PV_hi"] = resp["PV_lo"] / resp["PV_hi"]
print("\nResponse: index mean divided by the mean of a stable reference series with the same noise:")
display(resp.round(2))

# depth resolution: P(index of the deeper decline > index of the shallower decline) (AUC), rank-based
def auc(a, b):
    return stats.mannwhitneyu(b, a).statistic / (len(a) * len(b))

PAIRS = [(2, 10), (10, 100), (20, 50), (50, 100)]
arows = []
for noise in DECL_NOISES:
    for onset in DECL_ONSETS:
        for f1, f2 in PAIRS:
            r = dict(noise=noise, onset=onset, pair=f"{f1} vs {f2}")
            for m in RESP_MEASURES:
                r[m] = auc(decl_runs[(noise, onset, f1)][m], decl_runs[(noise, onset, f2)][m])
            arows.append(r)
        r = dict(noise=noise, onset=onset, pair="stable vs 2")   # detectability of a shallow decline
        for m in RESP_MEASURES:
            r[m] = auc(decl_ref[noise][m], decl_runs[(noise, onset, 2)][m])
        arows.append(r)
decl_auc = pd.DataFrame(arows).set_index(["noise", "onset", "pair"])
decl_auc.to_csv(OUT_DIR / "decline_auc.csv")
print("\nAUC (1 = deeper decline always ranks higher, 0.5 = no discrimination):")
display(decl_auc.round(3))
```

    Mean index values (end of the series, n = 100):
    Mean index values (end of the series, n = 100):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>CV</th>
      <th>SDL</th>
      <th>pv_lo</th>
      <th>PV_lo</th>
      <th>PV_hi</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>onset</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">0.05</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>0.226</td>
      <td>0.239</td>
      <td>0.341</td>
      <td>0.254</td>
      <td>0.224</td>
      <td>0.057</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.470</td>
      <td>0.545</td>
      <td>1.073</td>
      <td>0.518</td>
      <td>0.406</td>
      <td>0.059</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.619</td>
      <td>0.777</td>
      <td>2.031</td>
      <td>0.670</td>
      <td>0.499</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.740</td>
      <td>1.011</td>
      <td>3.618</td>
      <td>0.783</td>
      <td>0.566</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.865</td>
      <td>1.324</td>
      <td>7.614</td>
      <td>0.884</td>
      <td>0.631</td>
      <td>0.068</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.940</td>
      <td>1.568</td>
      <td>13.706</td>
      <td>0.932</td>
      <td>0.667</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>0.210</td>
      <td>0.233</td>
      <td>0.319</td>
      <td>0.242</td>
      <td>0.208</td>
      <td>0.057</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.409</td>
      <td>0.531</td>
      <td>1.004</td>
      <td>0.501</td>
      <td>0.366</td>
      <td>0.059</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.518</td>
      <td>0.758</td>
      <td>1.923</td>
      <td>0.658</td>
      <td>0.444</td>
      <td>0.062</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.600</td>
      <td>0.985</td>
      <td>3.476</td>
      <td>0.777</td>
      <td>0.500</td>
      <td>0.066</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.679</td>
      <td>1.289</td>
      <td>7.459</td>
      <td>0.882</td>
      <td>0.552</td>
      <td>0.072</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.723</td>
      <td>1.526</td>
      <td>13.636</td>
      <td>0.932</td>
      <td>0.581</td>
      <td>0.078</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>0.175</td>
      <td>0.204</td>
      <td>0.252</td>
      <td>0.201</td>
      <td>0.167</td>
      <td>0.057</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.320</td>
      <td>0.463</td>
      <td>0.768</td>
      <td>0.434</td>
      <td>0.279</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.393</td>
      <td>0.661</td>
      <td>1.472</td>
      <td>0.595</td>
      <td>0.333</td>
      <td>0.066</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.444</td>
      <td>0.859</td>
      <td>2.677</td>
      <td>0.728</td>
      <td>0.372</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.492</td>
      <td>1.123</td>
      <td>5.810</td>
      <td>0.853</td>
      <td>0.407</td>
      <td>0.080</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.517</td>
      <td>1.328</td>
      <td>10.707</td>
      <td>0.915</td>
      <td>0.426</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.15</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>0.268</td>
      <td>0.280</td>
      <td>0.417</td>
      <td>0.294</td>
      <td>0.255</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.495</td>
      <td>0.564</td>
      <td>1.145</td>
      <td>0.534</td>
      <td>0.421</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.640</td>
      <td>0.791</td>
      <td>2.120</td>
      <td>0.679</td>
      <td>0.510</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.760</td>
      <td>1.021</td>
      <td>3.738</td>
      <td>0.789</td>
      <td>0.576</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.885</td>
      <td>1.329</td>
      <td>7.759</td>
      <td>0.886</td>
      <td>0.639</td>
      <td>0.177</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.959</td>
      <td>1.572</td>
      <td>13.861</td>
      <td>0.933</td>
      <td>0.675</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>0.255</td>
      <td>0.275</td>
      <td>0.404</td>
      <td>0.288</td>
      <td>0.247</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.436</td>
      <td>0.550</td>
      <td>1.088</td>
      <td>0.521</td>
      <td>0.393</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.541</td>
      <td>0.772</td>
      <td>2.023</td>
      <td>0.669</td>
      <td>0.469</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.622</td>
      <td>0.996</td>
      <td>3.606</td>
      <td>0.783</td>
      <td>0.524</td>
      <td>0.176</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.699</td>
      <td>1.295</td>
      <td>7.625</td>
      <td>0.884</td>
      <td>0.576</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.743</td>
      <td>1.530</td>
      <td>13.803</td>
      <td>0.932</td>
      <td>0.604</td>
      <td>0.180</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>0.226</td>
      <td>0.251</td>
      <td>0.353</td>
      <td>0.261</td>
      <td>0.223</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.353</td>
      <td>0.486</td>
      <td>0.871</td>
      <td>0.465</td>
      <td>0.328</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.421</td>
      <td>0.677</td>
      <td>1.588</td>
      <td>0.613</td>
      <td>0.381</td>
      <td>0.176</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.470</td>
      <td>0.871</td>
      <td>2.817</td>
      <td>0.738</td>
      <td>0.419</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.516</td>
      <td>1.131</td>
      <td>5.992</td>
      <td>0.857</td>
      <td>0.453</td>
      <td>0.182</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.540</td>
      <td>1.334</td>
      <td>10.903</td>
      <td>0.916</td>
      <td>0.473</td>
      <td>0.185</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.30</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>0.376</td>
      <td>0.420</td>
      <td>0.750</td>
      <td>0.423</td>
      <td>0.331</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.571</td>
      <td>0.645</td>
      <td>1.494</td>
      <td>0.596</td>
      <td>0.455</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.707</td>
      <td>0.850</td>
      <td>2.543</td>
      <td>0.716</td>
      <td>0.533</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.824</td>
      <td>1.067</td>
      <td>4.306</td>
      <td>0.810</td>
      <td>0.594</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.947</td>
      <td>1.363</td>
      <td>8.661</td>
      <td>0.896</td>
      <td>0.654</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.021</td>
      <td>1.586</td>
      <td>14.439</td>
      <td>0.935</td>
      <td>0.687</td>
      <td>0.364</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>0.367</td>
      <td>0.418</td>
      <td>0.748</td>
      <td>0.422</td>
      <td>0.328</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.517</td>
      <td>0.634</td>
      <td>1.453</td>
      <td>0.589</td>
      <td>0.440</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.614</td>
      <td>0.833</td>
      <td>2.462</td>
      <td>0.709</td>
      <td>0.508</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.690</td>
      <td>1.044</td>
      <td>4.199</td>
      <td>0.806</td>
      <td>0.559</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.765</td>
      <td>1.330</td>
      <td>8.529</td>
      <td>0.894</td>
      <td>0.608</td>
      <td>0.374</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.808</td>
      <td>1.547</td>
      <td>14.391</td>
      <td>0.935</td>
      <td>0.636</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>0.346</td>
      <td>0.402</td>
      <td>0.703</td>
      <td>0.407</td>
      <td>0.316</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.444</td>
      <td>0.578</td>
      <td>1.238</td>
      <td>0.549</td>
      <td>0.397</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.504</td>
      <td>0.746</td>
      <td>2.020</td>
      <td>0.666</td>
      <td>0.446</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.548</td>
      <td>0.925</td>
      <td>3.363</td>
      <td>0.769</td>
      <td>0.481</td>
      <td>0.374</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.590</td>
      <td>1.173</td>
      <td>6.809</td>
      <td>0.871</td>
      <td>0.515</td>
      <td>0.376</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.613</td>
      <td>1.359</td>
      <td>11.457</td>
      <td>0.919</td>
      <td>0.534</td>
      <td>0.373</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>CV</th>
      <th>SDL</th>
      <th>pv_lo</th>
      <th>PV_lo</th>
      <th>PV_hi</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>onset</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">0.05</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>0.226</td>
      <td>0.239</td>
      <td>0.341</td>
      <td>0.254</td>
      <td>0.224</td>
      <td>0.057</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.470</td>
      <td>0.545</td>
      <td>1.073</td>
      <td>0.518</td>
      <td>0.406</td>
      <td>0.059</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.619</td>
      <td>0.777</td>
      <td>2.031</td>
      <td>0.670</td>
      <td>0.499</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.740</td>
      <td>1.011</td>
      <td>3.618</td>
      <td>0.783</td>
      <td>0.566</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.865</td>
      <td>1.324</td>
      <td>7.614</td>
      <td>0.884</td>
      <td>0.631</td>
      <td>0.068</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.940</td>
      <td>1.568</td>
      <td>13.706</td>
      <td>0.932</td>
      <td>0.667</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>0.210</td>
      <td>0.233</td>
      <td>0.319</td>
      <td>0.242</td>
      <td>0.208</td>
      <td>0.057</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.409</td>
      <td>0.531</td>
      <td>1.004</td>
      <td>0.501</td>
      <td>0.366</td>
      <td>0.059</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.518</td>
      <td>0.758</td>
      <td>1.923</td>
      <td>0.658</td>
      <td>0.444</td>
      <td>0.062</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.600</td>
      <td>0.985</td>
      <td>3.476</td>
      <td>0.777</td>
      <td>0.500</td>
      <td>0.066</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.679</td>
      <td>1.289</td>
      <td>7.459</td>
      <td>0.882</td>
      <td>0.552</td>
      <td>0.072</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.723</td>
      <td>1.526</td>
      <td>13.636</td>
      <td>0.932</td>
      <td>0.581</td>
      <td>0.078</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>0.175</td>
      <td>0.204</td>
      <td>0.252</td>
      <td>0.201</td>
      <td>0.167</td>
      <td>0.057</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.320</td>
      <td>0.463</td>
      <td>0.768</td>
      <td>0.434</td>
      <td>0.279</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.393</td>
      <td>0.661</td>
      <td>1.472</td>
      <td>0.595</td>
      <td>0.333</td>
      <td>0.066</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.444</td>
      <td>0.859</td>
      <td>2.677</td>
      <td>0.728</td>
      <td>0.372</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.492</td>
      <td>1.123</td>
      <td>5.810</td>
      <td>0.853</td>
      <td>0.407</td>
      <td>0.080</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.517</td>
      <td>1.328</td>
      <td>10.707</td>
      <td>0.915</td>
      <td>0.426</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.15</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>0.268</td>
      <td>0.280</td>
      <td>0.417</td>
      <td>0.294</td>
      <td>0.255</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.495</td>
      <td>0.564</td>
      <td>1.145</td>
      <td>0.534</td>
      <td>0.421</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.640</td>
      <td>0.791</td>
      <td>2.120</td>
      <td>0.679</td>
      <td>0.510</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.760</td>
      <td>1.021</td>
      <td>3.738</td>
      <td>0.789</td>
      <td>0.576</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.885</td>
      <td>1.329</td>
      <td>7.759</td>
      <td>0.886</td>
      <td>0.639</td>
      <td>0.177</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.959</td>
      <td>1.572</td>
      <td>13.861</td>
      <td>0.933</td>
      <td>0.675</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>0.255</td>
      <td>0.275</td>
      <td>0.404</td>
      <td>0.288</td>
      <td>0.247</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.436</td>
      <td>0.550</td>
      <td>1.088</td>
      <td>0.521</td>
      <td>0.393</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.541</td>
      <td>0.772</td>
      <td>2.023</td>
      <td>0.669</td>
      <td>0.469</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.622</td>
      <td>0.996</td>
      <td>3.606</td>
      <td>0.783</td>
      <td>0.524</td>
      <td>0.176</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.699</td>
      <td>1.295</td>
      <td>7.625</td>
      <td>0.884</td>
      <td>0.576</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.743</td>
      <td>1.530</td>
      <td>13.803</td>
      <td>0.932</td>
      <td>0.604</td>
      <td>0.180</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>0.226</td>
      <td>0.251</td>
      <td>0.353</td>
      <td>0.261</td>
      <td>0.223</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.353</td>
      <td>0.486</td>
      <td>0.871</td>
      <td>0.465</td>
      <td>0.328</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.421</td>
      <td>0.677</td>
      <td>1.588</td>
      <td>0.613</td>
      <td>0.381</td>
      <td>0.176</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.470</td>
      <td>0.871</td>
      <td>2.817</td>
      <td>0.738</td>
      <td>0.419</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.516</td>
      <td>1.131</td>
      <td>5.992</td>
      <td>0.857</td>
      <td>0.453</td>
      <td>0.182</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.540</td>
      <td>1.334</td>
      <td>10.903</td>
      <td>0.916</td>
      <td>0.473</td>
      <td>0.185</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.30</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>0.376</td>
      <td>0.420</td>
      <td>0.750</td>
      <td>0.423</td>
      <td>0.331</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.571</td>
      <td>0.645</td>
      <td>1.494</td>
      <td>0.596</td>
      <td>0.455</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.707</td>
      <td>0.850</td>
      <td>2.543</td>
      <td>0.716</td>
      <td>0.533</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.824</td>
      <td>1.067</td>
      <td>4.306</td>
      <td>0.810</td>
      <td>0.594</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.947</td>
      <td>1.363</td>
      <td>8.661</td>
      <td>0.896</td>
      <td>0.654</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.021</td>
      <td>1.586</td>
      <td>14.439</td>
      <td>0.935</td>
      <td>0.687</td>
      <td>0.364</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>0.367</td>
      <td>0.418</td>
      <td>0.748</td>
      <td>0.422</td>
      <td>0.328</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.517</td>
      <td>0.634</td>
      <td>1.453</td>
      <td>0.589</td>
      <td>0.440</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.614</td>
      <td>0.833</td>
      <td>2.462</td>
      <td>0.709</td>
      <td>0.508</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.690</td>
      <td>1.044</td>
      <td>4.199</td>
      <td>0.806</td>
      <td>0.559</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.765</td>
      <td>1.330</td>
      <td>8.529</td>
      <td>0.894</td>
      <td>0.608</td>
      <td>0.374</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.808</td>
      <td>1.547</td>
      <td>14.391</td>
      <td>0.935</td>
      <td>0.636</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>0.346</td>
      <td>0.402</td>
      <td>0.703</td>
      <td>0.407</td>
      <td>0.316</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.444</td>
      <td>0.578</td>
      <td>1.238</td>
      <td>0.549</td>
      <td>0.397</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.504</td>
      <td>0.746</td>
      <td>2.020</td>
      <td>0.666</td>
      <td>0.446</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.548</td>
      <td>0.925</td>
      <td>3.363</td>
      <td>0.769</td>
      <td>0.481</td>
      <td>0.374</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.590</td>
      <td>1.173</td>
      <td>6.809</td>
      <td>0.871</td>
      <td>0.515</td>
      <td>0.376</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.613</td>
      <td>1.359</td>
      <td>11.457</td>
      <td>0.919</td>
      <td>0.534</td>
      <td>0.373</td>
    </tr>
  </tbody>
</table>
</div>


    
    Response: index mean divided by the mean of a stable reference series with the same noise:
    
    Response: index mean divided by the mean of a stable reference series with the same noise:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>CV</th>
      <th>SDL</th>
      <th>PV_lo</th>
      <th>PV_hi</th>
      <th>D</th>
      <th>PV_lo / PV_hi</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>onset</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">0.05</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>4.53</td>
      <td>4.77</td>
      <td>4.55</td>
      <td>4.14</td>
      <td>1.01</td>
      <td>1.10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.40</td>
      <td>10.87</td>
      <td>9.27</td>
      <td>7.52</td>
      <td>1.04</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12.39</td>
      <td>15.52</td>
      <td>12.00</td>
      <td>9.22</td>
      <td>1.08</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>20</th>
      <td>14.82</td>
      <td>20.18</td>
      <td>14.03</td>
      <td>10.47</td>
      <td>1.12</td>
      <td>1.34</td>
    </tr>
    <tr>
      <th>50</th>
      <td>17.33</td>
      <td>26.42</td>
      <td>15.83</td>
      <td>11.66</td>
      <td>1.21</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>100</th>
      <td>18.81</td>
      <td>31.30</td>
      <td>16.70</td>
      <td>12.34</td>
      <td>1.29</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>4.21</td>
      <td>4.65</td>
      <td>4.33</td>
      <td>3.84</td>
      <td>1.01</td>
      <td>1.13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.18</td>
      <td>10.59</td>
      <td>8.98</td>
      <td>6.77</td>
      <td>1.05</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.36</td>
      <td>15.12</td>
      <td>11.78</td>
      <td>8.21</td>
      <td>1.11</td>
      <td>1.43</td>
    </tr>
    <tr>
      <th>20</th>
      <td>12.01</td>
      <td>19.67</td>
      <td>13.91</td>
      <td>9.25</td>
      <td>1.17</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>50</th>
      <td>13.59</td>
      <td>25.74</td>
      <td>15.80</td>
      <td>10.21</td>
      <td>1.28</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th>100</th>
      <td>14.47</td>
      <td>30.47</td>
      <td>16.69</td>
      <td>10.75</td>
      <td>1.38</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>3.51</td>
      <td>4.08</td>
      <td>3.61</td>
      <td>3.08</td>
      <td>1.02</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.41</td>
      <td>9.24</td>
      <td>7.78</td>
      <td>5.16</td>
      <td>1.08</td>
      <td>1.51</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.86</td>
      <td>13.19</td>
      <td>10.67</td>
      <td>6.17</td>
      <td>1.16</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8.90</td>
      <td>17.14</td>
      <td>13.04</td>
      <td>6.87</td>
      <td>1.26</td>
      <td>1.90</td>
    </tr>
    <tr>
      <th>50</th>
      <td>9.84</td>
      <td>22.42</td>
      <td>15.28</td>
      <td>7.52</td>
      <td>1.41</td>
      <td>2.03</td>
    </tr>
    <tr>
      <th>100</th>
      <td>10.35</td>
      <td>26.52</td>
      <td>16.38</td>
      <td>7.88</td>
      <td>1.53</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.15</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>1.79</td>
      <td>1.82</td>
      <td>1.77</td>
      <td>1.68</td>
      <td>1.00</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.31</td>
      <td>3.66</td>
      <td>3.21</td>
      <td>2.78</td>
      <td>1.00</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4.28</td>
      <td>5.14</td>
      <td>4.09</td>
      <td>3.37</td>
      <td>1.01</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5.08</td>
      <td>6.63</td>
      <td>4.75</td>
      <td>3.81</td>
      <td>1.02</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>50</th>
      <td>5.92</td>
      <td>8.63</td>
      <td>5.33</td>
      <td>4.23</td>
      <td>1.02</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>100</th>
      <td>6.41</td>
      <td>10.21</td>
      <td>5.61</td>
      <td>4.47</td>
      <td>1.03</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>1.70</td>
      <td>1.78</td>
      <td>1.73</td>
      <td>1.63</td>
      <td>1.00</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.91</td>
      <td>3.57</td>
      <td>3.13</td>
      <td>2.60</td>
      <td>1.01</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.62</td>
      <td>5.01</td>
      <td>4.03</td>
      <td>3.10</td>
      <td>1.01</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4.16</td>
      <td>6.47</td>
      <td>4.71</td>
      <td>3.47</td>
      <td>1.02</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>50</th>
      <td>4.68</td>
      <td>8.41</td>
      <td>5.32</td>
      <td>3.81</td>
      <td>1.03</td>
      <td>1.40</td>
    </tr>
    <tr>
      <th>100</th>
      <td>4.97</td>
      <td>9.94</td>
      <td>5.61</td>
      <td>4.00</td>
      <td>1.04</td>
      <td>1.40</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>1.51</td>
      <td>1.63</td>
      <td>1.57</td>
      <td>1.47</td>
      <td>1.00</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.36</td>
      <td>3.15</td>
      <td>2.80</td>
      <td>2.17</td>
      <td>1.01</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.81</td>
      <td>4.40</td>
      <td>3.69</td>
      <td>2.52</td>
      <td>1.02</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3.14</td>
      <td>5.66</td>
      <td>4.44</td>
      <td>2.77</td>
      <td>1.03</td>
      <td>1.60</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3.45</td>
      <td>7.35</td>
      <td>5.15</td>
      <td>3.00</td>
      <td>1.05</td>
      <td>1.72</td>
    </tr>
    <tr>
      <th>100</th>
      <td>3.61</td>
      <td>8.67</td>
      <td>5.51</td>
      <td>3.12</td>
      <td>1.07</td>
      <td>1.76</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.30</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>1.26</td>
      <td>1.20</td>
      <td>1.18</td>
      <td>1.19</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.91</td>
      <td>1.85</td>
      <td>1.67</td>
      <td>1.63</td>
      <td>1.00</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.37</td>
      <td>2.43</td>
      <td>2.00</td>
      <td>1.91</td>
      <td>1.00</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.76</td>
      <td>3.05</td>
      <td>2.27</td>
      <td>2.13</td>
      <td>1.00</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3.17</td>
      <td>3.90</td>
      <td>2.51</td>
      <td>2.35</td>
      <td>1.00</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>100</th>
      <td>3.42</td>
      <td>4.54</td>
      <td>2.62</td>
      <td>2.46</td>
      <td>0.98</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>1.23</td>
      <td>1.20</td>
      <td>1.18</td>
      <td>1.18</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.73</td>
      <td>1.82</td>
      <td>1.65</td>
      <td>1.58</td>
      <td>1.00</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.06</td>
      <td>2.38</td>
      <td>1.98</td>
      <td>1.82</td>
      <td>1.00</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.31</td>
      <td>2.99</td>
      <td>2.26</td>
      <td>2.01</td>
      <td>1.00</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2.56</td>
      <td>3.81</td>
      <td>2.50</td>
      <td>2.18</td>
      <td>1.00</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2.71</td>
      <td>4.43</td>
      <td>2.62</td>
      <td>2.28</td>
      <td>0.99</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>1.16</td>
      <td>1.15</td>
      <td>1.14</td>
      <td>1.13</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.49</td>
      <td>1.65</td>
      <td>1.54</td>
      <td>1.43</td>
      <td>1.00</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.69</td>
      <td>2.13</td>
      <td>1.86</td>
      <td>1.60</td>
      <td>1.00</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.84</td>
      <td>2.65</td>
      <td>2.15</td>
      <td>1.73</td>
      <td>1.01</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1.98</td>
      <td>3.36</td>
      <td>2.44</td>
      <td>1.85</td>
      <td>1.01</td>
      <td>1.32</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2.05</td>
      <td>3.89</td>
      <td>2.57</td>
      <td>1.91</td>
      <td>1.00</td>
      <td>1.34</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>CV</th>
      <th>SDL</th>
      <th>PV_lo</th>
      <th>PV_hi</th>
      <th>D</th>
      <th>PV_lo / PV_hi</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>onset</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">0.05</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>4.53</td>
      <td>4.77</td>
      <td>4.55</td>
      <td>4.14</td>
      <td>1.01</td>
      <td>1.10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.40</td>
      <td>10.87</td>
      <td>9.27</td>
      <td>7.52</td>
      <td>1.04</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12.39</td>
      <td>15.52</td>
      <td>12.00</td>
      <td>9.22</td>
      <td>1.08</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>20</th>
      <td>14.82</td>
      <td>20.18</td>
      <td>14.03</td>
      <td>10.47</td>
      <td>1.12</td>
      <td>1.34</td>
    </tr>
    <tr>
      <th>50</th>
      <td>17.33</td>
      <td>26.42</td>
      <td>15.83</td>
      <td>11.66</td>
      <td>1.21</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>100</th>
      <td>18.81</td>
      <td>31.30</td>
      <td>16.70</td>
      <td>12.34</td>
      <td>1.29</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>4.21</td>
      <td>4.65</td>
      <td>4.33</td>
      <td>3.84</td>
      <td>1.01</td>
      <td>1.13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.18</td>
      <td>10.59</td>
      <td>8.98</td>
      <td>6.77</td>
      <td>1.05</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.36</td>
      <td>15.12</td>
      <td>11.78</td>
      <td>8.21</td>
      <td>1.11</td>
      <td>1.43</td>
    </tr>
    <tr>
      <th>20</th>
      <td>12.01</td>
      <td>19.67</td>
      <td>13.91</td>
      <td>9.25</td>
      <td>1.17</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>50</th>
      <td>13.59</td>
      <td>25.74</td>
      <td>15.80</td>
      <td>10.21</td>
      <td>1.28</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th>100</th>
      <td>14.47</td>
      <td>30.47</td>
      <td>16.69</td>
      <td>10.75</td>
      <td>1.38</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>3.51</td>
      <td>4.08</td>
      <td>3.61</td>
      <td>3.08</td>
      <td>1.02</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.41</td>
      <td>9.24</td>
      <td>7.78</td>
      <td>5.16</td>
      <td>1.08</td>
      <td>1.51</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.86</td>
      <td>13.19</td>
      <td>10.67</td>
      <td>6.17</td>
      <td>1.16</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8.90</td>
      <td>17.14</td>
      <td>13.04</td>
      <td>6.87</td>
      <td>1.26</td>
      <td>1.90</td>
    </tr>
    <tr>
      <th>50</th>
      <td>9.84</td>
      <td>22.42</td>
      <td>15.28</td>
      <td>7.52</td>
      <td>1.41</td>
      <td>2.03</td>
    </tr>
    <tr>
      <th>100</th>
      <td>10.35</td>
      <td>26.52</td>
      <td>16.38</td>
      <td>7.88</td>
      <td>1.53</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.15</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>1.79</td>
      <td>1.82</td>
      <td>1.77</td>
      <td>1.68</td>
      <td>1.00</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.31</td>
      <td>3.66</td>
      <td>3.21</td>
      <td>2.78</td>
      <td>1.00</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4.28</td>
      <td>5.14</td>
      <td>4.09</td>
      <td>3.37</td>
      <td>1.01</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5.08</td>
      <td>6.63</td>
      <td>4.75</td>
      <td>3.81</td>
      <td>1.02</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>50</th>
      <td>5.92</td>
      <td>8.63</td>
      <td>5.33</td>
      <td>4.23</td>
      <td>1.02</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>100</th>
      <td>6.41</td>
      <td>10.21</td>
      <td>5.61</td>
      <td>4.47</td>
      <td>1.03</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>1.70</td>
      <td>1.78</td>
      <td>1.73</td>
      <td>1.63</td>
      <td>1.00</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.91</td>
      <td>3.57</td>
      <td>3.13</td>
      <td>2.60</td>
      <td>1.01</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.62</td>
      <td>5.01</td>
      <td>4.03</td>
      <td>3.10</td>
      <td>1.01</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4.16</td>
      <td>6.47</td>
      <td>4.71</td>
      <td>3.47</td>
      <td>1.02</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>50</th>
      <td>4.68</td>
      <td>8.41</td>
      <td>5.32</td>
      <td>3.81</td>
      <td>1.03</td>
      <td>1.40</td>
    </tr>
    <tr>
      <th>100</th>
      <td>4.97</td>
      <td>9.94</td>
      <td>5.61</td>
      <td>4.00</td>
      <td>1.04</td>
      <td>1.40</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>1.51</td>
      <td>1.63</td>
      <td>1.57</td>
      <td>1.47</td>
      <td>1.00</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.36</td>
      <td>3.15</td>
      <td>2.80</td>
      <td>2.17</td>
      <td>1.01</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.81</td>
      <td>4.40</td>
      <td>3.69</td>
      <td>2.52</td>
      <td>1.02</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3.14</td>
      <td>5.66</td>
      <td>4.44</td>
      <td>2.77</td>
      <td>1.03</td>
      <td>1.60</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3.45</td>
      <td>7.35</td>
      <td>5.15</td>
      <td>3.00</td>
      <td>1.05</td>
      <td>1.72</td>
    </tr>
    <tr>
      <th>100</th>
      <td>3.61</td>
      <td>8.67</td>
      <td>5.51</td>
      <td>3.12</td>
      <td>1.07</td>
      <td>1.76</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.30</th>
      <th rowspan="6" valign="top">0.3</th>
      <th>2</th>
      <td>1.26</td>
      <td>1.20</td>
      <td>1.18</td>
      <td>1.19</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.91</td>
      <td>1.85</td>
      <td>1.67</td>
      <td>1.63</td>
      <td>1.00</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.37</td>
      <td>2.43</td>
      <td>2.00</td>
      <td>1.91</td>
      <td>1.00</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.76</td>
      <td>3.05</td>
      <td>2.27</td>
      <td>2.13</td>
      <td>1.00</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3.17</td>
      <td>3.90</td>
      <td>2.51</td>
      <td>2.35</td>
      <td>1.00</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>100</th>
      <td>3.42</td>
      <td>4.54</td>
      <td>2.62</td>
      <td>2.46</td>
      <td>0.98</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.5</th>
      <th>2</th>
      <td>1.23</td>
      <td>1.20</td>
      <td>1.18</td>
      <td>1.18</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.73</td>
      <td>1.82</td>
      <td>1.65</td>
      <td>1.58</td>
      <td>1.00</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.06</td>
      <td>2.38</td>
      <td>1.98</td>
      <td>1.82</td>
      <td>1.00</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.31</td>
      <td>2.99</td>
      <td>2.26</td>
      <td>2.01</td>
      <td>1.00</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2.56</td>
      <td>3.81</td>
      <td>2.50</td>
      <td>2.18</td>
      <td>1.00</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2.71</td>
      <td>4.43</td>
      <td>2.62</td>
      <td>2.28</td>
      <td>0.99</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">0.7</th>
      <th>2</th>
      <td>1.16</td>
      <td>1.15</td>
      <td>1.14</td>
      <td>1.13</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.49</td>
      <td>1.65</td>
      <td>1.54</td>
      <td>1.43</td>
      <td>1.00</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.69</td>
      <td>2.13</td>
      <td>1.86</td>
      <td>1.60</td>
      <td>1.00</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.84</td>
      <td>2.65</td>
      <td>2.15</td>
      <td>1.73</td>
      <td>1.01</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1.98</td>
      <td>3.36</td>
      <td>2.44</td>
      <td>1.85</td>
      <td>1.01</td>
      <td>1.32</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2.05</td>
      <td>3.89</td>
      <td>2.57</td>
      <td>1.91</td>
      <td>1.00</td>
      <td>1.34</td>
    </tr>
  </tbody>
</table>
</div>


    
    AUC (1 = deeper decline always ranks higher, 0.5 = no discrimination):
    
    AUC (1 = deeper decline always ranks higher, 0.5 = no discrimination):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>CV</th>
      <th>SDL</th>
      <th>PV_lo</th>
      <th>PV_hi</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>onset</th>
      <th>pair</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="15" valign="top">0.05</th>
      <th rowspan="5" valign="top">0.3</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.699</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.932</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.713</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.682</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.524</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.5</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.764</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.986</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.807</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.774</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.537</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.7</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.878</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.999</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.902</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.853</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.553</td>
    </tr>
    <tr>
      <th rowspan="15" valign="top">0.15</th>
      <th rowspan="5" valign="top">0.3</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.523</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.558</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.523</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.519</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.505</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.5</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.533</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.579</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.536</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.523</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.508</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.7</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.561</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.652</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.557</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.999</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.982</td>
      <td>0.561</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.499</td>
    </tr>
    <tr>
      <th rowspan="15" valign="top">0.30</th>
      <th rowspan="5" valign="top">0.3</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.998</td>
      <td>1.000</td>
      <td>0.506</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.445</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.986</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.445</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>0.991</td>
      <td>0.875</td>
      <td>0.877</td>
      <td>0.979</td>
      <td>0.496</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.5</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.996</td>
      <td>1.000</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.466</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>0.998</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.949</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.996</td>
      <td>0.459</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>0.981</td>
      <td>0.866</td>
      <td>0.869</td>
      <td>0.971</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.7</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.993</td>
      <td>1.000</td>
      <td>0.510</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.505</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>0.955</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.981</td>
      <td>0.511</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.832</td>
      <td>1.000</td>
      <td>0.999</td>
      <td>0.876</td>
      <td>0.483</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>0.933</td>
      <td>0.810</td>
      <td>0.815</td>
      <td>0.927</td>
      <td>0.495</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>CV</th>
      <th>SDL</th>
      <th>PV_lo</th>
      <th>PV_hi</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>onset</th>
      <th>pair</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="15" valign="top">0.05</th>
      <th rowspan="5" valign="top">0.3</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.699</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.932</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.713</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.682</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.524</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.5</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.764</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.986</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.807</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.774</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.537</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.7</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.878</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.999</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.902</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.853</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.553</td>
    </tr>
    <tr>
      <th rowspan="15" valign="top">0.15</th>
      <th rowspan="5" valign="top">0.3</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.523</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.558</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.523</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.519</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.505</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.5</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.533</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.579</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.536</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.523</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.508</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.7</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.561</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.652</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.557</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.999</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.982</td>
      <td>0.561</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.499</td>
    </tr>
    <tr>
      <th rowspan="15" valign="top">0.30</th>
      <th rowspan="5" valign="top">0.3</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.998</td>
      <td>1.000</td>
      <td>0.506</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.445</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.986</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.445</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>0.991</td>
      <td>0.875</td>
      <td>0.877</td>
      <td>0.979</td>
      <td>0.496</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.5</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.996</td>
      <td>1.000</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.466</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>0.998</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.949</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.996</td>
      <td>0.459</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>0.981</td>
      <td>0.866</td>
      <td>0.869</td>
      <td>0.971</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.7</th>
      <th>2 vs 10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.993</td>
      <td>1.000</td>
      <td>0.510</td>
    </tr>
    <tr>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.505</td>
    </tr>
    <tr>
      <th>20 vs 50</th>
      <td>0.955</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.981</td>
      <td>0.511</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.832</td>
      <td>1.000</td>
      <td>0.999</td>
      <td>0.876</td>
      <td>0.483</td>
    </tr>
    <tr>
      <th>stable vs 2</th>
      <td>0.933</td>
      <td>0.810</td>
      <td>0.815</td>
      <td>0.927</td>
      <td>0.495</td>
    </tr>
  </tbody>
</table>
</div>



```python
# ------------------------------------------------------------------
# Cell B: decline regime, expanding window ("monitoring that accumulates over time")
# Requires Cell A (gen_decline, measures_with_d, draw_normal_pos, rng_dec, ...).
# For every prefix length k the index is compared with the 95th percentile of the same index
# in stable reference series of length k (same noise). Power = share of decline series above it.
# ------------------------------------------------------------------
DET_FACTORS = [2, 10, 100]
DET_ONSETS  = [0.3, 0.5, 0.7]
DET_NOISES  = [0.15, 0.30]
DET_DELAYS  = [10, 20, 30]                       # time steps after the onset of the decline
DET_KS      = list(range(10, N_STEPS + 1, 2))    # prefix lengths
DET_MEAS    = ["CV", "SDL", "PV_lo", "PV_hi", "D"]
N_DET_REF, N_DET, DET_Q = 2000, 1000, 0.95

drows = []
for noise in DET_NOISES:
    ref = {m: np.empty((N_DET_REF, len(DET_KS))) for m in DET_MEAS}
    for i in range(N_DET_REF):
        z = draw_normal_pos(rng_dec, 100, 100 * noise, N_STEPS)
        for j, k in enumerate(DET_KS):
            r = measures_with_d(z[:k])
            for m in DET_MEAS:
                ref[m][i, j] = r[m]
    thr = {m: np.quantile(ref[m], DET_Q, axis=0) for m in DET_MEAS}
    for onset in DET_ONSETS:
        t0 = int(round(onset * N_STEPS))
        for F in DET_FACTORS:
            hit = {m: np.zeros((N_DET, len(DET_KS)), dtype=bool) for m in DET_MEAS}
            for i in range(N_DET):
                z = gen_decline(rng_dec, F, onset, noise)
                for j, k in enumerate(DET_KS):
                    r = measures_with_d(z[:k])
                    for m in DET_MEAS:
                        hit[m][i, j] = r[m] > thr[m][j]
            for m in DET_MEAS:
                row = dict(noise=noise, onset=onset, factor=F, measure=m)
                for d in DET_DELAYS:
                    if t0 + d <= N_STEPS:
                        row[f"power +{d}"] = hit[m][:, DET_KS.index(t0 + d)].mean()
                row["power end"] = hit[m][:, -1].mean()
                drows.append(row)
detect = pd.DataFrame(drows)
detect.to_csv(OUT_DIR / "decline_detection.csv", index=False)
print(f"Share of decline series above the {DET_Q:.0%} threshold of stable series of the same length")
print("(+d = d time steps after the onset of the decline; false-positive rate of the threshold is 5 % per prefix length):")
display(detect.pivot_table(index=["noise", "onset", "factor"], columns="measure",
                           values=["power +10", "power +20", "power +30", "power end"])
        [["power +10", "power +20", "power end"]].round(2))
```

    Share of decline series above the 95% threshold of stable series of the same length
    (+d = d time steps after the onset of the decline; false-positive rate of the threshold is 5 % per prefix length):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th colspan="5" halign="left">power +10</th>
      <th colspan="5" halign="left">power +20</th>
      <th colspan="5" halign="left">power end</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>measure</th>
      <th>CV</th>
      <th>D</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>onset</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9" valign="top">0.15</th>
      <th rowspan="3" valign="top">0.3</th>
      <th>2</th>
      <td>0.07</td>
      <td>0.05</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.19</td>
      <td>0.05</td>
      <td>0.20</td>
      <td>0.19</td>
      <td>0.18</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.36</td>
      <td>0.05</td>
      <td>0.38</td>
      <td>0.35</td>
      <td>0.34</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.98</td>
      <td>0.07</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0.5</th>
      <th>2</th>
      <td>0.07</td>
      <td>0.05</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.33</td>
      <td>0.04</td>
      <td>0.33</td>
      <td>0.33</td>
      <td>0.30</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.61</td>
      <td>0.04</td>
      <td>0.61</td>
      <td>0.66</td>
      <td>0.68</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.08</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.11</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0.7</th>
      <th>2</th>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.14</td>
      <td>0.15</td>
      <td>0.14</td>
      <td>0.89</td>
      <td>0.05</td>
      <td>0.90</td>
      <td>0.92</td>
      <td>0.91</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.98</td>
      <td>0.04</td>
      <td>0.98</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.16</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">0.30</th>
      <th rowspan="3" valign="top">0.3</th>
      <th>2</th>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>0.95</td>
      <td>0.06</td>
      <td>0.91</td>
      <td>0.23</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.10</td>
      <td>0.05</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.53</td>
      <td>0.06</td>
      <td>0.48</td>
      <td>0.14</td>
      <td>0.15</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.29</td>
      <td>0.06</td>
      <td>0.26</td>
      <td>0.12</td>
      <td>0.12</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>0.95</td>
      <td>0.96</td>
      <td>1.00</td>
      <td>0.03</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0.5</th>
      <th>2</th>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.12</td>
      <td>0.07</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.90</td>
      <td>0.07</td>
      <td>0.85</td>
      <td>0.21</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.84</td>
      <td>0.05</td>
      <td>0.84</td>
      <td>0.30</td>
      <td>0.36</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.50</td>
      <td>0.08</td>
      <td>0.53</td>
      <td>0.20</td>
      <td>0.21</td>
      <td>1.00</td>
      <td>0.07</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0.7</th>
      <th>2</th>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.20</td>
      <td>0.05</td>
      <td>0.19</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.65</td>
      <td>0.04</td>
      <td>0.65</td>
      <td>0.11</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.25</td>
      <td>0.04</td>
      <td>0.25</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>0.97</td>
      <td>0.99</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.90</td>
      <td>0.05</td>
      <td>0.92</td>
      <td>0.63</td>
      <td>0.72</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>


    Share of decline series above the 95% threshold of stable series of the same length
    (+d = d time steps after the onset of the decline; false-positive rate of the threshold is 5 % per prefix length):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th colspan="5" halign="left">power +10</th>
      <th colspan="5" halign="left">power +20</th>
      <th colspan="5" halign="left">power end</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>measure</th>
      <th>CV</th>
      <th>D</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>onset</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9" valign="top">0.15</th>
      <th rowspan="3" valign="top">0.3</th>
      <th>2</th>
      <td>0.07</td>
      <td>0.05</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.19</td>
      <td>0.05</td>
      <td>0.20</td>
      <td>0.19</td>
      <td>0.18</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.36</td>
      <td>0.05</td>
      <td>0.38</td>
      <td>0.35</td>
      <td>0.34</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.98</td>
      <td>0.07</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0.5</th>
      <th>2</th>
      <td>0.07</td>
      <td>0.05</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.33</td>
      <td>0.04</td>
      <td>0.33</td>
      <td>0.33</td>
      <td>0.30</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.61</td>
      <td>0.04</td>
      <td>0.61</td>
      <td>0.66</td>
      <td>0.68</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.08</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.11</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0.7</th>
      <th>2</th>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.14</td>
      <td>0.15</td>
      <td>0.14</td>
      <td>0.89</td>
      <td>0.05</td>
      <td>0.90</td>
      <td>0.92</td>
      <td>0.91</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.98</td>
      <td>0.04</td>
      <td>0.98</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.16</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">0.30</th>
      <th rowspan="3" valign="top">0.3</th>
      <th>2</th>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>0.95</td>
      <td>0.06</td>
      <td>0.91</td>
      <td>0.23</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.10</td>
      <td>0.05</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.53</td>
      <td>0.06</td>
      <td>0.48</td>
      <td>0.14</td>
      <td>0.15</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.29</td>
      <td>0.06</td>
      <td>0.26</td>
      <td>0.12</td>
      <td>0.12</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>0.95</td>
      <td>0.96</td>
      <td>1.00</td>
      <td>0.03</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0.5</th>
      <th>2</th>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.12</td>
      <td>0.07</td>
      <td>0.12</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.90</td>
      <td>0.07</td>
      <td>0.85</td>
      <td>0.21</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.84</td>
      <td>0.05</td>
      <td>0.84</td>
      <td>0.30</td>
      <td>0.36</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.50</td>
      <td>0.08</td>
      <td>0.53</td>
      <td>0.20</td>
      <td>0.21</td>
      <td>1.00</td>
      <td>0.07</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0.7</th>
      <th>2</th>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.20</td>
      <td>0.05</td>
      <td>0.19</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.65</td>
      <td>0.04</td>
      <td>0.65</td>
      <td>0.11</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.25</td>
      <td>0.04</td>
      <td>0.25</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>0.97</td>
      <td>0.99</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.90</td>
      <td>0.05</td>
      <td>0.92</td>
      <td>0.63</td>
      <td>0.72</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.04</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>


## Decline regime: final analysis for the paper (Section 3.3)

Fixed design from the parameter exploration above: stable phase at 100 for the first 50 % of the series, then exponential decline to `100 / factor` at the last step. Main case: factor 10, relative noise 0.15. Sensitivity: factor 100. Stress test: relative noise 0.30. Factor 50 is only used for the depth-resolution AUC (50 vs 100). Cell C uses its own random generator; Cells A and B are not affected. Cell C produces the numbers of Tables 4 and 5 of the paper and the AUC values for the depth of the decline reported in the text of Section 3.3.


```python
# ------------------------------------------------------------------
# Cell C: final decline analysis for Section 3.3
# Requires Cells A and B (gen_decline, measures_with_d, draw_normal_pos, ...).
# ------------------------------------------------------------------
PAPER_NOISES  = [0.15, 0.30]        # main case, stress test (relative SD of the scatter)
PAPER_ONSET   = 0.5                 # decline starts after 50 % of the series
PAPER_FACTORS = [10, 50, 100]       # 10: main case, 100: sensitivity, 50: only for the 50-vs-100 AUC
N_PAPER       = 50000
PAPER_MEAS    = ["PV_hi", "PV_lo", "SDL", "CV", "D"]
rng_paper     = np.random.default_rng(2027)

runs, rows, arows = {}, [], []
for noise in PAPER_NOISES:
    runs[(noise, "stable")] = pd.DataFrame([measures_with_d(draw_normal_pos(rng_paper, 100, 100 * noise, N_STEPS))
                                            for _ in range(N_PAPER)])
    for F in PAPER_FACTORS:
        runs[(noise, F)] = pd.DataFrame([measures_with_d(gen_decline(rng_paper, F, PAPER_ONSET, noise))
                                         for _ in range(N_PAPER)])
    ref_mean = runs[(noise, "stable")][PAPER_MEAS].mean()
    for scen in ["stable"] + PAPER_FACTORS:
        df = runs[(noise, scen)]
        for m in PAPER_MEAS:
            rows.append(dict(noise=noise, scenario=scen, measure=m, mean=df[m].mean(), sd=df[m].std(),
                             response=df[m].mean() / ref_mean[m], pv_lo_mean=df["pv_lo"].mean()))
    for f1, f2 in [(10, 100), (50, 100)]:
        r = dict(noise=noise, pair=f"{f1} vs {f2}")
        for m in PAPER_MEAS:
            r[m] = auc(runs[(noise, f1)][m], runs[(noise, f2)][m])
        arows.append(r)

paper_end = pd.DataFrame(rows)
paper_end.to_csv(OUT_DIR / "decline_paper_end.csv", index=False)
print("End of the series (n = %d), mean and SD over %d series, response = mean / mean of the stable series:" % (N_STEPS, N_PAPER))
display(paper_end.pivot_table(index=["noise", "scenario"], columns="measure", values=["mean", "sd", "response"])
        .loc[:, (slice(None), PAPER_MEAS)].round(3))
paper_auc = pd.DataFrame(arows).set_index(["noise", "pair"])
paper_auc.to_csv(OUT_DIR / "decline_paper_auc.csv")
print("\nDepth resolution (AUC, deeper vs shallower decline):")
display(paper_auc.round(3))

# expanding window: detection power and mean index at selected prefix lengths
PD_FACTORS = [10, 100]
PD_DELAYS  = [10, 20, 30]
PD_KS      = list(range(10, N_STEPS + 1, 2))
PD_SHOW    = [30, 50, 60, 70, 80, 100]          # prefix lengths for the mean-index trajectory
N_PD_REF, N_PD = 5000, 2000
t0 = int(round(PAPER_ONSET * N_STEPS))

prow, traj = [], []
for noise in PAPER_NOISES:
    ref = {m: np.empty((N_PD_REF, len(PD_KS))) for m in PAPER_MEAS}
    for i in range(N_PD_REF):
        z = draw_normal_pos(rng_paper, 100, 100 * noise, N_STEPS)
        for j, k in enumerate(PD_KS):
            r = measures_with_d(z[:k])
            for m in PAPER_MEAS:
                ref[m][i, j] = r[m]
    thr = {m: np.quantile(ref[m], 0.95, axis=0) for m in PAPER_MEAS}
    for m in PAPER_MEAS:
        for k in PD_SHOW:
            traj.append(dict(noise=noise, factor="stable", measure=m, k=k, mean=ref[m][:, PD_KS.index(k)].mean()))
    for F in PD_FACTORS:
        hit  = {m: np.zeros((N_PD, len(PD_KS)), dtype=bool) for m in PAPER_MEAS}
        vals = {m: np.zeros((N_PD, len(PD_KS))) for m in PAPER_MEAS}
        for i in range(N_PD):
            z = gen_decline(rng_paper, F, PAPER_ONSET, noise)
            for j, k in enumerate(PD_KS):
                r = measures_with_d(z[:k])
                for m in PAPER_MEAS:
                    vals[m][i, j] = r[m]
                    hit[m][i, j] = r[m] > thr[m][j]
        for m in PAPER_MEAS:
            row = dict(noise=noise, factor=F, measure=m)
            for d in PD_DELAYS:
                row[f"power +{d}"] = hit[m][:, PD_KS.index(t0 + d)].mean()
            row["power end"] = hit[m][:, -1].mean()
            prow.append(row)
            for k in PD_SHOW:
                traj.append(dict(noise=noise, factor=F, measure=m, k=k, mean=vals[m][:, PD_KS.index(k)].mean()))
paper_det = pd.DataFrame(prow)
paper_det.to_csv(OUT_DIR / "decline_paper_detection.csv", index=False)
paper_traj = pd.DataFrame(traj)
paper_traj.to_csv(OUT_DIR / "decline_paper_trajectory.csv", index=False)
print("\nShare of decline series above the 95 % threshold of stable series of the same length (d steps after the onset):")
display(paper_det.pivot_table(index=["noise", "factor"], columns="measure",
                              values=["power +10", "power +20", "power +30", "power end"])
        .loc[:, (slice(None), PAPER_MEAS)].round(2))
print("\nMean index at prefix length k (stable = stable reference series):")
display(paper_traj.pivot_table(index=["noise", "factor", "k"], columns="measure", values="mean")[PAPER_MEAS].round(3))
```

    End of the series (n = 100), mean and SD over 50000 series, response = mean / mean of the stable series:
    End of the series (n = 100), mean and SD over 50000 series, response = mean / mean of the stable series:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
    </tr>
    <tr>
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>CV</th>
      <th>CV</th>
      <th>CV</th>
      <th>D</th>
      <th>D</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>scenario</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">0.15</th>
      <th>10</th>
      <td>0.469</td>
      <td>3.099</td>
      <td>0.006</td>
      <td>0.669</td>
      <td>4.019</td>
      <td>0.009</td>
      <td>0.771</td>
      <td>5.001</td>
      <td>0.016</td>
      <td>0.541</td>
      <td>3.614</td>
      <td>0.009</td>
      <td>0.175</td>
      <td>1.011</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.576</td>
      <td>3.803</td>
      <td>0.005</td>
      <td>0.884</td>
      <td>5.311</td>
      <td>0.005</td>
      <td>1.296</td>
      <td>8.400</td>
      <td>0.017</td>
      <td>0.699</td>
      <td>4.670</td>
      <td>0.007</td>
      <td>0.178</td>
      <td>1.032</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.604</td>
      <td>3.992</td>
      <td>0.004</td>
      <td>0.932</td>
      <td>5.601</td>
      <td>0.003</td>
      <td>1.530</td>
      <td>9.920</td>
      <td>0.019</td>
      <td>0.743</td>
      <td>4.961</td>
      <td>0.007</td>
      <td>0.180</td>
      <td>1.041</td>
      <td>0.018</td>
    </tr>
    <tr>
      <th>stable</th>
      <td>0.151</td>
      <td>1.000</td>
      <td>0.010</td>
      <td>0.166</td>
      <td>1.000</td>
      <td>0.012</td>
      <td>0.154</td>
      <td>1.000</td>
      <td>0.012</td>
      <td>0.150</td>
      <td>1.000</td>
      <td>0.011</td>
      <td>0.173</td>
      <td>1.000</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">0.30</th>
      <th>10</th>
      <td>0.508</td>
      <td>1.821</td>
      <td>0.010</td>
      <td>0.709</td>
      <td>1.979</td>
      <td>0.024</td>
      <td>0.833</td>
      <td>2.380</td>
      <td>0.038</td>
      <td>0.614</td>
      <td>2.056</td>
      <td>0.020</td>
      <td>0.373</td>
      <td>1.002</td>
      <td>0.042</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.608</td>
      <td>2.182</td>
      <td>0.008</td>
      <td>0.894</td>
      <td>2.496</td>
      <td>0.009</td>
      <td>1.331</td>
      <td>3.802</td>
      <td>0.036</td>
      <td>0.765</td>
      <td>2.562</td>
      <td>0.019</td>
      <td>0.374</td>
      <td>1.003</td>
      <td>0.042</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.635</td>
      <td>2.279</td>
      <td>0.007</td>
      <td>0.935</td>
      <td>2.608</td>
      <td>0.005</td>
      <td>1.547</td>
      <td>4.420</td>
      <td>0.033</td>
      <td>0.808</td>
      <td>2.704</td>
      <td>0.018</td>
      <td>0.368</td>
      <td>0.989</td>
      <td>0.042</td>
    </tr>
    <tr>
      <th>stable</th>
      <td>0.279</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.358</td>
      <td>1.000</td>
      <td>0.056</td>
      <td>0.350</td>
      <td>1.000</td>
      <td>0.050</td>
      <td>0.299</td>
      <td>1.000</td>
      <td>0.023</td>
      <td>0.372</td>
      <td>1.000</td>
      <td>0.043</td>
    </tr>
  </tbody>
</table>
</div>


    
    Depth resolution (AUC, deeper vs shallower decline):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>pair</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0.15</th>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.582</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.524</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.30</th>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.470</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.995</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.949</td>
      <td>0.465</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
      <th>mean</th>
      <th>response</th>
      <th>sd</th>
    </tr>
    <tr>
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>CV</th>
      <th>CV</th>
      <th>CV</th>
      <th>D</th>
      <th>D</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>scenario</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">0.15</th>
      <th>10</th>
      <td>0.469</td>
      <td>3.099</td>
      <td>0.006</td>
      <td>0.669</td>
      <td>4.019</td>
      <td>0.009</td>
      <td>0.771</td>
      <td>5.001</td>
      <td>0.016</td>
      <td>0.541</td>
      <td>3.614</td>
      <td>0.009</td>
      <td>0.175</td>
      <td>1.011</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.576</td>
      <td>3.803</td>
      <td>0.005</td>
      <td>0.884</td>
      <td>5.311</td>
      <td>0.005</td>
      <td>1.296</td>
      <td>8.400</td>
      <td>0.017</td>
      <td>0.699</td>
      <td>4.670</td>
      <td>0.007</td>
      <td>0.178</td>
      <td>1.032</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.604</td>
      <td>3.992</td>
      <td>0.004</td>
      <td>0.932</td>
      <td>5.601</td>
      <td>0.003</td>
      <td>1.530</td>
      <td>9.920</td>
      <td>0.019</td>
      <td>0.743</td>
      <td>4.961</td>
      <td>0.007</td>
      <td>0.180</td>
      <td>1.041</td>
      <td>0.018</td>
    </tr>
    <tr>
      <th>stable</th>
      <td>0.151</td>
      <td>1.000</td>
      <td>0.010</td>
      <td>0.166</td>
      <td>1.000</td>
      <td>0.012</td>
      <td>0.154</td>
      <td>1.000</td>
      <td>0.012</td>
      <td>0.150</td>
      <td>1.000</td>
      <td>0.011</td>
      <td>0.173</td>
      <td>1.000</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">0.30</th>
      <th>10</th>
      <td>0.508</td>
      <td>1.821</td>
      <td>0.010</td>
      <td>0.709</td>
      <td>1.979</td>
      <td>0.024</td>
      <td>0.833</td>
      <td>2.380</td>
      <td>0.038</td>
      <td>0.614</td>
      <td>2.056</td>
      <td>0.020</td>
      <td>0.373</td>
      <td>1.002</td>
      <td>0.042</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.608</td>
      <td>2.182</td>
      <td>0.008</td>
      <td>0.894</td>
      <td>2.496</td>
      <td>0.009</td>
      <td>1.331</td>
      <td>3.802</td>
      <td>0.036</td>
      <td>0.765</td>
      <td>2.562</td>
      <td>0.019</td>
      <td>0.374</td>
      <td>1.003</td>
      <td>0.042</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.635</td>
      <td>2.279</td>
      <td>0.007</td>
      <td>0.935</td>
      <td>2.608</td>
      <td>0.005</td>
      <td>1.547</td>
      <td>4.420</td>
      <td>0.033</td>
      <td>0.808</td>
      <td>2.704</td>
      <td>0.018</td>
      <td>0.368</td>
      <td>0.989</td>
      <td>0.042</td>
    </tr>
    <tr>
      <th>stable</th>
      <td>0.279</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.358</td>
      <td>1.000</td>
      <td>0.056</td>
      <td>0.350</td>
      <td>1.000</td>
      <td>0.050</td>
      <td>0.299</td>
      <td>1.000</td>
      <td>0.023</td>
      <td>0.372</td>
      <td>1.000</td>
      <td>0.043</td>
    </tr>
  </tbody>
</table>
</div>


    
    Depth resolution (AUC, deeper vs shallower decline):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>pair</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0.15</th>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.582</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>1.000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.524</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.30</th>
      <th>10 vs 100</th>
      <td>1.000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.470</td>
    </tr>
    <tr>
      <th>50 vs 100</th>
      <td>0.995</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.949</td>
      <td>0.465</td>
    </tr>
  </tbody>
</table>
</div>


    
    Share of decline series above the 95 % threshold of stable series of the same length (d steps after the onset):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
    </tr>
    <tr>
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>CV</th>
      <th>CV</th>
      <th>CV</th>
      <th>CV</th>
      <th>D</th>
      <th>D</th>
      <th>D</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0.15</th>
      <th>10</th>
      <td>0.64</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.66</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.67</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.63</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.30</th>
      <th>10</th>
      <td>0.13</td>
      <td>0.85</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>0.27</td>
      <td>0.99</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.32</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>0.85</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.50</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.18</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.49</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
</div>


    
    Share of decline series above the 95 % threshold of stable series of the same length (d steps after the onset):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
      <th>power +10</th>
      <th>power +20</th>
      <th>power +30</th>
      <th>power end</th>
    </tr>
    <tr>
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>SDL</th>
      <th>CV</th>
      <th>CV</th>
      <th>CV</th>
      <th>CV</th>
      <th>D</th>
      <th>D</th>
      <th>D</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0.15</th>
      <th>10</th>
      <td>0.64</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.66</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.67</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.63</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.30</th>
      <th>10</th>
      <td>0.13</td>
      <td>0.85</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>0.27</td>
      <td>0.99</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.32</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>0.85</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.50</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.18</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.49</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
</div>


    
    Mean index at prefix length k (stable = stable reference series):
    
    Mean index at prefix length k (stable = stable reference series):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>factor</th>
      <th>k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">0.15</th>
      <th rowspan="6" valign="top">10</th>
      <th>30</th>
      <td>0.152</td>
      <td>0.166</td>
      <td>0.153</td>
      <td>0.149</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.152</td>
      <td>0.167</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.178</td>
      <td>0.201</td>
      <td>0.189</td>
      <td>0.178</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.251</td>
      <td>0.308</td>
      <td>0.304</td>
      <td>0.259</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.332</td>
      <td>0.440</td>
      <td>0.452</td>
      <td>0.355</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.469</td>
      <td>0.669</td>
      <td>0.772</td>
      <td>0.541</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">100</th>
      <th>30</th>
      <td>0.150</td>
      <td>0.165</td>
      <td>0.152</td>
      <td>0.148</td>
      <td>0.171</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.149</td>
      <td>0.172</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.219</td>
      <td>0.270</td>
      <td>0.268</td>
      <td>0.226</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.344</td>
      <td>0.514</td>
      <td>0.546</td>
      <td>0.373</td>
      <td>0.177</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.453</td>
      <td>0.727</td>
      <td>0.865</td>
      <td>0.513</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.605</td>
      <td>0.932</td>
      <td>1.531</td>
      <td>0.743</td>
      <td>0.180</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">stable</th>
      <th>30</th>
      <td>0.152</td>
      <td>0.166</td>
      <td>0.153</td>
      <td>0.149</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.30</th>
      <th rowspan="6" valign="top">10</th>
      <th>30</th>
      <td>0.278</td>
      <td>0.347</td>
      <td>0.340</td>
      <td>0.297</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.278</td>
      <td>0.352</td>
      <td>0.345</td>
      <td>0.297</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.291</td>
      <td>0.370</td>
      <td>0.364</td>
      <td>0.314</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.336</td>
      <td>0.435</td>
      <td>0.437</td>
      <td>0.370</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.395</td>
      <td>0.528</td>
      <td>0.551</td>
      <td>0.446</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.507</td>
      <td>0.708</td>
      <td>0.831</td>
      <td>0.613</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">100</th>
      <th>30</th>
      <td>0.278</td>
      <td>0.349</td>
      <td>0.342</td>
      <td>0.297</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.278</td>
      <td>0.353</td>
      <td>0.346</td>
      <td>0.297</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.318</td>
      <td>0.412</td>
      <td>0.411</td>
      <td>0.345</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.413</td>
      <td>0.587</td>
      <td>0.630</td>
      <td>0.462</td>
      <td>0.374</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.504</td>
      <td>0.758</td>
      <td>0.919</td>
      <td>0.587</td>
      <td>0.374</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.635</td>
      <td>0.935</td>
      <td>1.548</td>
      <td>0.807</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">stable</th>
      <th>30</th>
      <td>0.278</td>
      <td>0.349</td>
      <td>0.343</td>
      <td>0.297</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.278</td>
      <td>0.352</td>
      <td>0.345</td>
      <td>0.298</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.278</td>
      <td>0.353</td>
      <td>0.346</td>
      <td>0.298</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.278</td>
      <td>0.354</td>
      <td>0.347</td>
      <td>0.298</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.278</td>
      <td>0.355</td>
      <td>0.347</td>
      <td>0.298</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.278</td>
      <td>0.357</td>
      <td>0.349</td>
      <td>0.298</td>
      <td>0.372</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>factor</th>
      <th>k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="18" valign="top">0.15</th>
      <th rowspan="6" valign="top">10</th>
      <th>30</th>
      <td>0.152</td>
      <td>0.166</td>
      <td>0.153</td>
      <td>0.149</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.152</td>
      <td>0.167</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.178</td>
      <td>0.201</td>
      <td>0.189</td>
      <td>0.178</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.251</td>
      <td>0.308</td>
      <td>0.304</td>
      <td>0.259</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.332</td>
      <td>0.440</td>
      <td>0.452</td>
      <td>0.355</td>
      <td>0.174</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.469</td>
      <td>0.669</td>
      <td>0.772</td>
      <td>0.541</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">100</th>
      <th>30</th>
      <td>0.150</td>
      <td>0.165</td>
      <td>0.152</td>
      <td>0.148</td>
      <td>0.171</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.149</td>
      <td>0.172</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.219</td>
      <td>0.270</td>
      <td>0.268</td>
      <td>0.226</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.344</td>
      <td>0.514</td>
      <td>0.546</td>
      <td>0.373</td>
      <td>0.177</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.453</td>
      <td>0.727</td>
      <td>0.865</td>
      <td>0.513</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.605</td>
      <td>0.932</td>
      <td>1.531</td>
      <td>0.743</td>
      <td>0.180</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">stable</th>
      <th>30</th>
      <td>0.152</td>
      <td>0.166</td>
      <td>0.153</td>
      <td>0.149</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.151</td>
      <td>0.166</td>
      <td>0.154</td>
      <td>0.150</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th rowspan="18" valign="top">0.30</th>
      <th rowspan="6" valign="top">10</th>
      <th>30</th>
      <td>0.278</td>
      <td>0.347</td>
      <td>0.340</td>
      <td>0.297</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.278</td>
      <td>0.352</td>
      <td>0.345</td>
      <td>0.297</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.291</td>
      <td>0.370</td>
      <td>0.364</td>
      <td>0.314</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.336</td>
      <td>0.435</td>
      <td>0.437</td>
      <td>0.370</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.395</td>
      <td>0.528</td>
      <td>0.551</td>
      <td>0.446</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.507</td>
      <td>0.708</td>
      <td>0.831</td>
      <td>0.613</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">100</th>
      <th>30</th>
      <td>0.278</td>
      <td>0.349</td>
      <td>0.342</td>
      <td>0.297</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.278</td>
      <td>0.353</td>
      <td>0.346</td>
      <td>0.297</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.318</td>
      <td>0.412</td>
      <td>0.411</td>
      <td>0.345</td>
      <td>0.373</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.413</td>
      <td>0.587</td>
      <td>0.630</td>
      <td>0.462</td>
      <td>0.374</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.504</td>
      <td>0.758</td>
      <td>0.919</td>
      <td>0.587</td>
      <td>0.374</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.635</td>
      <td>0.935</td>
      <td>1.548</td>
      <td>0.807</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">stable</th>
      <th>30</th>
      <td>0.278</td>
      <td>0.349</td>
      <td>0.343</td>
      <td>0.297</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.278</td>
      <td>0.352</td>
      <td>0.345</td>
      <td>0.298</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.278</td>
      <td>0.353</td>
      <td>0.346</td>
      <td>0.298</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.278</td>
      <td>0.354</td>
      <td>0.347</td>
      <td>0.298</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.278</td>
      <td>0.355</td>
      <td>0.347</td>
      <td>0.298</td>
      <td>0.371</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.278</td>
      <td>0.357</td>
      <td>0.349</td>
      <td>0.298</td>
      <td>0.372</td>
    </tr>
  </tbody>
</table>
</div>


## Can crashes be told from ordinary fluctuations of unknown amplitude?

Question raised during the revision: a larger *response factor* of an index (e.g. 10-fold instead of 4-fold in the Rare regime) is a statement about its scale, not about its ability to separate crashes from ordinary fluctuations. Here the separation is measured directly. Ordinary series: fluctuations of unknown amplitude (Gaussian with relative SD ~ U(0.05, 0.30), or log-normal with log-SD ~ U(0.05, 0.50)). Crash series: the same family and amplitude distribution plus five crashes at random positions to `100 / f` (relative SD 0.5, as in the Rare regime). AUC = probability that a crash series has a higher index value than an ordinary series (0.5 = no separation). Exploratory. The Gaussian version with random amplitude is repeated, together with known-amplitude controls, in Cell F (Table 6 of the paper); the log-normal variant is not used in the paper.


```python
# ------------------------------------------------------------------
# Cell D: discrimination of crash series from ordinary fluctuations of unknown amplitude
# ------------------------------------------------------------------
D_N       = 10000
D_FACTORS = [2, 5, 10, 100]                 # crash depth: abundance during crashes = 100 / f
D_N_CRASH = 5
D_MEAS    = ["PV_hi", "PV_lo", "SDL", "CV", "D"]
rng_d     = np.random.default_rng(8080)

def ordinary_gauss(rng):
    s = rng.uniform(0.05, 0.30)
    return draw_normal_pos(rng, 100, 100 * s, N_STEPS)

def ordinary_lognorm(rng):
    s = rng.uniform(0.05, 0.50)
    return np.maximum(np.round(100 * np.exp(rng.normal(0, s, N_STEPS))), 1)

def with_crashes(rng, base, f):
    z = base(rng).copy()
    idx = rng.choice(N_STEPS, D_N_CRASH, replace=False)
    z[idx] = draw_around_path(rng, np.full(D_N_CRASH, 100.0 / f), 0.5)
    return z

drows = []
for fam, base in [("Gaussian, rel. SD 0.05-0.30", ordinary_gauss), ("log-normal, log-SD 0.05-0.50", ordinary_lognorm)]:
    ordv = pd.DataFrame([measures_with_d(base(rng_d)) for _ in range(D_N)])
    for f in D_FACTORS:
        cr = pd.DataFrame([measures_with_d(with_crashes(rng_d, base, f)) for _ in range(D_N)])
        r = dict(family=fam, crash_factor=f)
        for m in D_MEAS:
            r[m] = auc(ordv[m], cr[m])
        drows.append(r)
crash_auc = pd.DataFrame(drows).set_index(["family", "crash_factor"])
crash_auc.to_csv(OUT_DIR / "crash_vs_ordinary_auc.csv")
print("AUC: crash series (5 crashes to 100/f) vs ordinary fluctuations of unknown amplitude (%d series per class):" % D_N)
display(crash_auc.round(3))
```

    AUC: crash series (5 crashes to 100/f) vs ordinary fluctuations of unknown amplitude (10000 series per class):
    AUC: crash series (5 crashes to 100/f) vs ordinary fluctuations of unknown amplitude (10000 series per class):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>family</th>
      <th>crash_factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Gaussian, rel. SD 0.05-0.30</th>
      <th>2</th>
      <td>0.633</td>
      <td>0.755</td>
      <td>0.796</td>
      <td>0.665</td>
      <td>0.677</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.741</td>
      <td>0.976</td>
      <td>0.989</td>
      <td>0.814</td>
      <td>0.863</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.771</td>
      <td>0.999</td>
      <td>1.000</td>
      <td>0.867</td>
      <td>0.946</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.800</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.914</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">log-normal, log-SD 0.05-0.50</th>
      <th>2</th>
      <td>0.573</td>
      <td>0.656</td>
      <td>0.671</td>
      <td>0.569</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.652</td>
      <td>0.887</td>
      <td>0.900</td>
      <td>0.639</td>
      <td>0.735</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.678</td>
      <td>0.993</td>
      <td>0.993</td>
      <td>0.667</td>
      <td>0.821</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.702</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.695</td>
      <td>0.977</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>family</th>
      <th>crash_factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Gaussian, rel. SD 0.05-0.30</th>
      <th>2</th>
      <td>0.633</td>
      <td>0.755</td>
      <td>0.796</td>
      <td>0.665</td>
      <td>0.677</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.741</td>
      <td>0.976</td>
      <td>0.989</td>
      <td>0.814</td>
      <td>0.863</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.771</td>
      <td>0.999</td>
      <td>1.000</td>
      <td>0.867</td>
      <td>0.946</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.800</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.914</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">log-normal, log-SD 0.05-0.50</th>
      <th>2</th>
      <td>0.573</td>
      <td>0.656</td>
      <td>0.671</td>
      <td>0.569</td>
      <td>0.598</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.652</td>
      <td>0.887</td>
      <td>0.900</td>
      <td>0.639</td>
      <td>0.735</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.678</td>
      <td>0.993</td>
      <td>0.993</td>
      <td>0.667</td>
      <td>0.821</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.702</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.695</td>
      <td>0.977</td>
    </tr>
  </tbody>
</table>
</div>


## Data for the trajectory figure (decline regime, expanding window)

Two quantities per prefix length k for the decline regime with f = 10 and onset after 50 steps, at relative SD 0.15 (main case) and 0.30 (stress test): the mean response (mean index at length k divided by the mean of the same index in stable series of length k; Fig. panel a) and the detection power (share of decline series above the 95th percentile of the same index in stable series of length k; Fig. panel b). Both are written to `decline_trajectory_fig.csv` for all k (step 2); the cell displays them for selected k only.


```python
# ------------------------------------------------------------------
# Cell E: trajectory data for the figure (written to decline_trajectory_fig.csv)
# ------------------------------------------------------------------
TR_NOISES, TR_F, TR_ONSET = [0.15, 0.30], 10, 0.5
TR_KS   = list(range(10, N_STEPS + 1, 2))
TR_MEAS = ["PV_hi", "PV_lo", "SDL", "CV", "D"]
N_TR_REF, N_TR = 5000, 2000
rng_tr = np.random.default_rng(6060)

trows = []
for noise in TR_NOISES:
    ref = {m: np.empty((N_TR_REF, len(TR_KS))) for m in TR_MEAS}
    for i in range(N_TR_REF):
        z = draw_normal_pos(rng_tr, 100, 100 * noise, N_STEPS)
        for j, k in enumerate(TR_KS):
            r = measures_with_d(z[:k])
            for m in TR_MEAS:
                ref[m][i, j] = r[m]
    thr = {m: np.quantile(ref[m], 0.95, axis=0) for m in TR_MEAS}
    vals = {m: np.empty((N_TR, len(TR_KS))) for m in TR_MEAS}
    for i in range(N_TR):
        z = gen_decline(rng_tr, TR_F, TR_ONSET, noise)
        for j, k in enumerate(TR_KS):
            r = measures_with_d(z[:k])
            for m in TR_MEAS:
                vals[m][i, j] = r[m]
    for m in TR_MEAS:
        for j, k in enumerate(TR_KS):
            trows.append(dict(noise=noise, measure=m, k=k,
                              mean_stable=ref[m][:, j].mean(), mean_decline=vals[m][:, j].mean(),
                              response=vals[m][:, j].mean() / ref[m][:, j].mean(),
                              power=(vals[m][:, j] > thr[m][j]).mean()))
traj_fig = pd.DataFrame(trows)
traj_fig.to_csv(OUT_DIR / "decline_trajectory_fig.csv", index=False)
print("saved", len(traj_fig), "rows")
SHOW_K = [50, 60, 70, 80, 100]
print("\nMean response (mean index at length k / mean index of stable series of length k):")
display(traj_fig[traj_fig.k.isin(SHOW_K)].pivot_table(
    index=["noise", "k"], columns="measure", values="response")[TR_MEAS].round(2))
print("\nDetection power (share of decline series above the 95th percentile of stable series of length k; nominal 0.05):")
display(traj_fig[traj_fig.k.isin(SHOW_K)].pivot_table(
    index=["noise", "k"], columns="measure", values="power")[TR_MEAS].round(2))
```

    saved 460 rows
    
    Mean response (mean index at length k / mean index of stable series of length k):
    saved 460 rows
    
    Mean response (mean index at length k / mean index of stable series of length k):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0.15</th>
      <th>50</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1.18</td>
      <td>1.21</td>
      <td>1.23</td>
      <td>1.19</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1.66</td>
      <td>1.85</td>
      <td>1.97</td>
      <td>1.73</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2.19</td>
      <td>2.65</td>
      <td>2.93</td>
      <td>2.37</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>100</th>
      <td>3.10</td>
      <td>4.01</td>
      <td>4.99</td>
      <td>3.61</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.30</th>
      <th>50</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1.04</td>
      <td>1.05</td>
      <td>1.05</td>
      <td>1.05</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1.21</td>
      <td>1.22</td>
      <td>1.25</td>
      <td>1.24</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1.42</td>
      <td>1.48</td>
      <td>1.58</td>
      <td>1.50</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.82</td>
      <td>1.98</td>
      <td>2.38</td>
      <td>2.06</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    Detection power (share of decline series above the 95th percentile of stable series of length k; nominal 0.05):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0.15</th>
      <th>50</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1.18</td>
      <td>1.21</td>
      <td>1.23</td>
      <td>1.19</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1.66</td>
      <td>1.85</td>
      <td>1.97</td>
      <td>1.73</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2.19</td>
      <td>2.65</td>
      <td>2.93</td>
      <td>2.37</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>100</th>
      <td>3.10</td>
      <td>4.01</td>
      <td>4.99</td>
      <td>3.61</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.30</th>
      <th>50</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1.04</td>
      <td>1.05</td>
      <td>1.05</td>
      <td>1.05</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1.21</td>
      <td>1.22</td>
      <td>1.25</td>
      <td>1.24</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1.42</td>
      <td>1.48</td>
      <td>1.58</td>
      <td>1.50</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.82</td>
      <td>1.98</td>
      <td>2.38</td>
      <td>2.06</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0.15</th>
      <th>50</th>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.66</td>
      <td>0.68</td>
      <td>0.68</td>
      <td>0.64</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.30</th>
      <th>50</th>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.12</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.12</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.82</td>
      <td>0.25</td>
      <td>0.31</td>
      <td>0.82</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1.00</td>
      <td>0.99</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>


    
    Detection power (share of decline series above the 95th percentile of stable series of length k; nominal 0.05):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>measure</th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>noise</th>
      <th>k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0.15</th>
      <th>50</th>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.66</td>
      <td>0.68</td>
      <td>0.68</td>
      <td>0.64</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">0.30</th>
      <th>50</th>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.12</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.12</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.82</td>
      <td>0.25</td>
      <td>0.31</td>
      <td>0.82</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1.00</td>
      <td>0.99</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>


## Controls: why do the crash analysis (Cell D) and the decline detection (Cells B, C, E) rank the indices differently?

Two design differences: (1) in Cell D the amplitude of the ordinary fluctuations is **unknown** (drawn at random per series), in the decline detection the threshold is calibrated on stable series with the **same, known** amplitude; (2) a crash series contains isolated lows far below anything ordinary fluctuations produce, a decline moves the bulk of the observations. Control 1 repeats the crash comparison with a known (fixed) amplitude and with different ranges of unknown amplitude. Control 2 repeats the decline comparison (as AUC against stable series of the same length) with known and unknown amplitude, at series length 70 (20 steps after the onset) and 100.

Cell F produces Table 6 of the paper (crash comparison) and the decline AUC values that Section 3.3 mentions as "not shown".


```python
# ------------------------------------------------------------------
# Cell F: controls (known vs unknown amplitude; crash vs decline)
# ------------------------------------------------------------------
N_CTRL   = 5000
CTRL_MEAS = ["PV_hi", "PV_lo", "SDL", "CV", "D"]
rng_c    = np.random.default_rng(1234)

def gauss_series(s):
    return draw_normal_pos(rng_c, 100, 100 * s, N_STEPS)

def add_crashes(z, f, n_crash=5):
    z = z.copy()
    idx = rng_c.choice(N_STEPS, n_crash, replace=False)
    z[idx] = draw_around_path(rng_c, np.full(n_crash, 100.0 / f), 0.5)
    return z

# Control 1: crash series vs ordinary series; amplitude known (fixed) or unknown (random per series)
amp_settings = [("fixed s = 0.05", lambda: 0.05), ("fixed s = 0.15", lambda: 0.15), ("fixed s = 0.30", lambda: 0.30),
                ("random s ~ U(0.14, 0.16)", lambda: rng_c.uniform(0.14, 0.16)),
                ("random s ~ U(0.05, 0.30)", lambda: rng_c.uniform(0.05, 0.30)),
                ("random s ~ U(0.02, 0.50)", lambda: rng_c.uniform(0.02, 0.50))]
c1 = []
for label, sfun in amp_settings:
    ordv = pd.DataFrame([measures_with_d(gauss_series(sfun())) for _ in range(N_CTRL)])
    for f in [2, 5, 10]:
        cr = pd.DataFrame([measures_with_d(add_crashes(gauss_series(sfun()), f)) for _ in range(N_CTRL)])
        c1.append(dict(amplitude=label, crash_factor=f, **{m: auc(ordv[m], cr[m]) for m in CTRL_MEAS}))
ctrl_crash = pd.DataFrame(c1).set_index(["amplitude", "crash_factor"])
ctrl_crash.to_csv(OUT_DIR / "control_crash_auc.csv")
print("Control 1: AUC, 5 crashes to 100/f vs ordinary Gaussian fluctuations")
display(ctrl_crash.round(3))

# Control 2: decline series vs stable series of the same length; amplitude known or unknown
c2 = []
for label, sfun in [("fixed s = 0.15", lambda: 0.15), ("fixed s = 0.30", lambda: 0.30),
                    ("random s ~ U(0.05, 0.30)", lambda: rng_c.uniform(0.05, 0.30))]:
    stable = [gauss_series(sfun()) for _ in range(N_CTRL)]
    for k in [70, 100]:
        ref = pd.DataFrame([measures_with_d(z[:k]) for z in stable])
        for F in [2, 10]:
            dd = pd.DataFrame([measures_with_d(gen_decline(rng_c, F, 0.5, sfun())[:k]) for _ in range(N_CTRL)])
            c2.append(dict(amplitude=label, length=k, factor=F, **{m: auc(ref[m], dd[m]) for m in CTRL_MEAS}))
ctrl_decline = pd.DataFrame(c2).set_index(["amplitude", "length", "factor"])
ctrl_decline.to_csv(OUT_DIR / "control_decline_auc.csv")
print("\nControl 2: AUC, decline (onset after 50 steps, to 100/f) vs stable series of the same length")
display(ctrl_decline.round(3))
```

    Control 1: AUC, 5 crashes to 100/f vs ordinary Gaussian fluctuations
    Control 1: AUC, 5 crashes to 100/f vs ordinary Gaussian fluctuations
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>amplitude</th>
      <th>crash_factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">fixed s = 0.05</th>
      <th>2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">fixed s = 0.15</th>
      <th>2</th>
      <td>0.974</td>
      <td>0.990</td>
      <td>0.994</td>
      <td>0.984</td>
      <td>0.972</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">fixed s = 0.30</th>
      <th>2</th>
      <td>0.773</td>
      <td>0.766</td>
      <td>0.769</td>
      <td>0.768</td>
      <td>0.752</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.967</td>
      <td>0.966</td>
      <td>0.978</td>
      <td>0.964</td>
      <td>0.973</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.989</td>
      <td>0.991</td>
      <td>0.999</td>
      <td>0.990</td>
      <td>0.997</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">random s ~ U(0.14, 0.16)</th>
      <th>2</th>
      <td>0.963</td>
      <td>0.987</td>
      <td>0.991</td>
      <td>0.977</td>
      <td>0.964</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">random s ~ U(0.05, 0.30)</th>
      <th>2</th>
      <td>0.625</td>
      <td>0.746</td>
      <td>0.788</td>
      <td>0.659</td>
      <td>0.668</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.730</td>
      <td>0.972</td>
      <td>0.986</td>
      <td>0.807</td>
      <td>0.854</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.766</td>
      <td>0.998</td>
      <td>0.999</td>
      <td>0.862</td>
      <td>0.944</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">random s ~ U(0.02, 0.50)</th>
      <th>2</th>
      <td>0.557</td>
      <td>0.610</td>
      <td>0.625</td>
      <td>0.572</td>
      <td>0.573</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.633</td>
      <td>0.783</td>
      <td>0.796</td>
      <td>0.657</td>
      <td>0.690</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.661</td>
      <td>0.907</td>
      <td>0.912</td>
      <td>0.692</td>
      <td>0.768</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>amplitude</th>
      <th>crash_factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">fixed s = 0.05</th>
      <th>2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">fixed s = 0.15</th>
      <th>2</th>
      <td>0.974</td>
      <td>0.990</td>
      <td>0.994</td>
      <td>0.984</td>
      <td>0.972</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">fixed s = 0.30</th>
      <th>2</th>
      <td>0.773</td>
      <td>0.766</td>
      <td>0.769</td>
      <td>0.768</td>
      <td>0.752</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.967</td>
      <td>0.966</td>
      <td>0.978</td>
      <td>0.964</td>
      <td>0.973</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.989</td>
      <td>0.991</td>
      <td>0.999</td>
      <td>0.990</td>
      <td>0.997</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">random s ~ U(0.14, 0.16)</th>
      <th>2</th>
      <td>0.963</td>
      <td>0.987</td>
      <td>0.991</td>
      <td>0.977</td>
      <td>0.964</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">random s ~ U(0.05, 0.30)</th>
      <th>2</th>
      <td>0.625</td>
      <td>0.746</td>
      <td>0.788</td>
      <td>0.659</td>
      <td>0.668</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.730</td>
      <td>0.972</td>
      <td>0.986</td>
      <td>0.807</td>
      <td>0.854</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.766</td>
      <td>0.998</td>
      <td>0.999</td>
      <td>0.862</td>
      <td>0.944</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">random s ~ U(0.02, 0.50)</th>
      <th>2</th>
      <td>0.557</td>
      <td>0.610</td>
      <td>0.625</td>
      <td>0.572</td>
      <td>0.573</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.633</td>
      <td>0.783</td>
      <td>0.796</td>
      <td>0.657</td>
      <td>0.690</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.661</td>
      <td>0.907</td>
      <td>0.912</td>
      <td>0.692</td>
      <td>0.768</td>
    </tr>
  </tbody>
</table>
</div>


    
    Control 2: AUC, decline (onset after 50 steps, to 100/f) vs stable series of the same length
    
    Control 2: AUC, decline (onset after 50 steps, to 100/f) vs stable series of the same length
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>amplitude</th>
      <th>length</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">fixed s = 0.15</th>
      <th rowspan="2" valign="top">70</th>
      <th>2</th>
      <td>0.828</td>
      <td>0.825</td>
      <td>0.818</td>
      <td>0.827</td>
      <td>0.502</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.524</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">100</th>
      <th>2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.508</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.542</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">fixed s = 0.30</th>
      <th rowspan="2" valign="top">70</th>
      <th>2</th>
      <td>0.587</td>
      <td>0.558</td>
      <td>0.554</td>
      <td>0.598</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.969</td>
      <td>0.887</td>
      <td>0.888</td>
      <td>0.969</td>
      <td>0.508</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">100</th>
      <th>2</th>
      <td>0.970</td>
      <td>0.862</td>
      <td>0.860</td>
      <td>0.980</td>
      <td>0.488</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>0.997</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.505</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">random s ~ U(0.05, 0.30)</th>
      <th rowspan="2" valign="top">70</th>
      <th>2</th>
      <td>0.570</td>
      <td>0.568</td>
      <td>0.568</td>
      <td>0.570</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.873</td>
      <td>0.892</td>
      <td>0.907</td>
      <td>0.870</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">100</th>
      <th>2</th>
      <td>0.873</td>
      <td>0.854</td>
      <td>0.850</td>
      <td>0.861</td>
      <td>0.502</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.505</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>PV_hi</th>
      <th>PV_lo</th>
      <th>SDL</th>
      <th>CV</th>
      <th>D</th>
    </tr>
    <tr>
      <th>amplitude</th>
      <th>length</th>
      <th>factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">fixed s = 0.15</th>
      <th rowspan="2" valign="top">70</th>
      <th>2</th>
      <td>0.828</td>
      <td>0.825</td>
      <td>0.818</td>
      <td>0.827</td>
      <td>0.502</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.524</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">100</th>
      <th>2</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.508</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.542</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">fixed s = 0.30</th>
      <th rowspan="2" valign="top">70</th>
      <th>2</th>
      <td>0.587</td>
      <td>0.558</td>
      <td>0.554</td>
      <td>0.598</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.969</td>
      <td>0.887</td>
      <td>0.888</td>
      <td>0.969</td>
      <td>0.508</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">100</th>
      <th>2</th>
      <td>0.970</td>
      <td>0.862</td>
      <td>0.860</td>
      <td>0.980</td>
      <td>0.488</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>0.997</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.505</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">random s ~ U(0.05, 0.30)</th>
      <th rowspan="2" valign="top">70</th>
      <th>2</th>
      <td>0.570</td>
      <td>0.568</td>
      <td>0.568</td>
      <td>0.570</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.873</td>
      <td>0.892</td>
      <td>0.907</td>
      <td>0.870</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">100</th>
      <th>2</th>
      <td>0.873</td>
      <td>0.854</td>
      <td>0.850</td>
      <td>0.861</td>
      <td>0.502</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.505</td>
    </tr>
  </tbody>
</table>
</div>

