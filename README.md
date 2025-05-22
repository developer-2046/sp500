# S\&P 500 Network-Entropy Toolkit

*A Swiss-army collection of one-off scripts for spectral, network and RMT analysis of US equity markets*

> **TL;DR** – This repo is **not** a single pipeline.
> Each script tackles a discrete quant-research question: entropy spikes, λ<sub>max</sub> eigen-vector drift, MST degree, Marchenko-Pastur gaps, community flips, box-plots of peak/valley windows, and more.
> Clone it, cherry-pick the script you need, and point it at `SP500_25May24_cleaned_filled.csv`.

---

## 0 · Directory map (every file that matters)

| path                     | what it is                                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `NeCe.py`                | **ALL-IN-ONE dashboard** – loops 20-day windows, computes Louvain entropy, global efficiency and mean ρ, plots three stacked time-series. |
| `adjacency matrix.py`    | Builds a **0.6-threshold binary graph** from the full sample, saves `adjacency_matrix.csv`, draws a spring-layout PNG.                    |
| `average degree.py`      | For each 20-day window: ⟨k⟩ of weighted graph, dual-axis line chart with mean correlation.                                                |
| `similarity_matrix.py`   | ε-power map (ε = 0.6) → √2(1−ρ) distance → pairwise similarity heat-map.                                                                  |
| `test.py/2/3.py`         | Quick “does this look right?” plots with different thresholds/layouts.                                                                    |
| **`distance/`**          | *Glue code around eigen-vectors* – see full table below (14 scripts). Also hosts >400 PNG eigen-vector snapshots + parquet matrices.      |
| **`shannonentropy/`**    | `main.py` : eigen-entropy per window → `shannonentropy.csv`.                                                                              |
| **`multiplex network/`** | Pickled multilayer NetworkX graphs (`layers.pkl`, `louvain_partitions.pkl`).                                                              |
| **`RMTpaper/`**          | `PCA.py`, `RMT_Models.py` reproduce Fig 2 of the draft paper: empirical spectrum vs Marchenko-Pastur.                                     |
| **`paper/`**             | Notebook-style utilities (`wishart_matrix.py`, `marchenko-pastur.py`) used in LaTeX draft.                                                |
| **`table/`**             | Feature engineering & plotting helpers (see table below).                                                                                 |
| `SP500_25May24*.csv`     | Raw & cleaned price matrix (ticker\_SECTOR columns, 2008-01-02 → 2024-05-24).                                                             |

---

### 0.1 · scripts inside **`distance/`**

| script                      | purpose (one-liner)                                                                                  |      |             |
| --------------------------- | ---------------------------------------------------------------------------------------------------- | ---- | ----------- |
| `distance_matrix.py`        | Rolling Pearson ρ → distance parquet writer (20-day, non-overlap).                                   |      |             |
| `eigenvectorplot.py`        | Scatter of λ<sub>max</sub> eigen-vector (v<sub>i</sub> components) at a chosen date.                 |      |             |
| `ev_20days_plt.py`          | Helper to validate eigen-vectors are unit-norm & finite, then call `plot_eigenvector()`.             |      |             |
| `ev_eva_plot.py`            | Two-panel figure: eigen-value spectrum + λ<sub>max</sub> eigen-vector bar-chart for a single window. |      |             |
| `icon_vector_plot.py`       | Saves PNG of icon-vector direction vs                                                                | v\_i | magnitude.  |
| `icon_vectors_save.py`      | Batch-runner for the above, loops every parquet.                                                     |      |             |
| `identifyflat.py`           | Finds windows where λ<sub>max</sub> eigen-vector is “flat” (                                         | v\_i | ≈ uniform). |
| `lambda_max_plot.py`        | Time-series of λ<sub>max</sub> itself, overlaid on crises.                                           |      |             |
| `lambda_max_vector.py`      | Serialises every λ<sub>max</sub> eigen-vector to CSV for later PCA.                                  |      |             |
| `lambda_max_vector_plot.py` | 3D PCA scatter of all λ<sub>max</sub> vectors (colour = year).                                       |      |             |
| `plot_flat_clusters.py`     | Highlight flat windows on λ<sub>max</sub> time-series.                                               |      |             |
| `svd_eigenvalue_plot.py`    | Sanity: SVD spectrum vs eigen-value spectrum.                                                        |      |             |
| `trend.py`                  | Simple linear trend regression on λ<sub>max</sub>.                                                   |      |             |

---

### 0.2 · scripts inside **`table/`**

| script                                 | focus                                                              |
| -------------------------------------- | ------------------------------------------------------------------ |
| `20_days.py`                           | Generates master CSV: date, mean ρ, ⟨k⟩, entropy, λ<sub>max</sub>. |
| `Table.py`                             | LaTeX tabulator for the same → `table/network_metrics.tex`.        |
| `boxplot.py`                           | Box-plot of ⟨k⟩ across bull vs bear windows.                       |
| `mean_correlation/mean_correlation.py` | Group-by sector, rolling mean-ρ CSV for dashboard.                 |
| `peak_valley_plots/*.py`               | Annotate ΔH peaks & valleys on time-series.                        |
| `selected_points_plot/*.py`            | Mark hand-picked crisis dates.                                     |
| `sample.py`, `test.py`                 | Quick head/tail printing & shape assertions.                       |

---

## 1 · Quick start (pick a script)

```bash
git clone https://github.com/developer-2046/sp500.git
cd sp500
python -m pip install -r requirements.txt     # numpy pandas scipy networkx python-louvain matplotlib pyarrow
python average\ degree.py                    # example: generates average_degree.png
# or
python distance/ev_eva_plot.py --date 2020-03-27
```

---

## 2 · Typical research flow

```mermaid
flowchart LR
  subgraph PREP
    A[CSV prices] -->|log returns| B[distance_matrix.py]
  end
  subgraph ANALYTICS
    B --> C[NeCe.py (Entropy Change & efficiency)]
    B --> D[lambda_max_vector.py]
    D --> E[PCA.py]
    B --> F[average_degree.py]
  end
  subgraph REPORT
    C --> G[table/20_days.py]
    F --> G
    E --> G
    G --> H[paper/main.tex]
  end
```

Feel free to jump in anywhere – every node in that graph is an *independent* script.

---

## 3 · Performance snapshot (N = 505, win = 20 days)

| script                     | total runtime        |
| -------------------------- | -------------------- |
| `distance_matrix.py`       | 1.9 s for one window |
| `NeCe.py` (entire 4 000 d) | \~2 min              |
| `average_degree.py`        | 45 s                 |
| `lambda_max_vector.py`     | 40 s                 |

(M1 MacBook Air, Python 3.11.4)

---

## 4 · Why a quant desk might care

* **Entropy slope (ΔH)** – early-warning for liquidity crunch.
* **λ<sub>max</sub> eigen-vector drift** – crowding into one risk-on mode.
* **Community breaks / Louvain flips** – regime-switch indicator.
* **Global efficiency** – proxy for diversification decay.
  Scripts here export each metric as a CSV so you can plug them straight into a factor model or VaR overlay.

---

## 5 · Citation

```bibtex
@misc{malik2025sp500,
  author = {Yuvraj Malik},
  title  = {S&P 500 Spectral and Network Analytics Toolkit},
  year   = {2025},
  url    = {https://github.com/developer-2046/sp500}
}
```

---

## 6 · Contact

Yuvraj Malik — [yuvrajmalik2046@gmail.com](mailto:yuvrajmalik2046@gmail.com) — [LinkedIn](https://www.linkedin.com/in/yuvraj-malik)

> Clone, pick your favourite script, and let network topology show you tomorrow’s risk.
