# S&P 500 Spectral- & Network-Analytics Toolkit  
**Systemic-Risk & Regime-Shift Detection via Random-Matrix Theory, Entropy and Multiplex Graphs**  

<p align="center">
  <img src="plots/teaser_network.png" width="640" alt="S&P500 Network snapshot">
</p>

> **Why this repo matters to a quant desk**  
> • Converts raw price series into high-fidelity correlation networks  
> • Blends Random-Matrix Theory (RMT), information entropy & multiplex graph statistics  
> • Produces trade-ready regime signals and feature tables you can feed straight into risk or alpha models  

---

## 1 . Motivation  

*Modern equity markets are not iid time-series; they are adaptive, high-dimensional networks whose structure warps during stress.*  
Traditional factors (β, size, value …) fail to capture this topology.  
Here we:  

1. **Denoise** rolling correlation matrices with RMT eigenvalue clipping.  
2. **Embed** the cleaned matrix into a **distance network** \( _d_{ij}=√{2(1−ρ_{ij})}_ \).  
3. **Characterise** each 20-day snapshot with:
   * Shannon **eigen-entropy** (spectral dispersion)  
   * **Average degree** & high-order centralities on the Minimum-Spanning-Tree  
   * **Similarity matrices** & *Δ-angles* between successive eigenvectors  
4. **Stack** single-layer graphs into a **multiplex network** to capture cross-sector spill-overs.  
5. **Generate signals** – entropy spikes, community-breaks, MST edge-breaches – that historically precede major sell-offs (2008-10-16, 2010-05-20, 2020-03-27 …).

---

## 2 . Repository Layout  

```text
.
├── NeCe.py                     # One-click orchestrator – executes full pipeline end-to-end
├── SP500_25May24.csv           # Raw OHLCV (Adj-Close) dataset
├── SP500_25May24_cleaned_filled.csv
│
├── adjacency matrix.py         # ρ → A(ρ>τ) binary adjacency
├── average degree.py           # <k> and other centralities per window
├── minus_one_to_one.py         # Maps raw correlations from [−1,1] → distance space
├── similarity_matrix.py        # Cosine-similarity between successive eigen-vectors
├── matrix.py                   # Helper: eigen-decomposition utilities
│
├── distance/                   # Rolling-window d_{ij} matrices (parquet)
├── shannonentropy/             # Per-window entropy & CSV summaries
├── multiplex network/          # Layered NetworkX objects & Louvain partitions
├── plots/                      # Auto-generated .png figures (MSTs, entropy time-series …)
├── table/                      # Publication-ready LaTeX tables of metrics
└── RMTpaper/ & paper/          # Draft write-up + BibTeX refs (aimed at a Q1 journal)
