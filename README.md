# Naval O&M CRI-Based Scenario Analysis — Simulation Code

Replication code for:

> Jung, S. and Lee, M. (2026).
> **"Handover-Induced Risk Peaks in Naval O&M: A CRI-Based Scenario Analysis."**
> *IEEE Access.* Manuscript ID: Access-2026-24786.

---

## Repository Structure

```
├── simulation.py                # Algorithm 1 — Monte Carlo LCC simulation (main)
├── cri_trend_lcc_breakeven.py   # Figure 3 & 4 — CRI sawtooth + LCC break-even
├── distributions.py             # Figure 2 — Stochastic parameter distributions
├── heatmap_sensitivity.py       # Figure 5 — Break-even sensitivity heatmap
├── sensitivity.py               # Figure 6 — LCC vs. CRI threshold sensitivity
├── requirements.txt             # Python dependencies
└── results/                     # Output directory (auto-created)
    └── summary.csv              # Simulation results table
```

All figure-generation scripts directly import `simulation.py` and draw
their data from the Monte Carlo engine; no hand-crafted analytic
curves are used.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full simulation (~2 min on a modern CPU)
python simulation.py

# 3. Quick test run (500 iterations, ~5 sec)
python simulation.py --quick

# 4. Custom iteration count
python simulation.py --iter 1000

# 5. Reproduce all figures
python distributions.py
python cri_trend_lcc_breakeven.py
python heatmap_sensitivity.py            # default 10×10 grid, 200 iter/cell
python heatmap_sensitivity.py --grid 6 --cell-iter 100   # fast variant
python sensitivity.py                    # default 11 tau points, 1000 iter
python sensitivity.py --points 6 --n-iter 300            # fast variant
```

---

## Parameters

All parameters match **Table III** in the paper:

| Symbol | Value | Description |
|--------|-------|-------------|
| N | 30 yr | Simulation horizon |
| I | 10,000 | Monte Carlo iterations |
| λ | 0.15 yr⁻¹ | CVE arrival rate (Poisson); also drives saturating cyber term in Eq. (5) |
| α_HEP, β_HEP | 2, 5 | Beta distribution (HEP) |
| μ_L, σ_L | 4.0, 1.2 | Lognormal (loss severity) |
| τ_th | 0.85 | CRI threshold (MIL-STD-882E) |
| r | 0.03 | Annual discount rate |
| α_CRI | 0.20 | CRI cyber-term weight (Eq. 5) |
| β_CRI | 2.50 | CRI human-term weight (Eq. 5) |
| η_learn | 0.10 /mo | Proficiency recovery rate (≈2.9 mo full recovery from mean HEP) |
| w | 1.0 | Operational criticality weight |
| Seed | 42 | Random seed (fixed) |

---

## Scenarios

| Scenario | δ_AI | CAPEX | p_compromise | Description |
|----------|------|-------|--------------|-------------|
| A | 0.15 | 1.0× | 0.00 | Manual baseline (current practice) |
| B | 0.60 | 1.2× | 0.20 | Centralized DB (no zero-trust) |
| C | 0.95 | 2.0× | 0.00 | Proposed AI-DB (Zero-Trust + LSTM + LLM-RAG) |
| D | 0.80 | 1.7× | 0.10 | AI only (no Zero-Trust) |
| E | 0.30 | 1.5× | 0.00 | Zero-Trust only (no AI) |

The `p_compromise` column captures the SPOF-failure mechanism described
in §VII-C: at each rotation, scenarios without a zero-trust continuity
layer (A, B, D) face a non-zero probability of central-node compromise,
in which case the effective δ_AI for that handover window collapses to 0.
This generates the wider risk-tail variance seen in Scenarios B and D
(see footnote of Table VI).

---

## Expected Output

Running `python simulation.py` with seed 42 (default I=10,000) produces:

```
======================================================================
  Scenario                             Med LCC                 95% CI   CRImax
  --------------------------------------------------------------------
  Scenario A (Manual Baseline)           670.3       [ 367.6, 1454.1]   1.000    (baseline)
  Scenario B (Centralized DB)            327.3       [ 239.6,  754.1]   0.966  (-51.2%)
  Scenario C (Proposed AI-DB)            240.4       [ 240.4,  240.4]   0.264  (-64.1%)
  Scenario D (AI Only, No Zero-Trust)    256.7       [ 240.1,  526.6]   0.832  (-61.7%)
  Scenario E (Zero-Trust Only, No AI)    498.8       [ 286.5, 1128.3]   0.996  (-25.6%)
======================================================================
```

Cost values are reported in a **relative scale** (per-month maintenance
unit = 1.0).  The reported LCC reductions are the headline figures used
in Table IV of the manuscript.

Empirical break-even between Scenario A and Scenario C, computed
directly from the cumulative discounted LCC curves of
`cri_trend_lcc_breakeven.py`, is at approximately **4.5 years**.

---

## Reproducibility

- Random seed is fixed at `SEED = 42` using `numpy.random.default_rng`.
- A single seeded generator drives the entire study so that all scenarios
  share identical stochastic realisations (paired Monte Carlo).
- All parameter values are printed at runtime.
- Results are saved to `results/summary.csv`.

---

## Citation

```bibtex
@article{jung2026handover,
  title   = {Handover-Induced Risk Peaks in Naval {O\&M}:
             A {CRI}-Based Scenario Analysis},
  author  = {Jung, Seunghoon and Lee, Minwoo},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript ID: Access-2026-15865}
}
```

---

## License

This code is provided for academic replication purposes.
© 2026 Seunghoon Jung, Minwoo Lee. All rights reserved.
