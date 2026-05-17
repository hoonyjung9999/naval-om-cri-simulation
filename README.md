# Naval O&M CRI-Based Scenario Analysis — Simulation Code

Replication code for:

> Jung, S. and Lee, M. (2026).  
> **"Handover-Induced Risk Peaks in Naval O&M: A CRI-Based Scenario Analysis."**  
> *IEEE Access.* Manuscript ID: Access-2026-15865.

---

## Repository Structure

```
├── simulation.py              # Algorithm 1 — Monte Carlo LCC simulation (main)
├── cri_trend_lcc_breakeven.py # Figure 3 & 4 — CRI sawtooth + LCC break-even
├── distributions.py           # Figure 2 — Stochastic parameter distributions
├── heatmap_sensitivity.py     # Figure 5 — Break-even sensitivity heatmap
├── sensitivity.py             # Figure 6 — LCC vs. CRI threshold sensitivity
├── requirements.txt           # Python dependencies
└── results/                   # Output directory (auto-created)
    └── summary.csv            # Simulation results table
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the simulation (full: ~2 min on a modern CPU)
python simulation.py

# 3. Quick test run (500 iterations, ~5 sec)
python simulation.py --quick

# 4. Custom iteration count
python simulation.py --iter 1000

# 5. Reproduce all figures
python distributions.py
python cri_trend_lcc_breakeven.py
python heatmap_sensitivity.py
python sensitivity.py
```

---

## Parameters

All parameters match **Table II** in the paper:

| Symbol | Value | Description |
|--------|-------|-------------|
| N | 30 yr | Simulation horizon |
| I | 10,000 | Monte Carlo iterations |
| λ | 0.15 yr⁻¹ | CVE arrival rate (Poisson) |
| α_HEP, β_HEP | 2, 5 | Beta distribution (HEP) |
| μ_L, σ_L | 4.0, 1.2 | Lognormal (loss severity) |
| τ_th | 0.85 | CRI threshold (MIL-STD-882E) |
| r | 0.03 | Annual discount rate |
| Seed | 42 | Random seed (fixed) |

---

## Scenarios

| Scenario | δ_AI | CAPEX | Description |
|----------|------|-------|-------------|
| A | 0.15 | 1.0× | Manual baseline (current practice) |
| B | 0.60 | 1.2× | Centralized DB only |
| C | 0.95 | 2.0× | Proposed AI-DB (Zero-Trust + LSTM + LLM-RAG) |
| D | 0.80 | 1.7× | AI only (no Zero-Trust) |
| E | 0.30 | 1.5× | Zero-Trust only (no AI) |

---

## Reproducibility

- Random seed is fixed at `SEED = 42` using `numpy.random.default_rng`.
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
