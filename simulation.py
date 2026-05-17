"""
simulation.py  —  Monte Carlo Life-Cycle Risk and Cost Simulation
Algorithm 1 implementation from:

  Jung, S. and Lee, M. (2026).
  "Handover-Induced Risk Peaks in Naval O&M: A CRI-Based Scenario Analysis"
  IEEE Access. Manuscript ID: Access-2026-15865.

Usage
-----
    python simulation.py              # full study (10,000 iterations)
    python simulation.py --quick      # fast test (500 iterations)
    python simulation.py --iter 1000  # custom iteration count

Results saved to results/summary.csv.  Seed fixed for reproducibility.

Requirements: Python >= 3.10, numpy >= 1.24, scipy >= 1.10
"""

import argparse, csv, os, time
import numpy as np

# ── Reproducibility ────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)
os.makedirs("results", exist_ok=True)

# ==============================================================
# Parameters  (Table II, Section IV)
# ==============================================================
N_YEARS   = 30           # simulation horizon (yr)
T_MAX     = N_YEARS * 12 # monthly steps = 360
I_DEFAULT = 10_000       # default iterations

LAMBDA    = 0.15         # CVE annual rate  [Poisson]
ALPHA_HEP = 2            # Beta α (HEP)
BETA_HEP  = 5            # Beta β
MU_L      = 4.0          # Lognormal μ (loss severity)
SIGMA_L   = 1.2          # Lognormal σ

TAU_TH    = 0.85         # CRI threshold (MIL-STD-882E)
R         = 0.03         # annual discount rate
ETA_LEARN = 0.05         # monthly proficiency recovery
W         = 1.0          # criticality weight
C_MAINT   = 1.0          # routine maintenance cost / month (relative)

# CRI weight factors (Eq. 10)
ALPHA_CRI = 0.20   # cyber-vulnerability growth weight
BETA_CRI  = 2.50   # human-factor sawtooth weight

# ==============================================================
# Scenario definitions  (Section IV-B)
# ==============================================================
SCENARIOS = {
    "A_Manual":  {"delta_ai": 0.15, "capex": 1.0,
                  "label": "Scenario A (Manual Baseline)"},
    "B_DB":      {"delta_ai": 0.60, "capex": 1.2,
                  "label": "Scenario B (Centralized DB)"},
    "C_AI_DB":   {"delta_ai": 0.95, "capex": 2.0,
                  "label": "Scenario C (Proposed AI-DB)"},
    "D_AI_only": {"delta_ai": 0.80, "capex": 1.7,
                  "label": "Scenario D (AI Only, No Zero-Trust)"},
    "E_ZT_only": {"delta_ai": 0.30, "capex": 1.5,
                  "label": "Scenario E (Zero-Trust Only, No AI)"},
}

# ==============================================================
# CRI  (Eq. 10)
# ==============================================================
def compute_cri(t_month: int, s_t: float, delta_ai: float) -> float:
    """
    CRI(t) = w * [alpha*exp(lambda*t_yr) + beta*(1-S(t))*(1-delta_AI)]
    clipped to [0, 1].
    """
    t_yr = t_month / 12.0
    cri  = W * (ALPHA_CRI * np.exp(LAMBDA * t_yr)
                + BETA_CRI * (1.0 - s_t) * (1.0 - delta_ai))
    return float(np.clip(cri, 0.0, 1.0))

# ==============================================================
# One iteration  (Algorithm 1 inner loop)
# ==============================================================
def run_one(delta_ai: float, c_capex: float,
            rng: np.random.Generator) -> tuple[float, float]:
    cost    = c_capex
    s_t     = 1.0
    cri_max = 0.0

    for t in range(1, T_MAX + 1):
        if t % 18 == 0:                              # rotation event
            hep = rng.beta(ALPHA_HEP, BETA_HEP)
            s_t = 1.0 - hep
        else:
            s_t = min(s_t + ETA_LEARN, 1.0)         # recovery

        cri_t   = compute_cri(t, s_t, delta_ai)
        cri_max = max(cri_max, cri_t)

        if cri_t > TAU_TH:                           # critical failure
            c_risk = rng.lognormal(mean=MU_L, sigma=SIGMA_L)
            cost  += c_risk * (1.0 + R) ** (-t / 12.0)

        cost += C_MAINT * (1.0 + R) ** (-t / 12.0)  # routine maintenance

    return cost, cri_max

# ==============================================================
# Scenario run
# ==============================================================
def run_scenario(name: str, delta_ai: float,
                 c_capex: float, n_iter: int) -> dict:
    costs   = np.empty(n_iter)
    cri_max = np.empty(n_iter)
    for i in range(n_iter):
        costs[i], cri_max[i] = run_one(delta_ai, c_capex, rng)
    return {
        "name":       name,
        "delta_ai":   delta_ai,
        "capex":      c_capex,
        "lcc_median": float(np.median(costs)),
        "lcc_mean":   float(np.mean(costs)),
        "lcc_p5":     float(np.percentile(costs,  5)),
        "lcc_p95":    float(np.percentile(costs, 95)),
        "cri_mean":   float(np.mean(cri_max)),
        "cri_p5":     float(np.percentile(cri_max,  5)),
        "cri_p95":    float(np.percentile(cri_max, 95)),
        "raw_costs":  costs,
        "raw_cri":    cri_max,
    }

# ==============================================================
# Main
# ==============================================================
def main(n_iter: int = I_DEFAULT) -> dict:
    print("=" * 68)
    print("  Naval O&M Monte Carlo LCC Simulation  (Algorithm 1)")
    print(f"  Iterations: {n_iter:,}  |  Horizon: {N_YEARS} yr  |  Seed: {SEED}")
    print(f"  ALPHA_CRI={ALPHA_CRI}, BETA_CRI={BETA_CRI}, "
          f"lambda={LAMBDA}/yr, tau={TAU_TH}")
    print("=" * 68)

    results = {}
    t0 = time.time()

    for key, cfg in SCENARIOS.items():
        print(f"\n  Running {cfg['label']}"
              f"  [delta_AI={cfg['delta_ai']}, CAPEX={cfg['capex']}x]",
              end="  ...", flush=True)
        t1 = time.time()
        results[key] = run_scenario(key, cfg["delta_ai"],
                                    cfg["capex"], n_iter)
        print(f"  {time.time()-t1:.1f} s")

    print(f"\n  Total: {time.time()-t0:.1f} s")

    # ── Summary table ──────────────────────────────────────────
    base = results["A_Manual"]["lcc_median"]
    print("\n" + "=" * 68)
    print(f"  {'Scenario':<34} {'Median LCC':>10}  {'95% CI':>21}  {'CRImax':>7}")
    print("  " + "-" * 66)
    for key, r in results.items():
        ci  = f"[{r['lcc_p5']:6.1f},{r['lcc_p95']:7.1f}]"
        red = (base - r["lcc_median"]) / base * 100
        tag = f"({red:+.1f}%)" if key != "A_Manual" else "  (baseline)"
        print(f"  {SCENARIOS[key]['label']:<34} {r['lcc_median']:>10.1f}"
              f"  {ci:>21}  {r['cri_mean']:>6.3f}  {tag}")
    print("=" * 68)

    # ── Convergence check (Section V-C) ───────────────────────
    ref     = results["A_Manual"]["raw_costs"]
    running = np.cumsum(ref) / np.arange(1, n_iter + 1)
    final   = running[-1]
    conv    = next((i+1 for i, v in enumerate(running)
                    if abs(v - final) / final < 0.005), n_iter)
    print(f"\n  Convergence (Scen A, ±0.5%): iteration {conv:,} / {n_iter:,}")

    # ── Save CSV ───────────────────────────────────────────────
    path = "results/summary.csv"
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Scenario", "delta_AI", "CAPEX",
                     "LCC_median", "LCC_mean", "LCC_p5", "LCC_p95",
                     "CRImax_mean", "CRImax_p5", "CRImax_p95",
                     "LCC_reduction_%_vs_A"])
        for key, r in results.items():
            red = (base - r["lcc_median"]) / base * 100
            wr.writerow([SCENARIOS[key]["label"],
                         r["delta_ai"], r["capex"],
                         f"{r['lcc_median']:.2f}", f"{r['lcc_mean']:.2f}",
                         f"{r['lcc_p5']:.2f}",    f"{r['lcc_p95']:.2f}",
                         f"{r['cri_mean']:.4f}",
                         f"{r['cri_p5']:.4f}",    f"{r['cri_p95']:.4f}",
                         f"{red:.1f}"])
    print(f"  Results → {path}\n")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="500 iterations (fast test)")
    ap.add_argument("--iter", type=int, default=None,
                    help="custom iteration count")
    args = ap.parse_args()
    n = args.iter or (500 if args.quick else I_DEFAULT)
    main(n_iter=n)
