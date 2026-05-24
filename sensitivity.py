"""
sensitivity.py  —  Figure 6 (LCC sensitivity to CRI threshold tau).
The simulation is re-run for each threshold value in the sweep using
the existing model in simulation.py; no surrogate analytic curves.

Output:  Sensitivity.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import simulation as sim
from simulation import SCENARIOS, SEED

mpl.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        17,
    'axes.titlesize':   19,
    'axes.titleweight': 'bold',
    'axes.titlepad':    6,
    'axes.labelsize':   18,
    'xtick.labelsize':  16,
    'ytick.labelsize':  16,
    'legend.fontsize':  15,
    'figure.dpi':       150,
})


def median_lcc(delta_ai: float, c_capex: float, p_comp: float,
               n_iter: int, rng: np.random.Generator) -> float:
    costs = np.empty(n_iter)
    for i in range(n_iter):
        c, _ = sim.run_one(delta_ai, c_capex, p_comp, rng)
        costs[i] = c
    return float(np.median(costs))


parser = argparse.ArgumentParser()
parser.add_argument("--points",     type=int, default=11,
                    help="number of threshold points (default 11)")
parser.add_argument("--n-iter",     type=int, default=1000,
                    help="MC iterations per threshold (default 1000)")
args, _ = parser.parse_known_args()

thresholds = np.linspace(0.75, 0.95, args.points)

# Save original TAU_TH so we can restore it
orig_tau = sim.TAU_TH

curves = {"A_Manual": [], "B_DB": [], "C_AI_DB": []}

for tau in thresholds:
    sim.TAU_TH = float(tau)
    for key in curves:
        cfg = SCENARIOS[key]
        rng = np.random.default_rng(SEED)        # paired sampling per tau
        curves[key].append(
            median_lcc(cfg["delta_ai"], cfg["capex"], cfg["p_comp"],
                       args.n_iter, rng)
        )
    print(f"  tau = {tau:.3f}  done")

sim.TAU_TH = orig_tau      # restore default

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.14)

ax.plot(thresholds, curves["A_Manual"], 'r--^',
        label='Scenario A (Manual)',          lw=2.5, ms=7)
ax.plot(thresholds, curves["B_DB"],     'b-o',
        label='Scenario B (Centralized DB)',  lw=2.5, ms=7)
ax.plot(thresholds, curves["C_AI_DB"],  'g-s',
        label='Scenario C (Proposed AI-DB)',  lw=3.0, ms=7)

ax.set_title('Sensitivity: 30-Year LCC vs. Critical Failure Threshold')
ax.set_xlabel(r'Critical Failure Threshold ($CRI_{\mathrm{threshold}}$)')
ax.set_ylabel('30-Year Median LCC (Relative Scale)')
ax.set_xlim(thresholds[0], thresholds[-1])
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='upper right')

fig.savefig('Sensitivity.png', dpi=300, bbox_inches='tight')
print("Saved: Sensitivity.png")
