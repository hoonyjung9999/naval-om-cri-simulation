"""
heatmap_sensitivity.py  —  Figure 5 (break-even sensitivity surface).
The surface is built by re-running the Monte Carlo simulation over a
grid of (delta_AI, CAPEX) settings rather than evaluating a hand-fit
analytic surrogate.  Per-cell iteration count (N_CELL) is reduced for
runtime; raising it yields a smoother surface at higher cost.

Output:  Heatmap_Sensitivity.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from simulation import (
    SCENARIOS, compute_cri, T_MAX, R, C_MAINT,
    ALPHA_HEP, BETA_HEP, ETA_LEARN, MU_L, SIGMA_L, TAU_TH, SEED,
)

mpl.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        17,
    'axes.titlesize':   20,
    'axes.titleweight': 'bold',
    'axes.titlepad':    6,
    'axes.labelsize':   18,
    'axes.labelweight': 'bold',
    'xtick.labelsize':  16,
    'ytick.labelsize':  16,
    'legend.fontsize':  16,
    'figure.dpi':       150,
})


def median_lcc_trajectory(delta_ai: float, c_capex: float,
                          p_comp: float, n_iter: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Median cumulative discounted LCC across n_iter independent runs."""
    traj = np.zeros((n_iter, T_MAX + 1))
    for i in range(n_iter):
        s_t       = 1.0
        delta_eff = delta_ai
        in_breach = False
        cost      = c_capex
        traj[i, 0] = cost
        for t in range(1, T_MAX + 1):
            if t % 18 == 0:
                s_t = 1.0 - rng.beta(ALPHA_HEP, BETA_HEP)
                delta_eff = 0.0 if (p_comp > 0 and rng.random() < p_comp) else delta_ai
            else:
                s_t = min(s_t + ETA_LEARN, 1.0)
            cri = compute_cri(t, s_t, delta_eff)
            disc = (1.0 + R) ** (-t / 12.0)
            if cri > TAU_TH:
                if not in_breach:
                    cost += rng.lognormal(mean=MU_L, sigma=SIGMA_L) * disc
                    in_breach = True
            else:
                in_breach = False
            cost += C_MAINT * disc
            traj[i, t] = cost
    return np.median(traj, axis=0)


def breakeven_year(delta_ai_c: float, capex_c: float,
                   n_iter: int, rng_seed_a: int,
                   rng_seed_c: int) -> float:
    """Return year of first up-crossing of the median A−C curve, or 30+
       if no break-even occurs within the simulation horizon."""
    rng_a = np.random.default_rng(rng_seed_a)
    lcc_a = median_lcc_trajectory(
        SCENARIOS["A_Manual"]["delta_ai"],
        SCENARIOS["A_Manual"]["capex"],
        SCENARIOS["A_Manual"]["p_comp"],
        n_iter, rng_a,
    )
    rng_c = np.random.default_rng(rng_seed_c)
    lcc_c = median_lcc_trajectory(
        delta_ai_c, capex_c,
        SCENARIOS["C_AI_DB"]["p_comp"],
        n_iter, rng_c,
    )
    diff = lcc_a - lcc_c
    crossings = np.where(np.diff(np.sign(diff)) > 0)[0]
    if crossings.size > 0:
        return (crossings[0] + 1) / 12.0
    # If no crossing, return horizon + small margin so colorbar saturates
    return 30.5


# --- Sweep grid -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--grid", type=int, default=10,
                    help="grid size per axis (default 10 → 100 cells)")
parser.add_argument("--cell-iter", type=int, default=200,
                    help="MC iterations per grid cell (default 200)")
args, _ = parser.parse_known_args()

delta_ai_axis = np.linspace(0.50, 0.99, args.grid)
capex_axis    = np.linspace(1.00, 4.00, args.grid)
X, Y = np.meshgrid(delta_ai_axis, capex_axis)
Z = np.zeros_like(X)

# Cache median Scenario A LCC trajectory once (it does not vary across cells)
print(f"Sweeping {args.grid}×{args.grid} grid with {args.cell_iter} iter/cell ...")
for i, capex in enumerate(capex_axis):
    for j, dai in enumerate(delta_ai_axis):
        Z[i, j] = breakeven_year(dai, capex, args.cell_iter,
                                 rng_seed_a=SEED, rng_seed_c=SEED + 1)
    print(f"  row {i+1}/{args.grid} done")

# Clip for visualisation
Z_disp = np.clip(Z, 2, 20)

fig, ax = plt.subplots(figsize=(8, 6))
fig.subplots_adjust(left=0.12, right=0.85, top=0.91, bottom=0.12)

contour = ax.contourf(X, Y, Z_disp,
                      levels=np.arange(2, 21, 1),
                      cmap='RdYlGn_r', extend='max')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Break-Even Time (Years)', fontsize=17, fontweight='bold')
cbar.ax.tick_params(labelsize=15)

# Mark the baseline AI-DB configuration
ax.plot(SCENARIOS["C_AI_DB"]["delta_ai"], SCENARIOS["C_AI_DB"]["capex"],
        'b*', markersize=18, markeredgecolor='white',
        label=f'Proposed AI-DB (δ={SCENARIOS["C_AI_DB"]["delta_ai"]}, '
              f'{SCENARIOS["C_AI_DB"]["capex"]}×)')

ax.set_title('Break-Even Sensitivity: (δ_AI, CAPEX)')
ax.set_xlabel(r'AI Defense Coefficient ($\delta_{AI}$)')
ax.set_ylabel('Initial CAPEX Multiplier')
ax.legend(loc='upper right', framealpha=0.9, handlelength=1.0)
ax.grid(color='white', linestyle=':', alpha=0.4)

fig.savefig('Heatmap_Sensitivity.png', dpi=300, bbox_inches='tight')
print("Saved: Heatmap_Sensitivity.png")
