"""
cri_trend_lcc_breakeven.py  —  Figure 3 (CRI trajectory) and Figure 4
(LCC break-even) generated directly from the Monte Carlo simulation
defined in simulation.py.  No hard-coded analytic curves.

Outputs:
    CRI_Trend.png       — single-trajectory CRI(t) for A, B, C over 120 mo
    LCC_Breakeven.png   — cumulative discounted LCC for A vs C with the
                          empirical break-even year marked
"""

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
    'axes.labelsize':   17,
    'xtick.labelsize':  16,
    'ytick.labelsize':  16,
    'legend.fontsize':  15,
    'lines.linewidth':  2.5,
    'figure.dpi':       150,
})

# =============================================================
# 1. CRI sawtooth — one paired trajectory per scenario
# =============================================================
def cri_trajectory(delta_ai: float, p_comp: float,
                   rng: np.random.Generator,
                   horizon_months: int = 120) -> np.ndarray:
    """Single-iteration CRI trajectory using the same model as simulation.py."""
    s_t       = 1.0
    delta_eff = delta_ai
    out = np.zeros(horizon_months + 1)
    out[0] = compute_cri(0, 1.0, delta_eff)
    for t in range(1, horizon_months + 1):
        if t % 18 == 0:
            s_t = 1.0 - rng.beta(ALPHA_HEP, BETA_HEP)
            delta_eff = 0.0 if (p_comp > 0 and rng.random() < p_comp) else delta_ai
        else:
            s_t = min(s_t + ETA_LEARN, 1.0)
        out[t] = compute_cri(t, s_t, delta_eff)
    return out

# Paired-RNG so the three scenarios see the same HEP draws → directly
# attributable differences come from delta_AI and ZT, not noise.
months = np.arange(0, 121)

cris = {}
for key in ("A_Manual", "B_DB", "C_AI_DB"):
    rng = np.random.default_rng(SEED)         # reset for paired sampling
    cris[key] = cri_trajectory(
        SCENARIOS[key]["delta_ai"],
        SCENARIOS[key]["p_comp"],
        rng,
        horizon_months=120,
    )

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.13)

ax.plot(months, cris["A_Manual"], 'r-',
        label=r'Scenario A (Manual, $\delta_{AI}=0.15$)',           lw=2.5)
ax.plot(months, cris["B_DB"],     'b--',
        label=r'Scenario B (Centralized DB, $\delta_{AI}=0.60$)',   lw=2.5)
ax.plot(months, cris["C_AI_DB"],  'g-',
        label=r'Scenario C (Proposed AI-DB, $\delta_{AI}=0.95$)',   lw=3.0)
ax.axhline(y=TAU_TH, color='black', linestyle=':', lw=2.5,
           label=f'Critical Threshold ({TAU_TH})')
ax.fill_between(months, TAU_TH, 1.0, color='red', alpha=0.10)

ax.set_title('10-Year Composite Risk Index (CRI) Trajectory')
ax.set_xlabel('Operational Time (Months)')
ax.set_ylabel('Composite Risk Index (CRI)')
ax.set_xticks(np.arange(0, 121, 18))
ax.set_xlim(0, 120)
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', framealpha=0.9,
          handlelength=1.8, labelspacing=0.25, borderpad=0.5)

fig.savefig('CRI_Trend.png', dpi=300, bbox_inches='tight')
print("Saved: CRI_Trend.png")
plt.close()

# =============================================================
# 2. LCC break-even — cumulative discounted cost from sim
# =============================================================
def cumulative_lcc(delta_ai: float, c_capex: float, p_comp: float,
                   n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Median cumulative discounted LCC trajectory across n_iter runs."""
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

# Use fewer iterations here for runtime; median is stable around 2k samples
N_TRAJ = 2000

rng = np.random.default_rng(SEED)
lcc_A = cumulative_lcc(SCENARIOS["A_Manual"]["delta_ai"],
                       SCENARIOS["A_Manual"]["capex"],
                       SCENARIOS["A_Manual"]["p_comp"], N_TRAJ, rng)
rng = np.random.default_rng(SEED)
lcc_C = cumulative_lcc(SCENARIOS["C_AI_DB"]["delta_ai"],
                       SCENARIOS["C_AI_DB"]["capex"],
                       SCENARIOS["C_AI_DB"]["p_comp"], N_TRAJ, rng)

# Empirical break-even: first month where median A >= median C
diff = lcc_A - lcc_C
crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
if crossings.size > 0:
    t_break_mo = crossings[0] + 1
    t_break_yr = t_break_mo / 12.0
else:
    t_break_yr = None

years_axis = np.arange(0, T_MAX + 1) / 12.0

fig2, ax2 = plt.subplots(figsize=(9, 5.5))
fig2.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.13)

ax2.plot(years_axis, lcc_A, 'r-',  label='Scenario A (Manual Baseline)',  lw=2.5)
ax2.plot(years_axis, lcc_C, 'g-',  label='Scenario C (Proposed AI-DB)',   lw=3.0)

if t_break_yr is not None:
    ax2.axvline(x=t_break_yr, color='gray', linestyle='--', lw=2.0)
    ax2.plot(t_break_yr, lcc_C[crossings[0] + 1],
             'kx', markersize=13, markeredgewidth=2.5)
    ax2.annotate(
        f'Break-Even Point\n({t_break_yr:.1f} Years)',
        xy=(t_break_yr + 0.3, lcc_C[crossings[0] + 1]),
        xytext=(t_break_yr + 6, lcc_C[crossings[0] + 1] * 0.55),
        arrowprops=dict(facecolor='black', shrink=0.05,
                        width=1.5, headwidth=8),
        fontsize=15, fontweight='bold',
    )

ax2.set_title('30-Year Cumulative Discounted LCC')
ax2.set_xlabel('Operational Time (Years)')
ax2.set_ylabel('Cumulative LCC (Relative Scale)')
ax2.set_xlim(0, 30)
ax2.set_ylim(0, max(lcc_A.max(), lcc_C.max()) * 1.08)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left')

fig2.savefig('LCC_Breakeven.png', dpi=300, bbox_inches='tight')
print("Saved: LCC_Breakeven.png")
if t_break_yr is not None:
    print(f"  Empirical break-even year: {t_break_yr:.2f}")
else:
    print("  No break-even within 30-year horizon")
plt.close()
