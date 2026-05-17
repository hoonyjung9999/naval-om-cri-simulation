"""
cri_trend_lcc_breakeven.py  (Final revision for IEEE Access 1-column)

Target printed font sizes (scale = 3.5/9 ≈ 0.389 for 9-inch wide fig):
  title:       22 pt → 8.6 pt  (≈caption)
  axis label:  19 pt → 7.4 pt  (above 6 pt minimum)
  tick label:  17 pt → 6.6 pt
  legend:      15 pt → 5.8 pt  (compact but legible)

Key fix: use subplots_adjust or pad in savefig to prevent ylabel clipping.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         17,
    'axes.titlesize':    20,
    'axes.titleweight':  'bold',
    'axes.titlepad':     6,
    'axes.labelsize':    18,
    'xtick.labelsize':   16,
    'ytick.labelsize':   16,
    'legend.fontsize':   14,
    'lines.linewidth':   2.5,
    'figure.dpi':        150,
})

# =============================================================
# 1. CRI Sawtooth Graph
# =============================================================
months = np.arange(0, 121)
rotation_cycle = 18

cri_A = np.zeros(121)
cri_B = np.zeros(121)
cri_C = np.zeros(121)

for t in months:
    tsr = t % rotation_cycle
    cri_A[t] = 0.9 * np.exp(-0.10 * tsr) + 0.05 + t * 0.001
    cri_B[t] = 0.6 * np.exp(-0.15 * tsr) + 0.05 + t * 0.0005
    cri_C[t] = 0.2 * np.exp(-0.20 * tsr) + 0.05 + t * 0.0002

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.13)

ax.plot(months, cri_A, 'r-',  label=r'Scenario A (Manual, $\delta_{AI}=0.15$)',          lw=2.5)
ax.plot(months, cri_B, 'b--', label=r'Scenario B (Centralized DB, $\delta_{AI}=0.60$)', lw=2.5)
ax.plot(months, cri_C, 'g-',  label=r'Scenario C (Proposed AI-DB, $\delta_{AI}=0.95$)', lw=3.0)
ax.axhline(y=0.85, color='black', linestyle=':', lw=2.5, label='Critical Threshold (0.85)')
ax.fill_between(months, 0.85, 1.0, color='red', alpha=0.10)

ax.set_title('10-Year Composite Risk Index (CRI) Fluctuation')
ax.set_xlabel('Operational Time (Months)')
ax.set_ylabel('Composite Risk Index (CRI)')
ax.set_xticks(np.arange(0, 121, 18))
ax.set_xlim(0, 120)
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', framealpha=0.9,
          handlelength=1.8, labelspacing=0.25, borderpad=0.5)

fig.savefig('/mnt/user-data/outputs/CRI_Trend.png', dpi=300, bbox_inches='tight')
print("Saved: CRI_Trend.png")
plt.close()

# =============================================================
# 2. LCC Break-even Chart
# =============================================================
years = np.arange(0, 31)
lcc_A = 100 + 15 * years + 0.5  * years**2
lcc_C = 200 +  5 * years + 0.05 * years**2

fig2, ax2 = plt.subplots(figsize=(9, 5.5))
fig2.subplots_adjust(left=0.15, right=0.97, top=0.91, bottom=0.13)

ax2.plot(years, lcc_A, 'r-o', label='Scenario A (Traditional Baseline)', markersize=5)
ax2.plot(years, lcc_C, 'g-s', label='Scenario C (Proposed AI-DB)',       markersize=5, lw=3.0)

ax2.axvline(x=7.5, color='gray', linestyle='--', lw=2.0)
ax2.plot(7.5, lcc_C[7] + (lcc_C[8] - lcc_C[7]) * 0.5,
         'kx', markersize=13, markeredgewidth=2.5)
ax2.annotate(
    'Break-Even Point\n(Approx. 7.5 Years)',
    xy=(7.7, 230), xytext=(12, 155),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    fontsize=15, fontweight='bold'
)

ax2.set_title('30-Year Cumulative Life Cycle Cost (LCC) Analysis')
ax2.set_xlabel('Operational Time (Years)')
ax2.set_ylabel('Cumulative Total LCC\n(Relative Scale)')   # 줄 바꿈으로 잘림 방지
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 800)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left')

fig2.savefig('/mnt/user-data/outputs/LCC_Breakeven.png', dpi=300, bbox_inches='tight')
print("Saved: LCC_Breakeven.png")
plt.close()
