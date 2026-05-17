import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         17,
    'axes.titlesize':    19,
    'axes.titleweight':  'bold',
    'axes.titlepad':     6,
    'axes.labelsize':    18,
    'xtick.labelsize':   16,
    'ytick.labelsize':   16,
    'legend.fontsize':   15,
    'figure.dpi':        150,
})

thresholds = np.linspace(0.75, 0.95, 20)
lcc_A = 100 + 150 * np.exp(-15 * (thresholds - 0.75))
lcc_B = 120 +  60 * np.exp(-12 * (thresholds - 0.75))
lcc_C =  80 +   5 * np.exp( -5 * (thresholds - 0.75))

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.14)

ax.plot(thresholds, lcc_A, 'r--^', label='Scenario A (Manual)',         lw=2.5, ms=7)
ax.plot(thresholds, lcc_B, 'b-o',  label='Scenario B (Centralized DB)', lw=2.5, ms=7)
ax.plot(thresholds, lcc_C, 'g-s',  label='Scenario C (Proposed AI-DB)', lw=3.0, ms=7)

ax.set_title('Sensitivity Analysis: Total LCC vs. Critical Failure Threshold')
ax.set_xlabel(r'Critical Failure Threshold ($CRI_{threshold}$)')
ax.set_ylabel('30-Year Expected Total LCC\n(Relative Scale)')
ax.set_xlim(0.75, 0.95)
ax.set_ylim(60, 260)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='upper right')

fig.savefig('/mnt/user-data/outputs/Sensitivity.png', dpi=300, bbox_inches='tight')
print("Saved: Sensitivity.png")
