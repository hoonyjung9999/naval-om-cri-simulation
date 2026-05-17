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
    'axes.labelweight':  'bold',
    'xtick.labelsize':   16,
    'ytick.labelsize':   16,
    'legend.fontsize':   16,
    'figure.dpi':        150,
})

delta_ai = np.linspace(0.70, 0.99, 100)
capex    = np.linspace(1.5,  3.0,  100)
X, Y = np.meshgrid(delta_ai, capex)
Z = np.clip(7.5 * (Y / 2.0) * (0.95 / X)**4, 4, 20)

fig, ax = plt.subplots(figsize=(8, 6))
fig.subplots_adjust(left=0.12, right=0.85, top=0.91, bottom=0.12)

contour = ax.contourf(X, Y, Z, levels=np.arange(4, 21, 1),
                      cmap='RdYlGn_r', extend='max')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Break-Even Time (Years)', fontsize=17, fontweight='bold')
cbar.ax.tick_params(labelsize=15)

ax.plot(0.95, 2.0, 'b*', markersize=18, markeredgecolor='white',
        label='Proposed AI-DB (7.5 Yrs)')

ax.set_title('Multi-Dimensional Sensitivity: Break-Even Time')
ax.set_xlabel(r'AI Defense Coefficient ($\delta_{AI}$)')
ax.set_ylabel('Initial CAPEX Multiplier')
ax.legend(loc='upper right', framealpha=0.9, handlelength=1.0)
ax.grid(color='white', linestyle=':', alpha=0.4)

fig.savefig('/mnt/user-data/outputs/Heatmap_Sensitivity.png', dpi=300, bbox_inches='tight')
print("Saved: Heatmap_Sensitivity.png")
