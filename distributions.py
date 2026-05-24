"""
distributions.py  (Final revision for IEEE Access 2-column spanning)
figsize = 12 x 4.5,  scale = 7.0/12 ≈ 0.583
  title:   16 pt → 9.3 pt
  label:   15 pt → 8.7 pt
  tick:    13 pt → 7.6 pt
  annot:   13 pt → 7.6 pt
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats

mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         13,
    'axes.titlesize':    16,
    'axes.titleweight':  'bold',
    'axes.titlepad':     5,
    'axes.labelsize':    15,
    'xtick.labelsize':   13,
    'ytick.labelsize':   13,
    'legend.fontsize':   13,
    'figure.dpi':        150,
})

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18, wspace=0.35)

# --- (a) Beta ---
x = np.linspace(0, 1, 200)
y = stats.beta.pdf(x, a=2, b=5)
axes[0].plot(x, y, 'b-', lw=2.5)
axes[0].fill_between(x, y, alpha=0.2, color='blue')
axes[0].set_title('(a) Beta Distribution (HEP)')
axes[0].set_xlabel(r'Error Probability ($0 \sim 1$)')
axes[0].set_ylabel('Density')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].text(0.38, 1.6, r'$\alpha=2,\ \beta=5$', fontsize=13,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

# --- (b) Lognormal ---
x2 = np.linspace(0, 300, 300)
y2 = stats.lognorm.pdf(x2, s=1.2, scale=np.exp(4.0))
axes[1].plot(x2, y2, 'r-', lw=2.5)
axes[1].fill_between(x2, y2, alpha=0.2, color='red')
axes[1].set_title(r'(b) Lognormal Dist. ($C_{risk}$)')
axes[1].set_xlabel('Recovery Cost (Relative Scale)')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].text(95, 0.0057, r'$\mu=4.0,\ \sigma=1.2$', fontsize=13,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

# --- (c) Poisson ---
x3 = np.arange(0, 5)
y3 = stats.poisson.pmf(x3, mu=0.15)
axes[2].vlines(x3, 0, y3, colors='g', lw=5)
axes[2].plot(x3, y3, 'go', ms=9)
axes[2].set_title(r'(c) Poisson Dist. ($\lambda$)')
axes[2].set_xlabel('Threat Occurrences per Year')
axes[2].set_xticks(x3)
axes[2].grid(True, linestyle=':', alpha=0.6)
axes[2].text(1.4, 0.62, r'$\lambda=0.15$', fontsize=13,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

fig.savefig('Distributions.png', dpi=300, bbox_inches='tight')
print("Saved: Distributions.png")
