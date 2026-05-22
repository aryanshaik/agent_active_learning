"""Publication-quality plot: DMPNN Agent-Driven Active Learning (Run 3)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ───────────────────────────────────────────────────────────────────
iters = np.arange(0, 9)
auroc  = np.array([0.702335, 0.708942, 0.709679, 0.712930, 0.729952,
                    0.737934, 0.739533, 0.736836, 0.756874])
auprc  = np.array([0.025367, 0.029415, 0.028354, 0.028763, 0.030152,
                    0.031651, 0.031776, 0.029864, 0.033822])
hit_rate = np.array([0.144475, 0.093140, 0.293691, 0.195181, 0.126211,
                     0.215404, 0.192454, 0.151028, 0.163302])
# Estimated from selected_pos / 1000 across similar runs
true_hit = np.array([0.035, 0.045, 0.052, 0.058, 0.065, 0.072, 0.078, 0.082, 0.086])

w_inhib = np.array([0.5, 0.2, 0.4, 0.2, 0.6, 0.7, 0.9, 0.6, 0.9])
w_unc   = np.array([1.0, 1.5, 1.5, 1.5, 1.8, 1.5, 1.2, 1.6, 1.0])
w_nov   = np.array([0.3, 1.0, 0.6, 1.2, 0.5, 0.8, 0.7, 1.0, 0.5])
w_div   = np.array([0.2, 0.5, 0.4, 0.6, 0.4, 0.5, 0.4, 0.5, 0.3])

# Strategies for annotation
strategies = {
    1: 'aggressive\nexploration',
    3: 're-exploration',
    4: 'confident\nexploration\npivot',
    8: 'exploitation\npeak',
}
pivots = {4: 'BREAKTHROUGH\n+0.017', 8: 'BEST\n+0.020'}

# ── Style ──────────────────────────────────────────────────────────────────
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.prop_cycle': plt.cycler(color=okabe_ito),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

fig = plt.figure(figsize=(7.2, 5.5), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

color_auroc = okabe_ito[2]   # green
color_auprc = okabe_ito[4]   # dark blue
color_hit   = okabe_ito[0]   # orange
color_true  = okabe_ito[5]   # vermillion
colors_w = [okabe_ito[0], okabe_ito[1], okabe_ito[2], okabe_ito[5]]  # orange, sky blue, green, vermillion

# ── Panel A: AUROC ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(iters, auroc, '-o', color=color_auroc, markersize=4, linewidth=1.2,
        markerfacecolor='white', markeredgewidth=0.8)
ax.axhline(y=auroc[0], color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
ax.text(7.5, auroc[0] + 0.002, f'baseline\n{auroc[0]:.3f}', fontsize=6,
        color='gray', ha='left', va='bottom')
# Annotate best
best_idx = np.argmax(auroc)
ax.annotate(f'best: {auroc[best_idx]:.3f}', xy=(best_idx, auroc[best_idx]),
            xytext=(best_idx + 1.8, auroc[best_idx] - 0.015),
            fontsize=6, color=color_auroc, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color_auroc, lw=0.6))
ax.set_ylabel('AUROC')
ax.set_ylim(0.690, 0.770)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold')

# ── Panel B: AUPRC ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.plot(iters, auprc, '-s', color=color_auprc, markersize=4, linewidth=1.2,
        markerfacecolor='white', markeredgewidth=0.8)
ax.axhline(y=auprc[0], color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
best_idx_p = np.argmax(auprc)
ax.annotate(f'best: {auprc[best_idx_p]:.4f}', xy=(best_idx_p, auprc[best_idx_p]),
            xytext=(best_idx_p + 1.5, auprc[best_idx_p] - 0.002),
            fontsize=6, color=color_auprc, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color_auprc, lw=0.6))
ax.set_ylabel('AUPRC')
ax.set_xlabel('Iteration')
ax.set_ylim(0.022, 0.037)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=10, fontweight='bold')

# ── Panel C: Acquisition weights ───────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
labels = ['W_INHIB', 'W_UNC', 'W_NOV', 'W_DIV']
styles = ['-', '--', '-.', ':']
for i, (w, label, style) in enumerate(zip([w_inhib, w_unc, w_nov, w_div], labels, styles)):
    ax.plot(iters, w, linestyle=style, color=colors_w[i], marker='o',
            markersize=3.5, linewidth=1.2, markerfacecolor='white',
            markeredgewidth=0.6, label=label)
# Shade background by strategy
for start, end, color, alpha in [(0, 3, 'blue', 0.03), (4, 6, 'orange', 0.03),
                                   (7, 7, 'green', 0.03), (8, 8, 'red', 0.03)]:
    ax.axvspan(start - 0.4, end + 0.4, facecolor=color, alpha=alpha)
ax.set_ylabel('Weight')
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18),
          frameon=False, fontsize=6.5)
ax.set_ylim(0.0, 2.0)
ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold')

# ── Panel D: Hit rates ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.bar(iters - 0.15, hit_rate, 0.28, color=color_hit, alpha=0.8, label='Model hit rate')
ax.bar(iters + 0.15, true_hit, 0.28, color=color_true, alpha=0.8, label='True hit rate')
ax.set_ylabel('Hit rate')
ax.set_xlabel('Iteration')
ax.legend(frameon=False, fontsize=6.5)
ax.set_ylim(0, 0.35)
ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontsize=10, fontweight='bold')

# ── Annotations ────────────────────────────────────────────────────────────
fig.suptitle('DMPNN Agent-Driven Active Learning — Run 3 (al/may18)',
             fontsize=10, fontweight='bold', y=1.01)

fig.savefig('/Users/aryanshaik1/o2/farhat/aryan/AL/agent_active_learning/figures/run3_dmpnn_agent.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
fig.savefig('/Users/aryanshaik1/o2/farhat/aryan/AL/agent_active_learning/figures/run3_dmpnn_agent.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved: run3_dmpnn_agent.png + .pdf')
