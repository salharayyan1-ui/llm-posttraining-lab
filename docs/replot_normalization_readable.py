"""Regenerate one plot from unchanged saved results; preserve experiment source."""
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--results', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
sys.path.insert(0, args.repo)
from scripts import plot_results as plots

results = Path(args.results)
out = Path(args.out)
out.mkdir(exist_ok=True)
plots.style()
fig, axes = plots.plt.subplots(1, 2, figsize=(5.8, 2.5))
for i, (label, name) in enumerate(zip(['GRPO (1/T)', 'Dr. GRPO (1/L_max)'], ['norm_grpo', 'norm_dr_grpo'])):
    data = plots.load_jsonl(results / f'task3_grpo/train/{name}/completions.jsonl')
    data = data[data['in_loss'] & (data['advantage'].abs() > 1e-8)]
    for ax, metric in zip(axes, ['grad_per_token', 'grad_mass']):
        ax.scatter(data['response_tokens'], data[metric], s=10, color=plots.SERIES[i],
                   edgecolor=plots.SURFACE, linewidth=0.5, label=label)
for ax, title in zip(axes, ['gradient weight\nper token', 'gradient mass\nper completion']):
    ax.set_title(title)
    ax.set_xlabel('completion length (tokens)')
axes[0].legend()
fig.tight_layout(w_pad=2.5)
plots.save(fig, out, 'task3_normalization_gradient_allocation')
