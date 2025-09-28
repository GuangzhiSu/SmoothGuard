import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _set_publication_style():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.linewidth': 1.0,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _read_reviews(path: str) -> List[Dict]:
    return [json.loads(line) for line in open(path)]


def _compute_category_means(reviews: List[Dict]) -> Dict[str, Tuple[float, float]]:
    bucket: Dict[str, List[Tuple[float, float]]] = {}
    for r in reviews:
        if 'tuple' not in r:
            continue
        cat = r.get('category', 'all')
        bucket.setdefault(cat, []).append(tuple(r['tuple']))
        bucket.setdefault('all', []).append(tuple(r['tuple']))

    means: Dict[str, Tuple[float, float]] = {}
    for cat, tuples in bucket.items():
        arr = np.asarray(tuples, dtype=float)
        m = arr.mean(axis=0).tolist()
        means[cat] = (float(m[0]), float(m[1]))
    return means


def _to_metrics(means: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cat, (a, b) in means.items():
        ratio = (b / a * 100.0) if a != 0 else 0.0
        out[cat] = {
            'ratio_percent': ratio,
            'score1_x10': a * 10.0,
            'score2_x10': b * 10.0,
        }
    return out


def _order_categories(categories: List[str]) -> List[str]:
    preferred = ['all', 'llava_bench_complex', 'llava_bench_conv', 'llava_bench_detail']
    ordered = [c for c in preferred if c in categories]
    for c in sorted(categories):
        if c not in ordered:
            ordered.append(c)
    return ordered


def _pretty_label(cat: str) -> str:
    mapping = {
        'all': 'All',
        'llava_bench_complex': 'Complex',
        'llava_bench_conv': 'Conversational',
        'llava_bench_detail': 'Detail',
    }
    return mapping.get(cat, cat)


def _compute_ylim(values_a: List[float], values_b: List[float], clamp: Tuple[float, float] = None) -> Tuple[float, float]:
    vmin = min(values_a + values_b)
    vmax = max(values_a + values_b)
    padding = max(0.02 * (vmax if vmax != 0 else 1.0), (vmax - vmin) * 0.15)
    y_low = vmin - padding
    y_high = vmax + padding
    if clamp is not None:
        y_low = max(clamp[0], y_low)
        y_high = min(clamp[1], y_high)
    if y_low >= y_high:
        y_low = max(0.0, vmin - 0.05)
        y_high = vmax + 0.05
    return y_low, y_high


def plot_wildbench(categories: List[str], metrics_a: Dict[str, Dict[str, float]], metrics_b: Dict[str, Dict[str, float]], label_a: str, label_b: str, outpath: str, suptitle: str = None) -> None:
    _set_publication_style()

    metrics_keys = [
        ('ratio_percent', 'Score2 / Score1 (%)'),
        ('score1_x10', 'Score1 (×10)'),
        ('score2_x10', 'Score2 (×10)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    bar_width = 0.35
    x_positions = list(range(len(categories)))

    for ax, (key, title) in zip(axes, metrics_keys):
        values_a = [metrics_a[c][key] for c in categories]
        values_b = [metrics_b[c][key] for c in categories]

        ax.bar([x - bar_width / 2 for x in x_positions], values_a, width=bar_width, label=label_a, color='#4C78A8', edgecolor='black', linewidth=0.6)
        ax.bar([x + bar_width / 2 for x in x_positions], values_b, width=bar_width, label=label_b, color='#F58518', edgecolor='black', linewidth=0.6)

        # Adaptive y limits; clamp plausible ranges for readability
        clamp = (0.0, None)
        if key == 'ratio_percent':
            clamp = (80.0, 120.0)
        elif key in ('score1_x10', 'score2_x10'):
            clamp = (60.0, 95.0)
        y_low, y_high = _compute_ylim(values_a, values_b, clamp=clamp)

        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([_pretty_label(c) for c in categories])
        ax.set_ylim(y_low, y_high)
        ax.grid(True, axis='y')

        for x, v in zip(x_positions, values_a):
            y_text = min(v + 0.5, y_high - 0.5)
            ax.text(x - bar_width / 2, y_text, f"{v:.1f}", ha='center', va='bottom', fontsize=9)
        for x, v in zip(x_positions, values_b):
            y_text = min(v + 0.5, y_high - 0.5)
            ax.text(x + bar_width / 2, y_text, f"{v:.1f}", ha='center', va='bottom', fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    if suptitle:
        fig.suptitle(suptitle, y=1.12)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    fig.savefig(os.path.splitext(outpath)[0] + '.png')
    plt.close(fig)


def write_csv(categories: List[str], metrics_a: Dict[str, Dict[str, float]], metrics_b: Dict[str, Dict[str, float]], label_a: str, label_b: str, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = [
        'category',
        f'{label_a}_ratio_percent', f'{label_b}_ratio_percent',
        f'{label_a}_score1_x10', f'{label_b}_score1_x10',
        f'{label_a}_score2_x10', f'{label_b}_score2_x10',
    ]
    lines = [','.join(header)]
    for c in categories:
        ma = metrics_a[c]
        mb = metrics_b[c]
        row = [
            c,
            f"{ma['ratio_percent']:.1f}", f"{mb['ratio_percent']:.1f}",
            f"{ma['score1_x10']:.1f}", f"{mb['score1_x10']:.1f}",
            f"{ma['score2_x10']:.1f}", f"{mb['score2_x10']:.1f}",
        ]
        lines.append(','.join(row))
    with open(out_csv, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Visualize WildBench (reviews) comparison between two runs.')
    parser.add_argument('--reviews-a', type=str, required=True, help='JSONL file for run A (e.g., normal)')
    parser.add_argument('--reviews-b', type=str, required=True, help='JSONL file for run B (e.g., new/noise)')
    parser.add_argument('--label-a', type=str, default='No noise', help='Legend label for run A')
    parser.add_argument('--label-b', type=str, default='With noise', help='Legend label for run B')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory for figures and CSV')
    parser.add_argument('--title', type=str, default='WildBench: With vs Without Noise', help='Figure super title')
    args = parser.parse_args()

    reviews_a = _read_reviews(args.reviews_a)
    reviews_b = _read_reviews(args.reviews_b)

    means_a = _compute_category_means(reviews_a)
    means_b = _compute_category_means(reviews_b)
    metrics_a = _to_metrics(means_a)
    metrics_b = _to_metrics(means_b)

    categories = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    categories = _order_categories(categories)

    os.makedirs(args.outdir, exist_ok=True)
    fig_out = os.path.join(args.outdir, 'wildbench_with_without_noise.pdf')
    plot_wildbench(categories, metrics_a, metrics_b, args.label_a, args.label_b, fig_out, suptitle=args.title)

    csv_out = os.path.join(args.outdir, 'wildbench_with_without_noise.csv')
    write_csv(categories, metrics_a, metrics_b, args.label_a, args.label_b, csv_out)

    print(f'Saved figure to: {fig_out} and {os.path.splitext(fig_out)[0] + ".png"}')
    print(f'Saved CSV summary to: {csv_out}')


if __name__ == '__main__':
    main()


