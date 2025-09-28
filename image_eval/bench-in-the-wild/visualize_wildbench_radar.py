import os
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def _set_publication_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


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


def parse_kv_pairs(pairs: str) -> Dict[str, float]:
    """
    Parse string like: "all:103.7,llava_bench_complex:100.9,llava_bench_conv:102.2,llava_bench_detail:110.8"
    into dict {key: float(value)}.
    """
    result: Dict[str, float] = {}
    if pairs is None:
        return result
    for chunk in pairs.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(f'Invalid pair: {chunk}. Expected key:value')
        key, value = chunk.split(':', 1)
        result[key.strip()] = float(value.strip())
    return result


def _radar_axes(num_vars: int):
    # Angles for radar (closed polygon)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
    return fig, ax, angles


def _round_to_step(x, step=5, mode="floor"):
    if mode == "floor":
        return step * np.floor(x / step)
    return step * np.ceil(x / step)

def plot_radar(categories, ratios_a, ratios_b, label_a, label_b, outpath, title=None):
    """
    categories: list[str]
    ratios_a, ratios_b: list[float]  (same length as categories)
    """
    _set_publication_style()
    fig, ax, angles = _radar_axes(len(categories))

    # Determine radial limits with padding and round to nice 5-pt ticks
    vmin = min(min(ratios_a), min(ratios_b))
    vmax = max(max(ratios_a), max(ratios_b))
    pad = max(1.0, 0.15 * (vmax - vmin))  # >= 1 for clarity
    rmin = max(80.0, vmin - pad)
    rmax = min(120.0, vmax + pad)

    # Round limits to 5-step boundaries
    rmin = float(_round_to_step(rmin, 5, "floor"))
    rmax = float(_round_to_step(rmax, 5, "ceil"))
    ax.set_ylim(rmin, rmax)

    # Radial ticks every 5; place labels down-left to avoid top overlap
    ticks = np.arange(rmin, rmax + 0.1, 5)
    ax.set_rgrids(ticks, angle=225)
    ax.set_rlabel_position(225)

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_pretty_label(c) for c in categories])
    # Increase angular tick label padding to separate from the title area
    ax.tick_params(axis='x', pad=14)

    # Prepare values (close loops)
    vals_a = ratios_a + ratios_a[:1]
    vals_b = ratios_b + ratios_b[:1]

    # Plot A (No noise) – default colors (no explicit color set)
    ax.plot(angles, vals_a, linewidth=2.0, label=label_a)
    ax.fill(angles, vals_a, alpha=0.15)

    # Plot B (With noise) – default second color
    ax.plot(angles, vals_b, linewidth=2.0, label=label_b)
    ax.fill(angles, vals_b, alpha=0.15)

    # Value labels with opposite offsets and small angular jitter
    jitter = 0.06  # radians
    up_offset = 0.012 * (rmax - rmin) + 0.6    # radial offset for label A
    down_offset = 0.012 * (rmax - rmin) + 0.6  # radial offset for label B

    def _clamp(val, lo, hi):
        return max(lo, min(hi, val))

    for ang, va, vb in zip(angles[:-1], ratios_a, ratios_b):
        # positions: A slightly ahead, B slightly behind the tick angle
        ax.text(ang + jitter,
                _clamp(va + up_offset, rmin + 0.5, rmax - 0.5),
                f"{va:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(ang - jitter,
                _clamp(vb - down_offset, rmin + 0.5, rmax - 0.5),
                f"{vb:.1f}", ha="center", va="top", fontsize=9)

    if title:
        # Raise title higher to avoid collision with top category label
        ax.set_title(title, y=1.14)

    # Legend above, outside the axes
    # Place legend just below the title
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)

    # Clean margins
    fig.tight_layout(pad=1.2)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(os.path.splitext(outpath)[0] + ".png", bbox_inches="tight")
    plt.close(fig)


def write_csv(categories: List[str], ratios_a: List[float], ratios_b: List[float], label_a: str, label_b: str, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = ['category', f'{label_a}_ratio_percent', f'{label_b}_ratio_percent']
    lines = [','.join(header)]
    for c, va, vb in zip(categories, ratios_a, ratios_b):
        lines.append(','.join([c, f'{va:.1f}', f'{vb:.1f}']))
    with open(out_csv, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Radar chart for WildBench relative scores (Score2/Score1 in %).')
    parser.add_argument('--ratios-a', type=str, required=True, help='Key:Value pairs for run A, comma-separated. Example: "all:103.7,llava_bench_complex:100.9,llava_bench_conv:102.2,llava_bench_detail:110.8"')
    parser.add_argument('--ratios-b', type=str, required=True, help='Key:Value pairs for run B, comma-separated.')
    parser.add_argument('--label-a', type=str, default='No noise')
    parser.add_argument('--label-b', type=str, default='With noise')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--title', type=str, default='WildBench Relative Score (Radar)')
    args = parser.parse_args()

    dict_a = parse_kv_pairs(args.ratios_a)
    dict_b = parse_kv_pairs(args.ratios_b)

    if set(dict_a.keys()) != set(dict_b.keys()):
        raise ValueError('Keys of --ratios-a and --ratios-b must match.')

    categories = _order_categories(sorted(dict_a.keys()))
    ratios_a = [dict_a[c] for c in categories]
    ratios_b = [dict_b[c] for c in categories]

    os.makedirs(args.outdir, exist_ok=True)
    fig_out = os.path.join(args.outdir, 'wildbench_radar_relative_scores.pdf')
    plot_radar(categories, ratios_a, ratios_b, args.label_a, args.label_b, fig_out, title=args.title)

    csv_out = os.path.join(args.outdir, 'wildbench_radar_relative_scores.csv')
    write_csv(categories, ratios_a, ratios_b, args.label_a, args.label_b, csv_out)

    print(f'Saved figure to: {fig_out} and {os.path.splitext(fig_out)[0] + ".png"}')
    print(f'Saved CSV summary to: {csv_out}')


if __name__ == '__main__':
    main()


