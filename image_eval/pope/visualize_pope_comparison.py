import os
import json
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl


def _normalize_text_to_binary_yes_no(text: str) -> int:
    """
    Normalize raw model output to binary prediction following eval_pope.py logic.
    Returns 1 for yes/positive, 0 for no/negative.
    """
    if text.find('.') != -1:
        text = text.split('.')[0]
    text = text.replace(',', '')
    words = text.split(' ')
    normalized = 'no' if ('No' in words or 'not' in words or 'no' in words) else 'yes'
    return 0 if normalized == 'no' else 1


def _load_questions(question_file: str) -> Dict[str, Dict]:
    questions = [json.loads(line) for line in open(question_file)]
    return {question['question_id']: question for question in questions}


def _load_answers(result_file: str) -> List[Dict]:
    return [json.loads(line) for line in open(result_file)]


def _load_labels(label_file: str) -> List[int]:
    label_list = [json.loads(line)['label'] for line in open(label_file, 'r')]
    normalized = []
    for label in label_list:
        if label == 'no':
            normalized.append(0)
        else:
            normalized.append(1)
    return normalized


def _compute_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    pos = 1
    neg = 0
    yes_ratio = predictions.count(1) / len(predictions) if predictions else 0.0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for pred, label in zip(predictions, labels):
        if pred == pos and label == pos:
            true_positive += 1
        elif pred == pos and label == neg:
            false_positive += 1
        elif pred == neg and label == neg:
            true_negative += 1
        elif pred == neg and label == pos:
            false_negative += 1

    precision = float(true_positive) / float(true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = float(true_positive) / float(true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (true_positive + true_negative) / max((true_positive + true_negative + false_positive + false_negative), 1)

    return {
        'TP': true_positive,
        'FP': false_positive,
        'TN': true_negative,
        'FN': false_negative,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'yes_ratio': yes_ratio,
    }


def evaluate_per_category(annotation_dir: str, question_file: str, result_file: str) -> Dict[str, Dict[str, float]]:
    """
    Compute POPE metrics per category for a given result file.
    Returns: {category: metrics_dict}
    """
    questions = _load_questions(question_file)
    answers = _load_answers(result_file)

    # Precompute normalized predictions per question_id
    question_id_to_pred = {}
    for answer in answers:
        raw_text = answer['text']
        pred = _normalize_text_to_binary_yes_no(raw_text)
        question_id_to_pred[answer['question_id']] = pred

    category_to_metrics: Dict[str, Dict[str, float]] = {}

    for filename in os.listdir(annotation_dir):
        if not (filename.startswith('coco_pope_') and filename.endswith('.json')):
            continue
        category = filename[10:-5]

        # Determine which answers belong to this category
        category_question_ids = [qid for qid, q in questions.items() if q['category'] == category]
        preds = [question_id_to_pred[qid] for qid in category_question_ids if qid in question_id_to_pred]
        labels = _load_labels(os.path.join(annotation_dir, filename))

        # Some result files may not include all questions; align lengths
        min_len = min(len(preds), len(labels))
        preds = preds[:min_len]
        labels = labels[:min_len]

        metrics = _compute_metrics(preds, labels)
        category_to_metrics[category] = metrics

    return category_to_metrics


def _set_publication_style():
    # Configure matplotlib for publication-quality output
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


def _order_categories(categories: List[str]) -> List[str]:
    preferred = ['popular', 'random', 'adversarial']
    ordered = [c for c in preferred if c in categories]
    for c in sorted(categories):
        if c not in ordered:
            ordered.append(c)
    return ordered


def plot_metrics_grid_four_conditions(categories: List[str], 
                                    metrics_qwen_normal: Dict[str, Dict[str, float]], 
                                    metrics_qwen_noise: Dict[str, Dict[str, float]],
                                    metrics_llava_normal: Dict[str, Dict[str, float]], 
                                    metrics_llava_noise: Dict[str, Dict[str, float]], 
                                    outpath: str, suptitle: str = None) -> None:
    _set_publication_style()

    metrics_keys: List[Tuple[str, str]] = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 score'),
    ]

    num_cols = 2
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12.0, 8.0), constrained_layout=True)
    axes = axes.flatten()

    # Define colors for the 4 conditions
    colors = ['#4C78A8', '#F58518', '#E45756', '#72B7B2']  # Blue, Orange, Red, Teal
    labels = ['Qwen Normal', 'Qwen Noise', 'LLaVA Normal', 'LLaVA Noise']
    
    bar_width = 0.2
    x_positions = range(len(categories))

    for ax, (key, title) in zip(axes, metrics_keys):
        # Get values for all 4 conditions
        values_qwen_normal = [metrics_qwen_normal[c][key] for c in categories]
        values_qwen_noise = [metrics_qwen_noise[c][key] for c in categories]
        values_llava_normal = [metrics_llava_normal[c][key] for c in categories]
        values_llava_noise = [metrics_llava_noise[c][key] for c in categories]

        # Plot bars for each condition
        ax.bar([x - 1.5*bar_width for x in x_positions], values_qwen_normal, width=bar_width, 
               label=labels[0], color=colors[0], edgecolor='black', linewidth=0.6)
        ax.bar([x - 0.5*bar_width for x in x_positions], values_qwen_noise, width=bar_width, 
               label=labels[1], color=colors[1], edgecolor='black', linewidth=0.6)
        ax.bar([x + 0.5*bar_width for x in x_positions], values_llava_normal, width=bar_width, 
               label=labels[2], color=colors[2], edgecolor='black', linewidth=0.6)
        ax.bar([x + 1.5*bar_width for x in x_positions], values_llava_noise, width=bar_width, 
               label=labels[3], color=colors[3], edgecolor='black', linewidth=0.6)

        # Adaptive y-limits per metric with headroom to avoid label overflow
        all_values = values_qwen_normal + values_qwen_noise + values_llava_normal + values_llava_noise
        vmin = min(all_values)
        vmax = max(all_values)
        padding = max(0.02, (vmax - vmin) * 0.15)
        y_low = max(0.0, vmin - padding)
        y_high = min(1.02, vmax + padding)

        ax.set_title(title)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.set_ylim(y_low, y_high)
        ax.grid(True, axis='y')

        # Place numeric labels on bars
        for i, (x, v) in enumerate(zip(x_positions, values_qwen_normal)):
            y_text = min(v + 0.01, y_high - 0.01)
            ax.text(x - 1.5*bar_width, y_text, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
        for i, (x, v) in enumerate(zip(x_positions, values_qwen_noise)):
            y_text = min(v + 0.01, y_high - 0.01)
            ax.text(x - 0.5*bar_width, y_text, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
        for i, (x, v) in enumerate(zip(x_positions, values_llava_normal)):
            y_text = min(v + 0.01, y_high - 0.01)
            ax.text(x + 0.5*bar_width, y_text, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
        for i, (x, v) in enumerate(zip(x_positions, values_llava_noise)):
            y_text = min(v + 0.01, y_high - 0.01)
            ax.text(x + 1.5*bar_width, y_text, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    # Create legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))

    if suptitle:
        fig.suptitle(suptitle, y=1.08)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    fig.savefig(os.path.splitext(outpath)[0] + '.png')
    plt.close(fig)


def plot_metrics_grid(categories: List[str], metrics_a: Dict[str, Dict[str, float]], metrics_b: Dict[str, Dict[str, float]], label_a: str, label_b: str, outpath: str, suptitle: str = None) -> None:
    _set_publication_style()

    metrics_keys: List[Tuple[str, str]] = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 score'),
    ]

    num_cols = 2
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8.0, 5.4), constrained_layout=True)
    axes = axes.flatten()

    bar_width = 0.35
    x_positions = range(len(categories))

    for ax, (key, title) in zip(axes, metrics_keys):
        values_a = [metrics_a[c][key] for c in categories]
        values_b = [metrics_b[c][key] for c in categories]

        ax.bar([x - bar_width / 2 for x in x_positions], values_a, width=bar_width, label=label_a, color='#4C78A8', edgecolor='black', linewidth=0.6)
        ax.bar([x + bar_width / 2 for x in x_positions], values_b, width=bar_width, label=label_b, color='#F58518', edgecolor='black', linewidth=0.6)

        # Adaptive y-limits per metric with headroom to avoid label overflow
        vmin = min(values_a + values_b)
        vmax = max(values_a + values_b)
        padding = max(0.02, (vmax - vmin) * 0.15)
        y_low = max(0.0, vmin - padding)
        y_high = min(1.02, vmax + padding)

        ax.set_title(title)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.set_ylim(y_low, y_high)
        ax.grid(True, axis='y')

        # Place numeric labels with clipping near the top limit
        for x, v in zip(x_positions, values_a):
            y_text = min(v + 0.01, y_high - 0.01)
            ax.text(x - bar_width / 2, y_text, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        for x, v in zip(x_positions, values_b):
            y_text = min(v + 0.01, y_high - 0.01)
            ax.text(x + bar_width / 2, y_text, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

    if suptitle:
        fig.suptitle(suptitle, y=1.08)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    fig.savefig(os.path.splitext(outpath)[0] + '.png')
    plt.close(fig)


def write_csv_summary_four_conditions(categories: List[str], 
                                    metrics_qwen_normal: Dict[str, Dict[str, float]], 
                                    metrics_qwen_noise: Dict[str, Dict[str, float]],
                                    metrics_llava_normal: Dict[str, Dict[str, float]], 
                                    metrics_llava_noise: Dict[str, Dict[str, float]], 
                                    out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = [
        'category',
        'qwen_normal_accuracy', 'qwen_noise_accuracy', 'llava_normal_accuracy', 'llava_noise_accuracy',
        'qwen_normal_precision', 'qwen_noise_precision', 'llava_normal_precision', 'llava_noise_precision',
        'qwen_normal_recall', 'qwen_noise_recall', 'llava_normal_recall', 'llava_noise_recall',
        'qwen_normal_f1', 'qwen_noise_f1', 'llava_normal_f1', 'llava_noise_f1',
        'qwen_normal_yes_ratio', 'qwen_noise_yes_ratio', 'llava_normal_yes_ratio', 'llava_noise_yes_ratio',
        'qwen_normal_TP', 'qwen_normal_FP', 'qwen_normal_TN', 'qwen_normal_FN',
        'qwen_noise_TP', 'qwen_noise_FP', 'qwen_noise_TN', 'qwen_noise_FN',
        'llava_normal_TP', 'llava_normal_FP', 'llava_normal_TN', 'llava_normal_FN',
        'llava_noise_TP', 'llava_noise_FP', 'llava_noise_TN', 'llava_noise_FN',
    ]
    lines = [','.join(header)]
    for c in categories:
        qn = metrics_qwen_normal[c]
        qno = metrics_qwen_noise[c]
        ln = metrics_llava_normal[c]
        lno = metrics_llava_noise[c]
        row = [
            c,
            f"{qn['accuracy']:.6f}", f"{qno['accuracy']:.6f}", f"{ln['accuracy']:.6f}", f"{lno['accuracy']:.6f}",
            f"{qn['precision']:.6f}", f"{qno['precision']:.6f}", f"{ln['precision']:.6f}", f"{lno['precision']:.6f}",
            f"{qn['recall']:.6f}", f"{qno['recall']:.6f}", f"{ln['recall']:.6f}", f"{lno['recall']:.6f}",
            f"{qn['f1']:.6f}", f"{qno['f1']:.6f}", f"{ln['f1']:.6f}", f"{lno['f1']:.6f}",
            f"{qn['yes_ratio']:.6f}", f"{qno['yes_ratio']:.6f}", f"{ln['yes_ratio']:.6f}", f"{lno['yes_ratio']:.6f}",
            str(qn['TP']), str(qn['FP']), str(qn['TN']), str(qn['FN']),
            str(qno['TP']), str(qno['FP']), str(qno['TN']), str(qno['FN']),
            str(ln['TP']), str(ln['FP']), str(ln['TN']), str(ln['FN']),
            str(lno['TP']), str(lno['FP']), str(lno['TN']), str(lno['FN']),
        ]
        lines.append(','.join(row))

    with open(out_csv, 'w') as f:
        f.write('\n'.join(lines))


def write_csv_summary(categories: List[str], metrics_a: Dict[str, Dict[str, float]], metrics_b: Dict[str, Dict[str, float]], label_a: str, label_b: str, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = [
        'category',
        f'{label_a}_accuracy', f'{label_b}_accuracy',
        f'{label_a}_precision', f'{label_b}_precision',
        f'{label_a}_recall', f'{label_b}_recall',
        f'{label_a}_f1', f'{label_b}_f1',
        f'{label_a}_yes_ratio', f'{label_b}_yes_ratio',
        f'{label_a}_TP', f'{label_a}_FP', f'{label_a}_TN', f'{label_a}_FN',
        f'{label_b}_TP', f'{label_b}_FP', f'{label_b}_TN', f'{label_b}_FN',
    ]
    lines = [','.join(header)]
    for c in categories:
        ma = metrics_a[c]
        mb = metrics_b[c]
        row = [
            c,
            f"{ma['accuracy']:.6f}", f"{mb['accuracy']:.6f}",
            f"{ma['precision']:.6f}", f"{mb['precision']:.6f}",
            f"{ma['recall']:.6f}", f"{mb['recall']:.6f}",
            f"{ma['f1']:.6f}", f"{mb['f1']:.6f}",
            f"{ma['yes_ratio']:.6f}", f"{mb['yes_ratio']:.6f}",
            str(ma['TP']), str(ma['FP']), str(ma['TN']), str(ma['FN']),
            str(mb['TP']), str(mb['FP']), str(mb['TN']), str(mb['FN']),
        ]
        lines.append(','.join(row))

    with open(out_csv, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Visualize POPE metrics: Qwen vs LLaVA with and without noise.')
    parser.add_argument('--annotation-dir', type=str, required=True, help='Directory with coco_pope_*.json files')
    parser.add_argument('--question-file', type=str, required=True, help='Path to llava_pope_test.jsonl (or equivalent)')
    parser.add_argument('--qwen-normal', type=str, required=True, help='Result file JSONL for Qwen without noise')
    parser.add_argument('--qwen-noise', type=str, required=True, help='Result file JSONL for Qwen with noise')
    parser.add_argument('--llava-normal', type=str, required=True, help='Result file JSONL for LLaVA without noise')
    parser.add_argument('--llava-noise', type=str, required=True, help='Result file JSONL for LLaVA with noise')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory for figures and CSV')
    parser.add_argument('--title', type=str, default='POPE Performance: Qwen vs LLaVA with/without Noise', help='Figure super title')

    args = parser.parse_args()

    # Load metrics for all 4 conditions
    metrics_qwen_normal = evaluate_per_category(args.annotation_dir, args.question_file, args.qwen_normal)
    metrics_qwen_noise = evaluate_per_category(args.annotation_dir, args.question_file, args.qwen_noise)
    metrics_llava_normal = evaluate_per_category(args.annotation_dir, args.question_file, args.llava_normal)
    metrics_llava_noise = evaluate_per_category(args.annotation_dir, args.question_file, args.llava_noise)

    categories = sorted(set(list(metrics_qwen_normal.keys()) + list(metrics_qwen_noise.keys()) + 
                        list(metrics_llava_normal.keys()) + list(metrics_llava_noise.keys())))
    categories = _order_categories(categories)

    os.makedirs(args.outdir, exist_ok=True)
    
    # Create the 4-condition comparison plot
    fig_out = os.path.join(args.outdir, 'pope_qwen_llava_comparison.pdf')
    plot_metrics_grid_four_conditions(categories, metrics_qwen_normal, metrics_qwen_noise, 
                                    metrics_llava_normal, metrics_llava_noise, fig_out, suptitle=args.title)

    # Create CSV summary
    csv_out = os.path.join(args.outdir, 'pope_qwen_llava_comparison.csv')
    write_csv_summary_four_conditions(categories, metrics_qwen_normal, metrics_qwen_noise, 
                                    metrics_llava_normal, metrics_llava_noise, csv_out)

    print(f'Saved figure to: {fig_out} and {os.path.splitext(fig_out)[0] + ".png"}')
    print(f'Saved CSV summary to: {csv_out}')


if __name__ == '__main__':
    main()


