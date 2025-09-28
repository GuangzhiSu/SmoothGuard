import argparse
import json
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def load_labels_and_categories(annotation_dir):
    question_id_to_label = {}
    question_id_to_category = {}
    for file in os.listdir(annotation_dir):
        if not (file.startswith('coco_pope_') and file.endswith('.json')):
            continue
        category = file[10:-5]
        with open(os.path.join(annotation_dir, file), 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                qid = obj['question_id']
                question_id_to_label[qid] = 1 if obj['label'] == 'yes' else 0
                question_id_to_category[qid] = category
    return question_id_to_label, question_id_to_category


def normalize_to_yes_no(text):
    # Match eval_pope.py normalization logic (first sentence, simple negation rules)
    if '.' in text:
        text = text.split('.')[0]
    text = text.replace(',', '')
    words = text.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        return 'no'
    return 'yes'


def compute_accuracy_per_sigma(answers_file, labels, categories):
    # Overall bucket
    sigma_to_preds_labels = defaultdict(lambda: {"preds": [], "labels": []})
    # Per-category buckets
    sigma_to_category_buckets = defaultdict(lambda: defaultdict(lambda: {"preds": [], "labels": []}))

    total_entries = 0
    skipped_no_sigma = 0
    skipped_no_label = 0
    processed_entries = 0

    with open(answers_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            total_entries += 1
            obj = json.loads(line)
            sigma = None
            metadata = obj.get('metadata', {})
            if isinstance(metadata, dict):
                sigma = metadata.get('sigma', None)

            # If sigma missing (e.g., legacy runs), skip
            if sigma is None:
                skipped_no_sigma += 1
                continue

            qid = obj.get('question_id')
            
            # Convert padded question_id format back to annotation format
            # Adversarial: 0-3000 (same as annotation)
            # Popular: 1000xxxxx -> xxxxx (remove 1000 prefix and leading zeros)
            # Random: 2000xxxxx -> xxxxx (remove 2000 prefix and leading zeros)
            if qid >= 20000000:  # Random set (2000xxxxx)
                qid = qid - 20000000
            elif qid >= 10000000:  # Popular set (1000xxxxx) 
                qid = qid - 10000000
            # Adversarial set (0-3000) remains unchanged
            
            if qid not in labels:
                skipped_no_label += 1
                continue

            pred_text = normalize_to_yes_no(obj.get('text', ''))
            pred = 0 if pred_text == 'no' else 1
            label = labels[qid]
            cat = categories.get(qid, 'unknown')
            sigma_to_preds_labels[sigma]["preds"].append(pred)
            sigma_to_preds_labels[sigma]["labels"].append(label)
            sigma_to_category_buckets[sigma][cat]["preds"].append(pred)
            sigma_to_category_buckets[sigma][cat]["labels"].append(label)
            processed_entries += 1

    # Compute metrics for overall
    rows = []
    for sigma, d in sigma_to_preds_labels.items():
        preds = d["preds"]
        golds = d["labels"]
        if not preds:
            continue
        preds = np.array(preds)
        golds = np.array(golds)
        acc = float(np.mean(preds == golds))
        rows.append((float(sigma), acc, len(preds)))

    rows.sort(key=lambda x: x[0])
    
    # Print debug information
    print(f"Debug info:")
    print(f"  Total entries in answers file: {total_entries}")
    print(f"  Skipped due to missing sigma: {skipped_no_sigma}")
    print(f"  Skipped due to missing label: {skipped_no_label}")
    print(f"  Successfully processed: {processed_entries}")
    print(f"  Available labels: {len(labels)}")
    
    # Compute metrics per category
    category_to_rows = defaultdict(list)
    for sigma, buckets in sigma_to_category_buckets.items():
        for cat, d in buckets.items():
            preds = d["preds"]
            golds = d["labels"]
            if not preds:
                continue
            preds = np.array(preds)
            golds = np.array(golds)
            acc = float(np.mean(preds == golds))
            category_to_rows[cat].append((float(sigma), acc, len(preds)))
    for cat in category_to_rows:
        category_to_rows[cat].sort(key=lambda x: x[0])

    return rows, category_to_rows


def save_csv_and_plot(rows, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    png_path = os.path.join(out_dir, f"{stem}.png")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")  # Also save as PDF for papers

    with open(csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["sigma", "accuracy", "num_samples"])
        for r in rows:
            writer.writerow([r[0], f"{r[1]:.6f}", r[2]])

    sigmas = [r[0] for r in rows]
    accs = [r[1] for r in rows]

    # Set style for publication-quality plots
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4

    # Create figure with better proportions for papers
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot with enhanced styling
    line = ax.plot(sigmas, accs, 'o-', linewidth=2.5, markersize=8, 
                   markerfacecolor='white', markeredgewidth=2, 
                   color='#1f77b4', alpha=0.9)
    
    # Customize axes
    ax.set_xlabel('Noise Level (σ)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Model Robustness to Gaussian Noise', fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits with some padding
    ax.set_xlim(min(sigmas) - 0.02, max(sigmas) + 0.02)
    ax.set_ylim(min(accs) - 0.05, max(accs) + 0.05)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)
    
    # Add value labels on points for precision
    for i, (sigma, acc) in enumerate(zip(sigmas, accs)):
        ax.annotate(f'{acc:.3f}', (sigma, acc), 
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')  # PDF for papers
    plt.close()  # Close to free memory
    
    print(f"Saved CSV to {csv_path}, PNG to {png_path}, and PDF to {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers-file', type=str, required=True, help='JSONL with per-sigma answers')
    parser.add_argument('--annotation-dir', type=str, required=True, help='POPE annotation dir (contains coco_pope_*.json)')
    parser.add_argument('--out-dir', type=str, default=None, help='Directory to save CSV/plot; defaults to answers-file dir')
    parser.add_argument('--out-stem', type=str, default='accuracy_vs_sigma', help='Output file stem (without extension)')
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir else os.path.dirname(args.answers_file) or '.'

    labels, categories = load_labels_and_categories(args.annotation_dir)
    rows, category_to_rows = compute_accuracy_per_sigma(args.answers_file, labels, categories)
    if not rows:
        print('No rows computed. Ensure your answers file contains metadata.sigma and matching question_ids.')
        return

    # Overall
    save_csv_and_plot(rows, out_dir, args.out_stem)
    # Per category
    for cat, cat_rows in category_to_rows.items():
        save_csv_and_plot(cat_rows, out_dir, f"{args.out_stem}_{cat}")


if __name__ == '__main__':
    main()


