"""
Sample 1000 records stratified by fused category, LLM-label them,
then compare LLM labels against rule-only, BERT-only, and fused strategies.

Usage:
    python src/validate_sample.py                    # full pipeline
    python src/validate_sample.py --sample-only       # just create sample JSONL
    python src/validate_sample.py --compare-only      # just compare existing labeled results
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CATEGORIES = ['adult', 'gambling', 'marketing', 'phishing', 'fraud']
CATEGORY_LABELS = {
    'adult': '色情淫秽', 'gambling': '赌博博彩',
    'marketing': '营销推广', 'phishing': '钓鱼诈骗', 'fraud': '假发票/诈骗',
}

random.seed(42)


def create_sample(strategy_file: str, output_file: str, n: int = 1000):
    """Create stratified sample from strategy comparison results."""
    strategy_file = Path(strategy_file)
    output_file = Path(output_file)

    # Load fused results
    with strategy_file.open('r', encoding='utf-8') as f:
        data = json.load(f)

    fused = data['results']['fused']
    rule = data['results']['rule']
    bert = data['results']['bert']

    # Group by fused category
    by_cat = {c: [] for c in CATEGORIES}
    for i, r in enumerate(fused):
        cat = r['category']
        if cat in CATEGORIES:
            by_cat[cat].append(i)

    # Calculate per-category quota (proportional)
    total_classified = sum(len(v) for v in by_cat.values())
    quotas = {}
    for cat in CATEGORIES:
        quotas[cat] = max(10, round(n * len(by_cat[cat]) / total_classified))
    # Adjust to exactly n
    diff = n - sum(quotas.values())
    while diff > 0:
        # Give extra to largest category
        largest = max(quotas, key=lambda c: len(by_cat[c]) - quotas[c])
        quotas[largest] += 1
        diff -= 1
    while diff < 0:
        # Remove from largest
        largest = max(quotas, key=quotas.get)
        if quotas[largest] > 10:
            quotas[largest] -= 1
            diff += 1

    # Sample
    sampled_indices = set()
    for cat, quota in quotas.items():
        indices = random.sample(by_cat[cat], min(quota, len(by_cat[cat])))
        sampled_indices.update(indices)

    print(f"Sampled {len(sampled_indices)} records (stratified):")
    for cat in CATEGORIES:
        cnt = sum(1 for i in sampled_indices if fused[i]['category'] == cat)
        print(f"  {CATEGORY_LABELS[cat]:10s}  {cnt:>4}")

    # Load original records to get from_name
    from parser import parse_file
    records = parse_file("database/spam_email_data.log")
    record_map = {r.record_id: r for r in records}

    # Build output JSONL
    samples = []
    for i in sorted(sampled_indices):
        rid = fused[i]['record_id']
        rec = record_map.get(rid)
        if rec:
            samples.append({
                'record_id': rid,
                'from_name': rec.from_name or '',
                'subject': rec.subject or '',
                'content': (rec.content or '')[:500],
                'fused_category': fused[i]['category'],
                'fused_confidence': fused[i]['confidence'],
                'fused_source': fused[i].get('source', ''),
                'rule_category': rule[i]['category'],
                'bert_category': bert[i]['category'],
            })

    with output_file.open('w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\nSample saved: {output_file} ({len(samples)} records)")
    return samples


def compare_with_llm(sample_file: str, strategy_file: str):
    """Compare LLM labels against the three strategies."""
    sample_file = Path(sample_file)
    strategy_file = Path(strategy_file)

    # Load samples
    samples = []
    with sample_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Find the labeled version (look for *_labeled.jsonl)
    labeled_file = sample_file.with_name(sample_file.stem + '_labeled.jsonl')
    if not labeled_file.exists():
        # Try alternative naming
        alt = sample_file.with_name(sample_file.stem.replace('_sample', '') + '_labeled.jsonl')
        if alt.exists():
            labeled_file = alt

    if not labeled_file.exists():
        print(f"[ERROR] Labeled file not found. Expected: {labeled_file}")
        print("  Run: python src/llm_label.py " + str(sample_file))
        return

    # Load LLM labels
    llm_labels = {}
    with labeled_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                llm_labels[r['record_id']] = r

    print(f"Loaded {len(llm_labels)} LLM labels from {labeled_file}")

    # Load strategy results for full comparison
    with strategy_file.open('r', encoding='utf-8') as f:
        strat_data = json.load(f)

    # Build quick lookup: record_id -> strategy results
    rule_map = {r['record_id']: r for r in strat_data['results']['rule']}
    bert_map = {r['record_id']: r for r in strat_data['results']['bert']}
    fused_map = {r['record_id']: r for r in strat_data['results']['fused']}

    # For each sample, compare LLM vs strategy
    total = 0
    results = {'rule': {'correct': 0, 'total': 0},
               'bert': {'correct': 0, 'total': 0},
               'fused': {'correct': 0, 'total': 0}}

    # Per-category metrics
    cat_metrics = {c: {s: {'tp': 0, 'fp': 0, 'fn': 0} for s in ['rule', 'bert', 'fused']}
                   for c in CATEGORIES}

    # Confusion matrices
    confusion = {s: Counter() for s in ['rule', 'bert', 'fused']}

    disagreements = {s: [] for s in ['rule', 'bert', 'fused']}

    for sample in samples:
        rid = sample['record_id']
        llm_r = llm_labels.get(rid)
        if not llm_r or llm_r.get('error'):
            continue
        llm_cat = llm_r.get('llm_category', '').strip()
        llm_conf = llm_r.get('llm_confidence', 0)
        if llm_cat not in CATEGORIES or llm_conf < 0.5:
            continue

        total += 1

        # Compare each strategy
        for strat_name, strat_map in [('rule', rule_map), ('bert', bert_map), ('fused', fused_map)]:
            strat_r = strat_map.get(rid)
            if not strat_r:
                continue
            strat_cat = strat_r['category']
            results[strat_name]['total'] += 1

            confusion[strat_name][(llm_cat, strat_cat)] += 1

            if strat_cat == llm_cat:
                results[strat_name]['correct'] += 1
            else:
                disagreements[strat_name].append({
                    'record_id': rid,
                    'llm': llm_cat,
                    'predicted': strat_cat,
                    'llm_conf': llm_conf,
                    'subject': sample.get('subject', '')[:80],
                    'llm_reason': llm_r.get('llm_reason', '')[:100],
                })

            # Per-category
            if llm_cat in CATEGORIES:
                # TP: strategy predicted this category AND LLM says this category
                if strat_cat == llm_cat:
                    cat_metrics[llm_cat][strat_name]['tp'] += 1
                else:
                    # FN: LLM says this category but strategy predicted something else
                    cat_metrics[llm_cat][strat_name]['fn'] += 1
                    # FP: strategy predicted this category but LLM says otherwise
                    if strat_cat in CATEGORIES:
                        cat_metrics[strat_cat][strat_name]['fp'] += 1

    # Print results
    print(f"\n{'='*60}")
    print(f"  LLM Validation Results ({total} valid samples)")
    print(f"{'='*60}")

    print(f"\n  -- Overall Accuracy --")
    for strat_name in ['rule', 'bert', 'fused']:
        r = results[strat_name]
        acc = r['correct'] / r['total'] * 100 if r['total'] else 0
        print(f"  {strat_name:6s}: {r['correct']:>4}/{r['total']}  {acc:.1f}%")

    print(f"\n  -- Per-Category F1 (LLM as ground truth) --")
    print(f"  {'Category':12s}  {'Rule':>8s}  {'BERT':>8s}  {'Fused':>8s}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}")
    for cat in CATEGORIES:
        f1_scores = {}
        for s in ['rule', 'bert', 'fused']:
            m = cat_metrics[cat][s]
            tp, fp, fn = m['tp'], m['fp'], m['fn']
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1_scores[s] = f1
        print(f"  {CATEGORY_LABELS[cat]:10s}  "
              f"{f1_scores['rule']:>7.3f}  {f1_scores['bert']:>7.3f}  {f1_scores['fused']:>7.3f}")

    # Macro F1
    print(f"  {'Macro-F1':12s}  {'':>8s}  ", end='')
    for s_name in ['rule', 'bert', 'fused']:
        macro = 0.0
        for c in CATEGORIES:
            m = cat_metrics[c][s_name]
            tp, fp, fn = m['tp'], m['fp'], m['fn']
            if (tp + fp) > 0 and (tp + fn) > 0:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                macro += 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        macro /= len(CATEGORIES)
        print(f"{macro:>7.3f}  ", end='')
    print()

    # IoU-based score (matching report style)
    print(f"\n  -- IoU Score (report metric) --")
    for s in ['rule', 'bert', 'fused']:
        ious = []
        for cat in CATEGORIES:
            m = cat_metrics[cat][s]
            # F = strategy predictions for this cat
            # FTrue = LLM labels for this cat
            stride = [cat_metrics[c][s] for c in CATEGORIES]
            # Actually compute F_k = TP + FP, FTrue_k = TP + FN
            union = m['tp'] + m['fp'] + m['fn']
            iou = m['tp'] / union if union > 0 else 0
            ious.append(iou)
        mean_iou = sum(ious) / len(ious)
        score = 100 * mean_iou
        print(f"  {s:6s}: Mean IoU = {mean_iou:.4f}  Score = {score:.2f}")

    # Show top disagreements
    print(f"\n  -- BERT Disagreements with LLM (top 10) --")
    for d in disagreements['bert'][:10]:
        print(f"  LLM={d['llm']:10s}  BERT={d['predicted']:10s}  "
              f"c={d['llm_conf']:.2f}  |  {d['subject'][:60]}")

    return {
        'accuracy': {s: results[s]['correct'] / results[s]['total'] if results[s]['total'] else 0
                     for s in ['rule', 'bert', 'fused']},
        'cat_metrics': {cat: {s: dict(cat_metrics[cat][s]) for s in ['rule', 'bert', 'fused']}
                       for cat in CATEGORIES},
        'confusion': {s: dict(confusion[s]) for s in ['rule', 'bert', 'fused']},
        'disagreements': {s: disagreements[s][:50] for s in ['rule', 'bert', 'fused']},
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description='LLM Validation: compare strategies against LLM labels')
    p.add_argument('--sample-only', action='store_true', help='Only create sample JSONL')
    p.add_argument('--compare-only', action='store_true', help='Only compare existing labeled results')
    p.add_argument('--n', type=int, default=1000, help='Sample size (default: 1000)')
    p.add_argument('--strategy-file', default='data/strategy_comparison.json')
    p.add_argument('--sample-output', default='data/validation_sample.jsonl')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    if args.compare_only:
        compare_with_llm(args.sample_output, args.strategy_file)
    elif args.sample_only:
        create_sample(args.strategy_file, args.sample_output, args.n)
    else:
        # Full pipeline
        print("=" * 60)
        print("  Step 1/3: Create stratified sample")
        print("=" * 60)
        create_sample(args.strategy_file, args.sample_output, args.n)

        print(f"\n{'='*60}")
        print(f"  Step 2/3: Run LLM labeling")
        print(f"{'='*60}")
        print(f"  Run: python src/llm_label.py {args.sample_output} --concurrency 10")
        print(f"  (Run this manually, then proceed to Step 3)")
        print(f"\n  Step 3/3: Compare")
        print(f"  Run: python src/validate_sample.py --compare-only")


if __name__ == '__main__':
    main()
