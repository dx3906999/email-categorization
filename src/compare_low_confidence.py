"""
Extract BERT low-confidence (<0.7) samples, LLM-label them,
then compare rule-only vs BERT-only vs fused on this subset.

Usage:
    python src/compare_low_confidence.py --extract     # create sample JSONL
    python src/compare_low_confidence.py --compare      # compare after LLM labeling
    python src/compare_low_confidence.py                # full pipeline (extract + print label cmd)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CATEGORIES = ['adult', 'gambling', 'marketing', 'phishing', 'fraud']
CATEGORY_LABELS = {
    'adult': 'adult', 'gambling': 'gambling',
    'marketing': 'marketing', 'phishing': 'phishing', 'fraud': 'fraud',
}
LABEL_CN = {
    'adult': '色情', 'gambling': '赌博', 'marketing': '营销',
    'phishing': '钓鱼', 'fraud': '发票',
}


def extract_low_confidence(strategy_file: str, output_file: str, threshold: float = 0.7):
    """Extract all low-confidence BERT predictions into a JSONL for LLM labeling."""
    strategy_file = Path(strategy_file)
    output_file = Path(output_file)

    with strategy_file.open('r', encoding='utf-8') as f:
        data = json.load(f)

    bert_results = data['results']['bert']
    rule_results = data['results']['rule']
    fused_results = data['results']['fused']

    # Find low-confidence indices
    low_indices = [i for i, r in enumerate(bert_results) if r['confidence'] < threshold]
    print(f"Low-confidence BERT predictions (< {threshold}): {len(low_indices)}")
    print(f"  BERT conf range: {min(bert_results[i]['confidence'] for i in low_indices):.3f} - "
          f"{max(bert_results[i]['confidence'] for i in low_indices):.3f}")

    # Distribution
    bert_cats = Counter(bert_results[i]['category'] for i in low_indices)
    print(f"\n  Distribution:")
    for cat in CATEGORIES:
        cnt = bert_cats.get(cat, 0)
        print(f"    {LABEL_CN[cat]:6s}  {cnt:>5}")

    # Load original records
    from parser import parse_file
    records = parse_file("database/spam_email_data.log")
    record_map = {r.record_id: r for r in records}

    # Build output
    samples = []
    for i in low_indices:
        rid = bert_results[i]['record_id']
        rec = record_map.get(rid)
        if rec:
            samples.append({
                'record_id': rid,
                'from_name': rec.from_name or '',
                'subject': rec.subject or '',
                'content': (rec.content or '')[:500],
                'bert_category': bert_results[i]['category'],
                'bert_confidence': bert_results[i]['confidence'],
                'rule_category': rule_results[i]['category'] or '',
                'rule_confidence': rule_results[i]['confidence'],
                'fused_category': fused_results[i]['category'],
                'fused_source': fused_results[i].get('source', ''),
            })

    with output_file.open('w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\nSample saved: {output_file} ({len(samples)} records)")
    print(f"\n  Next step: python src/llm_label.py {output_file} --concurrency 10")
    return samples


def compare_low_confidence(sample_file: str, strategy_file: str):
    """Compare strategies on low-confidence samples using LLM labels as ground truth."""
    sample_file = Path(sample_file)
    strategy_file = Path(strategy_file)

    # Load LLM-labeled samples
    labeled_file = sample_file.with_name(sample_file.stem + '_labeled.jsonl')
    if not labeled_file.exists():
        print(f"[ERROR] Labeled file not found: {labeled_file}")
        return

    samples = []
    with labeled_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} LLM-labeled low-confidence samples")

    # Load strategy results
    with strategy_file.open('r', encoding='utf-8') as f:
        strat_data = json.load(f)

    rule_map = {r['record_id']: r for r in strat_data['results']['rule']}
    bert_map = {r['record_id']: r for r in strat_data['results']['bert']}
    fused_map = {r['record_id']: r for r in strat_data['results']['fused']}

    # Compare
    total = 0
    valid = 0
    results = {s: {'correct': 0, 'total': 0, 'wrong_details': []} for s in ['rule', 'bert', 'fused']}

    cat_metrics = {c: {s: {'tp': 0, 'fp': 0, 'fn': 0} for s in ['rule', 'bert', 'fused']}
                   for c in CATEGORIES}

    confusion = {s: Counter() for s in ['rule', 'bert', 'fused']}
    bert_rule_compare = Counter()  # on low-conf: bert vs rule

    for sample in samples:
        rid = sample['record_id']
        llm_cat = sample.get('llm_category', '').strip()
        llm_conf = sample.get('llm_confidence', 0)
        if sample.get('error'):
            continue
        total += 1
        if llm_cat not in CATEGORIES or llm_conf < 0.5:
            continue
        valid += 1

        for strat_name, strat_map in [('rule', rule_map), ('bert', bert_map), ('fused', fused_map)]:
            sr = strat_map.get(rid)
            if not sr:
                continue
            sc = sr['category']
            results[strat_name]['total'] += 1
            confusion[strat_name][(llm_cat, sc)] += 1

            if sc == llm_cat:
                results[strat_name]['correct'] += 1
            else:
                results[strat_name]['wrong_details'].append({
                    'record_id': rid,
                    'llm': llm_cat, 'predicted': sc,
                    'llm_conf': llm_conf,
                    'subject': sample.get('subject', '')[:80],
                })

            # Per-category metrics
            if llm_cat in CATEGORIES:
                if sc == llm_cat:
                    cat_metrics[llm_cat][strat_name]['tp'] += 1
                else:
                    cat_metrics[llm_cat][strat_name]['fn'] += 1
                    if sc in CATEGORIES:
                        cat_metrics[sc][strat_name]['fp'] += 1

        # BERT vs rule on this sample
        br = bert_map.get(rid, {})
        rr = rule_map.get(rid, {})
        bc = br.get('category', '')
        rc = rr.get('category', '')
        if bc and rc:
            bert_rule_compare[(rc, bc)] += 1

    # ====================== Print Results ======================

    print(f"\n{'='*60}")
    print(f"  Low-Confidence Comparison ({valid} valid / {total} total)")
    print(f"  (BERT confidence < 0.7, LLM confidence >= 0.5)")
    print(f"{'='*60}")

    # Overall accuracy
    print(f"\n  -- Overall Accuracy --")
    for s_name in ['rule', 'bert', 'fused']:
        r = results[s_name]
        acc = r['correct'] / r['total'] * 100 if r['total'] else 0
        print(f"  {s_name:6s}: {r['correct']:>4}/{r['total']}  {acc:.1f}%")

    # Per-category F1
    print(f"\n  -- Per-Category F1 --")
    print(f"  {'Category':10s}  {'Rule':>8s}  {'BERT':>8s}  {'Fused':>8s}  {'Count':>6s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
    cat_f1s = {}
    for s_name in ['rule', 'bert', 'fused']:
        cat_f1s[s_name] = {}
    for cat in CATEGORIES:
        llm_cnt = sum(1 for s in samples
                      if s.get('llm_category', '').strip() == cat
                      and not s.get('error')
                      and s.get('llm_confidence', 0) >= 0.5)
        vals = []
        for s_name in ['rule', 'bert', 'fused']:
            m = cat_metrics[cat][s_name]
            tp, fp, fn = m['tp'], m['fp'], m['fn']
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            cat_f1s[s_name][cat] = f1
            vals.append(f"{f1:.3f}")
        print(f"  {LABEL_CN[cat]:8s}  {vals[0]:>8s}  {vals[1]:>8s}  {vals[2]:>8s}  {llm_cnt:>5}")

    # Macro-F1
    macro_vals = []
    for s_name in ['rule', 'bert', 'fused']:
        macro_vals.append(f"{sum(cat_f1s[s_name].values())/len(CATEGORIES):.3f}")
    print(f"  {'Macro-F1':8s}  {macro_vals[0]:>8s}  {macro_vals[1]:>8s}  {macro_vals[2]:>8s}")

    # IoU Score
    print(f"\n  -- IoU Score --")
    for s_name in ['rule', 'bert', 'fused']:
        ious = []
        for cat in CATEGORIES:
            m = cat_metrics[cat][s_name]
            union = m['tp'] + m['fp'] + m['fn']
            iou = m['tp'] / union if union > 0 else 0
            ious.append(iou)
        mean_iou = sum(ious) / len(ious)
        print(f"  {s_name:6s}: Mean IoU = {mean_iou:.4f}  Score = {100*mean_iou:.2f}")

    # BERT vs Rule analysis on low confidence
    print(f"\n  -- BERT vs Rule on Low Confidence --")
    bert_correct = results['bert']['correct']
    rule_correct = results['rule']['correct']
    print(f"  BERT correct: {bert_correct}/{valid} ({bert_correct/valid*100:.1f}%)")
    print(f"  Rule correct: {rule_correct}/{valid} ({rule_correct/valid*100:.1f}%)")

    # Where does rule beat BERT?
    rule_better = []
    bert_better = []
    both_wrong = []
    for sample in samples:
        rid = sample['record_id']
        llm_cat = sample.get('llm_category', '').strip()
        llm_conf = sample.get('llm_confidence', 0)
        if sample.get('error') or llm_cat not in CATEGORIES or llm_conf < 0.5:
            continue
        bc = bert_map[rid]['category'] if rid in bert_map else ''
        rc = rule_map[rid]['category'] if rid in rule_map else ''
        if rc == llm_cat and bc != llm_cat:
            rule_better.append({
                'rid': rid, 'llm': llm_cat, 'bert': bc, 'rule': rc,
                'subject': sample.get('subject', '')[:80],
            })
        elif bc == llm_cat and rc != llm_cat:
            bert_better.append({'rid': rid, 'llm': llm_cat, 'bert': bc, 'rule': rc})
        elif bc != llm_cat and rc != llm_cat:
            both_wrong.append({
                'rid': rid, 'llm': llm_cat, 'bert': bc, 'rule': rc,
                'subject': sample.get('subject', '')[:80],
            })

    print(f"\n  Rule wins, BERT loses: {len(rule_better)}")
    for d in rule_better[:10]:
        print(f"    LLM={d['llm']:10s}  BERT->{d['bert']:10s}  rule->{d['rule']:10s}  |  {d['subject'][:60]}")

    print(f"\n  BERT wins, Rule loses: {len(bert_better)}")

    print(f"\n  Both wrong: {len(both_wrong)}")
    for d in both_wrong[:10]:
        print(f"    LLM={d['llm']:10s}  BERT->{d['bert']:10s}  rule->{d['rule']:10s}  |  {d['subject'][:60]}")

    # Confusion matrix for BERT
    print(f"\n  -- BERT Confusion Matrix (rows=LLM, cols=BERT) --")
    print(f"  {'':10s}  " + "".join(f"{LABEL_CN[c]:>6s}" for c in CATEGORIES))
    for c1 in CATEGORIES:
        row = "".join(f"{confusion['bert'].get((c1, c2), 0):>6d}" for c2 in CATEGORIES)
        print(f"  {LABEL_CN[c1]:8s}  {row}")

    # Rule coverage analysis
    rule_covered = sum(1 for s in samples
                       if not s.get('error') and s.get('rule_category', '')
                       and s.get('llm_category', '').strip() in CATEGORIES
                       and s.get('llm_confidence', 0) >= 0.5)
    print(f"\n  -- Rule Coverage on Low-Confidence --")
    print(f"  Rule has opinion on: {rule_covered}/{valid} ({rule_covered/valid*100:.1f}%)")

    # Hybrid analysis: what if we trust rule when BERT < 0.7 and rule agrees with BERT?
    # Or: what if we use rule as alternative when BERT is uncertain?
    # Strategy: if BERT < 0.7 and rule has high confidence (>0.7), use rule instead
    hybrid_correct = 0
    hybrid_total = 0
    hybrid_source = Counter()
    for sample in samples:
        rid = sample['record_id']
        llm_cat = sample.get('llm_category', '').strip()
        llm_conf = sample.get('llm_confidence', 0)
        if sample.get('error') or llm_cat not in CATEGORIES or llm_conf < 0.5:
            continue
        bc = bert_map[rid]['category'] if rid in bert_map else ''
        rc = rule_map[rid]['category'] if rid in rule_map else ''
        bconf = bert_map[rid]['confidence'] if rid in bert_map else 0
        rconf = rule_map[rid]['confidence'] if rid in rule_map else 0

        # Simple hybrid: if rule confidence >= 0.5 and BERT < 0.7, prefer rule
        # Actually let's try a few strategies
        pass

    # Try various hybrid strategies
    strategies = {
        'BERT-only': lambda bc, rc, bconf, rconf: bc,
        'Rule-only': lambda bc, rc, bconf, rconf: rc if rc else bc,
        'Rule if BERT<0.7 & rule>=0.3': lambda bc, rc, bconf, rconf: rc if (bconf < 0.7 and rconf >= 0.3 and rc) else bc,
        'Rule if BERT<0.7 & rule>=0.5': lambda bc, rc, bconf, rconf: rc if (bconf < 0.7 and rconf >= 0.5 and rc) else bc,
        'Rule if BERT<0.7 & rule>=0.7': lambda bc, rc, bconf, rconf: rc if (bconf < 0.7 and rconf >= 0.7 and rc) else bc,
    }

    print(f"\n  -- Hybrid Strategy Search --")
    print(f"  {'Strategy':35s}  {'Correct':>8s}  {'Acc':>8s}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}")
    for name, strat_fn in strategies.items():
        correct = 0
        for sample in samples:
            rid = sample['record_id']
            llm_cat = sample.get('llm_category', '').strip()
            llm_conf = sample.get('llm_confidence', 0)
            if sample.get('error') or llm_cat not in CATEGORIES or llm_conf < 0.5:
                continue
            bc = bert_map[rid]['category'] if rid in bert_map else ''
            rc = rule_map[rid]['category'] if rid in rule_map else ''
            bconf = bert_map[rid]['confidence'] if rid in bert_map else 0
            rconf = rule_map[rid]['confidence'] if rid in rule_map else 0

            chosen = strat_fn(bc, rc, bconf, rconf)
            if chosen == llm_cat:
                correct += 1
        print(f"  {name:35s}  {correct:>7}  {correct/valid*100:>7.1f}%")

    # Category-specific hybrid (from report 4.10)
    print(f"\n  -- Category-Specific Hybrid --")
    # For adult and phishing: when BERT < 0.7, prefer rule if available
    cat_hybrid_correct = 0
    for sample in samples:
        rid = sample['record_id']
        llm_cat = sample.get('llm_category', '').strip()
        llm_conf = sample.get('llm_confidence', 0)
        if sample.get('error') or llm_cat not in CATEGORIES or llm_conf < 0.5:
            continue
        bc = bert_map[rid]['category'] if rid in bert_map else ''
        rc = rule_map[rid]['category'] if rid in rule_map else ''
        bconf = bert_map[rid]['confidence'] if rid in bert_map else 0
        rconf = rule_map[rid]['confidence'] if rid in rule_map else 0

        # Hybrid: if BERT predicts adult or phishing with low confidence, try rule
        if bconf < 0.7 and bc in ['adult', 'phishing'] and rc:
            chosen = rc
        else:
            chosen = bc
        if chosen == llm_cat:
            cat_hybrid_correct += 1

    print(f"  adult/phishing->rule: {cat_hybrid_correct}/{valid}  {cat_hybrid_correct/valid*100:.1f}%")

    return {
        'accuracy': {s: results[s]['correct']/results[s]['total'] for s in ['rule', 'bert', 'fused']},
        'cat_metrics': cat_metrics,
        'confusion': {s: dict(confusion[s]) for s in ['rule', 'bert', 'fused']},
        'rule_better': rule_better,
        'bert_better': bert_better,
        'both_wrong': both_wrong,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description='Low-confidence strategy comparison')
    p.add_argument('--extract', action='store_true', help='Extract low-confidence samples')
    p.add_argument('--compare', action='store_true', help='Compare strategies on labeled samples')
    p.add_argument('--threshold', type=float, default=0.7, help='BERT confidence threshold')
    p.add_argument('--strategy-file', default='data/strategy_comparison.json')
    p.add_argument('--sample-output', default='data/low_confidence_sample.jsonl')
    args = p.parse_args()

    if args.compare:
        compare_low_confidence(args.sample_output, args.strategy_file)
    elif args.extract:
        extract_low_confidence(args.strategy_file, args.sample_output, args.threshold)
    else:
        # Full pipeline
        extract_low_confidence(args.strategy_file, args.sample_output, args.threshold)


if __name__ == '__main__':
    main()
