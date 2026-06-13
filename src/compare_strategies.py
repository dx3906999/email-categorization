"""
Compare three classification strategies on all 24,000 records:
  1. Rule-only
  2. BERT-only
  3. Fused (BERT + rules)

Usage:
    python src/compare_strategies.py
    python src/compare_strategies.py --limit 1000  # quick test
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parser import parse_file
from classifier import EmailClassifier, CATEGORIES, CATEGORY_LABELS
from bert_train import BertPredictor


def run_rule_only(records):
    """Rule-based classification only."""
    clf = EmailClassifier(rule_threshold=0.2)
    clf.load_default_rules()
    results = []
    for r in records:
        cat, conf, _ = clf.classify(r)
        results.append({'record_id': r.record_id, 'category': cat, 'confidence': conf})
    return results


def run_bert_only(records, bert):
    """BERT-only classification."""
    results = []
    for i, r in enumerate(records):
        cat, conf = bert.predict(r.subject, r.content, from_name=r.from_name or '')
        results.append({'record_id': r.record_id, 'category': cat, 'confidence': conf})
        if (i + 1) % 2000 == 0:
            print(f"  BERT-only progress: {i+1}/{len(records)}")
    return results


def run_fused(records, clf):
    """Fused classification (BERT + rules)."""
    result = clf.classify_all_fused(records, update_record=False)
    return result['fusion_results']


def _save_checkpoint(path: str, data: dict) -> None:
    """Save partial results to avoid losing expensive computation."""
    import json as _json
    with open(path, 'w', encoding='utf-8') as f:
        _json.dump(data, f, ensure_ascii=False)


def compare(results_rule, results_bert, results_fused, total):
    """Print comparison table."""
    # Count categories for each strategy
    rule_cnt = Counter(r['category'] or 'unclassified' for r in results_rule)
    bert_cnt = Counter(r['category'] or 'unclassified' for r in results_bert)
    fused_cnt = Counter(r['category'] or 'unclassified' for r in results_fused)

    rule_cls = sum(1 for r in results_rule if r['category'])
    bert_cls = sum(1 for r in results_bert if r['category'])
    fused_cls = sum(1 for r in results_fused if r['category'])

    print(f"\n{'='*70}")
    print(f"  三种分类策略对比 ({total:,} 封邮件)")
    print(f"{'='*70}")

    # Coverage
    print(f"\n  ── 覆盖率 ──")
    print(f"  纯规则:   {rule_cls:>8,}  ({rule_cls/total*100:.1f}%)")
    print(f"  纯 BERT:  {bert_cls:>8,}  ({bert_cls/total*100:.1f}%)")
    print(f"  融合:     {fused_cls:>8,}  ({fused_cls/total*100:.1f}%)")

    # Category distribution
    print(f"\n  ── 类别分布 ──")
    print(f"  {'类别':12s}  {'纯规则':>8s}  {'纯BERT':>8s}  {'融合':>8s}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}")
    for cat in CATEGORIES:
        print(f"  {CATEGORY_LABELS[cat]:10s}  {rule_cnt.get(cat,0):>8,}  {bert_cnt.get(cat,0):>8,}  {fused_cnt.get(cat,0):>8,}")
    print(f"  {'未分类':10s}  {rule_cnt.get('unclassified',0):>8,}  {bert_cnt.get('unclassified',0):>8,}  {fused_cnt.get('unclassified',0):>8,}")

    # Pairwise agreement
    print(f"\n  ── 两两一致率 ──")

    # Rule vs BERT
    agree_rb = sum(1 for a, b in zip(results_rule, results_bert) if a['category'] == b['category'])
    print(f"  Rule <-> BERT:  {agree_rb/total*100:.1f}%  ({agree_rb:,}/{total:,})")

    # Rule vs Fused
    agree_rf = sum(1 for a, b in zip(results_rule, results_fused) if a['category'] == b['category'])
    print(f"  Rule <-> Fused: {agree_rf/total*100:.1f}%  ({agree_rf:,}/{total:,})")

    # BERT vs Fused
    agree_bf = sum(1 for a, b in zip(results_bert, results_fused) if a['category'] == b['category'])
    print(f"  BERT <-> Fused: {agree_bf/total*100:.1f}%  ({agree_bf:,}/{total:,})")

    # Disagreement analysis: where do they differ?
    print(f"\n  ── 分歧分析 ──")

    # BERT overrides rule
    bert_wins = 0
    rule_wins = 0
    for rb, bb in zip(results_rule, results_bert):
        if rb['category'] and bb['category'] and rb['category'] != bb['category']:
            bert_wins += 1
        elif bb['category'] and not rb['category']:
            bert_wins += 1
        elif rb['category'] and not bb['category']:
            rule_wins += 1
    print(f"  BERT 覆盖规则:   {bert_wins:,}  ({bert_wins/total*100:.1f}%)")
    print(f"  规则独自分:      {rule_wins:,}  ({rule_wins/total*100:.1f}%)")

    # Decision sources in fused
    source_cnt = Counter(r['source'] for r in results_fused if r['category'])
    print(f"\n  ── 融合决策来源 ──")
    for src, cnt in sorted(source_cnt.items(), key=lambda x: -x[1]):
        label = {'bert': 'BERT 主导', 'consensus': '规则+BERT 一致',
                 'rule_fallback': '规则兜底', 'ip': 'IP 传播'}.get(src, src)
        print(f"  {label:16s}  {cnt:>8,}  ({cnt/fused_cls*100:.1f}%)")

    # Confidence distribution
    print(f"\n  ── 策略置信度统计 ──")
    for name, res in [('规则', results_rule), ('BERT', results_bert), ('融合', results_fused)]:
        confs = [r['confidence'] for r in res if r['category']]
        if confs:
            avg = sum(confs) / len(confs)
            high = sum(1 for c in confs if c >= 0.9)
            mid = sum(1 for c in confs if 0.7 <= c < 0.9)
            low = sum(1 for c in confs if 0.5 <= c < 0.7)
            vlow = sum(1 for c in confs if c < 0.5)
            print(f"  {name:4s}: 均值={avg:.3f}  ≥0.9={high:,}  0.7-0.9={mid:,}  0.5-0.7={low:,}  <0.5={vlow:,}")

    return {
        'rule_cnt': dict(rule_cnt),
        'bert_cnt': dict(bert_cnt),
        'fused_cnt': dict(fused_cnt),
        'agree_rb': agree_rb,
        'agree_rf': agree_rf,
        'agree_bf': agree_bf,
        'bert_wins': bert_wins,
        'rule_wins': rule_wins,
        'sources': dict(source_cnt),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description='比较三种分类策略')
    p.add_argument('--limit', type=int, default=0, help='只分析前 N 条')
    p.add_argument('--log', default='database/spam_email_data.log')
    p.add_argument('--bert-model', default='models/bert_classifier')
    p.add_argument('--output', default='data/strategy_comparison.json', help='输出 JSON')
    args = p.parse_args()

    # Load data
    print(f"正在加载数据: {args.log}")
    records = parse_file(args.log)
    if args.limit:
        records = records[:args.limit]
    total = len(records)
    print(f"共 {total:,} 条")

    # Load BERT
    print(f"\n加载 BERT: {args.bert_model}")
    bert = BertPredictor(model_path=args.bert_model)

    # 1. Rule-only
    print(f"\n[1/3] Pure rule classification...")
    t0 = time.time()
    results_rule = run_rule_only(records)
    rule_time = time.time() - t0
    rule_cls = sum(1 for r in results_rule if r['category'])
    print(f"  Done: {rule_cls:,} classified  ({rule_time:.0f}s)")

    # Save checkpoint
    _save_checkpoint(args.output, {
        'total': total,
        'timing': {'rule': round(rule_time), 'bert': 0, 'fused': 0},
        'results_rule': results_rule,
    })

    # 2. BERT-only
    print(f"\n[2/3] Pure BERT classification...")
    t0 = time.time()
    results_bert = run_bert_only(records, bert)
    bert_time = time.time() - t0
    bert_cls = sum(1 for r in results_bert if r['category'])
    print(f"  Done: {bert_cls:,} classified  ({bert_time:.0f}s)")

    # Save checkpoint
    _save_checkpoint(args.output, {
        'total': total,
        'timing': {'rule': round(rule_time), 'bert': round(bert_time), 'fused': 0},
        'results_bert': results_bert,
    })

    # 3. Fused
    print(f"\n[3/3] 融合分类...")
    t0 = time.time()
    clf = EmailClassifier(rule_threshold=0.2)
    clf.load_default_rules()
    clf._bert = bert
    results_fused_raw = run_fused(records, clf)
    fused_time = time.time() - t0
    fused_cls = sum(1 for r in results_fused_raw if r['category'])
    print(f"  完成: {fused_cls:,} 已分类  耗时 {fused_time:.0f}s")

    # Compare
    stats = compare(results_rule, results_bert, results_fused_raw, total)

    # Save
    output_path = Path(args.output)
    output_data = {
        'total': total,
        'timing': {'rule': round(rule_time), 'bert': round(bert_time), 'fused': round(fused_time)},
        'stats': stats,
        'results': {
            'rule': [{'record_id': r['record_id'], 'category': r['category'], 'confidence': r['confidence']}
                     for r in results_rule],
            'bert': [{'record_id': r['record_id'], 'category': r['category'], 'confidence': r['confidence']}
                     for r in results_bert],
            'fused': [{'record_id': r['record_id'], 'category': r['category'],
                       'confidence': r['confidence'], 'source': r.get('source', ''),
                       'rule_cat': r.get('rule_cat', ''), 'bert_cat': r.get('bert_cat', '')}
                      for r in results_fused_raw],
        },
    }
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")

    # Save timing info
    print(f"\n  耗时对比: 规则 {rule_time:.0f}s  |  BERT {bert_time:.0f}s  |  融合 {fused_time:.0f}s")


if __name__ == '__main__':
    main()
