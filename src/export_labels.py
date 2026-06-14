"""
Export email classification results: record_id + spam_type (numeric).

Category mapping (v3):
    0 = adult_violence  (色情暴力)
    1 = commercial      (博彩营销)
    2 = phishing        (钓鱼邮件)
    3 = finance         (发票财务)
    4 = academic        (学术推广)

Usage:
    python src/export_labels.py                          # default: BERT results → CSV+JSONL
    python src/export_labels.py --source rule            # rule-only results
    python src/export_labels.py --source fused           # fused results
    python src/export_labels.py --input custom.json      # custom strategy file
    python src/export_labels.py --format csv             # CSV only
"""

import json
import sys
from collections import Counter
from pathlib import Path

LABEL2ID = {'adult_violence': 0, 'commercial': 1, 'phishing': 2, 'finance': 3, 'academic': 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def export(input_path: str, output_stem: str, source: str = 'bert',
           formats: tuple = ('csv', 'jsonl')):
    """Export labels from strategy comparison JSON."""
    input_path = Path(input_path)

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Select results by source
    if source == 'rule':
        results = data['results']['rule']
    elif source == 'fused':
        results = data['results']['fused']
    else:
        results = data['results']['bert']

    # Write JSONL
    if 'jsonl' in formats:
        jl_path = Path(output_stem + '.jsonl')
        with jl_path.open('w', encoding='utf-8') as f:
            for r in results:
                label = LABEL2ID.get(r['category'], -1)
                f.write(json.dumps({
                    'record_id': r['record_id'],
                    'spam_type': label,
                }, ensure_ascii=False) + '\n')
        print(f"  {jl_path}")

    # Write CSV
    if 'csv' in formats:
        csv_path = Path(output_stem + '.csv')
        with csv_path.open('w', encoding='utf-8') as f:
            f.write('record_id,spam_type\n')
            for r in results:
                label = LABEL2ID.get(r['category'], -1)
                f.write(f"{r['record_id']},{label}\n")
        print(f"  {csv_path}")

    # Stats
    cnt = Counter(LABEL2ID.get(r['category'], -1) for r in results)
    print(f"\n  Total: {len(results):,}")
    print(f"  Source: {source}")
    print(f"\n  Distribution:")
    for idx, cat in sorted(ID2LABEL.items()):
        print(f"    {idx}  {cat:12s}  {cnt.get(idx, 0):>6,}")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Export email_id + spam_type')
    p.add_argument('--input', default='data/strategy_comparison.json',
                   help='Strategy comparison JSON file')
    p.add_argument('--source', choices=['bert', 'rule', 'fused'], default='bert',
                   help='Which strategy results to use (default: bert)')
    p.add_argument('--output', default='data/email_labels',
                   help='Output file stem (without extension)')
    p.add_argument('--format', choices=['csv', 'jsonl', 'both'], default='both',
                   help='Output format (default: both)')
    args = p.parse_args()

    formats = ('csv', 'jsonl') if args.format == 'both' else (args.format,)
    export(args.input, args.output, args.source, formats)


if __name__ == '__main__':
    main()
