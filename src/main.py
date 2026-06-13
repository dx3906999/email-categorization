"""
Email Categorization System — 邮件分类系统
==========================================
Main entry point: unified CLI for parse → store → classify workflow.

Usage:
    # Step 1: Parse log and store in SQLite
    python src/main.py import

    # Step 2: Classify emails with rule engine
    python src/main.py classify

    # Step 3: Query results
    python src/main.py search "phishing"
    python src/main.py report
    python src/main.py categories

    # Or run the full pipeline at once:
    python src/main.py pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_import() -> None:
    """Parse log → SQLite."""
    from db import EmailDB
    with EmailDB() as db:
        db.create_tables()
        n = db.insert_from_log("database/spam_email_data.log")
        print(f"导入完成: {n:,} 条 → {db.db_path}")


def cmd_classify() -> None:
    """Two-pass classification: rules → IP propagation, write back to DB."""
    from parser import parse_file
    from db import EmailDB
    from classifier import EmailClassifier, CATEGORY_LABELS

    print("正在从日志重新解析并分类...")
    records = parse_file("database/spam_email_data.log")
    total = len(records)

    clf = EmailClassifier(rule_threshold=0.2)
    clf.load_default_rules()
    total = sum(clf.rule_count().values())
    print(f"规则数: {total} 条  (来自 src/rules/*.json)")

    # Pass 1: rules
    print("\n[第一轮] 规则分类...")
    rule_results = clf.classify_all(records, update_record=True)
    n_rule = sum(1 for r in rule_results if r['category'])

    print("规则分类结果:")
    for cat in ['adult', 'gambling', 'marketing', 'phishing', 'fraud']:
        cnt = sum(1 for r in rule_results if r['category'] == cat)
        if cnt:
            label = CATEGORY_LABELS[cat]
            print(f"  {label:10s}  {cnt:>6,}  ({cnt/total*100:5.1f}%)")

    # Pass 2: IP propagation
    print("\n[第二轮] IP 传播分类...")
    ip_results = clf.propagate_by_ip(records, update_record=True)
    n_ip = len(ip_results)
    ip_stats = clf.ip_classifier.stats()
    print(f"  可靠 IP 数: {ip_stats['total_ips']}")
    for cat, ip_cnt in sorted(ip_stats['categories'].items(), key=lambda x: -x[1]):
        label = CATEGORY_LABELS.get(cat, cat)
        print(f"  {label:10s}  {ip_cnt:>6} 个IP")

    ip_summary = {}
    from collections import Counter
    ip_cnt = Counter(r['category'] for r in ip_results)
    for cat in ['adult', 'gambling', 'marketing', 'phishing', 'fraud']:
        c = ip_cnt.get(cat, 0)
        if c:
            ip_summary[cat] = c
            label = CATEGORY_LABELS[cat]
            print(f"  {label:10s}  +{c:>6,}  被传播")

    final_classified = n_rule + n_ip
    final_uncl = total - final_classified
    print(f"\n最终: 已分类 {final_classified:,} / {total:,}"
          f"  (覆盖率 {final_classified/total*100:.1f}%)"
          f"  |  未分类 {final_uncl:,}")

    # Write all categories back to DB
    db = EmailDB()
    updates = {r.record_id: r.category for r in records if r.category}
    n = db.set_category_bulk(updates)
    db.close()
    print(f"已更新 {n} 条标签到数据库")


def cmd_pipeline() -> None:
    """Full pipeline: import → classify (two-pass)."""
    print("=" * 60)
    print("  邮件分类系统 — 完整流水线")
    print("=" * 60)
    print("\n[1/2] 导入日志到数据库...")
    cmd_import()
    print("\n[2/2] 两轮分类 (规则 + IP)...")
    cmd_classify()
    print("\n完成!")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description='邮件分类系统 — Email Categorization System',
    )
    sub = p.add_subparsers(dest='cmd')

    sub.add_parser('import', help='解析日志并存入 SQLite')
    sub.add_parser('classify', help='运行分类引擎')
    sub.add_parser('pipeline', help='一键执行 import + classify')
    sub.add_parser('search', help='全文搜索 (见 python src/db.py search --help)')
    sub.add_parser('report', help='统计报告 (见 python src/db.py report --help)')
    sub.add_parser('categories', help='分类分布 (见 python src/db.py categories --help)')

    args = p.parse_args()

    if args.cmd == 'import':
        cmd_import()
    elif args.cmd == 'classify':
        cmd_classify()
    elif args.cmd == 'pipeline':
        cmd_pipeline()
    elif args.cmd == 'search':
        from db import main as db_main
        sys.argv = ['db.py', 'search']
        db_main()
    elif args.cmd == 'report':
        from db import main as db_main
        sys.argv = ['db.py', 'report']
        db_main()
    elif args.cmd == 'categories':
        from db import main as db_main
        sys.argv = ['db.py', 'categories']
        db_main()
    else:
        p.print_help()


if __name__ == '__main__':
    main()
