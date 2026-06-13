"""
Generate datasets for LLM annotation → BERT training.

Output:
    data/quality_verification.jsonl   520 records, stratified
        Used to verify LLM annotation quality against rule labels.

    data/train.jsonl                2,400 records, stratified
        LLM-labeled → BERT fine-tuning.

    data/holdout.jsonl         ~21,080 remaining records
        Unused for now; available for future self-training rounds.

Sampling strategy:
    - Verification: 520 records covering all 5 classes + unclassified
    - Training: 2,400 records, balanced-ish across classes
    - Holdout: everything else
"""

import json
import random
import sqlite3
from pathlib import Path

random.seed(42)

DB_PATH = Path(__file__).resolve().parent.parent / "database" / "spam_emails.db"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

CONTENT_MAX_LEN = 500
CATEGORIES = ['adult', 'gambling', 'marketing', 'phishing', 'fraud']
CATEGORY_LABELS = {
    'adult': '色情淫秽', 'gambling': '赌博博彩',
    'marketing': '营销推广', 'phishing': '钓鱼诈骗', 'fraud': '假发票/诈骗',
}

# --- quotas ---
VERIF_SIZE = 520
TRAIN_SIZE = 2400

VERIF_QUOTA = {
    'gambling': 27,
    'adult': 90, 'fraud': 85, 'marketing': 85, 'phishing': 85,
}
VERIF_QUOTA['unclassified'] = VERIF_SIZE - sum(VERIF_QUOTA.values())

TRAIN_QUOTA = {
    'gambling': 30,      # all remaining ~30
    'phishing': 400,
    'fraud': 550,
    'marketing': 500,
    'adult': 600,
    'unclassified': TRAIN_SIZE - (30 + 400 + 550 + 500 + 600),
}

def fetch_ids(conn, category: str, exclude_ids: set, limit: int = 0) -> list:
    """Get record ids for a category, excluding already-selected ones."""
    if category:
        sql = "SELECT record_id FROM emails WHERE category=? AND record_id NOT IN ({})".format(
            ','.join('?' * len(exclude_ids)) if exclude_ids else ''
        )
        params = [category] + list(exclude_ids)
    else:
        sql = "SELECT record_id FROM emails WHERE (category='' OR category IS NULL) AND record_id NOT IN ({})".format(
            ','.join('?' * len(exclude_ids)) if exclude_ids else ''
        )
        params = list(exclude_ids)
    sql += " ORDER BY RANDOM()"
    if limit:
        sql += f" LIMIT {limit}"
    return [r[0] for r in conn.execute(sql, params).fetchall()]


def fetch_records(conn, ids: list[str]) -> list[dict]:
    """Fetch full records by id list."""
    if not ids:
        return []
    placeholders = ','.join('?' * len(ids))
    rows = conn.execute(
        f"SELECT * FROM emails WHERE record_id IN ({placeholders})", ids
    ).fetchall()
    return [dict(r) for r in rows]


def make_jsonl_entry(row: dict) -> dict:
    """Convert a DB row into a compact annotation-ready dict."""
    content = (row.get('content') or '')[:CONTENT_MAX_LEN].replace('\r', '')
    subject = row.get('subject') or ''
    return {
        'record_id': row['record_id'],
        'subject': subject.strip(),
        'content': content.strip(),
        'rule_category': row.get('category') or '',
        'rule_confidence': 1.0 if row.get('category') else 0.0,  # simplified: rule labels have no real confidence yet
    }


# --- main ---

def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Count per category
    counts = {}
    for cat in CATEGORIES + ['']:
        if cat:
            cnt = conn.execute("SELECT COUNT(*) FROM emails WHERE category=?", (cat,)).fetchone()[0]
        else:
            cnt = conn.execute("SELECT COUNT(*) FROM emails WHERE category='' OR category IS NULL").fetchone()[0]
        counts[cat or 'unclassified'] = cnt

    print("Current distribution:")
    for cat, cnt in counts.items():
        print(f"  {cat:15s}  {cnt:>6,}")
    total = sum(counts.values())

    # ---- Step 1: Verification set ----
    selected_ids: set[str] = set()

    print(f"\n{'='*50}")
    print("  Verification set")
    print(f"{'='*50}")
    verif_records = _sample_split(conn, VERIF_QUOTA, selected_ids, counts)
    selected_ids.update(r['record_id'] for r in verif_records)
    random.shuffle(verif_records)
    _write_jsonl(OUT_DIR / "quality_verification.jsonl", verif_records)
    print(f"  → {len(verif_records)} records")

    # ---- Step 2: Training set ----
    print(f"\n{'='*50}")
    print("  Training set")
    print(f"{'='*50}")
    train_records = _sample_split(conn, TRAIN_QUOTA, selected_ids, counts)
    selected_ids.update(r['record_id'] for r in train_records)
    random.shuffle(train_records)
    _write_jsonl(OUT_DIR / "train.jsonl", train_records)
    print(f"  → {len(train_records)} records")

    # ---- Step 3: Holdout set (everything else) ----
    holdout_count = total - len(selected_ids)
    print(f"\n{'='*50}")
    print(f"  Holdout set: {holdout_count:,} remaining → data/holdout.jsonl")
    print(f"{'='*50}")
    _write_remaining(conn, OUT_DIR / "holdout.jsonl", selected_ids)

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"  data/quality_verification.jsonl  {len(verif_records):>6} 条")
    print(f"  data/train.jsonl                 {len(train_records):>6} 条")
    print(f"  data/holdout.jsonl               {holdout_count:>6} 条")
    print(f"  {'─' * 35}")
    print(f"  Total                            {total:>6} 条")
    print(f"{'='*50}")

    conn.close()


def _sample_split(conn, quotas: dict, exclude: set, counts: dict) -> list[dict]:
    """Sample records according to quotas, respecting available counts."""
    records = []
    for cat in CATEGORIES:
        n = min(quotas[cat], counts[cat] - sum(1 for rid in exclude if rid.startswith(cat)))
        if n <= 0:
            continue
        ids = _fetch_ids(conn, cat, exclude, limit=n)
        exclude.update(ids)
        rows = _fetch_records(conn, ids)
        records.extend([_make_entry(r) for r in rows])
        label = CATEGORY_LABELS[cat]
        print(f"  {cat:15s}  {len(ids):>4} / {counts[cat]:<5}  ({label})")

    # Unclassified
    uncl_n = quotas.get('unclassified', 0)
    uncl_avail = counts['unclassified'] - sum(1 for rid in exclude if rid.startswith('unclass'))
    uncl_n = min(uncl_n, uncl_avail)
    if uncl_n > 0:
        ids = _fetch_ids(conn, '', exclude, limit=uncl_n)
        exclude.update(ids)
        rows = _fetch_records(conn, ids)
        records.extend([_make_entry(r) for r in rows])
        print(f"  unclassified     {len(ids):>4} / {counts['unclassified']:<5}")
    return records


def _fetch_ids(conn, category: str, exclude_ids: set, limit: int = 0) -> list[str]:
    ex_list = list(exclude_ids)
    if category:
        if ex_list:
            placeholders = ','.join('?' * len(ex_list))
            sql = f"SELECT record_id FROM emails WHERE category=? AND record_id NOT IN ({placeholders}) ORDER BY RANDOM()"
            params = [category] + ex_list
        else:
            sql = "SELECT record_id FROM emails WHERE category=? ORDER BY RANDOM()"
            params = [category]
    else:
        if ex_list:
            placeholders = ','.join('?' * len(ex_list))
            sql = f"SELECT record_id FROM emails WHERE (category='' OR category IS NULL) AND record_id NOT IN ({placeholders}) ORDER BY RANDOM()"
            params = ex_list
        else:
            sql = "SELECT record_id FROM emails WHERE (category='' OR category IS NULL) ORDER BY RANDOM()"
            params = []
    if limit:
        sql += f" LIMIT {limit}"
    return [r[0] for r in conn.execute(sql, params).fetchall()]


def _fetch_records(conn, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    placeholders = ','.join('?' * len(ids))
    rows = conn.execute(f"SELECT * FROM emails WHERE record_id IN ({placeholders})", ids).fetchall()
    return [dict(r) for r in rows]


def _make_entry(row: dict) -> dict:
    content = (row.get('content') or '')[:CONTENT_MAX_LEN].replace('\r', '')
    subject = row.get('subject') or ''
    from_name = row.get('from_name') or ''
    return {
        'record_id': row['record_id'],
        'from_name': from_name.strip(),
        'subject': subject.strip(),
        'content': content.strip(),
        'rule_category': row.get('category') or '',
        'rule_confidence': 1.0 if row.get('category') else 0.0,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _write_remaining(conn, path: Path, exclude_ids: set) -> None:
    ex_list = list(exclude_ids)
    placeholders = ','.join('?' * len(ex_list))
    batch_size = 1000
    written = 0
    with path.open('w', encoding='utf-8') as f:
        offset = 0
        while True:
            batch = conn.execute(
                f"SELECT * FROM emails WHERE record_id NOT IN ({placeholders}) LIMIT {batch_size} OFFSET {offset}",
                ex_list
            ).fetchall()
            if not batch:
                break
            for row in batch:
                f.write(json.dumps(_make_entry(dict(row)), ensure_ascii=False) + '\n')
                written += 1
            offset += batch_size
    print(f"  → {written:,} records")


if __name__ == '__main__':
    main()
