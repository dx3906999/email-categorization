"""
Email Categorization System — 邮件分类系统
==========================================
Database layer: SQLite storage with FTS5 full-text search,
category labeling, and aggregation queries.

Usage:
    from src.db import EmailDB

    db = EmailDB()                     # defaults to database/spam_emails.db
    db.create_tables()
    db.insert_from_log("database/spam_email_data.log")

    rows = db.search("phishing", limit=20)
    db.set_category("record_id", "phishing")
    stats = db.category_distribution()
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Allow importing parser from sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parser import EmailRecord, parse_file_iter

# Default paths (relative to project root)
DEFAULT_DB_PATH = "database/spam_emails.db"
DEFAULT_LOG_PATH = "database/spam_email_data.log"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Column order matches the 26 JSON fields in the log (plus derived record_id + raw_json).
DDL_EMAILS = """
CREATE TABLE IF NOT EXISTS emails (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   TEXT    UNIQUE NOT NULL,
    mid         TEXT,
    tid         TEXT,
    timestamp   TEXT,
    from_addr   TEXT,
    from_name   TEXT,
    sender      TEXT,
    recipient   TEXT,
    subject     TEXT,
    content     TEXT,
    doccontent  TEXT,
    htmlurl     TEXT,
    html_tag    TEXT,
    urls        TEXT,
    attach      TEXT,
    size        INTEGER,
    text_size   INTEGER,
    auth_user   TEXT,
    ip          TEXT,
    region_ip   TEXT,
    region      TEXT,
    recv_ip_list TEXT,
    domain_rep  TEXT,
    wlist_cnt   INTEGER,
    dwlist_cnt  INTEGER,
    license_id  TEXT,
    xmailer     TEXT,
    category    TEXT    DEFAULT '',   -- classification label
    raw_json    TEXT
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_emails_timestamp ON emails(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_emails_mid       ON emails(mid);",
    "CREATE INDEX IF NOT EXISTS idx_emails_sender    ON emails(sender);",
    "CREATE INDEX IF NOT EXISTS idx_emails_from_addr ON emails(from_addr);",
    "CREATE INDEX IF NOT EXISTS idx_emails_license   ON emails(license_id);",
    "CREATE INDEX IF NOT EXISTS idx_emails_region    ON emails(region);",
    "CREATE INDEX IF NOT EXISTS idx_emails_ip        ON emails(ip);",
    "CREATE INDEX IF NOT EXISTS idx_emails_rcpt      ON emails(recipient);",
    "CREATE INDEX IF NOT EXISTS idx_emails_category  ON emails(category);",
]

# FTS covers the main text-searchable fields
DDL_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS emails_fts USING fts5(
    subject,
    content,
    doccontent,
    sender,
    from_addr,
    recipient,
    content=emails,
    content_rowid=id
);
"""

DDL_FTS_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS emails_ai AFTER INSERT ON emails BEGIN
        INSERT INTO emails_fts(rowid, subject, content, doccontent, sender, from_addr, recipient)
        VALUES (new.id, new.subject, new.content, new.doccontent, new.sender, new.from_addr, new.recipient);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS emails_ad AFTER DELETE ON emails BEGIN
        INSERT INTO emails_fts(emails_fts, rowid, subject, content, doccontent, sender, from_addr, recipient)
        VALUES ('delete', old.id, old.subject, old.content, old.doccontent, old.sender, old.from_addr, old.recipient);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS emails_au AFTER UPDATE ON emails BEGIN
        INSERT INTO emails_fts(emails_fts, rowid, subject, content, doccontent, sender, from_addr, recipient)
        VALUES ('delete', old.id, old.subject, old.content, old.doccontent, old.sender, old.from_addr, old.recipient);
        INSERT INTO emails_fts(rowid, subject, content, doccontent, sender, from_addr, recipient)
        VALUES (new.id, new.subject, new.content, new.doccontent, new.sender, new.from_addr, new.recipient);
    END;
    """,
]

INSERT_SQL = """
INSERT OR IGNORE INTO emails (
    record_id, mid, tid, timestamp,
    from_addr, from_name, sender, recipient,
    subject, content, doccontent, htmlurl, html_tag, urls, attach,
    size, text_size, auth_user,
    ip, region_ip, region, recv_ip_list,
    domain_rep, wlist_cnt, dwlist_cnt, license_id, xmailer,
    category, raw_json
) VALUES (
    :record_id, :mid, :tid, :timestamp,
    :from_addr, :from_name, :sender, :recipient,
    :subject, :content, :doccontent, :htmlurl, :html_tag, :urls, :attach,
    :size, :text_size, :auth_user,
    :ip, :region_ip, :region, :recv_ip_list,
    :domain_rep, :wlist_cnt, :dwlist_cnt, :license_id, :xmailer,
    :category, :raw_json
);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class EmailDB:
    """Manages the SQLite database of spam-email records."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ---- connection management ----

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ---- schema ----

    def create_tables(self) -> None:
        """Create tables, indexes, and FTS support."""
        c = self.conn
        c.execute(DDL_EMAILS)
        for sql in DDL_INDEXES:
            c.execute(sql)
        c.execute(DDL_FTS)
        for sql in DDL_FTS_TRIGGERS:
            c.execute(sql)
        c.commit()

    # ---- data loading ----

    def insert_records(self, records: list[EmailRecord], batch_size: int = 1000) -> int:
        """Insert a list of EmailRecord objects. Returns count of newly inserted rows."""
        rows = [_record_to_row(r) for r in records]
        inserted = 0
        c = self.conn
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            cur = c.executemany(INSERT_SQL, batch)
            inserted += cur.rowcount
        c.commit()
        return inserted

    def insert_from_log(self, log_path: str | Path) -> int:
        """Parse a log file and insert all records in one pass."""
        total = 0
        batch: list[dict] = []
        c = self.conn

        for rec in parse_file_iter(log_path):
            batch.append(_record_to_row(rec))
            if len(batch) >= 1000:
                cur = c.executemany(INSERT_SQL, batch)
                total += cur.rowcount
                c.commit()
                batch.clear()
        # flush remaining
        if batch:
            cur = c.executemany(INSERT_SQL, batch)
            total += cur.rowcount
            c.commit()

        return total

    # ---- search ----

    def search(self, keyword: str, limit: int = 50, offset: int = 0) -> list[sqlite3.Row]:
        """Full-text search across subject, content, sender, recipient."""
        # FTS5 query — escape special chars except *
        safe = keyword.replace('"', '""')
        rows = self.conn.execute("""
            SELECT e.*
            FROM emails_fts f
            JOIN emails e ON e.id = f.rowid
            WHERE emails_fts MATCH ?
            ORDER BY rank
            LIMIT ? OFFSET ?
        """, (f'"{safe}"', limit, offset)).fetchall()
        return rows

    def search_like(self, keyword: str,
                    fields: tuple[str, ...] = ('subject', 'content', 'doccontent', 'sender', 'from_addr', 'recipient'),
                    limit: int = 50) -> list[sqlite3.Row]:
        """LIKE-based search (fallback when FTS is overkill)."""
        like = f'%{keyword}%'
        clauses = ' OR '.join(f'{f} LIKE ?' for f in fields)
        params = [like] * len(fields) + [limit]
        return self.conn.execute(
            f"SELECT * FROM emails WHERE {clauses} LIMIT ?", params
        ).fetchall()

    def search_by_field(self, field: str, value: str, limit: int = 100) -> list[sqlite3.Row]:
        """Exact or LIKE search on a specific column."""
        # defend against injection on field name
        allowed = {
            'sender', 'from_addr', 'from_name', 'recipient', 'license_id',
            'region', 'ip', 'subject', 'record_id', 'mid', 'auth_user',
            'doccontent', 'content',
        }
        if field not in allowed:
            raise ValueError(f"Field not allowed: {field}")
        return self.conn.execute(
            f"SELECT * FROM emails WHERE {field} LIKE ? LIMIT ?",
            (f'%{value}%', limit)
        ).fetchall()

    # ---- stats queries ----

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]

    def time_range(self) -> tuple[str, str]:
        row = self.conn.execute(
            "SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi FROM emails"
        ).fetchone()
        return (row['lo'], row['hi'])

    def top(self, column: str, limit: int = 30) -> list[sqlite3.Row]:
        """Top-N values for a given column."""
        allowed = {
            'sender', 'from_addr', 'from_name', 'recipient', 'license_id',
            'region', 'ip', 'region_ip', 'auth_user',
        }
        if column not in allowed:
            raise ValueError(f"Column not allowed: {column}")
        return self.conn.execute(
            f"SELECT {column}, COUNT(*) AS cnt FROM emails "
            f"WHERE {column} != '' GROUP BY {column} "
            f"ORDER BY cnt DESC LIMIT ?", (limit,)
        ).fetchall()

    def monthly_distribution(self) -> list[sqlite3.Row]:
        return self.conn.execute("""
            SELECT substr(timestamp, 1, 7) AS ym, COUNT(*) AS cnt
            FROM emails
            WHERE timestamp IS NOT NULL
            GROUP BY ym ORDER BY ym
        """).fetchall()

    def report(self) -> dict:
        """Return a dict suitable for pretty-printing or JSON export."""
        total = self.count()
        tr = self.time_range()
        size_row = self.conn.execute(
            "SELECT SUM(size) AS total_size, AVG(size) AS avg_size FROM emails"
        ).fetchone()

        return {
            'total': total,
            'time_range': tr,
            'total_size_kb': round((size_row['total_size'] or 0) / 1024, 1),
            'avg_size_bytes': round(size_row['avg_size'] or 0, 1),
            'top_senders': self.top('sender', 30),
            'top_recipients': self.top('recipient', 30),
            'top_regions': self.top('region', 20),
            'top_ips': self.top('ip', 20),
            'top_licenses': self.top('license_id', 30),
            'monthly': self.monthly_distribution(),
        }

    # ---- category management ----

    def set_category(self, record_id: str, category: str) -> None:
        """Label a single email by record_id."""
        self.conn.execute(
            "UPDATE emails SET category = ? WHERE record_id = ?",
            (category, record_id)
        )
        self.conn.commit()

    def set_category_bulk(self, updates: dict[str, str]) -> int:
        """Batch-update categories. `updates` maps record_id → category."""
        c = self.conn
        count = 0
        for rid, cat in updates.items():
            cur = c.execute("UPDATE emails SET category = ? WHERE record_id = ?", (cat, rid))
            count += cur.rowcount
        c.commit()
        return count

    def get_by_category(self, category: str, limit: int = 100) -> list[sqlite3.Row]:
        """Return records with a given category label."""
        return self.conn.execute(
            "SELECT * FROM emails WHERE category = ? LIMIT ?",
            (category, limit)
        ).fetchall()

    def unclassified(self, limit: int = 100) -> list[sqlite3.Row]:
        """Return records that have no category assigned yet."""
        return self.conn.execute(
            "SELECT * FROM emails WHERE category = '' OR category IS NULL LIMIT ?",
            (limit,)
        ).fetchall()

    def category_distribution(self) -> list[sqlite3.Row]:
        """Count records per category."""
        return self.conn.execute("""
            SELECT category, COUNT(*) AS cnt
            FROM emails
            GROUP BY category
            ORDER BY cnt DESC
        """).fetchall()

    # ---- raw SQL access ----

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Run an arbitrary SELECT query. Use with caution."""
        return self.conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_to_row(r: EmailRecord) -> dict:
    return {
        'record_id':   r.record_id,
        'mid':         r.mid,
        'tid':         r.tid,
        'timestamp':   r.timestamp,
        'from_addr':   r.from_addr,
        'from_name':   r.from_name,
        'sender':      r.sender,
        'recipient':   r.recipient,
        'subject':     r.subject,
        'content':     r.content,
        'doccontent':  r.doccontent,
        'htmlurl':     r.htmlurl,
        'html_tag':    r.html_tag,
        'urls':        r.urls,
        'attach':      r.attach,
        'size':        r.size,
        'text_size':   r.text_size,
        'auth_user':   r.auth_user,
        'ip':          r.ip,
        'region_ip':   r.region_ip,
        'region':      r.region,
        'recv_ip_list': r.recv_ip_list,
        'domain_rep':  r.domain_rep,
        'wlist_cnt':   r.wlist_cnt,
        'dwlist_cnt':  r.dwlist_cnt,
        'license_id':  r.license_id,
        'xmailer':     r.xmailer,
        'category':    r.category,
        'raw_json':    r.raw and json_dumps_compact(r.raw),
    }


def json_dumps_compact(obj: dict) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description='SQLite 存储管理')
    sub = p.add_subparsers(dest='cmd')

    # --- import ---
    imp = sub.add_parser('import', help='导入日志到数据库')
    imp.add_argument('log', nargs='?', default=DEFAULT_LOG_PATH,
                     help='日志文件路径')
    imp.add_argument('--db', default=DEFAULT_DB_PATH, help='SQLite 数据库路径')
    imp.add_argument('--force', action='store_true', help='重建表（清空已有数据）')

    # --- search ---
    sea = sub.add_parser('search', help='全文搜索邮件')
    sea.add_argument('keyword', help='搜索关键词')
    sea.add_argument('--db', default=DEFAULT_DB_PATH, help='SQLite 数据库路径')
    sea.add_argument('--limit', type=int, default=20)
    sea.add_argument('--like', action='store_true', help='使用 LIKE 而非 FTS')

    # --- report ---
    rep = sub.add_parser('report', help='打印汇总报告')
    rep.add_argument('--db', default=DEFAULT_DB_PATH, help='SQLite 数据库路径')

    # --- top ---
    top = sub.add_parser('top', help='按字段统计排名')
    top.add_argument('field', choices=['sender', 'from_addr', 'from_name', 'recipient', 'region', 'ip', 'license_id'])
    top.add_argument('--db', default=DEFAULT_DB_PATH, help='SQLite 数据库路径')
    top.add_argument('--limit', type=int, default=30)

    # --- categories ---
    cat = sub.add_parser('categories', help='查看分类分布')
    cat.add_argument('--db', default=DEFAULT_DB_PATH, help='SQLite 数据库路径')

    # --- label ---
    lab = sub.add_parser('label', help='给邮件设置分类标签')
    lab.add_argument('record_id', help='记录 ID')
    lab.add_argument('category', help='分类标签 (如 spam/phishing/promotional/scam/normal)')
    lab.add_argument('--db', default=DEFAULT_DB_PATH, help='SQLite 数据库路径')

    args = p.parse_args()

    if args.cmd == 'import':
        with EmailDB(args.db) as db:
            if args.force:
                db.conn.execute("DROP TABLE IF EXISTS emails;")
                db.conn.execute("DROP TABLE IF EXISTS emails_fts;")
            db.create_tables()
            print(f"正在解析并导入 {args.log} ...")
            n = db.insert_from_log(args.log)
            print(f"导入完成，数据库现有 {db.count()} 条记录")

    elif args.cmd == 'search':
        with EmailDB(args.db) as db:
            if args.like:
                rows = db.search_like(args.keyword, limit=args.limit)
            else:
                rows = db.search(args.keyword, limit=args.limit)
            print(f"搜索 '{args.keyword}': {len(rows)} 条结果\n")
            for i, r in enumerate(rows):
                print(f"--- {i+1} ---")
                print(f"  时间:     {r['timestamp']}")
                print(f"  发件人:   {r['sender']}")
                print(f"  主题:     {(r['subject'] or '')[:120]}")
                print(f"  收件人:   {r['recipient']}")
                print(f"  IP:       {r['ip']}  ({r['region']})")
                print(f"  单位:     {r['license_id']}")

    elif args.cmd == 'report':
        with EmailDB(args.db) as db:
            rep_data = db.report()
            print(f"数据库: {args.db}")
            print(f"总记录: {rep_data['total']:,}")
            print(f"时间范围: {rep_data['time_range'][0]} ~ {rep_data['time_range'][1]}")
            print(f"总大小: {rep_data['total_size_kb']:,} KB")
            print(f"平均大小: {rep_data['avg_size_bytes']:,.1f} bytes")
            print("\n每月分布:")
            for r in rep_data['monthly']:
                print(f"  {r['ym']}: {r['cnt']:,}")
            print("\nTop 发件人:")
            for r in rep_data['top_senders'][:15]:
                print(f"  {r['cnt']:>5}  {r['sender']}")
            print("\nTop 地区:")
            for r in rep_data['top_regions'][:15]:
                print(f"  {r['cnt']:>5}  {r['region']}")
            print("\nTop 单位:")
            for r in rep_data['top_licenses'][:15]:
                print(f"  {r['cnt']:>5}  {r['license_id']}")

    elif args.cmd == 'top':
        with EmailDB(args.db) as db:
            rows = db.top(args.field, args.limit)
            print(f"\n{args.field} 排名 (Top {args.limit}):")
            print("-" * 50)
            for r in rows:
                print(f"  {r['cnt']:>6}  {r[args.field]}")

    elif args.cmd == 'categories':
        with EmailDB(args.db) as db:
            rows = db.category_distribution()
            total = db.count()
            print(f"\n分类分布 ({total} 条):")
            print("-" * 50)
            for r in rows:
                label = r['category'] if r['category'] else '(未分类)'
                pct = r['cnt'] / total * 100
                bar = '█' * max(1, int(pct / 2))
                print(f"  {r['cnt']:>6}  ({pct:5.1f}%)  {bar}  {label}")

    elif args.cmd == 'label':
        with EmailDB(args.db) as db:
            db.set_category(args.record_id, args.category)
            print(f"已标记: {args.record_id} → '{args.category}'")

    else:
        p.print_help()


if __name__ == '__main__':
    main()
