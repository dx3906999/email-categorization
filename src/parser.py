"""
Email Categorization System — 邮件分类系统
==========================================
Log parser: reads spam_email_data.log (TSV+JSON) into structured EmailRecord objects.

Log format (TSV):
    <record_id><tab><json_payload>

Each JSON payload contains 26 fields of metadata for one intercepted email,
including sender, recipient, subject, content, IPs, URLs, reputation scores, etc.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Default paths (relative to project root)
DEFAULT_LOG_PATH = "database/spam_email_data.log"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# All 26 JSON fields in the log (for reference):
# @timestamp, attach, authuser, content, doccontent, domainrep,
# dwlistcnt, from, fromname, htmltag, htmlurl, ip, licenseid, mid,
# rcpt, recviplist, region, regionip, sender, size, subject,
# textsize, tid, url, wlistcnt, xmailer

@dataclass
class EmailRecord:
    """Parsed fields from one spam-email log line (all 26 JSON fields + derived)."""

    record_id: str
    raw: dict  # original JSON payload, unmodified

    # ---- core identity fields ----
    mid: str = ""                 # message-id
    tid: str = ""                 # tracking-id (gateway internal)

    # ---- addressing ----
    from_addr: str = ""           # full From header: "Name" <email>
    from_name: str = ""           # display-name part
    sender: str = ""              # envelope sender email
    recipient: str = ""           # rcpt field (semicolon-delimited)

    # ---- content ----
    subject: str = ""
    content: str = ""
    doccontent: str = ""          # extracted document text
    attach: str = ""              # attachment info
    htmlurl: str = ""             # HTML rendering URL
    html_tag: str = ""            # extracted HTML tag sequence
    urls: str = ""                # extracted URLs (space-delimited)

    # ---- metadata ----
    timestamp: Optional[str] = None
    size: int = 0
    text_size: int = 0
    auth_user: str = ""

    # ---- network / geo ----
    ip: str = ""
    region_ip: str = ""
    region: str = ""
    recv_ip_list: str = ""

    # ---- reputation / scoring ----
    domain_rep: str = ""          # domain reputation scores
    wlist_cnt: int = 0
    dwlist_cnt: int = 0

    # ---- categorization ----
    license_id: str = ""          # org / license key
    xmailer: str = ""
    category: str = ""            # classified category (set by classifier)

    # ---- derived fields ----
    parsed_urls: list[str] = field(default_factory=list)
    parsed_ips: list[str] = field(default_factory=list)
    parsed_rcpts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _extract_urls(raw: str) -> list[str]:
    """Extract URLs from a space-delimited string."""
    if not raw:
        return []
    url_pattern = re.compile(r'https?://[^\s]*|[a-zA-Z0-9][^\s]*\.[a-zA-Z]{2,}[^\s]*')
    return url_pattern.findall(raw)


def _extract_ips(raw: str) -> list[str]:
    """Extract IP addresses from a string."""
    if not raw:
        return []
    ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    return ip_pattern.findall(raw)


# ---------------------------------------------------------------------------
# Core parse functions
# ---------------------------------------------------------------------------

def parse_line(line: str) -> Optional[EmailRecord]:
    """Parse a single TSV line into an EmailRecord.

    Returns None when the line is malformed or empty.
    """
    line = line.strip()
    if not line:
        return None

    try:
        record_id, json_str = line.split('\t', 1)
    except ValueError:
        print(f"[WARN] skipping malformed line (no tab separator): {line[:80]}...", file=sys.stderr)
        return None

    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError as exc:
        print(f"[WARN] {record_id}: invalid JSON — {exc}", file=sys.stderr)
        return None

    record = EmailRecord(
        record_id=record_id,
        raw=raw,
        # core identity
        mid=raw.get('mid', ''),
        tid=raw.get('tid', ''),
        # addressing
        from_addr=raw.get('from', ''),
        from_name=raw.get('fromname', ''),
        sender=raw.get('sender', ''),
        recipient=raw.get('rcpt', ''),
        # content
        subject=raw.get('subject', ''),
        content=raw.get('content', ''),
        doccontent=raw.get('doccontent', ''),
        attach=raw.get('attach', ''),
        htmlurl=raw.get('htmlurl', ''),
        html_tag=raw.get('htmltag', ''),
        urls=raw.get('url', ''),
        # metadata
        timestamp=raw.get('@timestamp'),
        size=_safe_int(raw.get('size')),
        text_size=_safe_int(raw.get('textsize')),
        auth_user=raw.get('authuser', ''),
        # network / geo
        ip=raw.get('ip', ''),
        region_ip=raw.get('regionip', ''),
        region=raw.get('region', ''),
        recv_ip_list=raw.get('recviplist', ''),
        # reputation
        domain_rep=raw.get('domainrep', ''),
        wlist_cnt=_safe_int(raw.get('wlistcnt')),
        dwlist_cnt=_safe_int(raw.get('dwlistcnt')),
        # categorization
        license_id=raw.get('licenseid', ''),
        xmailer=raw.get('xmailer', ''),
    )

    # derived fields
    record.parsed_urls = _extract_urls(record.urls)
    record.parsed_ips = _extract_ips(record.urls + ' ' + record.ip)
    record.parsed_rcpts = [r.strip() for r in record.recipient.split(';') if r.strip()]

    return record


def parse_file(path: str | Path) -> list[EmailRecord]:
    """Parse the entire log file and return a list of EmailRecord objects."""
    records: list[EmailRecord] = []
    path = Path(path)
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            rec = parse_line(line)
            if rec is not None:
                records.append(rec)
    return records


def parse_file_iter(path: str | Path):
    """Generator yielding EmailRecord objects — memory-friendly for large files."""
    path = Path(path)
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            rec = parse_line(line)
            if rec is not None:
                yield rec


# ---------------------------------------------------------------------------
# Analysis / statistics
# ---------------------------------------------------------------------------

def compute_stats(records: list[EmailRecord]) -> dict:
    """Compute aggregate statistics over parsed records."""
    total = len(records)
    if total == 0:
        return {'total': 0}

    # -- distributions --
    regions: Counter[str] = Counter()
    senders: Counter[str] = Counter()
    recipients: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    url_domains: Counter[str] = Counter()
    license_ids: Counter[str] = Counter()
    year_months: Counter[str] = Counter()
    ip_list: Counter[str] = Counter()

    total_size = 0
    content_count = 0
    attach_count = 0
    wlist_total = 0
    dwlist_total = 0

    for r in records:
        if r.region.strip():
            regions[r.region.strip()] += 1
        if r.sender.strip():
            senders[r.sender.strip().lower()] += 1
        for rcpt in r.parsed_rcpts:
            recipients[rcpt.lower()] += 1
        if r.license_id.strip():
            license_ids[r.license_id.strip()] += 1
        if r.ip.strip():
            ip_list[r.ip.strip()] += 1
        if r.timestamp:
            try:
                ym = r.timestamp[:7]  # "2019-10"
                year_months[ym] += 1
            except Exception:
                pass
        for u in r.parsed_urls:
            try:
                dom = re.sub(r'^(https?://)?([^/:?#]+).*', r'\2', u)
                url_domains[dom] += 1
            except Exception:
                pass
        total_size += r.size
        if r.content.strip():
            content_count += 1
        if r.attach.strip():
            attach_count += 1
        wlist_total += r.wlist_cnt
        dwlist_total += r.dwlist_cnt

    # parse time range
    timestamps = [r.timestamp for r in records if r.timestamp]
    time_range = (min(timestamps), max(timestamps)) if timestamps else (None, None)

    return {
        'total': total,
        'time_range': time_range,
        'total_size_kb': round(total_size / 1024, 1),
        'avg_size_bytes': round(total_size / total, 1),
        'with_content': content_count,
        'with_attach': attach_count,
        'avg_wlist_cnt': round(wlist_total / total, 4),
        'avg_dwlist_cnt': round(dwlist_total / total, 4),
        'regions': regions.most_common(20),
        'senders': senders.most_common(50),
        'recipients': recipients.most_common(50),
        'license_ids': license_ids.most_common(30),
        'top_ips': ip_list.most_common(20),
        'top_url_domains': url_domains.most_common(20),
        'year_month_dist': sorted(year_months.items()),
    }


def print_report(records: list[EmailRecord]) -> None:
    """Print a human-readable summary report to stdout."""
    stats = compute_stats(records)
    print("=" * 70)
    print("  SPAM EMAIL DATA — 解析报告")
    print("=" * 70)
    print(f"  总记录数:         {stats['total']:,}")
    print(f"  时间范围:         {stats['time_range'][0]}  ~  {stats['time_range'][1]}")
    print(f"  总大小:           {stats['total_size_kb']:,.1f} KB")
    print(f"  平均邮件大小:     {stats['avg_size_bytes']:,.1f} bytes")
    print(f"  有正文的邮件:     {stats['with_content']:,} ({stats['with_content']/stats['total']*100:.1f}%)")
    print(f"  有附件的邮件:     {stats['with_attach']:,} ({stats['with_attach']/stats['total']*100:.1f}%)")
    print(f"  平均白名单计数:   {stats['avg_wlist_cnt']:.3f}")
    print(f"  平均动态白名单数: {stats['avg_dwlist_cnt']:.3f}")

    print("\n" + "-" * 70)
    print("  每月分布")
    print("-" * 70)
    for ym, cnt in stats['year_month_dist']:
        bar = '█' * max(1, cnt // (stats['total'] // 80 or 1))
        print(f"  {ym}:  {cnt:>6,}  {bar}")

    print("\n" + "-" * 70)
    print("  地区分布 (Top 20)")
    print("-" * 70)
    for region, cnt in stats['regions']:
        print(f"  {cnt:>6,}  {region}")

    print("\n" + "-" * 70)
    print("  发送域名 (Top 30)")
    print("-" * 70)
    for sender, cnt in stats['senders'][:30]:
        print(f"  {cnt:>6,}  {sender}")

    print("\n" + "-" * 70)
    print("  收件人 (Top 30)")
    print("-" * 70)
    for rcpt, cnt in stats['recipients'][:30]:
        print(f"  {cnt:>6,}  {rcpt}")

    print("\n" + "-" * 70)
    print("  许可证ID / 所属单位 (Top 30)")
    print("-" * 70)
    for lid, cnt in stats['license_ids']:
        print(f"  {cnt:>6,}  {lid}")

    print("\n" + "-" * 70)
    print("  URL 域名 (Top 20)")
    print("-" * 70)
    for dom, cnt in stats['top_url_domains']:
        print(f"  {cnt:>6,}  {dom}")

    print("\n" + "-" * 70)
    print("  发送 IP (Top 20)")
    print("-" * 70)
    for ip, cnt in stats['top_ips']:
        print(f"  {cnt:>6,}  {ip}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Filter / search helpers
# ---------------------------------------------------------------------------

def filter_by_keyword(records: list[EmailRecord], keyword: str,
                      fields: tuple[str, ...] = ('subject', 'content', 'sender', 'recipient'),
                      case_sensitive: bool = False) -> list[EmailRecord]:
    """Return records where *keyword* appears in one of the given fields."""
    results: list[EmailRecord] = []
    kw = keyword if case_sensitive else keyword.lower()
    for r in records:
        for fname in fields:
            text = getattr(r, fname, '')
            if not case_sensitive:
                text = text.lower()
            if kw in text:
                results.append(r)
                break
    return results


def filter_by_date_range(records: list[EmailRecord],
                         start: str, end: str) -> list[EmailRecord]:
    """Return records within [start, end] timestamps (ISO format)."""
    return [r for r in records
            if r.timestamp and start <= r.timestamp <= end]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(records: list[EmailRecord], dest: str | Path,
               fields: Optional[list[str]] = None) -> Path:
    """Export parsed records to CSV."""
    import csv
    if fields is None:
        fields = ['record_id', 'mid', 'tid', 'timestamp',
                  'from_addr', 'from_name', 'sender', 'recipient',
                  'subject', 'content', 'doccontent', 'htmlurl',
                  'ip', 'region_ip', 'region', 'recv_ip_list',
                  'size', 'text_size', 'attach', 'urls',
                  'domain_rep', 'html_tag', 'license_id', 'auth_user',
                  'wlist_cnt', 'dwlist_cnt', 'xmailer']

    dest = Path(dest)
    with dest.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for r in records:
            writer.writerow([getattr(r, f, '') for f in fields])
    return dest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse the log file and print a summary report.

    Usage:
        python src/parser.py                        # parse default log file
        python src/parser.py --csv output.csv       # export to CSV
        python src/parser.py --search <keyword>     # search emails
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='解析 spam_email_data.log 垃圾邮件日志')
    parser.add_argument('--file', '-f', default=DEFAULT_LOG_PATH,
                        help='日志文件路径 (默认: spam_email_data.log)')
    parser.add_argument('--csv', metavar='PATH',
                        help='导出 CSV 到指定路径')
    parser.add_argument('--search', '-s', metavar='KEYWORD',
                        help='按关键词搜索邮件')
    parser.add_argument('--fields', '-F', nargs='*',
                        default=['subject', 'content', 'sender', 'recipient'],
                        help='搜索的字段 (默认: subject content sender recipient)')
    parser.add_argument('--stats-only', action='store_true',
                        help='仅显示统计摘要，不搜索')

    args = parser.parse_args()

    log_path = Path(args.file)
    if not log_path.exists():
        print(f"[ERROR] 文件不存在: {log_path}", file=sys.stderr)
        sys.exit(1)

    print(f"正在解析 {log_path} ...")
    records = parse_file(log_path)
    print(f"解析完成: {len(records):,} 条记录\n")

    # Search mode
    if args.search:
        matched = filter_by_keyword(records, args.search,
                                    fields=tuple(args.fields))
        print(f"搜索 '{args.search}': 匹配 {len(matched)} 条")
        for i, r in enumerate(matched[:50]):
            print(f"\n--- 结果 {i+1} ---")
            print(f"  时间:     {r.timestamp}")
            print(f"  发件人:   {r.sender}")
            print(f"  主题:     {r.subject[:120]}")
            print(f"  收件人:   {r.recipient}")
            print(f"  IP:       {r.ip}  ({r.region})")
            print(f"  大小:     {r.size} bytes")
            if r.parsed_urls:
                print(f"  URLs:     {r.parsed_urls[:5]}")
        if len(matched) > 50:
            print(f"\n  ... 还有 {len(matched) - 50} 条结果未显示")
        return

    # CSV export mode
    if args.csv:
        dest = export_csv(records, args.csv)
        print(f"已导出: {dest}  ({len(records):,} 条)")
        return

    # Default: print summary
    print_report(records)


if __name__ == '__main__':
    main()
