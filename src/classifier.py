"""
Email Categorization System — 邮件分类系统
==========================================
Two-pass classifier:
  1. Rule-based  — regex rules with confidence scores (0.0–1.0)
  2. IP-based    — same-IP propagation (spammers reuse IPs)

All scores use 0.0–1.0 confidence range for compatibility with future ML models.

Categories (all emails in this dataset are spam):
    adult        — 色情淫秽：成人内容、色情网站、约炮交友
    gambling     — 赌博博彩：线上赌场、体育博彩、彩票投注
    marketing    — 营销推广：展会搭建、招标采购、广告营销、会议征稿
    phishing     — 钓鱼诈骗：账号验证、密码窃取、虚假通知、邮箱升级
    fraud        — 假发票/诈骗：代开发票、增值税发票、中奖诈骗、转账诈骗

Usage:
    from src.classifier import EmailClassifier

    clf = EmailClassifier()
    clf.load_default_rules()
    clf.load_rules_from_dir("src/rules/")

    records = parse_file("database/spam_email_data.log")
    results = clf.classify_all(records)         # rule-based
    propagated = clf.propagate_by_ip(records)   # IP-based (second pass)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parser import EmailRecord


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A single classification rule.

    field:      which EmailRecord attribute to match against
    pattern:    regex pattern (case-insensitive)
    category:   target category label
    confidence: confidence score 0.0–1.0 (was "weight" in v1)
    reason:     human-readable explanation
    """
    field: str
    pattern: str
    category: str
    confidence: float = 0.5
    reason: str = ""

    _compiled: re.Pattern[str] = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def match(self, record: EmailRecord) -> tuple[bool, str]:
        """Test rule against a record. Returns (matched, match_text)."""
        text = getattr(record, self.field, '') or ''
        m = self._compiled.search(text)
        if m:
            return True, m.group(0)
        return False, ""


# ---------------------------------------------------------------------------
# IP-based classifier
# ---------------------------------------------------------------------------

# Minimum votes for an IP to be considered a reliable signal
IP_MIN_SAMPLES = 3        # IP must appear at least this many times
IP_MIN_PURITY = 0.6       # at least 60% of emails from this IP must be same category
IP_CONFIDENCE = 0.5       # confidence assigned to IP-propagated labels


class IPClassifier:
    """Propagates labels via IP: if IP X sent 10 phishing emails,
    the 11th unclassified email from IP X is likely phishing too."""

    def __init__(self, min_samples: int = IP_MIN_SAMPLES,
                 min_purity: float = IP_MIN_PURITY,
                 confidence: float = IP_CONFIDENCE):
        self.min_samples = min_samples
        self.min_purity = min_purity
        self.confidence = confidence
        self.ip_map: dict[str, tuple[str, float, int]] = {}  # ip → (category, purity, count)

    def fit(self, records: list[EmailRecord]) -> None:
        """Build IP→category mapping from already-classified records."""
        # Collect: ip → list of categories
        ip_cats: dict[str, list[str]] = defaultdict(list)
        for r in records:
            if r.category and r.ip.strip():
                ip_cats[r.ip.strip()].append(r.category)

        for ip, cats in ip_cats.items():
            if len(cats) < self.min_samples:
                continue
            counter = Counter(cats)
            top_cat, top_cnt = counter.most_common(1)[0]
            purity = top_cnt / len(cats)
            if purity >= self.min_purity:
                self.ip_map[ip] = (top_cat, purity, len(cats))

    def predict(self, record: EmailRecord) -> tuple[str, float]:
        """Return (category, confidence) if IP is known, else ('', 0.0)."""
        ip = record.ip.strip()
        if ip in self.ip_map:
            cat, purity, count = self.ip_map[ip]
            # scale confidence: high-purity + high-count → higher confidence
            scaled = self.confidence * purity * min(1.0, count / 10)
            return cat, round(scaled, 3)
        return '', 0.0

    def propagate(self, records: list[EmailRecord],
                  update_record: bool = True) -> list[dict]:
        """Second pass: classify unclassified records using IP heuristics.

        Returns list of newly-classified records."""
        results = []
        for r in records:
            if r.category:
                continue  # already classified
            cat, conf = self.predict(r)
            if cat:
                if update_record:
                    r.category = cat
                results.append({
                    'record_id': r.record_id,
                    'category': cat,
                    'confidence': conf,
                    'source': 'ip',
                })
        return results

    def stats(self) -> dict:
        """Return summary statistics about the IP map."""
        return {
            'total_ips': len(self.ip_map),
            'categories': dict(Counter(c for c, _, _ in self.ip_map.values())),
        }


# ---------------------------------------------------------------------------
# Combined classifier
# ---------------------------------------------------------------------------

CATEGORIES = ['adult_violence', 'commercial', 'phishing', 'finance', 'academic']

CATEGORY_LABELS = {
    'adult_violence': '色情暴力',
    'commercial':     '博彩营销',
    'phishing':       '钓鱼邮件',
    'finance':        '发票财务',
    'academic':       '学术推广',
}


class EmailClassifier:
    """Three-pass email classifier: rules → BERT → IP propagation.

    Fusion strategy (BERT primary, rules as safety net):
        - BERT >= 0.8              → trust BERT
        - BERT == rule             → consensus, max confidence
        - BERT >= 0.5, disagrees   → trust BERT (rules can't do context)
        - BERT < 0.5, rule >= 0.7  → trust rule as fallback
        - otherwise                → unclassified
    """

    def __init__(self, rule_threshold: float = 0.2):
        self.rules: dict[str, list[Rule]] = {c: [] for c in CATEGORIES}
        self.threshold = rule_threshold
        self.ip_classifier = IPClassifier()
        self._bert = None  # lazy-loaded BertPredictor

    # ---- rule management ----

    def add_rule(self, rule: Rule) -> None:
        if rule.category not in self.rules:
            self.rules[rule.category] = []
        self.rules[rule.category].append(rule)

    def add_rules(self, rules: list[Rule]) -> None:
        for r in rules:
            self.add_rule(r)

    def load_default_rules(self) -> None:
        """Load built-in rules from src/rules/ directory."""
        rules_dir = Path(__file__).resolve().parent / 'rules'
        self.load_rules_from_dir(rules_dir)

    def load_rules_from_file(self, path: str | Path) -> int:
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        count = 0
        for item in data:
            self.add_rule(Rule(
                field=item['field'],
                pattern=item['pattern'],
                category=item['category'],
                confidence=item.get('confidence', item.get('weight', 0.5)),
                reason=item.get('reason', ''),
            ))
            count += 1
        return count

    def load_rules_from_dir(self, dir_path: str | Path) -> int:
        total = 0
        for f in sorted(Path(dir_path).glob('*.json')):
            total += self.load_rules_from_file(f)
        return total

    # ---- rule-based classification ----

    def classify(self, record: EmailRecord) -> tuple[str, float, list[dict]]:
        """Rule-based classification.

        Returns: (category, confidence, list_of_matched_rules)

        Confidence is sum of matched rule confidences, capped at 1.0.
        """
        scores: dict[str, float] = {c: 0.0 for c in self.rules}
        matches: list[dict] = []

        for category, rules in self.rules.items():
            for rule in rules:
                matched, text = rule.match(record)
                if matched:
                    scores[category] += rule.confidence
                    matches.append({
                        'category': category,
                        'field': rule.field,
                        'pattern': rule.pattern,
                        'matched': text[:200],
                        'reason': rule.reason,
                        'confidence': rule.confidence,
                    })

        # Cap each category score at 1.0
        for cat in scores:
            scores[cat] = min(scores[cat], 1.0)

        best_cat = ''
        best_score = 0.0
        for cat, score in scores.items():
            if score > best_score:
                best_score = score
                best_cat = cat

        if best_score < self.threshold:
            best_cat = ''

        return best_cat, round(best_score, 3), matches

    def classify_all(self, records: list[EmailRecord],
                     update_record: bool = True) -> list[dict]:
        """First pass: rule-based classification of all records."""
        results = []
        for r in records:
            cat, score, matches = self.classify(r)
            if update_record:
                r.category = cat
            results.append({
                'record_id': r.record_id,
                'category': cat,
                'confidence': score,
                'source': 'rule',
                'matches': matches,
            })
        return results

    # ---- IP propagation (second pass) ----

    def propagate_by_ip(self, records: list[EmailRecord],
                        update_record: bool = True) -> list[dict]:
        """Second pass: spread labels from classified → unclassified via shared IP.

        Workflow:
          1. Build IP→category map from rule-classified records
          2. For each unclassified record, check if its IP is known
          3. If yes, assign that category with IP_CONFIDENCE

        Returns list of newly-classified records.
        """
        self.ip_classifier.fit(records)
        return self.ip_classifier.propagate(records, update_record=update_record)

    def full_pipeline(self, records: list[EmailRecord]) -> dict:
        """Run both passes and return a summary dict.

        Returns:
            {
                'rule_results':   [...],      # first-pass results
                'ip_propagated':  [...],      # second-pass additions
                'total':          N,
                'rule_classified': N,
                'ip_classified':  N,
                'final_unclassified': N,
                'ip_stats':       {...},
            }
        """
        # Pass 1: rules
        rule_results = self.classify_all(records, update_record=True)
        n_rule = sum(1 for r in rule_results if r['category'])

        # Pass 2: IP propagation
        ip_results = self.propagate_by_ip(records, update_record=True)
        n_ip = len(ip_results)

        return {
            'rule_results': rule_results,
            'ip_propagated': ip_results,
            'total': len(records),
            'rule_classified': n_rule,
            'ip_classified': n_ip,
            'final_unclassified': len(records) - n_rule - n_ip,
            'ip_stats': self.ip_classifier.stats(),
        }

    # ---- BERT fusion (third pass) ----

    def load_bert(self, model_path: str = "models/bert_classifier") -> None:
        """Load BERT model for fusion classification."""
        from bert_train import BertPredictor
        self._bert = BertPredictor(model_path=model_path)

    def classify_fused(self, record: EmailRecord) -> tuple[str, float, str, dict]:
        """Single-record fusion classification.

        Returns: (category, confidence, source, detail)
            source: 'bert' | 'consensus' | 'rule_fallback' | ''
            detail: {'rule_cat': ..., 'rule_conf': ..., 'bert_cat': ..., 'bert_conf': ...}
        """
        # Run both classifiers in parallel
        rule_cat, rule_conf, rule_matches = self.classify(record)
        detail = {'rule_cat': rule_cat, 'rule_conf': rule_conf,
                  'bert_cat': '', 'bert_conf': 0.0}

        bert_cat, bert_conf = '', 0.0
        if self._bert is not None:
            try:
                bert_cat, bert_conf = self._bert.predict(
                    record.subject, record.content,
                    from_name=record.from_name or ''
                )
            except Exception:
                bert_cat, bert_conf = '', 0.0
            detail['bert_cat'] = bert_cat
            detail['bert_conf'] = bert_conf

        # ---- fusion logic ----
        if bert_cat and bert_conf >= 0.8:
            # BERT very confident → trust BERT
            return bert_cat, bert_conf, 'bert', detail
        elif bert_cat and bert_cat == rule_cat:
            # Consensus → mutual confirmation, take max confidence
            return rule_cat, max(rule_conf, bert_conf), 'consensus', detail
        elif bert_cat and bert_conf >= 0.5:
            # BERT disagrees with moderate confidence → trust BERT (rules can't do context)
            return bert_cat, bert_conf - 0.1, 'bert', detail
        elif rule_cat and rule_conf >= 0.7:
            # BERT uncertain, rule strong → fall back to rule
            return rule_cat, rule_conf, 'rule_fallback', detail
        elif bert_cat:
            # BERT has weak signal, rule has nothing → use BERT
            return bert_cat, bert_conf, 'bert', detail
        elif rule_cat:
            # Only rule has signal
            return rule_cat, rule_conf, 'rule_fallback', detail
        else:
            return '', 0.0, '', detail

    def classify_all_fused(self, records: list[EmailRecord],
                           update_record: bool = True) -> dict[str, Any]:
        """Three-pass classification: rule evaluation → BERT fusion → IP propagation.

        Returns list of dicts: {record_id, category, confidence, source, detail}.
        """
        from collections import Counter

        # ---- Pass 1: run rules (evaluate, don't commit yet) ----
        rule_results = self.classify_all(records, update_record=False)

        # ---- Pass 2: BERT fusion ----
        fusion_results = []
        for r in records:
            cat, conf, source, detail = self.classify_fused(r)
            if update_record:
                r.category = cat
            fusion_results.append({
                'record_id': r.record_id,
                'category': cat,
                'confidence': conf,
                'source': source,
                'rule_cat': detail['rule_cat'],
                'rule_conf': detail['rule_conf'],
                'bert_cat': detail['bert_cat'],
                'bert_conf': detail['bert_conf'],
            })

        # ---- Pass 3: IP propagation ----
        self.ip_classifier.fit(records)
        ip_results = self.ip_classifier.propagate(records, update_record=update_record)
        for r in ip_results:
            r['source'] = 'ip'

        # ---- Build summary ----
        source_counts = Counter(r['source'] for r in fusion_results if r['category'])
        cat_counts = Counter(r['category'] for r in fusion_results if r['category'])

        return {
            'total': len(records),
            'classified': cat_counts.total(),
            'unclassified': len(records) - cat_counts.total(),
            'sources': dict(source_counts),
            'categories': dict(cat_counts),
            'ip_propagated': len(ip_results),
            'fusion_results': fusion_results,
            'ip_results': ip_results,
        }

    # ---- report data collection ----

    def build_report(self, pipeline_result: dict) -> dict:
        """Collect comprehensive stats from a fused pipeline run.

        Returns a dict suitable for JSON export. All the data you need for a report.
        """
        total = pipeline_result['total']
        fusion = pipeline_result['fusion_results']
        classified = [r for r in fusion if r['category']]
        unclassified_num = pipeline_result['unclassified']

        # 1. Confidence distribution (buckets)
        conf_buckets = {'0.0-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0}
        for r in classified:
            c = r['confidence']
            if c < 0.3: conf_buckets['0.0-0.3'] += 1
            elif c < 0.5: conf_buckets['0.3-0.5'] += 1
            elif c < 0.7: conf_buckets['0.5-0.7'] += 1
            elif c < 0.9: conf_buckets['0.7-0.9'] += 1
            else: conf_buckets['0.9-1.0'] += 1

        # 2. Per-category confidence stats (min/max/mean)
        cat_confs = {c: [] for c in CATEGORIES}
        for r in classified:
            cat_confs[r['category']].append(r['confidence'])
        cat_conf_stats = {}
        for cat, confs in cat_confs.items():
            if confs:
                cat_conf_stats[cat] = {
                    'count': len(confs),
                    'mean': round(sum(confs) / len(confs), 3),
                    'min': round(min(confs), 3),
                    'max': round(max(confs), 3),
                    'median': round(sorted(confs)[len(confs)//2], 3),
                }

        # 3. Source contribution
        source_counts = Counter(r['source'] for r in classified)
        bert_dominant = source_counts.get('bert', 0)
        consensus = source_counts.get('consensus', 0)
        rule_fallback = source_counts.get('rule_fallback', 0)
        ip_propagated = pipeline_result.get('ip_propagated', 0)

        # 4. Rule-BERT agreement matrix
        agreement = Counter()
        for r in fusion:
            rc = r.get('rule_cat') or 'unclassified'
            bc = r.get('bert_cat') or 'unclassified'
            agreement[f'{rc}→{bc}'] += 1

        # 5. Where BERT overrides rules (rule false positive analysis)
        bert_overrides = []
        for r in fusion:
            if r['source'] == 'bert' and r.get('rule_cat') and r.get('rule_cat') != r['category']:
                bert_overrides.append({
                    'rule': r['rule_cat'],
                    'bert': r['category'],
                    'rule_conf': r.get('rule_conf', 0),
                    'bert_conf': r.get('bert_conf', 0),
                })
        override_summary = Counter(f"{o['rule']}→{o['bert']}" for o in bert_overrides)

        # 6. Rule-only breakdown (what rules would have done alone)
        rule_only = Counter()
        rule_only_classified = 0
        for r in fusion:
            rc = r.get('rule_cat') or ''
            rule_only[rc or 'unclassified'] += 1
            if rc:
                rule_only_classified += 1

        return {
            'total': total,
            'classified': len(classified),
            'unclassified': unclassified_num,
            'coverage': round(len(classified) / total * 100, 1) if total else 0,
            'confidence_distribution': conf_buckets,
            'confidence_by_category': cat_conf_stats,
            'source_contribution': {
                'bert': bert_dominant,
                'consensus': consensus,
                'rule_fallback': rule_fallback,
                'ip': ip_propagated,
            },
            'rule_bert_agreement': dict(agreement.most_common(30)),
            'bert_overrides_rule': dict(override_summary.most_common(20)),
            'rule_only_coverage': {
                'classified': rule_only_classified,
                'unclassified': rule_only.get('unclassified', 0),
                'rate': round(rule_only_classified / total * 100, 1) if total else 0,
                'distribution': {c: rule_only.get(c, 0) for c in CATEGORIES},
            },
            'ip_stats': self.ip_classifier.stats() if self.ip_classifier.ip_map else {},
        }

    def print_report(self, report: dict) -> None:
        """Pretty-print a report dict."""
        total = report['total']
        print(f"\n{'='*60}")
        print(f"  融合分类报告")
        print(f"{'='*60}")
        print(f"  总邮件数:       {total:>8,}")
        print(f"  已分类:         {report['classified']:>8,}  ({report['coverage']}%)")
        print(f"  未分类:         {report['unclassified']:>8,}")

        # Source contribution
        src = report['source_contribution']
        print(f"\n  ── 决策来源 ──")
        print(f"  BERT 主导:      {src['bert']:>8,}  (BERT 自行判定)")
        print(f"  规则+BERT一致:  {src['consensus']:>8,}  (双重确认)")
        print(f"  规则兜底:       {src['rule_fallback']:>8,}  (BERT 不确定时)")
        print(f"  IP 传播:        {src['ip']:>8,}  (同 IP 归类)")

        # Confidence
        conf = report['confidence_distribution']
        print(f"\n  ── 置信度分布 ──")
        for bucket in ['0.9-1.0', '0.7-0.9', '0.5-0.7', '0.3-0.5', '0.0-0.3']:
            cnt = conf.get(bucket, 0)
            bar = '█' * max(1, cnt // max(1, report['classified'] // 60))
            print(f"  {bucket:8s}  {cnt:>6,}  {bar}")

        # Per-category
        print(f"\n  ── 各类置信度 ──")
        print(f"  {'类别':12s}  {'数量':>6}  {'均值':>6}  {'中位':>6}  {'最低':>6}  {'最高':>6}")
        for cat in CATEGORIES:
            s = report['confidence_by_category'].get(cat, {})
            if s:
                print(f"  {CATEGORY_LABELS[cat]:10s}  {s['count']:>6}  {s['mean']:>6.3f}  "
                      f"{s['median']:>6.3f}  {s['min']:>6.3f}  {s['max']:>6.3f}")

        # Rule-only
        ro = report['rule_only_coverage']
        print(f"\n  ── 纯规则对比 ──")
        print(f"  规则单独覆盖率: {ro['rate']}% ({ro['classified']}/{total})")
        print(f"  融合后覆盖率:   {report['coverage']}% ({report['classified']}/{total})")
        print(f"  融合提升:       +{report['coverage'] - ro['rate']}%")

        # BERT overrides
        overrides = report.get('bert_overrides_rule', {})
        if overrides:
            print(f"\n  ── BERT 纠正规则的 Top 错误 ──")
            for pat, cnt in sorted(overrides.items(), key=lambda x: -x[1])[:8]:
                print(f"  {cnt:>4}  规则判 {pat}")

        # Rule-BERT agreement
        ag = report.get('rule_bert_agreement', {})
        agree = sum(v for k, v in ag.items() if '→' in k and k.split('→')[0] == k.split('→')[1])
        disagree = sum(v for k, v in ag.items() if '→' in k and k.split('→')[0] != k.split('→')[1])
        if agree + disagree > 0:
            print(f"\n  ── 规则-BERT 一致率 ──")
            print(f"  一致:   {agree:>6,}  ({agree/(agree+disagree)*100:.1f}%)")
            print(f"  不一致: {disagree:>6,}  ({disagree/(agree+disagree)*100:.1f}%)")

    # ---- summary ----

    def summary(self, results: list[dict]) -> dict[str, int]:
        cnt = Counter(r['category'] or 'unclassified' for r in results)
        return dict(cnt)

    def summary_by_record(self, records: list[EmailRecord]) -> dict[str, int]:
        cnt = Counter(r.category or 'unclassified' for r in records)
        return dict(cnt)

    # ---- rule info ----

    def rule_count(self) -> dict[str, int]:
        return {c: len(rs) for c, rs in self.rules.items()}

    def list_rules(self, category: Optional[str] = None) -> list[Rule]:
        if category:
            return list(self.rules.get(category, []))
        result = []
        for rs in self.rules.values():
            result.extend(rs)
        return result

# All rules are now loaded from src/rules/*.json at runtime.
# See: src/rules/adult.json, gambling.json, marketing.json, phishing.json, fraud.json

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description='邮件分类器 (规则 + IP传播)')
    sub = p.add_subparsers(dest='cmd')

    # classify
    cla = sub.add_parser('classify', help='规则分类 + BERT融合 + IP传播')
    cla.add_argument('log', nargs='?', default='database/spam_email_data.log',
                     help='日志文件路径')
    cla.add_argument('--db', default='database/spam_emails.db',
                     help='SQLite 数据库路径（分类结果写回此库）')
    cla.add_argument('--limit', type=int, default=0,
                     help='只分类前 N 条（0=全部分类）')
    cla.add_argument('--threshold', type=float, default=0.2,
                     help='规则最低置信度 (默认 0.2)')
    cla.add_argument('--no-ip', action='store_true',
                     help='跳过 IP 传播')
    cla.add_argument('--use-bert', action='store_true',
                     help='启用 BERT 融合分类')
    cla.add_argument('--bert-model', default='models/bert_classifier',
                     help='BERT 模型路径 (默认: models/bert_classifier)')
    cla.add_argument('--report', metavar='PATH',
                     help='导出详细报告 JSON (用于写论文/报告)')

    # rules
    rules_cmd = sub.add_parser('rules', help='查看分类规则')
    rules_cmd.add_argument('--category', '-c', choices=CATEGORIES,
                           help='按分类筛选')

    # predict
    pred = sub.add_parser('predict', help='预测单封邮件')
    pred.add_argument('subject', nargs='?', default='', help='邮件主题 (使用 --record-id 时可省略)')
    pred.add_argument('content', nargs='?', default='', help='邮件内容')
    pred.add_argument('--use-bert', action='store_true', help='启用 BERT 融合')
    pred.add_argument('--bert-model', default='models/bert_classifier', help='BERT 模型路径')
    pred.add_argument('--record-id', help='从数据库按 record_id 查询')

    # ip-stats
    ip_cmd = sub.add_parser('ip-stats', help='查看 IP 分类器统计')
    ip_cmd.add_argument('log', nargs='?', default='database/spam_email_data.log',
                        help='日志文件路径')
    ip_cmd.add_argument('--limit', type=int, default=0,
                        help='分析前 N 条（0=全部）')

    args = p.parse_args()

    if args.cmd == 'classify':
        from parser import parse_file
        from db import EmailDB

        print(f"正在解析 {args.log} ...")
        records = parse_file(args.log)
        if args.limit:
            records = records[:args.limit]
        total = len(records)
        print(f"解析完成: {total:,} 条")

        clf = EmailClassifier(rule_threshold=args.threshold)
        clf.load_default_rules()
        rule_counts = clf.rule_count()
        total_rules = sum(rule_counts.values())
        result: dict[str, Any] | None = None
        print(f"加载规则: {total_rules} 条  {dict(rule_counts)}")

        if args.use_bert:
            # ======== BERT fusion pipeline ========
            print(f"\n加载 BERT 模型: {args.bert_model} ...")
            clf.load_bert(args.bert_model)

            print("\n[融合分类] 规则 × BERT × IP 三路...")
            result = clf.classify_all_fused(records, update_record=True)

            # Show per-source breakdown
            print(f"\n分类来源分布:")
            for src, cnt in sorted(result['sources'].items(), key=lambda x: -x[1]):
                label = {'bert': 'BERT主导', 'consensus': '规则+BERT一致',
                         'rule_fallback': '规则兜底', 'ip': 'IP传播'}.get(src, src)
                print(f"  {label:16s}  {cnt:>6,}")

            print(f"\n最终分类结果:")
            for cat in CATEGORIES:
                cnt = result['categories'].get(cat, 0)
                if cnt:
                    pct = cnt / total * 100
                    label = CATEGORY_LABELS[cat]
                    bar = '█' * max(1, int(pct))
                    print(f"  {label:10s}  {cnt:>6,}  ({pct:5.1f}%)  {bar}")

            classified = result['classified']
            uncl = result['unclassified']
            n_ip = result['ip_propagated']
            print(f"\n{'='*50}")
            print(f"  分类合计: {classified:,} / {total:,}  (覆盖率 {classified/total*100:.1f}%)")
            print(f"  其中 IP 传播: +{n_ip:,}")
            print(f"  未分类: {uncl:,}")

        else:
            # ======== Rule-only pipeline (original) ========
            print("\n[第一轮] 规则分类...")
            rule_results = clf.classify_all(records, update_record=True)
            n_rule = sum(1 for r in rule_results if r['category'])
            rule_summary = clf.summary(rule_results)

            print(f"\n规则分类结果:")
            for cat in CATEGORIES:
                cnt = rule_summary.get(cat, 0)
                if cnt:
                    pct = cnt / total * 100
                    label = CATEGORY_LABELS[cat]
                    bar = '█' * max(1, int(pct))
                    print(f"  {label:10s}  {cnt:>6,}  ({pct:5.1f}%)  {bar}")
            uncl = rule_summary.get('unclassified', 0)
            if uncl:
                print(f"  {'未分类':10s}  {uncl:>6,}  ({uncl/total*100:5.1f}%)")
            print(f"  规则分类合计: {n_rule:,} / {total:,}")

            n_ip = 0
            if not args.no_ip:
                print("\n[第二轮] IP 传播分类...")
                clf.ip_classifier.fit(records)
                print(f"  IP 库: {clf.ip_classifier.stats()['total_ips']} 个可靠 IP")
                ip_results = clf.propagate_by_ip(records, update_record=True)
                n_ip = len(ip_results)
                ip_summary: Counter[str] = Counter()
                for r in ip_results:
                    ip_summary[r['category']] += 1
                for cat in CATEGORIES:
                    cnt = ip_summary.get(cat, 0)
                    if cnt:
                        label = CATEGORY_LABELS[cat]
                        print(f"  {label:10s}  +{cnt:>6,}")
                print(f"  IP 传播合计: +{n_ip:,}")
            else:
                print("\n(已跳过 IP 传播)")

            final_uncl = total - n_rule - n_ip
            print(f"\n{'='*50}")
            print(f"  最终: 已分类 {n_rule + n_ip:,} (规则 {n_rule:,} + IP {n_ip:,})"
                  f"  |  未分类 {final_uncl:,}"
                  f"  |  覆盖率 {(n_rule+n_ip)/total*100:.1f}%")

        # Generate report
        if args.use_bert and args.report and result is not None:
            report = clf.build_report(result)
            clf.print_report(report)
            report_path = Path(args.report)
            with report_path.open('w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n报告已导出: {report_path}")

        # Write back to DB
        if args.db:
            print(f"\n正在写回数据库 {args.db} ...")
            db = EmailDB(args.db)
            updates = {r.record_id: r.category
                       for r in records if r.category}
            n = db.set_category_bulk(updates)
            print(f"已更新 {n} 条记录的分类标签")
            db.close()

    elif args.cmd == 'predict':
        from parser import EmailRecord

        # Fetch from DB if --record-id is given
        if args.record_id:
            from db import EmailDB
            db = EmailDB()
            rows = db.query("SELECT * FROM emails WHERE record_id=?", (args.record_id,))
            db.close()
            if rows:
                r = rows[0]
                subject = r['subject'] or ''
                content = r['content'] or ''
                print(f"从数据库加载: {args.record_id}")
            else:
                print(f"[ERROR] record_id 不存在: {args.record_id}")
                return
        elif args.subject:
            subject = args.subject
            content = args.content or ''
        else:
            print("[ERROR] 请提供 subject 或 --record-id")
            return

        record = EmailRecord(
            record_id='manual', raw={},
            subject=subject, content=content,
        )

        clf = EmailClassifier()
        clf.load_default_rules()

        # Rule classification
        rule_cat, rule_conf, rule_matches = clf.classify(record)
        print(f"\n{'='*55}")
        print(f"  主题: {subject[:120]}")
        print(f"  内容: {content[:200] if content else '(空)'}")
        print(f"{'='*55}")
        print(f"\n[规则引擎]")
        print(f"  分类:   {rule_cat or '未分类'}  ({CATEGORY_LABELS.get(rule_cat, '')})")
        print(f"  置信度: {rule_conf:.3f}")
        if rule_matches:
            print(f"  命中规则 ({len(rule_matches)} 条):")
            for m in rule_matches:
                print(f"    [{m['category']:12s}] c={m['confidence']:.2f}  {m['field']} ~ {m['matched'][:60]}")
                print(f"                     {m['reason']}")
        else:
            print(f"  (无规则命中)")

        # BERT
        if args.use_bert:
            clf.load_bert(args.bert_model)
            cat, conf, source, detail = clf.classify_fused(record)
            print(f"\n[BERT 预测]")
            print(f"  分类:   {detail['bert_cat'] or '未分类'}")
            print(f"  置信度: {detail['bert_conf']:.3f}")
            print(f"\n[融合决策]")
            print(f"  最终:   {cat or '未分类'}  ({CATEGORY_LABELS.get(cat, '')})")
            print(f"  置信度: {conf:.3f}")
            print(f"  来源:   {source}  ", end='')
            if source == 'bert':
                print('(BERT 主导)')
            elif source == 'consensus':
                print('(规则+BERT 一致)')
            elif source == 'rule_fallback':
                print('(规则兜底)')
            else:
                print()

    elif args.cmd == 'rules':
        clf = EmailClassifier()
        clf.load_default_rules()
        rules_list = clf.list_rules(args.category)
        print(f"\n分类规则 ({len(rules_list)} 条):")
        print("-" * 70)
        for r in rules_list:
            print(f"  [{r.category:12s}] c={r.confidence:<4.2f}  {r.field}.match(r'{r.pattern}')")
            if r.reason:
                print(f"                     → {r.reason}")

    elif args.cmd == 'ip-stats':
        from parser import parse_file
        print(f"正在分析 IP 分布: {args.log} ...")
        records = parse_file(args.log)
        if args.limit:
            records = records[:args.limit]

        # First classify with rules to get base labels
        clf = EmailClassifier()
        clf.load_default_rules()
        clf.classify_all(records, update_record=True)

        # Then fit IP classifier
        clf.ip_classifier.fit(records)
        stats = clf.ip_classifier.stats()
        print(f"\n可靠 IP 数量: {stats['total_ips']}")
        print(f"IP 分布:")
        for cat, cnt in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            label = CATEGORY_LABELS.get(cat, cat)
            print(f"  {label:10s}  {cnt:>6,} IPs")

    else:
        p.print_help()


if __name__ == '__main__':
    main()
