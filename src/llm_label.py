"""
LLM annotation script — 调用大模型对邮件分类打标

Usage:
    # 默认用 DeepSeek（最便宜）
    python src/llm_label.py data/quality_verification.jsonl

    # 指定模型
    python src/llm_label.py data/train.jsonl --provider anthropic --model claude-haiku-4-5

    # 控制并发和速率
    python src/llm_label.py data/quality_verification.jsonl --concurrency 5 --delay 0.5

Providers:
    deepseek  — DeepSeek-V3 (默认, ¥1/M in, ¥4/M out)
    openai    — GPT-4o-mini ($0.15/M in, $0.60/M out)
    anthropic — Claude Haiku 4.5 ($0.25/M in, $1.25/M out)

Requirements:
    pip install httpx python-dotenv

API keys are loaded from the .env file at the project root:
    .env  →  DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(_env_path)
except ImportError:
    print("提示: pip install python-dotenv 可自动加载 .env 文件")
    pass

try:
    import httpx
except ImportError:
    print("需要 httpx: pip install httpx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_OPEN = """你是一个垃圾邮件分析专家。请根据这封邮件的主题和内容，给它一个简洁的分类标签。

要求：
1. 标签用 2-4 个汉字概括这封邮件的核心意图（如"假发票"、"钓鱼验证"、"色情推广"等）
2. 不要使用预定义的分类，根据邮件实际内容自行命名
3. 同样意图的邮件尽量使用相同的标签名
4. 以 JSON 格式回复，不要带其他文字

回复格式：
{"category": "你命名的标签", "confidence": 0.XX, "reason": "一句话判断依据"}"""

SYSTEM_PROMPT = """你是一个垃圾邮件分类专家。根据邮件的主题和内容，判断它属于以下哪一类：

- adult（色情淫秽）：成人内容、色情网站、约炮交友、AV视频
- gambling（赌博博彩）：线上赌场、体育博彩、彩票投注
- marketing（营销推广）：展会搭建、招标采购、广告营销、会议征稿、新闻简报
- phishing（钓鱼诈骗）：账号验证、密码窃取、虚假安全通知、快递通知、邮箱升级
- fraud（假发票/诈骗）：代开发票、增值税发票、中奖诈骗、遗产诈骗、转账诈骗、兼职诈骗

要求：
1. 仔细阅读邮件的主题和内容，综合判断
2. 如果内容极少或信息不足以判断，归类为你认为最可能的一类，并给出低置信度
3. 以 JSON 格式回复，不要带其他文字

回复格式：
{"category": "类别英文名", "confidence": 0.XX, "reason": "一句话判断依据"}"""

USER_PROMPT_TEMPLATE = """请分类以下邮件：

发件人：{from_name}
主题：{subject}
内容：{content}"""

CATEGORIES = ['adult', 'gambling', 'marketing', 'phishing', 'fraud']

# Provider configs
PROVIDERS = {
    'deepseek': {
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'model': 'deepseek-chat',
        'key_env': 'DEEPSEEK_API_KEY',
        'price_in': 1.0 / 1_000_000,    # ¥1/M tokens
        'price_out': 2.0 / 1_000_000,   # ¥2/M tokens
    },
    'openai': {
        'url': 'https://api.openai.com/v1/chat/completions',
        'model': 'gpt-4o-mini',
        'key_env': 'OPENAI_API_KEY',
        'price_in': 0.15 / 1_000_000,   # $0.15/M tokens
        'price_out': 0.60 / 1_000_000,
    },
    'anthropic': {
        'url': 'https://api.anthropic.com/v1/messages',
        'model': 'claude-haiku-4-5-20251001',
        'key_env': 'ANTHROPIC_API_KEY',
        'price_in': 0.25 / 1_000_000,   # $0.25/M tokens
        'price_out': 1.25 / 1_000_000,
    },
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

@dataclass
class LabelResult:
    record_id: str
    from_name: str
    subject: str
    content: str
    rule_category: str

    llm_category: str = ""
    llm_confidence: float = 0.0
    llm_reason: str = ""
    error: str = ""
    cost: float = 0.0


class LLMClient:
    def __init__(self, provider: str = 'deepseek', model: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: int = 60,
                 mode: str = 'closed'):
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.mode = mode
        self.system_prompt = SYSTEM_PROMPT_OPEN if mode == 'open' else SYSTEM_PROMPT
        self.url = cfg['url']
        self.model = model or cfg['model']
        self.api_key = api_key or os.environ.get(cfg['key_env'], '')
        self.timeout = timeout
        self.price_in = cfg['price_in']
        self.price_out = cfg['price_out']

        if not self.api_key:
            raise ValueError(f"缺少 API Key: 请设置环境变量 {cfg['key_env']}")

        if provider == 'anthropic':
            self._headers = {
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json',
            }
        else:
            self._headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }

    def _build_request(self, from_name: str, subject: str, content: str) -> dict:
        user_text = USER_PROMPT_TEMPLATE.format(from_name=from_name, subject=subject, content=content)

        if self.provider == 'anthropic':
            return {
                'model': self.model,
                'max_tokens': 200,
                'system': self.system_prompt,
                'messages': [{'role': 'user', 'content': user_text}],
            }
        else:
            return {
                'model': self.model,
                'max_tokens': 200,
                'temperature': 0.1,
                'messages': [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': user_text},
                ],
            }

    def _parse_response(self, resp_data: dict) -> tuple[str, float, str]:
        if self.provider == 'anthropic':
            text = resp_data['content'][0]['text'].strip()
        else:
            text = resp_data['choices'][0]['message']['content'].strip()

        # Extract JSON from response (handle markdown fences)
        if '```' in text:
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        category = result.get('category', '').strip()
        # In closed mode, validate against predefined categories
        if self.mode != 'open' and category.lower() not in CATEGORIES:
            category = ''
        confidence = float(result.get('confidence', 0))
        reason = result.get('reason', '')
        return category, confidence, reason

    async def label_one(self, client: httpx.AsyncClient, record: dict,
                        semaphore: asyncio.Semaphore) -> LabelResult:
        result = LabelResult(
            record_id=record['record_id'],
            from_name=record.get('from_name', ''),
            subject=record['subject'],
            content=record['content'],
            rule_category=record.get('rule_category', ''),
        )

        async with semaphore:
            # Retry loop
            for attempt in range(3):
                try:
                    body = self._build_request(record.get('from_name', ''), record['subject'], record['content'])
                    resp = await client.post(
                        self.url,
                        headers=self._headers,
                        json=body,
                        timeout=self.timeout,
                    )
                    if resp.status_code == 429:
                        # Rate limited — wait and retry
                        wait = 2 ** attempt
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()

                    # Estimate token usage & cost
                    if self.provider == 'anthropic':
                        in_tokens = data.get('usage', {}).get('input_tokens', 0)
                        out_tokens = data.get('usage', {}).get('output_tokens', 0)
                    else:
                        in_tokens = data.get('usage', {}).get('prompt_tokens', 0)
                        out_tokens = data.get('usage', {}).get('completion_tokens', 0)
                    result.cost = in_tokens * self.price_in + out_tokens * self.price_out

                    cat, conf, reason = self._parse_response(data)
                    result.llm_category = cat
                    result.llm_confidence = conf
                    result.llm_reason = reason
                    return result

                except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
                    result.error = str(e)[:200]
                    if attempt < 2:
                        await asyncio.sleep(1)
                except Exception as e:
                    result.error = f"{type(e).__name__}: {str(e)[:200]}"
                    break

        return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    provider: str = 'deepseek',
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    concurrency: int = 5,
    delay: float = 0.2,
    limit: int = 0,
    mode: str = 'closed',
) -> tuple[list[LabelResult], dict]:
    """Run LLM labeling on a JSONL file.

    Returns (results, stats_dict).
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_labeled{input_path.suffix}")
    output_path = Path(output_path)

    # Load records
    records = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit and limit < len(records):
        records = records[:limit]

    print(f"文件: {input_path}")
    print(f"记录数: {len(records)}")
    print(f"Provider: {provider}  Model: {model or PROVIDERS[provider]['model']}")
    print(f"并发: {concurrency}  输出: {output_path}")
    print()

    if not records:
        return [], {}

    # Run
    client = LLMClient(provider=provider, model=model, api_key=api_key, mode=mode)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[LabelResult] = []
    start_time = time.time()
    total_cost = 0.0

    async with httpx.AsyncClient() as http:
        tasks = []
        for i, rec in enumerate(records):
            # Small delay between requests to avoid burst
            if i > 0 and delay:
                await asyncio.sleep(delay)
            tasks.append(client.label_one(http, rec, semaphore))

            # Flush batches for progress reporting
            if len(tasks) >= concurrency * 2:
                batch = await asyncio.gather(*tasks)
                results.extend(batch)
                tasks.clear()
                elapsed = time.time() - start_time
                done = len(results)
                rate = done / elapsed
                eta = (len(records) - done) / rate if rate > 0 else 0
                print(f"\r  进度: {done}/{len(records)} ({done/len(records)*100:.0f}%)  "
                      f"{rate:.1f}条/秒  ETA {eta:.0f}秒", end='', flush=True)

        # Final flush
        if tasks:
            batch = await asyncio.gather(*tasks)
            results.extend(batch)

    elapsed = time.time() - start_time

    # Save first (so data survives even if terminal encoding fails)
    with output_path.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps({
                'record_id': r.record_id,
                'from_name': r.from_name,
                'subject': r.subject,
                'content': r.content,
                'rule_category': r.rule_category,
                'llm_category': r.llm_category,
                'llm_confidence': r.llm_confidence,
                'llm_reason': r.llm_reason,
                'error': r.error,
            }, ensure_ascii=False) + '\n')

    total_cost = sum(r.cost for r in results)
    print(f"\r  完成: {len(results)}/{len(records)}  耗时 {elapsed:.0f}秒  "
          f"速率 {len(results)/elapsed:.1f}条/秒  "
          f"费用 ~{total_cost:.4f} {'CNY' if provider == 'deepseek' else 'USD'}  "
          f"-> {output_path}")

    # Compute stats
    stats = _compute_stats(results)
    stats['mode'] = mode
    _print_stats(stats)

    return results, stats


def _compute_stats(results: list[LabelResult]) -> dict:
    from collections import Counter

    total = len(results)
    success = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    labeled = [r for r in success if r.llm_category]

    # Agreement between LLM and rules (only for originally-classified records)
    classified = [r for r in labeled if r.rule_category]
    agree = sum(1 for r in classified if r.llm_category == r.rule_category)
    disagree = len(classified) - agree

    # Confusion matrix-like
    llm_dist = Counter(r.llm_category for r in labeled)
    rule_dist = Counter(r.rule_category for r in classified)

    # Disagreement examples
    disagreements = []
    for r in classified:
        if r.llm_category != r.rule_category:
            disagreements.append({
                'record_id': r.record_id,
                'subject': r.subject[:80],
                'rule': r.rule_category,
                'llm': r.llm_category,
                'llm_reason': r.llm_reason[:100],
            })

    return {
        'total': total,
        'success': len(success),
        'errors': len(errors),
        'labeled': len(labeled),
        'agreement': {'agree': agree, 'disagree': disagree,
                      'rate': agree / len(classified) if classified else 0},
        'llm_dist': dict(llm_dist.most_common()),
        'rule_dist': dict(rule_dist.most_common()),
        'disagreements': disagreements[:20],
    }


def _print_stats(stats: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  标注统计")
    print(f"{'='*55}")
    print(f"  总数:        {stats['total']:>5}")
    print(f"  成功:        {stats['success']:>5}")
    print(f"  失败:        {stats['errors']:>5}")
    print(f"  已标注:      {stats['labeled']:>5}")

    a = stats['agreement']
    if stats['labeled'] > 0:
        print(f"\n  LLM vs 规则 一致率:")
        print(f"    一致:      {a['agree']:>5}")
        print(f"    不一致:    {a['disagree']:>5}")
        print(f"    一致率:    {a['rate']:.1%}")
        if a['rate'] < 1.0:
            print(f"    (不一致的样本将影响后续 ML 训练)")

    if stats.get('mode') == 'open':
        print(f"\n  开放分类 — LLM 自行命名的类别 ({len(stats['llm_dist'])} 个):")
        for cat, cnt in stats['llm_dist'].items():
            bar = '█' * max(1, cnt // 3)
            print(f"    {cat:16s}  {cnt:>4}  {bar}")
    else:
        print(f"\n  LLM 分类分布:")
        for cat, cnt in stats['llm_dist'].items():
            print(f"    {cat:12s}  {cnt:>5}")

        if stats['disagreements']:
            print(f"\n  不一致示例 (前 10):")
            for d in stats['disagreements'][:10]:
                print(f"    规则={d['rule']:12s}  LLM={d['llm']:12s}  |  {d['subject'][:60]}")
                print(f"      LLM理由: {d['llm_reason']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(description='LLM 邮件标注脚本')
    p.add_argument('input', help='输入 JSONL 文件路径')
    p.add_argument('-o', '--output', help='输出文件路径 (默认: 输入文件名_labeled.jsonl)')
    p.add_argument('--provider', choices=['deepseek', 'openai', 'anthropic'],
                   default='deepseek', help='LLM 提供商 (默认: deepseek)')
    p.add_argument('--model', help='模型名称 (默认: 提供商默认模型)')
    p.add_argument('--api-key', help='API Key (默认: 从环境变量读取)')
    p.add_argument('--concurrency', type=int, default=5, help='并发数 (默认: 5)')
    p.add_argument('--delay', type=float, default=0.2, help='请求间隔秒数 (默认: 0.2)')
    p.add_argument('--limit', type=int, default=0, help='只处理前 N 条 (0=全部)')
    p.add_argument('--mode', choices=['closed', 'open'], default='closed',
                   help='closed=预定义5类  open=LLM自行命名类别 (默认: closed)')
    p.add_argument('--dry-run', action='store_true', help='打印 prompt 但不实际调用 API')

    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    if args.dry_run:
        records = []
        with input_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if args.limit:
            records = records[:args.limit]

        print(f"文件: {input_path} ({len(records)} 条)")
        print(f"\n{'='*55}")
        print("  SYSTEM PROMPT")
        print(f"{'='*55}")
        sys_prompt = SYSTEM_PROMPT_OPEN if args.mode == 'open' else SYSTEM_PROMPT
        print(sys_prompt)
        print(f"\n{'='*55}")
        print("  SAMPLE (第1条)")
        print(f"{'='*55}")
        print(USER_PROMPT_TEMPLATE.format(
            from_name=records[0].get('from_name', ''),
            subject=records[0]['subject'],
            content=records[0]['content'][:300]
        ))
        return

    asyncio.run(run_batch(
        input_path=input_path,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        concurrency=args.concurrency,
        delay=args.delay,
        limit=args.limit,
        mode=args.mode,
    ))


if __name__ == '__main__':
    main()
