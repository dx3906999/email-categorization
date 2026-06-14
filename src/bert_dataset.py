"""
BERT dataset preparation — JSONL annotated data → HuggingFace Dataset.

Usage:
    python src/bert_dataset.py                           # auto-find annotated files
    python src/bert_dataset.py --data-dir data/          # custom dir
    python src/bert_dataset.py --model hfl/chinese-roberta-wwm-ext

Output (saved to data/processed/):
    train_dataset/   val_dataset/   test_dataset/
    label_map.json   class_weights.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

# Suppress HF warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CATEGORIES = ['adult_violence', 'commercial', 'phishing', 'finance', 'academic']
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def load_labeled_data(data_dir: str | Path = "data") -> list[dict]:
    """Load all *_labeled.jsonl files, filter low-confidence, return clean records."""
    data_dir = Path(data_dir)
    records = []

    for pattern in ["*_labeled.jsonl", "*_labelled.jsonl"]:
        for f in sorted(data_dir.glob(pattern)):
            # Skip cluster samples — they have free-form labels, not our 5-class system
            if 'cluster' in f.name.lower():
                continue
            print(f"  loading {f.name}...")
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("error"):
                        continue

                    # Prefer LLM label; fall back to rule label
                    cat = r.get("llm_category") or r.get("rule_category") or ""
                    conf = r.get("llm_confidence") or r.get("rule_confidence") or 0
                    cat = cat.strip()

                    if cat not in CATEGORIES:
                        continue
                    if conf < 0.5:  # filter unreliable labels
                        continue

                    from_name = r.get('from_name', '') or ''
                    records.append({
                        "text": f"{from_name} [SEP] {r['subject']} [SEP] {r['content']}",
                        "label": LABEL2ID[cat],
                        "confidence": conf,
                        "record_id": r["record_id"],
                    })

    return records


def compute_class_weights(labels: list[int]) -> dict[str, float]:
    """Compute inverse-frequency weights for weighted CrossEntropyLoss."""
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(CATEGORIES)
    weights = {}
    for i, cat in enumerate(CATEGORIES):
        cnt = counts.get(i, 1)
        # 1 / sqrt(count) is smoother than 1/count for extreme imbalance
        weights[cat] = round(np.sqrt(total / cnt) / n_classes, 4)
    return weights


def tokenize_and_split(
    records: list[dict],
    model_name: str = "hfl/chinese-roberta-wwm-ext",
    max_length: int = 256,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    output_dir: str | Path = "data/processed",
):
    """Tokenize records, split into train/val/test, save to disk."""
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split: train + val + test, stratified by label
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]

    # First split out test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    # Then split out val from train
    val_frac = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_frac, random_state=seed, stratify=train_labels
    )

    print(f"\nSplit: train={len(train_texts)}  val={len(val_texts)}  test={len(test_texts)}")
    for name, labs in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        cnt = Counter(labs)
        dist = "  ".join(f"{ID2LABEL[i]}={cnt.get(i,0)}" for i in range(len(CATEGORIES)))
        print(f"  {name}: {dist}")

    # Tokenize
    def _tokenize(texts: list[str]):
        return tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors=None,
        )

    print(f"\nTokenizing (max_length={max_length})...")
    train_enc = _tokenize(train_texts)
    val_enc = _tokenize(val_texts)
    test_enc = _tokenize(test_texts)

    # Build datasets
    def _to_dataset(enc, labs):
        return Dataset.from_dict({**enc, "labels": labs})

    dataset = DatasetDict({
        "train": _to_dataset(train_enc, train_labels),
        "val": _to_dataset(val_enc, val_labels),
        "test": _to_dataset(test_enc, test_labels),
    })

    # Save
    dataset.save_to_disk(str(output_dir))
    print(f"Dataset saved to {output_dir}/")

    # Save label map + class weights
    with (output_dir / "label_map.json").open("w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, ensure_ascii=False, indent=2)

    cw = compute_class_weights(train_labels)
    with (output_dir / "class_weights.json").open("w") as f:
        json.dump(cw, f, indent=2)
    print(f"Class weights: {cw}")

    return dataset, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(description="BERT 数据集准备")
    p.add_argument("--data-dir", default="data", help="标注 JSONL 所在目录")
    p.add_argument("--model", default="hfl/chinese-roberta-wwm-ext", help="预训练模型名")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    print("Loading annotated data...")
    records = load_labeled_data(args.data_dir)

    if not records:
        print("[ERROR] 未找到已标注数据。请先运行 LLM 标注。")
        print("  python src/llm_label.py data/quality_verification.jsonl --concurrency 10")
        sys.exit(1)

    # Show distribution
    dist = Counter(r["label"] for r in records)
    print(f"\nTotal usable records: {len(records)}")
    for i, cat in enumerate(CATEGORIES):
        print(f"  {cat:12s}  {dist.get(i, 0):>6,}")

    tokenize_and_split(
        records,
        model_name=args.model,
        max_length=args.max_length,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
