"""
Compare two classification label files where numeric category IDs may differ
in meaning between the two files.

Finds the optimal alignment between label sets, then computes:
  - Agreement rate (overall and per-category)
  - Confusion matrix (before and after alignment)
  - Per-category precision/recall (after alignment)

Usage:
    python src/compare_labels.py data/email_labels.jsonl data/other_labels.jsonl
    python src/compare_labels.py data/email_labels.csv data/other_labels.csv
    python src/compare_labels.py file_a.jsonl file_b.jsonl --key-a spam_type --key-b category
    python src/compare_labels.py file_a.jsonl file_b.jsonl --show-mismatches 20
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: str, key: str) -> dict[str, int]:
    """Load a JSONL file, returning {record_id: label}."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r.get("record_id", "")
            label = r.get(key)
            if rid and label is not None:
                # Normalize to int if possible
                try:
                    label = int(label)
                except (ValueError, TypeError):
                    pass
                records[rid] = label
    return records


def load_csv(path: str, id_col: str = "record_id", label_col: str = "spam_type") -> dict[str, int]:
    """Load a CSV file, returning {record_id: label}."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get(id_col, "")
            label = row.get(label_col)
            if rid and label is not None:
                try:
                    label = int(label)
                except (ValueError, TypeError):
                    pass
                records[rid] = label
    return records


def load_file(path: str, key: str) -> dict[str, int]:
    """Auto-detect format and load."""
    path = Path(path)
    if path.suffix == ".csv":
        return load_csv(str(path))
    elif path.suffix == ".jsonl":
        return load_jsonl(str(path), key)
    else:
        # Try JSONL first
        try:
            return load_jsonl(str(path), key)
        except json.JSONDecodeError:
            return load_csv(str(path))


def find_optimal_alignment(
    confusion: dict[tuple, int],
    labels_a: set,
    labels_b: set,
) -> dict:
    """
    Find the best 1-to-1 mapping from labels_b to labels_a.

    Uses greedy assignment: for each label in A, pick the most-frequently-
    paired label in B (that hasn't been taken yet).

    Returns {label_b: label_a} mapping.
    """
    # Build a score matrix: for each (a, b), count how many records agree
    scores = defaultdict(int)
    for a in labels_a:
        for b in labels_b:
            scores[(a, b)] = confusion.get((a, b), 0)

    # Greedy bipartite matching
    assigned_a = set()
    assigned_b = set()
    mapping = {}  # b -> a

    # Sort pairs by score descending
    pairs = sorted(scores.items(), key=lambda x: -x[1])

    for (a, b), score in pairs:
        if a not in assigned_a and b not in assigned_b:
            mapping[b] = a
            assigned_a.add(a)
            assigned_b.add(b)

    # Handle unmatched labels (if |A| != |B|)
    for b in labels_b - assigned_b:
        # Assign to the most frequent a not yet assigned, or any a
        best_a = None
        best_score = -1
        for a in labels_a:
            s = confusion.get((a, b), 0)
            if s > best_score:
                best_score = s
                best_a = a
        mapping[b] = best_a if best_a is not None else (list(labels_a)[0] if labels_a else None)

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Compare two classification label files with different category numbering"
    )
    parser.add_argument("file_a", help="First label file (JSONL or CSV)")
    parser.add_argument("file_b", help="Second label file (JSONL or CSV)")
    parser.add_argument(
        "--key-a", default="spam_type",
        help="Key/column name for label in file A (default: spam_type)"
    )
    parser.add_argument(
        "--key-b", default="spam_type",
        help="Key/column name for label in file B (default: spam_type)"
    )
    parser.add_argument(
        "--show-mismatches", type=int, default=0,
        help="Show N mismatched record_id pairs (default: 0)"
    )
    parser.add_argument(
        "--show-confusion", action="store_true",
        help="Print raw confusion matrix"
    )
    args = parser.parse_args()

    # ---- Load ----
    print(f"Loading A: {args.file_a}  (key={args.key_a})")
    data_a = load_file(args.file_a, args.key_a)
    print(f"  {len(data_a)} records")

    print(f"Loading B: {args.file_b}  (key={args.key_b})")
    data_b = load_file(args.file_b, args.key_b)
    print(f"  {len(data_b)} records")

    # ---- Match by record_id ----
    common_ids = sorted(set(data_a) & set(data_b))
    only_a = set(data_a) - set(data_b)
    only_b = set(data_b) - set(data_a)

    print(f"\nCommon record_ids: {len(common_ids)}")
    if only_a:
        print(f"  Only in A: {len(only_a)}")
    if only_b:
        print(f"  Only in B: {len(only_b)}")

    if not common_ids:
        print("\nERROR: No common record_ids found. Cannot compare.")
        sys.exit(1)

    # ---- Build confusion matrix on common IDs ----
    labels_a = sorted(set(data_a[rid] for rid in common_ids))
    labels_b = sorted(set(data_b[rid] for rid in common_ids))

    print(f"\nUnique labels in A: {labels_a}")
    print(f"Unique labels in B: {labels_b}")

    confusion = Counter()
    for rid in common_ids:
        confusion[(data_a[rid], data_b[rid])] += 1

    # ---- Find optimal alignment (B -> A) ----
    alignment = find_optimal_alignment(confusion, set(labels_a), set(labels_b))

    print(f"\nOptimal alignment (B label → A label):")
    for b in sorted(alignment):
        a = alignment[b]
        agreement = confusion.get((a, b), 0)
        total_b = sum(confusion.get((la, b), 0) for la in labels_a)
        pct = agreement / total_b * 100 if total_b else 0
        print(f"  B:{b} → A:{a}  (agree on {agreement}/{total_b} = {pct:.1f}%)")

    # ---- Compute agreement after alignment ----
    correct = 0
    mismatches = []
    for rid in common_ids:
        a_label = data_a[rid]
        b_label = data_b[rid]
        mapped_b = alignment.get(b_label)
        if mapped_b == a_label:
            correct += 1
        else:
            mismatches.append((rid, a_label, b_label, mapped_b))

    total = len(common_ids)
    agree_rate = correct / total * 100

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total compared:  {total}")
    print(f"  Agreement:       {correct} / {total}  ({agree_rate:.2f}%)")
    print(f"  Disagreement:    {total - correct} / {total}  ({100 - agree_rate:.2f}%)")

    # ---- Per-category breakdown (aligned) ----
    print(f"\n{'─' * 60}")
    print(f"  Per-category breakdown (aligned by optimal mapping)")
    print(f"{'─' * 60}")
    print(f"  {'A label':>8s}  {'B label':>8s}  {'Count':>6}  {'% of A':>8s}  {'Agree':>6}  {'Acc%':>7s}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 6}  {'─' * 8}  {'─' * 6}  {'─' * 7}")

    for a in sorted(labels_a):
        count_a = sum(1 for rid in common_ids if data_a[rid] == a)
        # Which B label maps to this A?
        b_mapped = None
        for b, mapped_a in alignment.items():
            if mapped_a == a:
                b_mapped = b
                break

        if b_mapped is not None:
            count_b_in_a = sum(1 for rid in common_ids
                               if data_a[rid] == a and data_b[rid] == b_mapped)
            agree_in_a = sum(1 for rid in common_ids
                             if data_a[rid] == a and alignment.get(data_b[rid]) == a)
            acc = agree_in_a / count_a * 100 if count_a else 0
        else:
            b_mapped = "—"
            count_b_in_a = 0
            acc = 0

        print(f"  {a:>8}  {str(b_mapped):>8}  {count_a:>6}  {count_a/total*100:>7.1f}%  "
              f"{count_b_in_a:>6}  {acc:>6.1f}%")

    # ---- Confusion matrix (raw) ----
    if args.show_confusion:
        print(f"\n{'─' * 60}")
        print(f"  Raw confusion matrix (rows=A, cols=B)")
        print(f"{'─' * 60}")
        header = "A\\B  " + "  ".join(f"{b:>6}" for b in sorted(labels_b))
        print(header)
        for a in sorted(labels_a):
            row = f"{a:>4}  "
            for b in sorted(labels_b):
                row += f"{confusion.get((a, b), 0):>6}  "
            print(row)

    # ---- Aligned confusion matrix ----
    print(f"\n{'─' * 60}")
    print(f"  Aligned confusion matrix (rows=A, cols=B remapped to A)")
    print(f"{'─' * 60}")
    # Build reverse mapping: a -> b (which b maps to this a)
    rev_map = {a: [] for a in labels_a}
    for b, a in alignment.items():
        if a is not None:
            rev_map[a].append(b)

    header = "A\\A'" + "".join(f"{a2:>6}" for a2 in sorted(labels_a))
    print(header)
    for a1 in sorted(labels_a):
        row = f"{a1:>4}  "
        for a2 in sorted(labels_a):
            # Count records where A=a1 AND aligned_B=a2
            cnt = 0
            for rid in common_ids:
                if data_a[rid] == a1:
                    b_label = data_b[rid]
                    if alignment.get(b_label) == a2:
                        cnt += 1
            row += f"{cnt:>6}  "
        print(row)

    # ---- Show mismatches ----
    if args.show_mismatches > 0 and mismatches:
        print(f"\n{'─' * 60}")
        print(f"  Sample mismatches (showing {min(args.show_mismatches, len(mismatches))})")
        print(f"{'─' * 60}")
        print(f"  {'record_id':16s}  {'A':>4}  {'B':>4}  {'B→A':>6}")
        print(f"  {'─' * 16}  {'─' * 4}  {'─' * 4}  {'─' * 6}")
        for rid, a_label, b_label, mapped_b in mismatches[: args.show_mismatches]:
            print(f"  {rid:16s}  {a_label:>4}  {b_label:>4}  {mapped_b:>6}")

    # ---- Quick copy-paste alignment mapping ----
    print(f"\n{'─' * 60}")
    print(f"  Mapping to remap B labels to A labels")
    print(f"{'─' * 60}")
    for b in sorted(alignment):
        a = alignment[b]
        print(f"  b_label == {b}  →  a_label = {a}")


if __name__ == "__main__":
    main()
