"""
Cluster 103 open LLM labels into 5 balanced, semantically coherent categories.

Algorithm: Balanced agglomerative clustering with character-bigram Jaccard similarity.
The balance parameter β controls the trade-off: β=0 means pure semantic clustering,
β=2 means heavy enforcement of equal-sized clusters.
"""

import json
import itertools
import statistics
from collections import Counter
from pathlib import Path


def char_bigrams(text: str) -> set:
    """Extract character bigrams for Jaccard similarity."""
    chars = list(text)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)} | set(chars)


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_similarity_matrix(labels: list[str]) -> dict:
    """Compute pairwise similarities between all label names."""
    bigrams = {label: char_bigrams(label) for label in labels}
    sim = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                sim[(a, b)] = jaccard(bigrams[a], bigrams[b])
    return sim


def balanced_agglomerative(labels: list[str], counts: dict,
                           sim: dict, target: float, beta: float = 1.0):
    """
    Agglomerative clustering with size balance constraint.
    Uses Union-Find to track cluster membership.
    """
    total = sum(counts.values())

    # Union-Find: each label starts as its own cluster
    parent = {label: label for label in labels}
    cluster_sizes = {label: counts[label] for label in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        # Merge rb into ra
        parent[rb] = ra
        cluster_sizes[ra] = cluster_sizes.get(ra, 0) + cluster_sizes.get(rb, 0)
        del cluster_sizes[rb]
        return ra

    # Get all unique pairs sorted by similarity
    pairs = sorted(sim.items(), key=lambda x: -x[1])

    while len(cluster_sizes) > 5:
        best_score = -float("inf")
        best_pair = None

        for (a, b), similarity in pairs:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue

            merged_size = cluster_sizes[ra] + cluster_sizes[rb]

            # Balance factor: penalize merges that make clusters too large
            balance_factor = 1.0 - beta * abs(merged_size / target - 1.0)
            if balance_factor <= 0:
                continue

            score = similarity * balance_factor
            if score > best_score:
                best_score = score
                best_pair = (a, b, ra, rb, similarity)

        if best_pair is None:
            # Fallback: merge two smallest clusters
            sorted_reps = sorted(cluster_sizes.keys(), key=lambda k: cluster_sizes[k])
            ra, rb = sorted_reps[0], sorted_reps[1]
            # Find any members to pass to union
            a = next(l for l in labels if find(l) == ra)
            b = next(l for l in labels if find(l) == rb)
            union(a, b)
            continue

        a, b, ra, rb, sim_val = best_pair
        union(a, b)

    # Collect final clusters
    clusters = {}
    for label in labels:
        r = find(label)
        if r not in clusters:
            clusters[r] = set()
        clusters[r].add(label)

    # Build cluster_sizes from final state
    final_sizes = {}
    for rep, members in clusters.items():
        final_sizes[rep] = sum(counts.get(m, 0) for m in members)

    return clusters, final_sizes


def name_cluster(members: set[str], counts: dict) -> str:
    """Pick the best name for a cluster: the label with max count."""
    return max(members, key=lambda m: counts.get(m, 0))


def main():
    # Load open labels
    path = Path("data/open_sample_500_labeled.jsonl")
    labels_data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r and not r.get("error"):
                labels_data.append(r)

    total = len(labels_data)
    counts = Counter(l["llm_category"] for l in labels_data)
    unique_labels = list(counts.keys())
    target = total / 5
    n_labels = len(unique_labels)

    print(f"Total: {total} emails, {n_labels} unique labels")
    print(f"Target per cluster: {target:.0f} (20%)")
    print()

    # Compute similarities
    print("Computing label similarity matrix...")
    sim = compute_similarity_matrix(unique_labels)
    print(f"  {len(sim)} pairwise similarities computed")

    # Run clustering with different β values
    for beta in [0.0, 0.5, 1.0, 1.5]:
        print(f"\n{'='*60}")
        print(f"  β = {beta}")
        if beta == 0:
            print(f"  (pure semantic, no balance constraint)")
        elif beta >= 1.5:
            print(f"  (strong balance enforcement)")
        else:
            print(f"  (moderate balance)")
        print(f"{'='*60}")

        clusters, cluster_sizes = balanced_agglomerative(
            unique_labels, counts, sim, target, beta=beta
        )

        # Sort clusters by size
        cluster_list = []
        for rep, members in clusters.items():
            size = cluster_sizes.get(rep, 0)
            name = name_cluster(members, counts)
            # Get top 5 labels in this cluster
            top_labels = sorted(members, key=lambda m: counts.get(m, 0), reverse=True)[:8]
            cluster_list.append((name, size, members, top_labels))

        cluster_list.sort(key=lambda x: -x[1])

        print(f"\n  {'Cluster':25s}  {'Size':>5}  {'Pct':>6s}  {'Top labels'}")
        print(f"  {'─'*25}  {'─'*5}  {'─'*6}  {'─'*40}")
        sizes = []
        for name, size, members, top_labels in cluster_list:
            pct = size / total * 100
            sizes.append(size)
            top_str = ", ".join(top_labels[:5])
            print(f"  {name:25s}  {size:>5}  {pct:>5.1f}%  {top_str}")

        # Metrics
        rg = max(sizes) - min(sizes)
        sd = statistics.stdev(sizes)
        print(f"\n  Range: {rg}  StdDev: {sd:.0f}")

        # Semantic cohesion: avg pairwise similarity within each cluster
        cohesion_scores = []
        for name, size, members, _ in cluster_list:
            if len(members) <= 1:
                continue
            pair_sims = []
            mem_list = list(members)
            for i in range(min(len(mem_list), 15)):
                for j in range(i + 1, min(len(mem_list), 15)):
                    a, b = mem_list[i], mem_list[j]
                    if (a, b) in sim:
                        pair_sims.append(sim[(a, b)])
                    elif (b, a) in sim:
                        pair_sims.append(sim[(b, a)])
            if pair_sims:
                cohesion_scores.append(sum(pair_sims) / len(pair_sims))
        avg_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0
        print(f"  Avg intra-cluster similarity: {avg_cohesion:.3f}")

    # Show top similar pairs for context
    print(f"\n{'='*60}")
    print(f"  Top 20 most similar label pairs")
    print(f"{'='*60}")
    for (a, b), s in sorted(sim.items(), key=lambda x: -x[1])[:20]:
        print(f"  {a:20s} <-> {b:20s}  sim={s:.3f}")


if __name__ == "__main__":
    main()
