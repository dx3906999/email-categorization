"""
Traditional clustering pipeline for 103 open LLM labels → 5 balanced categories.

Pipeline:
  1. Vectorize: embed each label's representative emails with Chinese BERT
  2. Reduce: PCA to 64 dims
  3. Cluster: constrained k-means (k=5) with size balancing
  4. Evaluate: silhouette score + size balance

Contrast with the keyword-priority method.
"""

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer


def load_emails_by_label(path: str) -> tuple[dict, dict, list]:
    """Load emails grouped by LLM label.

    Returns:
        label_emails: label → list of email dicts
        label_counts: label → count
        all_labels: list of unique labels sorted by count desc
    """
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f
                   if line.strip() and not json.loads(line.strip()).get("error")]

    label_emails = {}
    for r in records:
        cat = r["llm_category"]
        if cat not in label_emails:
            label_emails[cat] = []
        label_emails[cat].append(r)

    label_counts = {k: len(v) for k, v in label_emails.items()}
    all_labels = sorted(label_counts, key=label_counts.get, reverse=True)
    return label_emails, label_counts, all_labels


def embed_labels(label_emails: dict, all_labels: list,
                 model_name: str = "shibing624/text2vec-base-chinese",
                 max_emails_per_label: int = 50) -> np.ndarray:
    """Compute a 768-dim embedding for each label by averaging its emails.

    To save memory/time, sample up to max_emails_per_label emails per label.
    """
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {dim}")

    embeddings = np.zeros((len(all_labels), dim), dtype=np.float32)

    for i, label in enumerate(all_labels):
        emails = label_emails[label]
        # Sample if too many
        if len(emails) > max_emails_per_label:
            rng = np.random.RandomState(42 + i)
            emails = rng.choice(emails, max_emails_per_label, replace=False)

        # Concatenate subject + content
        texts = [(e.get("subject", "") + " " + e.get("content", ""))[:300]
                 for e in emails]

        # Batch encode
        embs = model.encode(texts, show_progress_bar=False, batch_size=32)
        embeddings[i] = embs.mean(axis=0)

        if (i + 1) % 20 == 0:
            print(f"  Embedded {i+1}/{len(all_labels)} labels")

    print(f"  Done: {len(all_labels)} label embeddings ({dim} dim)")
    return embeddings


def constrained_kmeans(X: np.ndarray, weights: np.ndarray, k: int,
                        max_iter: int = 100, random_state: int = 42) -> np.ndarray:
    """K-means with cluster size balancing via iterative reassignment.

    Algorithm:
      1. Run standard k-means
      2. Find oversized/undersized clusters
      3. Reassign boundary points from oversized to undersized clusters
      4. Repeat until balanced or max_iter

    Args:
        X: (n, d) data matrix
        weights: (n,) weight of each point (email counts for labels)
        k: number of clusters
        max_iter: max balancing iterations
        random_state: seed
    Returns:
        labels: (n,) cluster assignments (0..k-1)
    """
    n = X.shape[0]
    target = weights.sum() / k  # ideal size per cluster

    # Step 1: standard k-means
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # Step 2: iterative balancing
    for iteration in range(max_iter):
        # Compute current cluster sizes
        cluster_sizes = np.array([weights[labels == j].sum() for j in range(k)])
        imbalance = cluster_sizes.max() - cluster_sizes.min()

        if imbalance / target < 0.3:  # Converged: within 30% of target
            print(f"  Balancing converged at iter {iteration}: "
                  f"range={imbalance:.0f}, target={target:.0f}")
            break

        # Find most oversized and undersized clusters
        over_idx = np.argmax(cluster_sizes)
        under_idx = np.argmin(cluster_sizes)

        # Find points currently in over_idx that are closest to under_idx center
        over_points = np.where(labels == over_idx)[0]
        if len(over_points) == 0:
            break

        # Score each point: distance to under center - distance to over center
        dist_to_over = np.linalg.norm(X[over_points] - centers[over_idx], axis=1)
        dist_to_under = np.linalg.norm(X[over_points] - centers[under_idx], axis=1)
        scores = dist_to_under - dist_to_over  # negative = closer to under

        # Move the best candidate (most natural fit for under_idx)
        best_candidate_idx = over_points[np.argmin(scores)]
        labels[best_candidate_idx] = under_idx

        # Update centers
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centers[j] = (X[mask] * weights[mask, np.newaxis]).sum(axis=0) / weights[mask].sum()

    return labels


def evaluate(labels: np.ndarray, X: np.ndarray, weights: np.ndarray,
             all_labels: list, k: int):
    """Print cluster quality and balance metrics."""
    total = weights.sum()
    target = total / k

    print(f"\n{'='*60}")
    print(f"  Clustering Result (k={k})")
    print(f"{'='*60}")

    # Silhouette score
    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
        print(f"\n  Silhouette score: {sil:.4f}  (-1=bad, 1=good)")

    # Cluster sizes
    cluster_info = []
    for j in range(k):
        mask = labels == j
        size = int(weights[mask].sum())
        pct = size / total * 100
        n_labels = mask.sum()
        # Top labels in this cluster
        indices = np.where(mask)[0]
        top_indices = sorted(indices, key=lambda i: weights[i], reverse=True)[:6]
        top_names = [all_labels[i] for i in top_indices]
        cluster_info.append((j, size, pct, n_labels, top_names))

    cluster_info.sort(key=lambda x: -x[1])

    print(f"\n  {'Cluster':5s}  {'Size':>5}  {'%':>6s}  {'#labels':>7s}  {'Top labels'}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*50}")
    sizes = []
    for j, size, pct, n_lbls, top_names in cluster_info:
        sizes.append(size)
        bar = "█" * max(1, int(size / target * 10))
        top_str = " | ".join(top_names[:4])
        print(f"  C{j:<4d}  {size:>5}  {pct:>5.1f}%  {n_lbls:>7d}  {top_str}")

    # Balance metrics
    rg = max(sizes) - min(sizes)
    sd = statistics.stdev(sizes)
    print(f"\n  Target: {target:.0f}/cluster")
    print(f"  Range: {rg}  StdDev: {sd:.0f}")
    print(f"  CV: {sd/target*100:.1f}%  (coefficient of variation, lower=more balanced)")

    # Per-cluster semantic coherence
    print(f"\n  ── Full label assignments ──")
    for j, size, pct, n_lbls, top_names in cluster_info:
        indices = np.where(labels == j)[0]
        all_members = sorted(indices, key=lambda i: weights[i], reverse=True)
        member_names = [f"{all_labels[i]}({int(weights[i])})" for i in all_members]
        print(f"\n  C{j} [{pct:.0f}%, {n_lbls} labels, {size} emails]:")
        # Show all if few, or top 15
        show = member_names if len(member_names) <= 15 else member_names[:15] + [f"...+{len(member_names)-15} more"]
        print(f"    {', '.join(show)}")


def main():
    path = Path("data/open_sample_500_labeled.jsonl")
    if not path.exists():
        print(f"File not found: {path}")
        return

    # ---- Step 1: Load ----
    print("Step 1: Loading emails by label...")
    label_emails, label_counts, all_labels = load_emails_by_label(str(path))
    total = sum(label_counts.values())
    n_labels = len(all_labels)
    print(f"  {total} emails, {n_labels} unique labels, target {total//5}/cluster")
    weights = np.array([label_counts[l] for l in all_labels], dtype=np.float32)

    # ---- Step 2: Embed ----
    print(f"\nStep 2: Embedding labels...")
    X = embed_labels(label_emails, all_labels)
    # X = embed_labels(label_emails, all_labels,
    #                  model_name="BAAI/bge-small-zh-v1.5")
    # Alternative models: "shibing624/text2vec-base-chinese" (good for Chinese)
    #                    "BAAI/bge-small-zh-v1.5" (faster, 512dim)

    # ---- Step 3: PCA ----
    print(f"\nStep 3: PCA dimension reduction...")
    pca_dim = min(64, n_labels - 1, X.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  {X.shape[1]} → {pca_dim} dims, explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # ---- Step 4: Cluster ----
    print(f"\nStep 4: Constrained k-means (k=5)...")
    labels = constrained_kmeans(X_pca, weights, k=5)

    # ---- Step 5: Evaluate ----
    evaluate(labels, X_pca, weights, all_labels, k=5)

    # ---- Bonus: compare with standard k-means ----
    print(f"\n\n{'='*60}")
    print(f"  Comparison: Standard k-means (no balance constraint)")
    print(f"{'='*60}")
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels_std = km.fit_predict(X_pca)
    evaluate(labels_std, X_pca, weights, all_labels, k=5)


if __name__ == "__main__":
    main()
