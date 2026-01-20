#!/usr/bin/env python3
"""
VecDB Benchmarks

Measures insert throughput, search latency, and recall for the HNSW implementation.
"""

import time
import numpy as np
from typing import Tuple, List

# Force mock import for comparison
import sys


def get_mock_index():
    """Get mock HNSW index class."""
    from vecdb._hnsw_mock import HNSWIndex
    return HNSWIndex


def get_cpp_index():
    """Get C++ HNSW index class."""
    from vecdb._hnsw_cpp import HNSWIndex
    return HNSWIndex


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random normalized vectors."""
    np.random.seed(seed)
    vectors = np.random.randn(n, dim).astype(np.float32)
    # Normalize for cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors


def brute_force_search(vectors: np.ndarray, query: np.ndarray, k: int) -> List[int]:
    """Brute force k-NN search."""
    distances = np.linalg.norm(vectors - query, axis=1)
    return np.argsort(distances)[:k].tolist()


def benchmark_insert(IndexClass, vectors: np.ndarray, name: str) -> Tuple[float, float]:
    """Benchmark insert throughput."""
    n, dim = vectors.shape
    index = IndexClass(dim, 'l2')

    start = time.perf_counter()
    for i, vec in enumerate(vectors):
        index.add(i, vec.tolist())
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"[{name}] Insert {n} vectors: {elapsed:.3f}s ({throughput:.0f} vec/s)")
    return elapsed, throughput


def benchmark_search(IndexClass, vectors: np.ndarray, n_queries: int, k: int, name: str) -> Tuple[float, float]:
    """Benchmark search latency."""
    n, dim = vectors.shape
    index = IndexClass(dim, 'l2')

    for i, vec in enumerate(vectors):
        index.add(i, vec.tolist())

    queries = generate_vectors(n_queries, dim, seed=123)

    start = time.perf_counter()
    for query in queries:
        index.search(query.tolist(), k=k)
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / n_queries) * 1000
    print(f"[{name}] Search {n_queries} queries (k={k}): {elapsed:.3f}s ({avg_latency_ms:.2f} ms/query)")
    return elapsed, avg_latency_ms


def benchmark_recall(IndexClass, vectors: np.ndarray, n_queries: int, k: int, name: str) -> float:
    """Benchmark recall@k vs brute force."""
    n, dim = vectors.shape
    index = IndexClass(dim, 'l2')

    for i, vec in enumerate(vectors):
        index.add(i, vec.tolist())

    queries = generate_vectors(n_queries, dim, seed=123)

    total_recall = 0.0
    for query in queries:
        # HNSW results
        hnsw_results = index.search(query.tolist(), k=k)
        hnsw_ids = set(r[0] for r in hnsw_results)

        # Brute force ground truth
        bf_ids = set(brute_force_search(vectors, query, k))

        # Recall
        recall = len(hnsw_ids & bf_ids) / k
        total_recall += recall

    avg_recall = total_recall / n_queries
    print(f"[{name}] Recall@{k}: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    return avg_recall


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 60)
    print("VecDB Benchmark Suite")
    print("=" * 60)

    # Test sizes
    sizes = [1000, 10000]
    dim = 128
    k = 10
    n_queries = 100

    MockIndex = get_mock_index()
    try:
        CppIndex = get_cpp_index()
        has_cpp = True
    except ImportError:
        has_cpp = False
        print("Warning: C++ HNSW not available, benchmarking mock only")

    results = []

    for n in sizes:
        print(f"\n{'='*60}")
        print(f"Dataset: {n} vectors, {dim} dimensions")
        print("=" * 60)

        vectors = generate_vectors(n, dim)

        # Insert benchmarks
        print("\n--- Insert Throughput ---")
        mock_insert = benchmark_insert(MockIndex, vectors, "Mock")
        if has_cpp:
            cpp_insert = benchmark_insert(CppIndex, vectors, "C++")
            speedup = mock_insert[0] / cpp_insert[0]
            print(f"C++ speedup: {speedup:.2f}x")

        # Search benchmarks
        print("\n--- Search Latency ---")
        mock_search = benchmark_search(MockIndex, vectors, n_queries, k, "Mock")
        if has_cpp:
            cpp_search = benchmark_search(CppIndex, vectors, n_queries, k, "C++")
            speedup = mock_search[0] / cpp_search[0]
            print(f"C++ speedup: {speedup:.2f}x")

        # Recall benchmarks
        print("\n--- Recall@10 ---")
        mock_recall = benchmark_recall(MockIndex, vectors, n_queries, k, "Mock")
        if has_cpp:
            cpp_recall = benchmark_recall(CppIndex, vectors, n_queries, k, "C++")

        results.append({
            'n': n,
            'dim': dim,
            'mock_insert_throughput': mock_insert[1],
            'cpp_insert_throughput': cpp_insert[1] if has_cpp else None,
            'mock_search_latency_ms': mock_search[1],
            'cpp_search_latency_ms': cpp_search[1] if has_cpp else None,
            'mock_recall': mock_recall,
            'cpp_recall': cpp_recall if has_cpp else None,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'N':>10} | {'Insert (vec/s)':>20} | {'Search (ms)':>15} | {'Recall@10':>10}")
    print(f"{'':>10} | {'Mock':>10} {'C++':>10} | {'Mock':>7} {'C++':>7} | {'Mock':>5} {'C++':>5}")
    print("-" * 60)
    for r in results:
        cpp_ins = f"{r['cpp_insert_throughput']:.0f}" if r['cpp_insert_throughput'] else "N/A"
        cpp_lat = f"{r['cpp_search_latency_ms']:.2f}" if r['cpp_search_latency_ms'] else "N/A"
        cpp_rec = f"{r['cpp_recall']:.2f}" if r['cpp_recall'] else "N/A"
        print(f"{r['n']:>10} | {r['mock_insert_throughput']:>10.0f} {cpp_ins:>10} | "
              f"{r['mock_search_latency_ms']:>7.2f} {cpp_lat:>7} | "
              f"{r['mock_recall']:>5.2f} {cpp_rec:>5}")


if __name__ == '__main__':
    run_benchmarks()
