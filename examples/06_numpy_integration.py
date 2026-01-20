#!/usr/bin/env python3
"""
Example 6: NumPy Integration

Demonstrates using VecDB with NumPy arrays for efficient vector operations.
"""

import numpy as np
from vecdb import VecDB
import shutil


def main():
    print("NumPy Integration Demo")
    print("=" * 50)

    with VecDB('./numpy_db') as db:
        col = db.create_collection('vectors', dimension=128, metric='l2')

        # Generate random vectors using NumPy
        print("\n1. Generating vectors with NumPy")
        np.random.seed(42)
        num_vectors = 1000
        vectors = np.random.randn(num_vectors, 128).astype(np.float32)
        print(f"   Generated {num_vectors} vectors of shape {vectors.shape}")

        # Insert NumPy arrays directly
        print("\n2. Inserting NumPy arrays")
        for i, vec in enumerate(vectors):
            col.insert(i, vec)  # NumPy array accepted directly
        print(f"   Inserted {col.count()} vectors")

        # Search with NumPy array
        print("\n3. Searching with NumPy query")
        query = np.random.randn(128).astype(np.float32)
        results = col.search(query, k=5)  # NumPy array accepted
        print(f"   Query shape: {query.shape}")
        print("   Results:")
        for id, dist in results:
            print(f"     ID {id}: distance={dist:.4f}")

        # Retrieve and convert to NumPy
        print("\n4. Converting results to NumPy")
        retrieved_vec = col.get(results[0][0])
        np_vec = np.array(retrieved_vec)
        print(f"   Retrieved vector type: {type(retrieved_vec)}")
        print(f"   As NumPy array shape: {np_vec.shape}")

        # Batch operations with NumPy
        print("\n5. Batch operations")

        # Find centroid of top-k results
        top_k_vectors = []
        for id, dist in results:
            top_k_vectors.append(col.get(id))

        centroid = np.mean(top_k_vectors, axis=0)
        print(f"   Centroid of top-5 results: shape={centroid.shape}")

        # Search for vectors near centroid
        centroid_results = col.search(centroid, k=3)
        print("   Vectors nearest to centroid:")
        for id, dist in centroid_results:
            print(f"     ID {id}: distance={dist:.4f}")

        # Normalize vectors example
        print("\n6. Working with normalized vectors (for cosine)")
        col_cosine = db.create_collection('normalized', dimension=128, metric='cosine')

        # Insert normalized vectors
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        for i, vec in enumerate(normalized[:100]):
            col_cosine.insert(i, vec)

        # Search
        query_normalized = query / np.linalg.norm(query)
        cos_results = col_cosine.search(query_normalized, k=3)
        print("   Cosine search results:")
        for id, dist in cos_results:
            similarity = 1 - dist
            print(f"     ID {id}: similarity={similarity:.4f}")

    # Cleanup
    shutil.rmtree('./numpy_db')
    print("\nDemo complete!")


if __name__ == '__main__':
    main()
