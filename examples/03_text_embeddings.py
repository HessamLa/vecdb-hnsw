#!/usr/bin/env python3
"""
Example 3: Text Embeddings (Simulated)

Demonstrates a realistic use case: storing and searching text embeddings.
In production, you'd use a real embedding model (OpenAI, Sentence Transformers, etc.)
"""

import numpy as np
from vecdb import VecDB
import shutil

# Simulated documents
DOCUMENTS = {
    1: "The quick brown fox jumps over the lazy dog",
    2: "A fast auburn fox leaps above a sleepy canine",
    3: "Machine learning is transforming the technology industry",
    4: "Deep learning neural networks are revolutionizing AI",
    5: "The weather forecast predicts sunny skies tomorrow",
    6: "Climate predictions indicate warm temperatures ahead",
    7: "Python is a popular programming language",
    8: "JavaScript dominates web development",
}


def fake_embedding(text: str, dim: int = 64) -> list:
    """
    Generate a fake embedding for demonstration.
    In production, use a real embedding model!

    This creates deterministic pseudo-embeddings where similar
    texts have similar vectors (using hash-based seeding).
    """
    # Use words to create a deterministic seed
    words = text.lower().split()
    seed = sum(hash(w) for w in words) % (2**32)
    np.random.seed(seed)

    # Base embedding from hash
    embedding = np.random.randn(dim).astype(np.float32)

    # Add some word-based features to make similar docs closer
    for word in words:
        word_seed = hash(word) % (2**32)
        np.random.seed(word_seed)
        embedding += 0.1 * np.random.randn(dim)

    # Normalize for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def main():
    print("Text Embedding Search Demo")
    print("=" * 50)

    # Create database with cosine metric (standard for embeddings)
    with VecDB('./embeddings_db') as db:
        collection = db.create_collection(
            name='documents',
            dimension=64,
            metric='cosine'
        )

        # Index all documents
        print("\nIndexing documents...")
        for doc_id, text in DOCUMENTS.items():
            embedding = fake_embedding(text)
            collection.insert(doc_id, embedding)
            print(f"  [{doc_id}] {text[:50]}...")

        print(f"\nIndexed {collection.count()} documents")

        # Search queries
        queries = [
            "fox jumping over dog",
            "artificial intelligence and machine learning",
            "weather and temperature",
            "programming languages",
        ]

        for query in queries:
            print(f"\n{'='*50}")
            print(f"Query: '{query}'")
            print("-" * 50)

            query_embedding = fake_embedding(query)
            results = collection.search(query_embedding, k=3)

            for rank, (doc_id, distance) in enumerate(results, 1):
                similarity = 1 - distance  # Convert distance to similarity
                print(f"  {rank}. [ID {doc_id}] (sim: {similarity:.3f})")
                print(f"     {DOCUMENTS[doc_id]}")

    # Cleanup
    shutil.rmtree('./embeddings_db')
    print("\n" + "=" * 50)
    print("Demo complete!")


if __name__ == '__main__':
    main()
