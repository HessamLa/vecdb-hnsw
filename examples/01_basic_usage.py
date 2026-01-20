#!/usr/bin/env python3
"""
Example 1: Basic VecDB Usage

Demonstrates creating a database, inserting vectors, and searching.
"""

from vecdb import VecDB

# Create a database (uses context manager for automatic save)
with VecDB('./example_db') as db:
    # Create a collection for 4-dimensional vectors
    collection = db.create_collection(
        name='my_vectors',
        dimension=4,
        metric='l2'  # Euclidean distance
    )

    # Insert some vectors with user IDs
    collection.insert(user_id=1, vector=[1.0, 0.0, 0.0, 0.0])
    collection.insert(user_id=2, vector=[0.0, 1.0, 0.0, 0.0])
    collection.insert(user_id=3, vector=[0.0, 0.0, 1.0, 0.0])
    collection.insert(user_id=4, vector=[0.5, 0.5, 0.0, 0.0])
    collection.insert(user_id=5, vector=[0.7, 0.7, 0.1, 0.0])

    print(f"Inserted {collection.count()} vectors")

    # Search for similar vectors
    query = [0.6, 0.6, 0.0, 0.0]
    results = collection.search(query, k=3)

    print(f"\nSearching for vectors similar to {query}")
    print("Results (id, distance):")
    for user_id, distance in results:
        vec = collection.get(user_id)
        print(f"  ID {user_id}: distance={distance:.4f}, vector={vec}")

# Database is automatically saved when exiting the context

# Reopen to verify persistence
print("\n--- Reopening database ---")
db = VecDB('./example_db')
collection = db.get_collection('my_vectors')
print(f"Collection still has {collection.count()} vectors")

# Clean up
import shutil
shutil.rmtree('./example_db')
print("\nExample complete!")
