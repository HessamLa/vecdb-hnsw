#!/usr/bin/env python3
"""
Example 2: Distance Metrics

Demonstrates the three distance metrics: L2, Cosine, and Dot Product.
"""

from vecdb import VecDB
import shutil

def demo_l2():
    """L2 (Euclidean) distance - measures geometric distance."""
    print("=" * 50)
    print("L2 (Euclidean) Distance")
    print("=" * 50)

    with VecDB('./demo_l2') as db:
        col = db.create_collection('vectors', dimension=2, metric='l2')

        # Insert points on a 2D plane
        col.insert(1, [0.0, 0.0])  # Origin
        col.insert(2, [1.0, 0.0])  # 1 unit right
        col.insert(3, [3.0, 4.0])  # 5 units from origin (3-4-5 triangle)

        # Search from origin
        results = col.search([0.0, 0.0], k=3)
        print("Search from origin [0, 0]:")
        for id, dist in results:
            print(f"  ID {id}: distance = {dist:.2f}")

    shutil.rmtree('./demo_l2')


def demo_cosine():
    """Cosine distance - measures angular similarity (direction matters, magnitude doesn't)."""
    print("\n" + "=" * 50)
    print("Cosine Distance")
    print("=" * 50)

    with VecDB('./demo_cosine') as db:
        col = db.create_collection('vectors', dimension=2, metric='cosine')

        # Insert vectors with different directions
        col.insert(1, [1.0, 0.0])   # Pointing right
        col.insert(2, [10.0, 0.0])  # Same direction, 10x magnitude
        col.insert(3, [0.0, 1.0])   # Pointing up (perpendicular)
        col.insert(4, [-1.0, 0.0])  # Pointing left (opposite)
        col.insert(5, [1.0, 1.0])   # 45 degrees

        # Search for vectors similar to [1, 0]
        results = col.search([1.0, 0.0], k=5)
        print("Search for direction [1, 0] (pointing right):")
        for id, dist in results:
            vec = col.get(id)
            print(f"  ID {id}: cosine_dist = {dist:.4f}, vector = {vec}")

        print("\nNote: IDs 1 and 2 have same distance (same direction)")
        print("ID 3 (perpendicular) has distance 1.0")
        print("ID 4 (opposite) has distance 2.0")

    shutil.rmtree('./demo_cosine')


def demo_dot():
    """Dot product distance - for Maximum Inner Product Search (MIPS)."""
    print("\n" + "=" * 50)
    print("Dot Product Distance (MIPS)")
    print("=" * 50)

    with VecDB('./demo_dot') as db:
        col = db.create_collection('vectors', dimension=2, metric='dot')

        # Insert vectors - higher dot product = lower distance
        col.insert(1, [1.0, 1.0])   # dot with query = 2
        col.insert(2, [2.0, 2.0])   # dot with query = 4
        col.insert(3, [3.0, 3.0])   # dot with query = 6
        col.insert(4, [0.5, 0.5])   # dot with query = 1
        col.insert(5, [-1.0, -1.0]) # dot with query = -2

        # Search - finds vectors with highest dot product
        results = col.search([1.0, 1.0], k=5)
        print("Search with query [1, 1]:")
        print("(Higher dot product = better match = lower distance)")
        for id, dist in results:
            vec = col.get(id)
            actual_dot = vec[0] * 1.0 + vec[1] * 1.0
            print(f"  ID {id}: dist = {dist:.2f}, dot_product = {actual_dot:.1f}, vector = {vec}")

    shutil.rmtree('./demo_dot')


if __name__ == '__main__':
    demo_l2()
    demo_cosine()
    demo_dot()
    print("\nAll metric examples complete!")
