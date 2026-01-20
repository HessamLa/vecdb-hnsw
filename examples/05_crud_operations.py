#!/usr/bin/env python3
"""
Example 5: CRUD Operations

Demonstrates Create, Read, Update, Delete operations on vectors.
"""

from vecdb import VecDB, DuplicateIDError
import shutil


def main():
    print("CRUD Operations Demo")
    print("=" * 50)

    with VecDB('./crud_db') as db:
        col = db.create_collection('vectors', dimension=3, metric='l2')

        # CREATE - Insert vectors
        print("\n1. CREATE - Inserting vectors")
        col.insert(1, [1.0, 0.0, 0.0])
        col.insert(2, [0.0, 1.0, 0.0])
        col.insert(3, [0.0, 0.0, 1.0])
        print(f"   Inserted 3 vectors. Count: {col.count()}")

        # Try duplicate insert
        print("\n   Trying duplicate ID...")
        try:
            col.insert(1, [9.9, 9.9, 9.9])
        except DuplicateIDError as e:
            print(f"   Caught expected error: {e}")

        # READ - Get vectors
        print("\n2. READ - Retrieving vectors")
        for id in [1, 2, 3, 999]:
            vec = col.get(id)
            if vec:
                print(f"   ID {id}: {vec}")
            else:
                print(f"   ID {id}: Not found")

        # Check existence
        print("\n   Checking existence:")
        print(f"   ID 1 exists: {col.contains(1)}")
        print(f"   ID 999 exists: {col.contains(999)}")
        print(f"   Using 'in': {1 in col}, {999 in col}")

        # SEARCH - Find similar vectors
        print("\n3. SEARCH - Finding similar vectors")
        query = [0.5, 0.5, 0.0]
        results = col.search(query, k=3)
        print(f"   Query: {query}")
        for id, dist in results:
            print(f"   ID {id}: distance={dist:.4f}")

        # DELETE - Remove vectors
        print("\n4. DELETE - Removing vectors")
        print(f"   Before delete: count={col.count()}")

        deleted = col.delete(2)
        print(f"   Deleted ID 2: {deleted}")
        print(f"   After delete: count={col.count()}")

        # Verify deletion
        print(f"   Get ID 2: {col.get(2)}")
        print(f"   ID 2 in collection: {2 in col}")

        # Search no longer returns deleted vector
        results = col.search(query, k=3)
        print(f"\n   Search after delete (ID 2 not in results):")
        for id, dist in results:
            print(f"   ID {id}: distance={dist:.4f}")

        # Delete non-existent
        deleted = col.delete(999)
        print(f"\n   Delete non-existent ID 999: {deleted}")

        # "UPDATE" - Delete and re-insert (VecDB doesn't have update)
        print("\n5. UPDATE (via delete + insert)")
        print(f"   Original ID 1: {col.get(1)}")
        col.delete(1)
        col.insert(1, [0.9, 0.9, 0.9])  # New vector for same ID
        print(f"   Updated ID 1: {col.get(1)}")

    # Cleanup
    shutil.rmtree('./crud_db')
    print("\nDemo complete!")


if __name__ == '__main__':
    main()
