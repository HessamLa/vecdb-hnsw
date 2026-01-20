#!/usr/bin/env python3
"""
Example 4: Multiple Collections

Demonstrates managing multiple collections with different configurations.
Useful for multi-tenant applications or different embedding models.
"""

import numpy as np
from vecdb import VecDB, CollectionExistsError, CollectionNotFoundError
import shutil


def main():
    print("Multiple Collections Demo")
    print("=" * 50)

    with VecDB('./multi_collection_db') as db:
        # Create collections for different use cases
        print("\nCreating collections...")

        # Collection for image embeddings (high dimensional)
        images = db.create_collection(
            name='images',
            dimension=512,
            metric='cosine'
        )
        print(f"  Created 'images' (dim={images.dimension}, metric={images.metric})")

        # Collection for text embeddings
        texts = db.create_collection(
            name='texts',
            dimension=384,
            metric='cosine'
        )
        print(f"  Created 'texts' (dim={texts.dimension}, metric={texts.metric})")

        # Collection for user preference vectors (for recommendations)
        preferences = db.create_collection(
            name='user_preferences',
            dimension=64,
            metric='dot'  # Dot product for recommendation scores
        )
        print(f"  Created 'user_preferences' (dim={preferences.dimension}, metric={preferences.metric})")

        # List all collections
        print(f"\nAll collections: {db.list_collections()}")

        # Try to create duplicate (will fail)
        print("\nTrying to create duplicate collection...")
        try:
            db.create_collection('images', dimension=512)
        except CollectionExistsError as e:
            print(f"  Caught expected error: {e}")

        # Insert data into each collection
        print("\nInserting data...")
        np.random.seed(42)

        # Add image embeddings
        for i in range(10):
            images.insert(i, np.random.randn(512).tolist())
        print(f"  Images: {images.count()} vectors")

        # Add text embeddings
        for i in range(20):
            texts.insert(i, np.random.randn(384).tolist())
        print(f"  Texts: {texts.count()} vectors")

        # Add user preferences
        for user_id in range(100, 105):
            preferences.insert(user_id, np.random.randn(64).tolist())
        print(f"  User preferences: {preferences.count()} vectors")

        # Search in specific collections
        print("\nSearching each collection...")

        img_results = images.search(np.random.randn(512).tolist(), k=3)
        print(f"  Images search: found {len(img_results)} results")

        txt_results = texts.search(np.random.randn(384).tolist(), k=5)
        print(f"  Texts search: found {len(txt_results)} results")

        pref_results = preferences.search(np.random.randn(64).tolist(), k=2)
        print(f"  Preferences search: found {len(pref_results)} results")

        # Delete a collection
        print("\nDeleting 'user_preferences' collection...")
        db.delete_collection('user_preferences')
        print(f"  Remaining collections: {db.list_collections()}")

        # Try to access deleted collection
        try:
            db.get_collection('user_preferences')
        except CollectionNotFoundError as e:
            print(f"  Caught expected error: {e}")

    # Verify persistence
    print("\n--- Reopening database ---")
    db = VecDB('./multi_collection_db')
    print(f"Collections after reopen: {db.list_collections()}")
    print(f"Images count: {db.get_collection('images').count()}")
    print(f"Texts count: {db.get_collection('texts').count()}")

    # Cleanup
    shutil.rmtree('./multi_collection_db')
    print("\nDemo complete!")


if __name__ == '__main__':
    main()
