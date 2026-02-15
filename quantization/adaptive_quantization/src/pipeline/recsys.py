import numpy as np
from typing import List, Dict, Tuple


class ProductRecommender:
    def __init__(self):
        # Product catalog
        self.products = [
            {"id": 0, "name": "Wireless Headphones", "category": "Electronics", "price": "$89"},
            {"id": 1, "name": "Running Shoes", "category": "Sports", "price": "$120"},
            {"id": 2, "name": "Coffee Maker", "category": "Kitchen", "price": "$65"},
            {"id": 3, "name": "Yoga Mat", "category": "Sports", "price": "$35"},
            {"id": 4, "name": "Smart Watch", "category": "Electronics", "price": "$250"},
            {"id": 5, "name": "Blender", "category": "Kitchen", "price": "$55"},
            {"id": 6, "name": "Laptop Stand", "category": "Electronics", "price": "$45"},
            {"id": 7, "name": "Dumbbell Set", "category": "Sports", "price": "$80"},
            {"id": 8, "name": "Air Fryer", "category": "Kitchen", "price": "$95"},
            {"id": 9, "name": "Bluetooth Speaker", "category": "Electronics", "price": "$70"}
        ]

        # Generate random embeddings for products (8 dimensions)
        np.random.seed(42)
        self.product_embeddings = {
            product['id']: np.random.randn(8) for product in self.products
        }

        # Customer profiles with purchase history
        self.customers = [
            {"id": 0, "name": "Alex (Tech Enthusiast)", "purchased": [0, 4, 6]},
            {"id": 1, "name": "Sarah (Fitness Lover)", "purchased": [1, 3, 7]},
            {"id": 2, "name": "Mike (Home Chef)", "purchased": [2, 5, 8]}
        ]

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b)

    def get_customer_embedding(self, purchased_ids: List[int]) -> np.ndarray:
        """Generate customer embedding from purchase history (average of purchased products)."""
        purchased_embeddings = [self.product_embeddings[pid] for pid in purchased_ids]
        return np.mean(purchased_embeddings, axis=0)

    def get_recommendations(self, customer_id: int, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Get top-k product recommendations for a customer."""
        customer = self.customers[customer_id]
        customer_embedding = self.get_customer_embedding(customer['purchased'])

        # Calculate similarity for products not yet purchased
        recommendations = []
        for product in self.products:
            if product['id'] not in customer['purchased']:
                product_embedding = self.product_embeddings[product['id']]
                similarity = self.cosine_similarity(customer_embedding, product_embedding)
                recommendations.append((product, similarity))

        # Sort by similarity (descending) and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

    def display_customer_profile(self, customer_id: int):
        """Display customer profile and purchase history."""
        customer = self.customers[customer_id]
        print(f"\n{'=' * 70}")
        print(f"CUSTOMER PROFILE: {customer['name']}")
        print(f"{'=' * 70}")
        print(f"\nPurchase History:")
        for pid in customer['purchased']:
            product = next(p for p in self.products if p['id'] == pid)
            print(f"  • {product['name']} ({product['category']}) - {product['price']}")

        print(f"\nCustomer Embedding Vector:")
        customer_emb = self.get_customer_embedding(customer['purchased'])
        print(f"  {np.array2string(customer_emb, precision=4, suppress_small=True)}")

    def display_recommendations(self, customer_id: int, top_k: int = 5):
        """Display recommendations for a customer."""
        recommendations = self.get_recommendations(customer_id, top_k)

        print(f"\n{'=' * 70}")
        print(f"TOP {top_k} RECOMMENDATIONS")
        print(f"{'=' * 70}\n")

        for rank, (product, similarity) in enumerate(recommendations, 1):
            print(f"Rank #{rank}")
            print(f"  Product: {product['name']}")
            print(f"  Category: {product['category']}")
            print(f"  Price: {product['price']}")
            print(f"  Match Score: {similarity * 100:.2f}%")
            print(f"  Cosine Similarity: {similarity:.4f}")

            # Show product embedding
            product_emb = self.product_embeddings[product['id']]
            print(f"  Embedding: {np.array2string(product_emb, precision=4, suppress_small=True)}")
            print()

    def run_demo(self):
        """Run a demo for all customers."""
        print("\n" + "=" * 70)
        print("PRODUCT RECOMMENDATION SYSTEM - HACKATHON DEMO")
        print("=" * 70)
        print("\nAlgorithm: Cosine Similarity based Collaborative Filtering")
        print("Embedding Dimensions: 8")
        print(f"Total Products: {len(self.products)}")
        print(f"Total Customers: {len(self.customers)}")

        for customer in self.customers:
            self.display_customer_profile(customer['id'])
            self.display_recommendations(customer['id'], top_k=5)
            print("\n" + "-" * 70 + "\n")


def main():
    # Initialize recommender system
    recommender = ProductRecommender()

    # Run demo for all customers
    recommender.run_demo()

    # Optional: Get recommendations for a specific customer
    print("\n" + "=" * 70)
    print("CUSTOM QUERY EXAMPLE")
    print("=" * 70)
    print("\nGetting recommendations for Customer #1 (Sarah - Fitness Lover)...")
    recommender.display_customer_profile(1)
    recommender.display_recommendations(1, top_k=3)


if __name__ == "__main__":
    main()