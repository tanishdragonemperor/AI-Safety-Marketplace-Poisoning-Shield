"""
Search Pipeline for Marketplace Poisoning Shield
FAISS-based semantic search with embedding generation.
"""

import json
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from dataset_generator import Product


@dataclass
class SearchResult:
    """Represents a search result."""
    product: Product
    score: float
    rank: int
    is_poisoned: bool  # Ground truth: actually poisoned
    is_suspicious: bool  # Defense detection: flagged by defense system
    threat_score: float  # Defense threat score
    defense_applied: bool


class EmbeddingGenerator:
    """
    Generates text embeddings for semantic search.
    Uses a simple TF-IDF based approach for demo (can swap with sentence-transformers).
    """
    
    def __init__(self, use_neural: bool = False):
        """
        Initialize embedding generator.
        
        Args:
            use_neural: Whether to use neural embeddings (requires sentence-transformers)
        """
        self.use_neural = use_neural
        self.model = None
        self.tfidf_vectorizer = None
        self.embedding_dim = 384 if use_neural else 100
        
        if use_neural:
            self._init_neural_model()
        else:
            self._init_tfidf()
    
    def _init_neural_model(self):
        """Initialize neural embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            print("Loaded neural embedding model: all-MiniLM-L6-v2")
        except ImportError:
            print("sentence-transformers not available, falling back to TF-IDF")
            self.use_neural = False
            self._init_tfidf()
    
    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit the vectorizer on corpus (for TF-IDF)."""
        if not self.use_neural and self.tfidf_vectorizer:
            self.tfidf_vectorizer.fit(texts)
            self.is_fitted = True
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if self.use_neural and self.model:
            return self.model.encode(text, normalize_embeddings=True)
        elif self.tfidf_vectorizer and self.is_fitted:
            sparse = self.tfidf_vectorizer.transform([text])
            dense = sparse.toarray()[0]
            # Pad or truncate to embedding_dim
            if len(dense) < self.embedding_dim:
                dense = np.pad(dense, (0, self.embedding_dim - len(dense)))
            return dense / (np.linalg.norm(dense) + 1e-8)
        else:
            # Fallback: random embedding (for testing only)
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(self.embedding_dim)
            return emb / np.linalg.norm(emb)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if self.use_neural and self.model:
            return self.model.encode(texts, normalize_embeddings=True)
        else:
            return np.array([self.generate_embedding(t) for t in texts])


class FAISSIndex:
    """FAISS-based vector index for fast similarity search."""
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize FAISS index."""
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_map = {}  # Maps index position to product ID
        self.products = {}  # Maps product ID to Product object
        
        self._init_index()
    
    def _init_index(self):
        """Initialize the FAISS index."""
        try:
            import faiss
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.use_faiss = True
        except ImportError:
            print("FAISS not available, using numpy-based search")
            self.use_faiss = False
            self.vectors = []
    
    def add(self, product_id: str, embedding: np.ndarray, product: Product):
        """Add a product to the index."""
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        if self.use_faiss:
            idx = self.index.ntotal
            self.index.add(embedding)
        else:
            idx = len(self.vectors)
            self.vectors.append(embedding.flatten())
        
        self.id_map[idx] = product_id
        self.products[product_id] = product
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar products.
        
        Returns:
            List of (product_id, similarity_score) tuples
        """
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        if self.use_faiss:
            scores, indices = self.index.search(query_embedding, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self.id_map:
                    results.append((self.id_map[idx], float(score)))
            return results
        else:
            # Numpy-based fallback
            if not self.vectors:
                return []
            vectors = np.array(self.vectors)
            scores = np.dot(vectors, query_embedding.flatten())
            top_k = np.argsort(scores)[-k:][::-1]
            return [(self.id_map[idx], float(scores[idx])) for idx in top_k]
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID."""
        return self.products.get(product_id)
    
    def size(self) -> int:
        """Return number of indexed products."""
        if self.use_faiss:
            return self.index.ntotal
        return len(self.vectors)


class MarketplaceSearchPipeline:
    """
    Complete search pipeline with optional defense integration.
    """
    
    def __init__(self, use_neural_embeddings: bool = False):
        """Initialize search pipeline."""
        self.embedding_generator = EmbeddingGenerator(use_neural=use_neural_embeddings)
        self.index = FAISSIndex(embedding_dim=self.embedding_generator.embedding_dim)
        self.sanitized_index = FAISSIndex(embedding_dim=self.embedding_generator.embedding_dim)
        self.defender = None
        self.defense_results = {}
    
    def set_defender(self, defender):
        """Set the defense module for sanitized search."""
        self.defender = defender
    
    def _get_product_text(self, product: Product) -> str:
        """Extract searchable text from product."""
        parts = [
            product.title,
            product.description,
            product.category,
            " ".join(product.metadata.get("keywords", []))
        ]
        return " ".join(parts)
    
    def index_products(self, products: List[Product], with_defense: bool = False):
        """
        Index products for search.
        
        Args:
            products: List of products to index
            with_defense: Whether to also create sanitized index
        """
        # Extract texts for TF-IDF fitting
        texts = [self._get_product_text(p) for p in products]
        self.embedding_generator.fit(texts)
        
        # Index original products
        print(f"Indexing {len(products)} products...")
        for product in products:
            text = self._get_product_text(product)
            embedding = self.embedding_generator.generate_embedding(text)
            self.index.add(product.id, embedding, product)
        
        # Create sanitized index if defender is available
        if with_defense and self.defender:
            print("Creating sanitized index with defense...")
            self.defender.build_baseline(products)
            
            for product in products:
                result = self.defender.analyze_product(product, sanitize=True)
                self.defense_results[product.id] = result
                
                # Use sanitized product for sanitized index
                sanitized = result.sanitized_product or product
                text = self._get_product_text(sanitized)
                embedding = self.embedding_generator.generate_embedding(text)
                self.sanitized_index.add(product.id, embedding, sanitized)
        
        print(f"Indexed {self.index.size()} products")
        if with_defense:
            print(f"Sanitized index: {self.sanitized_index.size()} products")
    
    def search(self, query: str, k: int = 10, use_defense: bool = False) -> List[SearchResult]:
        """
        Search for products.
        
        Args:
            query: Search query
            k: Number of results to return
            use_defense: Whether to use sanitized index
        
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Choose index
        index = self.sanitized_index if use_defense and self.sanitized_index.size() > 0 else self.index
        
        # Search
        raw_results = index.search(query_embedding, k)
        
        # Build search results
        results = []
        for rank, (product_id, score) in enumerate(raw_results, 1):
            product = index.get_product(product_id)
            if product:
                defense_result = self.defense_results.get(product_id)
                # Get the original product to check is_poisoned (sanitized index has sanitized products)
                original_product = self.index.get_product(product_id)
                is_poisoned = original_product.is_poisoned if original_product else product.is_poisoned
                
                # When defense is enabled, apply penalty to suspicious products' scores
                # Use a subtractive penalty to demote suspicious items regardless of base score
                adjusted_score = score
                if use_defense and defense_result and defense_result.is_suspicious:
                    # Subtract a penalty based on threat score to push suspicious items down
                    penalty = defense_result.threat_score * 1.0  # penalty range: 0 to 1
                    adjusted_score = score - penalty
                
                results.append(SearchResult(
                    product=product,
                    score=adjusted_score,
                    rank=rank,
                    is_poisoned=is_poisoned,
                    is_suspicious=defense_result.is_suspicious if defense_result else False,
                    threat_score=defense_result.threat_score if defense_result else 0.0,
                    defense_applied=use_defense and defense_result is not None
                ))
        
        # When defense is enabled, re-rank by adjusted scores
        if use_defense:
            results.sort(key=lambda r: r.score, reverse=True)
            # Update ranks after sorting
            for i, result in enumerate(results):
                result.rank = i + 1
        
        return results
    
    def compare_search(self, query: str, k: int = 10) -> Dict:
        """
        Compare search results with and without defense.
        
        Returns:
            Dictionary with comparison analysis
        """
        baseline_results = self.search(query, k, use_defense=False)
        defended_results = self.search(query, k, use_defense=True)
        
        # Analyze poisoned products in results
        baseline_poisoned = [r for r in baseline_results if r.is_poisoned]
        defended_poisoned = [r for r in defended_results if r.is_poisoned]
        
        # Calculate rank changes
        baseline_ids = [r.product.id for r in baseline_results]
        defended_ids = [r.product.id for r in defended_results]
        
        rank_changes = []
        for product_id in set(baseline_ids) | set(defended_ids):
            baseline_rank = baseline_ids.index(product_id) + 1 if product_id in baseline_ids else None
            defended_rank = defended_ids.index(product_id) + 1 if product_id in defended_ids else None
            
            product = self.index.get_product(product_id)
            if product and (baseline_rank != defended_rank):
                rank_changes.append({
                    "product_id": product_id,
                    "title": product.title[:50],
                    "is_poisoned": product.is_poisoned,
                    "baseline_rank": baseline_rank,
                    "defended_rank": defended_rank
                })
        
        return {
            "query": query,
            "baseline": {
                "results": baseline_results,
                "poisoned_count": len(baseline_poisoned),
                "poisoned_in_top_5": len([r for r in baseline_poisoned if r.rank <= 5])
            },
            "defended": {
                "results": defended_results,
                "poisoned_count": len(defended_poisoned),
                "poisoned_in_top_5": len([r for r in defended_poisoned if r.rank <= 5])
            },
            "rank_changes": rank_changes,
            "defense_impact": {
                "poisoned_removed": len(baseline_poisoned) - len(defended_poisoned),
                "rank_changes_count": len(rank_changes)
            }
        }


class SearchEvaluator:
    """Evaluates search quality with and without poisoning/defense."""
    
    @staticmethod
    def calculate_precision_at_k(results: List[SearchResult], k: int, 
                                 relevant_func=lambda r: not r.is_poisoned) -> float:
        """
        Calculate precision@k.
        
        Args:
            results: Search results
            k: Top k results to consider
            relevant_func: Function to determine if result is relevant
        
        Returns:
            Precision value (0-1)
        """
        top_k = results[:k]
        relevant = sum(1 for r in top_k if relevant_func(r))
        return relevant / k if k > 0 else 0.0
    
    @staticmethod
    def calculate_ndcg(results: List[SearchResult], k: int,
                      relevance_func=lambda r: 0 if r.is_poisoned else 1) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain).
        
        Returns:
            NDCG value (0-1)
        """
        def dcg(relevances):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        
        top_k = results[:k]
        relevances = [relevance_func(r) for r in top_k]
        
        actual_dcg = dcg(relevances)
        ideal_dcg = dcg(sorted(relevances, reverse=True))
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def calculate_poisoning_impact(baseline_results: List[SearchResult],
                                  defended_results: List[SearchResult]) -> Dict:
        """
        Calculate the impact of poisoning and defense effectiveness.
        
        Returns:
            Dictionary with impact metrics
        """
        baseline_poisoned_ranks = [r.rank for r in baseline_results if r.is_poisoned]
        defended_poisoned_ranks = [r.rank for r in defended_results if r.is_poisoned]
        
        return {
            "baseline_poisoned_avg_rank": np.mean(baseline_poisoned_ranks) if baseline_poisoned_ranks else None,
            "defended_poisoned_avg_rank": np.mean(defended_poisoned_ranks) if defended_poisoned_ranks else None,
            "baseline_poisoned_count": len(baseline_poisoned_ranks),
            "defended_poisoned_count": len(defended_poisoned_ranks),
            "poisoning_promotion": len([r for r in baseline_poisoned_ranks if r <= 5]),
            "defense_blocked": len([r for r in baseline_poisoned_ranks if r <= 5]) - \
                             len([r for r in defended_poisoned_ranks if r <= 5])
        }


if __name__ == "__main__":
    from dataset_generator import MarketplaceDatasetGenerator
    from attack_simulator import AttackSimulator
    from defense_module import MarketplaceDefender
    
    # Setup
    print("=" * 60)
    print("SEARCH PIPELINE DEMO")
    print("=" * 60)
    
    # Generate dataset
    generator = MarketplaceDatasetGenerator(seed=42)
    clean_products = generator.generate_dataset(size=100)
    
    # Poison dataset
    attacker = AttackSimulator(seed=42)
    poisoned_products, attack_stats = attacker.poison_dataset(clean_products, poison_rate=0.25)
    
    print(f"\nDataset: {len(poisoned_products)} products")
    print(f"Poisoned: {attack_stats['poisoned']} ({attack_stats['poisoned']/len(poisoned_products)*100:.0f}%)")
    
    # Initialize pipeline with defense
    pipeline = MarketplaceSearchPipeline(use_neural_embeddings=False)
    defender = MarketplaceDefender()
    pipeline.set_defender(defender)
    
    # Index products
    pipeline.index_products(poisoned_products, with_defense=True)
    
    # Test search
    test_queries = [
        "wireless headphones",
        "premium quality",
        "best electronics",
    ]
    
    evaluator = SearchEvaluator()
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print("=" * 60)
        
        comparison = pipeline.compare_search(query, k=10)
        
        print(f"\n[BASELINE - No Defense]")
        print(f"  Poisoned products in results: {comparison['baseline']['poisoned_count']}")
        print(f"  Poisoned in top 5: {comparison['baseline']['poisoned_in_top_5']}")
        
        print(f"\n[DEFENDED - With Defense]")
        print(f"  Poisoned products in results: {comparison['defended']['poisoned_count']}")
        print(f"  Poisoned in top 5: {comparison['defended']['poisoned_in_top_5']}")
        
        print(f"\n[IMPACT]")
        print(f"  Rank changes: {comparison['defense_impact']['rank_changes_count']}")
        
        # Calculate metrics
        baseline_precision = evaluator.calculate_precision_at_k(
            comparison['baseline']['results'], k=5
        )
        defended_precision = evaluator.calculate_precision_at_k(
            comparison['defended']['results'], k=5
        )
        
        print(f"\n[METRICS]")
        print(f"  Baseline Precision@5 (clean results): {baseline_precision:.2f}")
        print(f"  Defended Precision@5 (clean results): {defended_precision:.2f}")
