"""
Unit Tests for Marketplace Poisoning Shield
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_generator import MarketplaceDatasetGenerator, Product
from attack_simulator import AttackSimulator
from defense_module import MarketplaceDefender
from search_pipeline import MarketplaceSearchPipeline, SearchEvaluator


class TestDatasetGenerator(unittest.TestCase):
    """Tests for dataset generation."""
    
    def setUp(self):
        self.generator = MarketplaceDatasetGenerator(seed=42)
    
    def test_generate_single_product(self):
        """Test generating a single product."""
        product = self.generator.generate_product()
        
        self.assertIsInstance(product, Product)
        self.assertIsNotNone(product.id)
        self.assertIsNotNone(product.title)
        self.assertGreater(len(product.title), 0)
        self.assertIn(product.category, ['electronics', 'clothing', 'home', 'sports', 'beauty'])
        self.assertGreater(product.price, 0)
        self.assertGreaterEqual(product.rating, 0)
        self.assertLessEqual(product.rating, 5)
    
    def test_generate_dataset(self):
        """Test generating a full dataset."""
        size = 50
        products = self.generator.generate_dataset(size=size)
        
        self.assertEqual(len(products), size)
        self.assertTrue(all(isinstance(p, Product) for p in products))
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = MarketplaceDatasetGenerator(seed=123)
        gen2 = MarketplaceDatasetGenerator(seed=123)
        
        products1 = gen1.generate_dataset(size=10)
        products2 = gen2.generate_dataset(size=10)
        
        for p1, p2 in zip(products1, products2):
            self.assertEqual(p1.title, p2.title)
            self.assertEqual(p1.category, p2.category)


class TestAttackSimulator(unittest.TestCase):
    """Tests for attack simulation."""
    
    def setUp(self):
        self.generator = MarketplaceDatasetGenerator(seed=42)
        self.attacker = AttackSimulator(seed=42)
        self.sample_product = self.generator.generate_product()
    
    def test_hidden_character_injection(self):
        """Test hidden character attack."""
        original_bytes = len(self.sample_product.title.encode())
        
        poisoned = self.attacker.inject_hidden_characters(self.sample_product, intensity=0.5)
        
        self.assertTrue(poisoned.is_poisoned)
        self.assertEqual(poisoned.poison_type, "hidden_characters")
        self.assertGreater(len(poisoned.title.encode()), original_bytes)
    
    def test_keyword_stuffing(self):
        """Test keyword stuffing attack."""
        original_len = len(self.sample_product.description)
        
        poisoned = self.attacker.stuff_keywords(self.sample_product, keyword_count=10)
        
        self.assertTrue(poisoned.is_poisoned)
        self.assertEqual(poisoned.poison_type, "keyword_stuffing")
    
    def test_homoglyph_attack(self):
        """Test homoglyph substitution attack."""
        poisoned = self.attacker.apply_homoglyphs(self.sample_product, replacement_rate=0.3)
        
        self.assertTrue(poisoned.is_poisoned)
        self.assertEqual(poisoned.poison_type, "homoglyph")
    
    def test_fake_review_injection(self):
        """Test fake review attack."""
        original_count = self.sample_product.review_count
        
        poisoned = self.attacker.inject_fake_reviews(self.sample_product, fake_count=5)
        
        self.assertTrue(poisoned.is_poisoned)
        self.assertEqual(poisoned.poison_type, "fake_reviews")
        self.assertEqual(poisoned.review_count, original_count + 5)
        self.assertGreaterEqual(poisoned.rating, self.sample_product.rating)
    
    def test_dataset_poisoning(self):
        """Test poisoning an entire dataset."""
        products = self.generator.generate_dataset(size=50)
        
        poisoned, stats = self.attacker.poison_dataset(products, poison_rate=0.3)
        
        self.assertEqual(len(poisoned), len(products))
        self.assertGreater(stats['poisoned'], 0)
        self.assertLess(stats['poisoned'], len(products))


class TestDefenseModule(unittest.TestCase):
    """Tests for defense mechanisms."""
    
    def setUp(self):
        self.generator = MarketplaceDatasetGenerator(seed=42)
        self.attacker = AttackSimulator(seed=42)
        self.defender = MarketplaceDefender()
        
        # Build baseline
        products = self.generator.generate_dataset(size=30)
        self.defender.build_baseline(products)
        
        self.sample_product = products[0]
    
    def test_unicode_detection(self):
        """Test Unicode anomaly detection."""
        text_with_hidden = "Hello\u200bWorld\u200b"
        score, anomalies = self.defender.detect_unicode_anomalies(text_with_hidden)
        
        self.assertGreater(score, 0)
        self.assertGreater(len(anomalies), 0)
    
    def test_homoglyph_detection(self):
        """Test homoglyph detection."""
        text_with_homoglyphs = "Hеllo Wоrld"  # Cyrillic е and о
        score, detected = self.defender.detect_homoglyphs(text_with_homoglyphs)
        
        self.assertGreater(score, 0)
        self.assertGreater(len(detected), 0)
    
    def test_keyword_density(self):
        """Test keyword density analysis."""
        spammy_text = "BEST BEST BEST TOP RATED #1 BESTSELLER AMAZING INCREDIBLE WOW"
        score, details = self.defender.analyze_keyword_density(spammy_text)
        
        self.assertGreater(score, 0.3)
    
    def test_clean_product_analysis(self):
        """Test that clean products have low threat scores."""
        result = self.defender.analyze_product(self.sample_product)
        
        self.assertLess(result.threat_score, 0.5)
    
    def test_poisoned_product_detection(self):
        """Test that poisoned products are detected."""
        poisoned = self.attacker.inject_hidden_characters(self.sample_product, intensity=0.5)
        result = self.defender.analyze_product(poisoned)
        
        # Should have elevated threat score
        self.assertGreater(result.threat_score, 0.1)
    
    def test_sanitization(self):
        """Test text sanitization."""
        dirty_text = "Hello\u200bWorld\u200b with е (Cyrillic)"
        clean_text = self.defender.sanitize_text(dirty_text)
        
        # Should remove zero-width chars
        self.assertNotIn('\u200b', clean_text)


class TestSearchPipeline(unittest.TestCase):
    """Tests for search pipeline."""
    
    def setUp(self):
        self.generator = MarketplaceDatasetGenerator(seed=42)
        self.pipeline = MarketplaceSearchPipeline(use_neural_embeddings=False)
        self.products = self.generator.generate_dataset(size=30)
    
    def test_indexing(self):
        """Test product indexing."""
        self.pipeline.index_products(self.products, with_defense=False)
        
        self.assertEqual(self.pipeline.index.size(), len(self.products))
    
    def test_search(self):
        """Test search functionality."""
        self.pipeline.index_products(self.products, with_defense=False)
        
        results = self.pipeline.search("premium quality", k=5)
        
        self.assertLessEqual(len(results), 5)
        self.assertTrue(all(hasattr(r, 'product') for r in results))
    
    def test_search_with_defense(self):
        """Test search with defense enabled."""
        defender = MarketplaceDefender()
        defender.build_baseline(self.products)
        
        self.pipeline.set_defender(defender)
        self.pipeline.index_products(self.products, with_defense=True)
        
        results = self.pipeline.search("premium", k=5, use_defense=True)
        
        self.assertLessEqual(len(results), 5)


class TestSearchEvaluator(unittest.TestCase):
    """Tests for search evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        # Create mock results
        from search_pipeline import SearchResult
        from dataset_generator import Product
        
        class MockProduct:
            is_poisoned = False
        
        class MockResult:
            def __init__(self, poisoned):
                self.is_poisoned = poisoned
                self.product = MockProduct()
                self.product.is_poisoned = poisoned
        
        # 3 clean, 2 poisoned in top 5
        results = [
            MockResult(False),
            MockResult(False),
            MockResult(True),
            MockResult(False),
            MockResult(True),
        ]
        
        precision = SearchEvaluator.calculate_precision_at_k(results, k=5)
        self.assertEqual(precision, 0.6)  # 3/5 clean


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end(self):
        """Test complete attack-defense-search pipeline."""
        # Generate
        generator = MarketplaceDatasetGenerator(seed=42)
        products = generator.generate_dataset(size=50)
        
        # Attack
        attacker = AttackSimulator(seed=42)
        poisoned_products, stats = attacker.poison_dataset(products, poison_rate=0.3)
        
        self.assertGreater(stats['poisoned'], 0)
        
        # Defend
        defender = MarketplaceDefender()
        defender.build_baseline(products)
        results, summary = defender.analyze_dataset(poisoned_products)
        
        self.assertEqual(len(results), len(poisoned_products))
        self.assertGreater(summary['detection_rate'], 0)
        
        # Search
        pipeline = MarketplaceSearchPipeline(use_neural_embeddings=False)
        pipeline.set_defender(defender)
        pipeline.index_products(poisoned_products, with_defense=True)
        
        comparison = pipeline.compare_search("premium", k=5)
        
        self.assertIn('baseline', comparison)
        self.assertIn('defended', comparison)


if __name__ == '__main__':
    unittest.main(verbosity=2)
