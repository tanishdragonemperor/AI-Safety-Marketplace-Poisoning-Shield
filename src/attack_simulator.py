"""
Attack Simulator for Marketplace Poisoning Shield
Implements various data poisoning attack techniques for red team testing.
"""

import random
import copy
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import numpy as np

from dataset_generator import Product


class AttackSimulator:
    """
    Simulates various data poisoning attacks on marketplace listings.
    
    Attack Categories:
    1. Hidden Character Injection - Unicode/invisible characters
    2. Keyword Stuffing - Excessive keyword repetition
    3. Adversarial Paraphrasing - Semantic manipulation
    4. Review Manipulation - Fake positive reviews
    5. Metadata Poisoning - Hidden metadata manipulation
    6. Homoglyph Attacks - Visual character substitution
    """
    
    # Unicode characters that are invisible or look like spaces
    HIDDEN_CHARS = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u2060',  # Word joiner
        '\ufeff',  # Zero-width no-break space
        '\u00ad',  # Soft hyphen
        '\u034f',  # Combining grapheme joiner
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
    ]
    
    # Homoglyphs - characters that look similar to ASCII
    HOMOGLYPHS = {
        'a': ['а', 'ɑ', 'α', 'а'],  # Cyrillic, Latin small alpha
        'e': ['е', 'ё', 'ε', 'ę'],  # Cyrillic, Greek
        'o': ['о', 'ο', 'ø', 'ö'],  # Cyrillic, Greek, Nordic
        'c': ['с', 'ϲ', 'ç'],       # Cyrillic, Greek
        'p': ['р', 'ρ'],             # Cyrillic, Greek
        'i': ['і', 'ι', 'ı'],        # Cyrillic, Greek, Turkish
        's': ['ѕ', 'ś', 'ş'],        # Cyrillic
        'x': ['х', 'χ'],             # Cyrillic, Greek
        'y': ['у', 'γ'],             # Cyrillic, Greek
    }
    
    # High-value keywords for stuffing
    SEO_KEYWORDS = {
        "electronics": ["best", "top rated", "premium", "professional", "high quality", 
                       "bestseller", "#1", "award winning", "certified", "original"],
        "clothing": ["designer", "luxury", "authentic", "trending", "bestseller",
                    "premium quality", "limited edition", "exclusive", "handmade"],
        "home": ["bestseller", "top rated", "premium", "luxury", "designer",
                "handcrafted", "artisan", "exclusive", "limited"],
        "sports": ["professional", "gym quality", "competition grade", "elite",
                  "top performance", "bestseller", "certified", "official"],
        "beauty": ["organic", "natural", "dermatologist tested", "clinical",
                  "bestseller", "award winning", "premium", "luxury"]
    }
    
    # Fake review templates
    FAKE_REVIEW_TEMPLATES = [
        "This is absolutely the BEST {category} product I've ever used! 5 stars!!!",
        "Amazing quality! Everyone should buy this. Best purchase ever!",
        "WOW! I can't believe how good this is. Exceeded all expectations!",
        "Perfect! Exactly what I needed. Fast shipping too! Highly recommend!",
        "Outstanding product! Best in its category. Will buy again!",
        "Incredible value for money! Works perfectly! A+++",
        "Must have product! Changed my life! Best {category} item ever!",
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize attack simulator."""
        random.seed(seed)
        np.random.seed(seed)
        self.attack_log = []
    
    def _log_attack(self, product_id: str, attack_type: str, details: Dict):
        """Log attack for analysis."""
        self.attack_log.append({
            "product_id": product_id,
            "attack_type": attack_type,
            "details": details
        })
    
    # ==================== ATTACK 1: Hidden Character Injection ====================
    
    def inject_hidden_characters(self, product: Product, intensity: float = 0.3) -> Product:
        """
        Inject invisible Unicode characters into product text.
        These can manipulate tokenization and embeddings without visible changes.
        
        Args:
            product: Original product
            intensity: Proportion of words to inject characters into (0-1)
        
        Returns:
            Poisoned product copy
        """
        poisoned = copy.deepcopy(product)
        
        def inject_chars(text: str) -> Tuple[str, int]:
            words = text.split()
            injection_count = 0
            for i in range(len(words)):
                if random.random() < intensity:
                    # Insert hidden char at random position in word
                    char = random.choice(self.HIDDEN_CHARS)
                    pos = random.randint(0, len(words[i]))
                    words[i] = words[i][:pos] + char + words[i][pos:]
                    injection_count += 1
            return ' '.join(words), injection_count
        
        # Inject into title and description
        poisoned.title, title_injections = inject_chars(poisoned.title)
        poisoned.description, desc_injections = inject_chars(poisoned.description)
        
        poisoned.is_poisoned = True
        poisoned.poison_type = "hidden_characters"
        
        self._log_attack(product.id, "hidden_characters", {
            "title_injections": title_injections,
            "description_injections": desc_injections,
            "intensity": intensity
        })
        
        return poisoned
    
    # ==================== ATTACK 2: Keyword Stuffing ====================
    
    def stuff_keywords(self, product: Product, keyword_count: int = 10) -> Product:
        """
        Add excessive SEO keywords to boost search rankings unfairly.
        
        Args:
            product: Original product
            keyword_count: Number of keywords to stuff
        
        Returns:
            Poisoned product copy
        """
        poisoned = copy.deepcopy(product)
        
        # Get category-specific keywords
        keywords = self.SEO_KEYWORDS.get(product.category, self.SEO_KEYWORDS["electronics"])
        selected_keywords = random.choices(keywords, k=keyword_count)
        
        # Various stuffing techniques
        technique = random.choice(["append", "prepend", "inject", "metadata"])
        
        if technique == "append":
            # Add keywords at end with separators
            keyword_string = " | ".join(selected_keywords)
            poisoned.description += f" Tags: {keyword_string}"
        
        elif technique == "prepend":
            # Add keywords at beginning
            keyword_string = " ".join(selected_keywords)
            poisoned.title = f"[{keyword_string[:30]}] {poisoned.title}"
        
        elif technique == "inject":
            # Inject keywords throughout description
            words = poisoned.description.split()
            for kw in selected_keywords[:5]:
                pos = random.randint(0, len(words))
                words.insert(pos, kw)
            poisoned.description = ' '.join(words)
        
        else:  # metadata
            # Stuff keywords in metadata
            poisoned.metadata["hidden_keywords"] = selected_keywords
            poisoned.metadata["seo_tags"] = " ".join(selected_keywords * 2)
        
        poisoned.is_poisoned = True
        poisoned.poison_type = "keyword_stuffing"
        
        self._log_attack(product.id, "keyword_stuffing", {
            "technique": technique,
            "keywords_added": selected_keywords,
            "count": keyword_count
        })
        
        return poisoned
    
    # ==================== ATTACK 3: Homoglyph Attack ====================
    
    def apply_homoglyphs(self, product: Product, replacement_rate: float = 0.2) -> Product:
        """
        Replace characters with visually similar Unicode homoglyphs.
        Text looks identical but has different character codes.
        
        Args:
            product: Original product
            replacement_rate: Proportion of eligible characters to replace
        
        Returns:
            Poisoned product copy
        """
        poisoned = copy.deepcopy(product)
        
        def replace_chars(text: str) -> Tuple[str, int]:
            result = list(text)
            replacements = 0
            for i, char in enumerate(result):
                if char.lower() in self.HOMOGLYPHS and random.random() < replacement_rate:
                    result[i] = random.choice(self.HOMOGLYPHS[char.lower()])
                    replacements += 1
            return ''.join(result), replacements
        
        poisoned.title, title_replacements = replace_chars(poisoned.title)
        poisoned.description, desc_replacements = replace_chars(poisoned.description)
        
        poisoned.is_poisoned = True
        poisoned.poison_type = "homoglyph"
        
        self._log_attack(product.id, "homoglyph", {
            "title_replacements": title_replacements,
            "description_replacements": desc_replacements,
            "replacement_rate": replacement_rate
        })
        
        return poisoned
    
    # ==================== ATTACK 4: Review Manipulation ====================
    
    def inject_fake_reviews(self, product: Product, fake_count: int = 5, 
                           boost_rating: bool = True) -> Product:
        """
        Inject fake positive reviews to artificially boost ratings.
        
        Args:
            product: Original product
            fake_count: Number of fake reviews to inject
            boost_rating: Whether to also boost the overall rating
        
        Returns:
            Poisoned product copy
        """
        poisoned = copy.deepcopy(product)
        
        fake_reviews = []
        for i in range(fake_count):
            template = random.choice(self.FAKE_REVIEW_TEMPLATES)
            review_text = template.format(category=product.category)
            
            fake_reviews.append({
                "id": f"fake_review_{i}_{random.randint(1000, 9999)}",
                "rating": 5,
                "text": review_text,
                "helpful_votes": random.randint(50, 200),  # Artificially high
                "verified_purchase": True,  # Falsely marked as verified
                "date": "2024-01-01T00:00:00",
                "is_fake": True  # Hidden marker for detection
            })
        
        # Inject fake reviews at various positions
        poisoned.reviews = fake_reviews + poisoned.reviews
        poisoned.review_count += fake_count
        
        # Boost the overall rating more aggressively for demo visibility
        if boost_rating:
            original_rating = poisoned.rating
            # More aggressive boost: add 0.5-1.0 stars depending on original rating
            if original_rating < 4.0:
                boost = min(1.2, 5.0 - original_rating)  # Bigger boost for lower ratings
            else:
                boost = 0.6  # Moderate boost for already high ratings
            
            poisoned.rating = min(5.0, round(original_rating + boost, 1))
        
        poisoned.is_poisoned = True
        poisoned.poison_type = "fake_reviews"
        
        self._log_attack(product.id, "fake_reviews", {
            "fake_count": fake_count,
            "original_rating": product.rating,
            "boosted_rating": poisoned.rating,
            "boost_applied": boost_rating
        })
        
        return poisoned
    
    # ==================== ATTACK 5: Adversarial Paraphrasing ====================
    
    def adversarial_paraphrase(self, product: Product, target_keywords: List[str] = None) -> Product:
        """
        Manipulate description semantically to include unrelated high-value keywords
        while maintaining grammatical correctness.
        
        Args:
            product: Original product
            target_keywords: Keywords to inject (uses competitor brand names if None)
        
        Returns:
            Poisoned product copy
        """
        poisoned = copy.deepcopy(product)
        
        if target_keywords is None:
            # Default: inject competitor/unrelated high-value terms
            target_keywords = ["Apple", "Samsung", "Nike", "Amazon", "Premium Brand", 
                             "Official", "Authentic", "Original"]
        
        # Paraphrasing techniques
        injections = []
        
        # Add comparison phrases
        brand = random.choice(target_keywords[:4])
        comparison = random.choice([
            f"Better than {brand}!",
            f"Alternative to {brand}.",
            f"Compatible with {brand} products.",
            f"{brand} quality at lower price.",
        ])
        poisoned.description += f" {comparison}"
        injections.append(comparison)
        
        # Add misleading authenticity claims
        authenticity = random.choice([
            "100% Authentic and Original.",
            "Genuine product guaranteed.",
            "Official licensed merchandise.",
        ])
        poisoned.description = authenticity + " " + poisoned.description
        injections.append(authenticity)
        
        poisoned.is_poisoned = True
        poisoned.poison_type = "adversarial_paraphrase"
        
        self._log_attack(product.id, "adversarial_paraphrase", {
            "target_keywords": target_keywords,
            "injections": injections
        })
        
        return poisoned
    
    # ==================== ATTACK 6: Metadata Poisoning ====================
    
    def poison_metadata(self, product: Product) -> Product:
        """
        Inject malicious or misleading metadata that may affect search/ranking.
        
        Args:
            product: Original product
        
        Returns:
            Poisoned product copy
        """
        poisoned = copy.deepcopy(product)
        
        # Various metadata manipulations
        manipulations = []
        
        # Inject hidden promotional text
        poisoned.metadata["__hidden_promo__"] = "BUY NOW BEST DEAL LOWEST PRICE " * 10
        manipulations.append("hidden_promo")
        
        # Falsify popularity metrics
        poisoned.metadata["view_count"] = random.randint(100000, 1000000)
        poisoned.metadata["purchase_count"] = random.randint(10000, 50000)
        manipulations.append("fake_metrics")
        
        # Add misleading category tags
        all_categories = ["electronics", "clothing", "home", "sports", "beauty"]
        poisoned.metadata["additional_categories"] = all_categories  # Appear in all searches
        manipulations.append("category_spam")
        
        # Inject timestamp manipulation (appear as newer)
        poisoned.metadata["boosted_timestamp"] = "2099-12-31T23:59:59"
        manipulations.append("timestamp_manipulation")
        
        poisoned.is_poisoned = True
        poisoned.poison_type = "metadata_poisoning"
        
        self._log_attack(product.id, "metadata_poisoning", {
            "manipulations": manipulations
        })
        
        return poisoned
    
    # ==================== Combined/Composite Attacks ====================
    
    def composite_attack(self, product: Product, attacks: List[str] = None) -> Product:
        """
        Apply multiple attack types to a single product.
        
        Args:
            product: Original product
            attacks: List of attack types to apply
        
        Returns:
            Poisoned product with multiple attack vectors
        """
        if attacks is None:
            attacks = ["hidden_characters", "keyword_stuffing", "fake_reviews"]
        
        poisoned = copy.deepcopy(product)
        applied_attacks = []
        
        for attack in attacks:
            if attack == "hidden_characters":
                poisoned = self.inject_hidden_characters(poisoned, intensity=0.15)
            elif attack == "keyword_stuffing":
                poisoned = self.stuff_keywords(poisoned, keyword_count=5)
            elif attack == "homoglyph":
                poisoned = self.apply_homoglyphs(poisoned, replacement_rate=0.1)
            elif attack == "fake_reviews":
                poisoned = self.inject_fake_reviews(poisoned, fake_count=3)
            elif attack == "adversarial_paraphrase":
                poisoned = self.adversarial_paraphrase(poisoned)
            elif attack == "metadata_poisoning":
                poisoned = self.poison_metadata(poisoned)
            applied_attacks.append(attack)
        
        poisoned.poison_type = f"composite:{'+'.join(applied_attacks)}"
        
        return poisoned
    
    # ==================== Dataset Poisoning ====================
    
    def poison_dataset(self, products: List[Product], poison_rate: float = 0.2,
                      attack_distribution: Dict[str, float] = None) -> Tuple[List[Product], Dict]:
        """
        Poison a portion of the dataset with various attacks.
        
        Args:
            products: List of clean products
            poison_rate: Proportion of products to poison (0-1)
            attack_distribution: Distribution of attack types
        
        Returns:
            Tuple of (poisoned dataset, statistics)
        """
        if attack_distribution is None:
            attack_distribution = {
                "hidden_characters": 0.2,
                "keyword_stuffing": 0.25,
                "homoglyph": 0.15,
                "fake_reviews": 0.2,
                "adversarial_paraphrase": 0.1,
                "metadata_poisoning": 0.1
            }
        
        result = []
        stats = {
            "total": len(products),
            "poisoned": 0,
            "clean": 0,
            "attack_counts": {k: 0 for k in attack_distribution.keys()}
        }
        
        attack_types = list(attack_distribution.keys())
        attack_probs = list(attack_distribution.values())
        
        for product in products:
            if random.random() < poison_rate:
                # Select attack type based on distribution
                attack_type = np.random.choice(attack_types, p=attack_probs)
                
                # Apply selected attack
                if attack_type == "hidden_characters":
                    poisoned = self.inject_hidden_characters(product)
                elif attack_type == "keyword_stuffing":
                    poisoned = self.stuff_keywords(product)
                elif attack_type == "homoglyph":
                    poisoned = self.apply_homoglyphs(product)
                elif attack_type == "fake_reviews":
                    poisoned = self.inject_fake_reviews(product)
                elif attack_type == "adversarial_paraphrase":
                    poisoned = self.adversarial_paraphrase(product)
                else:
                    poisoned = self.poison_metadata(product)
                
                result.append(poisoned)
                stats["poisoned"] += 1
                stats["attack_counts"][attack_type] += 1
            else:
                result.append(product)
                stats["clean"] += 1
        
        return result, stats
    
    def get_attack_log(self) -> List[Dict]:
        """Return the complete attack log."""
        return self.attack_log
    
    def clear_attack_log(self):
        """Clear the attack log."""
        self.attack_log = []


if __name__ == "__main__":
    from dataset_generator import MarketplaceDatasetGenerator
    
    # Generate clean dataset
    generator = MarketplaceDatasetGenerator(seed=42)
    clean_products = generator.generate_dataset(size=50)
    
    # Initialize attacker
    attacker = AttackSimulator(seed=42)
    
    # Demo each attack type
    print("=" * 60)
    print("ATTACK DEMONSTRATIONS")
    print("=" * 60)
    
    sample = clean_products[0]
    print(f"\nOriginal Product: {sample.title}")
    print(f"Description: {sample.description[:100]}...")
    
    # Test hidden character injection
    poisoned = attacker.inject_hidden_characters(sample)
    print(f"\n[ATTACK 1] Hidden Characters:")
    print(f"  Title (looks same but has hidden chars): {poisoned.title}")
    print(f"  Title bytes changed: {len(poisoned.title.encode()) != len(sample.title.encode())}")
    
    # Test keyword stuffing
    poisoned = attacker.stuff_keywords(sample)
    print(f"\n[ATTACK 2] Keyword Stuffing:")
    print(f"  Modified description: {poisoned.description[:150]}...")
    
    # Test homoglyph attack
    poisoned = attacker.apply_homoglyphs(sample)
    print(f"\n[ATTACK 3] Homoglyph Attack:")
    print(f"  Title (visual): {poisoned.title}")
    print(f"  Characters changed: {poisoned.title != sample.title}")
    
    # Poison entire dataset
    print("\n" + "=" * 60)
    print("DATASET POISONING")
    print("=" * 60)
    
    poisoned_dataset, stats = attacker.poison_dataset(clean_products, poison_rate=0.3)
    print(f"\nDataset Statistics:")
    print(f"  Total products: {stats['total']}")
    print(f"  Poisoned: {stats['poisoned']} ({stats['poisoned']/stats['total']*100:.1f}%)")
    print(f"  Clean: {stats['clean']}")
    print(f"\nAttack Distribution:")
    for attack, count in stats['attack_counts'].items():
        print(f"  {attack}: {count}")
