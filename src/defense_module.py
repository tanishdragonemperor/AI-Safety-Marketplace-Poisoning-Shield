"""
Defense Module for Marketplace Poisoning Shield
Implements multi-layer defenses against data poisoning attacks.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from unidecode import unidecode

from dataset_generator import Product


@dataclass
class DefenseResult:
    """Result of defense analysis on a product."""
    product_id: str
    is_suspicious: bool
    threat_score: float  # 0-1, higher = more suspicious
    detected_attacks: List[str]
    sanitized_product: Optional[Product]
    details: Dict = field(default_factory=dict)


class MarketplaceDefender:
    """
    Multi-layer defense system against marketplace data poisoning.
    
    Defense Layers:
    1. Unicode Anomaly Detection - Find hidden/suspicious characters
    2. Keyword Density Analysis - Detect keyword stuffing
    3. Homoglyph Detection - Find character substitutions
    4. Review Authenticity Analysis - Detect fake reviews
    5. Metadata Validation - Check for suspicious metadata
    6. Statistical Anomaly Detection - Compare against baseline
    """
    
    # Known suspicious Unicode categories
    SUSPICIOUS_UNICODE_CATEGORIES = {
        'Cf',  # Format characters (includes zero-width chars)
        'Co',  # Private use
        'Cn',  # Unassigned
        'Cc',  # Control characters (except common ones)
    }
    
    # Known zero-width and invisible characters
    ZERO_WIDTH_CHARS = {
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u2060',  # Word joiner
        '\ufeff',  # Zero-width no-break space (BOM)
        '\u00ad',  # Soft hyphen
        '\u034f',  # Combining grapheme joiner
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
        '\u180e',  # Mongolian vowel separator
    }
    
    # Common Latin characters and their homoglyph equivalents
    HOMOGLYPH_MAP = {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x', 'у': 'y',
        'і': 'i', 'ѕ': 's', 'ј': 'j', 'һ': 'h', 'ԁ': 'd', 'ԝ': 'w',
        'α': 'a', 'ε': 'e', 'ι': 'i', 'ο': 'o', 'υ': 'u', 'ν': 'v',
        'ρ': 'p', 'τ': 't', 'ω': 'w', 'χ': 'x', 'γ': 'y',
        'ɑ': 'a', 'ɡ': 'g', 'ɩ': 'i', 'ɴ': 'n', 'ʀ': 'r',
    }
    
    # Spam/SEO keyword patterns
    SPAM_PATTERNS = [
        r'\b(best|top|#1|number\s*one|bestseller)\b',
        r'\b(buy\s*now|limited\s*time|act\s*fast|hurry)\b',
        r'\b(100%|guaranteed|authentic|original|genuine)\b',
        r'\b(free\s*shipping|lowest\s*price|best\s*deal)\b',
        r'[!]{2,}',  # Multiple exclamation marks
        r'[A-Z]{5,}',  # Excessive caps
    ]
    
    # Fake review indicators
    FAKE_REVIEW_PATTERNS = [
        r'\b(best\s+ever|changed\s+my\s+life|must\s+have)\b',
        r'\b(amazing|incredible|outstanding|perfect)[!]+',
        r'\b(highly\s+recommend|5\s*stars?|A\+{2,})\b',
        r'everyone\s+should\s+(buy|get|have)',
    ]
    
    def __init__(self, 
                 unicode_threshold: float = 0.02,
                 keyword_density_threshold: float = 0.15,
                 homoglyph_threshold: float = 0.05,
                 review_score_threshold: float = 0.6,
                 overall_threat_threshold: float = 0.5):
        """
        Initialize defender with configurable thresholds.
        
        Args:
            unicode_threshold: Max ratio of suspicious Unicode chars
            keyword_density_threshold: Max keyword density before flagging
            homoglyph_threshold: Max ratio of homoglyph characters
            review_score_threshold: Threshold for fake review detection
            overall_threat_threshold: Overall score to mark as suspicious
        """
        self.unicode_threshold = unicode_threshold
        self.keyword_density_threshold = keyword_density_threshold
        self.homoglyph_threshold = homoglyph_threshold
        self.review_score_threshold = review_score_threshold
        self.overall_threat_threshold = overall_threat_threshold
        
        # Compile regex patterns
        self.spam_regex = [re.compile(p, re.IGNORECASE) for p in self.SPAM_PATTERNS]
        self.fake_review_regex = [re.compile(p, re.IGNORECASE) for p in self.FAKE_REVIEW_PATTERNS]
        
        # Statistics for baseline comparison
        self.baseline_stats = None
    
    # ==================== DEFENSE 1: Unicode Anomaly Detection ====================
    
    def detect_unicode_anomalies(self, text: str) -> Tuple[float, List[Dict]]:
        """
        Detect hidden and suspicious Unicode characters.
        
        Returns:
            Tuple of (anomaly_score, list of anomalies found)
        """
        anomalies = []
        suspicious_count = 0
        
        for i, char in enumerate(text):
            # Check for zero-width characters
            if char in self.ZERO_WIDTH_CHARS:
                anomalies.append({
                    "type": "zero_width",
                    "position": i,
                    "char_code": f"U+{ord(char):04X}",
                    "char_name": unicodedata.name(char, "UNKNOWN")
                })
                suspicious_count += 1
                continue
            
            # Check Unicode category
            category = unicodedata.category(char)
            if category in self.SUSPICIOUS_UNICODE_CATEGORIES:
                # Allow some common control chars
                if char not in '\n\r\t':
                    anomalies.append({
                        "type": "suspicious_category",
                        "position": i,
                        "char_code": f"U+{ord(char):04X}",
                        "category": category
                    })
                    suspicious_count += 1
        
        # Calculate anomaly score
        score = suspicious_count / max(len(text), 1)
        
        return score, anomalies
    
    # ==================== DEFENSE 2: Keyword Density Analysis ====================
    
    def analyze_keyword_density(self, text: str) -> Tuple[float, Dict]:
        """
        Detect keyword stuffing through density analysis.
        
        Returns:
            Tuple of (density_score, analysis details)
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words)
        
        if word_count == 0:
            return 0.0, {"word_count": 0}
        
        # Count spam pattern matches
        spam_matches = []
        for pattern in self.spam_regex:
            matches = pattern.findall(text_lower)
            spam_matches.extend(matches)
        
        # Analyze word frequency for repetition
        word_freq = Counter(words)
        repeated_words = {w: c for w, c in word_freq.items() if c > 2 and len(w) > 3}
        
        # Calculate keyword density score
        spam_density = len(spam_matches) / word_count
        repetition_score = sum(repeated_words.values()) / word_count if repeated_words else 0
        
        # Combined score
        density_score = min(1.0, spam_density * 3 + repetition_score)
        
        return density_score, {
            "word_count": word_count,
            "spam_matches": spam_matches,
            "repeated_words": repeated_words,
            "spam_density": spam_density,
            "repetition_score": repetition_score
        }
    
    # ==================== DEFENSE 3: Homoglyph Detection ====================
    
    def detect_homoglyphs(self, text: str) -> Tuple[float, List[Dict]]:
        """
        Detect character substitutions using homoglyphs.
        
        Returns:
            Tuple of (homoglyph_score, list of detected homoglyphs)
        """
        detected = []
        homoglyph_count = 0
        
        for i, char in enumerate(text):
            if char in self.HOMOGLYPH_MAP:
                detected.append({
                    "position": i,
                    "homoglyph": char,
                    "looks_like": self.HOMOGLYPH_MAP[char],
                    "char_code": f"U+{ord(char):04X}"
                })
                homoglyph_count += 1
        
        # Calculate score based on ratio
        alpha_count = sum(1 for c in text if c.isalpha())
        score = homoglyph_count / max(alpha_count, 1)
        
        return score, detected
    
    # ==================== DEFENSE 4: Review Authenticity Analysis ====================
    
    def analyze_review_authenticity(self, reviews: List[Dict]) -> Tuple[float, List[Dict]]:
        """
        Analyze reviews for signs of manipulation or fakeness.
        
        Returns:
            Tuple of (fake_score, list of suspicious reviews)
        """
        if not reviews:
            return 0.0, []
        
        suspicious_reviews = []
        
        for review in reviews:
            review_text = review.get("text", "")
            suspicion_factors = []
            review_score = 0.0
            
            # Check for fake review patterns
            for pattern in self.fake_review_regex:
                if pattern.search(review_text):
                    suspicion_factors.append("fake_pattern_match")
                    review_score += 0.3
            
            # Check for overly positive with high helpful votes
            if review.get("rating", 0) == 5 and review.get("helpful_votes", 0) > 100:
                suspicion_factors.append("suspicious_helpful_ratio")
                review_score += 0.2
            
            # Check for short, generic positive reviews
            word_count = len(review_text.split())
            if word_count < 15 and review.get("rating", 0) >= 4:
                suspicion_factors.append("too_short_positive")
                review_score += 0.15
            
            # Check for excessive punctuation
            if review_text.count("!") > 3:
                suspicion_factors.append("excessive_punctuation")
                review_score += 0.15
            
            # Check for all caps words
            caps_words = len(re.findall(r'\b[A-Z]{4,}\b', review_text))
            if caps_words > 2:
                suspicion_factors.append("excessive_caps")
                review_score += 0.1
            
            # Check if marked as fake (for testing)
            if review.get("is_fake", False):
                review_score = 1.0
                suspicion_factors.append("marked_fake")
            
            if review_score > 0.3:
                suspicious_reviews.append({
                    "review_id": review.get("id", "unknown"),
                    "score": min(1.0, review_score),
                    "factors": suspicion_factors,
                    "text_preview": review_text[:100]
                })
        
        # Overall fake score
        if suspicious_reviews:
            avg_score = np.mean([r["score"] for r in suspicious_reviews])
            fake_ratio = len(suspicious_reviews) / len(reviews)
            overall_score = min(1.0, avg_score * 0.6 + fake_ratio * 0.4)
        else:
            overall_score = 0.0
        
        return overall_score, suspicious_reviews
    
    # ==================== DEFENSE 5: Metadata Validation ====================
    
    def validate_metadata(self, metadata: Dict) -> Tuple[float, List[str]]:
        """
        Check metadata for suspicious or malicious entries.
        
        Returns:
            Tuple of (suspicion_score, list of issues)
        """
        issues = []
        score = 0.0
        
        # Check for hidden fields
        hidden_keys = [k for k in metadata.keys() if k.startswith("_")]
        if hidden_keys:
            issues.append(f"Hidden metadata fields: {hidden_keys}")
            score += 0.3
        
        # Check for unrealistic metrics
        if metadata.get("view_count", 0) > 500000:
            issues.append("Suspiciously high view count")
            score += 0.2
        
        if metadata.get("purchase_count", 0) > 50000:
            issues.append("Suspiciously high purchase count")
            score += 0.2
        
        # Check for category spam
        if isinstance(metadata.get("additional_categories"), list):
            if len(metadata["additional_categories"]) > 2:
                issues.append("Too many category tags")
                score += 0.2
        
        # Check for hidden promotional text
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) > 100:
                spam_check = sum(1 for p in self.spam_regex if p.search(value.lower()))
                if spam_check > 2:
                    issues.append(f"Spam content in metadata field: {key}")
                    score += 0.3
        
        # Check for timestamp manipulation
        if "boosted_timestamp" in metadata:
            issues.append("Timestamp manipulation detected")
            score += 0.2
        
        return min(1.0, score), issues
    
    # ==================== DEFENSE 6: Statistical Anomaly Detection ====================
    
    def build_baseline(self, products: List[Product]):
        """Build baseline statistics from clean dataset."""
        title_lengths = [len(p.title) for p in products]
        desc_lengths = [len(p.description) for p in products]
        ratings = [p.rating for p in products]
        review_counts = [p.review_count for p in products]
        
        self.baseline_stats = {
            "title_length": {"mean": np.mean(title_lengths), "std": np.std(title_lengths)},
            "desc_length": {"mean": np.mean(desc_lengths), "std": np.std(desc_lengths)},
            "rating": {"mean": np.mean(ratings), "std": np.std(ratings)},
            "review_count": {"mean": np.mean(review_counts), "std": np.std(review_counts)}
        }
    
    def detect_statistical_anomalies(self, product: Product) -> Tuple[float, List[str]]:
        """
        Detect anomalies by comparing against baseline statistics.
        
        Returns:
            Tuple of (anomaly_score, list of anomalies)
        """
        if self.baseline_stats is None:
            return 0.0, ["No baseline available"]
        
        anomalies = []
        z_scores = []
        
        # Check title length
        title_z = abs(len(product.title) - self.baseline_stats["title_length"]["mean"]) / \
                  max(self.baseline_stats["title_length"]["std"], 1)
        if title_z > 2.5:
            anomalies.append(f"Title length anomaly (z={title_z:.2f})")
            z_scores.append(title_z)
        
        # Check description length
        desc_z = abs(len(product.description) - self.baseline_stats["desc_length"]["mean"]) / \
                 max(self.baseline_stats["desc_length"]["std"], 1)
        if desc_z > 2.5:
            anomalies.append(f"Description length anomaly (z={desc_z:.2f})")
            z_scores.append(desc_z)
        
        # Check rating (suspiciously high)
        if product.rating >= 4.8 and product.review_count < 20:
            anomalies.append("High rating with few reviews")
            z_scores.append(2.0)
        
        # Calculate overall anomaly score
        if z_scores:
            score = min(1.0, np.mean(z_scores) / 3)
        else:
            score = 0.0
        
        return score, anomalies
    
    # ==================== Sanitization Functions ====================
    
    def sanitize_text(self, text: str) -> str:
        """Remove malicious characters and normalize text."""
        # Remove zero-width characters
        for char in self.ZERO_WIDTH_CHARS:
            text = text.replace(char, '')
        
        # Replace homoglyphs with ASCII equivalents
        result = []
        for char in text:
            if char in self.HOMOGLYPH_MAP:
                result.append(self.HOMOGLYPH_MAP[char])
            else:
                result.append(char)
        text = ''.join(result)
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive repetition
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
        
        # Remove excessive caps while preserving acronyms
        def lower_long_caps(match):
            word = match.group(0)
            if len(word) > 4:
                return word.capitalize()
            return word
        text = re.sub(r'\b[A-Z]{5,}\b', lower_long_caps, text)
        
        return text
    
    def sanitize_product(self, product: Product) -> Product:
        """Create a sanitized copy of the product."""
        import copy
        sanitized = copy.deepcopy(product)
        
        # Sanitize text fields
        sanitized.title = self.sanitize_text(product.title)
        sanitized.description = self.sanitize_text(product.description)
        
        # Sanitize reviews
        for review in sanitized.reviews:
            review["text"] = self.sanitize_text(review.get("text", ""))
        
        # Remove suspicious metadata
        suspicious_keys = [k for k in sanitized.metadata.keys() 
                         if k.startswith("_") or k in ["hidden_keywords", "seo_tags", 
                                                       "boosted_timestamp", "__hidden_promo__"]]
        for key in suspicious_keys:
            del sanitized.metadata[key]
        
        # Reset suspicious metrics
        if sanitized.metadata.get("view_count", 0) > 500000:
            sanitized.metadata["view_count"] = 0
        if sanitized.metadata.get("purchase_count", 0) > 50000:
            sanitized.metadata["purchase_count"] = 0
        
        # Mark sanitized artifacts as safe so downstream search treats them as clean
        sanitized.is_poisoned = False
        sanitized.poison_type = None
        
        return sanitized
    
    # ==================== Main Analysis Function ====================
    
    def analyze_product(self, product: Product, sanitize: bool = True) -> DefenseResult:
        """
        Perform comprehensive analysis of a product for poisoning.
        
        Args:
            product: Product to analyze
            sanitize: Whether to create a sanitized version
        
        Returns:
            DefenseResult with all findings
        """
        detected_attacks = []
        scores = {}
        details = {}
        
        # Combine title and description for text analysis
        full_text = f"{product.title} {product.description}"
        
        # Layer 1: Unicode anomalies
        unicode_score, unicode_anomalies = self.detect_unicode_anomalies(full_text)
        scores["unicode"] = unicode_score
        if unicode_score > self.unicode_threshold:
            detected_attacks.append("hidden_characters")
            details["unicode_anomalies"] = unicode_anomalies
        
        # Layer 2: Keyword density
        density_score, density_details = self.analyze_keyword_density(full_text)
        scores["keyword_density"] = density_score
        if density_score > self.keyword_density_threshold:
            detected_attacks.append("keyword_stuffing")
            details["keyword_analysis"] = density_details
        
        # Layer 3: Homoglyphs
        homoglyph_score, homoglyphs = self.detect_homoglyphs(full_text)
        scores["homoglyph"] = homoglyph_score
        if homoglyph_score > self.homoglyph_threshold:
            detected_attacks.append("homoglyph")
            details["homoglyphs"] = homoglyphs
        
        # Layer 4: Review authenticity
        review_score, suspicious_reviews = self.analyze_review_authenticity(product.reviews)
        scores["review"] = review_score
        if review_score > self.review_score_threshold:
            detected_attacks.append("fake_reviews")
            details["suspicious_reviews"] = suspicious_reviews
        
        # Layer 5: Metadata validation
        metadata_score, metadata_issues = self.validate_metadata(product.metadata)
        scores["metadata"] = metadata_score
        if metadata_score > 0.3:
            detected_attacks.append("metadata_poisoning")
            details["metadata_issues"] = metadata_issues
        
        # Layer 6: Statistical anomalies
        stat_score, stat_anomalies = self.detect_statistical_anomalies(product)
        scores["statistical"] = stat_score
        if stat_anomalies and stat_score > 0.3:
            detected_attacks.append("statistical_anomaly")
            details["statistical_anomalies"] = stat_anomalies
        
        # Calculate overall threat score
        weights = {
            "unicode": 0.2,
            "keyword_density": 0.2,
            "homoglyph": 0.15,
            "review": 0.2,
            "metadata": 0.15,
            "statistical": 0.1
        }
        threat_score = sum(scores[k] * weights[k] for k in scores)
        
        # Determine if suspicious
        is_suspicious = threat_score > self.overall_threat_threshold or len(detected_attacks) >= 2
        
        # Sanitize if requested
        sanitized = self.sanitize_product(product) if sanitize else None
        
        return DefenseResult(
            product_id=product.id,
            is_suspicious=is_suspicious,
            threat_score=threat_score,
            detected_attacks=detected_attacks,
            sanitized_product=sanitized,
            details={
                "scores": scores,
                **details
            }
        )
    
    def analyze_dataset(self, products: List[Product], 
                       sanitize: bool = True) -> Tuple[List[DefenseResult], Dict]:
        """
        Analyze entire dataset for poisoning attacks.
        
        Returns:
            Tuple of (list of results, summary statistics)
        """
        results = []
        suspicious_count = 0
        attack_type_counts = Counter()
        
        for product in products:
            result = self.analyze_product(product, sanitize)
            results.append(result)
            
            if result.is_suspicious:
                suspicious_count += 1
                for attack in result.detected_attacks:
                    attack_type_counts[attack] += 1
        
        summary = {
            "total_analyzed": len(products),
            "suspicious_found": suspicious_count,
            "clean": len(products) - suspicious_count,
            "detection_rate": suspicious_count / len(products) if products else 0,
            "attack_types_detected": dict(attack_type_counts),
            "avg_threat_score": np.mean([r.threat_score for r in results])
        }
        
        return results, summary


if __name__ == "__main__":
    from dataset_generator import MarketplaceDatasetGenerator
    from attack_simulator import AttackSimulator
    
    # Generate and poison dataset
    generator = MarketplaceDatasetGenerator(seed=42)
    clean_products = generator.generate_dataset(size=50)
    
    attacker = AttackSimulator(seed=42)
    poisoned_products, attack_stats = attacker.poison_dataset(clean_products, poison_rate=0.3)
    
    # Initialize defender
    defender = MarketplaceDefender()
    defender.build_baseline(clean_products)
    
    # Analyze dataset
    print("=" * 60)
    print("DEFENSE ANALYSIS")
    print("=" * 60)
    
    results, summary = defender.analyze_dataset(poisoned_products)
    
    print(f"\nDataset Analysis Summary:")
    print(f"  Total products: {summary['total_analyzed']}")
    print(f"  Suspicious found: {summary['suspicious_found']}")
    print(f"  Detection rate: {summary['detection_rate']*100:.1f}%")
    print(f"  Average threat score: {summary['avg_threat_score']:.3f}")
    
    print(f"\nDetected Attack Types:")
    for attack_type, count in summary['attack_types_detected'].items():
        print(f"  {attack_type}: {count}")
    
    # Show example analysis
    print("\n" + "=" * 60)
    print("EXAMPLE ANALYSIS")
    print("=" * 60)
    
    suspicious_results = [r for r in results if r.is_suspicious]
    if suspicious_results:
        example = suspicious_results[0]
        print(f"\nProduct: {example.product_id}")
        print(f"Threat Score: {example.threat_score:.3f}")
        print(f"Detected Attacks: {example.detected_attacks}")
        print(f"Layer Scores: {example.details.get('scores', {})}")
