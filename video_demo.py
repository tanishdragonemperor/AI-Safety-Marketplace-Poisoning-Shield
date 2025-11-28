#!/usr/bin/env python3
"""
VIDEO RECORDING DEMO SCRIPT
============================
Run this while screen recording for your AI Safety class presentation.

Usage: python video_demo.py

This script pauses at each step so you can explain what's happening.
Press ENTER to continue to the next section.
"""

import sys
import os
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset_generator import MarketplaceDatasetGenerator
from attack_simulator import AttackSimulator
from defense_module import MarketplaceDefender
from search_pipeline import MarketplaceSearchPipeline


def slow_print(text, delay=0.02):
    """Print text with typewriter effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def section_break(title):
    """Display a clear section break."""
    print("\n")
    print("=" * 70)
    print(f"   {title}")
    print("=" * 70)
    input("\n   [Press ENTER to continue...]\n")


def main():
    # =========================================================================
    # INTRO
    # =========================================================================
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║          MARKETPLACE POISONING SHIELD                                 ║
    ║                                                                       ║
    ║          Detecting and Defending Against Silent Data                  ║
    ║          Poisoning in AI-Powered Marketplaces                         ║
    ║                                                                       ║
    ║          Project by: Tanish Gupta                                     ║
    ║          Course: Intro to AI Security                                 ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("""
    📋 PROJECT OVERVIEW
    ─────────────────────────────────────────────────────────────────────────
    
    This project addresses a critical AI security vulnerability:
    
    Modern e-commerce platforms like Amazon, eBay, and Alibaba use AI for:
      • Search ranking and product discovery
      • Recommendation systems
      • Review analysis and fraud detection
      • Content moderation
    
    THE PROBLEM: These AI systems learn from USER-PROVIDED DATA.
    
    Attackers can inject POISONED DATA that:
      ✗ Manipulates search rankings (their products appear first)
      ✗ Boosts fraudulent listings
      ✗ Evades content moderation
      ✗ Bypasses fraud detection
    
    The attacks are SILENT - invisible to human moderators!
    
    ─────────────────────────────────────────────────────────────────────────
    """)
    
    section_break("PART 1: UNDERSTANDING THE THREAT")
    
    # =========================================================================
    # PART 1: THE THREAT
    # =========================================================================
    
    print("""
    🔍 PART 1: UNDERSTANDING SILENT DATA POISONING
    ─────────────────────────────────────────────────────────────────────────
    
    Let me show you how these invisible attacks work...
    """)
    
    print("\n    EXAMPLE 1: Hidden Character Injection")
    print("    " + "─" * 50)
    
    clean_text = "Premium Wireless Headphones"
    poisoned_text = "Premium\u200b Wireless\u200b Headphones"
    
    print(f"""
    Clean text:    "{clean_text}"
    Poisoned text: "{poisoned_text}"
    
    They look IDENTICAL, right?
    
    But let's check the bytes:
    """)
    
    time.sleep(1)
    
    print(f"    Clean bytes:    {len(clean_text.encode())} bytes")
    print(f"    Poisoned bytes: {len(poisoned_text.encode())} bytes")
    print(f"    Difference:     +{len(poisoned_text.encode()) - len(clean_text.encode())} hidden bytes!")
    
    print("""
    
    Those extra bytes are ZERO-WIDTH CHARACTERS (\\u200b)
    - Invisible to humans
    - But AI tokenizers see them differently
    - This can manipulate embeddings and search rankings!
    """)
    
    input("\n    [Press ENTER to see more attack types...]\n")
    
    print("""
    EXAMPLE 2: Homoglyph Attack (Character Substitution)
    """ + "─" * 50)
    
    print("""
    Original: "Premium" (Latin alphabet)
    Poisoned: "Prеmium" (with Cyrillic 'е')
    
    The Cyrillic 'е' (U+0435) looks identical to Latin 'e' (U+0065)
    But they have different Unicode code points!
    
    This tricks AI systems that rely on exact text matching.
    """)
    
    section_break("PART 2: DATASET GENERATION")
    
    # =========================================================================
    # PART 2: DATASET
    # =========================================================================
    
    print("""
    📦 PART 2: SYNTHETIC DATASET GENERATION
    ─────────────────────────────────────────────────────────────────────────
    
    First, I created a realistic e-commerce dataset generator.
    
    Let me generate some sample products...
    """)
    
    time.sleep(1)
    
    generator = MarketplaceDatasetGenerator(seed=42)
    products = generator.generate_dataset(size=100)
    
    print(f"""
    ✅ Generated {len(products)} products
    
    Categories: electronics, clothing, home, sports, beauty
    
    Sample products:
    """)
    
    for i, p in enumerate(products[:5]):
        print(f"    {i+1}. {p.title}")
        print(f"       Category: {p.category} | Price: ${p.price} | Rating: {p.rating}⭐")
        print()
    
    print("""
    Each product has:
      • Title and description
      • Category and price
      • Seller information
      • Reviews with ratings
      • Metadata (keywords, shipping, etc.)
    """)
    
    section_break("PART 3: ATTACK SIMULATION (RED TEAM)")
    
    # =========================================================================
    # PART 3: ATTACKS
    # =========================================================================
    
    print("""
    ⚔️  PART 3: ATTACK SIMULATION (RED TEAM)
    ─────────────────────────────────────────────────────────────────────────
    
    Now I'll demonstrate each attack type on a sample product.
    
    I implemented 6 different attack types:
    """)
    
    attacker = AttackSimulator(seed=42)
    sample = products[0]
    
    print(f"""
    📦 TARGET PRODUCT:
       Title: {sample.title}
       Description: {sample.description[:60]}...
       Rating: {sample.rating}⭐
       Reviews: {sample.review_count}
    """)
    
    input("\n    [Press ENTER to run Attack #1: Hidden Characters...]\n")
    
    # Attack 1
    print("    ATTACK 1: HIDDEN CHARACTER INJECTION")
    print("    " + "─" * 50)
    
    original = copy.deepcopy(sample)
    poisoned = attacker.inject_hidden_characters(original, intensity=0.4)
    
    print(f"""
    Original title: "{original.title}"
    Poisoned title: "{poisoned.title}"
    
    Bytes before: {len(original.title.encode())}
    Bytes after:  {len(poisoned.title.encode())}
    Hidden chars: {len(poisoned.title.encode()) - len(original.title.encode())}
    
    ⚠️  The poisoned text looks identical but contains invisible characters
       that can manipulate AI embeddings!
    """)
    
    input("\n    [Press ENTER for Attack #2: Keyword Stuffing...]\n")
    
    # Attack 2
    print("    ATTACK 2: KEYWORD STUFFING")
    print("    " + "─" * 50)
    
    original = copy.deepcopy(sample)
    poisoned = attacker.stuff_keywords(original, keyword_count=10)
    
    print(f"""
    Original description:
    "{original.description[:80]}..."
    
    Poisoned description:
    "{poisoned.description[:120]}..."
    
    ⚠️  SEO spam keywords are injected to artificially boost search rankings!
    """)
    
    input("\n    [Press ENTER for Attack #3: Fake Reviews...]\n")
    
    # Attack 3
    print("    ATTACK 3: FAKE REVIEW INJECTION")
    print("    " + "─" * 50)
    
    original = copy.deepcopy(sample)
    poisoned = attacker.inject_fake_reviews(original, fake_count=5, boost_rating=True)
    
    print(f"""
    Original rating: {original.rating}⭐ ({original.review_count} reviews)
    Boosted rating:  {poisoned.rating}⭐ ({poisoned.review_count} reviews)
    
    Fake reviews injected:
    """)
    
    for review in poisoned.reviews[:3]:
        if review.get('is_fake'):
            print(f'    ★★★★★ "{review["text"][:50]}..."')
    
    print("""
    ⚠️  Fake 5-star reviews artificially inflate the product rating!
    """)
    
    input("\n    [Press ENTER for Attack #4: Homoglyph...]\n")
    
    # Attack 4
    print("    ATTACK 4: HOMOGLYPH ATTACK")
    print("    " + "─" * 50)
    
    original = copy.deepcopy(sample)
    poisoned = attacker.apply_homoglyphs(original, replacement_rate=0.3)
    
    print(f"""
    Original: "{original.title}"
    Poisoned: "{poisoned.title}"
    
    Characters replaced: Latin → Cyrillic/Greek lookalikes
    
    Examples of homoglyphs used:
      • 'a' → 'а' (Cyrillic)
      • 'e' → 'е' (Cyrillic)  
      • 'o' → 'о' (Cyrillic)
      • 'p' → 'р' (Cyrillic)
    
    ⚠️  Text looks identical but has different character codes!
    """)
    
    input("\n    [Press ENTER to poison the full dataset...]\n")
    
    # Poison dataset
    print("    POISONING THE FULL DATASET")
    print("    " + "─" * 50)
    
    poisoned_products, attack_stats = attacker.poison_dataset(products, poison_rate=0.25)
    
    print(f"""
    Dataset size: {attack_stats['total']} products
    Poisoned:     {attack_stats['poisoned']} products ({attack_stats['poisoned']/attack_stats['total']*100:.0f}%)
    Clean:        {attack_stats['clean']} products
    
    Attack distribution:
    """)
    
    for attack_type, count in attack_stats['attack_counts'].items():
        bar = "█" * (count * 3) + "░" * (15 - count * 3)
        print(f"      {attack_type:25s} [{bar}] {count}")
    
    section_break("PART 4: DEFENSE SYSTEM (BLUE TEAM)")
    
    # =========================================================================
    # PART 4: DEFENSE
    # =========================================================================
    
    print("""
    🛡️  PART 4: DEFENSE SYSTEM (BLUE TEAM)
    ─────────────────────────────────────────────────────────────────────────
    
    I implemented a MULTI-LAYER defense system with 6 detection layers:
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  LAYER 1: Unicode Anomaly Detection                                 │
    │           → Scans for zero-width and hidden characters              │
    │           → Flags suspicious Unicode categories                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  LAYER 2: Keyword Density Analysis                                  │
    │           → Detects SEO spam patterns                               │
    │           → Analyzes word repetition frequency                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │  LAYER 3: Homoglyph Detection                                       │
    │           → Maps 50+ known character substitutions                  │
    │           → Calculates substitution ratio                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  LAYER 4: Review Authenticity Checker                               │
    │           → Pattern matching for fake reviews                       │
    │           → Sentiment and helpfulness analysis                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │  LAYER 5: Metadata Validation                                       │
    │           → Detects hidden fields                                   │
    │           → Validates metric plausibility                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  LAYER 6: Statistical Anomaly Detection                             │
    │           → Compares against baseline statistics                    │
    │           → Z-score calculation for outliers                        │
    └─────────────────────────────────────────────────────────────────────┘
    
    This is called "DEFENSE IN DEPTH" - no single layer catches everything,
    but together they provide robust protection!
    """)
    
    input("\n    [Press ENTER to run the defense analysis...]\n")
    
    print("    RUNNING DEFENSE ANALYSIS...")
    print("    " + "─" * 50)
    print()
    
    defender = MarketplaceDefender()
    defender.build_baseline(products)  # Learn what "normal" looks like
    
    results, summary = defender.analyze_dataset(poisoned_products)
    
    print(f"""
    ✅ Analysis complete!
    
    Products analyzed: {summary['total_analyzed']}
    Flagged as suspicious: {summary['suspicious_found']}
    Detection rate: {summary['detection_rate']*100:.1f}%
    Average threat score: {summary['avg_threat_score']:.3f}
    """)
    
    print("    Sample detections:")
    print("    " + "─" * 50)
    
    detected_count = 0
    for p, r in zip(poisoned_products, results):
        if p.is_poisoned and detected_count < 6:
            status = "✅ DETECTED" if r.is_suspicious else "❌ MISSED"
            print(f"      {p.poison_type:25s} | Threat: {r.threat_score:.2f} | {status}")
            detected_count += 1
    
    section_break("PART 5: SEARCH IMPACT DEMONSTRATION")
    
    # =========================================================================
    # PART 5: SEARCH IMPACT
    # =========================================================================
    
    print("""
    🔍 PART 5: SEARCH IMPACT DEMONSTRATION
    ─────────────────────────────────────────────────────────────────────────
    
    This is the key insight: How does poisoning affect REAL USERS?
    
    I built a semantic search pipeline using embeddings + vector similarity
    to simulate how marketplace search works.
    
    Let's compare search results WITH and WITHOUT defense...
    """)
    
    input("\n    [Press ENTER to build search index...]\n")
    
    print("    Building search index...")
    
    pipeline = MarketplaceSearchPipeline(use_neural_embeddings=False)
    pipeline.set_defender(defender)
    pipeline.index_products(poisoned_products, with_defense=True)
    
    print(f"    ✅ Indexed {pipeline.index.size()} products")
    print()
    
    queries = ["premium quality", "best headphones", "top rated"]
    
    for query in queries:
        print(f"\n    SEARCH: \"{query}\"")
        print("    " + "─" * 50)
        
        comparison = pipeline.compare_search(query, k=5)
        
        print("\n    WITHOUT DEFENSE (vulnerable):")
        for r in comparison["baseline"]["results"][:5]:
            marker = "🔴 POISONED" if r.is_poisoned else "🟢 Clean"
            print(f"      {r.rank}. {r.product.title[:35]:35s} | {marker}")
        print(f"      → Poisoned in top 5: {comparison['baseline']['poisoned_in_top_5']}")
        
        print("\n    WITH DEFENSE (protected):")
        for r in comparison["defended"]["results"][:5]:
            marker = "🔴 POISONED" if r.is_poisoned else "🟢 Clean"
            print(f"      {r.rank}. {r.product.title[:35]:35s} | {marker}")
        print(f"      → Poisoned in top 5: {comparison['defended']['poisoned_in_top_5']}")
        
        blocked = comparison['baseline']['poisoned_in_top_5'] - comparison['defended']['poisoned_in_top_5']
        print(f"\n      ✨ Defense blocked {blocked} poisoned products!")
        
        input("\n    [Press ENTER for next search...]\n")
    
    section_break("PART 6: EVALUATION METRICS")
    
    # =========================================================================
    # PART 6: EVALUATION
    # =========================================================================
    
    print("""
    📊 PART 6: EVALUATION METRICS
    ─────────────────────────────────────────────────────────────────────────
    
    How do we measure defense effectiveness?
    
    I use standard ML classification metrics:
    """)
    
    # Calculate metrics
    tp = sum(1 for p, r in zip(poisoned_products, results) if p.is_poisoned and r.is_suspicious)
    fp = sum(1 for p, r in zip(poisoned_products, results) if not p.is_poisoned and r.is_suspicious)
    fn = sum(1 for p, r in zip(poisoned_products, results) if p.is_poisoned and not r.is_suspicious)
    tn = sum(1 for p, r in zip(poisoned_products, results) if not p.is_poisoned and not r.is_suspicious)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(results)
    
    print(f"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                        CONFUSION MATRIX                                ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║                      │  Predicted     │  Predicted     │               ║
    ║                      │  POISONED      │  CLEAN         │               ║
    ║      ────────────────┼────────────────┼────────────────┤               ║
    ║      Actually        │                │                │               ║
    ║      POISONED        │  TP: {tp:4d}      │  FN: {fn:4d}      │               ║
    ║      ────────────────┼────────────────┼────────────────┤               ║
    ║      Actually        │                │                │               ║
    ║      CLEAN           │  FP: {fp:4d}      │  TN: {tn:4d}      │               ║
    ║                                                                        ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║                        KEY METRICS                                     ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║      PRECISION:  {precision*100:5.1f}%   (Of flagged items, how many are      ║
    ║                           actually poisoned?)                          ║
    ║                                                                        ║
    ║      RECALL:     {recall*100:5.1f}%   (Of poisoned items, how many did       ║
    ║                           we catch?)                                   ║
    ║                                                                        ║
    ║      F1 SCORE:   {f1*100:5.1f}%   (Harmonic mean - overall effectiveness)  ║
    ║                                                                        ║
    ║      ACCURACY:   {accuracy*100:5.1f}%   (Overall correct classifications)     ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("    Per-Attack Detection Rates:")
    print("    " + "─" * 50)
    
    attack_detection = {}
    for p, r in zip(poisoned_products, results):
        if p.is_poisoned:
            ptype = p.poison_type
            if ptype not in attack_detection:
                attack_detection[ptype] = {"detected": 0, "total": 0}
            attack_detection[ptype]["total"] += 1
            if r.is_suspicious:
                attack_detection[ptype]["detected"] += 1
    
    for attack, data in attack_detection.items():
        rate = data["detected"] / data["total"] if data["total"] > 0 else 0
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"      {attack:25s} [{bar}] {rate*100:5.1f}%")
    
    section_break("CONCLUSION")
    
    # =========================================================================
    # CONCLUSION
    # =========================================================================
    
    print("""
    🎯 CONCLUSION
    ─────────────────────────────────────────────────────────────────────────
    
    KEY TAKEAWAYS:
    
    1. SILENT DATA POISONING IS A REAL THREAT
       • AI systems learn from user-provided data
       • Attacks can be invisible to human reviewers
       • Real-world impact: biased rankings, fraud, manipulation
    
    2. MULTI-LAYER DEFENSE IS ESSENTIAL
       • No single defense catches all attacks
       • Different attacks require different detectors
       • "Defense in Depth" provides robust protection
    
    3. MEASURABLE RESULTS
       • Precision: {precision*100:.1f}% - Low false alarm rate
       • Recall: {recall*100:.1f}% - High detection rate
       • F1 Score: {f1*100:.1f}% - Strong overall performance
    
    4. PRACTICAL IMPACT
       • Search results protected from manipulation
       • Users see genuine products, not boosted fakes
       • Platform integrity maintained
    
    ─────────────────────────────────────────────────────────────────────────
    
    FUTURE WORK:
    • Neural embedding-based detection (transformers)
    • Real-time streaming defense
    • Adversarial training for robustness
    • Cross-platform attack detection
    
    ─────────────────────────────────────────────────────────────────────────
    
                        Thank you for watching!
    
                        Project: Marketplace Poisoning Shield
                        Author:  Tanish Gupta
                        Course:  Intro to AI Security
    
    ═══════════════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    main()
