#!/usr/bin/env python3
"""
Classroom Presentation Demo Script
==================================

This script provides an interactive demonstration for presenting the
Marketplace Poisoning Shield project in class.

Run: python presentation_demo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset_generator import MarketplaceDatasetGenerator
from attack_simulator import AttackSimulator
from defense_module import MarketplaceDefender
from search_pipeline import MarketplaceSearchPipeline


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def pause(message="Press Enter to continue..."):
    input(f"\n{message}")


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(step_num, description):
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)


def colored(text, color):
    """Simple color support."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'end': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"


def demo_intro():
    clear_screen()
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗███████╗████████╗██████╗ ██╗        ║
║   ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝╚══██╔══╝██╔══██╗██║        ║
║   ██╔████╔██║███████║██████╔╝█████╔╝ █████╗     ██║   ██████╔╝██║        ║
║   ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██╔══╝     ██║   ██╔═══╝ ██║        ║
║   ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗███████╗   ██║   ██║     ███████╗   ║
║   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝     ╚══════╝   ║
║                                                                           ║
║              ██████╗  ██████╗ ██╗███████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ║
║              ██╔══██╗██╔═══██╗██║██╔════╝██╔═══██╗████╗  ██║██║████╗  ██║ ║
║              ██████╔╝██║   ██║██║███████╗██║   ██║██╔██╗ ██║██║██╔██╗ ██║ ║
║              ██╔═══╝ ██║   ██║██║╚════██║██║   ██║██║╚██╗██║██║██║╚██╗██║ ║
║              ██║     ╚██████╔╝██║███████║╚██████╔╝██║ ╚████║██║██║ ╚████║ ║
║              ╚═╝      ╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ║
║                                                                           ║
║                      CLASSROOM DEMONSTRATION                              ║
║                                                                           ║
║              Detecting and Defending Against Silent Data                  ║
║                 Poisoning in AI-Powered Marketplaces                      ║
║                                                                           ║
║                      By: Tanish Gupta                                     ║
║                  Intro to AI Security Course                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nTopics covered in this demo:")
    print("  1. The Problem: Silent Data Poisoning")
    print("  2. Attack Simulation (Red Team)")
    print("  3. Defense Mechanisms (Blue Team)")
    print("  4. Search Impact Demonstration")
    print("  5. Evaluation Results")
    
    pause()


def demo_problem():
    clear_screen()
    print_header("THE PROBLEM: SILENT DATA POISONING")
    
    print("""
Modern e-commerce marketplaces use AI for:
  • Search ranking
  • Product recommendations
  • Content moderation
  • Review analysis

VULNERABILITY: These AI systems learn from user-provided data!

Attackers can inject POISONED DATA that:
  ✗ Manipulates search rankings
  ✗ Boosts fraudulent products
  ✗ Hides malicious content
  ✗ Bypasses content moderation

The attacks are SILENT - they're hard to detect by humans!
    """)
    
    print("\n" + colored("Example of invisible poisoning:", "yellow"))
    
    # Show hidden character example
    clean_text = "Premium Headphones"
    poisoned_text = "Pre\u200bmium\u200b Head\u200bphones"
    
    print(f"\n  Clean text:    '{clean_text}'")
    print(f"  Poisoned text: '{poisoned_text}'")
    print(f"\n  Look identical? YES!")
    print(f"  But bytes differ: {len(clean_text)} → {len(poisoned_text.encode())}")
    print(f"  Hidden zero-width characters change how AI processes the text!")
    
    pause()


def demo_attacks():
    clear_screen()
    print_header("🔴 RED TEAM: ATTACK SIMULATION")
    
    # Initialize
    gen = MarketplaceDatasetGenerator(seed=42)
    attacker = AttackSimulator(seed=42)
    sample = gen.generate_product(category="electronics")
    
    print(f"\n📦 Original Product:")
    print(f"   Title: {sample.title}")
    print(f"   Price: ${sample.price}")
    print(f"   Rating: {sample.rating}")
    
    pause("Press Enter to see Attack #1: Hidden Characters...")
    
    # Attack 1: Hidden Characters
    print_step(1, "HIDDEN CHARACTER INJECTION")
    import copy
    original = copy.deepcopy(sample)
    poisoned = attacker.inject_hidden_characters(original, intensity=0.5)
    
    print(f"   Original title bytes: {len(original.title.encode())}")
    print(f"   Poisoned title bytes: {len(poisoned.title.encode())}")
    print(f"   Added {len(poisoned.title.encode()) - len(original.title.encode())} invisible characters!")
    print(f"\n   Visual comparison:")
    print(f"   Original: '{original.title}'")
    print(f"   Poisoned: '{poisoned.title}'")
    print(colored("   ↳ They look identical but AI sees them differently!", "red"))
    
    pause("Press Enter to see Attack #2: Keyword Stuffing...")
    
    # Attack 2: Keyword Stuffing
    print_step(2, "KEYWORD STUFFING")
    original = copy.deepcopy(sample)
    poisoned = attacker.stuff_keywords(original, keyword_count=8)
    
    print(f"   Original description length: {len(original.description)}")
    print(f"   Poisoned description length: {len(poisoned.description)}")
    print(f"\n   Stuffed keywords boost SEO rankings unfairly!")
    print(f"   Added keywords appear in: {poisoned.description[-80:]}")
    
    pause("Press Enter to see Attack #3: Fake Reviews...")
    
    # Attack 3: Fake Reviews
    print_step(3, "FAKE REVIEW INJECTION")
    original = copy.deepcopy(sample)
    poisoned = attacker.inject_fake_reviews(original, fake_count=5)
    
    print(f"   Original rating: {original.rating}")
    print(f"   Boosted rating:  {poisoned.rating}")
    print(f"   Reviews added:   {poisoned.review_count - original.review_count}")
    print(f"\n   Sample fake review:")
    if poisoned.reviews:
        print(f"   '{poisoned.reviews[0]['text']}'")
    
    pause("Press Enter to see Attack #4: Homoglyph Attack...")
    
    # Attack 4: Homoglyphs
    print_step(4, "HOMOGLYPH ATTACK")
    original = copy.deepcopy(sample)
    poisoned = attacker.apply_homoglyphs(original, replacement_rate=0.3)
    
    print(f"   Original: '{original.title}'")
    print(f"   Poisoned: '{poisoned.title}'")
    print(f"\n   Characters replaced with lookalikes from other alphabets!")
    print(f"   e.g., Latin 'e' → Cyrillic 'е' (looks same, different code)")
    
    pause()


def demo_defense():
    clear_screen()
    print_header("🔵 BLUE TEAM: DEFENSE MECHANISMS")
    
    print("""
Our multi-layer defense system:

  Layer 1: UNICODE ANOMALY DETECTION
           → Scans for zero-width and hidden characters
           → Flags suspicious Unicode categories
           
  Layer 2: KEYWORD DENSITY ANALYSIS
           → Detects SEO spam patterns
           → Analyzes word repetition frequency
           
  Layer 3: HOMOGLYPH DETECTION
           → Maps 50+ known character substitutions
           → Calculates substitution ratio
           
  Layer 4: REVIEW AUTHENTICITY CHECKER
           → Pattern matching for fake reviews
           → Sentiment and helpfulness analysis
           
  Layer 5: METADATA VALIDATION
           → Detects hidden fields
           → Validates metric plausibility
           
  Layer 6: STATISTICAL ANOMALY DETECTION
           → Compares against baseline statistics
           → Z-score calculation for outliers
    """)
    
    pause("Press Enter to see defense in action...")
    
    # Demo defense
    gen = MarketplaceDatasetGenerator(seed=42)
    attacker = AttackSimulator(seed=42)
    defender = MarketplaceDefender()
    
    products = gen.generate_dataset(size=50)
    defender.build_baseline(products)
    
    poisoned_products, stats = attacker.poison_dataset(products, poison_rate=0.3)
    
    print_step("LIVE", "ANALYZING POISONED PRODUCTS")
    
    detected = 0
    missed = 0
    
    for p in poisoned_products:
        if p.is_poisoned:
            result = defender.analyze_product(p)
            if result.is_suspicious:
                detected += 1
                status = colored("✓ DETECTED", "green")
            else:
                missed += 1
                status = colored("✗ MISSED", "red")
            
            print(f"   {p.poison_type:25s} | Threat: {result.threat_score:.2f} | {status}")
            if detected + missed >= 8:
                break
    
    print(f"\n   Detection Summary:")
    print(f"   Detected: {colored(str(detected), 'green')}")
    print(f"   Missed:   {colored(str(missed), 'red')}")
    
    pause()


def demo_search_impact():
    clear_screen()
    print_header("🔍 SEARCH IMPACT DEMONSTRATION")
    
    print("\nThis shows how poisoning affects search results")
    print("and how defense protects users.\n")
    
    # Setup
    gen = MarketplaceDatasetGenerator(seed=42)
    attacker = AttackSimulator(seed=42)
    defender = MarketplaceDefender()
    
    products = gen.generate_dataset(size=50)
    defender.build_baseline(products)
    poisoned_products, _ = attacker.poison_dataset(products, poison_rate=0.25)
    
    pipeline = MarketplaceSearchPipeline(use_neural_embeddings=False)
    pipeline.set_defender(defender)
    pipeline.index_products(poisoned_products, with_defense=True)
    
    query = "premium quality"
    print(f"Search Query: '{query}'")
    print("-" * 50)
    
    comparison = pipeline.compare_search(query, k=5)
    
    print("\n📊 WITHOUT DEFENSE (Users see poisoned products):")
    for r in comparison["baseline"]["results"][:5]:
        marker = colored("⚠ POISONED", "red") if r.is_poisoned else colored("✓ Clean", "green")
        print(f"   {r.rank}. {r.product.title[:35]:35s} | {marker}")
    print(f"   Poisoned in top 5: {colored(str(comparison['baseline']['poisoned_in_top_5']), 'red')}")
    
    print("\n🛡️  WITH DEFENSE (Users protected):")
    for r in comparison["defended"]["results"][:5]:
        marker = colored("⚠ POISONED", "red") if r.is_poisoned else colored("✓ Clean", "green")
        print(f"   {r.rank}. {r.product.title[:35]:35s} | {marker}")
    print(f"   Poisoned in top 5: {colored(str(comparison['defended']['poisoned_in_top_5']), 'green')}")
    
    reduction = comparison['baseline']['poisoned_in_top_5'] - comparison['defended']['poisoned_in_top_5']
    print(f"\n✨ Defense blocked {colored(str(reduction), 'cyan')} poisoned products from top results!")
    
    pause()


def demo_evaluation():
    clear_screen()
    print_header("📈 EVALUATION RESULTS")
    
    # Run evaluation
    gen = MarketplaceDatasetGenerator(seed=42)
    attacker = AttackSimulator(seed=42)
    defender = MarketplaceDefender()
    
    products = gen.generate_dataset(size=100)
    defender.build_baseline(products)
    poisoned_products, stats = attacker.poison_dataset(products, poison_rate=0.25)
    results, summary = defender.analyze_dataset(poisoned_products)
    
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
╔════════════════════════════════════════════════════════════════════╗
║                     CONFUSION MATRIX                               ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║              │  Predicted    │  Predicted    │                     ║
║              │  Poisoned     │  Clean        │                     ║
║  ────────────┼───────────────┼───────────────┤                     ║
║  Actually    │               │               │                     ║
║  Poisoned    │  TP: {tp:4d}     │  FN: {fn:4d}     │                     ║
║  ────────────┼───────────────┼───────────────┤                     ║
║  Actually    │               │               │                     ║
║  Clean       │  FP: {fp:4d}     │  TN: {tn:4d}     │                     ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                      KEY METRICS                                   ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Precision:  {precision:.1%}  (How many flagged are truly poisoned)    ║
║  Recall:     {recall:.1%}  (How many poisoned were caught)             ║
║  F1 Score:   {f1:.1%}  (Overall detection effectiveness)           ║
║  Accuracy:   {accuracy:.1%}  (Overall correctness)                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📊 Per-Attack Detection Rates:")
    attack_stats = {}
    for p, r in zip(poisoned_products, results):
        if p.is_poisoned:
            ptype = p.poison_type
            if ptype not in attack_stats:
                attack_stats[ptype] = {"detected": 0, "total": 0}
            attack_stats[ptype]["total"] += 1
            if r.is_suspicious:
                attack_stats[ptype]["detected"] += 1
    
    for attack, data in attack_stats.items():
        rate = data["detected"] / data["total"] if data["total"] > 0 else 0
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"   {attack:25s} [{bar}] {rate:.0%}")
    
    pause()


def demo_conclusion():
    clear_screen()
    print_header("🎯 CONCLUSION")
    
    print("""
KEY TAKEAWAYS:
══════════════

1. SILENT DATA POISONING IS A REAL THREAT
   • AI systems learn from user-provided data
   • Attacks can be invisible to human reviewers
   • Impact: biased rankings, fraud, manipulation

2. MULTI-LAYER DEFENSE IS ESSENTIAL
   • No single defense catches all attacks
   • Different attacks require different detectors
   • Defense in depth provides robust protection

3. EVALUATION METRICS MATTER
   • Precision: Avoid false alarms
   • Recall: Don't miss real attacks
   • F1 Score: Balance both concerns

4. TRADE-OFFS EXIST
   • Security vs. User Experience
   • Detection Rate vs. False Positives
   • Performance vs. Thoroughness


FUTURE WORK:
════════════
• Neural embedding-based detection
• Real-time streaming defense
• Adversarial training for robustness
• Cross-platform attack detection
    """)
    
    print("\n" + "=" * 60)
    print("  Thank you for watching!")
    print("  Questions?")
    print("=" * 60)
    
    print("\n\nProject by: Tanish Gupta")
    print("Course: Intro to AI Security")
    print("\nGitHub: [Project Repository]")
    print("Demo: Run 'python main.py' for full demo")


def main():
    """Run the presentation demo."""
    try:
        demo_intro()
        demo_problem()
        demo_attacks()
        demo_defense()
        demo_search_impact()
        demo_evaluation()
        demo_conclusion()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Thanks for watching!")


if __name__ == "__main__":
    main()
