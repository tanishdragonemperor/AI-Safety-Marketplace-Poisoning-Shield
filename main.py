#!/usr/bin/env python3
"""
Marketplace Poisoning Shield - Main Runner
Complete demonstration of attack simulation and defense mechanisms.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset_generator import MarketplaceDatasetGenerator
from attack_simulator import AttackSimulator
from defense_module import MarketplaceDefender
from search_pipeline import MarketplaceSearchPipeline, SearchEvaluator
from evaluation import ComprehensiveEvaluator


def print_banner():
    """Print the project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗███████╗████████╗██████╗ ██╗      ██████╗  ║
║   ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝╚══██╔══╝██╔══██╗██║     ██╔════╝  ║
║   ██╔████╔██║███████║██████╔╝█████╔╝ █████╗     ██║   ██████╔╝██║     █████╗    ║
║   ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██╔══╝     ██║   ██╔═══╝ ██║     ██╔══╝    ║
║   ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗███████╗   ██║   ██║     ███████╗███████╗  ║
║   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝     ╚══════╝╚══════╝  ║
║                                                                              ║
║              ██████╗  ██████╗ ██╗███████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗ ║
║              ██╔══██╗██╔═══██╗██║██╔════╝██╔═══██╗████╗  ██║██║████╗  ██║██╔════╝ ║
║              ██████╔╝██║   ██║██║███████╗██║   ██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗║
║              ██╔═══╝ ██║   ██║██║╚════██║██║   ██║██║╚██╗██║██║██║╚██╗██║██║   ██║║
║              ██║     ╚██████╔╝██║███████║╚██████╔╝██║ ╚████║██║██║ ╚████║╚██████╔╝║
║              ╚═╝      ╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ║
║                                                                              ║
║                        ███████╗██╗  ██╗██╗███████╗██╗     ██████╗            ║
║                        ██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗           ║
║                        ███████╗███████║██║█████╗  ██║     ██║  ██║           ║
║                        ╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║           ║
║                        ███████║██║  ██║██║███████╗███████╗██████╔╝           ║
║                        ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝            ║
║                                                                              ║
║                 Detecting and Defending Against Silent Data Poisoning        ║
║                         in AI-Powered Marketplaces                           ║
║                                                                              ║
║                        AI Security Course Project                            ║
║                           By: Tanish Gupta                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_attacks(attacker: AttackSimulator, sample_product):
    """Demonstrate all attack types."""
    print("\n" + "=" * 70)
    print("🔴 RED TEAM: ATTACK DEMONSTRATIONS")
    print("=" * 70)
    
    print(f"\n📦 Original Product:")
    print(f"   Title: {sample_product.title}")
    print(f"   Description: {sample_product.description[:80]}...")
    print(f"   Rating: {sample_product.rating}")
    
    attacks = [
        ("Hidden Characters", attacker.inject_hidden_characters, {"intensity": 0.3}),
        ("Keyword Stuffing", attacker.stuff_keywords, {"keyword_count": 8}),
        ("Homoglyph Attack", attacker.apply_homoglyphs, {"replacement_rate": 0.2}),
        ("Fake Reviews", attacker.inject_fake_reviews, {"fake_count": 5}),
        ("Adversarial Paraphrase", attacker.adversarial_paraphrase, {}),
        ("Metadata Poisoning", attacker.poison_metadata, {}),
    ]
    
    for name, attack_func, kwargs in attacks:
        print(f"\n{'─' * 70}")
        print(f"⚔️  Attack: {name}")
        print("─" * 70)
        
        import copy
        original = copy.deepcopy(sample_product)
        poisoned = attack_func(original, **kwargs)
        
        print(f"   Poison Type: {poisoned.poison_type}")
        
        if name == "Hidden Characters":
            orig_bytes = len(original.title.encode())
            new_bytes = len(poisoned.title.encode())
            print(f"   Title bytes: {orig_bytes} → {new_bytes} (+{new_bytes - orig_bytes} hidden)")
            print(f"   Visual: '{poisoned.title}' (looks normal but has hidden chars)")
        
        elif name == "Keyword Stuffing":
            print(f"   Modified description: {poisoned.description[:100]}...")
        
        elif name == "Homoglyph Attack":
            print(f"   Original: {original.title}")
            print(f"   Poisoned: {poisoned.title}")
            print(f"   Characters differ: {original.title != poisoned.title}")
        
        elif name == "Fake Reviews":
            print(f"   Original rating: {original.rating} → New rating: {poisoned.rating}")
            print(f"   Reviews added: {poisoned.review_count - original.review_count}")
            if poisoned.reviews:
                print(f"   Sample fake review: '{poisoned.reviews[0]['text'][:60]}...'")
        
        elif name == "Adversarial Paraphrase":
            print(f"   Added claims: {poisoned.description[-100:]}")
        
        elif name == "Metadata Poisoning":
            print(f"   Hidden fields added: {[k for k in poisoned.metadata.keys() if k.startswith('_')]}")
            print(f"   Fake metrics: view_count={poisoned.metadata.get('view_count', 'N/A')}")


def demo_defense(defender: MarketplaceDefender, poisoned_products):
    """Demonstrate defense capabilities."""
    print("\n" + "=" * 70)
    print("🔵 BLUE TEAM: DEFENSE DEMONSTRATIONS")
    print("=" * 70)
    
    # Find poisoned products to analyze
    poisoned_samples = [p for p in poisoned_products if p.is_poisoned][:5]
    
    print(f"\n🛡️  Analyzing {len(poisoned_samples)} poisoned products...\n")
    
    for product in poisoned_samples:
        result = defender.analyze_product(product)
        
        status = "🚨 DETECTED" if result.is_suspicious else "✅ Missed"
        print(f"{'─' * 70}")
        print(f"Product: {product.title[:50]}...")
        print(f"  Actual poison: {product.poison_type}")
        print(f"  Defense result: {status}")
        print(f"  Threat score: {result.threat_score:.3f}")
        print(f"  Detected attacks: {result.detected_attacks}")
        
        if result.details.get("scores"):
            scores = result.details["scores"]
            print(f"  Layer scores:")
            for layer, score in scores.items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"    {layer:20s}: [{bar}] {score:.3f}")


def demo_search(pipeline: MarketplaceSearchPipeline, queries: list):
    """Demonstrate search impact."""
    print("\n" + "=" * 70)
    print("🔍 SEARCH IMPACT DEMONSTRATION")
    print("=" * 70)
    
    for query in queries[:3]:
        print(f"\n{'─' * 70}")
        print(f"Query: '{query}'")
        print("─" * 70)
        
        comparison = pipeline.compare_search(query, k=5)
        
        print("\n📊 WITHOUT DEFENSE (Baseline):")
        for r in comparison["baseline"]["results"][:5]:
            poison_marker = "🔴" if r.is_poisoned else "🟢"
            print(f"   {r.rank}. {poison_marker} {r.product.title[:40]}... (score: {r.score:.3f})")
        print(f"   Poisoned in top 5: {comparison['baseline']['poisoned_in_top_5']}")
        
        print("\n🛡️  WITH DEFENSE (Protected):")
        for r in comparison["defended"]["results"][:5]:
            poison_marker = "🔴" if r.is_poisoned else "🟢"
            print(f"   {r.rank}. {poison_marker} {r.product.title[:40]}... (score: {r.score:.3f})")
        print(f"   Poisoned in top 5: {comparison['defended']['poisoned_in_top_5']}")
        
        improvement = comparison['baseline']['poisoned_in_top_5'] - comparison['defended']['poisoned_in_top_5']
        print(f"\n   ✨ Defense blocked {improvement} poisoned products from top 5")


def run_full_demo(dataset_size: int = 100, poison_rate: float = 0.25, seed: int = 42):
    """Run the complete demonstration."""
    print_banner()
    
    print("\n" + "=" * 70)
    print("📊 PHASE 1: DATASET GENERATION")
    print("=" * 70)
    
    generator = MarketplaceDatasetGenerator(seed=seed)
    clean_products = generator.generate_dataset(size=dataset_size)
    
    print(f"\n✅ Generated {len(clean_products)} clean products")
    print(f"   Categories: {set(p.category for p in clean_products)}")
    
    # Sample product for attack demos
    sample_product = clean_products[0]
    
    print("\n" + "=" * 70)
    print("☠️  PHASE 2: ATTACK SIMULATION")
    print("=" * 70)
    
    attacker = AttackSimulator(seed=seed)
    poisoned_products, attack_stats = attacker.poison_dataset(
        clean_products, poison_rate=poison_rate
    )
    
    print(f"\n✅ Poisoned {attack_stats['poisoned']}/{attack_stats['total']} products ({poison_rate*100:.0f}%)")
    print(f"   Attack distribution:")
    for attack, count in attack_stats['attack_counts'].items():
        print(f"   - {attack}: {count}")
    
    # Demo individual attacks
    demo_attacks(attacker, sample_product)
    
    print("\n" + "=" * 70)
    print("🛡️  PHASE 3: DEFENSE ANALYSIS")
    print("=" * 70)
    
    defender = MarketplaceDefender()
    defender.build_baseline(clean_products)
    
    defense_results, defense_summary = defender.analyze_dataset(poisoned_products)
    
    print(f"\n✅ Defense Analysis Complete")
    print(f"   Products flagged as suspicious: {defense_summary['suspicious_found']}")
    print(f"   Detection rate: {defense_summary['detection_rate']*100:.1f}%")
    print(f"   Average threat score: {defense_summary['avg_threat_score']:.3f}")
    
    # Demo defense on specific products
    demo_defense(defender, poisoned_products)
    
    print("\n" + "=" * 70)
    print("🔍 PHASE 4: SEARCH PIPELINE")
    print("=" * 70)
    
    pipeline = MarketplaceSearchPipeline(use_neural_embeddings=False)
    pipeline.set_defender(defender)
    pipeline.index_products(poisoned_products, with_defense=True)
    
    print(f"\n✅ Indexed {pipeline.index.size()} products")
    print(f"   Sanitized index: {pipeline.sanitized_index.size()} products")
    
    # Demo search
    test_queries = [
        "premium headphones",
        "best quality",
        "wireless electronics",
        "top rated product"
    ]
    demo_search(pipeline, test_queries)
    
    print("\n" + "=" * 70)
    print("📈 PHASE 5: COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    evaluator = ComprehensiveEvaluator()
    
    # Calculate metrics
    # True positives: poisoned and detected
    # False positives: clean but flagged
    true_positives = sum(1 for p, r in zip(poisoned_products, defense_results)
                        if p.is_poisoned and r.is_suspicious)
    false_positives = sum(1 for p, r in zip(poisoned_products, defense_results)
                         if not p.is_poisoned and r.is_suspicious)
    false_negatives = sum(1 for p, r in zip(poisoned_products, defense_results)
                         if p.is_poisoned and not r.is_suspicious)
    true_negatives = sum(1 for p, r in zip(poisoned_products, defense_results)
                        if not p.is_poisoned and not r.is_suspicious)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(defense_results)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         EVALUATION RESULTS                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  CONFUSION MATRIX                                                    ║
║  ┌─────────────────────┬─────────────────────┐                       ║
║  │  TP: {true_positives:4d}            │  FP: {false_positives:4d}            │                       ║
║  │  (Correctly caught) │  (False alarms)     │                       ║
║  ├─────────────────────┼─────────────────────┤                       ║
║  │  FN: {false_negatives:4d}            │  TN: {true_negatives:4d}            │                       ║
║  │  (Missed attacks)   │  (Correctly clean)  │                       ║
║  └─────────────────────┴─────────────────────┘                       ║
║                                                                      ║
║  METRICS                                                             ║
║  ├── Precision:  {precision:.3f}  (How many flagged are actually poisoned)  ║
║  ├── Recall:     {recall:.3f}  (How many poisoned were caught)             ║
║  ├── F1 Score:   {f1:.3f}  (Harmonic mean of precision & recall)       ║
║  └── Accuracy:   {accuracy:.3f}  (Overall correct classifications)         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset_size": dataset_size,
            "poison_rate": poison_rate,
            "seed": seed
        },
        "attack_stats": attack_stats,
        "defense_summary": defense_summary,
        "metrics": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        }
    }
    
    output_path = "data/evaluation_results.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    print("\n" + "=" * 70)
    print("🎯 DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("""
Next Steps:
1. Run the API server: python src/api.py
2. Open the React demo: demo/App.jsx
3. Explore attack simulations and defense mechanisms
4. Customize thresholds and parameters for your use case

For more information, see README.md
    """)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Marketplace Poisoning Shield - AI Security Defense System"
    )
    parser.add_argument(
        "--dataset-size", type=int, default=100,
        help="Number of products to generate (default: 100)"
    )
    parser.add_argument(
        "--poison-rate", type=float, default=0.25,
        help="Proportion of products to poison (default: 0.25)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Start the API server instead of running demo"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run comprehensive evaluation only"
    )
    
    args = parser.parse_args()
    
    if args.api:
        print("Starting API server...")
        from api import app, init_system
        init_system(args.dataset_size, args.poison_rate, args.seed)
        app.run(host='0.0.0.0', port=5000, debug=True)
    elif args.evaluate:
        print_banner()
        evaluator = ComprehensiveEvaluator()
        report = evaluator.run_full_evaluation(
            dataset_size=args.dataset_size,
            poison_rate=args.poison_rate,
            seed=args.seed
        )
        evaluator.print_report(report)
        evaluator.export_report(report, "data/full_evaluation_report.json")
    else:
        run_full_demo(args.dataset_size, args.poison_rate, args.seed)


if __name__ == "__main__":
    main()
