"""
Evaluation Module for Marketplace Poisoning Shield
Comprehensive metrics for attack effectiveness and defense performance.
"""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

from dataset_generator import Product
from attack_simulator import AttackSimulator
from defense_module import MarketplaceDefender, DefenseResult
from search_pipeline import MarketplaceSearchPipeline, SearchResult, SearchEvaluator


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    dataset_stats: Dict
    attack_effectiveness: Dict
    defense_effectiveness: Dict
    search_quality: Dict
    overall_scores: Dict


class ComprehensiveEvaluator:
    """
    Evaluates the complete attack-defense pipeline.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_attack_effectiveness(self, 
                                      clean_products: List[Product],
                                      poisoned_products: List[Product],
                                      attack_stats: Dict) -> Dict:
        """
        Evaluate how effective the attacks are at evading detection
        and boosting rankings.
        """
        total = len(poisoned_products)
        poisoned_count = sum(1 for p in poisoned_products if p.is_poisoned)
        
        # Calculate visibility changes (using ratings as proxy)
        clean_ratings = {p.id: p.rating for p in clean_products}
        rating_boosts = []
        
        for p in poisoned_products:
            if p.is_poisoned and p.id in clean_ratings:
                boost = p.rating - clean_ratings.get(p.id, p.rating)
                if boost > 0:
                    rating_boosts.append(boost)
        
        return {
            "total_products": total,
            "poisoned_products": poisoned_count,
            "poison_rate": poisoned_count / total if total > 0 else 0,
            "attack_distribution": attack_stats.get("attack_counts", {}),
            "avg_rating_boost": np.mean(rating_boosts) if rating_boosts else 0,
            "max_rating_boost": max(rating_boosts) if rating_boosts else 0,
            "products_with_rating_boost": len(rating_boosts)
        }
    
    def evaluate_defense_effectiveness(self,
                                      poisoned_products: List[Product],
                                      defense_results: List[DefenseResult]) -> Dict:
        """
        Evaluate how well the defense detects poisoned products.
        """
        # Build lookup for ground truth
        ground_truth = {p.id: p.is_poisoned for p in poisoned_products}
        poison_types = {p.id: p.poison_type for p in poisoned_products if p.is_poisoned}
        
        # Calculate metrics
        true_positives = 0  # Correctly identified as poisoned
        false_positives = 0  # Clean but flagged as suspicious
        true_negatives = 0  # Correctly identified as clean
        false_negatives = 0  # Poisoned but not detected
        
        detection_by_type = defaultdict(lambda: {"detected": 0, "missed": 0})
        
        for result in defense_results:
            is_actually_poisoned = ground_truth.get(result.product_id, False)
            
            if result.is_suspicious and is_actually_poisoned:
                true_positives += 1
                ptype = poison_types.get(result.product_id, "unknown")
                detection_by_type[ptype]["detected"] += 1
            elif result.is_suspicious and not is_actually_poisoned:
                false_positives += 1
            elif not result.is_suspicious and not is_actually_poisoned:
                true_negatives += 1
            else:  # not suspicious but actually poisoned
                false_negatives += 1
                ptype = poison_types.get(result.product_id, "unknown")
                detection_by_type[ptype]["missed"] += 1
        
        # Calculate rates
        total_poisoned = true_positives + false_negatives
        total_clean = true_negatives + false_positives
        
        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0
        recall = true_positives / total_poisoned if total_poisoned > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(defense_results) \
            if defense_results else 0
        
        # Calculate per-attack-type detection rates
        per_type_rates = {}
        for ptype, counts in detection_by_type.items():
            total = counts["detected"] + counts["missed"]
            per_type_rates[ptype] = {
                "detection_rate": counts["detected"] / total if total > 0 else 0,
                "detected": counts["detected"],
                "missed": counts["missed"]
            }
        
        return {
            "confusion_matrix": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives
            },
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "false_positive_rate": false_positives / total_clean if total_clean > 0 else 0,
            "false_negative_rate": false_negatives / total_poisoned if total_poisoned > 0 else 0,
            "per_attack_type": per_type_rates
        }
    
    def evaluate_search_quality(self,
                               pipeline: MarketplaceSearchPipeline,
                               test_queries: List[str],
                               k: int = 10) -> Dict:
        """
        Evaluate search quality with and without defense.
        """
        evaluator = SearchEvaluator()
        
        baseline_metrics = {"precision": [], "ndcg": [], "poisoned_in_top_k": []}
        defended_metrics = {"precision": [], "ndcg": [], "poisoned_in_top_k": []}
        
        query_results = []
        
        for query in test_queries:
            comparison = pipeline.compare_search(query, k=k)
            
            # Baseline metrics
            baseline_precision = evaluator.calculate_precision_at_k(
                comparison['baseline']['results'], k=5
            )
            baseline_ndcg = evaluator.calculate_ndcg(
                comparison['baseline']['results'], k=k
            )
            baseline_metrics["precision"].append(baseline_precision)
            baseline_metrics["ndcg"].append(baseline_ndcg)
            baseline_metrics["poisoned_in_top_k"].append(
                comparison['baseline']['poisoned_in_top_5']
            )
            
            # Defended metrics
            defended_precision = evaluator.calculate_precision_at_k(
                comparison['defended']['results'], k=5
            )
            defended_ndcg = evaluator.calculate_ndcg(
                comparison['defended']['results'], k=k
            )
            defended_metrics["precision"].append(defended_precision)
            defended_metrics["ndcg"].append(defended_ndcg)
            defended_metrics["poisoned_in_top_k"].append(
                comparison['defended']['poisoned_in_top_5']
            )
            
            query_results.append({
                "query": query,
                "baseline_poisoned_top5": comparison['baseline']['poisoned_in_top_5'],
                "defended_poisoned_top5": comparison['defended']['poisoned_in_top_5'],
                "improvement": comparison['baseline']['poisoned_in_top_5'] - \
                              comparison['defended']['poisoned_in_top_5']
            })
        
        return {
            "baseline": {
                "avg_precision_at_5": np.mean(baseline_metrics["precision"]),
                "avg_ndcg": np.mean(baseline_metrics["ndcg"]),
                "avg_poisoned_in_top_5": np.mean(baseline_metrics["poisoned_in_top_k"]),
                "total_poisoned_exposure": sum(baseline_metrics["poisoned_in_top_k"])
            },
            "defended": {
                "avg_precision_at_5": np.mean(defended_metrics["precision"]),
                "avg_ndcg": np.mean(defended_metrics["ndcg"]),
                "avg_poisoned_in_top_5": np.mean(defended_metrics["poisoned_in_top_k"]),
                "total_poisoned_exposure": sum(defended_metrics["poisoned_in_top_k"])
            },
            "improvement": {
                "precision_gain": np.mean(defended_metrics["precision"]) - \
                                 np.mean(baseline_metrics["precision"]),
                "poisoned_exposure_reduction": sum(baseline_metrics["poisoned_in_top_k"]) - \
                                              sum(defended_metrics["poisoned_in_top_k"]),
                "poisoned_reduction_rate": (sum(baseline_metrics["poisoned_in_top_k"]) - \
                                           sum(defended_metrics["poisoned_in_top_k"])) / \
                                          max(sum(baseline_metrics["poisoned_in_top_k"]), 1)
            },
            "per_query_results": query_results
        }
    
    def run_full_evaluation(self,
                           dataset_size: int = 200,
                           poison_rate: float = 0.25,
                           test_queries: Optional[List[str]] = None,
                           seed: int = 42) -> EvaluationReport:
        """
        Run complete evaluation pipeline.
        """
        from dataset_generator import MarketplaceDatasetGenerator
        
        if test_queries is None:
            test_queries = [
                "wireless headphones",
                "premium quality electronics",
                "best rated product",
                "professional sports equipment",
                "organic beauty products",
                "smart home devices",
                "comfortable clothing",
                "top seller",
                "high quality",
                "authentic brand"
            ]
        
        print("=" * 60)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Step 1: Generate dataset
        print("\n[1/5] Generating dataset...")
        generator = MarketplaceDatasetGenerator(seed=seed)
        clean_products = generator.generate_dataset(size=dataset_size)
        
        # Step 2: Apply attacks
        print("[2/5] Applying poisoning attacks...")
        attacker = AttackSimulator(seed=seed)
        poisoned_products, attack_stats = attacker.poison_dataset(
            clean_products, poison_rate=poison_rate
        )
        
        # Step 3: Run defense
        print("[3/5] Running defense analysis...")
        defender = MarketplaceDefender()
        defender.build_baseline(clean_products)
        defense_results, defense_summary = defender.analyze_dataset(poisoned_products)
        
        # Step 4: Build search pipeline
        print("[4/5] Building search index...")
        pipeline = MarketplaceSearchPipeline(use_neural_embeddings=False)
        pipeline.set_defender(defender)
        pipeline.index_products(poisoned_products, with_defense=True)
        
        # Step 5: Evaluate
        print("[5/5] Evaluating...")
        
        attack_effectiveness = self.evaluate_attack_effectiveness(
            clean_products, poisoned_products, attack_stats
        )
        
        defense_effectiveness = self.evaluate_defense_effectiveness(
            poisoned_products, defense_results
        )
        
        search_quality = self.evaluate_search_quality(
            pipeline, test_queries, k=10
        )
        
        # Calculate overall scores
        overall_scores = {
            "defense_score": defense_effectiveness["f1_score"],
            "search_protection_score": search_quality["improvement"]["poisoned_reduction_rate"],
            "overall_effectiveness": (
                defense_effectiveness["f1_score"] * 0.5 +
                search_quality["improvement"]["poisoned_reduction_rate"] * 0.3 +
                (1 - defense_effectiveness["false_positive_rate"]) * 0.2
            )
        }
        
        report = EvaluationReport(
            dataset_stats={
                "total_products": dataset_size,
                "poison_rate": poison_rate,
                "test_queries": len(test_queries)
            },
            attack_effectiveness=attack_effectiveness,
            defense_effectiveness=defense_effectiveness,
            search_quality=search_quality,
            overall_scores=overall_scores
        )
        
        return report
    
    def print_report(self, report: EvaluationReport):
        """Pretty print evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        print("\n📊 DATASET STATISTICS")
        print("-" * 40)
        for key, value in report.dataset_stats.items():
            print(f"  {key}: {value}")
        
        print("\n🔴 ATTACK EFFECTIVENESS")
        print("-" * 40)
        ae = report.attack_effectiveness
        print(f"  Poisoned products: {ae['poisoned_products']}/{ae['total_products']} ({ae['poison_rate']*100:.1f}%)")
        print(f"  Products with rating boost: {ae['products_with_rating_boost']}")
        print(f"  Average rating boost: +{ae['avg_rating_boost']:.2f}")
        print("  Attack distribution:")
        for attack, count in ae['attack_distribution'].items():
            print(f"    - {attack}: {count}")
        
        print("\n🔵 DEFENSE EFFECTIVENESS")
        print("-" * 40)
        de = report.defense_effectiveness
        cm = de['confusion_matrix']
        print(f"  Precision: {de['precision']:.3f}")
        print(f"  Recall: {de['recall']:.3f}")
        print(f"  F1 Score: {de['f1_score']:.3f}")
        print(f"  Accuracy: {de['accuracy']:.3f}")
        print(f"  False Positive Rate: {de['false_positive_rate']:.3f}")
        print(f"  False Negative Rate: {de['false_negative_rate']:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {cm['true_positives']} | FP: {cm['false_positives']}")
        print(f"    FN: {cm['false_negatives']} | TN: {cm['true_negatives']}")
        print("\n  Per-Attack Detection Rates:")
        for ptype, rates in de['per_attack_type'].items():
            print(f"    - {ptype}: {rates['detection_rate']*100:.1f}% ({rates['detected']}/{rates['detected']+rates['missed']})")
        
        print("\n🔍 SEARCH QUALITY")
        print("-" * 40)
        sq = report.search_quality
        print("  Baseline (No Defense):")
        print(f"    - Avg Precision@5: {sq['baseline']['avg_precision_at_5']:.3f}")
        print(f"    - Avg Poisoned in Top 5: {sq['baseline']['avg_poisoned_in_top_5']:.2f}")
        print("  Defended:")
        print(f"    - Avg Precision@5: {sq['defended']['avg_precision_at_5']:.3f}")
        print(f"    - Avg Poisoned in Top 5: {sq['defended']['avg_poisoned_in_top_5']:.2f}")
        print("  Improvement:")
        print(f"    - Precision Gain: +{sq['improvement']['precision_gain']:.3f}")
        print(f"    - Poisoned Exposure Reduction: {sq['improvement']['poisoned_exposure_reduction']}")
        print(f"    - Reduction Rate: {sq['improvement']['poisoned_reduction_rate']*100:.1f}%")
        
        print("\n⭐ OVERALL SCORES")
        print("-" * 40)
        os = report.overall_scores
        print(f"  Defense Score (F1): {os['defense_score']:.3f}")
        print(f"  Search Protection: {os['search_protection_score']*100:.1f}%")
        print(f"  Overall Effectiveness: {os['overall_effectiveness']*100:.1f}%")
        
        print("\n" + "=" * 60)
    
    def export_report(self, report: EvaluationReport, filepath: str):
        """Export report to JSON."""
        data = asdict(report)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Report exported to {filepath}")


if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    
    # Run evaluation with different configurations
    report = evaluator.run_full_evaluation(
        dataset_size=150,
        poison_rate=0.25,
        seed=42
    )
    
    # Print results
    evaluator.print_report(report)
    
    # Export
    evaluator.export_report(report, "evaluation_report.json")
