"""
Flask API Backend for Marketplace Poisoning Shield Demo
Provides REST endpoints for red team (attacks) and blue team (defenses).
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dataclasses import asdict

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_generator import MarketplaceDatasetGenerator, Product
from attack_simulator import AttackSimulator
from defense_module import MarketplaceDefender
from search_pipeline import MarketplaceSearchPipeline
from evaluation import ComprehensiveEvaluator

app = Flask(__name__)
CORS(app)

# Global state
state = {
    "generator": None,
    "attacker": None,
    "defender": None,
    "pipeline": None,
    "clean_products": [],
    "poisoned_products": [],
    "attack_stats": {},
    "defense_results": [],
    "initialized": False
}


def init_system(dataset_size: int = 100, poison_rate: float = 0.25, seed: int = 42):
    """Initialize the complete system."""
    state["generator"] = MarketplaceDatasetGenerator(seed=seed)
    state["attacker"] = AttackSimulator(seed=seed)
    state["defender"] = MarketplaceDefender()
    
    # Generate clean dataset
    state["clean_products"] = state["generator"].generate_dataset(size=dataset_size)
    
    # Build baseline for defender
    state["defender"].build_baseline(state["clean_products"])
    
    # Apply poisoning
    state["poisoned_products"], state["attack_stats"] = state["attacker"].poison_dataset(
        state["clean_products"], poison_rate=poison_rate
    )
    
    # Analyze with defense
    state["defense_results"], defense_summary = state["defender"].analyze_dataset(
        state["poisoned_products"]
    )
    
    # Build search pipeline
    state["pipeline"] = MarketplaceSearchPipeline(use_neural_embeddings=False)
    state["pipeline"].set_defender(state["defender"])
    state["pipeline"].index_products(state["poisoned_products"], with_defense=True)
    
    state["initialized"] = True
    
    return {
        "status": "initialized",
        "dataset_size": dataset_size,
        "poison_rate": poison_rate,
        "poisoned_count": state["attack_stats"]["poisoned"],
        "attack_distribution": state["attack_stats"]["attack_counts"]
    }


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        "initialized": state["initialized"],
        "products_count": len(state["poisoned_products"]) if state["initialized"] else 0,
        "poisoned_count": state["attack_stats"].get("poisoned", 0) if state["initialized"] else 0
    })


@app.route('/api/init', methods=['POST'])
def initialize():
    """Initialize the system with configuration."""
    data = request.get_json() or {}
    result = init_system(
        dataset_size=data.get("dataset_size", 100),
        poison_rate=data.get("poison_rate", 0.25),
        seed=data.get("seed", 42)
    )
    return jsonify(result)


@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products with optional filtering."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    filter_type = request.args.get("filter", "all")  # all, clean, poisoned
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    
    products = state["poisoned_products"]
    
    if filter_type == "clean":
        products = [p for p in products if not p.is_poisoned]
    elif filter_type == "poisoned":
        products = [p for p in products if p.is_poisoned]
    
    # Get defense results for these products
    defense_map = {r.product_id: r for r in state["defense_results"]}
    
    result = []
    for p in products[offset:offset+limit]:
        defense = defense_map.get(p.id)
        result.append({
            "id": p.id,
            "title": p.title,
            "description": p.description[:200] + "..." if len(p.description) > 200 else p.description,
            "category": p.category,
            "price": p.price,
            "rating": p.rating,
            "review_count": p.review_count,
            "is_poisoned": p.is_poisoned,
            "poison_type": p.poison_type,
            "defense_result": {
                "is_suspicious": defense.is_suspicious if defense else False,
                "threat_score": defense.threat_score if defense else 0,
                "detected_attacks": defense.detected_attacks if defense else []
            } if defense else None
        })
    
    return jsonify({
        "products": result,
        "total": len(products),
        "offset": offset,
        "limit": limit
    })


@app.route('/api/product/<product_id>', methods=['GET'])
def get_product(product_id):
    """Get detailed product information."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    product = next((p for p in state["poisoned_products"] if p.id == product_id), None)
    if not product:
        return jsonify({"error": "Product not found"}), 404
    
    defense = next((r for r in state["defense_results"] if r.product_id == product_id), None)
    
    return jsonify({
        "product": asdict(product),
        "defense_result": {
            "is_suspicious": defense.is_suspicious,
            "threat_score": defense.threat_score,
            "detected_attacks": defense.detected_attacks,
            "details": defense.details
        } if defense else None
    })


@app.route('/api/attack/demo', methods=['POST'])
def demo_attack():
    """Demonstrate an attack on a specific product or new product."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    data = request.get_json() or {}
    attack_type = data.get("attack_type", "hidden_characters")
    product_id = data.get("product_id")
    
    # Get or generate product
    if product_id:
        product = next((p for p in state["clean_products"] if p.id == product_id), None)
        if not product:
            return jsonify({"error": "Product not found"}), 404
    else:
        product = state["generator"].generate_product()
    
    # Apply attack
    import copy
    original = copy.deepcopy(product)
    
    if attack_type == "hidden_characters":
        poisoned = state["attacker"].inject_hidden_characters(product, intensity=0.3)
    elif attack_type == "keyword_stuffing":
        poisoned = state["attacker"].stuff_keywords(product, keyword_count=10)
    elif attack_type == "homoglyph":
        poisoned = state["attacker"].apply_homoglyphs(product, replacement_rate=0.2)
    elif attack_type == "fake_reviews":
        poisoned = state["attacker"].inject_fake_reviews(product, fake_count=5)
    elif attack_type == "adversarial_paraphrase":
        poisoned = state["attacker"].adversarial_paraphrase(product)
    elif attack_type == "metadata_poisoning":
        poisoned = state["attacker"].poison_metadata(product)
    elif attack_type == "composite":
        poisoned = state["attacker"].composite_attack(product)
    else:
        return jsonify({"error": f"Unknown attack type: {attack_type}"}), 400
    
    # Analyze with defense
    defense_result = state["defender"].analyze_product(poisoned)
    
    return jsonify({
        "attack_type": attack_type,
        "original": {
            "title": original.title,
            "description": original.description,
            "rating": original.rating,
            "review_count": original.review_count
        },
        "poisoned": {
            "title": poisoned.title,
            "description": poisoned.description,
            "rating": poisoned.rating,
            "review_count": poisoned.review_count,
            "poison_type": poisoned.poison_type
        },
        "changes": {
            "title_changed": original.title != poisoned.title,
            "title_bytes_diff": len(poisoned.title.encode()) - len(original.title.encode()),
            "description_changed": original.description != poisoned.description,
            "rating_changed": original.rating != poisoned.rating,
            "rating_diff": poisoned.rating - original.rating
        },
        "defense_result": {
            "detected": defense_result.is_suspicious,
            "threat_score": defense_result.threat_score,
            "detected_attacks": defense_result.detected_attacks,
            "scores": defense_result.details.get("scores", {})
        }
    })


@app.route('/api/defense/analyze', methods=['POST'])
def analyze_text():
    """Analyze arbitrary text for poisoning indicators."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    data = request.get_json() or {}
    title = data.get("title", "")
    description = data.get("description", "")
    
    # Create temporary product for analysis
    temp_product = Product(
        id="temp",
        title=title,
        description=description,
        category="electronics",
        price=0,
        seller_id="temp",
        rating=4.0,
        review_count=0,
        reviews=[],
        metadata={},
        created_at=""
    )
    
    result = state["defender"].analyze_product(temp_product)
    
    return jsonify({
        "is_suspicious": result.is_suspicious,
        "threat_score": result.threat_score,
        "detected_attacks": result.detected_attacks,
        "details": result.details,
        "sanitized": {
            "title": result.sanitized_product.title if result.sanitized_product else title,
            "description": result.sanitized_product.description if result.sanitized_product else description
        }
    })


@app.route('/api/search', methods=['GET'])
def search():
    """Search products with optional defense."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    query = request.args.get("q", "")
    use_defense = request.args.get("defense", "false").lower() == "true"
    limit = int(request.args.get("limit", 10))
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    results = state["pipeline"].search(query, k=limit, use_defense=use_defense)
    
    return jsonify({
        "query": query,
        "defense_enabled": use_defense,
        "results": [
            {
                "rank": r.rank,
                "score": r.score,
                "product": {
                    "id": r.product.id,
                    "title": r.product.title,
                    "description": r.product.description[:150] + "...",
                    "category": r.product.category,
                    "price": r.product.price,
                    "rating": r.product.rating
                },
                "is_poisoned": r.is_poisoned,
                "poison_type": r.product.poison_type if r.is_poisoned else None
            }
            for r in results
        ],
        "poisoned_count": len([r for r in results if r.is_poisoned])
    })


@app.route('/api/search/compare', methods=['GET'])
def compare_search():
    """Compare search results with and without defense."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 10))
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    comparison = state["pipeline"].compare_search(query, k=limit)
    
    # Format results
    def format_results(results, include_detection=False):
        formatted = []
        for r in results:
            item = {
                "rank": r.rank,
                "score": r.score,
                "product_id": r.product.id,
                "title": r.product.title,
                "is_poisoned": r.is_poisoned,  # Ground truth (for metrics)
                "poison_type": r.product.poison_type if r.is_poisoned else None
            }
            # Only include detection info for defended results
            if include_detection:
                item["is_suspicious"] = r.is_suspicious  # Defense detected this
                item["threat_score"] = r.threat_score
            formatted.append(item)
        return formatted
    
    return jsonify({
        "query": query,
        "baseline": {
            "results": format_results(comparison["baseline"]["results"], include_detection=False),
            "poisoned_count": comparison["baseline"]["poisoned_count"],
            "poisoned_in_top_5": comparison["baseline"]["poisoned_in_top_5"]
        },
        "defended": {
            "results": format_results(comparison["defended"]["results"], include_detection=True),
            "poisoned_count": comparison["defended"]["poisoned_count"],
            "poisoned_in_top_5": comparison["defended"]["poisoned_in_top_5"]
        },
        "impact": comparison["defense_impact"],
        "rank_changes": comparison["rank_changes"]
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    total = len(state["poisoned_products"])
    poisoned = sum(1 for p in state["poisoned_products"] if p.is_poisoned)
    detected = sum(1 for r in state["defense_results"] if r.is_suspicious)
    
    # Calculate detection accuracy
    true_positives = sum(1 for p, r in zip(state["poisoned_products"], state["defense_results"])
                        if p.is_poisoned and r.is_suspicious)
    false_positives = sum(1 for p, r in zip(state["poisoned_products"], state["defense_results"])
                         if not p.is_poisoned and r.is_suspicious)
    
    precision = true_positives / detected if detected > 0 else 0
    recall = true_positives / poisoned if poisoned > 0 else 0
    
    return jsonify({
        "dataset": {
            "total_products": total,
            "poisoned_products": poisoned,
            "clean_products": total - poisoned,
            "poison_rate": poisoned / total if total > 0 else 0
        },
        "attacks": state["attack_stats"]["attack_counts"],
        "defense": {
            "suspicious_flagged": detected,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        },
        "threat_scores": {
            "avg": sum(r.threat_score for r in state["defense_results"]) / len(state["defense_results"]),
            "max": max(r.threat_score for r in state["defense_results"]),
            "min": min(r.threat_score for r in state["defense_results"])
        }
    })


@app.route('/api/evaluate', methods=['POST'])
def run_evaluation():
    """Run comprehensive evaluation."""
    data = request.get_json() or {}
    
    evaluator = ComprehensiveEvaluator()
    report = evaluator.run_full_evaluation(
        dataset_size=data.get("dataset_size", 100),
        poison_rate=data.get("poison_rate", 0.25),
        seed=data.get("seed", 42)
    )
    
    return jsonify(asdict(report))


# Auto-initialize on startup
@app.before_request
def ensure_initialized():
    if not state["initialized"] and request.endpoint not in ['initialize', 'get_status']:
        init_system(dataset_size=100, poison_rate=0.25, seed=42)


if __name__ == '__main__':
    print("Starting Marketplace Poisoning Shield API...")
    init_system(dataset_size=100, poison_rate=0.25, seed=42)
    app.run(host='0.0.0.0', port=5000, debug=True)
