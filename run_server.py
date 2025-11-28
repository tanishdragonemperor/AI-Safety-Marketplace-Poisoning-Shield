#!/usr/bin/env python3
"""
Marketplace Poisoning Shield - Full Stack Server (FIXED VERSION)
=================================================================

This script starts the Flask API backend and opens the frontend in your browser.

Usage:
    python run_server.py

The frontend will automatically open at: http://localhost:5000
"""

import os
import sys
import webbrowser
import threading
import time
import copy
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

# Import our modules
from dataset_generator import MarketplaceDatasetGenerator
from attack_simulator import AttackSimulator
from defense_module import MarketplaceDefender
from search_pipeline import MarketplaceSearchPipeline

app = Flask(__name__, static_folder='frontend')
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
    print("🔄 Initializing system...")
    
    state["generator"] = MarketplaceDatasetGenerator(seed=seed)
    state["attacker"] = AttackSimulator(seed=seed)
    state["defender"] = MarketplaceDefender()
    
    # Generate clean dataset
    print("📦 Generating dataset...")
    state["clean_products"] = state["generator"].generate_dataset(size=dataset_size)
    
    # Build baseline for defender
    state["defender"].build_baseline(state["clean_products"])
    
    # Apply poisoning
    print("☠️  Applying poisoning attacks...")
    state["poisoned_products"], state["attack_stats"] = state["attacker"].poison_dataset(
        state["clean_products"], poison_rate=poison_rate
    )
    
    # Analyze with defense
    print("🛡️  Running defense analysis...")
    state["defense_results"], defense_summary = state["defender"].analyze_dataset(
        state["poisoned_products"]
    )
    
    # Build search pipeline
    print("🔍 Building search index...")
    state["pipeline"] = MarketplaceSearchPipeline(use_neural_embeddings=False)
    state["pipeline"].set_defender(state["defender"])
    state["pipeline"].index_products(state["poisoned_products"], with_defense=True)
    
    state["initialized"] = True
    
    print("✅ System initialized!")
    print(f"   - Products: {dataset_size}")
    print(f"   - Poisoned: {state['attack_stats']['poisoned']}")
    print(f"   - Detection rate: {defense_summary['detection_rate']*100:.1f}%")
    
    return {
        "status": "initialized",
        "dataset_size": dataset_size,
        "poison_rate": poison_rate,
        "poisoned_count": state["attack_stats"]["poisoned"],
        "attack_distribution": state["attack_stats"]["attack_counts"]
    }


# ============== ROUTES ==============

@app.route('/')
def serve_frontend():
    """Serve the frontend HTML."""
    return send_from_directory('frontend', 'index.html')


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
    
    filter_type = request.args.get("filter", "all")
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    
    products = state["poisoned_products"]
    
    if filter_type == "clean":
        products = [p for p in products if not p.is_poisoned]
    elif filter_type == "poisoned":
        products = [p for p in products if p.is_poisoned]
    
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


@app.route('/api/attack/demo', methods=['POST'])
def demo_attack():
    """Demonstrate an attack on a product - FIXED VERSION."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    data = request.get_json() or {}
    attack_type = data.get("attack_type", "hidden_characters")
    
    # Generate a fresh product for demo
    product = state["generator"].generate_product()
    
    # Store original values BEFORE any modifications
    original_title = product.title
    original_description = product.description
    original_rating = product.rating
    original_review_count = product.review_count
    original_metadata = copy.deepcopy(product.metadata) if hasattr(product, 'metadata') else {}
    
    # Create a fresh copy for attacking
    product_copy = copy.deepcopy(product)
    
    # Apply attack based on type
    if attack_type == "hidden_characters":
        # Inject invisible Unicode characters
        poisoned = state["attacker"].inject_hidden_characters(product_copy, intensity=0.4)
        attack_details = {
            "method": "Zero-width Unicode injection",
            "chars_injected": len(poisoned.title.encode()) - len(original_title.encode())
        }
        
    elif attack_type == "keyword_stuffing":
        # Add SEO spam keywords
        poisoned = state["attacker"].stuff_keywords(product_copy, keyword_count=8)
        attack_details = {
            "method": "SEO keyword injection",
            "keywords_added": 8
        }
        
    elif attack_type == "homoglyph":
        # Replace characters with lookalikes
        poisoned = state["attacker"].apply_homoglyphs(product_copy, replacement_rate=0.3)
        # Count actual replacements
        replacements = sum(1 for a, b in zip(original_title, poisoned.title) if a != b)
        attack_details = {
            "method": "Character substitution (Cyrillic/Greek lookalikes)",
            "chars_replaced": replacements
        }
        
    elif attack_type == "fake_reviews":
        # Inject fake positive reviews and boost rating
        poisoned = state["attacker"].inject_fake_reviews(product_copy, fake_count=5, boost_rating=True)
        
        # Ensure rating is boosted visibly (force boost for demo)
        if poisoned.rating <= original_rating:
            boost = min(1.2, 5.0 - original_rating)
            poisoned.rating = min(5.0, round(original_rating + boost, 1))
        
        attack_details = {
            "method": "Fake 5-star review injection",
            "fake_reviews_added": 5,
            "rating_boost": round(poisoned.rating - original_rating, 1)
        }
        
    elif attack_type == "adversarial_paraphrase":
        # Add misleading brand associations
        poisoned = state["attacker"].adversarial_paraphrase(product_copy)
        attack_details = {
            "method": "Misleading brand/quality claims injection",
            "claims_added": ["Better than competitors", "Premium authentic quality"]
        }
        
    elif attack_type == "metadata_poisoning":
        # Poison hidden metadata
        poisoned = state["attacker"].poison_metadata(product_copy)
        attack_details = {
            "method": "Hidden metadata manipulation",
            "fields_poisoned": ["view_count", "hidden_promo", "fake_tags"]
        }
        
    else:
        return jsonify({"error": f"Unknown attack type: {attack_type}"}), 400
    
    # Analyze with defense
    defense_result = state["defender"].analyze_product(poisoned)
    
    # Build response with clear before/after comparison
    response = {
        "attack_type": attack_type,
        "attack_details": attack_details,
        "original": {
            "title": original_title,
            "description": original_description,
            "rating": original_rating,
            "review_count": original_review_count
        },
        "poisoned": {
            "title": poisoned.title,
            "description": poisoned.description,
            "rating": poisoned.rating,
            "review_count": poisoned.review_count,
            "poison_type": attack_type
        },
        "changes": {
            "title_changed": original_title != poisoned.title,
            "title_bytes_original": len(original_title.encode()),
            "title_bytes_poisoned": len(poisoned.title.encode()),
            "title_bytes_diff": len(poisoned.title.encode()) - len(original_title.encode()),
            "description_changed": original_description != poisoned.description,
            "description_length_diff": len(poisoned.description) - len(original_description),
            "rating_changed": original_rating != poisoned.rating,
            "rating_diff": round(poisoned.rating - original_rating, 1),
            "reviews_added": poisoned.review_count - original_review_count
        },
        "defense_result": {
            "detected": defense_result.is_suspicious,
            "threat_score": round(defense_result.threat_score, 3),
            "detected_attacks": defense_result.detected_attacks,
            "layer_scores": defense_result.details.get("scores", {})
        }
    }
    
    return jsonify(response)


@app.route('/api/search', methods=['GET'])
def search():
    """Search products."""
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
                "score": float(r.score),
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
    """Compare search results with and without defense - FIXED VERSION."""
    if not state["initialized"]:
        return jsonify({"error": "System not initialized"}), 400
    
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 10))
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    # Get comparison from pipeline
    comparison = state["pipeline"].compare_search(query, k=limit)
    
    def format_baseline_results(results):
        """Format baseline results - NO detection info (users can't tell what's poisoned)"""
        formatted = []
        for r in results:
            formatted.append({
                "rank": r.rank,
                "score": float(r.score),
                "product_id": r.product.id,
                "title": r.product.title,
                "category": r.product.category,
                "rating": r.product.rating,
                "is_poisoned": r.is_poisoned,  # Ground truth for metrics only
                "poison_type": r.product.poison_type if r.is_poisoned else None
            })
        return formatted
    
    def format_defended_results(results):
        """Format defended results - WITH detection info (defense detects suspicious items)"""
        formatted = []
        for r in results:
            formatted.append({
                "rank": r.rank,
                "score": float(r.score),
                "product_id": r.product.id,
                "title": r.product.title,
                "category": r.product.category,
                "rating": r.product.rating,
                "is_poisoned": r.is_poisoned,  # Ground truth
                "poison_type": r.product.poison_type if r.is_poisoned else None,
                "is_suspicious": r.is_suspicious,  # Defense detection flag
                "threat_score": round(r.threat_score, 3)  # Defense threat score
            })
        return formatted
    
    return jsonify({
        "query": query,
        "baseline": {
            "results": format_baseline_results(comparison["baseline"]["results"]),
            "poisoned_count": comparison["baseline"]["poisoned_count"],
            "poisoned_in_top_5": comparison["baseline"]["poisoned_in_top_5"]
        },
        "defended": {
            "results": format_defended_results(comparison["defended"]["results"]),
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
    false_negatives = sum(1 for p, r in zip(state["poisoned_products"], state["defense_results"])
                         if p.is_poisoned and not r.is_suspicious)
    true_negatives = sum(1 for p, r in zip(state["poisoned_products"], state["defense_results"])
                        if not p.is_poisoned and not r.is_suspicious)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / poisoned if poisoned > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return jsonify({
        "dataset": {
            "total_products": total,
            "poisoned_products": poisoned,
            "clean_products": total - poisoned,
            "poison_rate": round(poisoned / total, 2) if total > 0 else 0
        },
        "attacks": state["attack_stats"]["attack_counts"],
        "defense": {
            "suspicious_flagged": detected,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3)
        },
        "threat_scores": {
            "avg": round(sum(r.threat_score for r in state["defense_results"]) / len(state["defense_results"]), 3) if state["defense_results"] else 0,
            "max": round(max((r.threat_score for r in state["defense_results"]), default=0), 3),
            "min": round(min((r.threat_score for r in state["defense_results"]), default=0), 3)
        }
    })


def open_browser():
    """Open the browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')


if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          MARKETPLACE POISONING SHIELD                                     ║
║          Full Stack Server                                                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize the system
    init_system(dataset_size=100, poison_rate=0.25, seed=42)
    
    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("\n🌐 Starting server at http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
