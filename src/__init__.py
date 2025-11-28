"""
Marketplace Poisoning Shield
============================

A comprehensive defense system against data poisoning attacks in AI-powered marketplaces.

Modules:
- dataset_generator: Generate synthetic marketplace data
- attack_simulator: Red team attack simulation tools
- defense_module: Blue team multi-layer defenses
- search_pipeline: FAISS-based semantic search
- evaluation: Comprehensive metrics and evaluation
- api: REST API for interactive demo
"""

from .dataset_generator import MarketplaceDatasetGenerator, Product
from .attack_simulator import AttackSimulator
from .defense_module import MarketplaceDefender, DefenseResult
from .search_pipeline import MarketplaceSearchPipeline, SearchResult, SearchEvaluator
from .evaluation import ComprehensiveEvaluator, EvaluationReport

__version__ = "1.0.0"
__author__ = "Tanish Gupta"

__all__ = [
    "MarketplaceDatasetGenerator",
    "Product",
    "AttackSimulator",
    "MarketplaceDefender",
    "DefenseResult",
    "MarketplaceSearchPipeline",
    "SearchResult",
    "SearchEvaluator",
    "ComprehensiveEvaluator",
    "EvaluationReport",
]
