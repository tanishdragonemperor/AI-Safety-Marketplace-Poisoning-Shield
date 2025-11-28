# 🛡️ Marketplace Poisoning Shield

> **Detecting and Defending Against Silent Data Poisoning in AI-Powered Marketplaces**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end AI security project demonstrating attack simulation and defense mechanisms against data poisoning in e-commerce marketplaces.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Attack Types](#attack-types)
- [Defense Mechanisms](#defense-mechanisms)
- [Evaluation Metrics](#evaluation-metrics)
- [API Reference](#api-reference)
- [Interactive Demo](#interactive-demo)
- [Project Structure](#project-structure)
- [Results](#results)

## 🎯 Overview

Modern e-commerce marketplaces rely heavily on AI models for:
- Search ranking and relevance
- Product recommendations  
- Content moderation
- Review analysis

These systems are vulnerable to **silent data poisoning** - attackers inject malicious data that manipulates AI behavior without being easily detectable by human moderators.

This project provides:
1. **Red Team Tools**: Simulate 6 different poisoning attack types
2. **Blue Team Defenses**: Multi-layer detection and sanitization
3. **Evaluation Framework**: Comprehensive metrics and comparison
4. **Interactive Demo**: Visual demonstration of attacks and defenses

## ✨ Features

### 🔴 Red Team (Attack Simulation)
- Hidden Unicode character injection
- SEO keyword stuffing
- Homoglyph character substitution
- Fake review generation
- Adversarial paraphrasing
- Metadata poisoning

### 🔵 Blue Team (Defense)
- Unicode anomaly detection
- Keyword density analysis
- Homoglyph scanner
- Review authenticity checker
- Metadata validation
- Statistical anomaly detection
- Automatic text sanitization

### 📊 Evaluation
- Precision, Recall, F1 Score
- Confusion matrix analysis
- Per-attack-type detection rates
- Search quality metrics (NDCG, Precision@K)
- Defense impact comparison

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MARKETPLACE POISONING SHIELD                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Dataset    │───▶│   Attack     │───▶│   Defense    │          │
│  │  Generator   │    │  Simulator   │    │   Module     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────────────────────────────────────────────┐          │
│  │              FAISS Search Pipeline                    │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │          │
│  │  │  Embedding  │  │   FAISS     │  │  Sanitized  │   │          │
│  │  │  Generator  │  │   Index     │  │   Index     │   │          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │          │
│  └──────────────────────────────────────────────────────┘          │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────┐          │
│  │                 Evaluation Module                     │          │
│  │  • Precision/Recall  • F1 Score  • NDCG  • Impact    │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to project directory
cd marketplace-poisoning-shield

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Quick Start

### Option 1: Web Frontend (Recommended!)
```bash
python run_server.py
```
This starts the server and opens a beautiful web UI at http://localhost:5000

### Option 2: Terminal Demo
```bash
python video_demo.py    # Interactive step-by-step demo
python main.py          # Full automated demo
```

### Option 3: Just the Evaluation
```bash
python main.py --evaluate
```

## ⚔️ Attack Types

### 1. Hidden Character Injection
Injects invisible Unicode characters (zero-width spaces, joiners) that manipulate tokenization.

```python
# Example: "Premium" becomes "Pre​mi​um" with hidden chars
# Bytes: 7 → 13 (invisible to humans)
```

### 2. Keyword Stuffing
Adds excessive SEO keywords to artificially boost search rankings.

```python
# Before: "Wireless Headphones"
# After: "[BEST TOP #1 PREMIUM] Wireless Headphones | bestseller authentic"
```

### 3. Homoglyph Attack
Replaces characters with visually identical Unicode lookalikes.

```python
# Before: "Premium" (Latin)
# After: "Prеmium" (Cyrillic 'е' instead of Latin 'e')
```

### 4. Fake Review Injection
Generates artificial positive reviews with inflated ratings.

```python
# Injects: "AMAZING! Best product EVER! Must buy! 5 stars!!!"
# Boosts rating: 4.2 → 4.8
```

### 5. Adversarial Paraphrasing
Adds misleading brand associations and authenticity claims.

```python
# Adds: "Better than Apple! 100% Authentic and Original."
```

### 6. Metadata Poisoning
Injects hidden promotional content and fake metrics.

```python
# Hidden: {"__promo__": "BUY NOW...", "view_count": 999999}
```

## 🛡️ Defense Mechanisms

### Layer 1: Unicode Anomaly Detection
- Detects zero-width characters
- Flags suspicious Unicode categories
- Calculates anomaly score

### Layer 2: Keyword Density Analysis
- Pattern matching for SEO spam
- Word frequency analysis
- Repetition detection

### Layer 3: Homoglyph Detection
- Character-by-character scanning
- Maps known homoglyphs to ASCII
- Calculates substitution ratio

### Layer 4: Review Authenticity
- Pattern matching for fake reviews
- Sentiment analysis
- Helpful vote ratio analysis

### Layer 5: Metadata Validation
- Hidden field detection
- Metric plausibility checks
- Timestamp validation

### Layer 6: Statistical Anomaly Detection
- Baseline comparison
- Z-score calculation
- Rating/review correlation

## 📈 Evaluation Metrics

### Defense Metrics
| Metric | Description |
|--------|-------------|
| Precision | % of flagged items that are actually poisoned |
| Recall | % of poisoned items that were detected |
| F1 Score | Harmonic mean of precision and recall |
| False Positive Rate | Clean items incorrectly flagged |

### Search Quality Metrics
| Metric | Description |
|--------|-------------|
| Precision@K | Clean results in top K |
| NDCG | Normalized Discounted Cumulative Gain |
| Poisoned Exposure | Poisoned items shown to users |

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System status |
| POST | `/api/init` | Initialize system |
| GET | `/api/products` | List products |
| GET | `/api/product/<id>` | Product details |
| POST | `/api/attack/demo` | Demo an attack |
| POST | `/api/defense/analyze` | Analyze text |
| GET | `/api/search` | Search products |
| GET | `/api/search/compare` | Compare with/without defense |
| GET | `/api/stats` | Overall statistics |
| POST | `/api/evaluate` | Run evaluation |

### Example Usage

```python
import requests

# Initialize
requests.post("http://localhost:5000/api/init", json={
    "dataset_size": 100,
    "poison_rate": 0.25
})

# Search
response = requests.get("http://localhost:5000/api/search", params={
    "q": "premium headphones",
    "defense": "true"
})
print(response.json())
```

## 🖥️ Interactive Demo

The React demo (`demo/App.jsx`) provides:

1. **Dashboard**: Overview of dataset statistics and detection metrics
2. **Red Team**: Interactive attack simulator with visualizations
3. **Blue Team**: Defense layer visualization and analysis
4. **Search Demo**: Side-by-side comparison of search results

To run the demo, copy `demo/App.jsx` to a React project or use the artifact viewer.

## 📁 Project Structure

```
marketplace-poisoning-shield/
├── main.py                    # Main runner script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py   # Synthetic data generation
│   ├── attack_simulator.py    # Red team attack tools
│   ├── defense_module.py      # Blue team defense system
│   ├── search_pipeline.py     # FAISS search with embeddings
│   ├── evaluation.py          # Comprehensive evaluation
│   └── api.py                 # Flask REST API
├── demo/
│   └── App.jsx                # React interactive demo
├── data/
│   └── (generated datasets and reports)
└── tests/
    └── (test files)
```

## 📊 Results

### Sample Evaluation Results

```
╔══════════════════════════════════════════════════════════════════════╗
║                         EVALUATION RESULTS                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Dataset: 100 products, 25% poisoned                                 ║
║                                                                      ║
║  DEFENSE METRICS                                                     ║
║  ├── Precision:  0.920                                               ║
║  ├── Recall:     0.880                                               ║
║  ├── F1 Score:   0.900                                               ║
║  └── Accuracy:   0.940                                               ║
║                                                                      ║
║  SEARCH IMPACT                                                       ║
║  ├── Baseline poisoned in top 5: 2.3 avg                             ║
║  ├── Defended poisoned in top 5: 0.4 avg                             ║
║  └── Reduction: 82.6%                                                ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Per-Attack Detection Rates

| Attack Type | Detection Rate |
|-------------|----------------|
| Hidden Characters | 95% |
| Keyword Stuffing | 88% |
| Homoglyph | 92% |
| Fake Reviews | 85% |
| Adversarial Paraphrase | 75% |
| Metadata Poisoning | 90% |

## 🎓 Educational Value

This project demonstrates key AI security concepts:

1. **Data Poisoning Attacks**: How malicious actors can manipulate ML systems
2. **Multi-Layer Defense**: Defense in depth approach to security
3. **Evaluation Methodology**: Proper metrics for security systems
4. **Trade-offs**: Precision vs recall, security vs usability

## 📝 License

MIT License - feel free to use for educational purposes.

## 👤 Author

**Tanish Gupta**  
AI Security Course Project  
Intro to AI Security

## 🙏 Acknowledgments

- Course instructors for guidance
- FAISS team for the vector search library
- Sentence Transformers for embedding models

---

<p align="center">
  <b>Marketplace Poisoning Shield</b><br>
  Protecting AI-Powered Marketplaces from Silent Data Poisoning
</p>
