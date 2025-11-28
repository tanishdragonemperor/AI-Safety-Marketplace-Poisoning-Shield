"""
Dataset Generator for Marketplace Poisoning Shield
Generates synthetic e-commerce product listings for attack/defense simulations.
"""

import random
import json
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np


@dataclass
class Product:
    """Represents a product listing in the marketplace."""
    id: str
    title: str
    description: str
    category: str
    price: float
    seller_id: str
    rating: float
    review_count: int
    reviews: List[Dict]
    metadata: Dict
    created_at: str
    is_poisoned: bool = False
    poison_type: Optional[str] = None


class MarketplaceDatasetGenerator:
    """Generates realistic synthetic marketplace data for testing."""
    
    # Product categories and their typical attributes
    CATEGORIES = {
        "electronics": {
            "prefixes": ["Premium", "Pro", "Ultra", "Smart", "Wireless", "Portable"],
            "items": ["Headphones", "Speaker", "Charger", "Cable", "Power Bank", "Earbuds", "Mouse", "Keyboard"],
            "brands": ["TechPro", "SoundMax", "PowerCell", "ConnectX", "AudioWave"],
            "price_range": (15, 300),
            "keywords": ["bluetooth", "USB-C", "fast charging", "noise cancelling", "HD audio", "ergonomic"]
        },
        "clothing": {
            "prefixes": ["Classic", "Modern", "Vintage", "Organic", "Premium", "Casual"],
            "items": ["T-Shirt", "Jeans", "Jacket", "Hoodie", "Sweater", "Dress", "Shorts"],
            "brands": ["UrbanStyle", "ComfortWear", "EcoThread", "FitFlex", "ModernCut"],
            "price_range": (10, 150),
            "keywords": ["cotton", "breathable", "slim fit", "relaxed", "sustainable", "comfortable"]
        },
        "home": {
            "prefixes": ["Luxury", "Essential", "Modern", "Classic", "Compact", "Multi-purpose"],
            "items": ["Lamp", "Pillow", "Blanket", "Organizer", "Mirror", "Rug", "Vase"],
            "brands": ["HomeEssentials", "CozyLiving", "DecorPlus", "SpaceSaver", "ArtHome"],
            "price_range": (8, 200),
            "keywords": ["decorative", "functional", "space-saving", "elegant", "durable", "washable"]
        },
        "sports": {
            "prefixes": ["Pro", "Elite", "Training", "Competition", "Beginner", "Advanced"],
            "items": ["Yoga Mat", "Dumbbells", "Resistance Bands", "Jump Rope", "Water Bottle", "Gym Bag"],
            "brands": ["FitGear", "ProSport", "ActiveLife", "EnduranceX", "FlexFit"],
            "price_range": (5, 100),
            "keywords": ["non-slip", "lightweight", "portable", "durable", "professional", "grip"]
        },
        "beauty": {
            "prefixes": ["Natural", "Organic", "Premium", "Professional", "Daily", "Intensive"],
            "items": ["Face Cream", "Serum", "Cleanser", "Moisturizer", "Sunscreen", "Mask"],
            "brands": ["PureSkin", "GlowUp", "NaturaCare", "BeautyLab", "SkinScience"],
            "price_range": (8, 80),
            "keywords": ["hydrating", "anti-aging", "vitamin C", "SPF", "gentle", "nourishing"]
        }
    }
    
    # Review templates for realistic review generation
    REVIEW_TEMPLATES = {
        "positive": [
            "Absolutely love this {item}! {feature} is exactly what I needed.",
            "Great quality for the price. The {feature} works perfectly.",
            "Exceeded my expectations! {feature} is a game changer.",
            "Best {item} I've ever purchased. Highly recommend!",
            "Amazing product! The {feature} is top notch.",
        ],
        "neutral": [
            "Decent {item}. {feature} could be better but overall okay.",
            "It's alright. Nothing special but does the job.",
            "Average quality. {feature} is as expected.",
            "Good for the price point. {feature} is adequate.",
        ],
        "negative": [
            "Disappointed with this {item}. {feature} didn't meet expectations.",
            "Not worth the money. {feature} is subpar.",
            "Had issues with {feature}. Would not recommend.",
            "Poor quality. The {feature} broke after a week.",
        ]
    }
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        self.seller_pool = self._generate_sellers(50)
    
    def _generate_sellers(self, count: int) -> List[Dict]:
        """Generate a pool of seller profiles."""
        sellers = []
        for i in range(count):
            seller_id = f"seller_{hashlib.md5(str(i).encode()).hexdigest()[:8]}"
            sellers.append({
                "id": seller_id,
                "name": f"Shop{random.choice(['Pro', 'Plus', 'Direct', 'Store', 'Mart'])}{i}",
                "rating": round(random.uniform(3.5, 5.0), 1),
                "total_sales": random.randint(100, 10000),
                "joined_date": (datetime.now() - timedelta(days=random.randint(30, 1000))).isoformat()
            })
        return sellers
    
    def _generate_product_title(self, category: str) -> str:
        """Generate a realistic product title."""
        cat_data = self.CATEGORIES[category]
        prefix = random.choice(cat_data["prefixes"])
        item = random.choice(cat_data["items"])
        brand = random.choice(cat_data["brands"])
        
        # Various title formats
        formats = [
            f"{brand} {prefix} {item}",
            f"{prefix} {item} by {brand}",
            f"{brand} {item} - {prefix} Edition",
            f"{item} ({prefix}) - {brand}",
        ]
        return random.choice(formats)
    
    def _generate_description(self, category: str, title: str) -> str:
        """Generate a realistic product description."""
        cat_data = self.CATEGORIES[category]
        keywords = random.sample(cat_data["keywords"], min(3, len(cat_data["keywords"])))
        
        templates = [
            f"Introducing our {title.lower()}. Features include {', '.join(keywords)}. "
            f"Perfect for everyday use with premium quality materials.",
            
            f"Experience the best with {title}. This product offers {keywords[0]} design "
            f"with {keywords[1] if len(keywords) > 1 else 'excellent'} performance. "
            f"Backed by our satisfaction guarantee.",
            
            f"{title} - Your perfect choice for quality and value. "
            f"Key features: {', '.join(keywords)}. Order now and enjoy fast shipping!",
        ]
        return random.choice(templates)
    
    def _generate_reviews(self, rating: float, count: int, category: str) -> List[Dict]:
        """Generate realistic reviews based on product rating."""
        reviews = []
        cat_data = self.CATEGORIES[category]
        
        for i in range(count):
            # Determine review sentiment based on overall rating with some variance
            if rating >= 4.5:
                sentiment_probs = [0.8, 0.15, 0.05]
            elif rating >= 3.5:
                sentiment_probs = [0.5, 0.35, 0.15]
            else:
                sentiment_probs = [0.2, 0.3, 0.5]
            
            sentiment = np.random.choice(["positive", "neutral", "negative"], p=sentiment_probs)
            template = random.choice(self.REVIEW_TEMPLATES[sentiment])
            
            item = random.choice(cat_data["items"]).lower()
            feature = random.choice(cat_data["keywords"])
            
            review_text = template.format(item=item, feature=feature)
            
            # Generate review rating consistent with sentiment
            if sentiment == "positive":
                review_rating = random.randint(4, 5)
            elif sentiment == "neutral":
                review_rating = random.randint(3, 4)
            else:
                review_rating = random.randint(1, 3)
            
            reviews.append({
                "id": f"review_{hashlib.md5(f'{i}{review_text}'.encode()).hexdigest()[:8]}",
                "rating": review_rating,
                "text": review_text,
                "helpful_votes": random.randint(0, 50),
                "verified_purchase": random.random() > 0.2,
                "date": (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat()
            })
        
        return reviews
    
    def generate_product(self, category: Optional[str] = None, product_id: Optional[str] = None) -> Product:
        """Generate a single realistic product listing."""
        if category is None:
            category = random.choice(list(self.CATEGORIES.keys()))
        
        cat_data = self.CATEGORIES[category]
        
        title = self._generate_product_title(category)
        description = self._generate_description(category, title)
        
        price = round(random.uniform(*cat_data["price_range"]), 2)
        rating = round(random.uniform(2.5, 5.0), 1)
        review_count = random.randint(5, 500)
        
        seller = random.choice(self.seller_pool)
        
        if product_id is None:
            product_id = f"prod_{hashlib.md5(f'{title}{random.random()}'.encode()).hexdigest()[:12]}"
        
        return Product(
            id=product_id,
            title=title,
            description=description,
            category=category,
            price=price,
            seller_id=seller["id"],
            rating=rating,
            review_count=review_count,
            reviews=self._generate_reviews(rating, min(review_count, 10), category),
            metadata={
                "brand": random.choice(cat_data["brands"]),
                "keywords": random.sample(cat_data["keywords"], 3),
                "in_stock": random.random() > 0.1,
                "shipping_days": random.randint(1, 7),
            },
            created_at=datetime.now().isoformat(),
            is_poisoned=False,
            poison_type=None
        )
    
    def generate_dataset(self, size: int = 100, category_distribution: Optional[Dict[str, float]] = None) -> List[Product]:
        """Generate a complete dataset of products."""
        if category_distribution is None:
            # Equal distribution across categories
            categories = list(self.CATEGORIES.keys())
            category_distribution = {cat: 1/len(categories) for cat in categories}
        
        products = []
        for category, proportion in category_distribution.items():
            count = int(size * proportion)
            for _ in range(count):
                products.append(self.generate_product(category=category))
        
        # Fill remaining if rounding caused fewer products
        while len(products) < size:
            products.append(self.generate_product())
        
        random.shuffle(products)
        return products
    
    def save_dataset(self, products: List[Product], filepath: str):
        """Save dataset to JSON file."""
        data = [asdict(p) for p in products]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(products)} products to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Product]:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        products = []
        for item in data:
            products.append(Product(**item))
        return products


if __name__ == "__main__":
    # Demo: Generate sample dataset
    generator = MarketplaceDatasetGenerator(seed=42)
    dataset = generator.generate_dataset(size=200)
    
    print(f"Generated {len(dataset)} products")
    print(f"\nSample product:")
    sample = dataset[0]
    print(f"  Title: {sample.title}")
    print(f"  Category: {sample.category}")
    print(f"  Price: ${sample.price}")
    print(f"  Rating: {sample.rating}")
    print(f"  Reviews: {sample.review_count}")
    
    # Save to file
    generator.save_dataset(dataset, "data/marketplace_dataset.json")
