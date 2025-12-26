"""Dataset loading and generation utilities."""

import json
import random
from typing import List, Dict, Any, Optional


def load_dataset(input_path: Optional[str], num_samples: int, seed: int) -> List[Dict[str, Any]]:
    """
    Load dataset from file or generate synthetic dataset.
    
    Args:
        input_path: Path to JSON file with 'id' and 'text' fields. If None, generates synthetic.
        num_samples: Number of synthetic samples if input_path is None.
        seed: Random seed for reproducibility.
    
    Returns:
        List of dictionaries with 'id' and 'text' fields.
    """
    if input_path:
        with open(input_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON array")
        for item in data:
            if 'id' not in item or 'text' not in item:
                raise ValueError("Each item must have 'id' and 'text' fields")
        return data
    
    # Generate synthetic dataset
    random.seed(seed)
    synthetic_texts = [
        "This movie is absolutely fantastic and I loved every minute of it.",
        "The service was terrible and the food was cold when it arrived.",
        "I think this product is okay, nothing special but it works.",
        "Amazing experience! Highly recommend to everyone.",
        "Very disappointed with the quality and customer service.",
        "The book was interesting but the ending felt rushed.",
        "Outstanding performance by all actors in this production.",
        "Poor quality materials, would not buy again.",
        "It's a decent option if you're on a budget.",
        "Exceptional value for money, exceeded my expectations.",
        "The software crashed multiple times during use.",
        "Beautiful design and excellent craftsmanship throughout.",
        "Not worth the price, very overrated in my opinion.",
        "Great features but the interface could be more intuitive.",
        "Perfect for beginners, easy to understand and use.",
        "The delivery was late and the packaging was damaged.",
        "Incredible attention to detail, truly impressive work.",
        "Mediocre at best, expected much more for this price.",
        "Fast shipping and product exactly as described.",
        "Would not recommend, many better alternatives available.",
    ]
    
    # Use provided samples or repeat if needed
    samples = []
    for i in range(num_samples):
        text = synthetic_texts[i % len(synthetic_texts)]
        samples.append({
            'id': f"sample_{i:04d}",
            'text': text
        })
    
    return samples

