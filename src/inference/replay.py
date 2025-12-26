"""Deterministic inference replay system."""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available. Using mock inference.")


@dataclass
class InferenceResult:
    """Result of a single inference run."""
    input_text: str
    prediction: str
    confidence: float
    run_index: int
    metadata: Dict[str, Any]


class InferenceReplay:
    """
    Deterministic inference replay for model behavior analysis.
    
    Determinism serves as an experimental control to isolate input-driven
    variance from execution noise. Fixed seeds enable reproducible measurement
    and comparison of run variance vs. perturbation variance.
    """
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """
        Initialize inference replay system.
        
        Determinism (via fixed seeds) is an experimental control, not a
        property of the underlying model. It isolates input-driven variance
        from execution noise.
        
        Args:
            config: Inference configuration from config.yaml
            seed: Random seed for experimental control (determinism)
        """
        self.config = config
        self.seed = seed
        self.device = config.get('device', 'cpu')
        self.num_runs = config.get('num_runs', 1)
        self.batch_size = config.get('batch_size', 8)
        
        # Set seeds for determinism
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        
        # Load model if transformers is available
        self.model = None
        self.tokenizer = None
        self.model_name = config.get('model_name', 'distilbert-base-uncased-finetuned-sst-2-english')
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                # Set deterministic mode
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception as e:
                logging.warning(f"Could not load model {self.model_name}: {e}. Using mock inference.")
                self.model = None
        else:
            logging.warning("Using mock inference (transformers not available).")
    
    def run(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[InferenceResult]:
        """
        Run inference on text multiple times for consistency measurement.
        
        Repeated runs serve as:
        - Numerical stability probes: Detect variance in deterministic inference
        - Stochastic sensitivity checks: Even under deterministic settings, measure
          whether perturbation-induced variance dominates run variance
        
        Args:
            text: Input text
            metadata: Additional metadata to include in results
        
        Returns:
            List of InferenceResult objects (one per run)
        """
        results = []
        
        for run_idx in range(self.num_runs):
            # Ensure determinism for each run
            torch.manual_seed(self.seed + run_idx)
            np.random.seed(self.seed + run_idx)
            
            if self.model is not None and self.tokenizer is not None:
                prediction, confidence = self._run_real_inference(text)
            else:
                prediction, confidence = self._run_mock_inference(text, run_idx)
            
            result = InferenceResult(
                input_text=text,
                prediction=prediction,
                confidence=confidence,
                run_index=run_idx,
                metadata=metadata or {}
            )
            results.append(result)
        
        return results
    
    def _run_real_inference(self, text: str) -> Tuple[str, float]:
        """Run inference using actual model."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class_idx].item()
            
            # Map to label (assuming binary classification)
            label = "positive" if predicted_class_idx == 1 else "negative"
            
            return label, confidence
    
    def _run_mock_inference(self, text: str, run_idx: int) -> Tuple[str, float]:
        """
        Mock inference for testing without transformers.
        Uses simple heuristics to simulate model behavior.
        """
        text_lower = text.lower()
        
        # Simple sentiment heuristics
        positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'love', 'loved', 
                         'outstanding', 'perfect', 'beautiful', 'exceptional', 'highly', 'recommend']
        negative_words = ['bad', 'terrible', 'awful', 'poor', 'horrible', 'disappointed', 
                         'worst', 'hate', 'hated', 'mediocre', 'overrated', 'not worth']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Add some deterministic randomness based on text hash
        text_hash = hash(text) % 1000
        base_confidence = 0.5 + (pos_count - neg_count) * 0.1
        confidence = max(0.1, min(0.99, base_confidence + (text_hash % 20) / 100.0))
        
        # Add small variation per run for consistency testing
        confidence += (run_idx % 3) * 0.01
        
        if pos_count > neg_count:
            prediction = "positive"
        elif neg_count > pos_count:
            prediction = "negative"
        else:
            prediction = "positive" if confidence > 0.5 else "negative"
        
        return prediction, confidence

