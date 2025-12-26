"""Perturbation generation for controlled input transformations."""

import random
import string
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Perturbation:
    """Represents a single perturbation."""
    type: str
    invariance_class: str  # EXPECTED_INVARIANT or STRESS_TEST
    original_text: str
    perturbed_text: str
    metadata: Dict[str, Any]


class PerturbationGenerator:
    """
    Generates controlled, deterministic perturbations of text inputs.
    
    Perturbations serve as proxies for invariance expectations, not claims
    of linguistic completeness or real-world coverage. Their value lies in
    their conceptual role (testing invariance expectations) rather than
    linguistic realism.
    """
    
    # Common English stopwords
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
        'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our',
        'their', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'now'
    }
    
    # Simple synonym dictionary (word -> list of synonyms)
    SYNONYMS = {
        'good': ['great', 'excellent', 'fine', 'nice', 'well'],
        'bad': ['terrible', 'awful', 'poor', 'horrible', 'worse'],
        'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
        'small': ['tiny', 'little', 'mini', 'petite', 'compact'],
        'happy': ['glad', 'pleased', 'joyful', 'delighted', 'cheerful'],
        'sad': ['unhappy', 'depressed', 'sorrowful', 'melancholy', 'down'],
        'fast': ['quick', 'rapid', 'swift', 'speedy', 'brisk'],
        'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried', 'delayed'],
        'beautiful': ['pretty', 'lovely', 'gorgeous', 'attractive', 'stunning'],
        'ugly': ['unattractive', 'hideous', 'repulsive', 'unsightly', 'plain'],
        'important': ['significant', 'crucial', 'vital', 'essential', 'key'],
        'easy': ['simple', 'straightforward', 'effortless', 'uncomplicated', 'basic'],
        'difficult': ['hard', 'challenging', 'tough', 'complex', 'complicated'],
        'new': ['fresh', 'recent', 'modern', 'novel', 'latest'],
        'old': ['ancient', 'aged', 'elderly', 'outdated', 'vintage'],
        'smart': ['intelligent', 'clever', 'bright', 'brilliant', 'wise'],
        'stupid': ['foolish', 'dumb', 'silly', 'unwise', 'ignorant'],
    }
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """
        Initialize perturbation generator.
        
        Args:
            config: Perturbation configuration from config.yaml
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        random.seed(seed)
        
        # Build invariance class mapping
        self.invariance_class_map = {}
        invariance_classes = config.get('invariance_classes', {})
        for inv_class, types in invariance_classes.items():
            for ptype in types:
                self.invariance_class_map[ptype] = inv_class
    
    def _get_invariance_class(self, perturbation_type: str) -> str:
        """Get invariance class for a perturbation type."""
        return self.invariance_class_map.get(perturbation_type, 'STRESS_TEST')
    
    def generate(self, text: str, input_id: str) -> List[Perturbation]:
        """
        Generate perturbations for a given text input.
        
        Args:
            text: Original input text
            input_id: Unique identifier for the input
        
        Returns:
            List of Perturbation objects
        """
        perturbations = []
        num_perturbations = self.config.get('num_per_input', 5)
        types_config = self.config.get('types', {})
        
        # Collect enabled perturbation types with their frequencies
        enabled_types = []
        for ptype, pconfig in types_config.items():
            if pconfig.get('enabled', False):
                freq = pconfig.get('frequency', 0.0)
                enabled_types.extend([ptype] * int(freq * 100))
        
        if not enabled_types:
            # Default: use all types equally
            enabled_types = list(types_config.keys())
        
        # Generate perturbations
        for i in range(num_perturbations):
            # Select perturbation type based on frequency
            ptype = random.choice(enabled_types) if enabled_types else 'stopword_removal'
            
            try:
                if ptype == 'stopword_removal':
                    perturbed = self._remove_stopwords(text)
                elif ptype == 'synonym_substitution':
                    max_subs = types_config.get('synonym_substitution', {}).get('max_substitutions', 2)
                    perturbed = self._substitute_synonyms(text, max_subs)
                elif ptype == 'punctuation_casing':
                    ops = types_config.get('punctuation_casing', {}).get('operations', ['lowercase'])
                    perturbed = self._change_punctuation_casing(text, random.choice(ops))
                elif ptype == 'character_noise':
                    noise_rate = types_config.get('character_noise', {}).get('noise_rate', 0.05)
                    perturbed = self._add_character_noise(text, noise_rate)
                else:
                    # Fallback to stopword removal
                    perturbed = self._remove_stopwords(text)
                
                invariance_class = self._get_invariance_class(ptype)
                
                perturbations.append(Perturbation(
                    type=ptype,
                    invariance_class=invariance_class,
                    original_text=text,
                    perturbed_text=perturbed,
                    metadata={
                        'input_id': input_id,
                        'perturbation_index': i,
                        'type': ptype,
                        'invariance_class': invariance_class
                    }
                ))
            except Exception as e:
                # If perturbation fails, skip it
                continue
        
        return perturbations
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered = [w for w in words if w.lower().strip(string.punctuation) not in self.STOPWORDS]
        return ' '.join(filtered) if filtered else text
    
    def _substitute_synonyms(self, text: str, max_substitutions: int) -> str:
        """Substitute words with synonyms from dictionary."""
        words = text.split()
        substitutions = 0
        
        for i, word in enumerate(words):
            if substitutions >= max_substitutions:
                break
            
            # Remove punctuation for lookup
            clean_word = word.lower().strip(string.punctuation)
            if clean_word in self.SYNONYMS:
                synonym = random.choice(self.SYNONYMS[clean_word])
                # Preserve original casing and punctuation
                if word[0].isupper():
                    synonym = synonym.capitalize()
                # Preserve trailing punctuation
                if word[-1] in string.punctuation:
                    synonym = synonym + word[-1]
                words[i] = synonym
                substitutions += 1
        
        return ' '.join(words)
    
    def _change_punctuation_casing(self, text: str, operation: str) -> str:
        """Change punctuation or casing."""
        if operation == 'lowercase':
            return text.lower()
        elif operation == 'uppercase':
            return text.upper()
        elif operation == 'remove_punctuation':
            return ''.join(c for c in text if c not in string.punctuation)
        else:
            return text
    
    def _add_character_noise(self, text: str, noise_rate: float) -> str:
        """Add character-level noise (typos)."""
        chars = list(text)
        num_noise = max(1, int(len(chars) * noise_rate))
        
        for _ in range(num_noise):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            # Random character substitution (simple typo simulation)
            if chars[idx].isalpha():
                chars[idx] = random.choice(string.ascii_letters)
        
        return ''.join(chars)

