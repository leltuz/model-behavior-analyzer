"""Cross-input behavioral pattern analysis."""

from typing import List, Dict, Any
from dataclasses import dataclass
import statistics


@dataclass
class BehavioralPatterns:
    """Cross-input behavioral pattern analysis results."""
    # Inputs with highest instability
    highest_instability_inputs: List[Dict[str, Any]]
    
    # Common perturbation types causing failures
    failure_causing_perturbations: List[Dict[str, Any]]
    
    # Clustering by input length
    by_length_groups: Dict[str, List[str]]  # length_range -> input_ids
    
    # Clustering by confidence range
    by_confidence_groups: Dict[str, List[str]]  # conf_range -> input_ids
    
    # Clustering by perturbation class
    by_perturbation_class_groups: Dict[str, List[str]]  # class -> input_ids
    
    # Clustering by instability regime
    by_regime_groups: Dict[str, List[str]]  # regime -> input_ids
    
    # Summary statistics
    pattern_summary: Dict[str, Any]


class PatternAnalyzer:
    """Identifies behavioral patterns across inputs."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pattern analyzer.
        
        Args:
            config: Pattern analysis configuration from config.yaml
        """
        self.config = config.get('pattern_analysis', {})
        self.enabled = self.config.get('enabled', True)
        self.grouping_dimensions = self.config.get('grouping_dimensions', [])
    
    def analyze(
        self,
        per_input_stabilities: List[Any],  # List of PerInputStability
        trace_data: Dict[str, Any]  # Trace data for text length info
    ) -> BehavioralPatterns:
        """
        Analyze patterns across inputs.
        
        Args:
            per_input_stabilities: List of PerInputStability objects
            trace_data: Trace data containing input texts
        
        Returns:
            BehavioralPatterns object
        """
        if not self.enabled or not per_input_stabilities:
            return BehavioralPatterns(
                highest_instability_inputs=[],
                failure_causing_perturbations=[],
                by_length_groups={},
                by_confidence_groups={},
                by_perturbation_class_groups={},
                by_regime_groups={},
                pattern_summary={}
            )
        
        # Find inputs with highest instability
        highest_instability = sorted(
            per_input_stabilities,
            key=lambda x: (x.flip_rate, -x.decision_consistency, x.worst_case_deviation),
            reverse=True
        )[:5]
        
        highest_instability_inputs = [
            {
                'input_id': s.input_id,
                'flip_rate': s.flip_rate,
                'decision_consistency': s.decision_consistency,
                'worst_case_deviation': s.worst_case_deviation,
                'instability_regime': s.instability_regime
            }
            for s in highest_instability
        ]
        
        # Find common perturbation types causing failures
        failure_counts = {}
        for stability in per_input_stabilities:
            for detail in stability.perturbation_details:
                if detail['flipped']:
                    ptype = detail['perturbation_type']
                    inv_class = detail.get('invariance_class', 'STRESS_TEST')
                    key = f"{ptype} ({inv_class})"
                    failure_counts[key] = failure_counts.get(key, 0) + 1
        
        failure_causing_perturbations = sorted(
            [{'perturbation': k, 'failure_count': v} for k, v in failure_counts.items()],
            key=lambda x: x['failure_count'],
            reverse=True
        )
        
        # Group by input length
        by_length_groups = self._group_by_length(per_input_stabilities, trace_data)
        
        # Group by confidence range
        by_confidence_groups = self._group_by_confidence(per_input_stabilities)
        
        # Group by perturbation class (invariance class)
        by_perturbation_class_groups = self._group_by_perturbation_class(per_input_stabilities)
        
        # Group by instability regime
        by_regime_groups = self._group_by_regime(per_input_stabilities)
        
        # Summary statistics
        pattern_summary = self._compute_pattern_summary(
            per_input_stabilities,
            by_length_groups,
            by_confidence_groups,
            by_regime_groups
        )
        
        return BehavioralPatterns(
            highest_instability_inputs=highest_instability_inputs,
            failure_causing_perturbations=failure_causing_perturbations,
            by_length_groups=by_length_groups,
            by_confidence_groups=by_confidence_groups,
            by_perturbation_class_groups=by_perturbation_class_groups,
            by_regime_groups=by_regime_groups,
            pattern_summary=pattern_summary
        )
    
    def _group_by_length(
        self,
        per_input_stabilities: List[Any],
        trace_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Group inputs by text length."""
        groups = {
            'short (1-5 words)': [],
            'medium (6-10 words)': [],
            'long (11-15 words)': [],
            'very_long (16+ words)': []
        }
        
        # Build input_id -> text mapping
        input_texts = {}
        for input_data in trace_data.get('inputs', []):
            input_texts[input_data['input_id']] = input_data['original_text']
        
        for stability in per_input_stabilities:
            text = input_texts.get(stability.input_id, stability.original_text)
            length = len(text.split())
            
            if length <= 5:
                groups['short (1-5 words)'].append(stability.input_id)
            elif length <= 10:
                groups['medium (6-10 words)'].append(stability.input_id)
            elif length <= 15:
                groups['long (11-15 words)'].append(stability.input_id)
            else:
                groups['very_long (16+ words)'].append(stability.input_id)
        
        return {k: v for k, v in groups.items() if v}
    
    def _group_by_confidence(
        self,
        per_input_stabilities: List[Any]
    ) -> Dict[str, List[str]]:
        """Group inputs by confidence range."""
        groups = {
            'low_confidence (0.0-0.5)': [],
            'medium_confidence (0.5-0.7)': [],
            'high_confidence (0.7-0.9)': [],
            'very_high_confidence (0.9-1.0)': []
        }
        
        for stability in per_input_stabilities:
            conf = stability.original_confidence
            
            if conf < 0.5:
                groups['low_confidence (0.0-0.5)'].append(stability.input_id)
            elif conf < 0.7:
                groups['medium_confidence (0.5-0.7)'].append(stability.input_id)
            elif conf < 0.9:
                groups['high_confidence (0.7-0.9)'].append(stability.input_id)
            else:
                groups['very_high_confidence (0.9-1.0)'].append(stability.input_id)
        
        return {k: v for k, v in groups.items() if v}
    
    def _group_by_perturbation_class(
        self,
        per_input_stabilities: List[Any]
    ) -> Dict[str, List[str]]:
        """Group inputs by which perturbation class caused failures."""
        groups = {
            'failed_under_expected_invariant': [],
            'failed_under_stress_test': [],
            'failed_under_both': [],
            'stable_under_all': []
        }
        
        for stability in per_input_stabilities:
            has_expected_failures = stability.expected_invariant_flip_rate > 0
            has_stress_failures = stability.stress_test_flip_rate > 0
            
            if has_expected_failures and has_stress_failures:
                groups['failed_under_both'].append(stability.input_id)
            elif has_expected_failures:
                groups['failed_under_expected_invariant'].append(stability.input_id)
            elif has_stress_failures:
                groups['failed_under_stress_test'].append(stability.input_id)
            else:
                groups['stable_under_all'].append(stability.input_id)
        
        return {k: v for k, v in groups.items() if v}
    
    def _group_by_regime(
        self,
        per_input_stabilities: List[Any]
    ) -> Dict[str, List[str]]:
        """Group inputs by instability regime."""
        groups = {'STABLE': [], 'SENSITIVE': [], 'BRITTLE': []}
        
        for stability in per_input_stabilities:
            regime = stability.instability_regime
            groups[regime].append(stability.input_id)
        
        return groups
    
    def _compute_pattern_summary(
        self,
        per_input_stabilities: List[Any],
        by_length_groups: Dict[str, List[str]],
        by_confidence_groups: Dict[str, List[str]],
        by_regime_groups: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Compute summary statistics for patterns."""
        # Compute average metrics by length group
        length_stats = {}
        for length_range, input_ids in by_length_groups.items():
            matching_stabilities = [s for s in per_input_stabilities if s.input_id in input_ids]
            if matching_stabilities:
                length_stats[length_range] = {
                    'count': len(matching_stabilities),
                    'mean_flip_rate': statistics.mean([s.flip_rate for s in matching_stabilities]),
                    'mean_consistency': statistics.mean([s.decision_consistency for s in matching_stabilities])
                }
        
        # Compute average metrics by confidence group
        confidence_stats = {}
        for conf_range, input_ids in by_confidence_groups.items():
            matching_stabilities = [s for s in per_input_stabilities if s.input_id in input_ids]
            if matching_stabilities:
                confidence_stats[conf_range] = {
                    'count': len(matching_stabilities),
                    'mean_flip_rate': statistics.mean([s.flip_rate for s in matching_stabilities]),
                    'mean_consistency': statistics.mean([s.decision_consistency for s in matching_stabilities])
                }
        
        # Regime distribution
        regime_stats = {
            regime: {
                'count': len(input_ids),
                'percentage': (len(input_ids) / len(per_input_stabilities) * 100) if per_input_stabilities else 0.0
            }
            for regime, input_ids in by_regime_groups.items()
        }
        
        return {
            'by_length': length_stats,
            'by_confidence': confidence_stats,
            'by_regime': regime_stats
        }

