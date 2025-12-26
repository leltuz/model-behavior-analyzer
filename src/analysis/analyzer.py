"""Stability and consistency analysis metrics computation."""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import statistics
import math


@dataclass
class PerInputStability:
    """Stability profile for a single input."""
    input_id: str
    original_text: str
    original_prediction: str
    original_confidence: float
    
    # Metrics
    decision_consistency: float  # Fraction of perturbations preserving prediction
    flip_rate: float  # Percentage of perturbations that change prediction
    confidence_variance: float  # Variance of confidence across perturbations
    confidence_mean: float
    confidence_std: float
    worst_case_deviation: float  # Maximum confidence drop
    worst_case_flip: bool  # Whether any perturbation flipped the decision
    
    # Instability regime classification
    instability_regime: str  # STABLE, SENSITIVE, or BRITTLE
    
    # Run variance analysis
    run_variance: float  # Variance across repeated inference runs
    perturbation_variance_dominates: bool  # Whether perturbation variance > run variance
    
    # Per-invariance-class metrics
    expected_invariant_consistency: float  # Consistency for EXPECTED_INVARIANT perturbations
    expected_invariant_flip_rate: float
    stress_test_consistency: float  # Consistency for STRESS_TEST perturbations
    stress_test_flip_rate: float
    
    # Detailed breakdown
    num_perturbations: int
    num_consistent: int
    num_flipped: int
    perturbation_details: List[Dict[str, Any]]


@dataclass
class GlobalStabilityReport:
    """Global stability metrics across all inputs."""
    total_inputs: int
    total_perturbations: int
    
    # Aggregate metrics
    mean_decision_consistency: float
    mean_flip_rate: float
    mean_confidence_variance: float
    mean_worst_case_deviation: float
    
    # Distribution statistics
    consistency_distribution: Dict[str, float]  # min, max, median, q25, q75
    flip_rate_distribution: Dict[str, float]
    confidence_variance_distribution: Dict[str, float]
    
    # Per-perturbation-type statistics
    per_type_stats: Dict[str, Dict[str, float]]
    
    # Per-invariance-class statistics
    per_invariance_class_stats: Dict[str, Dict[str, float]]
    
    # Instability regime distribution
    regime_distribution: Dict[str, int]  # Count of inputs per regime
    
    # Run variance statistics
    mean_run_variance: float
    perturbation_variance_dominates_count: int  # Number of inputs where perturbation variance > run variance


class StabilityAnalyzer:
    """
    Computes stability and consistency metrics from inference results.
    
    Operates at the input–output behavior level, making no assumptions
    about model architecture, training method, or modality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stability analyzer.
        
        Args:
            config: Analysis configuration from config.yaml
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
        # Instability regime thresholds
        regime_config = config.get('instability_regimes', {})
        self.stable_threshold = regime_config.get('stable_threshold', 0.95)
        self.stable_flip_threshold = regime_config.get('stable_flip_threshold', 0.05)
        self.brittle_threshold = regime_config.get('brittle_threshold', 0.80)
        self.brittle_flip_threshold = regime_config.get('brittle_flip_threshold', 0.20)
        self.stable_deviation_threshold = regime_config.get('stable_deviation_threshold', 0.10)
        self.brittle_deviation_threshold = regime_config.get('brittle_deviation_threshold', 0.30)
    
    def _classify_instability_regime(
        self,
        decision_consistency: float,
        flip_rate: float,
        worst_case_deviation: float
    ) -> str:
        """
        Classify input into instability regime.
        
        Returns:
            'STABLE', 'SENSITIVE', or 'BRITTLE'
        """
        # BRITTLE: Low consistency OR high flip rate OR high deviation
        if (decision_consistency < self.brittle_threshold or
            flip_rate >= self.brittle_flip_threshold or
            worst_case_deviation >= self.brittle_deviation_threshold):
            return 'BRITTLE'
        
        # STABLE: High consistency AND low flip rate AND low deviation
        if (decision_consistency >= self.stable_threshold and
            flip_rate <= self.stable_flip_threshold and
            worst_case_deviation <= self.stable_deviation_threshold):
            return 'STABLE'
        
        # SENSITIVE: Between stable and brittle
        return 'SENSITIVE'
    
    def analyze_input(
        self,
        input_id: str,
        original_text: str,
        original_results: List[Any],  # List of InferenceResult
        perturbation_results: List[Dict[str, Any]]  # List of {perturbation, results}
    ) -> PerInputStability:
        """
        Analyze stability for a single input.
        
        Args:
            input_id: Unique identifier for input
            original_text: Original input text
            original_results: Inference results for original input
            perturbation_results: List of dicts with 'perturbation' and 'results' keys
        
        Returns:
            PerInputStability object
        """
        # Get original prediction (use first run as reference)
        original_result = original_results[0]
        original_prediction = original_result.prediction
        original_confidence = original_result.confidence
        
        # Compute run variance (variance across repeated runs of original input)
        original_confidences = [r.confidence for r in original_results]
        run_variance = statistics.variance(original_confidences) if len(original_confidences) > 1 else 0.0
        
        # Collect all perturbation predictions and confidences
        perturbation_predictions = []
        perturbation_confidences = []
        perturbation_details = []
        
        # Separate by invariance class
        expected_invariant_predictions = []
        expected_invariant_confidences = []
        stress_test_predictions = []
        stress_test_confidences = []
        
        for pert_data in perturbation_results:
            pert = pert_data['perturbation']
            results = pert_data['results']
            
            # Use first run result for consistency
            result = results[0]
            pred = result.prediction
            conf = result.confidence
            
            perturbation_predictions.append(pred)
            perturbation_confidences.append(conf)
            
            # Group by invariance class
            if pert.invariance_class == 'EXPECTED_INVARIANT':
                expected_invariant_predictions.append(pred)
                expected_invariant_confidences.append(conf)
            else:  # STRESS_TEST
                stress_test_predictions.append(pred)
                stress_test_confidences.append(conf)
            
            # Check if decision flipped
            flipped = (pred != original_prediction)
            
            # Compute confidence deviation
            confidence_deviation = original_confidence - conf
            
            perturbation_details.append({
                'perturbation_type': pert.type,
                'invariance_class': pert.invariance_class,
                'perturbed_text': pert.perturbed_text,
                'prediction': pred,
                'confidence': conf,
                'flipped': flipped,
                'confidence_deviation': confidence_deviation
            })
        
        # Compute metrics
        num_perturbations = len(perturbation_predictions)
        num_consistent = sum(1 for p in perturbation_predictions if p == original_prediction)
        num_flipped = num_perturbations - num_consistent
        
        decision_consistency = num_consistent / num_perturbations if num_perturbations > 0 else 1.0
        flip_rate = num_flipped / num_perturbations if num_perturbations > 0 else 0.0
        
        # Confidence statistics
        all_confidences = [original_confidence] + perturbation_confidences
        confidence_mean = statistics.mean(all_confidences)
        confidence_variance = statistics.variance(all_confidences) if len(all_confidences) > 1 else 0.0
        confidence_std = math.sqrt(confidence_variance) if confidence_variance > 0 else 0.0
        
        # Worst-case deviation
        worst_case_deviation = max(
            [original_confidence - conf for conf in perturbation_confidences] + [0.0]
        )
        
        # Check if any perturbation flipped
        worst_case_flip = any(p != original_prediction for p in perturbation_predictions)
        
        # Compute per-invariance-class metrics
        expected_invariant_consistent = sum(1 for p in expected_invariant_predictions if p == original_prediction)
        expected_invariant_total = len(expected_invariant_predictions)
        expected_invariant_consistency = (expected_invariant_consistent / expected_invariant_total 
                                         if expected_invariant_total > 0 else 1.0)
        expected_invariant_flip_rate = 1.0 - expected_invariant_consistency
        
        stress_test_consistent = sum(1 for p in stress_test_predictions if p == original_prediction)
        stress_test_total = len(stress_test_predictions)
        stress_test_consistency = (stress_test_consistent / stress_test_total 
                                  if stress_test_total > 0 else 1.0)
        stress_test_flip_rate = 1.0 - stress_test_consistency
        
        # Classify instability regime
        instability_regime = self._classify_instability_regime(
            decision_consistency, flip_rate, worst_case_deviation
        )
        
        # Check if perturbation variance dominates run variance
        perturbation_variance_dominates = confidence_variance > run_variance if run_variance > 0 else True
        
        return PerInputStability(
            input_id=input_id,
            original_text=original_text,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            decision_consistency=decision_consistency,
            flip_rate=flip_rate,
            confidence_variance=confidence_variance,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            worst_case_deviation=worst_case_deviation,
            worst_case_flip=worst_case_flip,
            instability_regime=instability_regime,
            run_variance=run_variance,
            perturbation_variance_dominates=perturbation_variance_dominates,
            expected_invariant_consistency=expected_invariant_consistency,
            expected_invariant_flip_rate=expected_invariant_flip_rate,
            stress_test_consistency=stress_test_consistency,
            stress_test_flip_rate=stress_test_flip_rate,
            num_perturbations=num_perturbations,
            num_consistent=num_consistent,
            num_flipped=num_flipped,
            perturbation_details=perturbation_details
        )
    
    def analyze_global(
        self,
        per_input_stabilities: List[PerInputStability]
    ) -> GlobalStabilityReport:
        """
        Compute global stability metrics across all inputs.
        
        Args:
            per_input_stabilities: List of PerInputStability objects
        
        Returns:
            GlobalStabilityReport object
        """
        if not per_input_stabilities:
            return GlobalStabilityReport(
                total_inputs=0,
                total_perturbations=0,
                mean_decision_consistency=0.0,
                mean_flip_rate=0.0,
                mean_confidence_variance=0.0,
                mean_worst_case_deviation=0.0,
                consistency_distribution={},
                flip_rate_distribution={},
                confidence_variance_distribution={},
                per_type_stats={}
            )
        
        # Aggregate metrics
        total_inputs = len(per_input_stabilities)
        total_perturbations = sum(s.num_perturbations for s in per_input_stabilities)
        
        mean_decision_consistency = statistics.mean([s.decision_consistency for s in per_input_stabilities])
        mean_flip_rate = statistics.mean([s.flip_rate for s in per_input_stabilities])
        mean_confidence_variance = statistics.mean([s.confidence_variance for s in per_input_stabilities])
        mean_worst_case_deviation = statistics.mean([s.worst_case_deviation for s in per_input_stabilities])
        
        # Distribution statistics
        consistencies = [s.decision_consistency for s in per_input_stabilities]
        flip_rates = [s.flip_rate for s in per_input_stabilities]
        conf_variances = [s.confidence_variance for s in per_input_stabilities]
        
        def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            return {
                'min': min(values),
                'max': max(values),
                'median': sorted_vals[n // 2] if n > 0 else 0.0,
                'q25': sorted_vals[n // 4] if n >= 4 else sorted_vals[0],
                'q75': sorted_vals[3 * n // 4] if n >= 4 else sorted_vals[-1],
                'mean': statistics.mean(values)
            }
        
        consistency_distribution = compute_distribution_stats(consistencies)
        flip_rate_distribution = compute_distribution_stats(flip_rates)
        confidence_variance_distribution = compute_distribution_stats(conf_variances)
        
        # Per-perturbation-type statistics
        per_type_stats = {}
        type_data = {}
        
        # Per-invariance-class statistics
        per_invariance_class_stats = {}
        invariance_class_data = {}
        
        for stability in per_input_stabilities:
            for detail in stability.perturbation_details:
                ptype = detail['perturbation_type']
                inv_class = detail.get('invariance_class', 'STRESS_TEST')
                
                # Per-type stats
                if ptype not in type_data:
                    type_data[ptype] = {'flipped': [], 'confidence_deviation': []}
                type_data[ptype]['flipped'].append(1.0 if detail['flipped'] else 0.0)
                type_data[ptype]['confidence_deviation'].append(abs(detail['confidence_deviation']))
                
                # Per-invariance-class stats
                if inv_class not in invariance_class_data:
                    invariance_class_data[inv_class] = {'flipped': [], 'confidence_deviation': []}
                invariance_class_data[inv_class]['flipped'].append(1.0 if detail['flipped'] else 0.0)
                invariance_class_data[inv_class]['confidence_deviation'].append(abs(detail['confidence_deviation']))
        
        for ptype, data in type_data.items():
            per_type_stats[ptype] = {
                'flip_rate': statistics.mean(data['flipped']) if data['flipped'] else 0.0,
                'mean_confidence_deviation': statistics.mean(data['confidence_deviation']) if data['confidence_deviation'] else 0.0,
                'count': len(data['flipped'])
            }
        
        for inv_class, data in invariance_class_data.items():
            per_invariance_class_stats[inv_class] = {
                'flip_rate': statistics.mean(data['flipped']) if data['flipped'] else 0.0,
                'mean_confidence_deviation': statistics.mean(data['confidence_deviation']) if data['confidence_deviation'] else 0.0,
                'count': len(data['flipped'])
            }
        
        # Instability regime distribution
        regime_distribution = {'STABLE': 0, 'SENSITIVE': 0, 'BRITTLE': 0}
        for stability in per_input_stabilities:
            regime = stability.instability_regime
            regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
        
        # Run variance statistics
        run_variances = [s.run_variance for s in per_input_stabilities]
        mean_run_variance = statistics.mean(run_variances) if run_variances else 0.0
        perturbation_variance_dominates_count = sum(
            1 for s in per_input_stabilities if s.perturbation_variance_dominates
        )
        
        return GlobalStabilityReport(
            total_inputs=total_inputs,
            total_perturbations=total_perturbations,
            mean_decision_consistency=mean_decision_consistency,
            mean_flip_rate=mean_flip_rate,
            mean_confidence_variance=mean_confidence_variance,
            mean_worst_case_deviation=mean_worst_case_deviation,
            consistency_distribution=consistency_distribution,
            flip_rate_distribution=flip_rate_distribution,
            confidence_variance_distribution=confidence_variance_distribution,
            per_type_stats=per_type_stats,
            per_invariance_class_stats=per_invariance_class_stats,
            regime_distribution=regime_distribution,
            mean_run_variance=mean_run_variance,
            perturbation_variance_dominates_count=perturbation_variance_dominates_count
        )

