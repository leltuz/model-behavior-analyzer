#!/usr/bin/env python3
"""
Comprehensive analysis of model behavior stability results.
"""

import json
import statistics
from collections import defaultdict
from typing import Dict, List, Any


def load_results(trace_path: str, summary_path: str) -> tuple:
    """Load trace and summary JSON files."""
    with open(trace_path, 'r') as f:
        trace = json.load(f)
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    return trace, summary


def analyze_perturbation_impact(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze impact of different perturbation types."""
    type_stats = defaultdict(lambda: {
        'total': 0,
        'flips': 0,
        'confidence_changes': [],
        'confidence_drops': [],
        'confidence_increases': []
    })
    
    for input_data in trace['inputs']:
        original_pred = input_data['original_inference_results'][0]['prediction']
        original_conf = input_data['original_inference_results'][0]['confidence']
        
        for pert in input_data['perturbations']:
            ptype = pert['perturbation_type']
            pert_pred = pert['inference_results'][0]['prediction']
            pert_conf = pert['inference_results'][0]['confidence']
            
            type_stats[ptype]['total'] += 1
            type_stats[ptype]['confidence_changes'].append(pert_conf - original_conf)
            
            if pert_pred != original_pred:
                type_stats[ptype]['flips'] += 1
            
            if pert_conf < original_conf:
                type_stats[ptype]['confidence_drops'].append(original_conf - pert_conf)
            else:
                type_stats[ptype]['confidence_increases'].append(pert_conf - original_conf)
    
    # Compute statistics
    analysis = {}
    for ptype, stats in type_stats.items():
        analysis[ptype] = {
            'flip_rate': stats['flips'] / stats['total'] if stats['total'] > 0 else 0.0,
            'mean_confidence_change': statistics.mean(stats['confidence_changes']) if stats['confidence_changes'] else 0.0,
            'mean_confidence_drop': statistics.mean(stats['confidence_drops']) if stats['confidence_drops'] else 0.0,
            'mean_confidence_increase': statistics.mean(stats['confidence_increases']) if stats['confidence_increases'] else 0.0,
            'max_confidence_drop': max(stats['confidence_drops']) if stats['confidence_drops'] else 0.0,
            'total_perturbations': stats['total']
        }
    
    return analysis


def identify_extreme_cases(summary: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Identify most and least stable inputs."""
    per_input = summary['per_input_summary']
    
    # Sort by different metrics
    by_consistency = sorted(per_input, key=lambda x: x['decision_consistency'])
    by_flip_rate = sorted(per_input, key=lambda x: x['flip_rate'], reverse=True)
    by_variance = sorted(per_input, key=lambda x: x['confidence_variance'], reverse=True)
    by_deviation = sorted(per_input, key=lambda x: x['worst_case_deviation'], reverse=True)
    
    return {
        'most_stable': by_consistency[-5:],  # Top 5 most consistent
        'least_stable': by_consistency[:5],   # Top 5 least consistent
        'highest_flip_rate': by_flip_rate[:5],
        'highest_variance': by_variance[:5],
        'highest_deviation': by_deviation[:5]
    }


def analyze_confidence_distribution(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze confidence score distributions."""
    original_confs = []
    perturbed_confs = []
    conf_changes = []
    
    for input_data in trace['inputs']:
        orig_conf = input_data['original_inference_results'][0]['confidence']
        original_confs.append(orig_conf)
        
        for pert in input_data['perturbations']:
            pert_conf = pert['inference_results'][0]['confidence']
            perturbed_confs.append(pert_conf)
            conf_changes.append(pert_conf - orig_conf)
    
    return {
        'original': {
            'mean': statistics.mean(original_confs),
            'median': statistics.median(original_confs),
            'stdev': statistics.stdev(original_confs) if len(original_confs) > 1 else 0.0,
            'min': min(original_confs),
            'max': max(original_confs)
        },
        'perturbed': {
            'mean': statistics.mean(perturbed_confs),
            'median': statistics.median(perturbed_confs),
            'stdev': statistics.stdev(perturbed_confs) if len(perturbed_confs) > 1 else 0.0,
            'min': min(perturbed_confs),
            'max': max(perturbed_confs)
        },
        'changes': {
            'mean': statistics.mean(conf_changes),
            'median': statistics.median(conf_changes),
            'stdev': statistics.stdev(conf_changes) if len(conf_changes) > 1 else 0.0,
            'min': min(conf_changes),
            'max': max(conf_changes)
        }
    }


def analyze_prediction_patterns(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze prediction patterns and correlations."""
    prediction_counts = defaultdict(int)
    flip_patterns = defaultdict(int)
    
    for input_data in trace['inputs']:
        orig_pred = input_data['original_inference_results'][0]['prediction']
        prediction_counts[orig_pred] += 1
        
        flips = sum(1 for pert in input_data['perturbations'] 
                   if pert['inference_results'][0]['prediction'] != orig_pred)
        flip_patterns[flips] += 1
    
    return {
        'original_predictions': dict(prediction_counts),
        'flip_frequency_distribution': dict(flip_patterns)
    }


def analyze_text_characteristics(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze relationship between text characteristics and stability."""
    text_lengths = []
    stability_by_length = defaultdict(list)
    
    for input_data in trace['inputs']:
        text = input_data['original_text']
        length = len(text.split())
        text_lengths.append(length)
        
        # Find corresponding stability (need to match with summary)
        # For now, just collect lengths
    
    return {
        'mean_length': statistics.mean(text_lengths) if text_lengths else 0.0,
        'median_length': statistics.median(text_lengths) if text_lengths else 0.0,
        'min_length': min(text_lengths) if text_lengths else 0,
        'max_length': max(text_lengths) if text_lengths else 0
    }


def print_analysis_report(trace: Dict[str, Any], summary: Dict[str, Any]):
    """Print comprehensive analysis report."""
    print("=" * 80)
    print("MODEL BEHAVIOR STABILITY & CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    # Global Summary
    gs = summary['global_stability']
    print("\n1. GLOBAL STABILITY OVERVIEW")
    print("-" * 80)
    print(f"Total Inputs Analyzed: {gs['total_inputs']}")
    print(f"Total Perturbations Generated: {gs['total_perturbations']}")
    print(f"\nOverall Metrics:")
    print(f"  Mean Decision Consistency: {gs['mean_decision_consistency']:.1%}")
    print(f"  Mean Flip Rate: {gs['mean_flip_rate']:.1%}")
    print(f"  Mean Confidence Variance: {gs['mean_confidence_variance']:.6f}")
    print(f"  Mean Worst-Case Deviation: {gs['mean_worst_case_deviation']:.3f}")
    
    # Distribution Analysis
    print(f"\nConsistency Distribution:")
    cd = gs['consistency_distribution']
    print(f"  Range: [{cd['min']:.3f}, {cd['max']:.3f}]")
    print(f"  Median: {cd['median']:.3f}")
    print(f"  Q25-Q75: [{cd['q25']:.3f}, {cd['q75']:.3f}]")
    
    # Perturbation Type Impact
    print("\n2. PERTURBATION TYPE IMPACT ANALYSIS")
    print("-" * 80)
    pert_analysis = analyze_perturbation_impact(trace)
    
    for ptype, stats in sorted(pert_analysis.items(), key=lambda x: x[1]['flip_rate'], reverse=True):
        print(f"\n{ptype.upper().replace('_', ' ')}:")
        print(f"  Total Perturbations: {stats['total_perturbations']}")
        print(f"  Flip Rate: {stats['flip_rate']:.1%}")
        print(f"  Mean Confidence Change: {stats['mean_confidence_change']:+.4f}")
        print(f"  Mean Confidence Drop: {stats['mean_confidence_drop']:.4f}")
        print(f"  Max Confidence Drop: {stats['max_confidence_drop']:.4f}")
    
    # Confidence Distribution
    print("\n3. CONFIDENCE SCORE DISTRIBUTION")
    print("-" * 80)
    conf_dist = analyze_confidence_distribution(trace)
    print(f"Original Inputs:")
    print(f"  Mean: {conf_dist['original']['mean']:.4f}")
    print(f"  Median: {conf_dist['original']['median']:.4f}")
    print(f"  Std Dev: {conf_dist['original']['stdev']:.4f}")
    print(f"  Range: [{conf_dist['original']['min']:.4f}, {conf_dist['original']['max']:.4f}]")
    
    print(f"\nPerturbed Inputs:")
    print(f"  Mean: {conf_dist['perturbed']['mean']:.4f}")
    print(f"  Median: {conf_dist['perturbed']['median']:.4f}")
    print(f"  Std Dev: {conf_dist['perturbed']['stdev']:.4f}")
    print(f"  Range: [{conf_dist['perturbed']['min']:.4f}, {conf_dist['perturbed']['max']:.4f}]")
    
    print(f"\nConfidence Changes (Perturbed - Original):")
    print(f"  Mean: {conf_dist['changes']['mean']:+.4f}")
    print(f"  Median: {conf_dist['changes']['median']:+.4f}")
    print(f"  Std Dev: {conf_dist['changes']['stdev']:.4f}")
    print(f"  Range: [{conf_dist['changes']['min']:+.4f}, {conf_dist['changes']['max']:+.4f}]")
    
    # Extreme Cases
    print("\n4. EXTREME CASES ANALYSIS")
    print("-" * 80)
    extremes = identify_extreme_cases(summary)
    
    print("\nMost Stable Inputs (Highest Consistency):")
    for item in reversed(extremes['most_stable']):
        print(f"  {item['input_id']}: Consistency={item['decision_consistency']:.3f}, "
              f"FlipRate={item['flip_rate']:.3f}, Variance={item['confidence_variance']:.6f}")
    
    print("\nLeast Stable Inputs (Lowest Consistency):")
    for item in extremes['least_stable']:
        print(f"  {item['input_id']}: Consistency={item['decision_consistency']:.3f}, "
              f"FlipRate={item['flip_rate']:.3f}, Variance={item['confidence_variance']:.6f}")
    
    print("\nInputs with Highest Flip Rates:")
    for item in extremes['highest_flip_rate']:
        if item['flip_rate'] > 0:
            print(f"  {item['input_id']}: FlipRate={item['flip_rate']:.3f}, "
                  f"Consistency={item['decision_consistency']:.3f}")
    
    print("\nInputs with Highest Confidence Variance:")
    for item in extremes['highest_variance'][:3]:
        print(f"  {item['input_id']}: Variance={item['confidence_variance']:.6f}, "
              f"Consistency={item['decision_consistency']:.3f}")
    
    # Prediction Patterns
    print("\n5. PREDICTION PATTERNS")
    print("-" * 80)
    patterns = analyze_prediction_patterns(trace)
    print(f"Original Prediction Distribution:")
    for pred, count in patterns['original_predictions'].items():
        pct = (count / gs['total_inputs']) * 100
        print(f"  {pred}: {count} ({pct:.1f}%)")
    
    print(f"\nFlip Frequency Distribution:")
    for num_flips, count in sorted(patterns['flip_frequency_distribution'].items()):
        pct = (count / gs['total_inputs']) * 100
        print(f"  {num_flips} flips: {count} inputs ({pct:.1f}%)")
    
    # Text Characteristics
    print("\n6. TEXT CHARACTERISTICS")
    print("-" * 80)
    text_chars = analyze_text_characteristics(trace)
    print(f"Input Text Length Statistics (words):")
    print(f"  Mean: {text_chars['mean_length']:.1f}")
    print(f"  Median: {text_chars['median_length']:.1f}")
    print(f"  Range: [{text_chars['min_length']}, {text_chars['max_length']}]")
    
    # Key Insights
    print("\n7. KEY INSIGHTS")
    print("-" * 80)
    
    # Find most disruptive perturbation type
    most_disruptive = max(pert_analysis.items(), key=lambda x: x[1]['flip_rate'])
    print(f"• Most Disruptive Perturbation Type: {most_disruptive[0]}")
    print(f"  - Causes {most_disruptive[1]['flip_rate']:.1%} of predictions to flip")
    print(f"  - Average confidence drop: {most_disruptive[1]['mean_confidence_drop']:.4f}")
    
    # Find most stable perturbation type
    most_stable_pert = min(pert_analysis.items(), key=lambda x: x[1]['flip_rate'])
    print(f"\n• Most Stable Perturbation Type: {most_stable_pert[0]}")
    print(f"  - Causes {most_stable_pert[1]['flip_rate']:.1%} of predictions to flip")
    
    # Overall stability assessment
    if gs['mean_decision_consistency'] >= 0.95:
        stability_level = "HIGHLY STABLE"
    elif gs['mean_decision_consistency'] >= 0.80:
        stability_level = "MODERATELY STABLE"
    else:
        stability_level = "UNSTABLE"
    
    print(f"\n• Overall Stability Assessment: {stability_level}")
    print(f"  - {gs['mean_decision_consistency']:.1%} of perturbations preserve predictions")
    print(f"  - {gs['mean_flip_rate']:.1%} of perturbations cause prediction flips")
    
    # Confidence stability
    if conf_dist['changes']['stdev'] < 0.05:
        conf_stability = "LOW VARIANCE"
    elif conf_dist['changes']['stdev'] < 0.10:
        conf_stability = "MODERATE VARIANCE"
    else:
        conf_stability = "HIGH VARIANCE"
    
    print(f"\n• Confidence Score Stability: {conf_stability}")
    print(f"  - Std Dev of confidence changes: {conf_dist['changes']['stdev']:.4f}")
    print(f"  - Mean change: {conf_dist['changes']['mean']:+.4f}")
    
    # Brittleness indicator
    num_brittle = sum(1 for item in summary['per_input_summary'] if item['worst_case_flip'])
    brittle_pct = (num_brittle / gs['total_inputs']) * 100
    print(f"\n• Brittleness Indicator:")
    print(f"  - {num_brittle} inputs ({brittle_pct:.1f}%) experienced at least one prediction flip")
    print(f"  - This indicates potential brittleness in decision boundaries")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


def main():
    """Main analysis function."""
    trace_path = "outputs/behavior_trace.json"
    summary_path = "outputs/stability_summary.json"
    
    try:
        trace, summary = load_results(trace_path, summary_path)
        print_analysis_report(trace, summary)
    except FileNotFoundError as e:
        print(f"Error: Could not find result files. {e}")
        print("Please run the evaluation pipeline first: python main.py")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

