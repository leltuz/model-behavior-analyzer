"""Behavior tracing and observability system."""

import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class BehaviorTrace:
    """Complete behavior trace for evaluation run."""
    timestamp: str
    config: Dict[str, Any]
    
    # Per-input traces
    inputs: List[Dict[str, Any]]
    
    # Aggregated analysis
    per_input_stability: List[Dict[str, Any]]
    global_stability: Dict[str, Any]
    behavioral_patterns: Dict[str, Any]
    
    # Metadata
    metadata: Dict[str, Any]


class BehaviorTracer:
    """Traces and exports model behavior for analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize behavior tracer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.trace_data = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'inputs': [],
            'per_input_stability': [],
            'global_stability': {},
            'behavioral_patterns': {},
            'metadata': {}
        }
    
    def add_input_trace(
        self,
        input_id: str,
        original_text: str,
        original_results: List[Any],
        perturbations: List[Any],
        perturbation_results: List[Dict[str, Any]]
    ):
        """
        Add trace data for a single input.
        
        Args:
            input_id: Unique identifier
            original_text: Original input text
            original_results: Inference results for original
            perturbations: List of Perturbation objects
            perturbation_results: List of dicts with 'perturbation' and 'results'
        """
        # Format original results
        original_trace = []
        for result in original_results:
            original_trace.append({
                'run_index': result.run_index,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'metadata': result.metadata
            })
        
        # Format perturbation results
        perturbation_traces = []
        for pert, pert_data in zip(perturbations, perturbation_results):
            pert_results = pert_data['results']
            pert_trace = {
                'perturbation_type': pert.type,
                'invariance_class': pert.invariance_class,
                'original_text': pert.original_text,
                'perturbed_text': pert.perturbed_text,
                'perturbation_metadata': pert.metadata,
                'inference_results': []
            }
            
            for result in pert_results:
                pert_trace['inference_results'].append({
                    'run_index': result.run_index,
                    'prediction': result.prediction,
                    'confidence': result.confidence,
                    'metadata': result.metadata
                })
            
            perturbation_traces.append(pert_trace)
        
        self.trace_data['inputs'].append({
            'input_id': input_id,
            'original_text': original_text,
            'original_inference_results': original_trace,
            'perturbations': perturbation_traces
        })
    
    def add_stability_analysis(
        self,
        per_input_stabilities: List[Any],
        global_stability: Any,
        behavioral_patterns: Any = None
    ):
        """
        Add stability analysis results to trace.
        
        Args:
            per_input_stabilities: List of PerInputStability objects
            global_stability: GlobalStabilityReport object
            behavioral_patterns: BehavioralPatterns object (optional)
        """
        # Convert dataclasses to dicts
        self.trace_data['per_input_stability'] = [
            asdict(s) for s in per_input_stabilities
        ]
        self.trace_data['global_stability'] = asdict(global_stability)
        
        if behavioral_patterns:
            self.trace_data['behavioral_patterns'] = asdict(behavioral_patterns)
    
    def export_json(self, filepath: str):
        """
        Export trace to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        # Convert to JSON-serializable format
        trace = BehaviorTrace(
            timestamp=self.trace_data['timestamp'],
            config=self.trace_data['config'],
            inputs=self.trace_data['inputs'],
            per_input_stability=self.trace_data['per_input_stability'],
            global_stability=self.trace_data['global_stability'],
            behavioral_patterns=self.trace_data.get('behavioral_patterns', {}),
            metadata=self.trace_data['metadata']
        )
        
        with open(filepath, 'w') as f:
            json.dump(asdict(trace), f, indent=2)
    
    def get_trace(self) -> Dict[str, Any]:
        """Get current trace data."""
        return self.trace_data.copy()

