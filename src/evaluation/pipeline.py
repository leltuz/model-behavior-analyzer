"""End-to-end evaluation pipeline orchestration."""

import os
import logging
import json
from typing import Dict, Any
from dataclasses import asdict
import yaml

from ..utils.dataset import load_dataset
from ..perturbations import PerturbationGenerator
from ..inference import InferenceReplay
from ..analysis import StabilityAnalyzer, PatternAnalyzer
from ..tracing import BehaviorTracer


class EvaluationPipeline:
    """Orchestrates the complete evaluation pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize evaluation pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.perturbation_generator = PerturbationGenerator(
            self.config['perturbations'],
            seed=self.config['dataset']['seed']
        )
        
        self.inference_replay = InferenceReplay(
            self.config['inference'],
            seed=self.config['inference']['seed']
        )
        
        self.analyzer = StabilityAnalyzer(self.config['analysis'])
        self.pattern_analyzer = PatternAnalyzer(self.config['analysis'])
        self.tracer = BehaviorTracer(self.config)
        
        # Create output directory
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        log_file = os.path.join(output_dir, self.config['output']['log_file'])
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run(self):
        """Execute the complete evaluation pipeline."""
        self.logger.info("Starting Model Behavior Stability & Consistency Analysis")
        self.logger.info("=" * 70)
        
        # Stage 1: Load Input Anchors
        self.logger.info("Stage 1: Loading Input Anchors")
        dataset = load_dataset(
            self.config['dataset'].get('input_path'),
            self.config['dataset']['num_samples'],
            self.config['dataset']['seed']
        )
        self.logger.info(f"Loaded {len(dataset)} input anchors")
        
        # Stage 2: Generate Perturbations
        self.logger.info("Stage 2: Generating Perturbations")
        all_perturbations = {}
        for item in dataset:
            input_id = item['id']
            text = item['text']
            perturbations = self.perturbation_generator.generate(text, input_id)
            all_perturbations[input_id] = {
                'text': text,
                'perturbations': perturbations
            }
            self.logger.debug(f"Generated {len(perturbations)} perturbations for {input_id}")
        
        total_perturbations = sum(len(p['perturbations']) for p in all_perturbations.values())
        self.logger.info(f"Generated {total_perturbations} total perturbations")
        
        # Stage 3: Deterministic Inference Replay
        self.logger.info("Stage 3: Running Deterministic Inference Replay")
        all_results = {}
        
        for input_id, data in all_perturbations.items():
            text = data['text']
            perturbations = data['perturbations']
            
            # Run inference on original
            self.logger.debug(f"Running inference on original: {input_id}")
            original_results = self.inference_replay.run(text, metadata={'input_id': input_id})
            
            # Run inference on perturbations
            perturbation_results = []
            for pert in perturbations:
                self.logger.debug(f"Running inference on perturbation: {input_id} - {pert.type}")
                pert_results = self.inference_replay.run(
                    pert.perturbed_text,
                    metadata={'input_id': input_id, 'perturbation_type': pert.type}
                )
                perturbation_results.append({
                    'perturbation': pert,
                    'results': pert_results
                })
            
            all_results[input_id] = {
                'original_text': text,
                'original_results': original_results,
                'perturbations': perturbations,
                'perturbation_results': perturbation_results
            }
            
            # Add to tracer
            self.tracer.add_input_trace(
                input_id,
                text,
                original_results,
                perturbations,
                perturbation_results
            )
        
        self.logger.info("Completed inference replay for all inputs and perturbations")
        
        # Stage 4: Stability & Consistency Analysis
        self.logger.info("Stage 4: Computing Stability & Consistency Metrics")
        per_input_stabilities = []
        
        for input_id, data in all_results.items():
            stability = self.analyzer.analyze_input(
                input_id,
                data['original_text'],
                data['original_results'],
                data['perturbation_results']
            )
            per_input_stabilities.append(stability)
            
            self.logger.info(
                f"{input_id}: Consistency={stability.decision_consistency:.3f}, "
                f"FlipRate={stability.flip_rate:.3f}, "
                f"ConfVar={stability.confidence_variance:.4f}"
            )
        
        # Compute global stability
        global_stability = self.analyzer.analyze_global(per_input_stabilities)
        
        self.logger.info("Global Stability Metrics:")
        self.logger.info(f"  Mean Decision Consistency: {global_stability.mean_decision_consistency:.3f}")
        self.logger.info(f"  Mean Flip Rate: {global_stability.mean_flip_rate:.3f}")
        self.logger.info(f"  Mean Confidence Variance: {global_stability.mean_confidence_variance:.4f}")
        self.logger.info(f"  Mean Worst-Case Deviation: {global_stability.mean_worst_case_deviation:.3f}")
        self.logger.info(f"  Instability Regime Distribution:")
        for regime, count in global_stability.regime_distribution.items():
            pct = (count / global_stability.total_inputs * 100) if global_stability.total_inputs > 0 else 0.0
            self.logger.info(f"    {regime}: {count} ({pct:.1f}%)")
        
        # Cross-input pattern analysis
        self.logger.info("Computing Cross-Input Behavioral Patterns")
        trace_data = self.tracer.get_trace()
        behavioral_patterns = self.pattern_analyzer.analyze(per_input_stabilities, trace_data)
        
        # Add analysis to tracer
        self.tracer.add_stability_analysis(per_input_stabilities, global_stability, behavioral_patterns)
        
        # Export results
        self.logger.info("Exporting Results")
        output_dir = self.config['output']['output_dir']
        
        trace_file = os.path.join(output_dir, self.config['output']['trace_file'])
        self.tracer.export_json(trace_file)
        self.logger.info(f"Exported behavior trace to {trace_file}")
        
        summary_file = os.path.join(output_dir, self.config['output']['summary_file'])
        summary = {
            'global_stability': asdict(global_stability),
            'per_input_summary': [
                {
                    'input_id': s.input_id,
                    'decision_consistency': s.decision_consistency,
                    'flip_rate': s.flip_rate,
                    'confidence_variance': s.confidence_variance,
                    'worst_case_deviation': s.worst_case_deviation,
                    'worst_case_flip': s.worst_case_flip,
                    'instability_regime': s.instability_regime,
                    'run_variance': s.run_variance,
                    'perturbation_variance_dominates': s.perturbation_variance_dominates,
                    'expected_invariant_consistency': s.expected_invariant_consistency,
                    'expected_invariant_flip_rate': s.expected_invariant_flip_rate,
                    'stress_test_consistency': s.stress_test_consistency,
                    'stress_test_flip_rate': s.stress_test_flip_rate
                }
                for s in per_input_stabilities
            ],
            'behavioral_patterns': asdict(behavioral_patterns) if behavioral_patterns else {}
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Exported stability summary to {summary_file}")
        
        self.logger.info("=" * 70)
        self.logger.info("Evaluation Complete")
        
        return {
            'per_input_stabilities': per_input_stabilities,
            'global_stability': global_stability
        }

