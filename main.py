#!/usr/bin/env python3
"""
Model Behavior Stability & Consistency Analyzer

Main entry point for the evaluation pipeline.
"""

import sys
import argparse
from src.evaluation.pipeline import EvaluationPipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Model Behavior Stability & Consistency Analyzer"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = EvaluationPipeline(config_path=args.config)
        results = pipeline.run()
        
        print("\n" + "=" * 70)
        print("Evaluation completed successfully.")
        print(f"Results exported to: {pipeline.config['output']['output_dir']}")
        print("=" * 70)
        
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

