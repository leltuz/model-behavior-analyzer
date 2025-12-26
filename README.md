# Model Behavior Stability & Consistency Analyzer

## TL;DR

A research-grade evaluation instrument for measuring whether machine learning models exhibit unstable or brittle behavior under controlled, normatively defined input perturbations—independent of correctness or real-world distribution shift.

**Why it matters:**
- Treats model behavior itself as a measurable object
- Makes invariance expectations explicit rather than implicit
- Uses deterministic replay as experimental control, not as a model claim
- Produces interpretable, comparative behavioral metrics beyond accuracy

## How to Read This README

This README is written as a behavioral evaluation methodology, not a model benchmark or robustness toolkit.

- **For the core idea and epistemic stance:** Start with _Research Question_ and _Conceptual Framework_. These sections explain what is being measured (behavioral stability) and what is explicitly not being claimed (correctness, safety, or real-world reliability).

- **For the methodological design:** Read _System Architecture_, especially Stage 2 (Perturbation Generation) and Stage 3 (Deterministic Inference Replay). These define the normative structure (invariance classes) and experimental controls used to isolate behavioral effects.

- **For the measurement outputs:** Focus on _Stage 4: Stability & Consistency Analysis_ and _Observability & Tracing_ to understand the metrics, instability regimes, and how results are exposed for downstream analysis.

- **For interpretation boundaries:** Read _What This Does NOT Prove_ carefully. This section is integral to the project and defines the epistemic limits of the measurements.

- **For practical execution:** Sections on _Configuration_, _Installation_, and _Usage_ are secondary and included to support reproducible evaluation rather than to position the system as a deployable tool.

## Research Question

**This system is designed to measure whether a model's behavior changes in ways that violate explicit, reasonable invariance expectations under controlled input variation, rather than whether those changes are correct or whether the perturbations capture real-world distribution shift.**

This question anchors all components of the system: invariance classes, instability regimes, pattern analysis, and deterministic replay. The system operates at the input–output behavior level, making no assumptions about model architecture, training method, or modality. Any classifier with confidence outputs can be evaluated using this instrument.

## Conceptual Framework

### Behavioral Measurement vs. Correctness

Traditional model evaluation focuses on aggregate accuracy metrics that measure correctness against ground truth labels. However, these metrics fail to capture a critical dimension: **behavioral stability and consistency** under controlled input variations.

A model may achieve high accuracy while exhibiting brittle, unstable behavior. Small, semantically-preserving perturbations can cause prediction flips, confidence volatility, and inconsistent decisions across semantically equivalent inputs.

This system treats **model behavior as an object of measurement**, independent of correctness. It quantifies behavioral stability under controlled conditions, exposing brittleness and instability that aggregate accuracy metrics cannot capture.

### Model-Agnostic Design

The system makes no assumptions about:
- Model architecture (transformer, CNN, RNN, etc.)
- Training method (supervised, self-supervised, etc.)
- Modality (text, though current perturbations are text-based)
- Deployment context

It operates purely at the **input–output behavior level**. Any classifier that produces:
- Predictions (class labels)
- Confidence scores

can be evaluated using this instrument. The system measures behavioral characteristics, not architectural properties.

## System Architecture

The evaluation pipeline consists of four stages:

### Stage 1: Input Anchors

Loads a base dataset of inputs, each treated as a behavioral anchor with stable IDs for traceability. Supports loading from JSON files or synthetic dataset generation for testing.

### Stage 2: Perturbation Generation

Generates controlled perturbations per input. **Important**: These perturbations serve as **proxies for invariance expectations**, not claims of linguistic completeness or real-world coverage.

**Perturbation Types** (conceptual placeholders for broader behavioral categories):
- **Stopword Removal**: Removes common function words
- **Synonym Substitution**: Replaces words with synonyms from a fixed dictionary
- **Punctuation/Casing Changes**: Modifies capitalization or removes punctuation
- **Character-Level Noise**: Introduces controlled typos

**Invariance Classes** (normative categorization):
- **EXPECTED_INVARIANT**: Perturbations that should not change model behavior (e.g., casing, punctuation, stopword removal). Instability under these is a strong signal of brittleness.
- **STRESS_TEST**: Perturbations that test robustness boundaries (e.g., character noise). Instability under these is expected and provides weaker signal.

**Conceptual Role**: The perturbation types are intentionally simple placeholders. Their value lies in their **conceptual role** (testing invariance expectations) rather than **linguistic realism** (capturing real-world variation). The system measures behavioral response to controlled variation, not coverage of real-world distribution shift.

### Stage 3: Deterministic Inference Replay

Runs inference on original inputs and all perturbations, with optional repeated runs.

**Determinism as Experimental Control**: Fixed seeds and deterministic inference settings serve as **experimental controls**, not assertions about model properties. Determinism isolates input-driven variance from execution noise, enabling:
- Comparison of run variance vs. perturbation variance
- Reproducible measurement across runs
- Isolation of behavioral signal from numerical noise

Repeated inference is used to **compare run variance vs. perturbation variance**, not to simulate stochastic deployment. This methodological choice ensures that observed behavioral changes are attributable to input variation, not execution variability.

### Stage 4: Stability & Consistency Analysis

Computes behavioral metrics:

- **Decision Consistency**: Fraction of perturbations preserving original prediction
- **Flip Rate**: Percentage of perturbations changing predicted class
- **Confidence Variance**: Variance of confidence scores across perturbations
- **Worst-Case Deviation**: Maximum confidence drop or decision change
- **Run Variance**: Variance across repeated inference runs (numerical stability probe)
- **Instability Regime Classification**: STABLE, SENSITIVE, or BRITTLE (configurable thresholds)
- **Per-Invariance-Class Metrics**: Separate metrics for EXPECTED_INVARIANT vs STRESS_TEST

**Instability Regimes** (configurable thresholds):
- **STABLE**: High consistency (≥95%), low flip rate (≤5%), low deviation (≤10%)
- **SENSITIVE**: Between stable and brittle thresholds
- **BRITTLE**: Low consistency (<80%), high flip rate (≥20%), or high deviation (≥30%)

Aggregates metrics globally and performs cross-input behavioral pattern analysis.


## Observability & Tracing

Every evaluation run produces a `BehaviorTrace` exported to JSON, containing:
- Per-input traces (original inputs, perturbations, inference results)
- Aggregated analysis (per-input stability summaries, global metrics, behavioral patterns)

Output files:
- `behavior_trace.json`: Complete trace
- `stability_summary.json`: Aggregated metrics
- `evaluation.log`: Execution log

All outputs are machine-readable JSON suitable for downstream analysis.


## Evaluation Methodology

The system is designed for **offline, deterministic evaluation**:

- **Deterministic Replay**: Fixed seeds ensure identical inputs yield identical outputs (experimental control)
- **Reproducible**: Same configuration → same results
- **Comparable**: Supports comparison across perturbation types, confidence thresholds, inference configurations, and model versions

Evaluation is **offline only**—no online learning, A/B testing, or production serving.


## Configuration

All parameters are configurable via `config.yaml`:
- Dataset: Input path, number of samples, seed
- Perturbations: Types, frequencies, invariance class mappings
- Inference: Model name, number of runs, batch size, device
- Analysis: Confidence thresholds, instability regime thresholds, metrics
- Output: Directory and filenames

No hard-coded constants in logic—everything is configurable.


## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: For real transformer model inference
pip install transformers torch
```

**Important Note on Mock Inference**: The system includes a mock inference mode that activates automatically if `transformers`/`torch` are not installed. This mock mode is:
- **Test harness only**: Designed for CI, unit testing, and dependency-free execution
- **Not representative**: Mock inference uses simple heuristics and does not reflect real model behavior
- **For pipeline testing**: Useful for verifying the evaluation pipeline works, but not for actual behavioral analysis

**For real evaluation**: Install `transformers` and `torch` to use actual model inference. Mock inference results should not be used to draw conclusions about model stability.


## Usage

```bash
# Run with default config.yaml
python main.py

# Run with custom config
python main.py --config custom_config.yaml
```

The pipeline executes all four stages sequentially and exports results to the output directory specified in the config.


## What This Does NOT Prove

This system measures behavioral stability under controlled perturbations. It is important to understand what these measurements do and do not establish:

**This system does NOT prove:**

1. **Stability ≠ Correctness**: A stable model (consistent predictions under perturbations) may still be systematically wrong. Stability measures consistency, not accuracy.

2. **Robustness ≠ Fairness**: A model that is stable under the perturbations tested here may still exhibit bias, discrimination, or unfair behavior. Perturbation stability is orthogonal to fairness concerns.

3. **Perturbation Coverage ≠ Real-World Reliability**: The perturbations tested represent a limited, controlled set serving as proxies for invariance expectations. Real-world inputs may exhibit variations not captured by these perturbations. High stability on these perturbations does not guarantee reliability in production.

4. **Behavioral Measurement ≠ Deployment Safety**: This system measures behavior; it does not validate deployment safety. A model may be stable under these perturbations but still pose risks (privacy violations, harmful outputs, etc.) that are not captured by stability metrics.

5. **Invariance Expectations ≠ Universal Truths**: The classification of perturbations into EXPECTED_INVARIANT vs STRESS_TEST is a normative, domain-specific judgment. What is "expected" to be invariant depends on the use case and may not hold universally.

**What this system DOES provide:**

- Quantitative measurement of behavioral consistency under controlled conditions
- Identification of inputs and perturbation types that cause instability
- Comparative analysis across models, configurations, or perturbation types
- Research-grade data for understanding model behavior characteristics

**Epistemic Position**: This is a measurement instrument, not a validation tool. The measurements provide signals about behavioral characteristics, but do not establish guarantees about correctness, fairness, or safety. Interpretation of results requires domain expertise and consideration of the limitations above.

## Out of Scope

This system explicitly **does not**:
- Train or fine-tune models
- Perform online learning or feedback loops
- Conduct A/B testing
- Provide UI or dashboards
- Serve models in production
- Generate adversarial attacks
- Perform generative paraphrasing
- Require heavy infrastructure or concurrency

This is an **evaluation instrument**, not a product. The focus is on measurement, not optimization.


## Design Principles

1. **Deterministic**: Same inputs → same outputs (via fixed seeds, as experimental control)
2. **Observable**: Every decision is traceable through JSON exports
3. **Evaluative**: Focused on measurement, not performance optimization
4. **Configurable**: No magic constants—everything via `config.yaml`
5. **Modular**: Perturbations, inference, and analysis are separable
6. **Model-Agnostic**: Operates at input–output behavior level, no architectural assumptions


## Framing

This project frames model behavior as a **measurable system property**, independent of correctness. By quantifying stability and consistency under controlled perturbations, we expose behavioral characteristics that accuracy metrics cannot capture.

The system measures whether behavior changes in ways that violate explicit, reasonable invariance expectations. It distinguishes between:
- Instability under expected invariants (strong signal of brittleness)
- Instability under stress tests (weaker signal, expected to some degree)

This normative structure, combined with experimental controls (determinism) and model-agnostic design, positions the system as a research-grade measurement instrument for analyzing behavioral reliability and invariance in AI systems.

**Note**: This analyzer measures behavior, not performance. It identifies brittleness and instability, enabling comparative analysis of model behavioral characteristics. The measurements are interpretative and require careful consideration of what they do and do not establish.

## License

This project is released under the MIT License. See the LICENSE file for details.
Provided for research, educational, and demonstration purposes, without warranty of any kind.
