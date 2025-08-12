# Custom LLM Evaluator (C-Eval)

**C-Eval** is a framework for evaluating Large Language Model (LLM) outputs using **custom, user-defined metrics** with integrated **Chain-of-Thought (CoT)** reasoning to enhance evaluation steps.

## Features
- **Define your own metrics** in plain language or load them from JSON files.
- Automatically **generate or refine evaluation steps** based on given criteria.
- Evaluate outputs considering **context**, **expected outputs**, and **queries**.
- Get **scored results** (0–1) with detailed reasoning for each metric.

## Key Capabilities
- Step improvement with correct/incorrect examples.
- Support for datasets from CSV or Excel.
- Structured results including scores, reasoning, and step improvements.
- Flexible for both simple and complex evaluation tasks.

## Use Cases
- Assessing response quality in NLP applications.
- Testing accuracy and safety in AI-generated outputs.
- Building domain-specific evaluation frameworks.

## Example Workflow
1. Load your dataset (CSV or Excel).
2. Define or import custom metrics.
3. Run C-Eval to score each test case.
4. Review results with scores, reasoning, and improvements.

> 🚀 **Future Enhancements:** Step refinement with user feedback and score normalization for higher accuracy.

## High‑level architecture
custom_metric_evaluator.py
├─ run_model_octo_ai()         # Thin OctoAI chat completion wrapper
├─ TestCaseParams              # String constants for dataframe keys
├─ Eval_metric                 # Container for a single metric's config
└─ CustomLLMEvaluator          # Orchestrates step-gen, step-improve, scoring
   ├─ run_model()                        # Model call helper
   ├─ generate_eval_steps()              # CoT → initial steps from criteria
   ├─ generate_improving_steps()         # Refine steps (optional examples)
   ├─ extract_improving_steps_response() # Parse JSON/regex → DataFrame
   ├─ improved_steps_check()             # Cache per-metric refined steps
   ├─ custom_evaluator_prompt()          # Score one test case
   └─ custom_Eval()                      # Score a DataFrame of cases

