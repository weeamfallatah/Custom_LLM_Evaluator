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
