# Analysis of LLM Post-Training: Optimization Mechanics, Feedback Fidelity, and Safety Calibration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

This project is an empirical investigation into what actually happens when you post-train a modern instruction model (`Qwen/Qwen2.5-1.5B-Instruct`). Rather than just asking *"which algorithm gets the highest score"*, this work explores what behaviors each objective secretly rewards, what capabilities drift away from the reference model, and which critical failures stay completely invisible to training metrics.

Completed as part of the **Advanced Topics in Machine Learning** graduate course (Fall 2026).

**Author:** Rayyan Salha (<salharayyan1@gmail.com>)  
**Repository:** [https://github.com/salharayyan1-ui/llm-posttraining-lab](https://github.com/salharayyan1-ui/llm-posttraining-lab)

---

### Key Findings & Empirical Takeaways

We compare three core alignment algorithms—**Direct Preference Optimization (DPO)**, **Proximal Policy Optimization (PPO)**, and **Group Relative Policy Optimization (GRPO)**—and then shift the nature of the feedback itself by comparing **exact outcome verification (RLVR)** against **AI preference feedback (RLAIF)** on mathematical reasoning.

* **Length bias in DPO:** DPO’s implicit reward easily exploits dataset verbosity. Balancing preference pairs across length strata curbs response length inflation while preserving preference accuracy.
* **The "Inactive Clipping" reality in PPO:** During short policy continuations, online clipping fractions remained virtually zero ($\epsilon$-clipping was largely inactive), meaning clipping sweeps tested geometric boundaries rather than actively binding constraints.
* **Gradient allocation in GRPO:** Critic-free GRPO relies heavily on group informativeness. Switching from canonical sequence averaging ($1/T_k$) to fixed sequence normalization (Dr. GRPO-style $1/L_{\max}$) radically alters how gradient mass is distributed across completion lengths.
* **The AI judge's over-refusal blindspot:** In our blind human audit of 240 responses on XSTest, the automated AI judge had a **0.0% predicted over-refusal rate**, blindly treating every benign refusal (like asking how to terminate a Python process) as "justified". Human auditing revealed that models suffer from ~35% over-refusal on harmless prompts containing sensitive keywords.
* **RLVR vs. RLAIF in math:** Exact answer verification (RLVR) provides clean, sparse signal. Direct RLAIF with an AI pairwise judge collapses into ties over 90% of the time on GSM8K and is easily fooled by verbose, persuasive formatting over actual mathematical validity.

---

### Project Structure

```text
pa2-llm-posttraining/
├── common/             # Shared tokenizers, metrics, GAE, and data utilities
├── configs/            # Experiment YAML configurations (fixed seeds, LoRA ranks, hyperparameters)
├── task1_dpo/          # DPO objective, length balancing, and beta ablation pipeline
├── task2_ppo/          # PPO continuation, clipping forks, and rollout cache analysis
├── task3_grpo/         # GRPO group-relative advantages, K-sweep, and normalization ablation
├── task4_safety/       # XSTest safety generation, AI judge scoring, and manual human audit
├── task5_feedback/     # Math reasoning: exact outcome verifier (RLVR) vs. pairwise AI judge (RLAIF)
├── results/            # Raw machine logs, JSON/CSV evaluation metrics, and response transcripts
├── report/             # 8-page NeurIPS-style final report and 11 generated PDF figures
├── scripts/            # Asset validation, environment checking, and verification scripts
└── requirements.txt    # Pinned dependency specification
```

---

### Quickstart & Reproduction

#### 1. Setup Environment & Assets
```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

#### 2. Run / Verify Experiments
All reported values and metrics can be verified against the saved run artifacts:
```bash
# Verify experiment schedules, token budgets, and output IDs
python -m scripts.verify_pipeline_outputs

# Run objective unit checks (verifying the corrected mathematical formulations)
python -m tests.test_objectives

# Reproduce normalization and diagnostic figures
python docs/replot_normalization_readable.py
```

---

### Attribution & Acknowledgments

- **Starter Scaffold:** Course starter infrastructure, baseline model checkpoints, and evaluation splits were provided by the course staff for EE-5102/CS-6304.
- **Implementations & Corrections:** Corrected algorithmic formulations (the reference subtraction term in DPO, pessimistic lower-bound clipping in PPO, and within-group advantage normalization in GRPO), experimental orchestration, ablation loops, manual safety auditing, and synthesis report were completed by the author.
- **Reference Papers:** Based on foundational methodologies from Rafailov et al. (DPO, 2023), Schulman et al. (PPO, 2017), Shao et al. (DeepSeekMath / GRPO, 2024), Liu et al. (Dr. GRPO, 2025), and Röttger et al. (XSTest, 2023).
