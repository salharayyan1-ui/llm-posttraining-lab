# Analysis of LLM Post-Training, Optimization Mechanics, Feedback Fidelity, and Safety Calibration
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
This project is an investigation into what happens when you post-train a modern instruction model (`Qwen/Qwen2.5-1.5B-Instruct`). This work explores what behaviors each objective secretly rewards, what capabilities drift away from the reference model, and which critical failures stay completely invisible to training metrics.
Completed as part of the **Advanced Topics in Machine Learning** graduate course (Fall 2026).
---
### What’s in this repository?
We compare three core alignment algorithms—**Direct Preference Optimization (DPO)**, **Proximal Policy Optimization (PPO)**, and **Group Relative Policy Optimization (GRPO)**—and then shift the nature of the feedback itself by comparing **exact outcome verification (RLVR)** against **AI preference feedback (RLAIF)** on mathematical reasoning. Further analysis in report.

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

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

This repository contains the experimental code, training continuations, evaluation pipelines, and a report for a study of language model post-training across **Direct Preference Optimization (DPO)**, **Proximal Policy Optimization (PPO)**, **Group Relative Policy Optimization (GRPO)**, **Reinforcement Learning with Verifiable Rewards (RLVR)**, and **Reinforcement Learning from AI Feedback (RLAIF)**, evaluated on `Qwen/Qwen2.5-1.5B-Instruct`. This was a part of my Advanced Topics in Machine Learning course.
